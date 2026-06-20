#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_atomic.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_Atomic, RS_SpinLock, and RS_TicketLock provide thread safe
 * primitives for device memory that work on both the cpu and the
 * cuda backend. Many threads contend for a shared counter and the
 * host checks the aggregate.
 *
 * - Lock Free Add
 *   deviceAtomicAdd lets every thread add to one counter
 *
 * - SpinLock
 *   lock(), unlock(), clear() guard a plain counter increment
 *
 * - TicketLock
 *   lock(), unlock(), clear() guard a plain counter increment in
 *   first in first out order
 *****************************************************************/

class RS_AtomicTest : public ::testing::Test {
   protected:
    void SetUp() override {
        RS_TEST_SETUP_DEVICE_PROVIDER();
        host_provider.init(0);
    }

    void TearDown() override {
        provider.free();
        host_provider.free();
    }

    RS_TEST_DEVICE_PROVIDER_TYPE provider;
    RS_MemoryCPUProvider host_provider;
};

/*****************************************************************
 * Compute functions for atomic and lock based increments
 *****************************************************************/

/// @brief Each thread adds one to a single counter with a device atomic add.
class RSCF_AtomicAddCounter : public RS_ComputeFunction<RSCF_AtomicAddCounter> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_Array<int, 1> counter, int num_threads) {
        RIndex idx = index1D();
        if (idx < static_cast<RIndex>(num_threads)) {
            deviceAtomicAdd(&counter[0], 1);
        }
    }
};

/// @brief Each thread takes a spinlock, adds one to the counter, then releases.
class RSCF_SpinLockCounter : public RS_ComputeFunction<RSCF_SpinLockCounter> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_SpinLock* lock, int* counter, int num_threads) {
        RIndex idx = index1D();
        if (idx < static_cast<RIndex>(num_threads)) {
            lock->lock();
            counter[0] = counter[0] + 1;
            lock->unlock();
        }
    }
};

/// @brief Each thread takes a ticket lock, adds one to the counter, then releases.
class RSCF_TicketLockCounter : public RS_ComputeFunction<RSCF_TicketLockCounter> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_TicketLock* lock, int* counter, int num_threads) {
        RIndex idx = index1D();
        if (idx < static_cast<RIndex>(num_threads)) {
            lock->lock();
            counter[0] = counter[0] + 1;
            lock->unlock();
        }
    }
};

/// @brief Clears the spinlock and reports its raw state back to the host.
class RSCF_ClearSpinLock : public RS_ComputeFunction<RSCF_ClearSpinLock> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_SpinLock* lock, unsigned int* state) {
        RIndex idx = index1D();
        if (idx == 0) {
            lock->clear();
            state[0] = lock->state;
        }
    }
};

/*****************************************************************
 * Test cases for device atomic add
 *
 * Test cases:
 * 1. Many threads add to one counter on the provider.
 * 2. Many threads add to one counter on the host provider.
 *****************************************************************/

TEST_F(RS_AtomicTest, AtomicAddCounterOnProvider) {
    const int num_threads = 100;

    RS_Array<int, 1> counter({1}, &provider);
    counter.fill(0);

    provider.run<RSCF_AtomicAddCounter>({static_cast<RSize>(num_threads)}, counter, num_threads);

    int result = -1;
    counter.read(&result);
    EXPECT_EQ(result, num_threads);

    counter.free();
}

TEST_F(RS_AtomicTest, AtomicAddCounterOnHostProvider) {
    const int num_threads = 100;

    RS_Array<int, 1> counter({1}, &host_provider);
    counter.fill(0);

    host_provider.run<RSCF_AtomicAddCounter>(
        {static_cast<RSize>(num_threads)}, counter, num_threads
    );

    int result = -1;
    counter.read(&result);
    EXPECT_EQ(result, num_threads);

    counter.free();
}

/*****************************************************************
 * Test cases for RS_SpinLock
 *
 * Test cases:
 * 1. Spinlock guarded counter on the provider.
 * 2. Spinlock guarded counter on the host provider.
 * 3. Clear resets the spinlock state to zero.
 *****************************************************************/

TEST_F(RS_AtomicTest, SpinLockCounterOnProvider) {
    const int num_threads = 100;

    // The lock holds an atomic and cannot be copied, so allocate it in device
    // memory and zero its bytes, which leaves it in the cleared state.
    RS_MemoryPtr lock = provider.allocate(sizeof(RS_SpinLock));
    lock.set(0, sizeof(RS_SpinLock));
    RS_MemoryPtr counter = provider.allocate(sizeof(int));
    counter.set(0, sizeof(int));

    provider.run<RSCF_SpinLockCounter>(
        {static_cast<RSize>(num_threads)}, lock.as<RS_SpinLock>(), counter.as<int>(), num_threads
    );

    int result = -1;
    counter.copyTo(RS_MemoryPtr(&result, sizeof(int)), sizeof(int));
    EXPECT_EQ(result, num_threads);

    lock.free();
    counter.free();
}

TEST_F(RS_AtomicTest, SpinLockCounterOnHostProvider) {
    const int num_threads = 100;

    RS_MemoryPtr lock = host_provider.allocate(sizeof(RS_SpinLock));
    lock.set(0, sizeof(RS_SpinLock));
    RS_MemoryPtr counter = host_provider.allocate(sizeof(int));
    counter.set(0, sizeof(int));

    host_provider.run<RSCF_SpinLockCounter>(
        {static_cast<RSize>(num_threads)}, lock.as<RS_SpinLock>(), counter.as<int>(), num_threads
    );

    int result = -1;
    counter.copyTo(RS_MemoryPtr(&result, sizeof(int)), sizeof(int));
    EXPECT_EQ(result, num_threads);

    lock.free();
    counter.free();
}

TEST_F(RS_AtomicTest, SpinLockClearOnProvider) {
    RS_MemoryPtr lock = provider.allocate(sizeof(RS_SpinLock));
    RS_MemoryPtr state = provider.allocate(sizeof(unsigned int));

    // Start from a non zero state to confirm clear resets it.
    lock.set(1, sizeof(RS_SpinLock));
    unsigned int init_state = 1;
    state.copyFrom(RS_MemoryPtr(&init_state, sizeof(unsigned int)), sizeof(unsigned int));

    provider.run<RSCF_ClearSpinLock>({1}, lock.as<RS_SpinLock>(), state.as<unsigned int>());

    unsigned int host_state = 1;
    state.copyTo(RS_MemoryPtr(&host_state, sizeof(unsigned int)), sizeof(unsigned int));
    EXPECT_EQ(host_state, 0u);

    lock.free();
    state.free();
}

/*****************************************************************
 * Test cases for RS_TicketLock
 *
 * Test cases:
 * 1. Ticket lock guarded counter on the provider.
 * 2. Ticket lock guarded counter on the host provider.
 *****************************************************************/

TEST_F(RS_AtomicTest, TicketLockCounterOnProvider) {
    const int num_threads = 100;

    // The lock holds an atomic and cannot be copied, so allocate it in device
    // memory and zero its bytes, which leaves it in the cleared state.
    RS_MemoryPtr lock = provider.allocate(sizeof(RS_TicketLock));
    lock.set(0, sizeof(RS_TicketLock));
    RS_MemoryPtr counter = provider.allocate(sizeof(int));
    counter.set(0, sizeof(int));

    provider.run<RSCF_TicketLockCounter>(
        {static_cast<RSize>(num_threads)}, lock.as<RS_TicketLock>(), counter.as<int>(), num_threads
    );

    int result = -1;
    counter.copyTo(RS_MemoryPtr(&result, sizeof(int)), sizeof(int));
    EXPECT_EQ(result, num_threads);

    lock.free();
    counter.free();
}

TEST_F(RS_AtomicTest, TicketLockCounterOnHostProvider) {
    const int num_threads = 100;

    RS_MemoryPtr lock = host_provider.allocate(sizeof(RS_TicketLock));
    lock.set(0, sizeof(RS_TicketLock));
    RS_MemoryPtr counter = host_provider.allocate(sizeof(int));
    counter.set(0, sizeof(int));

    host_provider.run<RSCF_TicketLockCounter>(
        {static_cast<RSize>(num_threads)}, lock.as<RS_TicketLock>(), counter.as<int>(), num_threads
    );

    int result = -1;
    counter.copyTo(RS_MemoryPtr(&result, sizeof(int)), sizeof(int));
    EXPECT_EQ(result, num_threads);

    lock.free();
    counter.free();
}
