#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/utils/memory/rs_memory.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_CPUBlock backs the block sync abstraction used by compute
 * functions. On the CPU each thread in a block runs as a fiber,
 * and syncThreadsInBlock hands control between fibers so that all
 * threads reach the same point before any of them moves on.
 *
 * - Shared Memory
 *   sharedMemAt1D, sharedMemAt for cross thread reads
 *
 * - Block Sync
 *   syncThreadsInBlock with IsSyncBlockRequired
 *
 * - Memory Providers
 *   CPU provider and the default provider as data source
 *****************************************************************/

class RS_CPUBlockTest : public ::testing::Test {
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
 * Compute function that exercises shared memory and block sync
 *
 * Each thread stores its input value into shared memory, then the
 * first half of the block adds the value of its partner thread in
 * the second half. The two sync points make sure all threads have
 * written before any thread reads.
 *****************************************************************/

class CPUBlockTestFunction : public RS_ComputeFunction<CPUBlockTestFunction> {
   public:
    using IsSyncBlockRequired = std::true_type;

    struct SharedMemLayout {
        int x;
    };

    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_Array<int, 1> input, RS_Array<int, 1> result) {
        int idx = index1D();
        sharedMemAt1D(&SharedMemLayout::x) = input[idx];
        syncThreadsInBlock();

        if (threadIdxX() < (blockDimX() / 2))
            sharedMemAt1D(&SharedMemLayout::x) =
                sharedMemAt1D(&SharedMemLayout::x) +
                sharedMemAt(&SharedMemLayout::x, blockDimX(), threadIdxX() + (blockDimX() / 2));

        syncThreadsInBlock();
        result[idx] = sharedMemAt1D(&SharedMemLayout::x);
    }
};

/*****************************************************************
 * Test cases for RS_CPUBlock shared memory and block sync
 *
 * Test cases:
 * 1. Run on the CPU provider with arrays from the same provider.
 * 2. Run on the CPU provider with arrays from the default provider.
 *****************************************************************/

TEST_F(RS_CPUBlockTest, CPUBlockTest) {
    RS_Array<int, 1> input_array({32}, &host_provider);
    RS_Array<int, 1> result_array({32}, &host_provider);
    input_array.fill(1);
    result_array.fill(0);

    host_provider.run<CPUBlockTestFunction>({32, 32}, input_array, result_array);
    for (int i = 0; i < 16; i++) {
        ASSERT_EQ(result_array[i], 2);
    }
    for (int i = 16; i < 32; i++) {
        ASSERT_EQ(result_array[i], 1);
    }
    input_array.free();
    result_array.free();
}

TEST_F(RS_CPUBlockTest, DefaultProviderCPUBlockTest) {
    RS_Array<int, 1> input_array({32}, &RS_MemoryDefaultProvider::getInstance());
    RS_Array<int, 1> result_array({32}, &RS_MemoryDefaultProvider::getInstance());
    input_array.fill(1);
    result_array.fill(0);

    host_provider.run<CPUBlockTestFunction>({32, 32}, input_array, result_array);
    for (int i = 0; i < 16; i++) {
        ASSERT_EQ(result_array[i], 2);
    }
    for (int i = 16; i < 32; i++) {
        ASSERT_EQ(result_array[i], 1);
    }
    input_array.free();
    result_array.free();
}
