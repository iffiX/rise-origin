#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_buffer.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary for RS_DoubleBuffer:
 *
 * RS_DoubleBuffer is a double buffered container with host/device
 * regions A/B. It supports swap and copy semantics where the inactive
 * buffer is copied while the active buffer is being used. This enables
 * pipelined data transfer between host and device.
 *
 * - Construction & Initialization
 *   Constructor with max_chunk_num and max_data_size, init(), free()
 *
 * - Host Operations
 *   pushHost(), popHost(), getHostNum(), getHostChunk(), clearHost()
 *
 * - Device Operations (via compute functions)
 *   pushDevice(), getDeviceNum(), getDeviceChunk()
 *
 * - Swap and Copy Operations
 *   swapAndCopy() for host to device and device to host transfers
 *
 * - Synchronization Verification
 *   Data integrity across buffer swaps, async copy with waitForCopyFinish()
 *****************************************************************/

/*****************************************************************
 * Test Summary for RS_DirectBuffer:
 *
 * RS_DirectBuffer is a single buffered container with immediate
 * synchronization between host and device. Unlike double buffer, it
 * provides a direct sync() method for explicit data transfer.
 *
 * - Construction & Initialization
 *   Constructor with max_data_size, init(), free()
 *
 * - Host Operations
 *   pushHost(), getHostChunk(), clearHost()
 *
 * - Device Operations (via compute functions)
 *   pushDevice(), getDeviceChunk()
 *
 * - Synchronization Operations
 *   sync() for immediate host to device data transfer
 *****************************************************************/

/*****************************************************************
 * Compute Functions for Device Side Buffer Operations
 *****************************************************************/

// Write data from a source array into a device double buffer chunk.
class RSCF_WriteSourceToDoubleBuffer
    : public RS_ComputeFunction<RSCF_WriteSourceToDoubleBuffer> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_DoubleBuffer buffer, RS_Array<int, 1> source, RSize num_elements, RS_Array<int, 1> success) {
        if (index1D() == 0) {
            RS_MemoryPtr chunk = buffer.pushDevice(sizeof(int) * num_elements);
            if (chunk.ptr() == nullptr) {
                success[0] = 0;
                return;
            }
            success[0] = 1;
            int* dst = chunk.as<int>();
            for (RSize i = 0; i < num_elements; ++i) {
                dst[i] = source[i];
            }
        }
    }
};

// Read data from a device double buffer chunk into a destination array.
class RSCF_ReadDoubleBufferToDestination
    : public RS_ComputeFunction<RSCF_ReadDoubleBufferToDestination> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_DoubleBuffer buffer, RIndex chunk_index, RS_Array<int, 1> dest, RSize num_elements) {
        if (index1D() == 0) {
            RS_MemoryPtr chunk = buffer.getDeviceChunk(chunk_index);
            int* src = chunk.as<int>();
            for (RSize i = 0; i < num_elements; ++i) {
                dest[i] = src[i];
            }
        }
    }
};

// Get the device chunk count of a double buffer.
class RSCF_GetDoubleBufferDeviceNum
    : public RS_ComputeFunction<RSCF_GetDoubleBufferDeviceNum> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_DoubleBuffer buffer, RS_Array<RSize, 1> result) {
        if (index1D() == 0) {
            result[0] = buffer.getDeviceNum();
        }
    }
};

// Write data from a source array into a direct buffer chunk.
class RSCF_WriteSourceToDirectBuffer
    : public RS_ComputeFunction<RSCF_WriteSourceToDirectBuffer> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_DirectBuffer buffer, RS_Array<int, 1> source, RSize num_elements, RS_Array<int, 1> success) {
        if (index1D() == 0) {
            RS_MemoryPtr chunk = buffer.pushDevice(sizeof(int) * num_elements);
            if (chunk.ptr() == nullptr) {
                success[0] = 0;
                return;
            }
            success[0] = 1;
            int* dst = chunk.as<int>();
            for (RSize i = 0; i < num_elements; ++i) {
                dst[i] = source[i];
            }
        }
    }
};

// Read data from a direct buffer chunk into a destination array.
class RSCF_ReadDirectBufferToDestination
    : public RS_ComputeFunction<RSCF_ReadDirectBufferToDestination> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_DirectBuffer buffer, RIndex chunk_index, RS_Array<int, 1> dest, RSize num_elements) {
        if (index1D() == 0) {
            RS_MemoryPtr chunk = buffer.getDeviceChunk(chunk_index);
            int* src = chunk.as<int>();
            for (RSize i = 0; i < num_elements; ++i) {
                dest[i] = src[i];
            }
        }
    }
};

/*****************************************************************
 * RS_DoubleBuffer Test Suite
 *****************************************************************/

class RS_DoubleBufferTest : public ::testing::Test {
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
 * Test cases for RS_DoubleBuffer construction and initialization
 *
 * Test cases:
 * 1. Initialize with valid parameters
 * 2. Constructor with valid parameters
 * 3. Free non-empty buffer
 * 4. Free empty buffer
 * 5. Multiple frees
 *****************************************************************/

TEST_F(RS_DoubleBufferTest, InitializeWithValidParameters) {
    RS_DoubleBuffer buffer;
    RSize max_chunk_num = 10;
    RSize max_data_size = 1024;

    buffer.init(max_chunk_num, max_data_size, &provider);

    EXPECT_NE(buffer.mem_provider, nullptr);
    EXPECT_TRUE(buffer.is_a_active_host);

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, ConstructorWithValidParameters) {
    RSize max_chunk_num = 20;
    RSize max_data_size = 2048;

    RS_DoubleBuffer buffer(max_chunk_num, max_data_size, &provider);

    EXPECT_NE(buffer.mem_provider, nullptr);
    EXPECT_TRUE(buffer.is_a_active_host);

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, FreeNonEmptyBuffer) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    buffer.pushHost(100);
    buffer.pushHost(200);

    EXPECT_NO_THROW(buffer.free());
    EXPECT_EQ(buffer.mem_provider, nullptr);
}

TEST_F(RS_DoubleBufferTest, FreeEmptyBuffer) {
    RS_DoubleBuffer buffer(10, 1024, &provider);
    EXPECT_NO_THROW(buffer.free());
    EXPECT_EQ(buffer.mem_provider, nullptr);
}

TEST_F(RS_DoubleBufferTest, MultipleFrees) {
    RS_DoubleBuffer buffer(10, 1024, &provider);
    buffer.free();
    EXPECT_NO_THROW(buffer.free());
    EXPECT_EQ(buffer.mem_provider, nullptr);
}

/*****************************************************************
 * Test cases for RS_DoubleBuffer host operations
 *
 * Test cases:
 * 1. Push single chunk
 * 2. Push multiple chunks
 * 3. Push until chunk capacity reached
 * 4. Push until data capacity reached
 * 5. Pop chunk
 * 6. Get host chunk
 * 7. Clear host buffer
 * 8. Check empty buffer
 *****************************************************************/

TEST_F(RS_DoubleBufferTest, PushSingleChunk) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    auto chunk = buffer.pushHost(100);

    EXPECT_NE(chunk.ptr(), nullptr);
    EXPECT_EQ(chunk.size(), 100);
    EXPECT_EQ(buffer.getHostNum(), 1);

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, PushMultipleChunks) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    auto chunk1 = buffer.pushHost(100);
    auto chunk2 = buffer.pushHost(200);
    auto chunk3 = buffer.pushHost(150);

    EXPECT_NE(chunk1.ptr(), nullptr);
    EXPECT_NE(chunk2.ptr(), nullptr);
    EXPECT_NE(chunk3.ptr(), nullptr);
    EXPECT_EQ(buffer.getHostNum(), 3);

    // Verify chunks are non-overlapping
    EXPECT_NE(chunk1.ptr(), chunk2.ptr());
    EXPECT_NE(chunk2.ptr(), chunk3.ptr());

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, PushUntilChunkCapacityReached) {
    RS_DoubleBuffer buffer(3, 500, &provider);

    auto chunk1 = buffer.pushHost(150);
    auto chunk2 = buffer.pushHost(150);
    auto chunk3 = buffer.pushHost(150);

    EXPECT_NE(chunk1.ptr(), nullptr);
    EXPECT_NE(chunk2.ptr(), nullptr);
    EXPECT_NE(chunk3.ptr(), nullptr);

    // This should fail since max chunks is reached
    auto chunk4 = buffer.pushHost(50);
    EXPECT_EQ(chunk4.ptr(), nullptr);

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, PushUntilDataCapacityReached) {
    RS_DoubleBuffer buffer(10, 300, &provider);

    auto chunk1 = buffer.pushHost(100);
    auto chunk2 = buffer.pushHost(100);
    auto chunk3 = buffer.pushHost(100);

    EXPECT_NE(chunk1.ptr(), nullptr);
    EXPECT_NE(chunk2.ptr(), nullptr);
    EXPECT_NE(chunk3.ptr(), nullptr);

    // This should fail since data capacity is reached
    auto chunk4 = buffer.pushHost(100);
    EXPECT_EQ(chunk4.ptr(), nullptr);

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, PopChunk) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    buffer.pushHost(100);
    buffer.pushHost(200);
    EXPECT_EQ(buffer.getHostNum(), 2);

    buffer.popHost();
    EXPECT_EQ(buffer.getHostNum(), 1);

    buffer.popHost();
    EXPECT_EQ(buffer.getHostNum(), 0);

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, GetHostChunk) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    auto chunk1 = buffer.pushHost(sizeof(int) * 10);
    auto chunk2 = buffer.pushHost(sizeof(int) * 20);

    int* data1 = chunk1.as<int>();
    for (int i = 0; i < 10; ++i) data1[i] = i;

    int* data2 = chunk2.as<int>();
    for (int i = 0; i < 20; ++i) data2[i] = 100 + i;

    auto retrieved1 = buffer.getHostChunk(0);
    auto retrieved2 = buffer.getHostChunk(1);

    EXPECT_EQ(retrieved1.ptr(), chunk1.ptr());
    EXPECT_EQ(retrieved2.ptr(), chunk2.ptr());

    int* r1 = retrieved1.as<int>();
    int* r2 = retrieved2.as<int>();

    for (int i = 0; i < 10; ++i) EXPECT_EQ(r1[i], i);
    for (int i = 0; i < 20; ++i) EXPECT_EQ(r2[i], 100 + i);

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, ClearHostBuffer) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    buffer.pushHost(100);
    buffer.pushHost(200);
    EXPECT_EQ(buffer.getHostNum(), 2);

    buffer.clearHost();
    EXPECT_EQ(buffer.getHostNum(), 0);
    EXPECT_TRUE(buffer.isHostEmpty());

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, CheckEmptyBuffer) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    EXPECT_TRUE(buffer.isHostEmpty());

    buffer.pushHost(100);
    EXPECT_FALSE(buffer.isHostEmpty());

    buffer.popHost();
    EXPECT_TRUE(buffer.isHostEmpty());

    buffer.free();
}

/*****************************************************************
 * Test cases for RS_DoubleBuffer device operations
 *
 * Test cases:
 * 1. Push chunk from device
 * 2. Push multiple chunks from device
 *****************************************************************/

TEST_F(RS_DoubleBufferTest, PushChunkFromDevice) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    RS_Array<int, 1> source({10}, &provider);
    int src[10];
    for (int i = 0; i < 10; ++i) src[i] = i * 100;
    source.write(src);

    RS_Array<int, 1> success({1}, &provider);
    provider.run<RSCF_WriteSourceToDoubleBuffer>({1}, buffer, source, 10, success);
    RS_Array<int, 1> host_success = success.move(&host_provider);
    EXPECT_EQ(host_success[0], 1);

    RS_Array<RSize, 1> num({1}, &provider);
    provider.run<RSCF_GetDoubleBufferDeviceNum>({1}, buffer, num);
    RS_Array<RSize, 1> host_num = num.move(&host_provider);
    EXPECT_EQ(host_num[0], 1u);

    source.free();
    success.free();
    host_success.free();
    num.free();
    host_num.free();
    buffer.free();
}

TEST_F(RS_DoubleBufferTest, PushMultipleChunksFromDevice) {
    RS_DoubleBuffer buffer(10, 2048, &provider);

    RS_Array<int, 1> success({1}, &provider);

    for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
        RSize num_elements = 10 + chunk_idx * 5;
        RS_Array<int, 1> source({num_elements}, &provider);
        std::vector<int> src(num_elements);
        for (RSize i = 0; i < num_elements; ++i) src[i] = chunk_idx * 1000 + static_cast<int>(i);
        source.write(src.data());

        provider.run<RSCF_WriteSourceToDoubleBuffer>({1}, buffer, source, num_elements, success);
        RS_Array<int, 1> host_success = success.move(&host_provider);
        EXPECT_EQ(host_success[0], 1);

        source.free();
        host_success.free();
    }

    RS_Array<RSize, 1> num({1}, &provider);
    provider.run<RSCF_GetDoubleBufferDeviceNum>({1}, buffer, num);
    RS_Array<RSize, 1> host_num = num.move(&host_provider);
    EXPECT_EQ(host_num[0], 3u);

    success.free();
    num.free();
    host_num.free();
    buffer.free();
}

/*****************************************************************
 * Test cases for RS_DoubleBuffer swap and copy host to device
 *
 * Test cases:
 * 1. Swap and copy single chunk host to device
 * 2. Swap and copy multiple chunks host to device
 * 3. Active buffer switches after swap
 * 4. New active buffer is cleared after swap
 *****************************************************************/

TEST_F(RS_DoubleBufferTest, SwapAndCopySingleChunkHostToDevice) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    auto chunk = buffer.pushHost(sizeof(int) * 10);
    int* host_data = chunk.as<int>();
    for (int i = 0; i < 10; ++i) host_data[i] = i * 10;

    EXPECT_TRUE(buffer.is_a_active_host);

    // Swap and copy host to device, now B is active
    buffer.swapAndCopy(false, true, true);
    EXPECT_FALSE(buffer.is_a_active_host);

    // Swap again so that A is active on the device side
    buffer.swapAndCopy(false, true, true);

    RS_Array<int, 1> dest({10}, &provider);
    provider.run<RSCF_ReadDoubleBufferToDestination>({1}, buffer, 0, dest, 10);
    RS_Array<int, 1> host_dest = dest.move(&host_provider);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(host_dest[i], i * 10);
    }

    dest.free();
    host_dest.free();
    buffer.free();
}

TEST_F(RS_DoubleBufferTest, SwapAndCopyMultipleChunksHostToDevice) {
    RS_DoubleBuffer buffer(10, 2048, &provider);

    auto chunk1 = buffer.pushHost(sizeof(int) * 10);
    auto chunk2 = buffer.pushHost(sizeof(int) * 15);

    int* data1 = chunk1.as<int>();
    for (int i = 0; i < 10; ++i) data1[i] = 100 + i;

    int* data2 = chunk2.as<int>();
    for (int i = 0; i < 15; ++i) data2[i] = 200 + i;

    buffer.swapAndCopy(false, true, true);
    buffer.swapAndCopy(false, true, true);

    RS_Array<RSize, 1> num({1}, &provider);
    provider.run<RSCF_GetDoubleBufferDeviceNum>({1}, buffer, num);
    RS_Array<RSize, 1> host_num = num.move(&host_provider);
    EXPECT_EQ(host_num[0], 2u);

    RS_Array<int, 1> dest1({10}, &provider);
    provider.run<RSCF_ReadDoubleBufferToDestination>({1}, buffer, 0, dest1, 10);
    RS_Array<int, 1> host_dest1 = dest1.move(&host_provider);
    for (int i = 0; i < 10; ++i) EXPECT_EQ(host_dest1[i], 100 + i);

    RS_Array<int, 1> dest2({15}, &provider);
    provider.run<RSCF_ReadDoubleBufferToDestination>({1}, buffer, 1, dest2, 15);
    RS_Array<int, 1> host_dest2 = dest2.move(&host_provider);
    for (int i = 0; i < 15; ++i) EXPECT_EQ(host_dest2[i], 200 + i);

    num.free();
    host_num.free();
    dest1.free();
    host_dest1.free();
    dest2.free();
    host_dest2.free();
    buffer.free();
}

TEST_F(RS_DoubleBufferTest, ActiveBufferSwitchesAfterSwap) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    EXPECT_TRUE(buffer.is_a_active_host);

    buffer.swapAndCopy(false, true, true);
    EXPECT_FALSE(buffer.is_a_active_host);

    buffer.swapAndCopy(false, true, true);
    EXPECT_TRUE(buffer.is_a_active_host);

    buffer.swapAndCopy(false, true, true);
    EXPECT_FALSE(buffer.is_a_active_host);

    buffer.free();
}

TEST_F(RS_DoubleBufferTest, NewActiveBufferIsClearedAfterSwap) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    buffer.pushHost(100);
    buffer.pushHost(200);
    EXPECT_EQ(buffer.getHostNum(), 2);

    // Swap to B
    buffer.swapAndCopy(false, true, true);
    EXPECT_EQ(buffer.getHostNum(), 0);

    buffer.pushHost(150);
    EXPECT_EQ(buffer.getHostNum(), 1);

    // Swap back to A
    buffer.swapAndCopy(false, true, true);
    EXPECT_EQ(buffer.getHostNum(), 0);

    buffer.free();
}

/*****************************************************************
 * Test cases for RS_DoubleBuffer swap and copy device to host
 *
 * Test cases:
 * 1. Swap and copy single chunk device to host
 * 2. Swap and copy multiple chunks device to host
 *****************************************************************/

TEST_F(RS_DoubleBufferTest, SwapAndCopySingleChunkDeviceToHost) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    RS_Array<int, 1> source({10}, &provider);
    int src[10];
    for (int i = 0; i < 10; ++i) src[i] = 50 + i;
    source.write(src);

    RS_Array<int, 1> success({1}, &provider);
    provider.run<RSCF_WriteSourceToDoubleBuffer>({1}, buffer, source, 10, success);

    // Swap and copy device to host, now B is active on host
    buffer.swapAndCopy(true, true, true);
    // Swap again so that A is active on host
    buffer.swapAndCopy(true, true, true);

    EXPECT_EQ(buffer.getHostNum(), 1);
    auto host_chunk = buffer.getHostChunk(0);
    int* h_data = host_chunk.as<int>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(h_data[i], 50 + i);
    }

    source.free();
    success.free();
    buffer.free();
}

TEST_F(RS_DoubleBufferTest, SwapAndCopyMultipleChunksDeviceToHost) {
    RS_DoubleBuffer buffer(10, 2048, &provider);

    RS_Array<int, 1> success({1}, &provider);

    RS_Array<int, 1> source1({10}, &provider);
    int s1[10];
    for (int i = 0; i < 10; ++i) s1[i] = 100 + i;
    source1.write(s1);
    provider.run<RSCF_WriteSourceToDoubleBuffer>({1}, buffer, source1, 10, success);

    RS_Array<int, 1> source2({15}, &provider);
    int s2[15];
    for (int i = 0; i < 15; ++i) s2[i] = 200 + i;
    source2.write(s2);
    provider.run<RSCF_WriteSourceToDoubleBuffer>({1}, buffer, source2, 15, success);

    buffer.swapAndCopy(true, true, true);
    buffer.swapAndCopy(true, true, true);

    EXPECT_EQ(buffer.getHostNum(), 2);

    auto chunk1 = buffer.getHostChunk(0);
    auto chunk2 = buffer.getHostChunk(1);

    int* data1 = chunk1.as<int>();
    int* data2 = chunk2.as<int>();

    for (int i = 0; i < 10; ++i) EXPECT_EQ(data1[i], 100 + i);
    for (int i = 0; i < 15; ++i) EXPECT_EQ(data2[i], 200 + i);

    source1.free();
    source2.free();
    success.free();
    buffer.free();
}

/*****************************************************************
 * Test cases for RS_DoubleBuffer pipeline behavior
 *
 * Test cases:
 * 1. Async copy with waitForCopyFinish
 * 2. Empty swap does not crash
 *****************************************************************/

TEST_F(RS_DoubleBufferTest, AsyncCopyWithWaitForFinish) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    auto chunk = buffer.pushHost(sizeof(int) * 100);
    int* data = chunk.as<int>();
    for (int i = 0; i < 100; ++i) data[i] = i * 2;

    // Start the copy without syncing
    buffer.swapAndCopy(false, false, true);
    buffer.waitForCopyFinish();

    // Make A active on device and read back
    buffer.swapAndCopy(false, true, true);

    RS_Array<int, 1> dest({100}, &provider);
    provider.run<RSCF_ReadDoubleBufferToDestination>({1}, buffer, 0, dest, 100);
    RS_Array<int, 1> host_dest = dest.move(&host_provider);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(host_dest[i], i * 2);
    }

    dest.free();
    host_dest.free();
    buffer.free();
}

TEST_F(RS_DoubleBufferTest, EmptySwapOperation) {
    RS_DoubleBuffer buffer(10, 1024, &provider);

    EXPECT_NO_THROW(buffer.swapAndCopy(false, true, true));
    EXPECT_NO_THROW(buffer.swapAndCopy(true, true, true));

    buffer.free();
}

/*****************************************************************
 * Test cases for RS_DoubleBuffer uninitialized operations
 *
 * Test cases:
 * 1. Operations on an uninitialized buffer throw
 *****************************************************************/

TEST_F(RS_DoubleBufferTest, UninitializedOperations) {
    RS_DoubleBuffer buffer;

    EXPECT_THROW(buffer.pushHost(100), RS_UninitializedException);
    EXPECT_THROW(buffer.popHost(), RS_UninitializedException);
    EXPECT_THROW(buffer.swapAndCopy(false), RS_UninitializedException);
}

/*****************************************************************
 * RS_DirectBuffer Test Suite
 *****************************************************************/

class RS_DirectBufferTest : public ::testing::Test {
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
 * Test cases for RS_DirectBuffer construction and initialization
 *
 * Test cases:
 * 1. Initialize with valid parameters
 * 2. Constructor with valid parameters
 * 3. Free non-empty buffer
 * 4. Free empty buffer
 * 5. Multiple frees
 *****************************************************************/

TEST_F(RS_DirectBufferTest, InitializeWithValidParameters) {
    RS_DirectBuffer buffer;
    RSize max_data_size = 1024;

    buffer.init(max_data_size, &provider);

    EXPECT_NE(buffer.mem_provider, nullptr);

    buffer.free();
}

TEST_F(RS_DirectBufferTest, ConstructorWithValidParameters) {
    RSize max_data_size = 2048;

    RS_DirectBuffer buffer(max_data_size, &provider);

    EXPECT_NE(buffer.mem_provider, nullptr);

    buffer.free();
}

TEST_F(RS_DirectBufferTest, FreeNonEmptyBuffer) {
    RS_DirectBuffer buffer(1024, &provider);

    buffer.pushHost(100);

    EXPECT_NO_THROW(buffer.free());
    EXPECT_EQ(buffer.mem_provider, nullptr);
}

TEST_F(RS_DirectBufferTest, FreeEmptyBuffer) {
    RS_DirectBuffer buffer(1024, &provider);
    EXPECT_NO_THROW(buffer.free());
    EXPECT_EQ(buffer.mem_provider, nullptr);
}

TEST_F(RS_DirectBufferTest, MultipleFrees) {
    RS_DirectBuffer buffer(1024, &provider);
    buffer.free();
    EXPECT_NO_THROW(buffer.free());
    EXPECT_EQ(buffer.mem_provider, nullptr);
}

/*****************************************************************
 * Test cases for RS_DirectBuffer host operations
 *
 * Test cases:
 * 1. Push single chunk
 * 2. Get host chunk with data
 *****************************************************************/

TEST_F(RS_DirectBufferTest, PushSingleChunk) {
    RS_DirectBuffer buffer(1024, &provider);

    auto chunk = buffer.pushHost(100);

    EXPECT_NE(chunk.ptr(), nullptr);

    buffer.free();
}

TEST_F(RS_DirectBufferTest, GetHostChunkWithData) {
    RS_DirectBuffer buffer(1024, &provider);

    auto chunk = buffer.pushHost(sizeof(double) * 5);
    double* data = chunk.as<double>();
    for (int i = 0; i < 5; ++i) data[i] = i * 3.14;

    auto retrieved = buffer.getHostChunk(0);
    double* r = retrieved.as<double>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(r[i], i * 3.14);
    }

    buffer.free();
}

/*****************************************************************
 * Test cases for RS_DirectBuffer device operations
 *
 * Test cases:
 * 1. Push chunk from device
 *****************************************************************/

TEST_F(RS_DirectBufferTest, PushChunkFromDevice) {
    RS_DirectBuffer buffer(1024, &provider);

    RS_Array<int, 1> source({10}, &provider);
    int src[10];
    for (int i = 0; i < 10; ++i) src[i] = 42 + i;
    source.write(src);

    RS_Array<int, 1> success({1}, &provider);
    provider.run<RSCF_WriteSourceToDirectBuffer>({1}, buffer, source, 10, success);
    RS_Array<int, 1> host_success = success.move(&host_provider);
    EXPECT_EQ(host_success[0], 1);

    source.free();
    success.free();
    host_success.free();
    buffer.free();
}

/*****************************************************************
 * Test cases for RS_DirectBuffer synchronization
 *
 * Test cases:
 * 1. Sync host to device single chunk
 * 2. Sync device to host single chunk
 * 3. Empty sync does not crash
 *****************************************************************/

TEST_F(RS_DirectBufferTest, SyncHostToDeviceSingleChunk) {
    RS_DirectBuffer buffer(1024, &provider);

    auto chunk = buffer.pushHost(sizeof(int) * 10);
    int* data = chunk.as<int>();
    for (int i = 0; i < 10; ++i) data[i] = i * 5;

    buffer.sync(false);  // host to device

    RS_Array<int, 1> dest({10}, &provider);
    provider.run<RSCF_ReadDirectBufferToDestination>({1}, buffer, 0, dest, 10);
    RS_Array<int, 1> host_dest = dest.move(&host_provider);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(host_dest[i], i * 5);
    }

    dest.free();
    host_dest.free();
    buffer.free();
}

TEST_F(RS_DirectBufferTest, SyncDeviceToHostSingleChunk) {
    RS_DirectBuffer buffer(1024, &provider);

    RS_Array<int, 1> source({10}, &provider);
    int src[10];
    for (int i = 0; i < 10; ++i) src[i] = 77 + i;
    source.write(src);

    RS_Array<int, 1> success({1}, &provider);
    provider.run<RSCF_WriteSourceToDirectBuffer>({1}, buffer, source, 10, success);

    buffer.sync(true);  // device to host

    auto chunk = buffer.getHostChunk(0);
    int* data = chunk.as<int>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(data[i], 77 + i);
    }

    source.free();
    success.free();
    buffer.free();
}

TEST_F(RS_DirectBufferTest, EmptySyncOperation) {
    RS_DirectBuffer buffer(1024, &provider);

    EXPECT_NO_THROW(buffer.sync(false));
    EXPECT_NO_THROW(buffer.sync(true));

    buffer.free();
}

/*****************************************************************
 * Test cases for RS_DirectBuffer uninitialized operations
 *
 * Test cases:
 * 1. Operations on an uninitialized buffer throw
 *****************************************************************/

TEST_F(RS_DirectBufferTest, UninitializedOperations) {
    RS_DirectBuffer buffer;

    EXPECT_THROW(buffer.pushHost(100), RS_UninitializedException);
    EXPECT_THROW(buffer.sync(false), RS_UninitializedException);
}
