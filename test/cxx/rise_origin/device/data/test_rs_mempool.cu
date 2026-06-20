#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_mempool.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

constexpr RSize ChunkSize = 40;

/*****************************************************************
 * Test Summary:
 *
 * RS_MemPool is a fixed size chunk based memory pool that stores a
 * pointer for each chunk and keeps the host side memory regions.
 *
 * RS_MemArrayPool is an array based memory pool that computes chunk
 * pointers from a base pointer instead of storing them.
 *
 * - Construction & Initialization
 *   Constructor with chunk count, init(), free()
 *
 * - Memory Management
 *   grow(), resize() with optional data copy, clear()
 *
 * - Data Transfer
 *   copy(), move()
 *
 * - Chunk Allocation
 *   acquire() for thread safe allocation on CPU and on the device
 *   release() to return chunks to the pool
 *   getChunkPtr() to retrieve chunk pointers
 *   getAllocatedNum(), getAllocatedRatio()
 *****************************************************************/

class RS_MemPoolTest : public ::testing::Test {
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

// A bare compute function that acquires chunks on the device and writes
// the acquired indices into a result buffer that is moved to the host.
class RSCF_AcquireChunks : public RS_ComputeFunction<RSCF_AcquireChunks> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_MemPool<ChunkSize> mem_pool, RS_Vector<RIndex> acquired, int num_acquisitions) {
        RIndex tid = index1D();
        RIndex last_chunk = RS_NULL_INDEX;

        for (int i = 0; i < num_acquisitions; ++i) {
            RIndex index = mem_pool.acquire(last_chunk, tid);
            acquired[tid * num_acquisitions + i] = index;
            last_chunk = index;
        }
    }
};

/*****************************************************************
 * Test cases for the RS_MemPool constructor
 *
 * Test cases:
 * 1. Initialize with zero chunks.
 * 2. Initialize with non zero chunks.
 *****************************************************************/

TEST_F(RS_MemPoolTest, InitializationWithZeroChunks) {
    constexpr RSize NumChunks = 0;
    RS_MemPool<ChunkSize> mem_pool(NumChunks, &provider);

    EXPECT_EQ(mem_pool.chunk_num, 0);
    EXPECT_TRUE(mem_pool.chunk_mask.isAllOff());
    EXPECT_EQ(mem_pool.chunk_ptrs.size(), 0);
    EXPECT_EQ(mem_pool.getAllocatedNum(), 0);

    mem_pool.free();
}

TEST_F(RS_MemPoolTest, InitializationWithNonZeroChunks) {
    constexpr RSize NumChunks = 100;
    RS_MemPool<ChunkSize> mem_pool(NumChunks, &provider);

    EXPECT_EQ(mem_pool.chunk_num, NumChunks);

    auto h_chunk_mask = mem_pool.chunk_mask.move(&host_provider);
    for (RSize i = 0; i < NumChunks; ++i) {
        EXPECT_FALSE(h_chunk_mask.isOn(i));
    }
    h_chunk_mask.free();

    auto h_chunk_ptrs = mem_pool.chunk_ptrs.move(&host_provider);
    for (RSize i = 0; i < NumChunks; ++i) {
        EXPECT_NE(h_chunk_ptrs.get(i), nullptr);
    }
    h_chunk_ptrs.free();

    EXPECT_EQ(mem_pool.getAllocatedNum(), 0);
    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemPool::grow()
 *
 * Test cases:
 * 1. Grow by non zero chunks.
 *****************************************************************/

TEST_F(RS_MemPoolTest, GrowNonZeroChunks) {
    constexpr RSize InitialChunks = 50;
    constexpr RSize AdditionalChunks = 50;
    constexpr RSize TotalChunks = InitialChunks + AdditionalChunks;
    RS_MemPool<ChunkSize> mem_pool(InitialChunks, &provider);

    mem_pool.grow(AdditionalChunks);
    EXPECT_EQ(mem_pool.chunk_num, TotalChunks);

    auto h_chunk_mask = mem_pool.chunk_mask.move(&host_provider);
    EXPECT_EQ(h_chunk_mask.size(), TotalChunks);
    for (RSize i = 0; i < TotalChunks; ++i) {
        EXPECT_FALSE(h_chunk_mask.isOn(i));
    }
    h_chunk_mask.free();

    auto h_chunk_ptrs = mem_pool.chunk_ptrs.move(&host_provider);
    EXPECT_EQ(h_chunk_ptrs.size(), TotalChunks);
    for (RSize i = 0; i < TotalChunks; ++i) {
        EXPECT_NE(h_chunk_ptrs.get(i), nullptr);
    }
    h_chunk_ptrs.free();

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemPool::move()
 *
 * Test cases:
 * 1. Move an empty memory pool.
 * 2. Move a memory pool with allocated chunks.
 * 3. Move to a different memory provider.
 *****************************************************************/

TEST_F(RS_MemPoolTest, MoveEmptyMemoryPool) {
    RS_MemPool<ChunkSize> mem_pool;
    EXPECT_THROW(mem_pool.move(&provider), RS_UninitializedException);
}

TEST_F(RS_MemPoolTest, MoveMemoryPoolWithAllocatedChunks) {
    constexpr RSize NumChunks = 100;
    RS_MemPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    RIndex chunk_index = mem_pool.acquire();
    void* chunk_ptr = mem_pool.getChunkPtr(chunk_index);
    ASSERT_NE(chunk_ptr, nullptr);

    char data[ChunkSize];
    memset(data, 42, ChunkSize);
    memset(chunk_ptr, 42, ChunkSize);

    auto moved_mem_pool = mem_pool.move(&provider);

    EXPECT_EQ(moved_mem_pool.chunk_num, NumChunks);
    EXPECT_EQ(moved_mem_pool.getAllocatedNum(), 1);
    EXPECT_EQ(moved_mem_pool.provider(), &provider);

    auto moved_mem_pool2 = moved_mem_pool.move(&host_provider);

    void* moved_chunk_ptr = moved_mem_pool2.getChunkPtr(chunk_index);
    ASSERT_NE(moved_chunk_ptr, nullptr);
    EXPECT_EQ(memcmp(data, moved_chunk_ptr, ChunkSize), 0);

    mem_pool.free();
    moved_mem_pool.free();
    moved_mem_pool2.free();
}

TEST_F(RS_MemPoolTest, MoveToDifferentMemoryProvider) {
    constexpr RSize NumChunks = 100;
    RS_MemPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    RS_MemPool<ChunkSize> moved_mem_pool = mem_pool.move(&provider);
    EXPECT_EQ(moved_mem_pool.provider(), &provider);

    mem_pool.free();
    moved_mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemPool::acquire() on the device
 *
 * Test cases:
 * 1. Acquire within capacity.
 * 2. Acquire beyond capacity.
 *****************************************************************/

TEST_F(RS_MemPoolTest, AcquireWithinCapacityTest) {
    constexpr RSize NumChunks = 100;
    constexpr int NumThreads = 2;
    constexpr int NumAcquisitions = 5;
    constexpr RSize TotalAcquisitions = NumThreads * NumAcquisitions;
    RS_MemPool<ChunkSize> mem_pool(NumChunks, &provider);

    RS_Vector<RIndex> d_acquired(TotalAcquisitions, &provider);
    d_acquired.fill(RS_NULL_INDEX);

    provider.run<RSCF_AcquireChunks>({NumThreads}, mem_pool, d_acquired, NumAcquisitions);

    RS_Vector<RIndex> h_acquired = d_acquired.move(&host_provider);
    auto h_chunk_mask = mem_pool.chunk_mask.move(&host_provider);

    int success_count = 0;
    for (RSize i = 0; i < TotalAcquisitions; ++i) {
        RIndex acquired_idx = h_acquired[i];
        if (acquired_idx != RS_NULL_INDEX) {
            success_count++;
            EXPECT_TRUE(h_chunk_mask.isOn(acquired_idx));
        }
    }
    EXPECT_EQ(success_count, TotalAcquisitions);
    EXPECT_EQ(h_chunk_mask.count(), TotalAcquisitions);
    EXPECT_EQ(mem_pool.getAllocatedNum(), TotalAcquisitions);

    h_chunk_mask.free();
    d_acquired.free();
    h_acquired.free();
    mem_pool.free();
}

TEST_F(RS_MemPoolTest, AcquireBeyondCapacityTest) {
    constexpr RSize NumChunks = 100;
    constexpr int NumThreads = 8;
    constexpr int NumAcquisitions = 20;
    constexpr RSize TotalAcquisitions = NumThreads * NumAcquisitions;
    RS_MemPool<ChunkSize> mem_pool(NumChunks, &provider);

    RS_Vector<RIndex> d_acquired(TotalAcquisitions, &provider);
    d_acquired.fill(RS_NULL_INDEX);

    provider.run<RSCF_AcquireChunks>({NumThreads}, mem_pool, d_acquired, NumAcquisitions);

    RS_Vector<RIndex> h_acquired = d_acquired.move(&host_provider);
    auto h_chunk_mask = mem_pool.chunk_mask.move(&host_provider);

    int success_count = 0;
    for (RSize i = 0; i < TotalAcquisitions; ++i) {
        RIndex acquired_idx = h_acquired[i];
        if (acquired_idx != RS_NULL_INDEX) {
            success_count++;
            EXPECT_TRUE(h_chunk_mask.isOn(acquired_idx));
        }
    }
    EXPECT_EQ(success_count, NumChunks);
    EXPECT_EQ(h_chunk_mask.count(), NumChunks);
    EXPECT_EQ(mem_pool.getAllocatedNum(), NumChunks);

    h_chunk_mask.free();
    d_acquired.free();
    h_acquired.free();
    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemPool::acquire() on the CPU
 *
 * Test cases:
 * 1. Acquire within capacity using host threads.
 * 2. Acquire beyond capacity using host threads.
 *****************************************************************/

TEST_F(RS_MemPoolTest, AcquireWithinCapacityTestHost) {
    constexpr RSize NumChunks = 100;
    constexpr int NumThreads = 2;
    constexpr int NumAcquisitions = 5;
    RS_MemPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    std::vector<RIndex> acquired_indices(NumThreads * NumAcquisitions);

    std::vector<std::thread> threads;
    for (int t = 0; t < NumThreads; ++t) {
        threads.emplace_back([&, t]() {
            RIndex last_chunk = RS_NULL_INDEX;
            for (int i = 0; i < NumAcquisitions; ++i) {
                RIndex index = mem_pool.acquire(last_chunk, t);
                acquired_indices[t * NumAcquisitions + i] = index;
                last_chunk = index;
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    int success_count = 0;
    for (int i = 0; i < NumThreads * NumAcquisitions; ++i) {
        if (acquired_indices[i] != RS_NULL_INDEX) {
            success_count++;
            EXPECT_TRUE(mem_pool.chunk_mask.isOn(acquired_indices[i]));
            EXPECT_EQ(
                mem_pool.getChunkPtr(acquired_indices[i]),
                mem_pool.chunk_ptrs.get(acquired_indices[i])
            );
        }
    }
    EXPECT_EQ(success_count, NumThreads * NumAcquisitions);
    EXPECT_EQ(mem_pool.chunk_mask.count(), NumThreads * NumAcquisitions);
    EXPECT_EQ(mem_pool.getAllocatedNum(), NumThreads * NumAcquisitions);

    mem_pool.free();
}

TEST_F(RS_MemPoolTest, AcquireBeyondCapacityTestHost) {
    constexpr RSize NumChunks = 100;
    constexpr int NumThreads = 8;
    constexpr int NumAcquisitions = 20;
    RS_MemPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    std::vector<RIndex> acquired_indices(NumThreads * NumAcquisitions);

    std::vector<std::thread> threads;
    for (int t = 0; t < NumThreads; ++t) {
        threads.emplace_back([&, t]() {
            RIndex last_chunk = RS_NULL_INDEX;
            for (int i = 0; i < NumAcquisitions; ++i) {
                RIndex index = mem_pool.acquire(last_chunk, t);
                acquired_indices[t * NumAcquisitions + i] = index;
                last_chunk = index;
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    int success_count = 0;
    for (int i = 0; i < NumThreads * NumAcquisitions; ++i) {
        if (acquired_indices[i] != RS_NULL_INDEX) {
            success_count++;
            EXPECT_TRUE(mem_pool.chunk_mask.isOn(acquired_indices[i]));
        }
    }
    EXPECT_EQ(success_count, NumChunks);
    EXPECT_EQ(mem_pool.chunk_mask.count(), NumChunks);
    EXPECT_EQ(mem_pool.getAllocatedNum(), NumChunks);

    mem_pool.free();
}

/*****************************************************************
 * RS_MemArrayPool tests
 *****************************************************************/

class RS_MemArrayPoolTest : public ::testing::Test {
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

// A bare compute function that acquires chunks from an array based pool.
class RSCF_AcquireChunksArray : public RS_ComputeFunction<RSCF_AcquireChunksArray> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_MemArrayPool<ChunkSize> mem_pool, RS_Vector<RIndex> acquired, int num_acquisitions) {
        RIndex tid = index1D();
        RIndex last_chunk = RS_NULL_INDEX;

        for (int i = 0; i < num_acquisitions; ++i) {
            RIndex index = mem_pool.acquire(last_chunk, tid);
            acquired[tid * num_acquisitions + i] = index;
            last_chunk = index;
        }
    }
};

/*****************************************************************
 * Test cases for the RS_MemArrayPool constructor
 *
 * Test cases:
 * 1. Initialize with zero chunks.
 * 2. Initialize with non zero chunks.
 * 3. Initialize with provider only.
 *****************************************************************/

TEST_F(RS_MemArrayPoolTest, InitializationWithZeroChunks) {
    constexpr RSize NumChunks = 0;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &provider);

    EXPECT_EQ(mem_pool.size(), 0);
    EXPECT_TRUE(mem_pool.empty());
    EXPECT_EQ(mem_pool.getAllocatedNum(), 0);

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, InitializationWithNonZeroChunks) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &provider);

    EXPECT_EQ(mem_pool.size(), NumChunks);
    EXPECT_FALSE(mem_pool.empty());
    EXPECT_EQ(mem_pool.getAllocatedNum(), 0);

    auto h_chunk_mask = mem_pool.chunk_mask.move(&host_provider);
    for (RSize i = 0; i < NumChunks; ++i) {
        EXPECT_FALSE(h_chunk_mask.isOn(i));
    }
    h_chunk_mask.free();

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, InitializationWithProviderOnly) {
    RS_MemArrayPool<ChunkSize> mem_pool(&provider);

    EXPECT_EQ(mem_pool.size(), 0);
    EXPECT_TRUE(mem_pool.empty());
    EXPECT_EQ(mem_pool.provider(), &provider);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemArrayPool::resize()
 *
 * Test cases:
 * 1. Resize larger with copy and verify allocations are preserved.
 * 2. Resize smaller with copy.
 * 3. Resize without copy.
 *****************************************************************/

TEST_F(RS_MemArrayPoolTest, ResizeLargerWithCopy) {
    constexpr RSize InitialSize = 50;
    constexpr RSize NewSize = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(InitialSize, &host_provider);

    RIndex idx1 = mem_pool.acquire();
    RIndex idx2 = mem_pool.acquire();

    mem_pool.resize(NewSize, true);

    EXPECT_EQ(mem_pool.size(), NewSize);
    EXPECT_EQ(mem_pool.getAllocatedNum(), 2);
    EXPECT_TRUE(mem_pool.chunk_mask.isOn(idx1));
    EXPECT_TRUE(mem_pool.chunk_mask.isOn(idx2));

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, ResizeSmallerWithCopy) {
    constexpr RSize InitialSize = 100;
    constexpr RSize NewSize = 50;
    RS_MemArrayPool<ChunkSize> mem_pool(InitialSize, &host_provider);

    mem_pool.acquire();
    mem_pool.acquire();

    mem_pool.resize(NewSize, true);

    EXPECT_EQ(mem_pool.size(), NewSize);

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, ResizeWithoutCopy) {
    constexpr RSize InitialSize = 50;
    constexpr RSize NewSize = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(InitialSize, &host_provider);

    mem_pool.acquire();
    mem_pool.acquire();

    mem_pool.resize(NewSize, false);

    EXPECT_EQ(mem_pool.size(), NewSize);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemArrayPool::clear()
 *
 * Test cases:
 * 1. Clear an empty memory pool.
 * 2. Clear a memory pool with allocated chunks.
 *****************************************************************/

TEST_F(RS_MemArrayPoolTest, ClearEmptyMemoryPool) {
    RS_MemArrayPool<ChunkSize> mem_pool(0, &provider);

    mem_pool.clear();

    EXPECT_EQ(mem_pool.getAllocatedNum(), 0);

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, ClearMemoryPoolWithAllocatedChunks) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    mem_pool.acquire();
    mem_pool.acquire();
    mem_pool.acquire();

    EXPECT_EQ(mem_pool.getAllocatedNum(), 3);

    mem_pool.clear();

    EXPECT_EQ(mem_pool.getAllocatedNum(), 0);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemArrayPool::copy()
 *
 * Test cases:
 * 1. Copy an empty memory pool.
 * 2. Copy a memory pool with allocated chunks.
 * 3. Copy to the same pool.
 *****************************************************************/

TEST_F(RS_MemArrayPoolTest, CopyEmptyMemoryPool) {
    RS_MemArrayPool<ChunkSize> mem_pool1(0, &provider);
    RS_MemArrayPool<ChunkSize> mem_pool2(&provider);

    mem_pool2.copy(mem_pool1);

    EXPECT_EQ(mem_pool2.size(), 0);
    EXPECT_EQ(mem_pool2.getAllocatedNum(), 0);

    mem_pool1.free();
    mem_pool2.free();
}

TEST_F(RS_MemArrayPoolTest, CopyMemoryPoolWithAllocatedChunks) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool1(NumChunks, &host_provider);
    RS_MemArrayPool<ChunkSize> mem_pool2(&provider);

    mem_pool1.acquire();
    mem_pool1.acquire();
    mem_pool1.acquire();

    mem_pool2.copy(mem_pool1);

    EXPECT_EQ(mem_pool2.size(), NumChunks);
    EXPECT_EQ(mem_pool2.getAllocatedNum(), mem_pool1.getAllocatedNum());
    EXPECT_EQ(mem_pool2.provider(), &provider);

    mem_pool1.free();
    mem_pool2.free();
}

TEST_F(RS_MemArrayPoolTest, CopySelfMemoryPool) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &provider);

    mem_pool.copy(mem_pool);

    EXPECT_EQ(mem_pool.size(), NumChunks);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemArrayPool::move()
 *
 * Test cases:
 * 1. Move a memory pool with allocated chunks and verify the data.
 * 2. Move to a different memory provider.
 *****************************************************************/

TEST_F(RS_MemArrayPoolTest, MoveMemoryPoolWithAllocatedChunks) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    RIndex chunk_index = mem_pool.acquire();
    void* chunk_ptr = mem_pool.getChunkPtr(chunk_index);
    ASSERT_NE(chunk_ptr, nullptr);

    char data[ChunkSize];
    memset(data, 42, ChunkSize);
    memset(chunk_ptr, 42, ChunkSize);

    auto moved_mem_pool = mem_pool.move(&provider);

    EXPECT_EQ(moved_mem_pool.size(), NumChunks);
    EXPECT_EQ(moved_mem_pool.getAllocatedNum(), 1);
    EXPECT_EQ(moved_mem_pool.provider(), &provider);

    auto moved_mem_pool2 = moved_mem_pool.move(&host_provider);

    void* moved_chunk_ptr = moved_mem_pool2.getChunkPtr(chunk_index);
    ASSERT_NE(moved_chunk_ptr, nullptr);
    EXPECT_EQ(memcmp(data, moved_chunk_ptr, ChunkSize), 0);

    mem_pool.free();
    moved_mem_pool.free();
    moved_mem_pool2.free();
}

TEST_F(RS_MemArrayPoolTest, MoveToDifferentMemoryProvider) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    RS_MemArrayPool<ChunkSize> moved_mem_pool = mem_pool.move(&provider);
    EXPECT_EQ(moved_mem_pool.provider(), &provider);

    mem_pool.free();
    moved_mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemArrayPool::release()
 *
 * Test cases:
 * 1. Release an acquired chunk.
 * 2. Release an invalid chunk.
 * 3. Release multiple chunks.
 *****************************************************************/

TEST_F(RS_MemArrayPoolTest, ReleaseAcquiredChunk) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    RIndex idx = mem_pool.acquire();
    EXPECT_NE(idx, RS_NULL_INDEX);
    EXPECT_EQ(mem_pool.getAllocatedNum(), 1);

    mem_pool.release(idx);
    EXPECT_EQ(mem_pool.getAllocatedNum(), 0);

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, ReleaseInvalidChunk) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    mem_pool.release(RS_NULL_INDEX);
    mem_pool.release(NumChunks + 10);

    EXPECT_EQ(mem_pool.getAllocatedNum(), 0);

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, ReleaseMultipleChunks) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    RIndex idx1 = mem_pool.acquire();
    RIndex idx2 = mem_pool.acquire();
    RIndex idx3 = mem_pool.acquire();

    EXPECT_EQ(mem_pool.getAllocatedNum(), 3);

    mem_pool.release(idx1);
    EXPECT_EQ(mem_pool.getAllocatedNum(), 2);

    mem_pool.release(idx2);
    EXPECT_EQ(mem_pool.getAllocatedNum(), 1);

    mem_pool.release(idx3);
    EXPECT_EQ(mem_pool.getAllocatedNum(), 0);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemArrayPool::getChunkPtr()
 *
 * Test cases:
 * 1. Get a valid chunk pointer.
 * 2. Get an invalid chunk pointer.
 *****************************************************************/

TEST_F(RS_MemArrayPoolTest, GetValidChunkPtr) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    RIndex idx = mem_pool.acquire();
    void* ptr = mem_pool.getChunkPtr(idx);

    EXPECT_NE(ptr, nullptr);

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, GetInvalidChunkPtr) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    void* ptr = mem_pool.getChunkPtr(RS_NULL_INDEX);

    EXPECT_EQ(ptr, nullptr);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemArrayPool::getAllocatedRatio()
 *
 * Test cases:
 * 1. Ratio with no allocations.
 * 2. Ratio with some allocations.
 * 3. Ratio with full allocations.
 *****************************************************************/

TEST_F(RS_MemArrayPoolTest, GetAllocatedRatioEmpty) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    RFloat ratio = mem_pool.getAllocatedRatio();

    EXPECT_FLOAT_EQ(ratio, 0.0f);

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, GetAllocatedRatioPartial) {
    constexpr RSize NumChunks = 100;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    mem_pool.acquire();
    mem_pool.acquire();
    mem_pool.acquire();
    mem_pool.acquire();
    mem_pool.acquire();

    RFloat ratio = mem_pool.getAllocatedRatio();

    EXPECT_FLOAT_EQ(ratio, 0.05f);

    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, GetAllocatedRatioFull) {
    constexpr RSize NumChunks = 10;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &host_provider);

    for (RSize i = 0; i < NumChunks; ++i) {
        mem_pool.acquire();
    }

    RFloat ratio = mem_pool.getAllocatedRatio();

    EXPECT_FLOAT_EQ(ratio, 1.0f);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_MemArrayPool::acquire() on the device
 *
 * Test cases:
 * 1. Acquire within capacity.
 * 2. Acquire beyond capacity.
 *****************************************************************/

TEST_F(RS_MemArrayPoolTest, AcquireWithinCapacityTest) {
    constexpr RSize NumChunks = 100;
    constexpr int NumThreads = 2;
    constexpr int NumAcquisitions = 5;
    constexpr RSize TotalAcquisitions = NumThreads * NumAcquisitions;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &provider);

    RS_Vector<RIndex> d_acquired(TotalAcquisitions, &provider);
    d_acquired.fill(RS_NULL_INDEX);

    provider.run<RSCF_AcquireChunksArray>({NumThreads}, mem_pool, d_acquired, NumAcquisitions);

    RS_Vector<RIndex> h_acquired = d_acquired.move(&host_provider);
    auto h_chunk_mask = mem_pool.chunk_mask.move(&host_provider);

    int success_count = 0;
    for (RSize i = 0; i < TotalAcquisitions; ++i) {
        RIndex acquired_idx = h_acquired[i];
        if (acquired_idx != RS_NULL_INDEX) {
            success_count++;
            EXPECT_TRUE(h_chunk_mask.isOn(acquired_idx));
        }
    }
    EXPECT_EQ(success_count, TotalAcquisitions);
    EXPECT_EQ(h_chunk_mask.count(), TotalAcquisitions);
    EXPECT_EQ(mem_pool.getAllocatedNum(), TotalAcquisitions);

    h_chunk_mask.free();
    d_acquired.free();
    h_acquired.free();
    mem_pool.free();
}

TEST_F(RS_MemArrayPoolTest, AcquireBeyondCapacityTest) {
    constexpr RSize NumChunks = 100;
    constexpr int NumThreads = 8;
    constexpr int NumAcquisitions = 20;
    constexpr RSize TotalAcquisitions = NumThreads * NumAcquisitions;
    RS_MemArrayPool<ChunkSize> mem_pool(NumChunks, &provider);

    RS_Vector<RIndex> d_acquired(TotalAcquisitions, &provider);
    d_acquired.fill(RS_NULL_INDEX);

    provider.run<RSCF_AcquireChunksArray>({NumThreads}, mem_pool, d_acquired, NumAcquisitions);

    RS_Vector<RIndex> h_acquired = d_acquired.move(&host_provider);
    auto h_chunk_mask = mem_pool.chunk_mask.move(&host_provider);

    int success_count = 0;
    for (RSize i = 0; i < TotalAcquisitions; ++i) {
        RIndex acquired_idx = h_acquired[i];
        if (acquired_idx != RS_NULL_INDEX) {
            success_count++;
            EXPECT_TRUE(h_chunk_mask.isOn(acquired_idx));
        }
    }
    EXPECT_EQ(success_count, NumChunks);
    EXPECT_EQ(h_chunk_mask.count(), NumChunks);
    EXPECT_EQ(mem_pool.getAllocatedNum(), NumChunks);

    h_chunk_mask.free();
    d_acquired.free();
    h_acquired.free();
    mem_pool.free();
}
