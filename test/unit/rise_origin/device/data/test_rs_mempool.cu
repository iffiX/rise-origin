#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_mempool.h"

using namespace rise;
using namespace rise::device;

constexpr RSize ChunkSize = 40;

class RS_MemPoolTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize CUDA stream and memory provider
        provider.init({RSE_DEVICE_TYPE_CUDA, 0});
        host_provider.init(0);
    }

    void TearDown() override {
        provider.free();
        host_provider.free();
    }

    RS_MemoryCudaProvider provider;
    RS_MemoryCPUProvider host_provider;
};
/*****************************************************************
 * Test cases for the RS_MemPool constructor
 *
 * Test cases:
 * 1. Initialize with 0 chunks.
 * 2. Initialize with non-zero chunks.
 *****************************************************************/
TEST_F(RS_MemPoolTest, InitializationWithZeroChunks) {
    constexpr RSize NumChunks = 0;
    RS_MemPool<ChunkSize> memPool(NumChunks, &provider);

    EXPECT_EQ(memPool.chunk_num, 0);

    // Ensure chunk_mask is of size chunk_num and all bits are set to off
    EXPECT_TRUE(memPool.chunk_mask.isAllOff());

    // Verify chunk_ptrs size is zero
    EXPECT_EQ(memPool.chunk_ptrs.size(), 0);

    // Confirm allocate_num is zero
    EXPECT_EQ(memPool.getAllocatedNum(), 0);
    memPool.free();
}

TEST_F(RS_MemPoolTest, InitializationWithNonZeroChunks) {
    constexpr RSize NumChunks = 100;
    RS_MemPool<ChunkSize> memPool(NumChunks, &provider);

    // Verify chunk_num equals the number of chunks initialized
    EXPECT_EQ(memPool.chunk_num, NumChunks);

    // Ensure chunk_mask is of size chunk_num and all bits are set to off
    auto h_chunk_mask = memPool.chunk_mask.move(&host_provider);
    for (RSize i = 0; i < NumChunks; ++i) {
        EXPECT_FALSE(h_chunk_mask.isOn(i));
    }
    h_chunk_mask.free();

    // Verify chunk_ptrs contains valid pointers (not nullptr)
    auto h_chunk_ptrs = memPool.chunk_ptrs.move(&host_provider);
    for (RSize i = 0; i < NumChunks; ++i) {
        EXPECT_NE(h_chunk_ptrs.get(i), nullptr);
    }
    h_chunk_ptrs.free();

    // Confirm allocate_num is zero
    EXPECT_EQ(memPool.getAllocatedNum(), 0);
    memPool.free();
}

/*****************************************************************
 * Test cases for the RS_MemPool::grow()
 *
 * Test cases:
 * 1. Grow by non-zero chunks.
 *****************************************************************/
TEST_F(RS_MemPoolTest, GrowNonZeroChunks) {
    constexpr RSize InitialChunks = 50;
    constexpr RSize AdditionalChunks = 50;
    constexpr RSize TotalChunks = InitialChunks + AdditionalChunks;
    RS_MemPool<ChunkSize> memPool(InitialChunks, &provider);

    memPool.grow(AdditionalChunks);
    EXPECT_EQ(memPool.chunk_num, TotalChunks);

    // Ensure chunk_mask and chunk_ptrs are resized to 128 entries
    auto h_chunk_mask = memPool.chunk_mask.move(&host_provider);
    EXPECT_EQ(h_chunk_mask.size(), TotalChunks);
    for (RSize i = 0; i < TotalChunks; ++i) {
        EXPECT_FALSE(h_chunk_mask.isOn(i));
    }
    h_chunk_mask.free();

    auto h_chunk_ptrs = memPool.chunk_ptrs.move(&host_provider);
    EXPECT_EQ(h_chunk_ptrs.size(), TotalChunks);
    for (RSize i = 0; i < TotalChunks; ++i) {
        EXPECT_NE(h_chunk_ptrs.get(i), nullptr);
    }
    h_chunk_ptrs.free();

    memPool.free();
}

/*****************************************************************
 * Test cases for the RS_MemPool::copy()
 *
 * TODO: Add more test cases
 *****************************************************************/

/*****************************************************************
 * Test cases for the RS_MemPool::move()
 *
 * Test cases:
 * 1. Move empty memory pool.
 * 2. Move memory pool with allocated chunks.
 * 3. Move to different memory provider.
 * 4. Verify data consistency after move.
 *****************************************************************/
TEST_F(RS_MemPoolTest, MoveEmptyMemoryPool) {
    RS_MemPool<ChunkSize> memPool;

    // Move to a new provider
    EXPECT_THROW(memPool.move(&provider), RS_UninitializedException);
}

TEST_F(RS_MemPoolTest, MoveMemoryPoolWithAllocatedChunks) {
    constexpr RSize NumChunks = 100;
    RS_MemPool<ChunkSize> memPool(NumChunks, &host_provider);

    // Allocate a chunk and write data
    RIndex chunkIndex = memPool.acquire();
    void* chunkPtr = memPool.getChunkPtr(chunkIndex);
    ASSERT_NE(chunkPtr, nullptr);

    char data[ChunkSize];
    memset(data, 42, ChunkSize);
    memset(chunkPtr, 42, ChunkSize);

    // Move the memory pool to device provider
    auto movedMemPool = memPool.move(&provider);

    // Verify chunk counts and provider
    EXPECT_EQ(movedMemPool.chunk_num, NumChunks);
    EXPECT_EQ(movedMemPool.getAllocatedNum(), 1);
    EXPECT_EQ(movedMemPool.provider(), &provider);

    auto movedMemPool2 = movedMemPool.move(&host_provider);

    // Verify data in moved pool
    void* movedChunkPtr = movedMemPool2.getChunkPtr(chunkIndex);
    ASSERT_NE(movedChunkPtr, nullptr);
    EXPECT_EQ(memcmp(data, movedChunkPtr, ChunkSize), 0);

    memPool.free();
    movedMemPool.free();
    movedMemPool2.free();
}

TEST_F(RS_MemPoolTest, MoveToDifferentMemoryProvider) {
    constexpr RSize NumChunks = 100;
    RS_MemPool<ChunkSize> memPool(NumChunks, &host_provider);

    // Move to device provider
    RS_MemPool<ChunkSize> movedMemPool = memPool.move(&provider);

    EXPECT_EQ(movedMemPool.provider(), &provider);

    memPool.free();
    movedMemPool.free();
}

/*****************************************************************
 * Test cases for the RS_MemPool::acquire()
 *
 * Test on CUDA device.
 *
 * Test cases:
 * 1. Acquire within capacity Test
 * 2. Acquire beyond Capacity Test
 *****************************************************************/
__global__ void acquire_chunks_kernel(
    RS_MemPool<ChunkSize> mem_pool,
    RIndex* acquired_indices,
    void** acquired_pointers,
    int num_acquisitions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    RIndex last_chunk = RS_NULL_INDEX;

    for (int i = 0; i < num_acquisitions; ++i) {
        RIndex index = mem_pool.acquire(last_chunk, tid);
        acquired_indices[tid * num_acquisitions + i] = index;
        acquired_pointers[tid * num_acquisitions + i] = mem_pool.getChunkPtr(index);
        last_chunk = index;
    }
}

TEST_F(RS_MemPoolTest, AcquireWithinCapacityTest) {
    constexpr RSize NumChunks = 100;
    constexpr int NumThreads = 2;
    constexpr int NumAcquisitions = 5;
    RS_MemPool<ChunkSize> memPool(NumChunks, &provider);

    // Allocate device memory for acquired indices and pointers
    RIndex* d_acquired_indices;
    void** d_acquired_pointers;
    cudaMalloc(&d_acquired_indices, NumThreads * NumAcquisitions * sizeof(RIndex));
    cudaMalloc(&d_acquired_pointers, NumThreads * NumAcquisitions * sizeof(void*));

    // Launch kernel
    acquire_chunks_kernel<<<1, NumThreads>>>(
        memPool, d_acquired_indices, d_acquired_pointers, NumAcquisitions
    );

    // Copy results back to host
    auto h_acquired_indices = new RIndex[NumThreads * NumAcquisitions];
    auto h_acquired_pointers = new void*[NumThreads * NumAcquisitions];
    cudaMemcpy(
        h_acquired_indices,
        d_acquired_indices,
        NumThreads * NumAcquisitions * sizeof(RIndex),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        h_acquired_pointers,
        d_acquired_pointers,
        NumThreads * NumAcquisitions * sizeof(void*),
        cudaMemcpyDeviceToHost
    );

    // Verify results
    auto h_chunk_mask = memPool.chunk_mask.move(&host_provider);
    auto h_chunk_ptrs = memPool.chunk_ptrs.move(&host_provider);
    auto root = memPool.host_memory_regions->at(0);
    EXPECT_EQ(h_chunk_mask.count(), NumThreads * NumAcquisitions);
    for (int i = 0; i < NumThreads * NumAcquisitions; ++i) {
        EXPECT_NE(h_acquired_indices[i], RS_NULL_INDEX);
        EXPECT_TRUE(h_chunk_mask.isOn(h_acquired_indices[i]));
        EXPECT_EQ(h_acquired_pointers[i], h_chunk_ptrs.get(h_acquired_indices[i]));
        EXPECT_EQ(h_acquired_pointers[i], (root + h_acquired_indices[i] * ChunkSize).data);
    }

    h_chunk_mask.free();
    h_chunk_ptrs.free();
    EXPECT_EQ(memPool.getAllocatedNum(), NumThreads * NumAcquisitions);
    delete[] h_acquired_indices;
    delete[] h_acquired_pointers;
    // Cleanup
    cudaFree(d_acquired_indices);
    cudaFree(d_acquired_pointers);
    memPool.free();
}

TEST_F(RS_MemPoolTest, AcquireBeyondCapacityTest) {
    constexpr RSize NumChunks = 100;
    constexpr int NumBlocks = 4;
    constexpr int NumThreads = 2;
    constexpr int NumAcquisitions = 20;  // Total acquisitions exceed NumChunks
    RS_MemPool<ChunkSize> memPool(NumChunks, &provider);

    // Allocate device memory for acquired indices and pointers
    RIndex* d_acquired_indices;
    void** d_acquired_pointers;
    cudaMalloc(&d_acquired_indices, NumBlocks * NumThreads * NumAcquisitions * sizeof(RIndex));
    cudaMalloc(&d_acquired_pointers, NumBlocks * NumThreads * NumAcquisitions * sizeof(void*));

    // Launch kernel
    acquire_chunks_kernel<<<NumBlocks, NumThreads>>>(
        memPool, d_acquired_indices, d_acquired_pointers, NumAcquisitions
    );

    // Copy results back to host
    auto h_acquired_indices = new RIndex[NumBlocks * NumThreads * NumAcquisitions];
    auto h_acquired_pointers = new void*[NumBlocks * NumThreads * NumAcquisitions];
    cudaMemcpy(
        h_acquired_indices,
        d_acquired_indices,
        NumBlocks * NumThreads * NumAcquisitions * sizeof(RIndex),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        h_acquired_pointers,
        d_acquired_pointers,
        NumBlocks * NumThreads * NumAcquisitions * sizeof(void*),
        cudaMemcpyDeviceToHost
    );

    // Verify results
    auto h_chunk_mask = memPool.chunk_mask.move(&host_provider);
    auto h_chunk_ptrs = memPool.chunk_ptrs.move(&host_provider);
    auto root = memPool.host_memory_regions->at(0);
    int success_count = 0;
    for (int i = 0; i < NumBlocks * NumThreads * NumAcquisitions; ++i) {
        if (h_acquired_indices[i] != RS_NULL_INDEX) {
            success_count++;
            EXPECT_TRUE(h_chunk_mask.isOn(h_acquired_indices[i]));
            EXPECT_EQ(h_acquired_pointers[i], h_chunk_ptrs.get(h_acquired_indices[i]));
            EXPECT_EQ(h_acquired_pointers[i], (root + h_acquired_indices[i] * ChunkSize).data);
        }
    }
    EXPECT_EQ(success_count, NumChunks);  // Only NumChunks acquisitions should succeed
    EXPECT_EQ(h_chunk_mask.count(), NumChunks);
    EXPECT_EQ(memPool.getAllocatedNum(), NumChunks);

    // Cleanup
    h_chunk_mask.free();
    h_chunk_ptrs.free();
    delete[] h_acquired_indices;
    delete[] h_acquired_pointers;
    cudaFree(d_acquired_indices);
    cudaFree(d_acquired_pointers);
    memPool.free();
}

/*****************************************************************
 * Test cases for the RS_MemPool::acquire()
 *
 * Test on CPU.
 *
 * Test cases:
 * 1. Acquire within capacity Test
 * 2. Acquire beyond Capacity Test
 *****************************************************************/

void acquire_chunks_host(
    RS_MemPool<ChunkSize>& mem_pool,
    RIndex* acquired_indices,
    void** acquired_pointers,
    int num_acquisitions,
    int tid
) {
    RIndex last_chunk = RS_NULL_INDEX;

    for (int i = 0; i < num_acquisitions; ++i) {
        RIndex index = mem_pool.acquire(last_chunk, tid);  // Use thread ID as partition seed
        acquired_indices[tid * num_acquisitions + i] = index;
        acquired_pointers[tid * num_acquisitions + i] = mem_pool.getChunkPtr(index);
        last_chunk = index;
    }
}

TEST_F(RS_MemPoolTest, AcquireWithinCapacityTestHost) {
    constexpr RSize NumChunks = 100;
    constexpr int NumThreads = 2;
    constexpr int NumAcquisitions = 5;
    RS_MemPool<ChunkSize> memPool(NumChunks, &host_provider);

    // Allocate memory for acquired indices and pointers
    std::vector<RIndex> acquired_indices(NumThreads * NumAcquisitions);
    std::vector<void*> acquired_pointers(NumThreads * NumAcquisitions);

    // Launch threads
    std::vector<std::thread> threads;
    for (int t = 0; t < NumThreads; ++t) {
        threads.emplace_back(
            acquire_chunks_host,
            std::ref(memPool),
            acquired_indices.data(),
            acquired_pointers.data(),
            NumAcquisitions,
            t
        );
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify results
    auto root = memPool.host_memory_regions->at(0);
    EXPECT_EQ(memPool.chunk_mask.count(), NumThreads * NumAcquisitions);
    for (int i = 0; i < NumThreads * NumAcquisitions; ++i) {
        EXPECT_NE(acquired_indices[i], RS_NULL_INDEX);
        EXPECT_TRUE(memPool.chunk_mask.isOn(acquired_indices[i]));
        EXPECT_EQ(acquired_pointers[i], memPool.chunk_ptrs.get(acquired_indices[i]));
        EXPECT_EQ(acquired_pointers[i], (root + acquired_indices[i] * ChunkSize).data);
    }
    EXPECT_EQ(memPool.getAllocatedNum(), NumThreads * NumAcquisitions);
    memPool.free();
}

TEST_F(RS_MemPoolTest, AcquireBeyondCapacityTestHost) {
    constexpr RSize NumChunks = 100;
    constexpr int NumThreads = 8;
    constexpr int NumAcquisitions = 20;  // Total acquisitions exceed NumChunks
    RS_MemPool<ChunkSize> memPool(NumChunks, &host_provider);

    // Allocate memory for acquired indices and pointers
    std::vector<RIndex> acquired_indices(NumThreads * NumAcquisitions);
    std::vector<void*> acquired_pointers(NumThreads * NumAcquisitions);

    // Launch threads
    std::vector<std::thread> threads;
    for (int t = 0; t < NumThreads; ++t) {
        threads.emplace_back(
            acquire_chunks_host,
            std::ref(memPool),
            acquired_indices.data(),
            acquired_pointers.data(),
            NumAcquisitions,
            t
        );
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify results
    auto root = memPool.host_memory_regions->at(0);
    int success_count = 0;
    for (int i = 0; i < NumThreads * NumAcquisitions; ++i) {
        if (acquired_indices[i] != RS_NULL_INDEX) {
            success_count++;
            EXPECT_TRUE(memPool.chunk_mask.isOn(acquired_indices[i]));
            EXPECT_EQ(acquired_pointers[i], memPool.chunk_ptrs.get(acquired_indices[i]));
            EXPECT_EQ(acquired_pointers[i], (root + acquired_indices[i] * ChunkSize).data);
        }
    }
    EXPECT_EQ(success_count, NumChunks);  // Only NumChunks acquisitions should succeed
    EXPECT_EQ(memPool.chunk_mask.count(), NumChunks);
    EXPECT_EQ(memPool.getAllocatedNum(), NumChunks);

    memPool.free();
}