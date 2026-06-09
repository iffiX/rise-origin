#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_list.h"
#include "rise_origin/device/data/rs_mempool.h"

using namespace rise;
using namespace rise::device;

constexpr RSize ChunkSize = 32;

class RS_ListHeaderTest : public ::testing::Test {
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

// TODO: Add more test cases, for device test

/*****************************************************************
 * Test cases for the RS_ListHeader constructor
 *
 * Test cases:
 * 1. Initialize RS_ListHeader with a valid memory pool.
 *    And verify the size is empty.
 *****************************************************************/
TEST_F(RS_ListHeaderTest, Initialization) {
    using ListType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListType list;

    EXPECT_EQ(list.elem_num, 0);
    EXPECT_EQ(list.head, nullptr);
    EXPECT_EQ(list.tail, nullptr);
}

/*****************************************************************
 * Test cases for the append() method
 *
 * Test cases:
 * 1. Append elements and verify elem_num increments.
 * 2. Append elements exceeding ChunkSize to test chunk creation.
 * 3. Append exactly ChunkSize elements.
 *****************************************************************/
TEST_F(RS_ListHeaderTest, AppendElements) {
    constexpr RSize NumChunks = 4;
    using ListType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListType::MemPool memPool(NumChunks, &host_provider);
    ListType list;

    // Append elements less than ChunkSize
    for (int i = 0; i < ChunkSize - 1; ++i) {
        list.append(&memPool, i);
        EXPECT_EQ(list.elem_num, i + 1);
    }
    EXPECT_EQ(list.head, list.tail);
    EXPECT_EQ(list.head->next, nullptr);

    // Append elements to reach exactly ChunkSize
    list.append(&memPool, ChunkSize - 1);
    EXPECT_EQ(list.elem_num, ChunkSize);
    EXPECT_EQ(list.head, list.tail);
    EXPECT_EQ(list.head->next, nullptr);

    // Append elements exceeding ChunkSize to create new chunk
    list.append(&memPool, ChunkSize);
    EXPECT_EQ(list.elem_num, ChunkSize + 1);
    EXPECT_NE(list.head, list.tail);
    EXPECT_NE(list.head->next, nullptr);
    EXPECT_EQ(list.tail->next, nullptr);
}

/*****************************************************************
 * Test cases for the operator[] method
 *
 * Test cases:
 * 1. Access elements using operator[] and verify values.
 * 2. Access boundary indices (0 and elem_num - 1).
 *****************************************************************/
TEST_F(RS_ListHeaderTest, ElementAccess) {
    constexpr RSize NumChunks = 4;
    using ListType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListType::MemPool memPool(NumChunks, &host_provider);
    ListType list;

    // Append elements
    for (int i = 0; i < ChunkSize * 2; ++i) {
        list.append(&memPool, i * 10);
    }

    // Access elements and verify values
    for (int i = 0; i < ChunkSize * 2; ++i) {
        EXPECT_EQ(list[i], i * 10);
    }

    // Access boundary indices
    EXPECT_EQ(list[0], 0);
    EXPECT_EQ(list[list.elem_num - 1], (list.elem_num - 1) * 10);
}

/*****************************************************************
 * Test cases for the pop() method
 *
 * Test cases:
 * 1. Pop elements and verify elem_num decrements.
 * 2. Ensure chunks are deallocated when empty.
 * 3. Pop elements until list is empty.
 *****************************************************************/
TEST_F(RS_ListHeaderTest, PopElements) {
    constexpr RSize NumChunks = 4;
    using ListType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListType::MemPool memPool(NumChunks, &host_provider);
    ListType list;

    // Append elements
    for (int i = 0; i < ChunkSize * 2; ++i) {
        list.append(&memPool, i);
    }

    // Pop elements and verify
    for (int i = ChunkSize * 2 - 1; i >= 0; --i) {
        EXPECT_EQ(list.elem_num, i + 1);
        list.pop(&memPool);
    }

    // Verify list is empty
    EXPECT_EQ(list.elem_num, 0);
    EXPECT_EQ(list.head, nullptr);
    EXPECT_EQ(list.tail, nullptr);
}

/*****************************************************************
 * Test cases for the clear() method
 *
 * Test cases:
 * 1. Clear the list after appending elements.
 * 2. Verify that all chunks are deallocated.
 * 3. Call clear() on an empty list.
 *****************************************************************/
TEST_F(RS_ListHeaderTest, ClearList) {
    constexpr RSize NumChunks = 4;
    using ListType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListType::MemPool memPool(NumChunks, &host_provider);
    ListType list;

    // Append elements
    for (int i = 0; i < ChunkSize * 2; ++i) {
        list.append(&memPool, i);
    }

    // Clear the list
    list.clear(&memPool);

    // Verify list is empty
    EXPECT_EQ(list.elem_num, 0);
    EXPECT_EQ(list.head, nullptr);
    EXPECT_EQ(list.tail, nullptr);

    // Clear again on empty list
    list.clear(&memPool);
    EXPECT_EQ(list.elem_num, 0);
    EXPECT_EQ(list.head, nullptr);
    EXPECT_EQ(list.tail, nullptr);
}

/*****************************************************************
 * Test cases for the findChunk() method
 *
 * Test cases:
 * 1. Find chunks from head when index is in the first half.
 * 2. Find chunks from tail when index is in the second half.
 * 3. Access elements at chunk boundaries.
 *****************************************************************/
TEST_F(RS_ListHeaderTest, FindChunk) {
    constexpr RSize NumChunks = 4;
    using ListType = RS_ListHeader<int, ChunkSize, RS_MemPool>;
    using ChunkType = typename ListType::ChunkType;

    ListType::MemPool memPool(NumChunks, &host_provider);
    ListType list;

    // Append elements
    for (int i = 0; i < ChunkSize * 3; ++i) {
        list.append(&memPool, i);
    }

    // Test findChunk from head
    RIndex index = ChunkSize;  // Boundary between first and second chunk
    ChunkType* chunk = list.findChunk(index);
    EXPECT_EQ(chunk, list.head->next);

    // Test findChunk from tail
    index = ChunkSize * 2 + 5;  // In the third chunk
    chunk = list.findChunk(index);
    EXPECT_EQ(chunk, list.tail);

    // Test chunk boundaries
    index = 0;
    chunk = list.findChunk(index);
    EXPECT_EQ(chunk, list.head);

    index = list.elem_num - 1;
    chunk = list.findChunk(index);
    EXPECT_EQ(chunk, list.tail);
}

/*****************************************************************
 * Test cases for edge cases and robustness
 *
 * Test cases:
 * 1. Attempt to access out-of-bounds index.
 * 2. Call pop() on an empty list.
 *****************************************************************/
TEST_F(RS_ListHeaderTest, EdgeCases) {
    constexpr RSize NumChunks = 4;
    using ListType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListType::MemPool memPool(NumChunks, &host_provider);
    ListType list;

    // Attempt to access out-of-bounds index
    // Note: Asserts terminate the program; for testing, we use death tests
    // Compile with -DGTEST_HAS_DEATH_TEST=1

    ASSERT_DEATH({ list[0]; }, ".*Assertion.*");

    // Call pop() on an empty list
    ASSERT_DEATH({ list.pop(&memPool); }, ".*Assertion.*");
}
