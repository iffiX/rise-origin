#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_list.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/data/rs_mempool.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_ListHeader is a chunk based list header that manages a chain
 * of fixed size chunks through a memory pool.
 *
 * - Construction & Initialization
 *   Default constructor
 *
 * - Element Access (Host)
 *   operator[], get()
 *
 * - Capacity Management
 *   size(), empty()
 *
 * - Modifiers (Host)
 *   append(), pop(), clear()
 *
 * - Internal Methods
 *   findChunk()
 *
 * RS_List is a dynamic list container built on top of RS_ListHeader.
 *
 * - Construction & Initialization
 *   Default constructor, constructor with provider, init()
 *
 * - Capacity Management
 *   size(), empty(), clear()
 *
 * - Element Access (Device)
 *   operator[], get(), set()
 *
 * - Modifiers (Device)
 *   append(), pop()
 *
 * - Operations
 *   copy(), move()
 *****************************************************************/

constexpr RSize ChunkSize = 32;

class RS_ListHeaderTest : public ::testing::Test {
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
 * Test cases for the RS_ListHeader constructor
 *
 * Test cases:
 * 1. Default construct and verify the header is empty.
 *****************************************************************/

TEST_F(RS_ListHeaderTest, Initialization) {
    using ListHeaderType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListHeaderType list;

    EXPECT_EQ(list.elem_num, 0);
    EXPECT_EQ(list.head, nullptr);
    EXPECT_EQ(list.tail, nullptr);
}

/*****************************************************************
 * Test cases for the append() method
 *
 * Test cases:
 * 1. Append elements and verify elem_num increments.
 * 2. Append exactly ChunkSize elements.
 * 3. Append elements exceeding ChunkSize to create a new chunk.
 *****************************************************************/

TEST_F(RS_ListHeaderTest, AppendElements) {
    constexpr RSize NumChunks = 4;
    using ListHeaderType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListHeaderType::MemPool mem_pool(NumChunks, &host_provider);
    ListHeaderType list;

    // Append elements less than ChunkSize
    for (int i = 0; i < ChunkSize - 1; ++i) {
        list.append(&mem_pool, i);
        EXPECT_EQ(list.elem_num, i + 1);
    }
    EXPECT_EQ(list.head, list.tail);
    EXPECT_EQ(list.head->next, nullptr);

    // Append elements to reach exactly ChunkSize
    list.append(&mem_pool, ChunkSize - 1);
    EXPECT_EQ(list.elem_num, ChunkSize);
    EXPECT_EQ(list.head, list.tail);
    EXPECT_EQ(list.head->next, nullptr);

    // Append elements exceeding ChunkSize to create a new chunk
    list.append(&mem_pool, ChunkSize);
    EXPECT_EQ(list.elem_num, ChunkSize + 1);
    EXPECT_NE(list.head, list.tail);
    EXPECT_NE(list.head->next, nullptr);
    EXPECT_EQ(list.tail->next, nullptr);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the operator[] method
 *
 * Test cases:
 * 1. Access elements using operator[] and verify values.
 * 2. Access boundary indices.
 *****************************************************************/

TEST_F(RS_ListHeaderTest, ElementAccess) {
    constexpr RSize NumChunks = 4;
    using ListHeaderType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListHeaderType::MemPool mem_pool(NumChunks, &host_provider);
    ListHeaderType list;

    for (int i = 0; i < ChunkSize * 2; ++i) {
        list.append(&mem_pool, i * 10);
    }

    for (int i = 0; i < ChunkSize * 2; ++i) {
        EXPECT_EQ(list[i], i * 10);
    }

    EXPECT_EQ(list[0], 0);
    EXPECT_EQ(list[list.elem_num - 1], (list.elem_num - 1) * 10);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the pop() method
 *
 * Test cases:
 * 1. Pop elements and verify elem_num decrements.
 * 2. Pop elements until the list is empty.
 *****************************************************************/

TEST_F(RS_ListHeaderTest, PopElements) {
    constexpr RSize NumChunks = 4;
    using ListHeaderType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListHeaderType::MemPool mem_pool(NumChunks, &host_provider);
    ListHeaderType list;

    for (int i = 0; i < ChunkSize * 2; ++i) {
        list.append(&mem_pool, i);
    }

    for (int i = ChunkSize * 2 - 1; i >= 0; --i) {
        EXPECT_EQ(list.elem_num, i + 1);
        list.pop(&mem_pool);
    }

    EXPECT_EQ(list.elem_num, 0);
    EXPECT_EQ(list.head, nullptr);
    EXPECT_EQ(list.tail, nullptr);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the clear() method
 *
 * Test cases:
 * 1. Clear the list after appending elements.
 * 2. Call clear() again on the empty list.
 *****************************************************************/

TEST_F(RS_ListHeaderTest, ClearList) {
    constexpr RSize NumChunks = 4;
    using ListHeaderType = RS_ListHeader<int, ChunkSize, RS_MemPool>;

    ListHeaderType::MemPool mem_pool(NumChunks, &host_provider);
    ListHeaderType list;

    for (int i = 0; i < ChunkSize * 2; ++i) {
        list.append(&mem_pool, i);
    }

    list.clear(&mem_pool);

    EXPECT_EQ(list.elem_num, 0);
    EXPECT_EQ(list.head, nullptr);
    EXPECT_EQ(list.tail, nullptr);

    list.clear(&mem_pool);
    EXPECT_EQ(list.elem_num, 0);
    EXPECT_EQ(list.head, nullptr);
    EXPECT_EQ(list.tail, nullptr);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the findChunk() method
 *
 * Test cases:
 * 1. Find chunks from head when the index is in the first half.
 * 2. Find chunks from tail when the index is in the second half.
 * 3. Access elements at chunk boundaries.
 *****************************************************************/

TEST_F(RS_ListHeaderTest, FindChunk) {
    constexpr RSize NumChunks = 4;
    using ListHeaderType = RS_ListHeader<int, ChunkSize, RS_MemPool>;
    using ChunkType = typename ListHeaderType::ChunkType;

    ListHeaderType::MemPool mem_pool(NumChunks, &host_provider);
    ListHeaderType list;

    for (int i = 0; i < ChunkSize * 3; ++i) {
        list.append(&mem_pool, i);
    }

    // Boundary between the first and second chunk
    RIndex index = ChunkSize;
    ChunkType* chunk = list.findChunk(index);
    EXPECT_EQ(chunk, list.head->next);

    // In the third chunk
    index = ChunkSize * 2 + 5;
    chunk = list.findChunk(index);
    EXPECT_EQ(chunk, list.tail);

    index = 0;
    chunk = list.findChunk(index);
    EXPECT_EQ(chunk, list.head);

    index = list.elem_num - 1;
    chunk = list.findChunk(index);
    EXPECT_EQ(chunk, list.tail);

    mem_pool.free();
}

/*****************************************************************
 * Test cases for the RS_List construction and initialization
 *
 * Test cases:
 * 1. Default construction.
 * 2. Construction with provider.
 * 3. Construction with chunks and provider.
 * 4. Init with provider.
 * 5. Init with chunks and provider.
 *****************************************************************/

class RS_ListTest : public ::testing::Test {
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

TEST_F(RS_ListTest, DefaultConstruction) {
    RS_List<int, ChunkSize> list;
    EXPECT_EQ(list.provider(), nullptr);
}

TEST_F(RS_ListTest, ConstructionWithProvider) {
    RS_List<int, ChunkSize> list(&provider);
    EXPECT_NE(list.provider(), nullptr);
    EXPECT_EQ(list.size(), 0);
    EXPECT_TRUE(list.empty());
    list.free();
}

TEST_F(RS_ListTest, ConstructionWithChunksAndProvider) {
    RS_List<int, ChunkSize> list(4, &provider);
    EXPECT_NE(list.provider(), nullptr);
    EXPECT_EQ(list.size(), 0);
    EXPECT_TRUE(list.empty());
    list.free();
}

TEST_F(RS_ListTest, InitWithProvider) {
    RS_List<int, ChunkSize> list;
    list.init(&provider);
    EXPECT_NE(list.provider(), nullptr);
    EXPECT_EQ(list.size(), 0);
    EXPECT_TRUE(list.empty());
    list.free();
}

TEST_F(RS_ListTest, InitWithChunksAndProvider) {
    RS_List<int, ChunkSize> list;
    list.init(4, &provider);
    EXPECT_NE(list.provider(), nullptr);
    EXPECT_EQ(list.size(), 0);
    EXPECT_TRUE(list.empty());
    list.free();
}

/*****************************************************************
 * Test cases for the RS_List capacity management
 *
 * Test cases:
 * 1. Size and empty on a fresh list.
 * 2. Clear an empty list.
 *****************************************************************/

TEST_F(RS_ListTest, SizeAndEmpty) {
    RS_List<int, ChunkSize> list(4, &provider);
    EXPECT_EQ(list.size(), 0);
    EXPECT_TRUE(list.empty());
    list.free();
}

TEST_F(RS_ListTest, ClearEmptyList) {
    RS_List<int, ChunkSize> list(4, &provider);
    list.clear();
    EXPECT_EQ(list.size(), 0);
    EXPECT_TRUE(list.empty());
    list.free();
}

/*****************************************************************
 * Test cases for the RS_List device side operations
 *
 * A bare compute function appends elements on the device using the
 * locked append, then reads back values into a result buffer that is
 * moved to the host for checking.
 *
 * Test cases:
 * 1. Device append single element.
 * 2. Device append multiple elements.
 * 3. Device append exceeding one chunk.
 * 4. Device pop.
 * 5. Device element access via operator[], get(), and set().
 *****************************************************************/

class RSCF_ListAppend : public RS_ComputeFunction<RSCF_ListAppend> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_List<int, ChunkSize> list, int count) {
        if (index1D() == 0) {
            for (int i = 0; i < count; i++) {
                list.append(i * 10);
            }
        }
    }
};

class RSCF_ListPop : public RS_ComputeFunction<RSCF_ListPop> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_List<int, ChunkSize> list, int push_count, int pop_count) {
        if (index1D() == 0) {
            for (int i = 0; i < push_count; i++) {
                list.append(i);
            }
            for (int i = 0; i < pop_count; i++) {
                list.pop();
            }
        }
    }
};

class RSCF_ListAccess : public RS_ComputeFunction<RSCF_ListAccess> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_List<int, ChunkSize> list, RS_Vector<int> results) {
        if (index1D() == 0) {
            for (int i = 0; i < 5; i++) {
                list.append(i * 10);
            }
            results[0] = list[0];
            results[1] = list[2];
            results[2] = list[4];
            results[3] = list.get(1);
            results[4] = list.get(3);
            list.set(2, 999);
            results[5] = list[2];
        }
    }
};

TEST_F(RS_ListTest, DeviceAppendSingle) {
    RS_List<int, ChunkSize> list(4, &provider);

    provider.run<RSCF_ListAppend>({1}, list, 1);

    EXPECT_EQ(list.size(), 1);
    EXPECT_FALSE(list.empty());
    list.free();
}

TEST_F(RS_ListTest, DeviceAppendMultiple) {
    RS_List<int, ChunkSize> list(4, &provider);

    provider.run<RSCF_ListAppend>({1}, list, 10);

    EXPECT_EQ(list.size(), 10);
    EXPECT_FALSE(list.empty());
    list.free();
}

TEST_F(RS_ListTest, DeviceAppendExceedingChunk) {
    RS_List<int, ChunkSize> list(4, &provider);

    provider.run<RSCF_ListAppend>({1}, list, ChunkSize + 5);

    EXPECT_EQ(list.size(), ChunkSize + 5);
    EXPECT_FALSE(list.empty());
    list.free();
}

TEST_F(RS_ListTest, DevicePop) {
    RS_List<int, ChunkSize> list(4, &provider);

    provider.run<RSCF_ListPop>({1}, list, 10, 3);

    EXPECT_EQ(list.size(), 7);
    list.free();
}

TEST_F(RS_ListTest, DeviceElementAccess) {
    RS_List<int, ChunkSize> list(4, &provider);

    RS_Vector<int> results(6, &provider);
    results.fill(0);

    provider.run<RSCF_ListAccess>({1}, list, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    EXPECT_EQ(host_results[0], 0);     // list[0]
    EXPECT_EQ(host_results[1], 20);    // list[2]
    EXPECT_EQ(host_results[2], 40);    // list[4]
    EXPECT_EQ(host_results[3], 10);    // list.get(1)
    EXPECT_EQ(host_results[4], 30);    // list.get(3)
    EXPECT_EQ(host_results[5], 999);   // list[2] after set

    results.free();
    host_results.free();
    list.free();
}

/*****************************************************************
 * Test cases for the RS_List copy and move operations
 *
 * Test cases:
 * 1. Copy an empty list.
 * 2. Move an empty list to a different provider.
 * 3. Move a populated list and verify the size is preserved.
 *****************************************************************/

TEST_F(RS_ListTest, CopyEmptyList) {
    RS_List<int, ChunkSize> src(4, &provider);
    RS_List<int, ChunkSize> dst(4, &provider);

    dst.copy(src);
    EXPECT_EQ(dst.size(), 0);
    EXPECT_TRUE(dst.empty());

    src.free();
    dst.free();
}

TEST_F(RS_ListTest, MoveEmptyList) {
    RS_List<int, ChunkSize> src(4, &provider);

    RS_List<int, ChunkSize> moved = src.move(&host_provider);
    EXPECT_EQ(moved.size(), 0);
    EXPECT_TRUE(moved.empty());

    src.free();
    moved.free();
}

TEST_F(RS_ListTest, MovePopulatedList) {
    RS_List<int, ChunkSize> src(4, &provider);

    provider.run<RSCF_ListAppend>({1}, src, ChunkSize + 10);
    EXPECT_EQ(src.size(), ChunkSize + 10);

    RS_List<int, ChunkSize> moved = src.move(&host_provider);
    EXPECT_EQ(moved.provider(), &host_provider);
    EXPECT_EQ(moved.size(), ChunkSize + 10);

    src.free();
    moved.free();
}
