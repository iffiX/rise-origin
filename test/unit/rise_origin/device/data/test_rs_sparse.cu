#include <gtest/gtest.h>

#include <iostream>

#include "rise_origin/device/data/rs_sparse.h"

using namespace rise;
using namespace rise::device;

struct LargeValue {
    int data[3];  // 12 bytes
};

class RS_SparseTest : public ::testing::Test {
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

/**
 * @brief Basic hierarchical insert and find:
 * Insert at a coarser level (level 1) and verify that a neighboring finer-level coordinate
 * implicitly returns the same value.
 */
TEST_F(RS_SparseTest, HierarchicalInsertAndFind) {
    RS_Sparse3DMap<
        int,
        RS_Sparse3DHashMapLevel<>,
        RS_Sparse3DArrayLevel<4, 4, 4>,
        RS_Sparse3DArrayLevel<8, 8, 8>>
        sparse(&host_provider);

    // Resize to ensure we have enough space
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    // Insert value=1 at level 1 for coordinate {100, 23, 47}
    // This implies that at the finer level (level 2), blocks under this coordinate are
    // implicitly 1.
    sparse.insert<1>({100, 23, 47}, 1, accessor);

    // Check a neighboring coordinate {99, 22, 46} which falls under the same finer block
    auto result = sparse.find({99, 22, 46}, accessor);
    ASSERT_TRUE(result != nullptr);
    if (result != nullptr) {
        ASSERT_EQ(*result, 1);
    }

    // Cleanup
    sparse.free();
}

/**
 * @brief Overriding a finer-level coordinate:
 * Insert at a coarser level, then insert a different value at a finer level
 * inside that region, and verify the override.
 */
TEST_F(RS_SparseTest, OverrideFinerLevel) {
    RS_Sparse3DMap<
        int,
        RS_Sparse3DHashMapLevel<>,
        RS_Sparse3DArrayLevel<4, 4, 4>,
        RS_Sparse3DArrayLevel<8, 8, 8>>
        sparse(&host_provider);

    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    // Insert coarser-level value=2
    sparse.insert<1>({100, 23, 47}, 2, accessor);

    // Override a specific finer-level coordinate in that block with value=5
    sparse.insert<2>({100, 23, 47}, 5, accessor);

    // The exact coordinate {100, 23, 47} now should return 5
    int* exact_result = sparse.find({100, 23, 47}, accessor);
    ASSERT_TRUE(exact_result != nullptr);
    ASSERT_EQ(*exact_result, 5);

    // A neighboring coordinate in the same coarser-level block but
    // not overridden should still be 2
    int* neighbor_result = sparse.find({101, 23, 47}, accessor);
    ASSERT_TRUE(neighbor_result != nullptr);
    ASSERT_EQ(*neighbor_result, 2);

    sparse.free();
}

/**
 * @brief Partial filling and large block queries:
 * Insert one coarser-level tile and verify that multiple queries within that range
 * return the same inherited value.
 */
TEST_F(RS_SparseTest, LargeBlockQueries) {
    RS_Sparse3DMap<
        int,
        RS_Sparse3DHashMapLevel<>,
        RS_Sparse3DArrayLevel<4, 4, 4>,
        RS_Sparse3DArrayLevel<8, 8, 8>>
        sparse(&host_provider);

    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    // Insert at level 1: large tile value=10
    sparse.insert<1>({200, 100, 50}, 10, accessor);

    // Test various coordinates around {200, 100, 50}
    // All should implicitly return 10
    std::vector<RCoord3> coords_to_test = {
        {202, 102, 52},
        {203, 103, 53},
        {200, 96, 48},  // Minimum boundary
        {207, 103, 55}  // Maximum boundary
    };

    for (auto c : coords_to_test) {
        int* val = sparse.find(c, accessor);
        ASSERT_TRUE(val != nullptr);
        if (val != nullptr) {
            ASSERT_EQ(*val, 10);
        }
    }

    int* invalid_val = sparse.find({208, 104, 56}, accessor);
    ASSERT_TRUE(invalid_val == nullptr);
    sparse.free();
}

/**
 * @brief Interacting with the HashMap root level:
 * Insert at the root hash map level and ensure queries at finer scales
 * inherit the root value.
 */
TEST_F(RS_SparseTest, HashMapRootInsert) {
    RS_Sparse3DMap<
        int,
        RS_Sparse3DHashMapLevel<>,
        RS_Sparse3DArrayLevel<4, 4, 4>,
        RS_Sparse3DArrayLevel<8, 8, 8>>
        sparse(&host_provider);

    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    // Insert at root (level 0)
    sparse.insert<0>({200, 100, 50}, 10, accessor);

    // Check nearby coordinates that should inherit 10
    std::vector<RCoord3> test_coords = {
        {202, 102, 52},
        {203, 103, 53},
        {192, 96, 32},  // Minimum boundary
        {223, 127, 63}  // Maximum boundary
    };

    for (auto c : test_coords) {
        int* val = sparse.find(c, accessor);
        ASSERT_TRUE(val != nullptr);
        if (val != nullptr) {
            ASSERT_EQ(*val, 10);
        }
    }

    // Now override one coordinate at a finer level
    sparse.insert<2>({200, 100, 50}, 50, accessor);
    int* overridden_val = sparse.find({200, 100, 50}, accessor);
    ASSERT_TRUE(overridden_val != nullptr);
    ASSERT_EQ(*overridden_val, 50);  // overridden

    // A nearby coordinate still inherits 10 from root
    int* nearby_val = sparse.find({201, 100, 50}, accessor);
    ASSERT_TRUE(nearby_val != nullptr);
    ASSERT_EQ(*nearby_val, 10);

    sparse.free();
}

/**
 * @brief Multiple regions and consistency:
 * Insert several coarser-level tiles with different values
 * and verify that queries return correct values.
 * Then add a finer-level override in one region and check no impact on others.
 */
TEST_F(RS_SparseTest, MultipleRegions) {
    RS_Sparse3DMap<
        int,
        RS_Sparse3DHashMapLevel<>,
        RS_Sparse3DArrayLevel<4, 4, 4>,
        RS_Sparse3DArrayLevel<8, 8, 8>>
        sparse(&host_provider);

    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    // Insert multiple distinct tiles
    sparse.insert<1>({0, 0, 0}, 11, accessor);
    sparse.insert<1>({256, 256, 256}, 22, accessor);
    sparse.insert<1>({1024, 1024, 1024}, 33, accessor);

    // Check values in those regions
    int* val1 = sparse.find({1, 1, 1}, accessor);  // near {0,0,0}
    ASSERT_TRUE(val1 != nullptr && *val1 == 11);

    int* val2 = sparse.find({257, 256, 256}, accessor);  // near {256,256,256}
    ASSERT_TRUE(val2 != nullptr && *val2 == 22);

    int* val3 = sparse.find({1025, 1024, 1024}, accessor);  // near {1024,1024,1024}
    ASSERT_TRUE(val3 != nullptr && *val3 == 33);

    // Override one coordinate in the {0,0,0} tile with a finer-level value
    sparse.insert<2>({0, 0, 0}, 99, accessor);
    int* overridden = sparse.find({0, 0, 0}, accessor);
    ASSERT_TRUE(overridden != nullptr && *overridden == 99);

    // Other tiles should remain unaffected
    val2 = sparse.find({257, 256, 256}, accessor);
    ASSERT_TRUE(val2 != nullptr && *val2 == 22);

    sparse.clear();
    sparse.free();
}

/**
 * @brief Demonstrate resizing after inserts and verifying hierarchical consistency.
 */
TEST_F(RS_SparseTest, ResizingConsistency) {
    RS_Sparse3DMap<
        int,
        RS_Sparse3DHashMapLevel<>,
        RS_Sparse3DArrayLevel<4, 4, 4>,
        RS_Sparse3DArrayLevel<8, 8, 8>>
        sparse(&host_provider);

    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    // Insert a tile at level 1
    sparse.insert<1>({500, 500, 500}, 7, accessor);

    // Resize root level
    sparse.resize(4096, true);

    // Accessor must be cleared on resize
    accessor.clear();

    // Check the value remains correct after resize
    int* val = sparse.find({500, 500, 500}, accessor);
    ASSERT_TRUE(val != nullptr && *val == 7);

    int* neighbor = sparse.find({501, 500, 500}, accessor);
    ASSERT_TRUE(neighbor != nullptr && *neighbor == 7);

    sparse.clear();
    sparse.free();
}