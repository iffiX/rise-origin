#include <gtest/gtest.h>

#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_sparse.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_Sparse3DMap is a hierarchical sparse 3D spatial map based on the
 * OpenVDB paper. It uses a multi level tree with a hash map or array
 * root level and array descendant levels.
 *
 * - Construction & Initialization
 *   Constructor with provider, default constructor, free()
 *
 * - Memory Management
 *   resize() for the root level, resizeMemPool() for descendant levels
 *   copy(), move() between providers
 *
 * - Hierarchical Operations (Host)
 *   insert() and find() with a cached accessor
 *   Coarse level values inherited by finer levels
 *   Fine level values override coarse level values
 *
 * - Device Operations
 *   insert() and find() driven from a bare compute function
 *
 * - Boundary Conditions
 *   Out of bound access, zero coordinate, bounds after resize
 *****************************************************************/

class RS_SparseTest : public ::testing::Test {
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

// Hash map root configuration
using TestSparseHashMap = RS_Sparse3DMap<
    int,
    RS_Sparse3DHashMapLevel<>,
    RS_Sparse3DArrayLevel<4, 4, 4>,
    RS_Sparse3DArrayLevel<8, 8, 8>>;

// 3D array root configuration
using TestSparseArray = RS_Sparse3DMap<
    int,
    RS_Sparse3DArrayLevel<4, 4, 4>,
    RS_Sparse3DArrayLevel<4, 4, 4>,
    RS_Sparse3DArrayLevel<8, 8, 8>>;

/*****************************************************************
 * Compute functions for sparse map device operations
 *
 * A bare compute function inserts or finds values at coordinates that
 * are provided through an RS_Vector buffer, and writes the outcome into
 * a result buffer that is moved to the host for checking.
 *****************************************************************/

template <typename SparseType, RIndex Level>
class RSCF_SparseInsert : public RS_ComputeFunction<RSCF_SparseInsert<SparseType, Level>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_SparseInsert<SparseType, Level>>;
    using BaseType::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void
    run(SparseType sparse, RS_Vector<RCoord3> coords, RS_Vector<int> values,
        RS_Vector<int> results) {
        RIndex idx = index1D();
        results[idx] = sparse.template insert<Level>(coords[idx], values[idx]) ? 1 : 0;
    }
};

template <typename SparseType>
class RSCF_SparseFind : public RS_ComputeFunction<RSCF_SparseFind<SparseType>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_SparseFind<SparseType>>;
    using BaseType::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void
    run(SparseType sparse, RS_Vector<RCoord3> coords, RS_Vector<int> values,
        RS_Vector<int> found) {
        RIndex idx = index1D();
        int* ptr = sparse.find(coords[idx]);
        if (ptr != nullptr) {
            values[idx] = *ptr;
            found[idx] = 1;
        } else {
            values[idx] = -1;
            found[idx] = 0;
        }
    }
};

/*****************************************************************
 * Test cases for the RS_Sparse3DMap constructor
 *
 * Test cases:
 * 1. Construct with a valid provider, hash map root.
 * 2. Construct with a valid provider, array root.
 * 3. Default constructor.
 *****************************************************************/

TEST_F(RS_SparseTest, ConstructWithValidProviderHashMap) {
    TestSparseHashMap sparse(&host_provider);
    EXPECT_TRUE(sparse.provider() != nullptr);
    sparse.free();
}

TEST_F(RS_SparseTest, ConstructWithValidProviderArray) {
    TestSparseArray sparse(&host_provider);
    EXPECT_TRUE(sparse.provider() != nullptr);
    sparse.free();
}

TEST_F(RS_SparseTest, DefaultConstructor) {
    TestSparseHashMap sparse;
    EXPECT_TRUE(sparse.provider() == nullptr);
}

/*****************************************************************
 * Test cases for the RS_Sparse3DMap::free()
 *
 * Test cases:
 * 1. Free after initialization.
 * 2. Multiple frees do not crash.
 *****************************************************************/

TEST_F(RS_SparseTest, FreeAfterInitialization) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    sparse.free();
    EXPECT_TRUE(sparse.provider() == nullptr);
}

TEST_F(RS_SparseTest, MultipleFrees) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);

    sparse.free();
    sparse.free();
    sparse.free();

    EXPECT_TRUE(sparse.provider() == nullptr);
}

/*****************************************************************
 * Test cases for the RS_Sparse3DMap::resize()
 *
 * Test cases:
 * 1. Resize the hash map root with copy and data preservation.
 * 2. Resize the array root dimensions and verify bounds.
 * 3. Resize the array root with non power of two should throw.
 *****************************************************************/

TEST_F(RS_SparseTest, ResizeHashMapWithCopy) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(1024, false);
    sparse.resizeMemPool(1, 512);
    sparse.resizeMemPool(2, 512);

    auto accessor = sparse.getAccessor();
    sparse.insert<1>({100, 100, 100}, 42, accessor);

    sparse.resize(2048, true);

    accessor.clear();
    int* result = sparse.find({100, 100, 100}, accessor);
    ASSERT_TRUE(result != nullptr);
    EXPECT_EQ(*result, 42);

    sparse.free();
}

TEST_F(RS_SparseTest, ResizeArrayDimensions) {
    TestSparseArray sparse(&host_provider);
    sparse.resizeMemPool(1, 512);
    sparse.resizeMemPool(2, 512);

    sparse.resize({16, 16, 16}, false);

    RCoord3 max_coord = sparse.maxCoord();
    EXPECT_EQ(max_coord.x, 255);
    EXPECT_EQ(max_coord.y, 255);
    EXPECT_EQ(max_coord.z, 255);

    RCoord3 min_coord = sparse.minCoord();
    EXPECT_EQ(min_coord.x, -256);
    EXPECT_EQ(min_coord.y, -256);
    EXPECT_EQ(min_coord.z, -256);

    sparse.free();
}

TEST_F(RS_SparseTest, ResizeArrayWithNonPowerOfTwo) {
    TestSparseArray sparse(&host_provider);

    EXPECT_THROW(sparse.resize({10, 10, 10}, false), RS_InvalidArgumentException);

    sparse.free();
}

/*****************************************************************
 * Test cases for the RS_Sparse3DMap::resizeMemPool()
 *
 * Test cases:
 * 1. Resize descendant level memory pools with data preservation.
 * 2. Invalid level index should throw.
 *****************************************************************/

TEST_F(RS_SparseTest, ResizeMemPoolWithDataPreservation) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 512);
    sparse.resizeMemPool(2, 512);

    auto accessor = sparse.getAccessor();
    sparse.insert<1>({100, 100, 100}, 42, accessor);

    sparse.resizeMemPool(1, 1024, true);
    sparse.resizeMemPool(2, 1024, true);

    accessor.clear();
    int* result = sparse.find({100, 100, 100}, accessor);
    ASSERT_TRUE(result != nullptr);
    EXPECT_EQ(*result, 42);

    sparse.free();
}

TEST_F(RS_SparseTest, ResizeMemPoolInvalidLevel) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);

    // Level 0 is the root and cannot be resized as a memory pool
    EXPECT_THROW(sparse.resizeMemPool(0, 512), RS_InvalidArgumentException);

    // Level equal to the level count is out of range
    EXPECT_THROW(sparse.resizeMemPool(3, 512), RS_InvalidArgumentException);

    sparse.free();
}

/*****************************************************************
 * Test cases for hierarchical insert and find on the host
 *
 * These tests verify the hierarchical inheritance behavior where
 * values inserted at coarse levels are inherited by finer levels,
 * and fine level values can override coarse level values.
 *
 * Test cases:
 * 1. Hierarchical insert and find, coarse level inheritance.
 * 2. Override a finer level coordinate.
 * 3. Large block queries within one tile.
 * 4. Hash map root insert and inheritance.
 * 5. Multiple isolated regions.
 *****************************************************************/

TEST_F(RS_SparseTest, HierarchicalInsertAndFind) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    sparse.insert<1>({100, 23, 47}, 1, accessor);

    auto result = sparse.find({99, 22, 46}, accessor);
    ASSERT_TRUE(result != nullptr);
    ASSERT_EQ(*result, 1);

    sparse.free();
}

TEST_F(RS_SparseTest, OverrideFinerLevel) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    sparse.insert<1>({100, 23, 47}, 2, accessor);
    sparse.insert<2>({100, 23, 47}, 5, accessor);

    int* exact_result = sparse.find({100, 23, 47}, accessor);
    ASSERT_TRUE(exact_result != nullptr);
    ASSERT_EQ(*exact_result, 5);

    int* neighbor_result = sparse.find({101, 23, 47}, accessor);
    ASSERT_TRUE(neighbor_result != nullptr);
    ASSERT_EQ(*neighbor_result, 2);

    sparse.free();
}

TEST_F(RS_SparseTest, LargeBlockQueries) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    sparse.insert<1>({200, 100, 50}, 10, accessor);

    std::vector<RCoord3> coords_to_test = {
        {202, 102, 52},
        {203, 103, 53},
        {200, 96, 48},
        {207, 103, 55}
    };

    for (auto c : coords_to_test) {
        int* val = sparse.find(c, accessor);
        ASSERT_TRUE(val != nullptr);
        ASSERT_EQ(*val, 10);
    }

    int* invalid_val = sparse.find({208, 104, 56}, accessor);
    ASSERT_TRUE(invalid_val == nullptr);

    sparse.free();
}

TEST_F(RS_SparseTest, HashMapRootInsert) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    sparse.insert<0>({200, 100, 50}, 10, accessor);

    std::vector<RCoord3> test_coords = {
        {202, 102, 52},
        {203, 103, 53},
        {192, 96, 32},
        {223, 127, 63}
    };

    for (auto c : test_coords) {
        int* val = sparse.find(c, accessor);
        ASSERT_TRUE(val != nullptr);
        ASSERT_EQ(*val, 10);
    }

    sparse.insert<2>({200, 100, 50}, 50, accessor);
    int* overridden_val = sparse.find({200, 100, 50}, accessor);
    ASSERT_TRUE(overridden_val != nullptr);
    ASSERT_EQ(*overridden_val, 50);

    int* nearby_val = sparse.find({201, 100, 50}, accessor);
    ASSERT_TRUE(nearby_val != nullptr);
    ASSERT_EQ(*nearby_val, 10);

    sparse.free();
}

TEST_F(RS_SparseTest, MultipleRegions) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    sparse.insert<1>({0, 0, 0}, 11, accessor);
    sparse.insert<1>({256, 256, 256}, 22, accessor);
    sparse.insert<1>({1024, 1024, 1024}, 33, accessor);

    int* val1 = sparse.find({1, 1, 1}, accessor);
    ASSERT_TRUE(val1 != nullptr && *val1 == 11);

    int* val2 = sparse.find({257, 256, 256}, accessor);
    ASSERT_TRUE(val2 != nullptr && *val2 == 22);

    int* val3 = sparse.find({1025, 1024, 1024}, accessor);
    ASSERT_TRUE(val3 != nullptr && *val3 == 33);

    sparse.insert<2>({0, 0, 0}, 99, accessor);
    int* overridden = sparse.find({0, 0, 0}, accessor);
    ASSERT_TRUE(overridden != nullptr && *overridden == 99);

    val2 = sparse.find({257, 256, 256}, accessor);
    ASSERT_TRUE(val2 != nullptr && *val2 == 22);

    sparse.clear();
    sparse.free();
}

TEST_F(RS_SparseTest, ResizingConsistency) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    sparse.insert<1>({500, 500, 500}, 7, accessor);

    sparse.resize(4096, true);

    accessor.clear();

    int* val = sparse.find({500, 500, 500}, accessor);
    ASSERT_TRUE(val != nullptr && *val == 7);

    int* neighbor = sparse.find({501, 500, 500}, accessor);
    ASSERT_TRUE(neighbor != nullptr && *neighbor == 7);

    sparse.clear();
    sparse.free();
}

/*****************************************************************
 * Test cases for the array root on the host
 *
 * Test cases:
 * 1. Insert and find a value with an array root.
 * 2. Insert and find with a negative coordinate.
 * 3. Find a non existent value.
 *****************************************************************/

TEST_F(RS_SparseTest, FindExistingValueArray) {
    TestSparseArray sparse(&host_provider);
    sparse.resize({16, 16, 16});
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    sparse.insert<1>({10, 10, 10}, 42, accessor);

    int* result = sparse.find({10, 10, 10}, accessor);
    ASSERT_TRUE(result != nullptr);
    EXPECT_EQ(*result, 42);

    sparse.free();
}

TEST_F(RS_SparseTest, InsertNegativeCoordinatesArray) {
    TestSparseArray sparse(&host_provider);
    sparse.resize({16, 16, 16});
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    bool result = sparse.insert<1>({-10, -10, -10}, 42, accessor);
    EXPECT_TRUE(result);

    int* found = sparse.find({-10, -10, -10}, accessor);
    ASSERT_TRUE(found != nullptr);
    EXPECT_EQ(*found, 42);

    sparse.free();
}

TEST_F(RS_SparseTest, FindNonExistentValueArray) {
    TestSparseArray sparse(&host_provider);
    sparse.resize({16, 16, 16});
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);
    auto accessor = sparse.getAccessor();

    int* result = sparse.find({10, 10, 10}, accessor);
    EXPECT_TRUE(result == nullptr);

    sparse.free();
}

/*****************************************************************
 * Test cases for clear operations on the host
 *
 * Test cases:
 * 1. Clear with populated data, data is gone afterwards.
 * 2. Insert after clear.
 *****************************************************************/

TEST_F(RS_SparseTest, ClearWithPopulatedData) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    auto accessor = sparse.getAccessor();
    sparse.insert<1>({100, 100, 100}, 42, accessor);

    int* result = sparse.find({100, 100, 100}, accessor);
    ASSERT_TRUE(result != nullptr);

    sparse.clear(true);

    accessor.clear();
    result = sparse.find({100, 100, 100}, accessor);
    EXPECT_TRUE(result == nullptr);

    sparse.free();
}

TEST_F(RS_SparseTest, InsertAfterClear) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    auto accessor = sparse.getAccessor();
    sparse.insert<1>({100, 100, 100}, 42, accessor);

    sparse.clear(true);

    accessor.clear();
    sparse.insert<1>({200, 200, 200}, 84, accessor);

    int* result = sparse.find({200, 200, 200}, accessor);
    ASSERT_TRUE(result != nullptr);
    EXPECT_EQ(*result, 84);

    sparse.free();
}

/*****************************************************************
 * Test cases for copy operations on the host
 *
 * Test cases:
 * 1. Copy a hash map sparse map and verify the data.
 * 2. Copy an array sparse map and verify the data.
 *****************************************************************/

TEST_F(RS_SparseTest, CopySparseMapHashMap) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    auto accessor = sparse.getAccessor();
    sparse.insert<1>({100, 100, 100}, 42, accessor);
    sparse.insert<1>({200, 200, 200}, 84, accessor);

    TestSparseHashMap sparse2(&host_provider);
    sparse2.copy(sparse);

    accessor.clear();
    int* result1 = sparse2.find({100, 100, 100}, accessor);
    ASSERT_TRUE(result1 != nullptr);
    EXPECT_EQ(*result1, 42);

    int* result2 = sparse2.find({200, 200, 200}, accessor);
    ASSERT_TRUE(result2 != nullptr);
    EXPECT_EQ(*result2, 84);

    sparse.free();
    sparse2.free();
}

TEST_F(RS_SparseTest, CopySparseMapArray) {
    TestSparseArray sparse(&host_provider);
    sparse.resize({16, 16, 16});
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    auto accessor = sparse.getAccessor();
    sparse.insert<1>({10, 10, 10}, 42, accessor);

    TestSparseArray sparse2(&host_provider);
    sparse2.copy(sparse);

    accessor.clear();
    int* result = sparse2.find({10, 10, 10}, accessor);
    ASSERT_TRUE(result != nullptr);
    EXPECT_EQ(*result, 42);

    sparse.free();
    sparse2.free();
}

/*****************************************************************
 * Test cases for boundary conditions on the host
 *
 * Test cases:
 * 1. Coordinate at the exact bounds.
 * 2. Coordinate just outside the bounds.
 * 3. Zero coordinate.
 *****************************************************************/

TEST_F(RS_SparseTest, CoordinateAtBounds) {
    TestSparseArray sparse(&host_provider);
    sparse.resize({16, 16, 16});
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    RCoord3 max_coord = sparse.maxCoord();
    RCoord3 min_coord = sparse.minCoord();

    auto accessor = sparse.getAccessor();

    bool result1 = sparse.insert<1>(max_coord, 42, accessor);
    bool result2 = sparse.insert<1>(min_coord, 84, accessor);

    EXPECT_TRUE(result1);
    EXPECT_TRUE(result2);

    sparse.free();
}

TEST_F(RS_SparseTest, CoordinateOutsideBounds) {
    TestSparseArray sparse(&host_provider);
    sparse.resize({16, 16, 16});
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    RCoord3 max_coord = sparse.maxCoord();

    auto accessor = sparse.getAccessor();

    bool result = sparse.insert<1>(max_coord + RCoord3(1, 1, 1), 42, accessor);
    EXPECT_FALSE(result);

    sparse.free();
}

TEST_F(RS_SparseTest, ZeroCoordinate) {
    TestSparseHashMap sparse(&host_provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    auto accessor = sparse.getAccessor();

    bool result = sparse.insert<1>({0, 0, 0}, 42, accessor);
    EXPECT_TRUE(result);

    int* found = sparse.find({0, 0, 0}, accessor);
    ASSERT_TRUE(found != nullptr);
    EXPECT_EQ(*found, 42);

    sparse.free();
}

/*****************************************************************
 * Test cases for insert and find on the device
 *
 * A bare compute function inserts values at coordinates and another
 * one finds them, with both coordinates and results passed through
 * RS_Vector buffers that are moved to the host for checking.
 *
 * Test cases:
 * 1. Insert at level 1 on the device.
 * 2. Insert then find on the device.
 *****************************************************************/

TEST_F(RS_SparseTest, DeviceInsertAtLevel1) {
    TestSparseHashMap sparse(&provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    RS_Vector<RCoord3> coords(1, &provider);
    RS_Vector<int> values(1, &provider);
    RS_Vector<int> results(1, &provider);
    results.fill(0);

    RCoord3 coord(100, 100, 100);
    int value = 42;
    coords.write(&coord);
    values.write(&value);

    provider.run<RSCF_SparseInsert<TestSparseHashMap, 1>>({1}, sparse, coords, values, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    EXPECT_EQ(host_results[0], 1);

    coords.free();
    values.free();
    results.free();
    host_results.free();
    sparse.free();
}

TEST_F(RS_SparseTest, DeviceInsertThenFind) {
    TestSparseHashMap sparse(&provider);
    sparse.resize(2048);
    sparse.resizeMemPool(1, 1024);
    sparse.resizeMemPool(2, 1024);

    RS_Vector<RCoord3> coords(1, &provider);
    RS_Vector<int> values(1, &provider);
    RS_Vector<int> results(1, &provider);
    RS_Vector<int> found(1, &provider);
    results.fill(0);
    found.fill(0);

    RCoord3 coord(100, 100, 100);
    int value = 42;
    coords.write(&coord);
    values.write(&value);

    provider.run<RSCF_SparseInsert<TestSparseHashMap, 1>>({1}, sparse, coords, values, results);
    provider.run<RSCF_SparseFind<TestSparseHashMap>>({1}, sparse, coords, values, found);

    RS_Vector<int> host_values = values.move(&host_provider);
    RS_Vector<int> host_found = found.move(&host_provider);

    EXPECT_EQ(host_found[0], 1);
    EXPECT_EQ(host_values[0], 42);

    coords.free();
    values.free();
    results.free();
    found.free();
    host_values.free();
    host_found.free();
    sparse.free();
}
