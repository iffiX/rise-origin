#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_array_tree.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_ArrayTree3DMap and RS_ArrayTree3DMultiMap are hierarchical 3D
 * spatial containers built from a fixed set of level configs. Each
 * coordinate is reached by a stack based descent through the levels,
 * with a per level claim and allocate pass before values are stored.
 *
 * - Hierarchical Build
 *   claim() marks a thread for a level, allocate() reserves the page
 *
 * - Map Values
 *   insert() stores one value at a coordinate, find() reads it back
 *
 * - MultiMap Values
 *   insert() pushes onto a fixed size stack at a coordinate,
 *   find() returns the first value and the value count
 *
 * - Boundary Conditions
 *   Out of bound coordinate fails to claim
 *****************************************************************/

class RS_ArrayTreeTest : public ::testing::Test {
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
 * Compute functions for array tree operations
 *****************************************************************/

template <typename ArrayTreeType, RIndex Level>
class RSCF_ClaimLevel : public RS_ComputeFunction<RSCF_ClaimLevel<ArrayTreeType, Level>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_ClaimLevel<ArrayTreeType, Level>>;
    using BaseType::index1D;
    /// @brief Claims the level for each coordinate using the thread index.
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(ArrayTreeType tree, RS_Array<RCoord3, 1> coords, RS_Array<bool, 1> results) {
        RSize idx = index1D();
        RCoord3 coord = coords[idx];
        results[idx] = tree.template claim<Level>(coord, idx);
    }
};

template <typename ArrayTreeType, RIndex Level>
class RSCF_AllocateLevel : public RS_ComputeFunction<RSCF_AllocateLevel<ArrayTreeType, Level>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_AllocateLevel<ArrayTreeType, Level>>;
    using BaseType::index1D;
    /// @brief Allocates the level for each coordinate using the thread index.
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(ArrayTreeType tree, RS_Array<RCoord3, 1> coords, RS_Array<bool, 1> results) {
        RSize idx = index1D();
        RCoord3 coord = coords[idx];
        results[idx] = tree.template allocate<Level>(coord, idx);
    }
};

template <typename ArrayTreeType>
class RSCF_InsertValues : public RS_ComputeFunction<RSCF_InsertValues<ArrayTreeType>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_InsertValues<ArrayTreeType>>;
    using BaseType::index1D;
    /// @brief Inserts one value per coordinate into the map.
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        ArrayTreeType tree,
        RS_Array<RCoord3, 1> coords,
        RS_Array<float, 1> values,
        RS_Array<bool, 1> results
    ) {
        RSize idx = index1D();
        RCoord3 coord = coords[idx];
        results[idx] = tree.insert(coord, values[idx]);
    }
};

template <typename ArrayTreeType>
class RSCF_FindValues : public RS_ComputeFunction<RSCF_FindValues<ArrayTreeType>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_FindValues<ArrayTreeType>>;
    using BaseType::index1D;
    /// @brief Reads back one value per coordinate from the map.
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        ArrayTreeType tree,
        RS_Array<RCoord3, 1> coords,
        RS_Array<float, 1> values,
        RS_Array<bool, 1> found
    ) {
        RSize idx = index1D();
        RCoord3 coord = coords[idx];
        float* ptr = tree.find(coord);
        if (ptr != nullptr) {
            values[idx] = *ptr;
            found[idx] = true;
        } else {
            values[idx] = -1.0f;
            found[idx] = false;
        }
    }
};

template <typename ArrayTreeType>
class RSCF_InsertMultiValues : public RS_ComputeFunction<RSCF_InsertMultiValues<ArrayTreeType>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_InsertMultiValues<ArrayTreeType>>;
    using BaseType::index1D;
    /// @brief Pushes a run of values onto the stack at each coordinate.
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        ArrayTreeType tree,
        RS_Array<RCoord3, 1> coords,
        RS_Array<float, 1> base_values,
        RS_Array<RSize, 1> value_counts,
        RS_Array<bool, 1> results
    ) {
        RSize idx = index1D();
        RCoord3 coord = coords[idx];
        typename ArrayTreeType::RS_ArrayTree3DMultiMapAccessor accessor = tree.getAccessor();
        RSize count = value_counts[idx];
        bool is_all_ok = true;
        for (RSize i = 0; i < count; ++i) {
            if (!tree.insert(coord, base_values[idx] + static_cast<float>(i), accessor)) {
                is_all_ok = false;
                break;
            }
        }
        results[idx] = is_all_ok;
    }
};

template <typename ArrayTreeType>
class RSCF_GetMultiSize : public RS_ComputeFunction<RSCF_GetMultiSize<ArrayTreeType>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_GetMultiSize<ArrayTreeType>>;
    using BaseType::index1D;
    /// @brief Reads the stack size at each coordinate.
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(ArrayTreeType tree, RS_Array<RCoord3, 1> coords, RS_Array<RSize, 1> sizes) {
        RSize idx = index1D();
        RCoord3 coord = coords[idx];
        RSize value_num = 0;
        tree.find(coord, value_num);
        sizes[idx] = value_num;
    }
};

/*****************************************************************
 * Test cases for RS_ArrayTree3DMap construction
 *
 * Test cases:
 * 1. Construct with valid provider.
 * 2. Default constructor has no provider.
 * 3. Free resets the tree.
 *****************************************************************/

using TestArrayTree = RS_ArrayTree3DMap<
    float,
    RS_ArrayTree3DLevel<8, 8, 8>,  // Level 0 (Root)
    RS_ArrayTree3DLevel<8, 8, 8>,  // Level 1 (Internal)
    RS_ArrayTree3DLevel<8, 8, 8>   // Level 2 (Leaf)
    >;

TEST_F(RS_ArrayTreeTest, ConstructWithValidProvider) {
    TestArrayTree tree(&provider);
    EXPECT_TRUE(tree.provider() != nullptr);
    tree.free();
}

TEST_F(RS_ArrayTreeTest, DefaultConstructor) {
    TestArrayTree tree;
    EXPECT_TRUE(tree.provider() == nullptr);
}

TEST_F(RS_ArrayTreeTest, FreeAfterInitialization) {
    TestArrayTree tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);
    tree.free();
    EXPECT_TRUE(tree.provider() == nullptr);
}

/*****************************************************************
 * Test cases for RS_ArrayTree3DMap claim and allocate
 *
 * Test cases:
 * 1. Claim a single position at level 0.
 * 2. Allocate after claim at level 0.
 * 3. Out of bound coordinate fails to claim.
 *****************************************************************/

TEST_F(RS_ArrayTreeTest, ClaimSinglePositionLevel0) {
    TestArrayTree tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    RCoord3 coord(4, 4, 4);
    RS_Array<RCoord3, 1> d_coords({1}, &provider);
    d_coords.write(&coord);

    RS_Array<bool, 1> d_results({1}, &provider);

    provider.run<RSCF_ClaimLevel<TestArrayTree, 0>>({1}, tree, d_coords, d_results);

    bool result = false;
    d_results.read(&result);
    EXPECT_TRUE(result);

    d_coords.free();
    d_results.free();
    tree.free();
}

TEST_F(RS_ArrayTreeTest, AllocateAfterClaim) {
    TestArrayTree tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    RCoord3 coord(4, 4, 4);
    RS_Array<RCoord3, 1> d_coords({1}, &provider);
    d_coords.write(&coord);

    RS_Array<bool, 1> d_claim_results({1}, &provider);
    RS_Array<bool, 1> d_alloc_results({1}, &provider);

    provider.run<RSCF_ClaimLevel<TestArrayTree, 0>>({1}, tree, d_coords, d_claim_results);
    provider.run<RSCF_AllocateLevel<TestArrayTree, 0>>({1}, tree, d_coords, d_alloc_results);

    bool result = false;
    d_alloc_results.read(&result);
    EXPECT_TRUE(result);

    d_coords.free();
    d_claim_results.free();
    d_alloc_results.free();
    tree.free();
}

TEST_F(RS_ArrayTreeTest, OutOfBoundAccess) {
    TestArrayTree tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    // The tree has a 512 wide grid, so half bound is 256.
    const RCoord3 test_coord(256, 256, 256);
    RS_Array<RCoord3, 1> d_coords({1}, &provider);
    d_coords.write(&test_coord);

    RS_Array<bool, 1> claim_result({1}, &provider);

    provider.run<RSCF_ClaimLevel<TestArrayTree, 0>>({1}, tree, d_coords, claim_result);

    bool host_result = true;
    claim_result.read(&host_result);
    EXPECT_FALSE(host_result) << "Should fail for out of bound coordinate";

    d_coords.free();
    claim_result.free();
    tree.free();
}

/*****************************************************************
 * Test cases for RS_ArrayTree3DMap insert and find
 *
 * Test cases:
 * 1. Insert one value and read it back.
 * 2. Find a value that was never inserted.
 * 3. Insert and retrieve many values across two spatial groups.
 *****************************************************************/

TEST_F(RS_ArrayTreeTest, InsertSingleValue) {
    TestArrayTree tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    RCoord3 coord(4, 4, 4);
    float insert_value = 42.0f;

    RS_Array<RCoord3, 1> d_coords({1}, &provider);
    d_coords.write(&coord);

    RS_Array<bool, 1> d_results({1}, &provider);
    RS_Array<float, 1> d_values({1}, &provider);
    d_values.write(&insert_value);

    provider.run<RSCF_ClaimLevel<TestArrayTree, 0>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayTree, 0>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_ClaimLevel<TestArrayTree, 1>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayTree, 1>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_InsertValues<TestArrayTree>>({1}, tree, d_coords, d_values, d_results);

    bool result = false;
    d_results.read(&result);
    EXPECT_TRUE(result);

    RS_Array<bool, 1> d_found({1}, &provider);
    provider.run<RSCF_FindValues<TestArrayTree>>({1}, tree, d_coords, d_values, d_found);

    float value = 0.0f;
    bool found = false;
    d_values.read(&value);
    d_found.read(&found);

    EXPECT_TRUE(found);
    EXPECT_EQ(value, 42.0f);

    d_coords.free();
    d_results.free();
    d_values.free();
    d_found.free();
    tree.free();
}

TEST_F(RS_ArrayTreeTest, FindNonExistentValue) {
    TestArrayTree tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    RCoord3 coord(4, 4, 4);
    RS_Array<RCoord3, 1> d_coords({1}, &provider);
    d_coords.write(&coord);

    RS_Array<float, 1> d_values({1}, &provider);
    RS_Array<bool, 1> d_found({1}, &provider);

    provider.run<RSCF_FindValues<TestArrayTree>>({1}, tree, d_coords, d_values, d_found);

    bool found = true;
    d_found.read(&found);
    EXPECT_FALSE(found);

    d_coords.free();
    d_values.free();
    d_found.free();
    tree.free();
}

TEST_F(RS_ArrayTreeTest, InsertAndRetrieveMultiLevel) {
    TestArrayTree tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    // Two spatial groups, one near the origin and one near the edge.
    std::vector<RCoord3> host_coords;
    const int group_size = 8;
    for (int x = 0; x < group_size; x++) {
        for (int y = 0; y < group_size; y++) {
            for (int z = 0; z < group_size; z++) {
                host_coords.emplace_back(x, y, z);
            }
        }
    }
    const int offset = 248;
    for (int x = offset; x < offset + group_size; x++) {
        for (int y = offset; y < offset + group_size; y++) {
            for (int z = offset; z < offset + group_size; z++) {
                host_coords.emplace_back(x, y, z);
            }
        }
    }

    const RSize num_coords = host_coords.size();
    std::vector<float> insert_values(num_coords);
    for (RSize i = 0; i < num_coords; ++i) {
        insert_values[i] = 1000.0f + static_cast<float>(i);
    }

    RS_Array<RCoord3, 1> d_coords({num_coords}, &provider);
    d_coords.write(host_coords.data());

    RS_Array<float, 1> d_values({num_coords}, &provider);
    d_values.write(insert_values.data());

    RS_Array<bool, 1> d_results({num_coords}, &provider);
    RS_Array<bool, 1> d_found({num_coords}, &provider);

    provider.run<RSCF_ClaimLevel<TestArrayTree, 0>>({num_coords}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayTree, 0>>({num_coords}, tree, d_coords, d_results);
    provider.run<RSCF_ClaimLevel<TestArrayTree, 1>>({num_coords}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayTree, 1>>({num_coords}, tree, d_coords, d_results);
    provider.run<RSCF_InsertValues<TestArrayTree>>({num_coords}, tree, d_coords, d_values, d_results);

    provider.run<RSCF_FindValues<TestArrayTree>>({num_coords}, tree, d_coords, d_values, d_found);

    std::vector<float> host_values(num_coords);
    RS_Vector<bool> found(num_coords, &host_provider);
    d_values.read(host_values.data());
    d_found.read(found.ptr());

    for (RSize i = 0; i < num_coords; i++) {
        EXPECT_TRUE(found[i]) << "Failed to find value at index " << i;
        EXPECT_EQ(host_values[i], 1000.0f + static_cast<float>(i))
            << "Failed to insert value at index " << i;
    }

    d_coords.free();
    d_values.free();
    d_results.free();
    d_found.free();
    found.free();
    tree.free();
}

/*****************************************************************
 * Test cases for RS_ArrayTree3DMultiMap insert, find, and size
 *
 * Test cases:
 * 1. Insert many values at one coordinate and read the size.
 * 2. Insert beyond the stack capacity and confirm the cap.
 * 3. Insert at several coordinates.
 *****************************************************************/

using TestArrayMultiMap = RS_ArrayTree3DMultiMap<
    float,
    8,                             // ValueNum, the stack capacity per coordinate
    RS_ArrayTree3DLevel<8, 8, 8>,  // Level 0 (Root)
    RS_ArrayTree3DLevel<8, 8, 8>,  // Level 1 (Internal)
    RS_ArrayTree3DLevel<8, 8, 8>   // Level 2 (Leaf)
    >;

TEST_F(RS_ArrayTreeTest, MultiMapInsertMultipleValues) {
    TestArrayMultiMap tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    RCoord3 coord(4, 4, 4);
    RS_Array<RCoord3, 1> d_coords({1}, &provider);
    d_coords.write(&coord);

    RS_Array<bool, 1> d_results({1}, &provider);
    float base_value = 100.0f;
    RS_Array<float, 1> d_values({1}, &provider);
    d_values.write(&base_value);

    RSize num_to_insert = 5;
    RS_Array<RSize, 1> d_num_values({1}, &provider);
    d_num_values.write(&num_to_insert);

    provider.run<RSCF_ClaimLevel<TestArrayMultiMap, 0>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayMultiMap, 0>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_ClaimLevel<TestArrayMultiMap, 1>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayMultiMap, 1>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_InsertMultiValues<TestArrayMultiMap>>(
        {1}, tree, d_coords, d_values, d_num_values, d_results
    );

    bool result = false;
    d_results.read(&result);
    EXPECT_TRUE(result);

    RS_Array<RSize, 1> d_sizes({1}, &provider);
    provider.run<RSCF_GetMultiSize<TestArrayMultiMap>>({1}, tree, d_coords, d_sizes);

    RSize size = 0;
    d_sizes.read(&size);
    EXPECT_EQ(size, num_to_insert);

    d_coords.free();
    d_results.free();
    d_values.free();
    d_num_values.free();
    d_sizes.free();
    tree.free();
}

TEST_F(RS_ArrayTreeTest, MultiMapInsertUntilFull) {
    TestArrayMultiMap tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    RCoord3 coord(4, 4, 4);
    RS_Array<RCoord3, 1> d_coords({1}, &provider);
    d_coords.write(&coord);

    RS_Array<bool, 1> d_results({1}, &provider);
    float base_value = 1.0f;
    RS_Array<float, 1> d_values({1}, &provider);
    d_values.write(&base_value);

    // The stack holds 8 values, so 10 inserts must report a failure.
    RSize num_to_insert = 10;
    RS_Array<RSize, 1> d_num_values({1}, &provider);
    d_num_values.write(&num_to_insert);

    provider.run<RSCF_ClaimLevel<TestArrayMultiMap, 0>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayMultiMap, 0>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_ClaimLevel<TestArrayMultiMap, 1>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayMultiMap, 1>>({1}, tree, d_coords, d_results);
    provider.run<RSCF_InsertMultiValues<TestArrayMultiMap>>(
        {1}, tree, d_coords, d_values, d_num_values, d_results
    );

    bool result = true;
    d_results.read(&result);
    EXPECT_FALSE(result) << "Should fail when exceeding ValueNum";

    RS_Array<RSize, 1> d_sizes({1}, &provider);
    provider.run<RSCF_GetMultiSize<TestArrayMultiMap>>({1}, tree, d_coords, d_sizes);

    RSize size = 0;
    d_sizes.read(&size);
    EXPECT_LE(size, 8u) << "Size should not exceed ValueNum";

    d_coords.free();
    d_results.free();
    d_values.free();
    d_num_values.free();
    d_sizes.free();
    tree.free();
}

TEST_F(RS_ArrayTreeTest, MultiMapInsertAtMultipleLocations) {
    TestArrayMultiMap tree(&provider);
    tree.resizeMemPool(1, 4);
    tree.resizeMemPool(2, 4);

    std::vector<RCoord3> coords = {RCoord3(0, 0, 0), RCoord3(4, 4, 4), RCoord3(7, 7, 7)};
    const RSize num_coords = coords.size();

    RS_Array<RCoord3, 1> d_coords({num_coords}, &provider);
    d_coords.write(coords.data());

    RS_Array<bool, 1> d_results({num_coords}, &provider);
    std::vector<float> base_values = {10.0f, 20.0f, 30.0f};
    RS_Array<float, 1> d_values({num_coords}, &provider);
    d_values.write(base_values.data());

    std::vector<RSize> counts(num_coords, 2);
    RS_Array<RSize, 1> d_num_values({num_coords}, &provider);
    d_num_values.write(counts.data());

    provider.run<RSCF_ClaimLevel<TestArrayMultiMap, 0>>({num_coords}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayMultiMap, 0>>({num_coords}, tree, d_coords, d_results);
    provider.run<RSCF_ClaimLevel<TestArrayMultiMap, 1>>({num_coords}, tree, d_coords, d_results);
    provider.run<RSCF_AllocateLevel<TestArrayMultiMap, 1>>({num_coords}, tree, d_coords, d_results);
    provider.run<RSCF_InsertMultiValues<TestArrayMultiMap>>(
        {num_coords}, tree, d_coords, d_values, d_num_values, d_results
    );

    RS_Vector<bool> results(num_coords, &host_provider);
    d_results.read(results.ptr());
    for (RSize i = 0; i < num_coords; ++i) {
        EXPECT_TRUE(results[i]) << "Failed to insert at location " << i;
    }

    RS_Array<RSize, 1> d_sizes({num_coords}, &provider);
    provider.run<RSCF_GetMultiSize<TestArrayMultiMap>>({num_coords}, tree, d_coords, d_sizes);

    std::vector<RSize> sizes(num_coords);
    d_sizes.read(sizes.data());
    for (RSize i = 0; i < num_coords; ++i) {
        EXPECT_EQ(sizes[i], 2u) << "Incorrect size at location " << i;
    }

    d_coords.free();
    d_results.free();
    d_values.free();
    d_num_values.free();
    d_sizes.free();
    results.free();
    tree.free();
}
