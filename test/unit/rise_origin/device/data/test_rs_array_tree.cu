#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_array_tree.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

template <typename ArrayTreeType, RIndex Level>
class RS_ClaimLevel : public RS_ComputeFunction<RS_ClaimLevel<ArrayTreeType, Level>> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(ArrayTreeType tree, RS_Array<RCoord3, 1> coords, RS_Array<bool, 1> results) {
        RSize idx = RS_ComputeFunction<RS_ClaimLevel<ArrayTreeType, Level>>::index1D();
        RCoord3 coord = coords[idx];
        results[idx] = tree.template claim<Level>(coord, idx);
    }
};

template <typename ArrayTreeType, RIndex Level>
class RS_AllocateLevel : public RS_ComputeFunction<RS_AllocateLevel<ArrayTreeType, Level>> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(ArrayTreeType tree, RS_Array<RCoord3, 1> coords, RS_Array<bool, 1> results) {
        RSize idx = RS_ComputeFunction<RS_AllocateLevel<ArrayTreeType, Level>>::index1D();
        RCoord3 coord = coords[idx];
        results[idx] = tree.template allocate<Level>(coord, idx);
    }
};

template <typename ArrayTreeType>
class RS_InsertValues : public RS_ComputeFunction<RS_InsertValues<ArrayTreeType>> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(ArrayTreeType tree, RS_Array<RCoord3, 1> coords, RS_Array<float, 1> values) {
        RSize idx = RS_ComputeFunction<RS_InsertValues<ArrayTreeType>>::index1D();
        RCoord3 coord = coords[idx];
        if (tree.insert(coord, idx)) {
            values[idx] = *tree.find(coord);
        } else {
            values[idx] = -1.0f;
        }
    }
};

template <typename ArrayTreeType>
class RS_CheckValues : public RS_ComputeFunction<RS_CheckValues<ArrayTreeType>> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        ArrayTreeType tree,
        RS_Array<RCoord3, 1> coords,
        RS_Array<float, 1> expected,
        RS_Array<bool, 1> results
    ) {
        RSize idx = RS_ComputeFunction<RS_CheckValues<ArrayTreeType>>::index1D();
        RCoord3 coord = coords[idx];
        float* ptr = tree.find(coord);
        bool valid = (ptr != nullptr) && (*ptr == expected[idx]);
        results[idx] = valid;
    }
};

class RS_ArrayTreeTest : public ::testing::Test {
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

TEST_F(RS_ArrayTreeTest, InsertAndRetrieveMultiLevel) {
    using TestArrayTree = RS_ArrayTree3DMap<
        float,
        RS_ArrayTree3DLevel<8, 8, 8>,  // Level 0 (Root)
        RS_ArrayTree3DLevel<8, 8, 8>,  // Level 1 (Internal)
        RS_ArrayTree3DLevel<8, 8, 8>   // Level 2 (Leaf)
        >;

    TestArrayTree tree(&provider);
    tree.resizeMemPool(1, 2);  // Level 1 mem pool
    tree.resizeMemPool(2, 2);  // Level 2 mem pool

    // Generate test coordinates - two spatial groups
    std::vector<RCoord3> host_coords;
    const int GROUP_SIZE = 8;

    // Group 1: [0,8) in all dimensions
    for (int x = 0; x < GROUP_SIZE; x++) {
        for (int y = 0; y < GROUP_SIZE; y++) {
            for (int z = 0; z < GROUP_SIZE; z++) {
                host_coords.emplace_back(x, y, z);
            }
        }
    }

    // Group 2: [248,256) in all dimensions (edge of 512^3 grid)
    const int OFFSET = 248;
    for (int x = OFFSET; x < OFFSET + GROUP_SIZE; x++) {
        for (int y = OFFSET; y < OFFSET + GROUP_SIZE; y++) {
            for (int z = OFFSET; z < OFFSET + GROUP_SIZE; z++) {
                host_coords.emplace_back(x, y, z);
            }
        }
    }

    const RSize num_coords = host_coords.size();
    RS_Array<RCoord3, 1> d_coords({num_coords}, &provider);
    d_coords.write(host_coords.data());

    // Buffers for results
    RS_Array<bool, 1> claim_results[3] = {
        RS_Array<bool, 1>({num_coords}, &provider),
        RS_Array<bool, 1>({num_coords}, &provider),
        RS_Array<bool, 1>({num_coords}, &provider)
    };

    RS_Array<bool, 1> alloc_results[3] = {
        RS_Array<bool, 1>({num_coords}, &provider),
        RS_Array<bool, 1>({num_coords}, &provider),
        RS_Array<bool, 1>({num_coords}, &provider)
    };

    RS_Array<float, 1> d_values({num_coords}, &provider);

    // Single batched execution
    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        // Level 0 operations
        p_ptr->template run<RS_ClaimLevel<TestArrayTree, 0>>(
            {num_coords}, tree, d_coords, claim_results[0]
        );
        p_ptr->template run<RS_AllocateLevel<TestArrayTree, 0>>(
            {num_coords}, tree, d_coords, alloc_results[0]
        );

        // Level 1 operations
        p_ptr->template run<RS_ClaimLevel<TestArrayTree, 1>>(
            {num_coords}, tree, d_coords, claim_results[1]
        );
        p_ptr->template run<RS_AllocateLevel<TestArrayTree, 1>>(
            {num_coords}, tree, d_coords, alloc_results[1]
        );

        p_ptr->sync();

        // Final insertion
        p_ptr->template run<RS_InsertValues<TestArrayTree>>({num_coords}, tree, d_coords, d_values);
    });

    // Verification
    std::vector<float> host_values(num_coords);
    d_values.read(host_values.data());

    for (RSize i = 0; i < num_coords; i++) {
        EXPECT_EQ(host_values[i], i) << "Failed to insert value at index " << i;
    }

    // Cleanup
    d_coords.free();
    for (auto& arr : claim_results) arr.free();
    for (auto& arr : alloc_results) arr.free();
    d_values.free();
    tree.free();
}

TEST_F(RS_ArrayTreeTest, OutOfBoundAccess) {
    using TestArrayTree = RS_ArrayTree3DMap<
        float,
        RS_ArrayTree3DLevel<8, 8, 8>,
        RS_ArrayTree3DLevel<8, 8, 8>,
        RS_ArrayTree3DLevel<8, 8, 8>>;

    TestArrayTree tree(&provider);
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    // Test coordinate at maximum grid boundary (512^3)
    const RCoord3 test_coord(256, 256, 256);  // Beyond 512^3/2 = 256
    RS_Array<RCoord3, 1> d_coords({1}, &provider);
    d_coords.write(&test_coord);

    RS_Array<bool, 1> claim_result({1}, &provider);

    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_ClaimLevel<TestArrayTree, 0>>({1}, tree, d_coords, claim_result);
    });

    bool host_result = true;
    claim_result.read(&host_result);
    EXPECT_FALSE(host_result) << "Should fail for out-of-bound coordinate";

    d_coords.free();
    claim_result.free();
    tree.free();
}

TEST_F(RS_ArrayTreeTest, ResizeRoot) {
    using TestArrayTree = RS_ArrayTree3DMap<
        float,
        RS_ArrayTree3DLevel<8, 8, 8>,
        RS_ArrayTree3DLevel<8, 8, 8>,
        RS_ArrayTree3DLevel<8, 8, 8>>;

    TestArrayTree tree(&provider);

    // Initial size (8^3 root)
    tree.resizeMemPool(1, 2);
    tree.resizeMemPool(2, 2);

    // Resize root to 16^3
    tree.resize({16, 16, 16}, false);
    tree.resizeMemPool(1, 4);  // Increase mem pool for larger root
    tree.resizeMemPool(2, 4);

    // Test coordinate in new range
    const RCoord3 test_coord(300, 300, 300);
    RS_Array<RCoord3, 1> d_coords({1}, &provider);
    d_coords.write(&test_coord);

    RS_Array<bool, 1> claim_result({1}, &provider);
    RS_Array<bool, 1> alloc_result({1}, &provider);
    RS_Array<float, 1> d_value({1}, &provider);

    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_ClaimLevel<TestArrayTree, 0>>({1}, tree, d_coords, claim_result);
        p_ptr->template run<RS_AllocateLevel<TestArrayTree, 0>>({1}, tree, d_coords, alloc_result);
        p_ptr->template run<RS_ClaimLevel<TestArrayTree, 1>>({1}, tree, d_coords, claim_result);
        p_ptr->template run<RS_AllocateLevel<TestArrayTree, 1>>({1}, tree, d_coords, alloc_result);
        p_ptr->template run<RS_InsertValues<TestArrayTree>>({1}, tree, d_coords, d_value);
    });

    std::vector<float> host_value(1);
    d_value.read(host_value.data());
    EXPECT_NE(host_value[0], -1.0f) << "Should handle resized coordinate";

    d_coords.free();
    claim_result.free();
    alloc_result.free();
    d_value.free();
    tree.free();
}
