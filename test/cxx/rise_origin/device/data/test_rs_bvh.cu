#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_bvh.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_BVH is a bounding volume hierarchy over object bounds, built
 * with a parallel Morton code radix tree and a bottom up bound
 * propagation pass. It supports build, point queries through an
 * accessor, and incremental insert, update, and erase.
 *
 * - Build and Query
 *   build() from host vectors, then a stack based descent from
 *   the root to find which leaf bound contains a point
 *
 * - Incremental Edits
 *   insert(), erase(), update() each rebuild the tree
 *****************************************************************/

/**
 * @brief Walks a bvh from the root and returns the handle of the leaf whose
 * bound contains the query point, or RS_NULL_INDEX when no leaf contains it.
 */
class RSCF_PointInBVH : public RS_ComputeFunction<RSCF_PointInBVH> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_BVH bvh, RS_Array<RVec3f, 1> points, RS_Array<RHandle, 1> results) {
        RSize idx = index1D();
        RVec3f pt = points[idx];
        RHandle found_handle = RS_NULL_INDEX;

        static constexpr int stack_size = 64;
        RIndex stack[stack_size];
        int top = 0;

        auto accessor = bvh.root();
        if (accessor.current_node == RS_NULL_INDEX) {
            results[idx] = found_handle;
            return;
        }

        stack[top++] = accessor.current_node;

        while (top > 0) {
            RIndex node_index = stack[--top];
            if (node_index == RS_NULL_INDEX) {
                continue;
            }
            accessor.moveTo(node_index);

            auto bound = accessor.bound();
            if (!bound.isContaining(pt)) {
                continue;
            }

            RIndex left_index = accessor.left();
            RIndex right_index = accessor.right();

            if (accessor.isLeaf()) {
                if (accessor.object()) {
                    found_handle = accessor.object()->object_handle;
                }
                break;
            }

            if (left_index != RS_NULL_INDEX && top < stack_size) {
                stack[top++] = left_index;
            }
            if (right_index != RS_NULL_INDEX && top < stack_size) {
                stack[top++] = right_index;
            }
        }

        results[idx] = found_handle;
    }
};

class RS_BVHTest : public ::testing::Test {
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

    /// @brief Builds a grid of unit boxes, each tagged with a unique handle.
    void createSampleAABBs(
        RS_Vector<RS_BVHObjectIdentifier>& obj_ids,
        RS_Vector<RBound3f>& bounds,
        int box_num_per_dim,
        float spacing = 2.0f
    ) {
        RIndex idx = 0;
        for (int x = 0; x < box_num_per_dim; x++) {
            for (int y = 0; y < box_num_per_dim; y++) {
                for (int z = 0; z < box_num_per_dim; z++) {
                    RS_BVHObjectIdentifier id;
                    id.object_metadata = idx;
                    id.object_handle = static_cast<RHandle>(idx);

                    float start_x = x * spacing;
                    float start_y = y * spacing;
                    float start_z = z * spacing;

                    RVec3f min_bound(start_x, start_y, start_z);
                    RVec3f max_bound(start_x + 1, start_y + 1, start_z + 1);

                    obj_ids.append(id);
                    bounds.append(RBound3f(min_bound, max_bound));
                    idx++;
                }
            }
        }
    }
};

/*****************************************************************
 * Test cases for building a bvh and querying points
 *
 * Test cases:
 * 1. Build on the device, then query each box center.
 * 2. Build on the host provider, then query each box center.
 *****************************************************************/

TEST_F(RS_BVHTest, BuildAndQueryOnDevice) {
    constexpr int box_num_per_dim = 3;  // 3^3 = 27 boxes
    constexpr int box_num = box_num_per_dim * box_num_per_dim * box_num_per_dim;
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(box_num, &host_provider);
    RS_Vector<RBound3f> host_bounds(box_num, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    RS_BVH bvh(&provider);
    bvh.build(host_obj_ids, host_bounds);

    RS_Vector<RVec3f> host_query_points(0, &host_provider);
    for (RSize i = 0; i < host_obj_ids.size(); i++) {
        host_query_points.append(host_bounds[i].center());
    }
    // A point outside every box.
    host_query_points.append(RVec3f(-10.f, -10.f, -10.f));

    RS_Array<RVec3f, 1> d_query_points({host_query_points.size()}, &provider);
    RS_Array<RHandle, 1> d_results({host_query_points.size()}, &provider);
    d_query_points.write(host_query_points.ptr());
    d_results.fill(RS_NULL_INDEX);

    provider.run<RSCF_PointInBVH>({host_query_points.size()}, bvh, d_query_points, d_results);

    std::vector<RHandle> results(host_query_points.size(), RS_NULL_INDEX);
    d_results.read(results.data());

    for (RSize i = 0; i < host_obj_ids.size(); i++) {
        EXPECT_EQ(results[i], host_obj_ids[i].object_handle);
    }
    EXPECT_EQ(results.back(), RS_NULL_INDEX);

    host_obj_ids.free();
    host_bounds.free();
    host_query_points.free();
    d_query_points.free();
    d_results.free();
    bvh.free();
}

TEST_F(RS_BVHTest, BuildAndQueryOnHost) {
    constexpr int box_num_per_dim = 3;  // 3^3 = 27 boxes
    constexpr int box_num = box_num_per_dim * box_num_per_dim * box_num_per_dim;
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(box_num, &host_provider);
    RS_Vector<RBound3f> host_bounds(box_num, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    RS_BVH bvh(&host_provider);
    bvh.build(host_obj_ids, host_bounds);

    RS_Vector<RVec3f> host_query_points(0, &host_provider);
    for (RSize i = 0; i < host_obj_ids.size(); i++) {
        host_query_points.append(host_bounds[i].center());
    }
    host_query_points.append(RVec3f(-10.f, -10.f, -10.f));

    RS_Array<RVec3f, 1> d_query_points({host_query_points.size()}, &host_provider);
    RS_Array<RHandle, 1> d_results({host_query_points.size()}, &host_provider);
    d_query_points.write(host_query_points.ptr());
    d_results.fill(RS_NULL_INDEX);

    host_provider.run<RSCF_PointInBVH>({host_query_points.size()}, bvh, d_query_points, d_results);

    std::vector<RHandle> results(host_query_points.size(), RS_NULL_INDEX);
    d_results.read(results.data());

    for (RSize i = 0; i < host_obj_ids.size(); i++) {
        EXPECT_EQ(results[i], host_obj_ids[i].object_handle);
    }
    EXPECT_EQ(results.back(), RS_NULL_INDEX);

    host_obj_ids.free();
    host_bounds.free();
    host_query_points.free();
    d_query_points.free();
    d_results.free();
    bvh.free();
}

/*****************************************************************
 * Test cases for incremental bvh edits
 *
 * Test cases:
 * 1. Erase a box and confirm its center is no longer found.
 * 2. Update a box and confirm the old center misses, new hits.
 * 3. Insert a new box and confirm its center is found.
 *****************************************************************/

TEST_F(RS_BVHTest, EraseOnDevice) {
    constexpr int box_num_per_dim = 2;  // 2^3 = 8 boxes
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(0, &host_provider);
    RS_Vector<RBound3f> host_bounds(0, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    RS_BVH bvh(&provider);
    bvh.build(host_obj_ids, host_bounds);

    auto object_to_erase = host_obj_ids.back();
    RVec3f erase_box_center = host_bounds.back().center();

    RS_Array<RVec3f, 1> d_query_points({1}, &provider);
    d_query_points.writeSlice(&erase_box_center, 1, 0);

    RS_Array<RHandle, 1> d_results({1}, &provider);
    d_results.fill(RS_NULL_INDEX);

    provider.run<RSCF_PointInBVH>({1}, bvh, d_query_points, d_results);

    RHandle first_result = RS_NULL_INDEX;
    d_results.readSlice(&first_result, 1, 0);
    EXPECT_EQ(first_result, object_to_erase.object_handle)
        << "Before erasing, the box should be found";

    bvh.erase(object_to_erase);

    d_results.fill(RS_NULL_INDEX);
    provider.run<RSCF_PointInBVH>({1}, bvh, d_query_points, d_results);

    RHandle second_result = RS_NULL_INDEX;
    d_results.readSlice(&second_result, 1, 0);
    EXPECT_EQ(second_result, RS_NULL_INDEX) << "After erasing, the box should not be found";

    d_query_points.free();
    d_results.free();
    bvh.free();
    host_obj_ids.free();
    host_bounds.free();
}

TEST_F(RS_BVHTest, UpdateOnDevice) {
    constexpr int box_num_per_dim = 1;  // 1 box
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(0, &host_provider);
    RS_Vector<RBound3f> host_bounds(0, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    RS_BVH bvh(&provider);
    bvh.build(host_obj_ids, host_bounds);

    RVec3f old_center = host_bounds[0].center();

    RS_Array<RVec3f, 1> d_query_points({1}, &provider);
    d_query_points.writeSlice(&old_center, 1, 0);

    RS_Array<RHandle, 1> d_results({1}, &provider);
    d_results.fill(RS_NULL_INDEX);

    provider.run<RSCF_PointInBVH>({1}, bvh, d_query_points, d_results);

    RHandle first_result = RS_NULL_INDEX;
    d_results.readSlice(&first_result, 1, 0);
    EXPECT_EQ(first_result, host_obj_ids[0].object_handle) << "Box not found before update";

    RVec3f shift(10.0f, 0.f, 0.f);
    RBound3f new_bound = host_bounds[0].shift(shift);
    bvh.update(host_obj_ids[0], new_bound);

    d_results.fill(RS_NULL_INDEX);
    provider.run<RSCF_PointInBVH>({1}, bvh, d_query_points, d_results);

    RHandle second_result = 0;
    d_results.readSlice(&second_result, 1, 0);
    EXPECT_EQ(second_result, RS_NULL_INDEX) << "Old center should miss after update";

    RVec3f new_center = new_bound.center();
    d_query_points.writeSlice(&new_center, 1, 0);
    d_results.fill(RS_NULL_INDEX);

    provider.run<RSCF_PointInBVH>({1}, bvh, d_query_points, d_results);

    RHandle third_result = RS_NULL_INDEX;
    d_results.readSlice(&third_result, 1, 0);
    EXPECT_EQ(third_result, host_obj_ids[0].object_handle) << "New center should hit after update";

    d_query_points.free();
    d_results.free();
    bvh.free();
    host_obj_ids.free();
    host_bounds.free();
}

TEST_F(RS_BVHTest, InsertOnDevice) {
    constexpr int box_num_per_dim = 2;  // 2^3 = 8 boxes
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(0, &host_provider);
    RS_Vector<RBound3f> host_bounds(0, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    RS_BVH bvh(&provider);
    bvh.build(host_obj_ids, host_bounds);

    RS_BVHObjectIdentifier new_id;
    new_id.object_metadata = 42;
    new_id.object_handle = 12345;

    RBound3f new_bound(10.f, 10.f, 10.f, 11.f, 11.f, 11.f);
    RVec3f new_box_center = new_bound.center();

    RS_Array<RVec3f, 1> d_query_points({1}, &provider);
    d_query_points.writeSlice(&new_box_center, 1, 0);

    RS_Array<RHandle, 1> d_results({1}, &provider);
    d_results.fill(RS_NULL_INDEX);

    provider.run<RSCF_PointInBVH>({1}, bvh, d_query_points, d_results);

    RHandle initial_result = RS_NULL_INDEX;
    d_results.readSlice(&initial_result, 1, 0);
    EXPECT_EQ(initial_result, RS_NULL_INDEX) << "New box should not be found before insert";

    bvh.insert(new_id, new_bound);

    d_results.fill(RS_NULL_INDEX);
    provider.run<RSCF_PointInBVH>({1}, bvh, d_query_points, d_results);

    RHandle second_result = RS_NULL_INDEX;
    d_results.readSlice(&second_result, 1, 0);
    EXPECT_EQ(second_result, new_id.object_handle) << "New box should be found after insert";

    bvh.free();
    host_obj_ids.free();
    host_bounds.free();
    d_query_points.free();
    d_results.free();
}
