#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_bvh.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/**
 * ComputeFunction that traverses a BVH to find which AABB a given point belongs to.
 *
 * - We assume each leaf's bounding box encloses one object.
 * - We'll do a simple DFS from the root:
 *   1. If the point is within the bounding box of the current node, then:
 *      - If it's a leaf, record the leaf index and stop.
 *      - Otherwise go deeper: check left and/or right child.
 *   2. If the point is not inside the bounding box, backtrack.
 *
 * - For simplicity, once the point is found to be in a *single* leaf’s bounding box,
 *   we mark that in `results[idx]` and return. If we never find a leaf that contains
 *   the point, store `RS_NULL_INDEX` (or some sentinel).
 */
class RS_TestPointInBVH : public RS_ComputeFunction<RS_TestPointInBVH> {
   public:
    /**
     * @param bvh         The BVH we are traversing.
     * @param points      The array of query points.
     * @param results     The array to store each query's resulting object handle
     *                    (or RS_NULL_INDEX if no containing leaf found).
     */
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_BVH bvh, RS_Array<RVec3f, 1> points, RS_Array<RHandle, 1> results) {
        // Each thread (on device or host) will handle one point
        RSize idx = index1D();

        RVec3f pt = points[idx];

        RHandle found_handle = RS_NULL_INDEX;

        // 3) Prepare a stack-based DFS
        static constexpr int STACK_SIZE = 64;
        RIndex stack[STACK_SIZE];
        int top = 0;

        auto accessor = bvh.root();

        if (accessor.current_node == RS_NULL_INDEX) {
            return;
        }

        // Push the root node index
        stack[top++] = accessor.current_node;

        // 5) DFS loop
        while (top > 0) {
            // Pop from the stack
            RIndex nodeIndex = stack[--top];
            if (nodeIndex == RS_NULL_INDEX) {
                continue;
            }

            // Move the accessor to that node
            accessor.moveTo(nodeIndex);

            // Check bounding box
            auto bound = accessor.bound();

            if (!bound.isContaining(pt)) {
                // Not inside -> skip
                continue;
            }

            // If it is a leaf => gather the handle and break out
            RIndex left_index = accessor.left();
            RIndex right_index = accessor.right();

            if (accessor.isLeaf()) {
                // For a leaf, accessor.object() is a pointer to the leaf's object identifier
                if (accessor.object()) {
                    found_handle = accessor.object()->object_handle;
                }
                break;  // We found a containing leaf, so we can stop
            }

            // If it is an internal node, push its children (DFS)
            // (Take care not to overflow the stack in practice.)
            if (left_index != RS_NULL_INDEX && top < STACK_SIZE) {
                stack[top++] = left_index;
            }
            if (right_index != RS_NULL_INDEX && top < STACK_SIZE) {
                stack[top++] = right_index;
            }
        }

        // 6) Write back the result
        results[idx] = found_handle;
    }
};

class RS_BVHTest : public ::testing::Test {
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
 * @brief create some AABBs in a grid. Each box is 1 unit in size
 */
static void createSampleAABBs(
    RS_Vector<RS_BVHObjectIdentifier> &obj_ids,
    RS_Vector<RBound3f> &bounds,
    int box_num_per_dim,
    float spacing = 2.0f
) {
    // Each box is 1 unit in size in this example, spaced by spacing
    // along x, y, z for a simple arrangement.
    // object_handle can be set to anything for demonstration
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

/**
 * Tests building a BVH, then performing point-in-AABB queries in device code.
 */
TEST_F(RS_BVHTest, BuildAndQueryOnDevice) {
    // 1) Construct the BVH on device
    constexpr int box_num_per_dim = 3;  // total boxes = 3^3 = 27
    constexpr int box_num = box_num_per_dim * box_num_per_dim * box_num_per_dim;
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(box_num, &host_provider);
    RS_Vector<RBound3f> host_bounds(box_num, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    RS_BVH bvh(&provider);

    // 1) Build the BVH from the host vectors (object IDs and bounding boxes).
    bvh.build(host_obj_ids, host_bounds);

    // 2) Create some query points
    // Use the center of each bounding box, plus some out-of-range points.
    RS_Vector<RVec3f> host_query_points(0, &host_provider);
    for (size_t i = 0; i < host_obj_ids.size(); i++) {
        auto bound = host_bounds[i];
        RVec3f center = bound.center();
        host_query_points.append(center);
    }
    // Add a point that is definitely outside
    host_query_points.append(RVec3f(-10.f, -10.f, -10.f));

    // 3) Create device arrays for the queries and the results
    RS_Array<RVec3f, 1> d_query_points({host_query_points.size()}, &provider);
    RS_Array<RIndex, 1> d_results({host_query_points.size()}, &provider);
    d_query_points.write(host_query_points.ptr());
    d_results.fill(RS_NULL_INDEX);

    // 4) Run our compute kernel to do the search
    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_TestPointInBVH>(
            {host_query_points.size()}, bvh, d_query_points, d_results
        );
    });

    // 5) Check results on host side
    std::vector<RHandle> results(host_query_points.size(), RS_NULL_INDEX);
    d_results.read(results.data());

    // For each center, we expect the result to be the leaf index (object identifier)
    // of the box that encloses that point.
    for (RSize i = 0; i < host_obj_ids.size(); i++) {
        EXPECT_EQ(results[i], host_obj_ids[i].object_handle);
    }

    // The last query (out-of-range) is not in any bounding box, so it should be RS_NULL_INDEX
    EXPECT_EQ(results.back(), RS_NULL_INDEX);

    host_obj_ids.free();
    host_bounds.free();
    host_query_points.free();
    d_query_points.free();
    d_results.free();
    bvh.free();
}

TEST_F(RS_BVHTest, BuildAndQueryOnHost) {
    // 1) Construct the BVH on device
    constexpr int box_num_per_dim = 3;  // total boxes = 3^3 = 27
    constexpr int box_num = box_num_per_dim * box_num_per_dim * box_num_per_dim;
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(box_num, &host_provider);
    RS_Vector<RBound3f> host_bounds(box_num, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    RS_BVH bvh(&host_provider);

    // 1) Build the BVH from the host vectors (object IDs and bounding boxes).
    bvh.build(host_obj_ids, host_bounds);

    // 2) Create some query points
    // Use the center of each bounding box, plus some out-of-range points.
    RS_Vector<RVec3f> host_query_points(0, &host_provider);
    for (size_t i = 0; i < host_obj_ids.size(); i++) {
        auto bound = host_bounds[i];
        RVec3f center = bound.center();
        host_query_points.append(center);
    }
    // Add a point that is definitely outside
    host_query_points.append(RVec3f(-10.f, -10.f, -10.f));

    // 3) Create device arrays for the queries and the results
    RS_Array<RVec3f, 1> d_query_points({host_query_points.size()}, &host_provider);
    RS_Array<RIndex, 1> d_results({host_query_points.size()}, &host_provider);
    d_query_points.write(host_query_points.ptr());
    d_results.fill(RS_NULL_INDEX);

    // 4) Run our compute kernel to do the search
    castProviderPtrAndCall(&host_provider, [&](auto p_ptr) {
        p_ptr->template run<RS_TestPointInBVH>(
            {host_query_points.size()}, bvh, d_query_points, d_results
        );
    });

    // 5) Check results on host side
    std::vector<RHandle> results(host_query_points.size(), RS_NULL_INDEX);
    d_results.read(results.data());

    // For each center, we expect the result to be the leaf index (object identifier)
    // of the box that encloses that point.
    for (RSize i = 0; i < host_obj_ids.size(); i++) {
        EXPECT_EQ(results[i], host_obj_ids[i].object_handle);
    }

    // The last query (out-of-range) is not in any bounding box, so it should be RS_NULL_INDEX
    EXPECT_EQ(results.back(), RS_NULL_INDEX);

    host_obj_ids.free();
    host_bounds.free();
    host_query_points.free();
    d_query_points.free();
    d_results.free();
    bvh.free();
}

/**
 * @brief Test erasing an object from the BVH on device, verifying via point-in-BVH queries.
 */
TEST_F(RS_BVHTest, EraseOnDevice) {
    // 1) Create sample boxes on HOST
    constexpr int box_num_per_dim = 2;  // total boxes = 2^3 = 8
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(0, &host_provider);
    RS_Vector<RBound3f> host_bounds(0, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    // 2) Create BVH on DEVICE and build
    RS_BVH bvh(&provider);
    bvh.build(host_obj_ids, host_bounds);

    // 3) Pick an object to erase: let's pick the last one
    auto object_to_erase = host_obj_ids.back();
    // We'll create a query point that is definitely inside that box
    RVec3f erase_box_center = host_bounds.back().center();

    // 4) Create the query array (just one point)
    RS_Array<RVec3f, 1> d_query_points({1}, &provider);
    d_query_points.writeSlice(&erase_box_center, 1, 0);

    // 5) Create results array
    RS_Array<RHandle, 1> d_results({1}, &provider);
    d_results.fill(RS_NULL_INDEX);

    // 6) First check that we can indeed find this box
    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_TestPointInBVH>(
            {1},  // one thread
            bvh,
            d_query_points,
            d_results
        );
    });

    // Copy result back
    RHandle first_result = RS_NULL_INDEX;
    d_results.readSlice(&first_result, 1, 0);

    // Expect the handle to match the object we plan to erase
    EXPECT_EQ(first_result, object_to_erase.object_handle)
        << "Before erasing, we should find the correct box handle!";

    // 7) Now erase
    bvh.erase(object_to_erase);

    // 8) Run the same point-in-BVH query again
    d_results.fill(RS_NULL_INDEX);  // reset
    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_TestPointInBVH>(
            {1},  // one thread
            bvh,
            d_query_points,
            d_results
        );
    });

    RHandle second_result = RS_NULL_INDEX;
    d_results.readSlice(&second_result, 1, 0);
    // Now we expect no object found
    EXPECT_EQ(second_result, RS_NULL_INDEX) << "After erasing, the box should no longer be found !";

    // Cleanup
    d_query_points.free();
    d_results.free();
    bvh.free();
    host_obj_ids.free();
    host_bounds.free();
}

/**
 * @brief Test updating an object in the BVH on device, verifying via point-in-BVH queries.
 */
TEST_F(RS_BVHTest, UpdateOnDevice) {
    // 1) Create a single box (or more, if you like)
    constexpr int box_num_per_dim = 1;  // just 1^3=1 box for clarity
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(0, &host_provider);
    RS_Vector<RBound3f> host_bounds(0, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    // 2) Build the BVH on device
    RS_BVH bvh(&provider);
    bvh.build(host_obj_ids, host_bounds);

    // 3) The single box's center
    RVec3f old_center = host_bounds[0].center();

    // 4) Query array (just 1 point)
    RS_Array<RVec3f, 1> d_query_points({1}, &provider);
    d_query_points.writeSlice(&old_center, 1, 0);

    RS_Array<RHandle, 1> d_results({1}, &provider);
    d_results.fill(RS_NULL_INDEX);

    // 5) Confirm the box is found initially
    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_TestPointInBVH>(
            {1},  // one thread
            bvh,
            d_query_points,
            d_results
        );
    });

    RHandle first_result = RS_NULL_INDEX;
    d_results.readSlice(&first_result, 1, 0);
    EXPECT_EQ(first_result, host_obj_ids[0].object_handle) << "Box not found before update!";

    // 6) Update the bounding box so it no longer encloses old_center
    //    Let's shift it 10 units away
    RVec3f shift(10.0f, 0.f, 0.f);
    RBound3f new_bound = host_bounds[0].shift(shift);
    bvh.update(host_obj_ids[0], new_bound);

    // 7) Re-check with old_center => should not be found now
    d_results.fill(RS_NULL_INDEX);
    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_TestPointInBVH>({1}, bvh, d_query_points, d_results);
    });

    RHandle second_result = 0;
    d_results.readSlice(&second_result, 1, 0);
    EXPECT_EQ(second_result, RS_NULL_INDEX) << "Box still found after update, but shouldn't be!";

    // 8) But if we query a point inside the new bounding box, we should find it
    RVec3f new_center = new_bound.center();
    d_query_points.writeSlice(&new_center, 1, 0);
    d_results.fill(RS_NULL_INDEX);

    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_TestPointInBVH>({1}, bvh, d_query_points, d_results);
    });

    RHandle third_result = RS_NULL_INDEX;
    d_results.readSlice(&third_result, 1, 0);
    EXPECT_EQ(third_result, host_obj_ids[0].object_handle)
        << "Updated box not found with new center!";

    // Cleanup
    d_query_points.free();
    d_results.free();
    bvh.free();
    host_obj_ids.free();
    host_bounds.free();
}

/**
 * @brief Test inserting an object into the BVH on device, verifying via point-in-BVH queries.
 */
TEST_F(RS_BVHTest, InsertOnDevice) {
    // 1) Create a small set of boxes on HOST
    constexpr int box_num_per_dim = 2;  // total = 2^3 = 8
    RS_Vector<RS_BVHObjectIdentifier> host_obj_ids(0, &host_provider);
    RS_Vector<RBound3f> host_bounds(0, &host_provider);

    createSampleAABBs(host_obj_ids, host_bounds, box_num_per_dim);

    // 2) Build the BVH on the device
    RS_BVH bvh(&provider);
    bvh.build(host_obj_ids, host_bounds);

    // 3) Define a new box that we haven't inserted yet
    RS_BVHObjectIdentifier new_id;
    new_id.object_metadata = 42;  // arbitrary
    new_id.object_handle = 12345;

    RBound3f new_bound(10.f, 10.f, 10.f, 11.f, 11.f, 11.f);

    // 4) We'll check that a point in this new box is NOT found yet
    RVec3f new_box_center = new_bound.center();

    RS_Array<RVec3f, 1> d_query_points({1}, &provider);
    d_query_points.writeSlice(&new_box_center, 1, 0);

    RS_Array<RHandle, 1> d_results({1}, &provider);
    d_results.fill(RS_NULL_INDEX);

    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_TestPointInBVH>({1}, bvh, d_query_points, d_results);
    });

    RHandle initial_result = RS_NULL_INDEX;
    d_results.readSlice(&initial_result, 1, 0);
    EXPECT_EQ(initial_result, RS_NULL_INDEX)
        << "Point inside new box should not be found before insert!";

    // 5) Now insert the new box
    bvh.insert(new_id, new_bound);

    // 6) Re-run the point-in-BVH query
    d_results.fill(RS_NULL_INDEX);
    castProviderPtrAndCall(&provider, [&](auto p_ptr) {
        p_ptr->template run<RS_TestPointInBVH>({1}, bvh, d_query_points, d_results);
    });

    RHandle second_result = RS_NULL_INDEX;
    d_results.readSlice(&second_result, 1, 0);
    EXPECT_EQ(second_result, new_id.object_handle)
        << "Point inside newly inserted box should now be found!";

    // Cleanup
    bvh.free();
    host_obj_ids.free();
    host_bounds.free();
    d_query_points.free();
    d_results.free();
}
