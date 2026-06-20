// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MESH_TRIANGLE_C_H
#define RS_MESH_TRIANGLE_C_H

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_bvh.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/mesh/rs_mesh_triangle_m.h"
#include "rise_origin/utils/math/rs_bound.h"
#include "rise_origin/utils/math/rs_triangle.h"

namespace rise {

namespace logic {

class RS_MeshTriangleCompute {
   public:
    /**
     * @brief Fill one floor-BVH leaf per triangle: write {object_metadata = 0,
     * object_handle = triangle} into obj_ids and the triangle AABB into leaf_bounds.
     * Used by the floor BVH builder (floor is static; there is no refit kernel).
     * @param k Kernel instance; unused here but required by RSCF_LaunchDirect.
     * @param mesh_triangles The flattened floor triangles, passed explicitly so the
     * kernel reads the current handle rather than a possibly stale constant-memory copy.
     * @param obj_ids Per-leaf object identifiers to write (sized to the triangle count).
     * @param leaf_bounds Per-leaf triangle AABBs to write (sized to the triangle count).
     */
    class RSCF_FillTriangleBVHLeafData
        : public device::RS_ComputeFunction<RSCF_FillTriangleBVHLeafData> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_FillTriangleBVHLeafData>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            device::RS_Array<RS_MeshTriangleMem, 1> mesh_triangles,
            device::RS_Vector<device::RS_BVHObjectIdentifier> obj_ids,
            device::RS_Vector<RBound3f> leaf_bounds
        ) {
            RIndex leaf_index = index1D();
            if (leaf_index >= mesh_triangles.size()) return;
            RHandle triangle = leaf_index;
            auto &tri = mesh_triangles[triangle];

            device::RS_BVHObjectIdentifier oid;
            oid.object_metadata = 0;
            oid.object_handle = triangle;
            obj_ids[leaf_index] = oid;

            leaf_bounds[leaf_index] =
                RS_Triangle<float>::bound(tri.vertex_0, tri.vertex_1, tri.vertex_2);
        }
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_MESH_TRIANGLE_C_H
