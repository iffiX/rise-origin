// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_INTERSECT_REGION_C_H
#define RS_INTERSECT_REGION_C_H

#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/record/rs_intersect_region_m.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * @brief Resolve the linked rigid body handle and seed the initial world pose of one region,
 *        mirroring RSCF_ResolveRigidBodyLinkedPerspectiveCameraDependency.
 */
class RSCF_ResolveIntersectRegionDependency
    : public device::RS_ComputeFunction<RSCF_ResolveIntersectRegionDependency> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Kernel* k, RHandle intersect_region, RSize rigid_body_segment_offset) {
        auto& ctx = k->ctx;
        RIndex rigid_body_sid = IR_G(rigid_body_sid);
        RHandle rigid_body = rigid_body_sid + rigid_body_segment_offset;
        IR_S(rigid_body, rigid_body);

        // Use double for universal precision
        RQuat3df rb_orientation = RB_G(orientation);
        RQuat3df orientation = rb_orientation * RQuat3df(IR_G(rigid_body_anchor_rb_orientation));
        RVec3df position = RVec3df(RB_G(origin)) +
                           rb_orientation.rotateVec3D(RVec3df(IR_G(rigid_body_anchor_rb_position)));
        IR_S(position, RVec3f(position));
        IR_S(orientation, RQuat3f(orientation));
    }
};

/**
 * @brief Per emitted pair: pack its record-gid-major sort key and its identity index.
 */
class RSCF_GetIntersectRegionPairKeys
    : public device::RS_ComputeFunction<RSCF_GetIntersectRegionPairKeys> {
   public:
    using BaseType = device::RS_ComputeFunction<RSCF_GetIntersectRegionPairKeys>;
    using BaseType::index1D;

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel* k,
        device::RS_Vector<RS_IntersectRegionPair> pairs,
        device::RS_Vector<RS_IntersectRegionSortKey> keys,
        device::RS_Vector<RIndex> indices
    ) {
        if (k == nullptr) return;
        RIndex i = index1D();
        keys[i] =
            RS_IntersectRegionSortKey{pairs[i].key(), pairs[i].closest_relative_offset.length2()};
        indices[i] = i;
    }
};

/**
 * @brief Per sorted pair: flag the first element of each unique-key run.
 */
class RSCF_FlagIntersectRegionPairRunStart
    : public device::RS_ComputeFunction<RSCF_FlagIntersectRegionPairRunStart> {
   public:
    using BaseType = device::RS_ComputeFunction<RSCF_FlagIntersectRegionPairRunStart>;
    using BaseType::index1D;

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel* k,
        device::RS_Vector<RS_IntersectRegionSortKey> sorted_keys,
        device::RS_Vector<RIndex> flags,
        RIndex total
    ) {
        if (k == nullptr) return;
        RIndex i = index1D();
        if (i >= total) return;
        bool is_run_start = (i == 0) || (sorted_keys[i].group_key != sorted_keys[i - 1].group_key);
        flags[i] = is_run_start ? 1 : 0;
    }
};

/**
 * @brief Per sorted pair: write each run-start pair into the compacted unique list.
 */
class RSCF_CompactIntersectRegionPairs
    : public device::RS_ComputeFunction<RSCF_CompactIntersectRegionPairs> {
   public:
    using BaseType = device::RS_ComputeFunction<RSCF_CompactIntersectRegionPairs>;
    using BaseType::index1D;

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel* k,
        device::RS_Vector<RS_IntersectRegionPair> pairs,
        device::RS_Vector<RIndex> sorted_indices,
        device::RS_Vector<RIndex> flags,
        device::RS_Vector<RIndex> prefix,
        device::RS_Vector<RS_IntersectRegionPair> out
    ) {
        if (k == nullptr) return;
        RIndex i = index1D();
        if (flags[i]) {
            out[prefix[i] - 1] = pairs[sorted_indices[i]];
        }
    }
};

/**
 * @brief Numeric per-step intersect-region kernels, parameterized by float compute accuracy.
 */
template <RSE_FloatType FloatComputeAccuracy = RSE_FLOAT_32>
class RS_IntersectRegionCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;

    /**
     * @brief Materialize one BVH leaf object identifier and its world bound from the source mem.
     *        For the linked type the world pose is recomputed from the rigid body's live pose.
     */
    class RSCF_UpdateIntersectRegion
        : public device::RS_ComputeFunction<RSCF_UpdateIntersectRegion> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_UpdateIntersectRegion>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel* k,
            device::RS_Vector<RS_IntersectRegionObjectIdentifier> obj_ids,
            device::RS_Vector<RBound3f> leaf_bounds
        ) {
            if (k == nullptr) return;
            auto& ctx = k->ctx;
            RIndex intersect_region = index1D();
            if (intersect_region >= obj_ids.size()) return;

            RS_IntersectRegionObjectIdentifier id;
            id.record_gid = IR_G(gid);
            id.type = IR_G(type);
            id.half_size = IR_G(half_size);
            id.radius = IR_G(radius);

            if (id.type == RSE_INTERSECT_REGION_CUBOID_RIGID_BODY_LINKED ||
                id.type == RSE_INTERSECT_REGION_SPHERE_RIGID_BODY_LINKED) {
                RHandle rigid_body = IR_G(rigid_body);
                Quat3C rb_orientation = RB_G(orientation);
                Quat3C orientation =
                    rb_orientation * Quat3C(IR_G(rigid_body_anchor_rb_orientation));
                Vec3C position =
                    Vec3C(RB_G(origin)) +
                    rb_orientation.rotateVec3D(Vec3C(IR_G(rigid_body_anchor_rb_position)));
                id.position = RVec3f(position);
                id.orientation = RQuat3f(orientation);
            } else {
                id.position = IR_G(position);
                id.orientation = IR_G(orientation);
            }

            obj_ids[intersect_region] = id;
            leaf_bounds[intersect_region] = id.worldBound();
        }
    };

    /**
     * @brief Per voxel: count how many regions currently contain its world center.
     */
    class RSCF_CountIntersectRegionMembership
        : public device::RS_ComputeFunction<RSCF_CountIntersectRegionMembership> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_CountIntersectRegionMembership>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel* k, RHandle voxel, RIndex dense_index, RS_MemoryPtr counts) {
            if (k == nullptr) return;
            auto& ctx = k->ctx;
            counts.template get<RSize>(dense_index) = 0;
            if (voxel == RS_NULL_HANDLE) return;
            if (ctx.intersect_region_bvh.object_identifiers.size() == 0) return;

            Vec3C center = Vec3C(V_G(voxel, position));

            RSize count = 0;
            auto accessor = ctx.intersect_region_bvh.root();
            static constexpr RSize STACK_SIZE = 32;
            RIndex stack[STACK_SIZE];
            RSize stack_top = 0;
            RIndex root_node = accessor.current();
            if (root_node != RS_NULL_INDEX) stack[stack_top++] = root_node;

            while (stack_top > 0) {
                RIndex node = stack[--stack_top];
                accessor.moveTo(node);
                if (accessor.isLeaf()) {
                    auto* obj = accessor.object();
                    if (obj != nullptr && obj->contains(RVec3f(center))) count++;
                } else {
                    auto internal_node = accessor.internalNode();
                    if (internal_node.left != RS_NULL_INDEX && stack_top < STACK_SIZE &&
                        internal_node.left_bound.isContaining(center)) {
                        stack[stack_top++] = internal_node.left;
                    }
                    if (internal_node.right != RS_NULL_INDEX && stack_top < STACK_SIZE &&
                        internal_node.right_bound.isContaining(center)) {
                        stack[stack_top++] = internal_node.right;
                    }
                }
            }
            counts.template get<RSize>(dense_index) = count;
        }
    };

    /**
     * @brief Per voxel: emit one (record_gid, structure_gid) pair for each region containing its
     *        world center, packed at the voxel's exclusive prefix-sum offset.
     */
    class RSCF_WriteIntersectRegionMembership
        : public device::RS_ComputeFunction<RSCF_WriteIntersectRegionMembership> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_WriteIntersectRegionMembership>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel* k,
            RHandle voxel,
            RIndex dense_index,
            RS_MemoryPtr offsets,
            device::RS_Vector<RS_IntersectRegionPair> pairs
        ) {
            if (k == nullptr) return;
            auto& ctx = k->ctx;
            if (voxel == RS_NULL_HANDLE) return;
            if (ctx.intersect_region_bvh.object_identifiers.size() == 0) return;

            Vec3C center = Vec3C(V_G(voxel, position));
            RIndex structure_gid = V_G(voxel, structure_gid);
            RSize out = offsets.template get<RSize>(dense_index);

            auto accessor = ctx.intersect_region_bvh.root();
            static constexpr RSize STACK_SIZE = 32;
            RIndex stack[STACK_SIZE];
            RSize stack_top = 0;
            RIndex root_node = accessor.current();
            if (root_node != RS_NULL_INDEX) stack[stack_top++] = root_node;

            while (stack_top > 0) {
                RIndex node = stack[--stack_top];
                accessor.moveTo(node);
                if (accessor.isLeaf()) {
                    auto* obj = accessor.object();
                    if (obj != nullptr && obj->contains(RVec3f(center))) {
                        RS_IntersectRegionPair pair;
                        pair.record_gid = obj->record_gid;
                        pair.structure_gid = structure_gid;
                        pair.closest_relative_offset = RVec3f(center) - obj->position;
                        pairs[out] = pair;
                        out++;
                    }
                } else {
                    auto internal_node = accessor.internalNode();
                    if (internal_node.left != RS_NULL_INDEX && stack_top < STACK_SIZE &&
                        internal_node.left_bound.isContaining(center)) {
                        stack[stack_top++] = internal_node.left;
                    }
                    if (internal_node.right != RS_NULL_INDEX && stack_top < STACK_SIZE &&
                        internal_node.right_bound.isContaining(center)) {
                        stack[stack_top++] = internal_node.right;
                    }
                }
            }
        }
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_INTERSECT_REGION_C_H
