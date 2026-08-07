// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RIGID_BODY_BOUND_VOXEL_C_H
#define RS_RIGID_BODY_BOUND_VOXEL_C_H

#include "rise_origin/device/rs_dev_util.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"
#include "rs_coupling_m.h"
#include "rs_rigid_body_bound_voxel_m.h"

namespace rise {

namespace logic {

/**
 * @brief Per rigid-body bound voxel: resolve and cache the voxel and rigid-body handles
 * from their stored sids and the given segment offsets.
 *
 * @param k                                       simulation kernel.
 * @param rigid_body_bound_voxel_segment_offset   offset of this segment in the bound voxel store.
 * @param voxel_segment_offset                    offset of this segment in the voxel store.
 * @param rigid_body_segment_offset               offset of this segment in the rigid body store.
 */
class RSCF_ResolveRigidBodyBoundVoxelDependency
    : public device::RS_ComputeFunction<RSCF_ResolveRigidBodyBoundVoxelDependency> {
   public:
    using BaseType = device::RS_ComputeFunction<RSCF_ResolveRigidBodyBoundVoxelDependency>;
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel *k,
        RSize rigid_body_bound_voxel_segment_offset,
        RSize voxel_segment_offset,
        RSize rigid_body_segment_offset
    ) {
        RHandle rigid_body_bound_voxel = index1D() + rigid_body_bound_voxel_segment_offset;
        auto &ctx = k->ctx;
        RHandle voxel = RBBV_G(voxel_sid) + voxel_segment_offset;
        RHandle rigid_body = RBBV_G(rigid_body_sid) + rigid_body_segment_offset;

        RBBV_S(voxel, voxel);
        RBBV_S(rigid_body, rigid_body);
    }
};

/**
 * @brief Classify one bound voxel into the surface subset.
 * @details A bound voxel is on the surface when the voxel it drives is, which the contact
 * launches need since only a voxel with an air face can touch another body. The bit is fixed
 * at build time for a rigid voxel, so the subset only changes when a structure does.
 *
 * @param k                       simulation kernel.
 * @param rigid_body_bound_voxel  handle of the bound voxel to classify.
 * @param flag_index              index of this bound voxel within the flag array.
 * @param surface_flags           per bound voxel output, 1 when it drives a surface voxel.
 */
class RSCF_GetRigidBodyBoundVoxelSurfaceFlags
    : public device::RS_ComputeFunction<RSCF_GetRigidBodyBoundVoxelSurfaceFlags> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        const Kernel *k,
        RHandle rigid_body_bound_voxel,
        RIndex flag_index,
        device::RS_Vector<RIndex> surface_flags
    ) {
        if (k == nullptr || rigid_body_bound_voxel == RS_NULL_HANDLE) return;
        auto &ctx = k->ctx;
        RHandle voxel = RBBV_G(voxel);
        if (voxel == RS_NULL_HANDLE) {
            surface_flags[flag_index] = 0;
            return;
        }
        RS_VoxelStateUnion s_union{V_G(voxel, state)};
        surface_flags[flag_index] = s_union.voxel_state.is_surface ? 1 : 0;
    }
};

/**
 * @brief Scatter the flagged bound voxel handles into a packed list using the prefix sum.
 * @details A bound voxel whose flag is set writes its handle into out_handles at
 * prefix_sums[flag_index]-1.
 *
 * @param k                       simulation kernel.
 * @param rigid_body_bound_voxel  handle of this bound voxel.
 * @param flag_index              index of this bound voxel within the flag and prefix arrays.
 * @param flags                   per bound voxel 0/1 membership flags for the subset being
 * compacted.
 * @param prefix_sums             inclusive prefix sum of flags giving each flagged bound voxel its
 * output slot.
 * @param out_handles             packed output list of the flagged bound voxel handles.
 */
class RSCF_CompactRigidBodyBoundVoxelHandles
    : public device::RS_ComputeFunction<RSCF_CompactRigidBodyBoundVoxelHandles> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        const Kernel *k,
        RHandle rigid_body_bound_voxel,
        RIndex flag_index,
        device::RS_Vector<RIndex> flags,
        device::RS_Vector<RIndex> prefix_sums,
        device::RS_Vector<RHandle> out_handles
    ) {
        if (k == nullptr || rigid_body_bound_voxel == RS_NULL_HANDLE) return;
        if (flags[flag_index]) {
            out_handles[prefix_sums[flag_index] - 1] = rigid_body_bound_voxel;
        }
    }
};

template <RSE_FloatType FloatComputeAccuracy>
class RS_RigidBodyBoundVoxelCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using Mat3C = RS_Mat3D<FloatC>;

    /**
     * @brief Per rigid-body bound voxel: write the underlying voxel's `position` and
     * `orientation` from the rigid body's current pose and the voxel's body-local offset.
     *
     * @param k                        simulation kernel.
     * @param rigid_body_bound_voxel   handle of the bound voxel to update.
     */
    class RSCF_UpdateVoxelFromRigidBody
        : public device::RS_ComputeFunction<RSCF_UpdateVoxelFromRigidBody> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_UpdateVoxelFromRigidBody>;
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body_bound_voxel) {
            if (k != nullptr && rigid_body_bound_voxel != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                RHandle voxel = RBBV_G(voxel);
                RHandle rigid_body = RBBV_G(rigid_body);

                Quat3C orientation = RB_G(orientation);
                Vec3C origin = RB_G(origin);
                Vec3C position = origin + orientation.rotateVec3D(Vec3C(RBBV_G(voxel_rb_position)));
                V_S(position, position);
                V_S(orientation, orientation);
            }
        }
    };

    /**
     * @brief Per rigid-body bound voxel: produce one soft-step coupling impulse for the
     * body's reduce-by-group input buffer.
     *
     * @param k                          simulation kernel.
     * @param rigid_body_couplings       reduce-by-group input buffer of RS_CouplingCorrection.
     * @param rigid_body_bound_voxel     handle of the bound voxel to contribute.
     * @param rigid_body_coupling_index  slot in `rigid_body_couplings` to write into.
     */
    class RSCF_GetRigidBodyCouplingCorrectionFromVoxel
        : public device::RS_ComputeFunction<RSCF_GetRigidBodyCouplingCorrectionFromVoxel> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_GetRigidBodyCouplingCorrectionFromVoxel>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            RHandle rigid_body_bound_voxel,
            RIndex rigid_body_coupling_index,
            RS_MemoryPtr rigid_body_couplings
        ) {
            using namespace device;
            if (k != nullptr && rigid_body_bound_voxel != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;

                RHandle voxel = RBBV_G(voxel);
                RHandle rigid_body = RBBV_G(rigid_body);

                FloatC passed_time =
                    k->getDeviceTimeSinceLastOperation(RSE_KERNEL_UPDATE_OPERATION_SOFT_BODY);
                Vec3C linear_impulse = Vec3C(V_G(ext_force)) * passed_time;
                Vec3C moment_impulse = Vec3C(V_G(ext_moment)) * passed_time;

                Quat3C orientation = RB_G(orientation);
                Vec3C relative_position = orientation.rotateVec3D(Vec3C(RBBV_G(voxel_rb_position)));
                Vec3C relative_com_position =
                    relative_position + Vec3C(RB_G(origin)) - Vec3C(RB_G(com));
                Vec3C angular_impulse =
                    moment_impulse + relative_com_position.cross(linear_impulse);

                rigid_body_couplings.get<RS_CouplingCorrection>(rigid_body_coupling_index) =
                    RS_CouplingCorrection(linear_impulse, angular_impulse);
            }
        }
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_RIGID_BODY_BOUND_VOXEL_C_H
