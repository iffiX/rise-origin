// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RIGID_BODY_SURFACE_VOXEL_C_H
#define RS_RIGID_BODY_SURFACE_VOXEL_C_H

#include "rise_origin/device/rs_dev_util.h"
#include "rs_coupling_m.h"
#include "rs_rigid_body_surface_voxel_m.h"

namespace rise {

namespace logic {

/**
 * @brief Per rigid-body surface voxel: resolve and cache the voxel and rigid-body handles
 * from their stored sids and the given segment offsets.
 *
 * @param k                                          simulation kernel.
 * @param rigid_body_surface_voxel_segment_offset    offset of this segment in the surface voxel
 * store.
 * @param voxel_segment_offset                       offset of this segment in the voxel store.
 * @param rigid_body_segment_offset                  offset of this segment in the rigid body store.
 */
class RSCF_ResolveRigidBodySurfaceVoxelDependency
    : public device::RS_ComputeFunction<RSCF_ResolveRigidBodySurfaceVoxelDependency> {
   public:
    using BaseType = device::RS_ComputeFunction<RSCF_ResolveRigidBodySurfaceVoxelDependency>;
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel k,
        RSize rigid_body_surface_voxel_segment_offset,
        RSize voxel_segment_offset,
        RSize rigid_body_segment_offset
    ) {
        RHandle rigid_body_surface_voxel = index1D() + rigid_body_surface_voxel_segment_offset;
        auto &ctx = k.ctx;
        RHandle voxel = RBSV_G(voxel_sid) + voxel_segment_offset;
        RHandle rigid_body = RBSV_G(rigid_body_sid) + rigid_body_segment_offset;

        RBSV_S(voxel, voxel);
        RBSV_S(rigid_body, rigid_body);
    }
};

template <RSE_FloatType FloatComputeAccuracy>
class RS_RigidBodySurfaceVoxelCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using Mat3C = RS_Mat3D<FloatC>;

    /**
     * @brief Per rigid-body surface voxel: write the underlying voxel's `position` and
     * `orientation` from the rigid body's current pose and the voxel's body-local offset.
     *
     * @param k                          simulation kernel.
     * @param rigid_body_surface_voxel   handle of the surface voxel to update.
     */
    class RSCF_UpdateVoxelFromRigidBody
        : public device::RS_ComputeFunction<RSCF_UpdateVoxelFromRigidBody> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_UpdateVoxelFromRigidBody>;
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body_surface_voxel) {
            if (k != nullptr && rigid_body_surface_voxel != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                RHandle voxel = RBSV_G(voxel);
                RHandle rigid_body = RBSV_G(rigid_body);

                Quat3C orientation = RB_G(orientation);
                Vec3C origin = RB_G(origin);
                Vec3C position = origin + orientation.rotateVec3D(Vec3C(RBSV_G(voxel_rb_position)));
                V_S(position, position);
                V_S(orientation, orientation);
            }
        }
    };

    /**
     * @brief Per rigid-body surface voxel: produce one soft-step coupling impulse for the
     * body's reduce-by-group input buffer.
     *
     * @param k                          simulation kernel.
     * @param rigid_body_couplings       reduce-by-group input buffer of RS_CouplingCorrection.
     * @param rigid_body_surface_voxel   handle of the surface voxel to contribute.
     * @param rigid_body_coupling_index  slot in `rigid_body_couplings` to write into.
     */
    class RSCF_GetRigidBodyCouplingCorrectionFromVoxel
        : public device::RS_ComputeFunction<RSCF_GetRigidBodyCouplingCorrectionFromVoxel> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_GetRigidBodyCouplingCorrectionFromVoxel>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            RS_MemoryPtr rigid_body_couplings,
            RHandle rigid_body_surface_voxel,
            RIndex rigid_body_coupling_index
        ) {
            using namespace device;
            if (k != nullptr && rigid_body_surface_voxel != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;

                RHandle voxel = RBSV_G(voxel);
                RHandle rigid_body = RBSV_G(rigid_body);

                FloatC passed_time =
                    k->getDeviceTimeSinceLastOperation(RSE_KERNEL_UPDATE_OPERATION_SOFT_BODY);
                Vec3C linear_impulse = Vec3C(V_G(ext_force)) * passed_time;
                Vec3C moment_impulse = Vec3C(V_G(ext_moment)) * passed_time;

                Quat3C orientation = RB_G(orientation);
                Vec3C relative_position = orientation.rotateVec3D(Vec3C(RBSV_G(voxel_rb_position)));
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

#endif  // RS_RIGID_BODY_SURFACE_VOXEL_C_H
