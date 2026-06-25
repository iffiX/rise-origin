// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_TRANSFORM_C_H
#define RS_TRANSFORM_C_H

#include "rise_origin/logic/rigid/rs_rigid_body_c.h"
#include "rise_origin/logic/signal/rs_transform_m.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

template <RSE_FloatType FloatComputeAccuracy>
class RS_TransformCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using FI = RS_FloatIntrinsics<FloatC>;

    /**
     * @brief Compute function to reset transform signals to identity.
     *
     * Resets the device translation, rotation, and pivot of every transform
     * memory entry to identity after the transform has been applied, so the
     * next read shows zero translation and a unit quaternion and the transform
     * is never applied twice.
     */
    class RSCF_ResetTransformSignal
        : public device::RS_ComputeFunction<RSCF_ResetTransformSignal> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ResetTransformSignal>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k) {
            if (k != nullptr) {
                auto &ctx = k->ctx;
                RIndex i = BaseType::index1D();

                if (i < ctx.transform_mem.size()) {
                    RHandle signal_handle = ctx.transform_mem[i].transform_signal;
                    ctx.transform_signals[signal_handle].translation_signals.translation[0] =
                        RVec3rf::zero();
                    ctx.transform_signals[signal_handle].rotation_signals.rotation[0] =
                        RQuat3rf::unit();
                    ctx.transform_signals[signal_handle].rotation_signals.pivot[0] =
                        RVec3rf::zero();
                }
            }
        }
    };

    /**
     * @brief Compute function to apply transform signals to voxels.
     *
     * Iterates through all transform memory entries and re-poses every matching
     * voxel of the structure regardless of fixed or rigid state. Rigid voxels
     * are moved too so a static rigid body, whose voxel poses are not otherwise
     * recomputed each step, follows the same rigid transform as its body.
     */
    class RSCF_ApplyVoxelTransformSignal
        : public device::RS_ComputeFunction<RSCF_ApplyVoxelTransformSignal> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ApplyVoxelTransformSignal>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle voxel) {
            if (k != nullptr && voxel != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;

                RIndex v_structure_gid = V_G(structure_gid);
                RS_VoxelStateUnion v_state{V_G(state)};

                // Iterate through all transform memories
                RSize transform_count = ctx.transform_mem.size();
                for (RIndex i = 0; i < transform_count; i++) {
                    auto &transform_mem = ctx.transform_mem[i];

                    if (transform_mem.structure_gid != v_structure_gid) continue;

                    // Read transform signal (one translation, rotation, pivot entry).
                    RHandle signal_handle = transform_mem.transform_signal;
                    RVec3rf translation_raw =
                        ctx.transform_signals[signal_handle].translation_signals.translation[0];
                    RQuat3rf rotation_raw =
                        ctx.transform_signals[signal_handle].rotation_signals.rotation[0];
                    RVec3rf pivot_raw =
                        ctx.transform_signals[signal_handle].rotation_signals.pivot[0];

                    // Identity transform applies nothing, skip.
                    if (translation_raw == RVec3rf::zero() && rotation_raw == RQuat3rf::unit())
                        continue;

                    Vec3C translation = Vec3C(translation_raw);
                    Quat3C rotation = Quat3C(rotation_raw);
                    Vec3C pivot = Vec3C(pivot_raw);

                    // Re-pose: new = rotation * (pos - pivot) + pivot + translation.
                    Vec3C position = Vec3C(V_G(position));
                    position = rotation.rotateVec3D(position - pivot) + pivot + translation;
                    V_S(position, position);

                    Quat3C orientation = Quat3C(V_G(orientation));
                    orientation = rotation * orientation;
                    V_S(orientation, orientation);

                    // A rigid voxel's momenta live on its rigid body, so only turn the momenta of
                    // soft voxels. Translation does not affect momenta; rotation turns them.
                    if (!v_state.voxel_state.is_rigid) {
                        Vec3C linear_momentum = Vec3C(V_G(linear_momentum));
                        V_S(linear_momentum, rotation.rotateVec3D(linear_momentum));

                        Vec3C angular_momentum = Vec3C(V_G(angular_momentum));
                        V_S(angular_momentum, rotation.rotateVec3D(angular_momentum));
                    }
                }
            }
        }
    };

    /**
     * @brief Compute function to apply transform signals to rigid bodies.
     *
     * Iterates through all transform memory entries and re-poses matching rigid
     * bodies of the structure regardless of fixed state.
     */
    class RSCF_ApplyRigidBodyTransformSignal
        : public device::RS_ComputeFunction<RSCF_ApplyRigidBodyTransformSignal> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ApplyRigidBodyTransformSignal>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body) {
            if (k != nullptr && rigid_body != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;

                RIndex rb_structure_gid = RB_G(structure_gid);

                // Iterate through all transform memories
                RSize transform_count = ctx.transform_mem.size();
                for (RIndex i = 0; i < transform_count; i++) {
                    auto &transform_mem = ctx.transform_mem[i];

                    if (transform_mem.structure_gid != rb_structure_gid) continue;

                    // Read transform signal (one translation, rotation, pivot entry).
                    RHandle signal_handle = transform_mem.transform_signal;
                    RVec3rf translation_raw =
                        ctx.transform_signals[signal_handle].translation_signals.translation[0];
                    RQuat3rf rotation_raw =
                        ctx.transform_signals[signal_handle].rotation_signals.rotation[0];
                    RVec3rf pivot_raw =
                        ctx.transform_signals[signal_handle].rotation_signals.pivot[0];

                    // Identity transform applies nothing, skip.
                    if (translation_raw == RVec3rf::zero() && rotation_raw == RQuat3rf::unit())
                        continue;

                    Vec3C translation = Vec3C(translation_raw);
                    Quat3C rotation = Quat3C(rotation_raw);
                    Vec3C pivot = Vec3C(pivot_raw);

                    // Re-pose: new = rotation * (pos - pivot) + pivot + translation.
                    Vec3C origin = Vec3C(RB_G(origin));
                    origin = rotation.rotateVec3D(origin - pivot) + pivot + translation;
                    RB_S(origin, origin);

                    Vec3C com = Vec3C(RB_G(com));
                    com = rotation.rotateVec3D(com - pivot) + pivot + translation;
                    RB_S(com, com);

                    Quat3C orientation = Quat3C(RB_G(orientation));
                    orientation = rotation * orientation;
                    RB_S(orientation, orientation);

                    // Translation does not affect momenta; rotation turns them.
                    Vec3C linear_momentum = Vec3C(RB_G(linear_momentum));
                    RB_S(linear_momentum, rotation.rotateVec3D(linear_momentum));

                    Vec3C angular_momentum = Vec3C(RB_G(angular_momentum));
                    RB_S(angular_momentum, rotation.rotateVec3D(angular_momentum));
                }
            }
        }
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_TRANSFORM_C_H
