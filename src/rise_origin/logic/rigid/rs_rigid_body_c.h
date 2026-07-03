// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RIGID_BODY_C_H
#define RS_RIGID_BODY_C_H

#include "rise_origin/utils/math/rs_float.h"
#include "rs_coupling_m.h"
#include "rs_rigid_body_m.h"

namespace rise {

namespace logic {

/**
 * @brief Per rigid body: write the new mass, inverse mass, center-of-mass offset and
 * local inverse inertia from the input arrays, then derive the world-space `com` and
 * `moment_inertia_inverse` from the body's current origin and orientation.
 *
 * @param k                            simulation kernel.
 * @param rigid_body_segment_offset    handle offset added to the 1D index to reach this
 *                                     segment's bodies.
 * @param mass                         per-body mass values, indexed by the 1D index.
 * @param mass_inverse                 per-body inverse mass values.
 * @param com_rb_position              per-body center-of-mass offset in body-local frame.
 * @param local_moment_inertia_inverse per-body inverse inertia tensor in body-local frame.
 */
class RSCF_UpdateRigidBodyProperties
    : public device::RS_ComputeFunction<RSCF_UpdateRigidBodyProperties> {
   public:
    using BaseType = device::RS_ComputeFunction<RSCF_UpdateRigidBodyProperties>;

    template <typename Kernel, typename FloatMemArray, typename VecMemArray, typename MatMemArray>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel *k,
        RSize rigid_body_segment_offset,
        FloatMemArray mass,
        FloatMemArray mass_inverse,
        VecMemArray com_rb_position,
        MatMemArray local_moment_inertia_inverse
    ) {
        RIndex idx = index1D();
        RHandle rigid_body = idx + rigid_body_segment_offset;
        auto &ctx = k->ctx;

        RFloat new_mass = mass[idx];
        RB_S(mass, new_mass);
        RB_S(mass_inverse, mass_inverse[idx]);

        RVec3rf new_com_rb_position = com_rb_position[idx];
        RMat3rf new_local_moment_inertia_inverse = local_moment_inertia_inverse[idx];

        RQuat3rf orientation = RB_G(orientation);
        RMat3rf rotation_matrix = orientation.toRotationMatrix();
        RVec3rf origin = RB_G(origin);

        RVec3rf new_com = origin + orientation.rotateVec3D(new_com_rb_position);
        RMat3rf new_moment_inertia_inverse =
            rotation_matrix * new_local_moment_inertia_inverse * rotation_matrix.transpose();

        RB_S(com_rb_position, new_com_rb_position);
        RB_S(com, new_com);
        RB_S(local_moment_inertia_inverse, new_local_moment_inertia_inverse);
        RB_S(moment_inertia_inverse, new_moment_inertia_inverse);
    };
};

/**
 * @brief Set or clear the effective immobility of every rigid body in a structure's segment.
 * @details Freezing forces is_static true so the body stops integrating; unfreezing restores
 * is_static to the body's permanent is_fixed, so a body fixed by its material stays immobile.
 *
 * @param k                         simulation kernel.
 * @param rigid_body_segment_offset handle offset added to the 1D index to reach this segment's
 *                                  bodies.
 * @param freeze                    true to freeze the bodies, false to restore them.
 */
class RSCF_SetRigidBodyStaticOverSegment
    : public device::RS_ComputeFunction<RSCF_SetRigidBodyStaticOverSegment> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Kernel *k, RSize rigid_body_segment_offset, bool freeze) {
        auto &ctx = k->ctx;
        RHandle rigid_body = index1D() + rigid_body_segment_offset;
        RS_RigidBodyStateUnion state_union{RB_G(state)};
        state_union.rigid_body_state.is_static = freeze ? true
                                                        : state_union.rigid_body_state.is_fixed;
        RB_S(state, state_union.rigid_body_state_packed);
    }
};

/**
 * @brief Classify one rigid body into the dynamic subset.
 * @details A body is dynamic when it is not static, so the per-step dynamics launches iterate only
 * the moving bodies while the full set still serves the transform and contact anchors. Static and
 * frozen bodies are simply excluded, exactly as the dynamics kernels already early-return on them.
 *
 * @param k             simulation kernel.
 * @param rigid_body    handle of the body to classify.
 * @param flag_index    index of this body within the flag array.
 * @param dynamic_flags per-body output, 1 when mobile.
 */
class RSCF_GetRigidBodyMobilityFlags
    : public device::RS_ComputeFunction<RSCF_GetRigidBodyMobilityFlags> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        const Kernel *k,
        RHandle rigid_body,
        RIndex flag_index,
        device::RS_Vector<RIndex> dynamic_flags
    ) {
        if (k == nullptr || rigid_body == RS_NULL_HANDLE) return;
        auto &ctx = k->ctx;
        bool is_static = RS_RigidBodyStateUnion{RB_G(state)}.rigid_body_state.is_static;
        dynamic_flags[flag_index] = is_static ? 0 : 1;
    }
};

/**
 * @brief Scatter the flagged rigid body handles into a packed list using the prefix sum.
 * @details A body whose flag is set writes its handle into out_handles at
 * prefix_sums[flag_index]-1.
 *
 * @param k           simulation kernel.
 * @param rigid_body  handle of this body.
 * @param flag_index  index of this body within the flag and prefix arrays.
 * @param flags       per-body 0/1 membership flags for the subset being compacted.
 * @param prefix_sums inclusive prefix sum of flags giving each flagged body its output slot.
 * @param out_handles packed output list of the flagged body handles.
 */
class RSCF_CompactRigidBodyHandles
    : public device::RS_ComputeFunction<RSCF_CompactRigidBodyHandles> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        const Kernel *k,
        RHandle rigid_body,
        RIndex flag_index,
        device::RS_Vector<RIndex> flags,
        device::RS_Vector<RIndex> prefix_sums,
        device::RS_Vector<RHandle> out_handles
    ) {
        if (k == nullptr || rigid_body == RS_NULL_HANDLE) return;
        if (flags[flag_index]) {
            out_handles[prefix_sums[flag_index] - 1] = rigid_body;
        }
    }
};

template <RSE_FloatType FloatComputeAccuracy>
class RS_RigidBodyCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using Mat3C = RS_Mat3D<FloatC>;

    /**
     * @brief Per rigid body: add the gravity impulse `m * g * passed_time` to linear momentum.
     *
     * @param k          simulation kernel.
     * @param rigid_body handle of the body to push with gravity.
     */
    class RSCF_ApplyGravity : public device::RS_ComputeFunction<RSCF_ApplyGravity> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ApplyGravity>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body) {
            if (k != nullptr && rigid_body != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                if (RS_RigidBodyStateUnion{RB_G(state)}.rigid_body_state.is_static) return;

                FloatC passed_time = k->getDeviceTimeSinceLastOperation(
                    RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_CONSTRAINT_AND_CONTACT
                );
                FloatC mass = RB_G(mass);
                Vec3C lm = Vec3C(RB_G(linear_momentum));
                lm.z += k->world_grav_acc * mass * passed_time;
                RB_S(linear_momentum, lm);
            }
        };
    };

    /**
     * @brief Per rigid body: scale linear and angular momentum by
     * `(1 - damping_z * passed_time)` for each axis.
     *
     * @param k          simulation kernel.
     * @param rigid_body handle of the body to damp.
     */
    class RSCF_ApplyDamping : public device::RS_ComputeFunction<RSCF_ApplyDamping> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ApplyDamping>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body) {
            if (k != nullptr && rigid_body != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                if (RS_RigidBodyStateUnion{RB_G(state)}.rigid_body_state.is_static) return;

                FloatC passed_time = k->getDeviceTimeSinceLastOperation(
                    RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_CONSTRAINT_AND_CONTACT
                );
                Vec3C linear_momentum = Vec3C(RB_G(linear_momentum));
                Vec3C angular_momentum = Vec3C(RB_G(angular_momentum));

                if (k->rigidbody_linear_damping_z != 0)
                    linear_momentum *= (FloatC(1) - k->rigidbody_linear_damping_z * passed_time);
                if (k->rigidbody_angular_damping_z != 0)
                    angular_momentum *= (FloatC(1) - k->rigidbody_angular_damping_z * passed_time);
                RB_S(linear_momentum, linear_momentum);
                RB_S(angular_momentum, angular_momentum);
            }
        };
    };

    /**
     * @brief Per rigid body: zero the `cumulative_coupling_work` accumulator.
     *
     * @param k          simulation kernel.
     * @param rigid_body handle of the body to reset.
     */
    class RSCF_ResetRigidBodyCouplingWork
        : public device::RS_ComputeFunction<RSCF_ResetRigidBodyCouplingWork> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ResetRigidBodyCouplingWork>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body) {
            if (k != nullptr && rigid_body != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                if (RS_RigidBodyStateUnion{RB_G(state)}.rigid_body_state.is_static) return;
                RB_S(cumulative_coupling_work, FloatC(0));
            }
        };
    };

    /**
     * @brief Per rigid body: add the soft coupling linear and angular impulse to body
     * momentum, accumulate the linear coupling work into `cumulative_coupling_work`, then
     * integrate `com` and `orientation` by `passed_time` from the resulting momentum.
     *
     * @param k                          simulation kernel.
     * @param rigid_body_couplings       reduce-by-group output buffer of summed
     *                                   RS_CouplingCorrection.
     * @param rigid_body                 handle of the body to apply the correction to.
     * @param rigid_body_coupling_index  slot in `rigid_body_couplings` holding this body's
     *                                   summed correction.
     */
    class RSCF_ApplyCouplingCorrectionAndUpdateRigidBody
        : public device::RS_ComputeFunction<RSCF_ApplyCouplingCorrectionAndUpdateRigidBody> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ApplyCouplingCorrectionAndUpdateRigidBody>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            RHandle rigid_body,
            RIndex rigid_body_coupling_index,
            RS_MemoryPtr rigid_body_couplings
        ) {
            using namespace device;
            if (k != nullptr && rigid_body != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                if (RS_RigidBodyStateUnion{RB_G(state)}.rigid_body_state.is_static) return;

                FloatC passed_time = k->getDeviceTimeSinceLastOperation(
                    RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_INTEGRATE
                );

                RS_CouplingCorrection coupling =
                    rigid_body_couplings.get<RS_CouplingCorrection>(rigid_body_coupling_index);

                FloatC mass_inverse = RB_G(mass_inverse);

                Vec3C linear_momentum_old = Vec3C(RB_G(linear_momentum));
                Vec3C linear_impulse = Vec3C(coupling.linear_impulse);
                Vec3C linear_momentum = linear_momentum_old + linear_impulse;
                Vec3C angular_momentum =
                    Vec3C(RB_G(angular_momentum)) + Vec3C(coupling.angular_impulse);
                RB_S(linear_momentum, linear_momentum);
                RB_S(angular_momentum, angular_momentum);

                // Linear-only approximation of work this step (j . v_pre).
                FloatC coupling_work_step = linear_impulse.dot(linear_momentum_old) * mass_inverse;
                RB_S(
                    cumulative_coupling_work,
                    FloatC(RB_G(cumulative_coupling_work)) + coupling_work_step
                );

                // Update position
                Vec3C translate = linear_momentum * passed_time * mass_inverse;
                Vec3C com = Vec3C(RB_G(com)) + translate;
                RB_S(com, com);

                // Update orientation
                Quat3C orientation = RB_G(orientation);
                Mat3C rotation_matrix = orientation.toRotationMatrix();
                Mat3C moment_inertia_inverse = rotation_matrix *
                                               Mat3C(RB_G(local_moment_inertia_inverse)) *
                                               rotation_matrix.transpose();
                RB_S(moment_inertia_inverse, moment_inertia_inverse);

                Quat3C rotation;
                rotation.fromRotationVector(
                    moment_inertia_inverse * (angular_momentum * passed_time)
                );
                orientation = (rotation * orientation).normalized();
                RB_S(orientation, orientation);

                // Update origin
                Vec3C com_rb_position = RB_G(com_rb_position);
                Vec3C origin = com - orientation.rotateVec3D(com_rb_position);
                RB_S(origin, origin);
            }
        };
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_RIGID_BODY_C_H