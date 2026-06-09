// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONSTRAINT_C_H
#define RS_CONSTRAINT_C_H

namespace rise {

namespace logic {

/**
 * @brief Resolves each joint's body and signal references.
 */
class RSCF_ResolveConstraintJointDependency
    : public device::RS_ComputeFunction<RSCF_ResolveConstraintJointDependency> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel k,
        RSize constraint_joint_segment_offset,
        RSize rigid_body_segment_offset,
        RSize structure_hinge_signal_segment_offset,
        RSize structure_prismatic_signal_segment_offset
    ) {
        RHandle constraint_joint = index1D() + constraint_joint_segment_offset;
        auto &ctx = k.ctx;
        RHandle rigid_body_a = CJ_G(rigid_body_a_sid) + rigid_body_segment_offset;
        RHandle rigid_body_b = CJ_G(rigid_body_b_sid) + rigid_body_segment_offset;

        RIndex hinge_signal_sid = CJ_G(hinge_signal_sid);
        RHandle hinge_signal = hinge_signal_sid == RS_NULL_INDEX
                                   ? RS_NULL_HANDLE
                                   : hinge_signal_sid + structure_hinge_signal_segment_offset;

        RIndex prismatic_signal_sid = CJ_G(prismatic_signal_sid);
        RHandle prismatic_signal =
            prismatic_signal_sid == RS_NULL_INDEX
                ? RS_NULL_HANDLE
                : prismatic_signal_sid + structure_prismatic_signal_segment_offset;

        CJ_S(rigid_body_a, rigid_body_a);
        CJ_S(rigid_body_b, rigid_body_b);
        CJ_S(hinge_signal, hinge_signal);
        CJ_S(prismatic_signal, prismatic_signal);
    }
};

/**
 * @brief Sequential-impulse joint constraint solver for articulated rigid bodies.
 *
 * @details Resolves ball-and-socket, fixed, hinge, and prismatic joints, including their limits
 * and motors, in velocity space with Baumgarte stabilization. The work is split into two kernels:
 *
 *     RSCF_UpdateConstraintJoint:  per joint, build J, K_inv, and C once per update interval
 *     RSCF_PGSUpdateJoint:         per structure, block-shared PGS sweep applying delta_lambda
 *
 * Each sub-constraint contributes a 3-row block of the Jacobian; the projected Gauss-Seidel sweep
 * is graph-colored so a color's threads touch disjoint bodies and can write shared memory without
 * conflict.
 *
 * Symbols, shared by the kernels below and matching the rigid contact solver in RS_ContactCompute:
 *   a, b          the two bodies joined by the constraint
 *   v_a, v_b      linear velocity of a / b                       m/s
 *   w_a, w_b      angular velocity of a / b                      rad/s
 *   mass_inv_a/b  inverse mass of a / b                          kg^-1
 *   I_inv_a/b     inverse world-frame inertia of a / b           kg^-1 m^-2
 *   J             constraint Jacobian, rows J_{a,b}_{lin,ang}_0..2
 *   K_inv         inverse effective mass, the inverse of J M^-1 J^T
 *   C             constraint_value, the positional or orientational error
 *   rel_vel       constraint-space relative velocity J v
 *   lambda        constraint impulse, as accumulated_lambda and delta_lambda
 *   passed_time   rigid body update step, from the kernel           s
 */
template <RSE_FloatType FloatComputeAccuracy>
class RS_ConstraintCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using Mat3C = RS_Mat3D<FloatC>;
    using RS_FI = RS_FloatIntrinsics<FloatC>;

    /**
     * @brief Per joint: build the Jacobian, inverse effective mass, and constraint value.
     *
     * @details Runs once per constraint-update interval ahead of the PGS sweep to refresh the data
     * that stays constant across iterations: the 3x3 inverse effective mass K_inv, the 3x12
     * Jacobian rows J, and the constraint value C that both activates the constraint and feeds the
     * Baumgarte bias. Dispatches to the per-type builder selected by joint type. See the class
     * docstring for the shared symbol table.
     *
     * @param k                simulation kernel.
     * @param constraint_joint handle of the joint to refresh.
     */
    class RSCF_UpdateConstraintJoint
        : public device::RS_ComputeFunction<RSCF_UpdateConstraintJoint> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle constraint_joint) {
            if (k != nullptr && constraint_joint != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;

                RSE_ConstraintJointType type = CJ_G(type);
                RHandle rigid_body_a = CJ_G(rigid_body_a);
                RHandle rigid_body_b = CJ_G(rigid_body_b);

                RBigFloat passed_time = k->getDeviceTimeSinceLastOperation(
                    RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_CONSTRAINT_AND_CONTACT
                );

                _updateCompliance(
                    k, constraint_joint, type, rigid_body_a, rigid_body_b, passed_time
                );

                if (type & (RSE_BALL_AND_SOCKET_JOINT_TRANSLATION | RSE_FIXED_JOINT_TRANSLATION |
                            RSE_HINGE_JOINT_TRANSLATION)) {
                    _updateTranslationConstraint(k, constraint_joint, rigid_body_a, rigid_body_b);
                } else if (type & RSE_FIXED_JOINT_ROTATION) {
                    _updateFixedRotationConstraint(k, constraint_joint, rigid_body_a, rigid_body_b);
                } else if (type & RSE_HINGE_JOINT_ROTATION) {
                    _updateHingeRotationConstraint(k, constraint_joint, rigid_body_a, rigid_body_b);
                } else if (type & RSE_HINGE_JOINT_LIMIT) {
                    _updateHingeLimitConstraint(k, constraint_joint, rigid_body_a, rigid_body_b);
                } else if (type & RSE_HINGE_JOINT_MOTOR) {
                    _updateHingeMotorConstraint(
                        k,
                        constraint_joint,
                        rigid_body_a,
                        rigid_body_b,
                        passed_time,
                        ctx.structure_hinge_signals[CJ_G(hinge_signal)]
                    );
                } else if (type & RSE_PRISMATIC_JOINT_PERPENDICULAR_TRANSLATION) {
                    _updatePrismaticTranslationConstraint(
                        k, constraint_joint, rigid_body_a, rigid_body_b
                    );
                } else if (type & RSE_PRISMATIC_JOINT_ROTATION) {
                    _updateFixedRotationConstraint(k, constraint_joint, rigid_body_a, rigid_body_b);
                } else if (type & RSE_PRISMATIC_JOINT_LIMIT) {
                    _updatePrismaticLimitConstraint(
                        k, constraint_joint, rigid_body_a, rigid_body_b
                    );
                } else if (type & RSE_PRISMATIC_JOINT_MOTOR) {
                    _updatePrismaticMotorConstraint(
                        k,
                        constraint_joint,
                        rigid_body_a,
                        rigid_body_b,
                        passed_time,
                        ctx.structure_prismatic_signals[CJ_G(prismatic_signal)]
                    );
                }
            }
        }

       private:
        /**
         * @brief Adapt the joint's PGS velocity-correction scale (compliance) from the
         * cumulative coupling-work the soft delivered to the two adjacent bodies over the
         * slow rigid interval. Power = work / slow_dt is compared against the watt
         * threshold; compliance is scaled by (1 - step) when above, (1 + step) when below,
         * and clamped to [min_threshold, max_threshold]. Motors and null-body joints are
         * skipped.
         */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateCompliance(
            Kernel *k,
            RHandle constraint_joint,
            RSE_ConstraintJointType type,
            RHandle rigid_body_a,
            RHandle rigid_body_b,
            RBigFloat passed_time
        ) {
            auto &ctx = k->ctx;
            bool is_motor = (type & (RSE_HINGE_JOINT_MOTOR | RSE_PRISMATIC_JOINT_MOTOR)) != 0;
            if (is_motor || rigid_body_a == RS_NULL_HANDLE || rigid_body_b == RS_NULL_HANDLE)
                return;

            FloatC e_joint = FloatC(RB_G(rigid_body_a, cumulative_coupling_work)) +
                             FloatC(RB_G(rigid_body_b, cumulative_coupling_work));
            FloatC slow_dt = FloatC(passed_time);
            FloatC p_joint = (slow_dt > FloatC(0)) ? (e_joint / slow_dt) : FloatC(0);

            FloatC compliance = FloatC(CJ_G(compliance));
            FloatC step = k->rigidbody_constraint_compliance_auto_adjust_step;
            if (p_joint > k->rigidbody_constraint_compliance_auto_adjust_watt_threshold) {
                compliance *= (FloatC(1) - step);
            } else {
                compliance *= (FloatC(1) + step);
            }
            FloatC lo = k->rigidbody_constraint_compliance_min_threshold;
            FloatC hi = k->rigidbody_constraint_compliance_max_threshold;
            if (compliance < lo) compliance = lo;
            if (compliance > hi) compliance = hi;
            CJ_S(compliance, compliance);
        }

        /**
         * @brief Builds the 3-DOF point-to-point constraint pinning the two anchor points together.
         */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateTranslationConstraint(
            Kernel *k,
            RHandle constraint_joint,
            RHandle rigid_body_a,
            RHandle rigid_body_b
        ) {
            auto &ctx = k->ctx;

            Quat3C a_orientation = RB_G(rigid_body_a, orientation);
            Vec3C a_rb_position = CJ_G(rigid_body_a_anchor_rb_position);
            Quat3C b_orientation = RB_G(rigid_body_b, orientation);
            Vec3C b_rb_position = CJ_G(rigid_body_b_anchor_rb_position);

            // 1) Update constraint value.
            // Global position of anchor points.
            Vec3C anchor_a_g_pos =
                Vec3C(RB_G(rigid_body_a, origin)) + a_orientation.rotateVec3D(a_rb_position);
            Vec3C anchor_a_g_relative_to_com_pos = anchor_a_g_pos - Vec3C(RB_G(rigid_body_a, com));
            Vec3C anchor_b_g_pos =
                Vec3C(RB_G(rigid_body_b, origin)) + b_orientation.rotateVec3D(b_rb_position);
            Vec3C anchor_b_g_relative_to_com_pos = anchor_b_g_pos - Vec3C(RB_G(rigid_body_b, com));

            Vec3C constraint_value = anchor_b_g_pos - anchor_a_g_pos;
            CJ_S(constraint_value, constraint_value);

            Vec3C position = (anchor_b_g_pos + anchor_a_g_pos) * FloatC(0.5);
            CJ_S(position, position);

            // 2) Update jacobian (full 3x12: linear + angular for both bodies).
            // J_a_linear = -I, J_b_linear = +I.
            CJ_S(rigid_body_a_linear_J_0, Vec3C(-1, 0, 0));
            CJ_S(rigid_body_a_linear_J_1, Vec3C(0, -1, 0));
            CJ_S(rigid_body_a_linear_J_2, Vec3C(0, 0, -1));
            CJ_S(rigid_body_b_linear_J_0, Vec3C(1, 0, 0));
            CJ_S(rigid_body_b_linear_J_1, Vec3C(0, 1, 0));
            CJ_S(rigid_body_b_linear_J_2, Vec3C(0, 0, 1));
            // J_a_angular = skew(r_a), J_b_angular = -skew(r_b).
            CJ_S(
                rigid_body_a_angular_J_0,
                Vec3C(0, -anchor_a_g_relative_to_com_pos.z, anchor_a_g_relative_to_com_pos.y)
            );
            CJ_S(
                rigid_body_b_angular_J_0,
                Vec3C(0, anchor_b_g_relative_to_com_pos.z, -anchor_b_g_relative_to_com_pos.y)
            );
            CJ_S(
                rigid_body_a_angular_J_1,
                Vec3C(anchor_a_g_relative_to_com_pos.z, 0, -anchor_a_g_relative_to_com_pos.x)
            );
            CJ_S(
                rigid_body_b_angular_J_1,
                Vec3C(-anchor_b_g_relative_to_com_pos.z, 0, anchor_b_g_relative_to_com_pos.x)
            );
            CJ_S(
                rigid_body_a_angular_J_2,
                Vec3C(-anchor_a_g_relative_to_com_pos.y, anchor_a_g_relative_to_com_pos.x, 0)
            );
            CJ_S(
                rigid_body_b_angular_J_2,
                Vec3C(anchor_b_g_relative_to_com_pos.y, -anchor_b_g_relative_to_com_pos.x, 0)
            );

            // 3) Update effective mass.
            FloatC diag_mass_inverse =
                RB_G(rigid_body_a, mass_inverse) + RB_G(rigid_body_b, mass_inverse);
            Mat3C effective_mass =
                Mat3C::diagonal(diag_mass_inverse, diag_mass_inverse, diag_mass_inverse);

            Mat3C a_cross = anchor_a_g_relative_to_com_pos.toCrossMatrix();
            effective_mass +=
                a_cross * Mat3C(RB_G(rigid_body_a, moment_inertia_inverse)) * a_cross.transpose();
            Mat3C b_cross = anchor_b_g_relative_to_com_pos.toCrossMatrix();
            effective_mass +=
                b_cross * Mat3C(RB_G(rigid_body_b, moment_inertia_inverse)) * b_cross.transpose();

            Mat3C effective_mass_inverse = effective_mass.inverse();
            assert(effective_mass_inverse.isValid());
            CJ_S(effective_mass_inverse, effective_mass_inverse);
        }

        /**
         * @brief Builds the 3-DOF constraint locking the relative orientation of the two bodies.
         */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateFixedRotationConstraint(
            Kernel *k,
            RHandle constraint_joint,
            RHandle rigid_body_a,
            RHandle rigid_body_b
        ) {
            auto &ctx = k->ctx;

            // 1) Update constraint value.
            Quat3C initial_orientation_difference_inv =
                Quat3C(RB_G(rigid_body_b, initial_orientation)).inverse() *
                Quat3C(RB_G(rigid_body_a, initial_orientation));

            Quat3C error = Quat3C(RB_G(rigid_body_b, orientation)) *
                           initial_orientation_difference_inv *
                           Quat3C(RB_G(rigid_body_a, orientation)).inverse();

            CJ_S(constraint_value, error.toVec() * FloatC(2.0));

            // 2) Update Jacobian (full 3x12: linear = 0, angular = -I/+I).
            CJ_S(rigid_body_a_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_a_angular_J_0, Vec3C(-1, 0, 0));
            CJ_S(rigid_body_a_angular_J_1, Vec3C(0, -1, 0));
            CJ_S(rigid_body_a_angular_J_2, Vec3C(0, 0, -1));
            CJ_S(rigid_body_b_angular_J_0, Vec3C(1, 0, 0));
            CJ_S(rigid_body_b_angular_J_1, Vec3C(0, 1, 0));
            CJ_S(rigid_body_b_angular_J_2, Vec3C(0, 0, 1));

            // 3) Update effective mass.
            Mat3C effective_mass = RB_G(rigid_body_a, moment_inertia_inverse);
            effective_mass += RB_G(rigid_body_b, moment_inertia_inverse);
            Mat3C effective_mass_inverse = effective_mass.inverse();
            assert(effective_mass_inverse.isValid());
            CJ_S(effective_mass_inverse, effective_mass_inverse);
        }

        /**
         * @brief Builds the 2-DOF constraint aligning the hinge axis of a with the plane of b.
         */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateHingeRotationConstraint(
            Kernel *k,
            RHandle constraint_joint,
            RHandle rigid_body_a,
            RHandle rigid_body_b
        ) {
            auto &ctx = k->ctx;

            // Global axis of hinge.
            Vec3C hinge_a_g_axis = Quat3C(RB_G(rigid_body_a, orientation))
                                       .rotateVec3D(Vec3C(CJ_G(hinge_a_rb_axis)))
                                       .normalized();
            CJ_S(axis, hinge_a_g_axis);

            // Two orthogonal axes for enforcing a <-> b hinge axis alignment.
            Quat3C b_orientation = Quat3C(RB_G(rigid_body_b, orientation));
            Vec3C hinge_b_g_orthogonal_axis_1 =
                b_orientation.rotateVec3D(Vec3C(CJ_G(hinge_b_rb_orthogonal_axis_1))).normalized();
            Vec3C hinge_b_g_orthogonal_axis_2 =
                b_orientation.rotateVec3D(Vec3C(CJ_G(hinge_b_rb_orthogonal_axis_2))).normalized();

            // 1) Update constraint value.
            Vec3C constraint_value(
                hinge_a_g_axis.dot(hinge_b_g_orthogonal_axis_1),
                hinge_a_g_axis.dot(hinge_b_g_orthogonal_axis_2),
                0
            );
            CJ_S(constraint_value, constraint_value);

            // 2) Update jacobian (full 3x12: linear = 0, angular for 2 rows, row 2 = 0).
            CJ_S(rigid_body_a_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_2, Vec3C::zero());

            Vec3C axis_b1_cross_a = hinge_b_g_orthogonal_axis_1.cross(hinge_a_g_axis);
            Vec3C axis_b2_cross_a = hinge_b_g_orthogonal_axis_2.cross(hinge_a_g_axis);

            Vec3C a_angular_J_0 = -axis_b1_cross_a;
            CJ_S(rigid_body_a_angular_J_0, a_angular_J_0);
            CJ_S(rigid_body_b_angular_J_0, -a_angular_J_0);

            Vec3C a_angular_J_1 = -axis_b2_cross_a;
            CJ_S(rigid_body_a_angular_J_1, a_angular_J_1);
            CJ_S(rigid_body_b_angular_J_1, -a_angular_J_1);

            CJ_S(rigid_body_a_angular_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_angular_J_2, Vec3C::zero());

            // 3) Update effective mass.
            Mat3C a_moment_inertia_inverse = RB_G(rigid_body_a, moment_inertia_inverse);
            Mat3C b_moment_inertia_inverse = RB_G(rigid_body_b, moment_inertia_inverse);

            FloatC a = axis_b1_cross_a * a_moment_inertia_inverse * axis_b1_cross_a +
                       axis_b1_cross_a * b_moment_inertia_inverse * axis_b1_cross_a;
            FloatC b = axis_b1_cross_a * a_moment_inertia_inverse * axis_b2_cross_a +
                       axis_b1_cross_a * b_moment_inertia_inverse * axis_b2_cross_a;
            FloatC c = axis_b2_cross_a * a_moment_inertia_inverse * axis_b1_cross_a +
                       axis_b2_cross_a * b_moment_inertia_inverse * axis_b1_cross_a;
            FloatC d = axis_b2_cross_a * a_moment_inertia_inverse * axis_b2_cross_a +
                       axis_b2_cross_a * b_moment_inertia_inverse * axis_b2_cross_a;
            FloatC det = a * d - b * c;
            det = RS_ABS(det) < 1e-6 ? 1 : det;  // failover

            Mat3C effective_mass_inverse =
                Mat3C(d / det, -b / det, 0, -c / det, a / det, 0, 0, 0, 0);
            CJ_S(effective_mass_inverse, effective_mass_inverse);
        }

        /**
         * @brief Builds the hinge min and max angle limit constraint about the hinge axis.
         */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateHingeLimitConstraint(
            Kernel *k,
            RHandle constraint_joint,
            RHandle rigid_body_a,
            RHandle rigid_body_b
        ) {
            auto &ctx = k->ctx;

            Vec3C hinge_a_rb_axis = Vec3C(CJ_G(hinge_a_rb_axis));
            Vec3C hinge_a_g_axis =
                Quat3C(RB_G(rigid_body_a, orientation)).rotateVec3D(hinge_a_rb_axis).normalized();

            Quat3C initial_orientation_difference =
                Quat3C(RB_G(rigid_body_a, initial_orientation)).inverse() *
                Quat3C(RB_G(rigid_body_b, initial_orientation));

            Quat3C current_orientation_difference =
                Quat3C(RB_G(rigid_body_a, orientation)).inverse() *
                Quat3C(RB_G(rigid_body_b, orientation));

            // Compute relative rotation change in body A's initial frame
            // so that rotation_axis is in the same frame as hinge_a_rb_axis.
            Quat3C difference =
                (current_orientation_difference * initial_orientation_difference.inverse())
                    .normalized();

            Vec3C rotation_axis = difference.toVec();
            FloatC cos_half_angle = difference.w;
            FloatC sin_half_angle_abs = rotation_axis.length();
            FloatC dot_with_hinge_axis = rotation_axis * hinge_a_rb_axis;

            FloatC angle;
            if (dot_with_hinge_axis >= 0) {
                angle = FloatC(2) * RS_FI::rs_atan(sin_half_angle_abs / cos_half_angle);
            } else {
                angle = FloatC(2) * RS_FI::rs_atan(sin_half_angle_abs / -cos_half_angle);
            }

            CJ_S(angle, angle);

            // 1) & 2) Update constraint value & jacobian (full 3x12).
            // row_0 for min limit and row_1 for max limit, row_2 unused.
            CJ_S(rigid_body_a_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_2, Vec3C::zero());

            FloatC hinge_min = CJ_G(hinge_min);
            FloatC hinge_max = CJ_G(hinge_max);
            Vec3C constraint_value = Vec3C(angle - hinge_min, hinge_max - angle, 0);

            if (constraint_value.x < 0) {
                CJ_S(rigid_body_a_angular_J_0, -hinge_a_g_axis);
                CJ_S(rigid_body_b_angular_J_0, hinge_a_g_axis);
            } else {
                CJ_S(rigid_body_a_angular_J_0, Vec3C::zero());
                CJ_S(rigid_body_b_angular_J_0, Vec3C::zero());
                constraint_value.x = 0;
            }

            if (constraint_value.y < 0) {
                CJ_S(rigid_body_a_angular_J_1, hinge_a_g_axis);
                CJ_S(rigid_body_b_angular_J_1, -hinge_a_g_axis);
            } else {
                CJ_S(rigid_body_a_angular_J_1, Vec3C::zero());
                CJ_S(rigid_body_b_angular_J_1, Vec3C::zero());
                constraint_value.y = 0;
            }

            CJ_S(rigid_body_a_angular_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_angular_J_2, Vec3C::zero());

            CJ_S(constraint_value, constraint_value);

            // 3) Update effective mass.
            FloatC em =
                hinge_a_g_axis * Mat3C(RB_G(rigid_body_a, moment_inertia_inverse)) *
                    hinge_a_g_axis +
                hinge_a_g_axis * Mat3C(RB_G(rigid_body_b, moment_inertia_inverse)) * hinge_a_g_axis;
            FloatC em_inv = FloatC(1.0) / em;
            Mat3C effective_mass_inverse(em_inv, 0, 0, 0, em_inv, 0, 0, 0, 0);
            CJ_S(effective_mass_inverse, effective_mass_inverse);
        }

        /**
         * @brief Builds the hinge motor constraint driving the joint angle per its control mode.
         */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateHingeMotorConstraint(
            Kernel *k,
            RHandle constraint_joint,
            RHandle rigid_body_a,
            RHandle rigid_body_b,
            RBigFloat passed_time,
            FloatC hinge_signal
        ) {
            auto &ctx = k->ctx;

            CJ_S(hinge_motor_max_impulse, passed_time * CJ_G(hinge_max_torque));
            CJ_S(hinge_motor_target_impulse, 0);
            CJ_S(hinge_motor_impulse, 0);
            // Hinge motor impulse changes constantly, accumulated lambda only works for current
            // iteration loop.
            CJ_S(accumulated_lambda, Vec3C::zero());

            // Uniform J_linear = 0 and J_angular_2 = 0 for all motor modes.
            CJ_S(rigid_body_a_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_a_angular_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_angular_J_2, Vec3C::zero());

            Vec3C hinge_a_rb_axis = Vec3C(CJ_G(hinge_a_rb_axis));
            Vec3C hinge_a_g_axis =
                Quat3C(RB_G(rigid_body_a, orientation)).rotateVec3D(hinge_a_rb_axis).normalized();

            // Compute current angle and angular velocity (needed for most modes).
            Quat3C initial_orientation_difference =
                Quat3C(RB_G(rigid_body_a, initial_orientation)).inverse() *
                Quat3C(RB_G(rigid_body_b, initial_orientation));

            Quat3C current_orientation_difference =
                Quat3C(RB_G(rigid_body_a, orientation)).inverse() *
                Quat3C(RB_G(rigid_body_b, orientation));

            // Compute relative rotation change in body A's initial frame
            // so that rotation_axis is in the same frame as hinge_a_rb_axis.
            Quat3C difference =
                (current_orientation_difference * initial_orientation_difference.inverse())
                    .normalized();

            Vec3C rotation_axis = difference.toVec();
            FloatC cos_half_angle = difference.w;
            FloatC sin_half_angle_abs = rotation_axis.length();
            FloatC dot_with_hinge_axis = rotation_axis * hinge_a_rb_axis;

            FloatC angle;
            if (dot_with_hinge_axis >= 0) {
                angle = FloatC(2) * RS_FI::rs_atan(sin_half_angle_abs / cos_half_angle);
            } else {
                angle = FloatC(2) * RS_FI::rs_atan(sin_half_angle_abs / -cos_half_angle);
            }

            CJ_S(angle, angle);

            // Compute effective mass (needed for all modes).
            FloatC em =
                hinge_a_g_axis * Mat3C(RB_G(rigid_body_a, moment_inertia_inverse)) *
                    hinge_a_g_axis +
                hinge_a_g_axis * Mat3C(RB_G(rigid_body_b, moment_inertia_inverse)) * hinge_a_g_axis;
            FloatC em_inv = FloatC(1) / em;

            // Get control mode.
            int control_mode = CJ_G(hinge_control_mode);
            FloatC hinge_min = CJ_G(hinge_min);
            FloatC hinge_max = CJ_G(hinge_max);
            Vec3C constraint_value = Vec3C::zero();
            FloatC desired_torque = 0;

            // Mode-specific computation.
            if (control_mode == RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_BASIC_POSITION) {
                // Basic position control: drive a target speed that closes the angle error. Safe
                // torque-and-speed envelope.
                FloatC signal = RS_BOUND(hinge_signal, hinge_min, hinge_max);
                FloatC max_speed = CJ_G(hinge_max_speed);
                FloatC target_speed =
                    RS_BOUND((signal - angle) / passed_time, -max_speed, max_speed);

                CJ_S(rigid_body_a_angular_J_0, -hinge_a_g_axis);
                CJ_S(rigid_body_b_angular_J_0, hinge_a_g_axis);
                CJ_S(rigid_body_a_angular_J_1, Vec3C::zero());
                CJ_S(rigid_body_b_angular_J_1, Vec3C::zero());
                constraint_value.x = -target_speed;
            } else if (control_mode == RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_PD_POSITION) {
                // Stable PD: T = kp*(signal - angle) - kd*speed, deposited as a push with the
                // damping made implicit through the (1 + kd*dt*em) divisor so high gains stay
                // stable. The solver drives accumulated lambda toward this target.
                FloatC signal = RS_BOUND(hinge_signal, hinge_min, hinge_max);

                Vec3C a_angular_velocity = Mat3C(RB_G(rigid_body_a, moment_inertia_inverse)) *
                                           Vec3C(RB_G(rigid_body_a, angular_momentum));
                Vec3C b_angular_velocity = Mat3C(RB_G(rigid_body_b, moment_inertia_inverse)) *
                                           Vec3C(RB_G(rigid_body_b, angular_momentum));
                FloatC speed = (b_angular_velocity - a_angular_velocity) * hinge_a_g_axis;

                FloatC kp = CJ_G(hinge_position_control_kp);
                FloatC kd = CJ_G(hinge_position_control_kd);
                desired_torque = kp * (signal - angle) - kd * speed;

                // Single signed row about the hinge axis.
                CJ_S(rigid_body_a_angular_J_0, -hinge_a_g_axis);
                CJ_S(rigid_body_b_angular_J_0, hinge_a_g_axis);
                CJ_S(rigid_body_a_angular_J_1, Vec3C::zero());
                CJ_S(rigid_body_b_angular_J_1, Vec3C::zero());
                CJ_S(
                    hinge_motor_target_impulse,
                    passed_time * desired_torque / (FloatC(1) + kd * passed_time * em)
                );
            } else if (control_mode == RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_TORQUE) {
                // Torque command: drive toward max speed in the torque's direction, with the push
                // capped at the commanded torque. Safe torque-and-speed envelope.
                desired_torque =
                    RS_BOUND(hinge_signal, -CJ_G(hinge_max_torque), CJ_G(hinge_max_torque));
                FloatC max_speed = CJ_G(hinge_max_speed);

                CJ_S(rigid_body_a_angular_J_0, -hinge_a_g_axis);
                CJ_S(rigid_body_b_angular_J_0, hinge_a_g_axis);
                CJ_S(rigid_body_a_angular_J_1, Vec3C::zero());
                CJ_S(rigid_body_b_angular_J_1, Vec3C::zero());
                if (desired_torque > 0) {
                    constraint_value.x = -max_speed;
                } else if (desired_torque < 0) {
                    constraint_value.x = max_speed;
                }
                CJ_S(hinge_motor_max_impulse, RS_ABS(desired_torque) * passed_time);
            } else {
                // Velocity command, for wheels: drive toward the commanded angular velocity using
                // up to max torque. Safe torque-and-speed envelope.
                FloatC max_speed = CJ_G(hinge_max_speed);
                FloatC target_speed = RS_BOUND(hinge_signal, -max_speed, max_speed);

                CJ_S(rigid_body_a_angular_J_0, -hinge_a_g_axis);
                CJ_S(rigid_body_b_angular_J_0, hinge_a_g_axis);
                CJ_S(rigid_body_a_angular_J_1, Vec3C::zero());
                CJ_S(rigid_body_b_angular_J_1, Vec3C::zero());
                constraint_value.x = -target_speed;
            }

            CJ_S(constraint_value, constraint_value);
            CJ_S(hinge_desired_torque, desired_torque);

            Mat3C effective_mass_inverse(em_inv, 0, 0, 0, em_inv, 0, 0, 0, 0);
            CJ_S(effective_mass_inverse, effective_mass_inverse);
        }

        /**
         * @brief Builds the 2-DOF constraint confining motion to the prismatic slide axis.
         */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updatePrismaticTranslationConstraint(
            Kernel *k,
            RHandle constraint_joint,
            RHandle rigid_body_a,
            RHandle rigid_body_b
        ) {
            auto &ctx = k->ctx;

            Quat3C a_orientation = RB_G(rigid_body_a, orientation);
            Vec3C a_rb_position = CJ_G(rigid_body_a_anchor_rb_position);
            Quat3C b_orientation = RB_G(rigid_body_b, orientation);
            Vec3C b_rb_position = CJ_G(rigid_body_b_anchor_rb_position);

            // Global slide axis.
            Vec3C slide_g =
                a_orientation.rotateVec3D(Vec3C(CJ_G(prismatic_a_rb_axis))).normalized();

            // Two orthogonal axes for perpendicular translation.
            Vec3C n1 = b_orientation.rotateVec3D(Vec3C(CJ_G(prismatic_b_rb_orthogonal_axis_1)))
                           .normalized();
            Vec3C n2 = b_orientation.rotateVec3D(Vec3C(CJ_G(prismatic_b_rb_orthogonal_axis_2)))
                           .normalized();

            // 1) Update constraint value.
            Vec3C anchor_a_g_pos =
                Vec3C(RB_G(rigid_body_a, origin)) + a_orientation.rotateVec3D(a_rb_position);
            Vec3C r_a = anchor_a_g_pos - Vec3C(RB_G(rigid_body_a, com));
            Vec3C anchor_b_g_pos =
                Vec3C(RB_G(rigid_body_b, origin)) + b_orientation.rotateVec3D(b_rb_position);
            Vec3C r_b = anchor_b_g_pos - Vec3C(RB_G(rigid_body_b, com));

            // u = displacement between anchor points.
            Vec3C u = anchor_b_g_pos - anchor_a_g_pos;
            Vec3C constraint_value = Vec3C(u * n1, u * n2, 0);
            CJ_S(constraint_value, constraint_value);

            Vec3C position = (anchor_b_g_pos + anchor_a_g_pos) * FloatC(0.5);
            CJ_S(position, position);

            // 2) Update jacobian (full 3x12: 2 active rows perpendicular to slide axis).
            Vec3C r_a_plus_u = r_a + u;
            CJ_S(rigid_body_a_linear_J_0, -n1);
            CJ_S(rigid_body_a_linear_J_1, -n2);
            CJ_S(rigid_body_a_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_0, n1);
            CJ_S(rigid_body_b_linear_J_1, n2);
            CJ_S(rigid_body_b_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_a_angular_J_0, -(r_a_plus_u.cross(n1)));
            CJ_S(rigid_body_a_angular_J_1, -(r_a_plus_u.cross(n2)));
            CJ_S(rigid_body_a_angular_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_angular_J_0, r_b.cross(n1));
            CJ_S(rigid_body_b_angular_J_1, r_b.cross(n2));
            CJ_S(rigid_body_b_angular_J_2, Vec3C::zero());

            // 3) Update effective mass (2x2 in top-left of 3x3).
            FloatC diag_mass_inverse =
                RB_G(rigid_body_a, mass_inverse) + RB_G(rigid_body_b, mass_inverse);
            Mat3C a_I_inv = RB_G(rigid_body_a, moment_inertia_inverse);
            Mat3C b_I_inv = RB_G(rigid_body_b, moment_inertia_inverse);

            Vec3C ra_u_cross_n1 = r_a_plus_u.cross(n1);
            Vec3C ra_u_cross_n2 = r_a_plus_u.cross(n2);
            Vec3C rb_cross_n1 = r_b.cross(n1);
            Vec3C rb_cross_n2 = r_b.cross(n2);

            FloatC a = diag_mass_inverse + ra_u_cross_n1 * a_I_inv * ra_u_cross_n1 +
                       rb_cross_n1 * b_I_inv * rb_cross_n1;
            FloatC b =
                ra_u_cross_n1 * a_I_inv * ra_u_cross_n2 + rb_cross_n1 * b_I_inv * rb_cross_n2;
            FloatC d = diag_mass_inverse + ra_u_cross_n2 * a_I_inv * ra_u_cross_n2 +
                       rb_cross_n2 * b_I_inv * rb_cross_n2;
            FloatC det = a * d - b * b;
            det = RS_ABS(det) < 1e-6 ? 1 : det;

            Mat3C effective_mass_inverse =
                Mat3C(d / det, -b / det, 0, -b / det, a / det, 0, 0, 0, 0);
            CJ_S(effective_mass_inverse, effective_mass_inverse);
        }

        /**
         * @brief Builds the prismatic min and max travel limit constraint along the slide axis.
         */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updatePrismaticLimitConstraint(
            Kernel *k,
            RHandle constraint_joint,
            RHandle rigid_body_a,
            RHandle rigid_body_b
        ) {
            auto &ctx = k->ctx;

            Quat3C a_orientation = RB_G(rigid_body_a, orientation);
            Vec3C a_rb_position = CJ_G(rigid_body_a_anchor_rb_position);
            Quat3C b_orientation = RB_G(rigid_body_b, orientation);
            Vec3C b_rb_position = CJ_G(rigid_body_b_anchor_rb_position);

            // Global slide axis.
            Vec3C slide_g =
                a_orientation.rotateVec3D(Vec3C(CJ_G(prismatic_a_rb_axis))).normalized();

            Vec3C anchor_a_g_pos =
                Vec3C(RB_G(rigid_body_a, origin)) + a_orientation.rotateVec3D(a_rb_position);
            Vec3C r_a = anchor_a_g_pos - Vec3C(RB_G(rigid_body_a, com));
            Vec3C anchor_b_g_pos =
                Vec3C(RB_G(rigid_body_b, origin)) + b_orientation.rotateVec3D(b_rb_position);
            Vec3C r_b = anchor_b_g_pos - Vec3C(RB_G(rigid_body_b, com));

            // u = displacement between anchor points.
            Vec3C u = anchor_b_g_pos - anchor_a_g_pos;
            Vec3C r_a_plus_u = r_a + u;

            FloatC dist = u * slide_g;
            CJ_S(prismatic_distance, dist);

            // 1) & 2) Update constraint value & jacobian (full 3x12).
            CJ_S(rigid_body_a_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_2, Vec3C::zero());

            FloatC prismatic_min = CJ_G(prismatic_min);
            FloatC prismatic_max = CJ_G(prismatic_max);
            // C_min = u·a - d_min >= 0, C_max = d_max - u·a >= 0.
            Vec3C constraint_value = Vec3C(dist - prismatic_min, prismatic_max - dist, 0);

            if (constraint_value.x < 0) {
                // dist < min: C_min violated.
                CJ_S(rigid_body_a_linear_J_0, -slide_g);
                CJ_S(rigid_body_b_linear_J_0, slide_g);
                CJ_S(rigid_body_a_angular_J_0, -(r_a_plus_u.cross(slide_g)));
                CJ_S(rigid_body_b_angular_J_0, r_b.cross(slide_g));
            } else {
                CJ_S(rigid_body_a_angular_J_0, Vec3C::zero());
                CJ_S(rigid_body_b_angular_J_0, Vec3C::zero());
                constraint_value.x = 0;
            }

            if (constraint_value.y < 0) {
                // dist > max: C_max violated.
                CJ_S(rigid_body_a_linear_J_1, slide_g);
                CJ_S(rigid_body_b_linear_J_1, -slide_g);
                CJ_S(rigid_body_a_angular_J_1, r_a_plus_u.cross(slide_g));
                CJ_S(rigid_body_b_angular_J_1, -(r_b.cross(slide_g)));
            } else {
                CJ_S(rigid_body_a_angular_J_1, Vec3C::zero());
                CJ_S(rigid_body_b_angular_J_1, Vec3C::zero());
                constraint_value.y = 0;
            }

            CJ_S(rigid_body_a_angular_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_angular_J_2, Vec3C::zero());

            CJ_S(constraint_value, constraint_value);

            // 3) Update effective mass (2x2 in top-left of 3x3).
            Vec3C ra_u_cross_slide = r_a_plus_u.cross(slide_g);
            Vec3C rb_cross_slide = r_b.cross(slide_g);
            FloatC em =
                RB_G(rigid_body_a, mass_inverse) + RB_G(rigid_body_b, mass_inverse) +
                ra_u_cross_slide * Mat3C(RB_G(rigid_body_a, moment_inertia_inverse)) *
                    ra_u_cross_slide +
                rb_cross_slide * Mat3C(RB_G(rigid_body_b, moment_inertia_inverse)) * rb_cross_slide;
            FloatC em_inv = FloatC(1.0) / em;
            Mat3C effective_mass_inverse(em_inv, 0, 0, 0, em_inv, 0, 0, 0, 0);
            CJ_S(effective_mass_inverse, effective_mass_inverse);
        }

        /**
         * @brief Builds the prismatic motor constraint driving the slide distance per its control
         * mode.
         */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updatePrismaticMotorConstraint(
            Kernel *k,
            RHandle constraint_joint,
            RHandle rigid_body_a,
            RHandle rigid_body_b,
            RBigFloat passed_time,
            FloatC prismatic_signal
        ) {
            auto &ctx = k->ctx;

            CJ_S(prismatic_motor_max_impulse, passed_time * CJ_G(prismatic_max_force));
            CJ_S(prismatic_motor_target_impulse, 0);
            CJ_S(prismatic_motor_impulse, 0);
            // Prismatic motor impulse changes constantly, accumulated lambda only works for
            // current iteration loop.
            CJ_S(accumulated_lambda, Vec3C::zero());

            // Motor applies pure linear force along slide axis, no angular component.
            CJ_S(rigid_body_a_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_a_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_0, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_1, Vec3C::zero());
            CJ_S(rigid_body_b_linear_J_2, Vec3C::zero());
            CJ_S(rigid_body_a_angular_J_0, Vec3C::zero());
            CJ_S(rigid_body_a_angular_J_1, Vec3C::zero());
            CJ_S(rigid_body_a_angular_J_2, Vec3C::zero());
            CJ_S(rigid_body_b_angular_J_0, Vec3C::zero());
            CJ_S(rigid_body_b_angular_J_1, Vec3C::zero());
            CJ_S(rigid_body_b_angular_J_2, Vec3C::zero());

            Quat3C a_orientation = RB_G(rigid_body_a, orientation);
            Vec3C slide_g =
                a_orientation.rotateVec3D(Vec3C(CJ_G(prismatic_a_rb_axis))).normalized();

            Vec3C a_rb_position = CJ_G(rigid_body_a_anchor_rb_position);
            Quat3C b_orientation = RB_G(rigid_body_b, orientation);
            Vec3C b_rb_position = CJ_G(rigid_body_b_anchor_rb_position);

            Vec3C anchor_a_g_pos =
                Vec3C(RB_G(rigid_body_a, origin)) + a_orientation.rotateVec3D(a_rb_position);
            Vec3C anchor_b_g_pos =
                Vec3C(RB_G(rigid_body_b, origin)) + b_orientation.rotateVec3D(b_rb_position);

            // u = displacement between anchor points, dist = u · a.
            Vec3C u = anchor_b_g_pos - anchor_a_g_pos;
            FloatC dist = u * slide_g;
            CJ_S(prismatic_distance, dist);

            // Effective mass for pure linear motor: 1/m_a + 1/m_b.
            FloatC em = RB_G(rigid_body_a, mass_inverse) + RB_G(rigid_body_b, mass_inverse);
            FloatC em_inv = FloatC(1) / em;

            // Get control mode.
            int control_mode = CJ_G(prismatic_control_mode);
            FloatC prismatic_min = CJ_G(prismatic_min);
            FloatC prismatic_max = CJ_G(prismatic_max);
            Vec3C constraint_value = Vec3C::zero();
            FloatC desired_force = 0;

            // Interpolate signal (0-1) to position in [prismatic_min, prismatic_max].
            FloatC target_position =
                prismatic_min + prismatic_signal * (prismatic_max - prismatic_min);

            // Mode-specific computation.
            if (control_mode == RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_BASIC_POSITION) {
                // Basic position control: drive a target speed that closes the distance error.
                // Safe force-and-speed envelope.
                FloatC max_speed = CJ_G(prismatic_max_speed);
                FloatC target_speed =
                    RS_BOUND((target_position - dist) / passed_time, -max_speed, max_speed);

                CJ_S(rigid_body_a_linear_J_0, -slide_g);
                CJ_S(rigid_body_b_linear_J_0, slide_g);
                constraint_value.x = -target_speed;
            } else if (control_mode == RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_PD_POSITION) {
                // Stable PD: F = kp*(target - dist) - kd*speed, deposited as a push with the
                // damping made implicit through the (1 + kd*dt*em) divisor. The solver drives
                // accumulated lambda toward this target.
                Vec3C a_linear_velocity =
                    RB_G(rigid_body_a, mass_inverse) * Vec3C(RB_G(rigid_body_a, linear_momentum));
                Vec3C b_linear_velocity =
                    RB_G(rigid_body_b, mass_inverse) * Vec3C(RB_G(rigid_body_b, linear_momentum));
                FloatC speed = (b_linear_velocity - a_linear_velocity) * slide_g;

                FloatC kp = CJ_G(prismatic_position_control_kp);
                FloatC kd = CJ_G(prismatic_position_control_kd);
                desired_force = kp * (target_position - dist) - kd * speed;

                CJ_S(rigid_body_a_linear_J_0, -slide_g);
                CJ_S(rigid_body_b_linear_J_0, slide_g);
                CJ_S(
                    prismatic_motor_target_impulse,
                    passed_time * desired_force / (FloatC(1) + kd * passed_time * em)
                );
            } else if (control_mode == RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_FORCE) {
                // Force command: drive toward max speed in the force's direction, with the push
                // capped at the commanded force. Safe force-and-speed envelope.
                desired_force = RS_BOUND(
                    prismatic_signal * CJ_G(prismatic_max_force),
                    -CJ_G(prismatic_max_force),
                    CJ_G(prismatic_max_force)
                );
                FloatC max_speed = CJ_G(prismatic_max_speed);

                CJ_S(rigid_body_a_linear_J_0, -slide_g);
                CJ_S(rigid_body_b_linear_J_0, slide_g);
                if (desired_force > 0) {
                    constraint_value.x = -max_speed;
                } else if (desired_force < 0) {
                    constraint_value.x = max_speed;
                }
                CJ_S(prismatic_motor_max_impulse, RS_ABS(desired_force) * passed_time);
            } else {
                // Velocity command: drive toward the commanded slide velocity using up to max
                // force. Safe force-and-speed envelope.
                FloatC max_speed = CJ_G(prismatic_max_speed);
                FloatC target_speed = RS_BOUND(prismatic_signal, -max_speed, max_speed);

                CJ_S(rigid_body_a_linear_J_0, -slide_g);
                CJ_S(rigid_body_b_linear_J_0, slide_g);
                constraint_value.x = -target_speed;
            }

            CJ_S(constraint_value, constraint_value);
            CJ_S(prismatic_desired_force, desired_force);

            Mat3C effective_mass_inverse(em_inv, 0, 0, 0, em_inv, 0, 0, 0, 0);
            CJ_S(effective_mass_inverse, effective_mass_inverse);
        }
    };

    /**
     * @brief Block-shared-memory PGS sweep resolving a structure's joints in velocity space.
     *
     * @details One block per structure, one thread per sub-constraint. Rigid body momentum and mass
     * are cooperatively loaded into shared memory and Jacobian rows preloaded into registers, then
     * graph-colored Gauss-Seidel iterations apply
     * delta_lambda = -K_inv (rel_vel + correction), with syncThreadsInBlock between color passes in
     * place of separate kernel launches. For position-error rows the correction is the Baumgarte
     * velocity bias baumgarte_ratio C / passed_time; for motor rows C is already a target relative
     * velocity, so the correction is C directly. Limit and motor constraints clamp the accumulated
     * lambda, and the resolved momenta are written back to global memory. See the class docstring
     * for the shared symbol table.
     *
     * @param k               simulation kernel.
     * @param cj_layout       per-structure layout of constraint joints.
     * @param rb_layout       per-structure layout of rigid bodies.
     * @param max_colors      number of graph colors sequenced within each sweep.
     */
    class RSCF_PGSUpdateJoint : public device::RS_ComputeFunction<RSCF_PGSUpdateJoint> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_PGSUpdateJoint>;
        using BaseType::blockDimX;
        using BaseType::blockIdxX;
        using BaseType::sharedMem;
        using BaseType::syncThreadsInBlock;
        using BaseType::threadIdxX;
        using IsSyncBlockRequired = std::true_type;

        struct SharedRigidBodyState {
            RVec3f lin_mom;
            RVec3f ang_mom;
            float mass_inv;
            RMat3f I_inv;
        };

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            device::RS_Vector<RS_ConstraintJointBlockRange> block_ranges,
            RSize max_colors
        ) {
            if (k == nullptr) return;
            auto &ctx = k->ctx;
            RSize iterations = k->rigidbody_constraint_iterations;
            FloatC baumgarte_ratio = FloatC(k->rigidbody_constraint_position_correction_ratio);
            RIndex block_idx = blockIdxX();
            RIndex tid = threadIdxX();

            // One block = one structure; pull its four offsets/counts from the
            // packed descriptor instead of cross-indexing two pooled layouts.
            RS_ConstraintJointBlockRange range = block_ranges[block_idx];
            RIndex cj_offset = range.cj_offset;
            RSize cj_count = range.cj_count;
            RIndex rb_offset = range.rb_offset;
            RSize rb_count = range.rb_count;

            // Step 1: Cast shared memory.
            auto *shared_rb = reinterpret_cast<SharedRigidBodyState *>(sharedMem());

            // Step 2: Cooperative load rigid bodies into shared memory.
            for (RIndex rb = tid; rb < rb_count; rb += blockDimX()) {
                RHandle h = rb_offset + rb;
                shared_rb[rb].lin_mom = RB_G(h, linear_momentum);
                shared_rb[rb].ang_mom = RB_G(h, angular_momentum);
                shared_rb[rb].mass_inv = RB_G(h, mass_inverse);
                shared_rb[rb].I_inv = RB_G(h, moment_inertia_inverse);
            }
            syncThreadsInBlock();

            // Step 3: Preload constraint data into registers.
            bool is_active = (tid < cj_count);
            RHandle cj = cj_offset + tid;

            Vec3C J_a_lin_0, J_a_lin_1, J_a_lin_2;
            Vec3C J_a_ang_0, J_a_ang_1, J_a_ang_2;
            Vec3C J_b_lin_0, J_b_lin_1, J_b_lin_2;
            Vec3C J_b_ang_0, J_b_ang_1, J_b_ang_2;
            Mat3C K_inv;
            Vec3C constraint_value, accumulated_lambda;
            FloatC passed_time;
            FloatC motor_max_impulse = 0;
            FloatC motor_target_impulse = 0;
            bool is_target_impulse_motor = false;
            FloatC compliance = 1;
            RSE_ConstraintJointType my_type = RSE_NULL_JOINT_CONSTRAINT;
            RIndex my_color = RS_NULL_INDEX;
            RIndex rb_a_idx = 0, rb_b_idx = 0;

            if (is_active) {
                my_type = CJ_G(cj, type);
                my_color = CJ_G(cj, color);
                rb_a_idx = CJ_G(cj, rigid_body_a) - rb_offset;
                rb_b_idx = CJ_G(cj, rigid_body_b) - rb_offset;

                J_a_lin_0 = CJ_G(cj, rigid_body_a_linear_J_0);
                J_a_lin_1 = CJ_G(cj, rigid_body_a_linear_J_1);
                J_a_lin_2 = CJ_G(cj, rigid_body_a_linear_J_2);
                J_a_ang_0 = CJ_G(cj, rigid_body_a_angular_J_0);
                J_a_ang_1 = CJ_G(cj, rigid_body_a_angular_J_1);
                J_a_ang_2 = CJ_G(cj, rigid_body_a_angular_J_2);
                J_b_lin_0 = CJ_G(cj, rigid_body_b_linear_J_0);
                J_b_lin_1 = CJ_G(cj, rigid_body_b_linear_J_1);
                J_b_lin_2 = CJ_G(cj, rigid_body_b_linear_J_2);
                J_b_ang_0 = CJ_G(cj, rigid_body_b_angular_J_0);
                J_b_ang_1 = CJ_G(cj, rigid_body_b_angular_J_1);
                J_b_ang_2 = CJ_G(cj, rigid_body_b_angular_J_2);
                K_inv = CJ_G(cj, effective_mass_inverse);
                constraint_value = CJ_G(cj, constraint_value);
                accumulated_lambda = CJ_G(cj, accumulated_lambda);
                compliance = FloatC(CJ_G(cj, compliance));
                // Warm-start damping: scale previous frame's lambda to prevent buildup
                // damping_z=0 means no damping (full carry-over), damping_z=1 means reset to zero.
                accumulated_lambda *= (FloatC(1) - k->rigidbody_constraint_warm_start_damping_z);
                passed_time = FloatC(k->getDeviceTimeSinceLastOperation(
                    RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_CONSTRAINT_AND_CONTACT
                ));
                // Motors are recomputed fresh each step and never warm-started; the target-impulse
                // path also relies on a zero start so its push is deposited exactly once.
                if (my_type & RSE_HINGE_JOINT_MOTOR) {
                    motor_max_impulse = FloatC(CJ_G(cj, hinge_motor_max_impulse));
                    motor_target_impulse = FloatC(CJ_G(cj, hinge_motor_target_impulse));
                    is_target_impulse_motor = CJ_G(cj, hinge_control_mode) ==
                                              RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_PD_POSITION;
                    accumulated_lambda = Vec3C::zero();
                } else if (my_type & RSE_PRISMATIC_JOINT_MOTOR) {
                    motor_max_impulse = FloatC(CJ_G(cj, prismatic_motor_max_impulse));
                    motor_target_impulse = FloatC(CJ_G(cj, prismatic_motor_target_impulse));
                    is_target_impulse_motor =
                        CJ_G(cj, prismatic_control_mode) ==
                        RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_PD_POSITION;
                    accumulated_lambda = Vec3C::zero();
                }
            }
            syncThreadsInBlock();

            // Step 4: PGS iterations.
            for (RIndex iter = 0; iter < iterations; iter++) {
                for (RIndex c = 0; c < max_colors; c++) {
                    if (is_active && my_color == c) {
                        // Read velocities from shared memory (cast float→compute precision).
                        FloatC mass_inv_a = FloatC(shared_rb[rb_a_idx].mass_inv);
                        Mat3C I_inv_a = Mat3C(shared_rb[rb_a_idx].I_inv);
                        Vec3C v_a = Vec3C(shared_rb[rb_a_idx].lin_mom) * mass_inv_a;
                        Vec3C w_a = I_inv_a * Vec3C(shared_rb[rb_a_idx].ang_mom);
                        FloatC mass_inv_b = FloatC(shared_rb[rb_b_idx].mass_inv);
                        Mat3C I_inv_b = Mat3C(shared_rb[rb_b_idx].I_inv);
                        Vec3C v_b = Vec3C(shared_rb[rb_b_idx].lin_mom) * mass_inv_b;
                        Vec3C w_b = I_inv_b * Vec3C(shared_rb[rb_b_idx].ang_mom);

                        // J * v (uniform for all constraint types).
                        Vec3C rel_vel;
                        rel_vel.x =
                            J_a_lin_0 * v_a + J_a_ang_0 * w_a + J_b_lin_0 * v_b + J_b_ang_0 * w_b;
                        rel_vel.y =
                            J_a_lin_1 * v_a + J_a_ang_1 * w_a + J_b_lin_1 * v_b + J_b_ang_1 * w_b;
                        rel_vel.z =
                            J_a_lin_2 * v_a + J_a_ang_2 * w_a + J_b_lin_2 * v_b + J_b_ang_2 * w_b;

                        Vec3C delta_lambda;
                        if (is_target_impulse_motor) {
                            // PD motor: deposit the stable-PD push once, on the single signed row,
                            // capped by the max torque/force impulse.
                            FloatC target = RS_BOUND(
                                motor_target_impulse, -motor_max_impulse, motor_max_impulse
                            );
                            Vec3C new_lambda(target, FloatC(0), FloatC(0));
                            delta_lambda = new_lambda - accumulated_lambda;
                            accumulated_lambda = new_lambda;
                        } else {
                            // Standard velocity solve for every other row. Position and limit rows
                            // scale their position error into a Baumgarte velocity bias over the
                            // step; velocity-servo motors store a target relative velocity in
                            // constraint_value and take it directly.
                            bool is_velocity_target =
                                my_type & (RSE_HINGE_JOINT_MOTOR | RSE_PRISMATIC_JOINT_MOTOR);
                            Vec3C correction =
                                is_velocity_target
                                    ? constraint_value
                                    : baumgarte_ratio * constraint_value / passed_time;
                            delta_lambda = -(K_inv * (rel_vel + correction)) * compliance;

                            // Accumulate.
                            Vec3C old_lambda = accumulated_lambda;
                            accumulated_lambda = old_lambda + delta_lambda;

                            // Clamp only for limit/motor.
                            if (my_type & (RSE_HINGE_JOINT_LIMIT | RSE_PRISMATIC_JOINT_LIMIT)) {
                                accumulated_lambda = accumulated_lambda.max(Vec3C::zero());
                            } else if (is_velocity_target) {
                                accumulated_lambda.x = RS_BOUND(
                                    accumulated_lambda.x, -motor_max_impulse, motor_max_impulse
                                );
                                accumulated_lambda.y = RS_BOUND(
                                    accumulated_lambda.y, -motor_max_impulse, motor_max_impulse
                                );
                                accumulated_lambda.z = RS_BOUND(
                                    accumulated_lambda.z, -motor_max_impulse, motor_max_impulse
                                );
                            }

                            delta_lambda = accumulated_lambda - old_lambda;
                        }

                        // J^T * delta_lambda.
                        Vec3C imp_a_lin =
                            Mat3C(J_a_lin_0, J_a_lin_1, J_a_lin_2).transpose() * delta_lambda;
                        Vec3C imp_a_ang =
                            Mat3C(J_a_ang_0, J_a_ang_1, J_a_ang_2).transpose() * delta_lambda;
                        Vec3C imp_b_lin =
                            Mat3C(J_b_lin_0, J_b_lin_1, J_b_lin_2).transpose() * delta_lambda;
                        Vec3C imp_b_ang =
                            Mat3C(J_b_ang_0, J_b_ang_1, J_b_ang_2).transpose() * delta_lambda;

                        // Write to shared memory (no conflict within same color).
                        shared_rb[rb_a_idx].lin_mom += RVec3f(imp_a_lin);
                        shared_rb[rb_a_idx].ang_mom += RVec3f(imp_a_ang);
                        shared_rb[rb_b_idx].lin_mom += RVec3f(imp_b_lin);
                        shared_rb[rb_b_idx].ang_mom += RVec3f(imp_b_ang);
                    }
                    syncThreadsInBlock();
                }
            }

            // Step 5: Write back to global memory.
            if (is_active) {
                CJ_S(cj, accumulated_lambda, accumulated_lambda);
                if (my_type & RSE_HINGE_JOINT_MOTOR) {
                    CJ_S(
                        cj,
                        hinge_motor_impulse,
                        RS_MAX(RS_ABS(accumulated_lambda.x), RS_ABS(accumulated_lambda.y))
                    );
                } else if (my_type & RSE_PRISMATIC_JOINT_MOTOR) {
                    CJ_S(
                        cj,
                        prismatic_motor_impulse,
                        RS_MAX(RS_ABS(accumulated_lambda.x), RS_ABS(accumulated_lambda.y))
                    );
                }
            }
            for (RIndex rb = tid; rb < rb_count; rb += blockDimX()) {
                RHandle h = rb_offset + rb;
                if (RS_RigidBodyStateUnion{RB_G(h, state)}.rigid_body_state.is_fixed) continue;
                RB_S(h, linear_momentum, shared_rb[rb].lin_mom);
                RB_S(h, angular_momentum, shared_rb[rb].ang_mom);
            }
        }
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_CONSTRAINT_C_H