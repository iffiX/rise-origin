// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONSTRAINT_M_H
#define RS_CONSTRAINT_M_H

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/math/rs_mat.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

namespace logic {

RS_ENUM(
    RSE_ConstraintJointType,
    unsigned int,
    (RSE_NULL_JOINT_CONSTRAINT, 0),
    (RSE_BALL_AND_SOCKET_JOINT_TRANSLATION, 0b1),
    (RSE_FIXED_JOINT_TRANSLATION, 0b10),
    (RSE_FIXED_JOINT_ROTATION, 0b100),
    (RSE_HINGE_JOINT_TRANSLATION, 0b1000),
    (RSE_HINGE_JOINT_ROTATION, 0b10000),
    (RSE_HINGE_JOINT_LIMIT, 0b100000),
    (RSE_HINGE_JOINT_MOTOR, 0b1000000),
    (RSE_PRISMATIC_JOINT_PERPENDICULAR_TRANSLATION, 0b10000000),
    (RSE_PRISMATIC_JOINT_ROTATION, 0b100000000),
    (RSE_PRISMATIC_JOINT_LIMIT, 0b1000000000),
    (RSE_PRISMATIC_JOINT_MOTOR, 0b10000000000)
)

RS_ENUM(
    RSE_ConstraintJointHingeControlMode,
    unsigned int,
    (RSE_CONSTRAINT_JOINT_HINGE_CONTROL_BASIC_POSITION, 0),
    (RSE_CONSTRAINT_JOINT_HINGE_CONTROL_PD_POSITION, 1),
    (RSE_CONSTRAINT_JOINT_HINGE_CONTROL_TORQUE, 2),
    (RSE_CONSTRAINT_JOINT_HINGE_CONTROL_VELOCITY, 3)
)

RS_ENUM(
    RSE_ConstraintJointPrismaticControlMode,
    unsigned int,
    (RSE_CONSTRAINT_JOINT_PRISMATIC_CONTROL_BASIC_POSITION, 0),
    (RSE_CONSTRAINT_JOINT_PRISMATIC_CONTROL_PD_POSITION, 1),
    (RSE_CONSTRAINT_JOINT_PRISMATIC_CONTROL_FORCE, 2),
    (RSE_CONSTRAINT_JOINT_PRISMATIC_CONTROL_VELOCITY, 3)
)

union RS_ConstraintJointStateUnion {
    RState constraint_joint_state_packed;
    struct RS_ConstraintJointState {
        RSE_ConstraintJointType type : 11;
        RSE_ConstraintJointHingeControlMode hinge_control_mode : 3;
        RSE_ConstraintJointPrismaticControlMode prismatic_control_mode : 3;
    } constraint_joint_state;
};

struct RS_ConstraintJointBlockRange {
    /// Absolute offset of this structure's constraint joints in the pooled buffer
    RSize cj_offset;

    /// Number of constraint joints owned by this structure
    RSize cj_count;

    /// Absolute offset of this structure's rigid bodies in the pooled buffer
    RSize rb_offset;

    /// Number of rigid bodies owned by this structure
    RSize rb_count;
};

template <RSE_FloatType FloatMemAccuracy>
class RS_ConstraintJointMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;
    using Vec3M = RS_Vec3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;
    using Quat3M = RS_Quat3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;
    using Mat3M = RS_Mat3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;

    /**
     * Pre-set attributes
     */
    /// Constraint state
    RState state = 0;

    /// Objectified structure signal handle (index into ctx.structure_signals).
    RHandle structure_signal = RS_NULL_HANDLE;

    /// Structure gid of this constraint joint
    RIndex structure_gid = RS_NULL_INDEX;

    /// Structure scope ID of the constraint
    RIndex sid = RS_NULL_INDEX;

    /// Structure scope ID of the joint
    RIndex joint_sid = RS_NULL_INDEX;

    /// If the constraint is a hinge joint with motor enabled,
    /// this is the structure scope ID of the motor
    RIndex motor_sid = RS_NULL_INDEX;

    /// Structure scope ID of the rigid body a connected by the constraint
    RIndex rigid_body_a_sid = RS_NULL_INDEX;

    /// Structure scope ID of the rigid body b connected by the constraint
    RIndex rigid_body_b_sid = RS_NULL_INDEX;

    /// Rigid body scope anchor position of anchor A, relative to the origin of rigid body A
    Vec3M rigid_body_a_anchor_rb_position = Vec3M::zero();

    /// Rigid body scope anchor position of anchor B, relative to the origin of rigid body B
    Vec3M rigid_body_b_anchor_rb_position = Vec3M::zero();

    /// Rigid body scope axis of the hinge joint, relative to the rigid body A
    Vec3M hinge_a_rb_axis = Vec3M::zero();

    /// Rigid body orthogonal scope axis of the hinge joint, relative to the rigid body B
    Vec3M hinge_b_rb_orthogonal_axis_1 = Vec3M::zero();

    Vec3M hinge_b_rb_orthogonal_axis_2 = Vec3M::zero();

    FloatM hinge_min = 0;

    FloatM hinge_max = 0;

    FloatM hinge_position_control_kp = 0;

    FloatM hinge_position_control_kd = 0;

    FloatM hinge_max_speed = 0;

    FloatM hinge_max_torque = 0;

    FloatM hinge_desired_torque = 0;

    RIndex hinge_signal_sid = RS_NULL_INDEX;

    /// Rigid body scope slide axis of the prismatic joint, relative to the rigid body A
    Vec3M prismatic_a_rb_axis = Vec3M::zero();

    /// Rigid body scope orthogonal axes for perpendicular translation, relative to rigid body B
    Vec3M prismatic_b_rb_orthogonal_axis_1 = Vec3M::zero();

    Vec3M prismatic_b_rb_orthogonal_axis_2 = Vec3M::zero();

    FloatM prismatic_min = 0;

    FloatM prismatic_max = 0;

    FloatM prismatic_position_control_kp = 0;

    FloatM prismatic_position_control_kd = 0;

    FloatM prismatic_max_speed = 0;

    FloatM prismatic_max_force = 0;

    FloatM prismatic_desired_force = 0;

    RIndex prismatic_signal_sid = RS_NULL_INDEX;

    /// Color for determining update order in PGS
    RIndex color = RS_NULL_INDEX;
    /**
     * Post-set attributes
     */
    RHandle rigid_body_a = RS_NULL_HANDLE;

    RHandle rigid_body_b = RS_NULL_HANDLE;

    /// absolute position, averaged from position of anchor A and B
    /// only valid for translation constraints
    Vec3M position = Vec3M::zero();

    /// absolute axis from body a local axis, only valid for hinge axis rotation
    Vec3M axis = Vec3M::zero();

    /// Angle relative to body a local axis, only valid for hinge axis limit
    FloatM angle = 0;

    /// Hinge motor impulse max and applied, only valid for hinge axis motor
    FloatM hinge_motor_max_impulse = 0;

    FloatM hinge_motor_target_impulse = 0;

    FloatM hinge_motor_impulse = 0;

    /// Distance along slide axis, only valid for prismatic axis limit
    FloatM prismatic_distance = 0;

    /// Prismatic motor impulse max and applied, only valid for prismatic axis motor
    FloatM prismatic_motor_max_impulse = 0;

    FloatM prismatic_motor_target_impulse = 0;

    FloatM prismatic_motor_impulse = 0;

    /**
     * Depending on the constraint type, constraint may use 1 ~ 3
     * Jacobian rows and the corresponding effective_mass,
     * accumulated_lambda and constraint_value dimensions.
     * Eg: use x and y of constraint func value, x and y of the
     * accumulated_lambda and the left upper 2x2 part of effective_mass
     */
    Vec3M rigid_body_a_linear_J_0 = Vec3M::zero();
    Vec3M rigid_body_a_angular_J_0 = Vec3M::zero();

    Vec3M rigid_body_a_linear_J_1 = Vec3M::zero();
    Vec3M rigid_body_a_angular_J_1 = Vec3M::zero();

    Vec3M rigid_body_a_linear_J_2 = Vec3M::zero();
    Vec3M rigid_body_a_angular_J_2 = Vec3M::zero();

    Vec3M rigid_body_b_linear_J_0 = Vec3M::zero();
    Vec3M rigid_body_b_angular_J_0 = Vec3M::zero();

    Vec3M rigid_body_b_linear_J_1 = Vec3M::zero();
    Vec3M rigid_body_b_angular_J_1 = Vec3M::zero();

    Vec3M rigid_body_b_linear_J_2 = Vec3M::zero();
    Vec3M rigid_body_b_angular_J_2 = Vec3M::zero();

    Mat3M effective_mass_inverse = Mat3M::zero();

    /// When multiplied with beta/delta_t, it is the restitution bias
    /// used for Baumgarte stabilization
    Vec3M constraint_value = Vec3M::zero();

    /// Accumulated lambda for warm-starting
    Vec3M accumulated_lambda = Vec3M::zero();

    /// PGS velocity-correction scale; 1 = hard, 0 = compliant; auto-adapts
    FloatM stiffness = 1;
};

}  // namespace logic

}  // namespace rise

#define REFLECT_CONSTRAINT_JOINT_MEM(constraint_joint_mem_type) \
    REFL_AUTO(                                                  \
        type(constraint_joint_mem_type),                        \
        field(state),                                           \
        field(structure_signal),                                \
        field(structure_gid),                                   \
        field(sid),                                             \
        field(joint_sid),                                       \
        field(motor_sid),                                       \
        field(rigid_body_a_sid),                                \
        field(rigid_body_b_sid),                                \
        field(rigid_body_a_anchor_rb_position),                 \
        field(rigid_body_b_anchor_rb_position),                 \
        field(hinge_a_rb_axis),                                 \
        field(hinge_b_rb_orthogonal_axis_1),                    \
        field(hinge_b_rb_orthogonal_axis_2),                    \
        field(hinge_min),                                       \
        field(hinge_max),                                       \
        field(hinge_position_control_kp),                       \
        field(hinge_position_control_kd),                       \
        field(hinge_max_speed),                                 \
        field(hinge_max_torque),                                \
        field(hinge_desired_torque),                            \
        field(hinge_signal_sid),                                \
        field(prismatic_a_rb_axis),                             \
        field(prismatic_b_rb_orthogonal_axis_1),                \
        field(prismatic_b_rb_orthogonal_axis_2),                \
        field(prismatic_min),                                   \
        field(prismatic_max),                                   \
        field(prismatic_position_control_kp),                   \
        field(prismatic_position_control_kd),                   \
        field(prismatic_max_speed),                             \
        field(prismatic_max_force),                             \
        field(prismatic_desired_force),                         \
        field(prismatic_signal_sid),                            \
        field(rigid_body_a),                                    \
        field(rigid_body_b),                                    \
        field(color),                                           \
        field(position),                                        \
        field(axis),                                            \
        field(angle),                                           \
        field(hinge_motor_max_impulse),                         \
        field(hinge_motor_target_impulse),                      \
        field(hinge_motor_impulse),                             \
        field(prismatic_distance),                              \
        field(prismatic_motor_max_impulse),                     \
        field(prismatic_motor_target_impulse),                  \
        field(prismatic_motor_impulse),                         \
        field(rigid_body_a_linear_J_0),                         \
        field(rigid_body_a_angular_J_0),                        \
        field(rigid_body_a_linear_J_1),                         \
        field(rigid_body_a_angular_J_1),                        \
        field(rigid_body_a_linear_J_2),                         \
        field(rigid_body_a_angular_J_2),                        \
        field(rigid_body_b_linear_J_0),                         \
        field(rigid_body_b_angular_J_0),                        \
        field(rigid_body_b_linear_J_1),                         \
        field(rigid_body_b_angular_J_1),                        \
        field(rigid_body_b_linear_J_2),                         \
        field(rigid_body_b_angular_J_2),                        \
        field(effective_mass_inverse),                          \
        field(constraint_value),                                \
        field(accumulated_lambda),                              \
        field(stiffness)                                        \
    )

REFLECT_CONSTRAINT_JOINT_MEM(rise::logic::RS_ConstraintJointMem<rise::RSE_FLOAT_32>)
REFLECT_CONSTRAINT_JOINT_MEM(rise::logic::RS_ConstraintJointMem<rise::RSE_FLOAT_64>)

#endif  // RS_CONSTRAINT_M_H
