// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_STRUCTURE_CONFIG_H
#define RS_STRUCTURE_CONFIG_H

#include "rise_origin/core/rs_simulation_def.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_quat.h"
#include "rise_origin/utils/rs_vec.h"
#include "rs_base_config.h"

namespace rise {

RS_ENUM(
    RSE_StructureExpansionControlMode,
    RIndex,
    (RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION,
     0),  // Signal value is the target expansion ratio, clamped to material [min, max].
    (RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE,
     1)  // Signal value is d(expansion)/dt; integrated each step.
)

class RS_StructureBodyConfig : public RS_BaseConfig {
   public:
    /// Structure scope ID of the body, starting from 0.
    RIndex body_sid = RS_NULL_INDEX;

    /// Initial structure scope origin position relative to the structure origin.
    RVec3rf relative_origin_position = RVec3rf::zero();

    /// Initial structure scope relative orientation.
    RQuat3rf relative_orientation = RQuat3rf::unit();

    /// Voxels in each dimension X, max value is limited by MAX_VOXEL_DIM_SIZE
    RSize x_voxels = 0;

    /// Voxels in each dimension Y, max value is limited by MAX_VOXEL_DIM_SIZE
    RSize y_voxels = 0;

    /// Voxels in each dimension Z, max value is limited by MAX_VOXEL_DIM_SIZE
    RSize z_voxels = 0;

    // Note for vectors below:
    // Position in the array corresponds with the position in the 3D structure,
    // where the array is ordered: starting at (x0,x0,z0), proceeding to (xn,y0,z0),
    // next to (xn,yn,z0), and on to (xn,yn,zn)

    /// Array of material references structure scope IDs of each voxel,
    /// use RS_NULL_INDEX for an empty voxel.
    std::vector<RIndex> material_reference_sid;

    /// Array of segment body scope IDs, starting from 0, use RS_NULL_INDEX for unspecified
    /// values.
    std::vector<RIndex> segment_bid;

    /// Array of expansion signal structure scope IDs, starting from 0, use RS_NULL_INDEX for
    /// unspecified value. If not specified, all voxels have no expansion signal.
    std::vector<RIndex> expansion_signal_sid;

    /// Array of expansion control modes (one per signal sid). If empty, defaults to
    /// RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION for each expansion signal.
    std::vector<RSE_StructureExpansionControlMode> expansion_control_mode;

    /// Array of link specification, has value 0~63, the lowest bit (0) corresponds to X_POS
    /// and the highest bit (5) corresponds to Z_NEG.
    /// If not specified, all voxels will connect 6 faces when adjacent voxels are present.
    std::vector<RIndex> link;

    template <class Serializer>
    void serialize(Serializer &s) {
        s("body_sid", body_sid);
        s("relative_origin_position", relative_origin_position);
        s("relative_orientation", relative_orientation);
        s("x_voxels", x_voxels);
        s("y_voxels", y_voxels);
        s("z_voxels", z_voxels);
        s("material_reference_sid", material_reference_sid);
        s("segment_bid", segment_bid);
        s("expansion_signal_sid", expansion_signal_sid);
        s("expansion_control_mode", expansion_control_mode);
        s("link", link);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_StructureTendonConfig : public RS_BaseConfig {
   public:
    /// Structure scope ID of the used material reference.
    RIndex material_reference_sid = RS_NULL_INDEX;

    /// Structure scope ID of the first body in the structure.
    RIndex a_body_sid = RS_NULL_INDEX;

    /// X index of the connected voxel in the first body.
    RSignedIndex a_voxel_x = RS_NULL_SIGNED_INDEX;

    /// Y index of the connected voxel in the first body.
    RSignedIndex a_voxel_y = RS_NULL_SIGNED_INDEX;

    /// Z index of the connected voxel in the first body.
    RSignedIndex a_voxel_z = RS_NULL_SIGNED_INDEX;

    /// Connecting face of the voxel in the first connected body.
    RSE_LinkDirection a_voxel_face = RSE_X_POS;

    /// ID of the second body in the structure.
    RIndex b_body_sid = RS_NULL_INDEX;

    /// X index of the connected voxel in the second body.
    RSignedIndex b_voxel_x = RS_NULL_SIGNED_INDEX;

    /// Y index of the connected voxel in the second body.
    RSignedIndex b_voxel_y = RS_NULL_SIGNED_INDEX;

    /// Z index of the connected voxel in the second body.
    RSignedIndex b_voxel_z = RS_NULL_SIGNED_INDEX;

    /// Connecting face of the voxel in the second connected body.
    RSE_LinkDirection b_voxel_face = RSE_X_POS;

    template <class Serializer>
    void serialize(Serializer &s) {
        s("material_reference_sid", material_reference_sid);
        s("a_body_sid", a_body_sid);
        s("a_voxel_x", a_voxel_x);
        s("a_voxel_y", a_voxel_y);
        s("a_voxel_z", a_voxel_z);
        s("a_voxel_face", a_voxel_face);
        s("b_body_sid", b_body_sid);
        s("b_voxel_x", b_voxel_x);
        s("b_voxel_y", b_voxel_y);
        s("b_voxel_z", b_voxel_z);
        s("b_voxel_face", b_voxel_face);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

RS_ENUM(
    RSE_StructureConstraintType,
    RIndex,
    (RSE_BALL_AND_SOCKET_JOINT, 0),
    (RSE_FIXED_JOINT, 1),
    (RSE_HINGE_JOINT, 2),
    (RSE_PRISMATIC_JOINT, 3),
    (RSE_NULL_JOINT, 4)
)

RS_ENUM(
    RSE_StructureHingeJointControlMode,
    RIndex,
    (RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_BASIC_POSITION,
     0),  // Drive a target speed that closes the angle error. Bounded by the safe
          // torque-and-speed envelope: at most max torque, at most max speed.
    (RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_PD_POSITION,
     1),  // T = kp*(p_destination - p_actual) + kd * (v_destination - v_actual), where
          // v_destination is 0. Applied as a stable torque, bounded by max torque only.
    (RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_TORQUE,
     2),  // Interpret rotation angle signal as a torque command. Bounded by the safe
          // torque-and-speed envelope: applies that torque, at most max speed.
    (RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_VELOCITY,
     3)  // Interpret rotation angle signal as a target angular velocity, for wheels. Bounded by
         // the safe torque-and-speed envelope: at most max torque, at most max speed.
)

RS_ENUM(
    RSE_StructurePrismaticJointControlMode,
    RIndex,
    (RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_BASIC_POSITION,
     0),  // Drive a target speed that closes the distance error. Bounded by the safe
          // force-and-speed envelope: at most max force, at most max speed.
    (RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_PD_POSITION,
     1),  // F = kp*(p_destination - p_actual) + kd * (v_destination - v_actual), where
          // v_destination is 0. Applied as a stable force, bounded by max force only.
    (RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_FORCE,
     2),  // Interpret translation rate signal as a force command. Bounded by the safe
          // force-and-speed envelope: applies that force, at most max speed.
    (RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_VELOCITY,
     3)  // Interpret translation rate signal as a target slide velocity. Bounded by the safe
         // force-and-speed envelope: at most max force, at most max speed.
)

class RS_StructureConstraintConfig : public RS_BaseConfig {
   public:
    /// Constraint type.
    int type = RSE_NULL_JOINT;

    /// Structure scope ID of the first body in the structure.
    RIndex a_body_sid = RS_NULL_INDEX;

    /// Body scope ID of the constrained rigid body segment in the first body.
    RIndex a_segment_bid = RS_NULL_INDEX;

    /// Structure scope ID of the second body in the structure.
    RIndex b_body_sid = RS_NULL_INDEX;

    /// Body scope ID of the constrained rigid body segment in the second body.
    RIndex b_segment_bid = RS_NULL_INDEX;

    /// Structure scope anchor location of the joint.
    RVec3rf relative_anchor = RVec3rf::zero();

    // For hinge only
    /// Control mode of the hinge joint.
    int hinge_control_mode = RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_BASIC_POSITION;

    /// Structure scope hinge axis. Positive rotation follows the right-hand rule around this axis.
    RVec3rf hinge_relative_axis = RVec3rf::zero();

    /// Minimum angle allowed by hinge, relative to initial two rigid body positions (unit: rad).
    RFloat hinge_min = -RS_PI;

    /// Maximum angle allowed by hinge, relative to initial two rigid body positions (unit: rad).
    RFloat hinge_max = RS_PI;

    /// Proportional gain of the position control of the hinge joint.
    /// Only valid when hinge_control_mode is RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_PD_POSITION.
    RFloat hinge_position_control_kp = 0;

    /// Derivative gain of the position control of the hinge joint.
    /// Only valid when hinge_control_mode is RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_PD_POSITION.
    RFloat hinge_position_control_kd = 0;

    /// Maximum speed of the hinge joint (unit: rad/s).
    RFloat hinge_max_speed = 10;

    /// Maximum motor torque of hinge (unit: N-m).
    RFloat hinge_max_torque = 0;

    /// Structure scope signal ID applied on hinge, starting from 0.
    RIndex hinge_signal_sid = RS_NULL_INDEX;

    // For prismatic only
    /// Control mode of the prismatic joint.
    int prismatic_control_mode = RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_BASIC_POSITION;

    /// Structure scope prismatic axis. Translation is allowed along this axis.
    RVec3rf prismatic_relative_axis = RVec3rf::zero();

    /// Minimum distance allowed by prismatic joint, relative to initial two rigid body positions.
    /// (unit: m).
    RFloat prismatic_min = -1.0;

    /// Maximum distance allowed by prismatic joint, relative to initial two rigid body positions.
    /// (unit: m).
    RFloat prismatic_max = 1.0;

    /// Proportional gain of the position control of the prismatic joint.
    /// Only valid when prismatic_control_mode is
    /// RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_PD_POSITION.
    RFloat prismatic_position_control_kp = 0;

    /// Derivative gain of the position control of the prismatic joint.
    /// Only valid when prismatic_control_mode is
    /// RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_PD_POSITION.
    RFloat prismatic_position_control_kd = 0;

    /// Maximum speed of the prismatic joint (unit: m/s).
    RFloat prismatic_max_speed = 10;

    /// Maximum motor force of prismatic joint (unit: m/s).
    RFloat prismatic_max_force = 0;

    /// Structure scope signal ID applied on prismatic joint, starting from 0.
    RIndex prismatic_signal_sid = RS_NULL_INDEX;

    template <class Serializer>
    void serialize(Serializer &s) {
        s("type", type);
        s("a_body_sid", a_body_sid);
        s("a_segment_bid", a_segment_bid);
        s("b_body_sid", b_body_sid);
        s("b_segment_bid", b_segment_bid);
        s("relative_anchor", relative_anchor);
        s("hinge_control_mode", hinge_control_mode);
        s("hinge_relative_axis", hinge_relative_axis);
        s("hinge_min", hinge_min);
        s("hinge_max", hinge_max);
        s("hinge_position_control_kp", hinge_position_control_kp);
        s("hinge_position_control_kd", hinge_position_control_kd);
        s("hinge_max_speed", hinge_max_speed);
        s("hinge_max_torque", hinge_max_torque);
        s("hinge_signal_sid", hinge_signal_sid);
        s("prismatic_control_mode", prismatic_control_mode);
        s("prismatic_relative_axis", prismatic_relative_axis);
        s("prismatic_min", prismatic_min);
        s("prismatic_max", prismatic_max);
        s("prismatic_position_control_kp", prismatic_position_control_kp);
        s("prismatic_position_control_kd", prismatic_position_control_kd);
        s("prismatic_max_speed", prismatic_max_speed);
        s("prismatic_max_force", prismatic_max_force);
        s("prismatic_signal_sid", prismatic_signal_sid);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_StructureConfig : public RS_BaseConfig {
   public:
    /// Name of the structure.
    std::string name;

    /// Flag indicating whether voxels made of this structure is
    /// static and never changes position or orientation .
    bool is_fixed = false;

    /// Size of the voxel (unit: m).
    RFloat voxel_size = 0;

    /// Number of expansion signals.
    RSize expansion_signal_num = 0;

    /// Number of hinge signals.
    RSize hinge_signal_num = 0;

    /// Number of prismatic signals.
    RSize prismatic_signal_num = 0;

    /// Initial global scope origin position.
    RVec3rf origin_position = RVec3rf::zero();

    /// Initial global scope orientation.
    RQuat3rf orientation = RQuat3rf::unit();

    /// Dependent material references (names)
    std::vector<std::string> material_references;

    std::vector<RS_StructureBodyConfig> bodies;
    std::vector<RS_StructureTendonConfig> tendons;
    std::vector<RS_StructureConstraintConfig> constraints;

    template <class Serializer>
    void serialize(Serializer &s) {
        s("name", name);
        s("is_fixed", is_fixed);
        s("voxel_size", voxel_size);
        s("expansion_signal_num", expansion_signal_num);
        s("hinge_signal_num", hinge_signal_num);
        s("prismatic_signal_num", prismatic_signal_num);
        s("origin_position", origin_position);
        s("orientation", orientation);
        s("material_references", material_references);
        s("bodies", bodies);
        s("tendons", tendons);
        s("constraints", constraints);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;

   private:
    void _validateSIDMatchOrder() const;
    void _validateStructureMaterialSIDExist() const;
    void _validateStructureExpansionSignalSIDExist() const;
    void _validateStructureHingeSignalSIDExist() const;
    void _validateStructurePrismaticSignalSIDExist() const;
    void _validateStructureExpansionControlModeSize() const;
    void _validateStructureLinkExistInAdjacentVoxels() const;
    void _validateTendon() const;
    void _validateConstraint() const;
};

}  // namespace rise
#endif  // RS_STRUCTURE_CONFIG_H
