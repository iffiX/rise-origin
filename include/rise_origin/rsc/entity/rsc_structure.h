// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_STRUCTURE_H
#define RSC_STRUCTURE_H

#include <string>
#include <vector>

#include "rise_origin/rsc/record/rsc_record_base.h"
#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/rsc/signal/rsc_signal_base.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_entity_base.h"

namespace rise {

namespace rsc {

/*****************************************************************
 * Structure base
 *****************************************************************/

class RSC_StructureBase : public RSC_EntityBase {
   public:
    /// Initial global scope origin position.
    RVec3rf origin_position = RVec3rf::zero();

    /// Initial global scope orientation.
    RQuat3rf orientation = RQuat3rf::unit();

    /// Dependent material references (names).
    std::vector<std::string> material_references;

    bool operator==(const RSC_StructureBase& other) const {
        return RSC_EntityBase::operator==(other) && origin_position == other.origin_position &&
               orientation == other.orientation && material_references == other.material_references;
    }

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_EntityBase::serialize(s);
        s("origin_position", origin_position);
        s("orientation", orientation);
        s("material_references", material_references);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override { (void)validator; }
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
};

/*****************************************************************
 * Voxel structure sub-characterizations (bodies, constraints)
 *****************************************************************/

RS_ENUM(
    RSE_StructureExpansionControlMode,
    RIndex,

    /**
     * @brief Basic position control mode.
     *
     * Signal value is the target expansion ratio, clamped to material [min, max].
     */
    (RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION, 0),

    /**
     * @brief Rate control mode.
     *
     * Signal value is d(expansion)/dt; integrated each step.
     */
    (RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE, 1)
)

class RSC_StructureVoxelBody : public RSC_Base {
   public:
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

    /// Array of expansion control modes (one per expansion signal). If empty, defaults to
    /// RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION for each expansion signal.
    std::vector<RSE_StructureExpansionControlMode> expansion_control_mode;

    /// Array of link specification, has value 0~63, the bit correspondence is:
    /// 0 (lowest bit): X_POS, 1: X_NEG, 2: Y_POS, 3: Y_NEG, 4: Z_POS, 5(highest bit): Z_NEG
    /// If not specified, all voxels will connect 6 faces when adjacent voxels are present.
    std::vector<RIndex> link;

    /// Voxel size (unit: m).
    RFloat voxel_size = 0;

    bool operator==(const RSC_StructureVoxelBody& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
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
        s("voxel_size", voxel_size);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }

   private:
    void _validateNullSegment(RS_RSCValidator& validator) const;
    void _validateLinkAdjacency(RS_RSCValidator& validator) const;
};

RS_ENUM(
    RSE_StructureVoxelConstraintType,
    RIndex,
    (RSE_STRUCTURE_VOXEL_BALL_AND_SOCKET_JOINT, 0),
    (RSE_STRUCTURE_VOXEL_FIXED_JOINT, 1),
    (RSE_STRUCTURE_VOXEL_HINGE_JOINT, 2),
    (RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT, 3),
    (RSE_STRUCTURE_VOXEL_NULL_JOINT, 4)
)

RS_ENUM(
    RSE_StructureVoxelHingeControlMode,
    RIndex,

    /**
     * @brief Basic position control mode.
     *
     * Drive a target speed that closes the hinge angle error. Bounded by the safe
     * torque-and-speed envelope: at most max torque, at most max speed.
     */
    (RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION, 0),

    /**
     * @brief Position control mode with proportional-derivative (PD) control.
     *
     * T = kp*(p_destination - p_actual) + kd * (v_destination - v_actual), where
     * v_destination is 0. Applied as a stable torque, bounded by max torque only.
     */
    (RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION, 1),

    /**
     * @brief Torque control mode.
     *
     * Interpret the hinge signal as a torque command. Bounded by the safe
     * torque-and-speed envelope: applies that torque, at most max speed.
     */
    (RSE_STRUCTURE_VOXEL_HINGE_CONTROL_TORQUE, 2),

    /**
     * @brief Velocity control mode.
     *
     * Interpret the hinge signal as a target angular velocity, for wheels. Bounded by
     * the safe torque-and-speed envelope: at most max torque, at most max speed.
     */
    (RSE_STRUCTURE_VOXEL_HINGE_CONTROL_VELOCITY, 3)
)

RS_ENUM(
    RSE_StructureVoxelPrismaticControlMode,
    RIndex,

    /**
     * @brief Basic position control mode.
     *
     * Drive a target speed that closes the prismatic distance error. Bounded by the safe
     * force-and-speed envelope: at most max force, at most max speed.
     */
    (RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION, 0),

    /**
     * @brief Position control mode with proportional-derivative (PD) control.
     *
     * F = kp*(p_destination - p_actual) + kd * (v_destination - v_actual), where
     * v_destination is 0. Applied as a stable force, bounded by max force only.
     */
    (RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION, 1),

    /**
     * @brief Force control mode.
     *
     * Interpret the prismatic signal as a force command. Bounded by the safe
     * force-and-speed envelope: applies that force, at most max speed.
     */
    (RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_FORCE, 2),

    /**
     * @brief Velocity control mode.
     *
     * Interpret the prismatic signal as a target slide velocity. Bounded by the safe
     * force-and-speed envelope: at most max force, at most max speed.
     */
    (RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_VELOCITY, 3)
)

class RSC_StructureVoxelConstraint : public RSC_Base {
   public:
    /// Constraint type.
    RSE_StructureVoxelConstraintType type = RSE_STRUCTURE_VOXEL_NULL_JOINT;

    /// Whether the two bodies joined by this constraint are allowed to collide with each other.
    bool is_collision_enabled = false;

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

    // Hinge joint fields

    /// Control mode of the hinge joint.
    RSE_StructureVoxelHingeControlMode hinge_control_mode =
        RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION;

    /// Structure scope hinge axis. Positive rotation follows the right-hand rule around this axis.
    RVec3rf hinge_relative_axis = RVec3rf::zero();

    /// Minimum angle allowed by hinge, relative to initial two rigid body positions (unit: rad).
    RFloat hinge_min = -RS_PI;

    /// Maximum angle allowed by hinge, relative to initial two rigid body positions (unit: rad).
    RFloat hinge_max = RS_PI;

    /// Proportional gain of the position control of the hinge joint.
    /// Only valid when hinge_control_mode is RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION.
    RFloat hinge_position_control_kp = 0;

    /// Derivative gain of the position control of the hinge joint.
    /// Only valid when hinge_control_mode is RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION.
    RFloat hinge_position_control_kd = 0;

    /// Maximum speed of the hinge joint (unit: rad/s).
    RFloat hinge_max_speed = 10;

    /// Maximum motor torque of hinge (unit: N-m).
    RFloat hinge_max_torque = 0;

    /// Structure scope signal ID applied on hinge, starting from 0.
    RIndex hinge_signal_sid = RS_NULL_INDEX;

    // Prismatic joint fields

    /// Control mode of the prismatic joint.
    RSE_StructureVoxelPrismaticControlMode prismatic_control_mode =
        RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION;

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
    /// RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION.
    RFloat prismatic_position_control_kp = 0;

    /// Derivative gain of the position control of the prismatic joint.
    /// Only valid when prismatic_control_mode is
    /// RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION.
    RFloat prismatic_position_control_kd = 0;

    /// Maximum speed of the prismatic joint (unit: m/s).
    RFloat prismatic_max_speed = 10;

    /// Maximum motor force of prismatic joint (unit: m/s).
    RFloat prismatic_max_force = 0;

    /// Structure scope signal ID applied on prismatic joint, starting from 0.
    RIndex prismatic_signal_sid = RS_NULL_INDEX;

    bool operator==(const RSC_StructureVoxelConstraint& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        s("type", type);
        s("is_collision_enabled", is_collision_enabled);
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
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/*****************************************************************
 * Per-voxel-structure embedded record/signal/controller
 *****************************************************************/

RS_ENUM(
    RSE_StructureVoxelRecordVoxelRange,
    RIndex,
    (RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_NONE, 0),
    (RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_ALL, 1),
    (RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_RANDOM, 2)
)

RS_ENUM(
    RSE_StructureVoxelRecordLinkRange,
    RIndex,
    (RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_NONE, 0),
    (RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_ALL, 1),
    (RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_RANDOM, 2)
)

RS_ENUM(
    RSE_StructureVoxelRecordRigidBodyRange,
    RIndex,
    (RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_NONE, 0),
    (RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_ALL, 1)
)

RS_ENUM(
    RSE_StructureVoxelRecordJointRange,
    RIndex,
    (RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_NONE, 0),
    (RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_ALL, 1)
)

/**
 * @brief Per-voxel-structure record characterization — describes which sub-data to sample.
 */
class RSC_StructureVoxelRecord : public RSC_RecordBase {
   public:
    RSC_StructureVoxelRecord() { name = "default"; }

    /// Range of voxels to record.
    RSE_StructureVoxelRecordVoxelRange voxel_record_range =
        RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_NONE;

    /// Range of links to record.
    RSE_StructureVoxelRecordLinkRange link_record_range =
        RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_NONE;

    /// Range of rigid bodies to record.
    RSE_StructureVoxelRecordRigidBodyRange rigid_body_record_range =
        RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_NONE;

    /// Range of joints to record.
    RSE_StructureVoxelRecordJointRange joint_record_range =
        RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_NONE;

    /// Number of voxels to sample. Only valid when random sampling is enabled.
    RSize voxel_sample_num = 0;

    /// Number of links to sample. Only valid when random sampling is enabled.
    RSize link_sample_num = 0;

    bool operator==(const RSC_StructureVoxelRecord& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_RecordBase::serialize(s);
        s("voxel_record_range", voxel_record_range);
        s("link_record_range", link_record_range);
        s("rigid_body_record_range", rigid_body_record_range);
        s("joint_record_range", joint_record_range);
        s("voxel_sample_num", voxel_sample_num);
        s("link_sample_num", link_sample_num);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Per-voxel-structure signal characterization — declares how many actuation signals
 * the structure expects from a controller.
 */
class RSC_StructureVoxelSignal : public RSC_SignalBase {
   public:
    RSC_StructureVoxelSignal() { name = "default"; }

    /// Number of expansion signals.
    RSize expansion_signal_num = 0;

    /// Number of hinge signals.
    RSize hinge_signal_num = 0;

    /// Number of prismatic signals.
    RSize prismatic_signal_num = 0;

    bool operator==(const RSC_StructureVoxelSignal& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_SignalBase::serialize(s);
        s("expansion_signal_num", expansion_signal_num);
        s("hinge_signal_num", hinge_signal_num);
        s("prismatic_signal_num", prismatic_signal_num);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/*****************************************************************
 * Voxel structure (concrete, inherits StructureBase)
 *****************************************************************/

class RSC_StructureVoxel : public RSC_StructureBase {
   public:
    /// If true, this structure is static — voxels never move.
    bool is_fixed = false;

    /// Bodies in the structure.
    RSCVec<RSC_StructureVoxelBody> bodies;

    /// Constraints in the structure.
    RSCVec<RSC_StructureVoxelConstraint> constraints;

    /// Structure record.
    RSCRef<RSC_StructureVoxelRecord> record = RSC_StructureVoxelRecord();

    /// Structure signal.
    RSCRef<RSC_StructureVoxelSignal> signal = RSC_StructureVoxelSignal();

    bool operator==(const RSC_StructureVoxel& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_StructureBase::serialize(s);
        s("is_static", is_fixed);
        s("bodies", bodies);
        s("constraints", constraints);
        s("record", record);
        s("signal", signal);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
    void _deepCopyShared(MemoRef memo) override;

   private:
    void _validateMaterialSIDExist(RS_RSCValidator& validator) const;
    void _validateExpansionSignals(RS_RSCValidator& validator) const;
    void _validateConstraints(RS_RSCValidator& validator) const;
    void _validateConstraintSignals(RS_RSCValidator& validator) const;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_STRUCTURE_H
