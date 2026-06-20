// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_structure.h"

#include <map>
#include <set>
#include <unordered_map>

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rise_origin/utils/string/rs_string.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_StructureVoxelBody::operator==(const RSC_StructureVoxelBody &other) const {
    return RSC_Base::operator==(other) &&
           relative_origin_position == other.relative_origin_position &&
           relative_orientation == other.relative_orientation && x_voxels == other.x_voxels &&
           y_voxels == other.y_voxels && z_voxels == other.z_voxels &&
           material_reference_sid == other.material_reference_sid &&
           segment_bid == other.segment_bid && expansion_signal_sid == other.expansion_signal_sid &&
           expansion_control_mode == other.expansion_control_mode && link == other.link &&
           voxel_size == other.voxel_size;
}

void RSC_StructureVoxelBody::_validate() const {
    RSC_EXPECT_GT(RSC_StructureVoxelBody, x_voxels, 0);
    RSC_EXPECT_GT(RSC_StructureVoxelBody, y_voxels, 0);
    RSC_EXPECT_GT(RSC_StructureVoxelBody, z_voxels, 0);

    RSize voxel_num = x_voxels * y_voxels * z_voxels;
    if (material_reference_sid.size() != voxel_num) {
        RS_THROW(RS_InvalidCharacterizationException(
            "RSC_StructureVoxelBody material_reference_sid size mismatch"
        ));
    }
    if (segment_bid.size() != voxel_num) {
        RS_THROW(
            RS_InvalidCharacterizationException("RSC_StructureVoxelBody segment_bid size mismatch")
        );
    }
    if (!expansion_signal_sid.empty() && expansion_signal_sid.size() != voxel_num) {
        RS_THROW(RS_InvalidCharacterizationException(
            "RSC_StructureVoxelBody expansion_signal_sid must be empty or match voxel count"
        ));
    }
    if (!expansion_control_mode.empty() && expansion_control_mode.size() != voxel_num) {
        RS_THROW(RS_InvalidCharacterizationException(
            "RSC_StructureVoxelBody expansion_control_mode must be empty or match voxel count"
        ));
    }
    if (!link.empty() && link.size() != voxel_num) {
        RS_THROW(RS_InvalidCharacterizationException(
            "RSC_StructureVoxelBody link must be empty or match voxel count"
        ));
    }
    RSC_EXPECT_GT(RSC_StructureVoxelBody, voxel_size, 0);
}

void RSC_StructureVoxelBody::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    relative_origin_position = readVec3rf(rsc_tree.get<string>("RelativeOriginPosition"));
    relative_orientation = readQuat3rf(rsc_tree.get<string>("RelativeOrientation"));
    relative_orientation = relative_orientation.normalized();

    x_voxels = rsc_tree.get<RSize>("X_Voxels");
    y_voxels = rsc_tree.get<RSize>("Y_Voxels");
    z_voxels = rsc_tree.get<RSize>("Z_Voxels");

    RSize voxel_num = x_voxels * y_voxels * z_voxels;
    RSize voxel_per_layer_num = x_voxels * y_voxels;

    readIndexLayersFromXML(
        rsc_tree,
        "MaterialReferenceSID",
        voxel_num,
        voxel_per_layer_num,
        material_reference_sid,
        true
    );
    readIndexLayersFromXML(
        rsc_tree, "SegmentBID", voxel_num, voxel_per_layer_num, segment_bid, true
    );
    readIndexLayersFromXML(
        rsc_tree,
        "ExpansionSignalSID",
        voxel_num,
        voxel_per_layer_num,
        expansion_signal_sid,
        false
    );
    readEnumLayersFromXML(
        rsc_tree,
        "ExpansionControlMode",
        voxel_num,
        voxel_per_layer_num,
        expansion_control_mode,
        false
    );
    readIndexLayersFromXML(rsc_tree, "Link", voxel_num, voxel_per_layer_num, link, false);

    voxel_size = rsc_tree.get<RFloat>("VoxelSize", 0);
}

bool RSC_StructureVoxelConstraint::operator==(const RSC_StructureVoxelConstraint &other) const {
    return RSC_Base::operator==(other) && type == other.type &&
           is_collision_enabled == other.is_collision_enabled && a_body_sid == other.a_body_sid &&
           a_segment_bid == other.a_segment_bid && b_body_sid == other.b_body_sid &&
           b_segment_bid == other.b_segment_bid && relative_anchor == other.relative_anchor &&
           hinge_control_mode == other.hinge_control_mode &&
           hinge_relative_axis == other.hinge_relative_axis && hinge_min == other.hinge_min &&
           hinge_max == other.hinge_max &&
           hinge_position_control_kp == other.hinge_position_control_kp &&
           hinge_position_control_kd == other.hinge_position_control_kd &&
           hinge_max_speed == other.hinge_max_speed && hinge_max_torque == other.hinge_max_torque &&
           hinge_signal_sid == other.hinge_signal_sid &&
           prismatic_control_mode == other.prismatic_control_mode &&
           prismatic_relative_axis == other.prismatic_relative_axis &&
           prismatic_min == other.prismatic_min && prismatic_max == other.prismatic_max &&
           prismatic_position_control_kp == other.prismatic_position_control_kp &&
           prismatic_position_control_kd == other.prismatic_position_control_kd &&
           prismatic_max_speed == other.prismatic_max_speed &&
           prismatic_max_force == other.prismatic_max_force &&
           prismatic_signal_sid == other.prismatic_signal_sid;
}

void RSC_StructureVoxelConstraint::_validate() const {
    RSC_EXPECT_NOT_NULL(RSC_StructureVoxelConstraint, a_body_sid, RS_NULL_INDEX);
    RSC_EXPECT_NOT_NULL(RSC_StructureVoxelConstraint, a_segment_bid, RS_NULL_INDEX);
    RSC_EXPECT_NOT_NULL(RSC_StructureVoxelConstraint, b_body_sid, RS_NULL_INDEX);
    RSC_EXPECT_NOT_NULL(RSC_StructureVoxelConstraint, b_segment_bid, RS_NULL_INDEX);
    if (type == RSE_STRUCTURE_VOXEL_HINGE_JOINT) {
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, hinge_control_mode, 0);
        RSC_EXPECT_LE(RSC_StructureVoxelConstraint, hinge_control_mode, 3);
        RSC_EXPECT_CLOSE(RSC_StructureVoxelConstraint, hinge_relative_axis.length(), 1.0f, 1e-3);
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, hinge_min, -RS_PI);
        RSC_EXPECT_LE(RSC_StructureVoxelConstraint, hinge_max, RS_PI);
        RSC_EXPECT_GT(RSC_StructureVoxelConstraint, hinge_max, hinge_min);
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, hinge_position_control_kp, 0);
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, hinge_position_control_kd, 0);
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, hinge_max_speed, 0);
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, hinge_max_torque, 0);
    } else if (type == RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT) {
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, prismatic_control_mode, 0);
        RSC_EXPECT_LE(RSC_StructureVoxelConstraint, prismatic_control_mode, 3);
        RSC_EXPECT_CLOSE(
            RSC_StructureVoxelConstraint, prismatic_relative_axis.length(), 1.0f, 1e-3
        );
        RSC_EXPECT_GT(RSC_StructureVoxelConstraint, prismatic_max, prismatic_min);
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, prismatic_position_control_kp, 0);
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, prismatic_position_control_kd, 0);
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, prismatic_max_speed, 0);
        RSC_EXPECT_GE(RSC_StructureVoxelConstraint, prismatic_max_force, 0);
    }
}

void RSC_StructureVoxelConstraint::_read(const RS_CharacterizationPropertyTree &constraint_tree) {
    readEnum(constraint_tree.get<string>("Type"), type, "Type");
    is_collision_enabled = constraint_tree.get<bool>("IsCollisionEnabled", is_collision_enabled);
    a_body_sid = constraint_tree.get<RIndex>("RigidBodyA.BodySID");
    a_segment_bid = constraint_tree.get<RIndex>("RigidBodyA.SegmentBID");
    b_body_sid = constraint_tree.get<RIndex>("RigidBodyB.BodySID");
    b_segment_bid = constraint_tree.get<RIndex>("RigidBodyB.SegmentBID");
    relative_anchor = readVec3rf(constraint_tree.get<string>("RelativeAnchor"));
    if (type == RSE_STRUCTURE_VOXEL_HINGE_JOINT) {
        auto hinge_tree = constraint_tree.getChild("Hinge");
        readEnum(hinge_tree.get<string>("ControlMode"), hinge_control_mode, "Hinge.ControlMode");
        hinge_relative_axis = readVec3rf(hinge_tree.get<string>("RelativeAxis")).normalized();
        hinge_min = hinge_tree.get<RFloat>("Min", -RS_PI);
        hinge_max = hinge_tree.get<RFloat>("Max", RS_PI);
        hinge_position_control_kp = hinge_tree.get<RFloat>("Kp", 0);
        hinge_position_control_kd = hinge_tree.get<RFloat>("Kd", 0);
        hinge_max_speed = hinge_tree.get<RFloat>("MaxSpeed", 10);
        hinge_max_torque = hinge_tree.get<RFloat>("MaxTorque", 0);
        hinge_signal_sid = hinge_tree.get<RIndex>("SignalSID", RS_NULL_INDEX);
    } else if (type == RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT) {
        auto prismatic_tree = constraint_tree.getChild("Prismatic");
        readEnum(
            prismatic_tree.get<string>("ControlMode"),
            prismatic_control_mode,
            "Prismatic.ControlMode"
        );
        prismatic_relative_axis =
            readVec3rf(prismatic_tree.get<string>("RelativeAxis")).normalized();
        prismatic_min = prismatic_tree.get<RFloat>("Min", -1.0);
        prismatic_max = prismatic_tree.get<RFloat>("Max", 1.0);
        prismatic_position_control_kp = prismatic_tree.get<RFloat>("Kp", 0);
        prismatic_position_control_kd = prismatic_tree.get<RFloat>("Kd", 0);
        prismatic_max_speed = prismatic_tree.get<RFloat>("MaxSpeed", 10);
        prismatic_max_force = prismatic_tree.get<RFloat>("MaxForce", 0);
        prismatic_signal_sid = prismatic_tree.get<RIndex>("SignalSID", RS_NULL_INDEX);
    }
}

bool RSC_StructureVoxelRecord::operator==(const RSC_StructureVoxelRecord &other) const {
    return RSC_RecordBase::operator==(other) && voxel_record_range == other.voxel_record_range &&
           link_record_range == other.link_record_range &&
           rigid_body_record_range == other.rigid_body_record_range &&
           joint_record_range == other.joint_record_range &&
           voxel_sample_num == other.voxel_sample_num && link_sample_num == other.link_sample_num;
}

void RSC_StructureVoxelRecord::_validate() const {
    RSC_RecordBase::_validate();
    RSC_EXPECT_EQ(RSC_StructureVoxelRecord, name, "default");
}

void RSC_StructureVoxelRecord::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_RecordBase::_read(rsc_tree);
    name = "default";
    readEnum(rsc_tree.get<string>("VoxelRecordRange"), voxel_record_range, "VoxelRecordRange");
    readEnum(rsc_tree.get<string>("LinkRecordRange"), link_record_range, "LinkRecordRange");
    readEnum(
        rsc_tree.get<string>("RigidBodyRecordRange"),
        rigid_body_record_range,
        "RigidBodyRecordRange"
    );
    readEnum(rsc_tree.get<string>("JointRecordRange"), joint_record_range, "JointRecordRange");
    voxel_sample_num = rsc_tree.get<RSize>("VoxelSampleNum", voxel_sample_num);
    link_sample_num = rsc_tree.get<RSize>("LinkSampleNum", link_sample_num);
}

bool RSC_StructureVoxelSignal::operator==(const RSC_StructureVoxelSignal &other) const {
    return RSC_SignalBase::operator==(other) &&
           expansion_signal_num == other.expansion_signal_num &&
           hinge_signal_num == other.hinge_signal_num &&
           prismatic_signal_num == other.prismatic_signal_num;
}

void RSC_StructureVoxelSignal::_validate() const {
    RSC_SignalBase::_validate();
    RSC_EXPECT_EQ(RSC_StructureVoxelSignal, name, "default");
}

void RSC_StructureVoxelSignal::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_SignalBase::_read(rsc_tree);
    name = "default";
    expansion_signal_num = rsc_tree.get("ExpansionSignalNum", expansion_signal_num);
    hinge_signal_num = rsc_tree.get("HingeSignalNum", hinge_signal_num);
    prismatic_signal_num = rsc_tree.get("PrismaticSignalNum", prismatic_signal_num);
}

static void _validateStructureVoxelMaterialSIDExist(const RSC_StructureVoxel &s) {
    for (auto &&[body_sid, body_rsc] : enumerate(s.bodies)) {
        for (auto m : body_rsc.material_reference_sid) {
            if (m != RS_NULL_INDEX && m >= s.material_references.size()) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Material SID {:d} referenced in body with SID {:d} doesn't exist", m, body_sid
                ));
            }
        }
    }
}

static void _validateStructureVoxelNullSegment(const RSC_StructureVoxel &s) {
    for (auto &&[body_sid, body_rsc] : enumerate(s.bodies)) {
        RSize voxel_num = body_rsc.x_voxels * body_rsc.y_voxels * body_rsc.z_voxels;
        for (RSize i = 0; i < voxel_num; i++) {
            RIndex seg = body_rsc.segment_bid[i];
            RIndex mat = body_rsc.material_reference_sid[i];
            if (mat == RS_NULL_INDEX && seg != RS_NULL_INDEX) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Voxel {:d} in body with SID {:d} is empty (null material) "
                    "but has non-null segment BID {:d}",
                    i,
                    body_sid,
                    seg
                ));
            }
        }
    }
}

static void _validateStructureVoxelLinkExistInAdjacentVoxels(const RSC_StructureVoxel &s) {
    static int link_to_x_offset[] = {1, -1, 0, 0, 0, 0};
    static int link_to_y_offset[] = {0, 0, 1, -1, 0, 0};
    static int link_to_z_offset[] = {0, 0, 0, 0, 1, -1};
    static int opposite_link_direction[] = {1, 0, 3, 2, 5, 4};

    for (auto &&[body_sid, body_rsc] : enumerate(s.bodies)) {
        if (body_rsc.link.empty()) continue;
        RSize layer_size = body_rsc.y_voxels * body_rsc.x_voxels;
        for (int z = 0; z < (int)body_rsc.z_voxels; z++) {
            for (int y = 0; y < (int)body_rsc.y_voxels; y++) {
                for (int x = 0; x < (int)body_rsc.x_voxels; x++) {
                    RSize idx = z * layer_size + y * body_rsc.x_voxels + x;
                    for (int l = 0; l < 6; l++) {
                        bool has_link = body_rsc.link[idx] & (1 << l);
                        if (has_link) continue;
                        int neighbor_x = x + link_to_x_offset[l];
                        int neighbor_y = y + link_to_y_offset[l];
                        int neighbor_z = z + link_to_z_offset[l];
                        if (neighbor_x >= 0 && neighbor_x < (int)body_rsc.x_voxels &&
                            neighbor_y >= 0 && neighbor_y < (int)body_rsc.y_voxels &&
                            neighbor_z >= 0 && neighbor_z < (int)body_rsc.z_voxels) {
                            RSize neighbor_idx = neighbor_z * layer_size +
                                                 neighbor_y * body_rsc.x_voxels + neighbor_x;

                            bool neighbor_has_link =
                                body_rsc.link[neighbor_idx] & (1 << opposite_link_direction[l]);
                            if (neighbor_has_link) {
                                RS_THROW(RS_InvalidCharacterizationException(
                                    "Voxel {:d} at layer {:d} has inconsistent link "
                                    "with neighbor voxel {:d} at layer {:d}, in body "
                                    "with SID {:d}",
                                    y * body_rsc.x_voxels + x,
                                    z,
                                    neighbor_y * body_rsc.x_voxels + neighbor_x,
                                    neighbor_z,
                                    body_sid
                                ));
                            } else if (body_rsc.segment_bid[idx] ==
                                       body_rsc.segment_bid[neighbor_idx]) {
                                RS_THROW(RS_InvalidCharacterizationException(
                                    "Voxel {:d} at layer {:d} is not connected "
                                    "with neighbor voxel {:d} at layer {:d}, "
                                    "but belong to the same segment in body "
                                    "with SID {:d}",
                                    y * body_rsc.x_voxels + x,
                                    z,
                                    neighbor_y * body_rsc.x_voxels + neighbor_x,
                                    neighbor_z,
                                    body_sid
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}

static void _validateStructureVoxelConstraint(const RSC_StructureVoxel &s) {
    for (auto &&[idx, constraint_rsc] : enumerate(s.constraints)) {
        map<RIndex, set<RIndex>> body_segment_ids;
        for (auto &&[body_sid, body_rsc] : enumerate(s.bodies)) {
            auto &bsi = body_segment_ids[body_sid];
            for (auto seg_id : body_rsc.segment_bid) {
                bsi.insert(seg_id);
            }
        }
        if (constraint_rsc.a_body_sid > s.bodies.size()) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Body A with SID {:d} referenced in constraint {:d} doesn't exist",
                constraint_rsc.a_body_sid,
                idx
            ));
        }
        if (constraint_rsc.b_body_sid > s.bodies.size()) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Body B with SID {:d} referenced in constraint {:d} doesn't exist",
                constraint_rsc.b_body_sid,
                idx
            ));
        }
        auto &a_bsi = body_segment_ids[constraint_rsc.a_body_sid];
        auto &b_bsi = body_segment_ids[constraint_rsc.b_body_sid];
        if (a_bsi.find(constraint_rsc.a_segment_bid) == a_bsi.end()) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Segment with BID {:d} from Body A with SID {:d} referenced in "
                "constraint {:d} doesn't exist",
                constraint_rsc.a_segment_bid,
                constraint_rsc.a_body_sid,
                idx
            ));
        }
        if (b_bsi.find(constraint_rsc.b_segment_bid) == b_bsi.end()) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Segment with BID {:d} from Body B with SID {:d} referenced in "
                "constraint {:d} doesn't exist",
                constraint_rsc.b_segment_bid,
                constraint_rsc.b_body_sid,
                idx
            ));
        }
        if (constraint_rsc.type == RSE_STRUCTURE_VOXEL_HINGE_JOINT) {
            if (constraint_rsc.hinge_max <= constraint_rsc.hinge_min) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Hinge min value must be smaller than hinge max value in constraint {:d}", idx
                ));
            }
        } else if (constraint_rsc.type == RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT) {
            if (constraint_rsc.prismatic_max <= constraint_rsc.prismatic_min) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Prismatic min value must be smaller than prismatic max value in constraint "
                    "{:d}",
                    idx
                ));
            }
        }
    }
}

static void _validateStructureVoxelExpansionSignal(const RSC_StructureVoxel &s) {
    for (auto &&[body_sid, body_rsc] : enumerate(s.bodies)) {
        for (auto e : body_rsc.expansion_signal_sid) {
            if (e != RS_NULL_INDEX && e >= s.signal.expansion_signal_num) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Expansion signal SID {:d} referenced in body with SID {:d} doesn't exist",
                    e,
                    body_sid
                ));
            }
        }
    }
}

static void _validateStructureVoxelConstraintSignals(const RSC_StructureVoxel &s) {
    for (auto &&[idx, c] : enumerate(s.constraints)) {
        if (c.hinge_signal_sid != RS_NULL_INDEX &&
            c.hinge_signal_sid >= s.signal.hinge_signal_num) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Hinge signal SID {:d} referenced in constraint with SID {:d} doesn't exist",
                c.hinge_signal_sid,
                idx
            ));
        }
        if (c.prismatic_signal_sid != RS_NULL_INDEX &&
            c.prismatic_signal_sid >= s.signal.prismatic_signal_num) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Prismatic signal SID {:d} referenced in constraint with SID {:d} doesn't exist",
                c.prismatic_signal_sid,
                idx
            ));
        }
    }
}

bool RSC_StructureVoxel::operator==(const RSC_StructureVoxel &other) const {
    return RSC_StructureBase::operator==(other) && is_fixed == other.is_fixed &&
           bodies == other.bodies && constraints == other.constraints && record == other.record &&
           signal == other.signal;
}

void RSC_StructureVoxel::_validate() const {
    RSC_EntityBase::_validate();
    for (auto &body : bodies) body.validate();
    for (auto &constraint : constraints) constraint.validate();

    _validateStructureVoxelMaterialSIDExist(*this);
    _validateStructureVoxelNullSegment(*this);
    _validateStructureVoxelExpansionSignal(*this);
    _validateStructureVoxelConstraintSignals(*this);
    _validateStructureVoxelLinkExistInAdjacentVoxels(*this);
    _validateStructureVoxelConstraint(*this);
}

void RSC_StructureVoxel::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_EntityBase::_read(rsc_tree);
    is_fixed = rsc_tree.get("IsFixed", is_fixed);
    origin_position = readVec3rf(rsc_tree.get<string>("OriginPosition"));
    orientation = readQuat3rf(rsc_tree.get<string>("Orientation"));
    for (auto &str : split(rsc_tree.get<string>("MaterialReferences"), ",")) {
        material_references.emplace_back(strip(str));
    }

    for (auto &child : rsc_tree.getChild("Bodies").getChildrenValues()) {
        RSC_StructureVoxelBody body;
        body.read(child, false);
        bodies.emplace_back(body);
    }
    if (rsc_tree.isChildExist("Constraints")) {
        for (auto &child : rsc_tree.getChild("Constraints").getChildrenValues()) {
            RSC_StructureVoxelConstraint constraint;
            constraint.read(child, false);
            constraints.emplace_back(constraint);
        }
    }
    if (rsc_tree.isChildExist("Record")) {
        record.read(rsc_tree.getChild("Record"), false);
    }
    if (rsc_tree.isChildExist("Signal")) {
        signal.read(rsc_tree.getChild("Signal"), false);
    }
}

}  // namespace rsc

}  // namespace rise
