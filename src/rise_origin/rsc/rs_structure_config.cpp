// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_structure_config.h"

#include <filesystem>
#include <highfive/highfive.hpp>
#include <map>
#include <set>

#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rise_origin/utils/rs_string.h"
#include "rs_rsc_util.h"

using namespace std;
using namespace HighFive;

namespace rise {

void RS_StructureBodyConfig::_validate() const {
    RS_CONFIG_EXPECT_NE(RS_StructureBodyConfig, body_sid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_GT(RS_StructureBodyConfig, x_voxels, 0);
    RS_CONFIG_EXPECT_GT(RS_StructureBodyConfig, y_voxels, 0);
    RS_CONFIG_EXPECT_GT(RS_StructureBodyConfig, z_voxels, 0);

    if (material_reference_sid.size() != x_voxels * y_voxels * z_voxels) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid structure body config, material_reference_sid must be of the same size as the "
            "number of voxels"
        ));
    }
    if (segment_bid.size() != x_voxels * y_voxels * z_voxels) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid structure body config, segment_bid must be of the same size as the "
            "number of voxels"
        ));
    }
    if (!expansion_signal_sid.empty() &&
        expansion_signal_sid.size() != x_voxels * y_voxels * z_voxels) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid structure body config, expansion_signal_sid must be empty or of "
            "the same size as the number of voxels"
        ));
    }
    if (!expansion_control_mode.empty() &&
        expansion_control_mode.size() != x_voxels * y_voxels * z_voxels) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid structure body config, expansion_control_mode must be empty or of "
            "the same size as the number of voxels"
        ));
    }
    if (!link.empty() && link.size() != x_voxels * y_voxels * z_voxels) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid structure body config, link must be empty or of "
            "the same size as the number of voxels"
        ));
    }

    // Every voxel in a segment must share one material.
    map<RIndex, RIndex> segment_to_material;
    for (RSize i = 0; i < material_reference_sid.size(); i++) {
        RIndex material = material_reference_sid[i];
        RIndex segment = segment_bid[i];
        if (material == RS_NULL_INDEX || segment == RS_NULL_INDEX) continue;
        auto [it, inserted] = segment_to_material.try_emplace(segment, material);
        if (!inserted && it->second != material) {
            RS_THROW(RS_InvalidConfigurationException(
                "Invalid structure body config, segment {} contains more than one material; "
                "each segment must use a single material",
                segment
            ));
        }
    }
}

void RS_StructureBodyConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Structures(Structure[S]).Bodies(Body[B])
    body_sid = config_tree.get<RIndex>("BodySid");
    relative_origin_position = readVec3rf(config_tree.get<string>("RelativeOriginPosition"));
    relative_orientation = readQuat3rf(config_tree.get<string>("RelativeOrientation"));
    relative_orientation = relative_orientation.normalized();

    // Requires size to present, no default values
    x_voxels = config_tree.get<RSize>("XVoxels");
    y_voxels = config_tree.get<RSize>("YVoxels");
    z_voxels = config_tree.get<RSize>("ZVoxels");

    RSize voxel_num = x_voxels * y_voxels * z_voxels;
    RSize voxel_per_layer_num = x_voxels * y_voxels;

    if (config_tree.isChildExist("Hdf5File")) {
        auto file_path = config_tree.get<string>("Hdf5File");
        if (!filesystem::exists(file_path)) {
            RS_THROW(
                RS_InvalidConfigurationException("Referred HDF5 file {} doesn't exist", file_path)
            );
        } else {
            File hdf5_file(file_path, File::ReadOnly);
            readIndexLayersFromHDF5(
                hdf5_file,
                config_tree,
                "MaterialReferenceSid",
                voxel_num,
                material_reference_sid,
                true
            );

            readIndexLayersFromHDF5(
                hdf5_file, config_tree, "SegmentBid", voxel_num, segment_bid, true
            );
            readIndexLayersFromHDF5(
                hdf5_file,
                config_tree,
                "ExpansionSignalSid",
                voxel_num,
                expansion_signal_sid,
                false
            );
            readEnumLayersFromHDF5(
                hdf5_file,
                config_tree,
                "ExpansionControlMode",
                voxel_num,
                expansion_control_mode,
                false
            );
            readIndexLayersFromHDF5(hdf5_file, config_tree, "Link", voxel_num, link, false);
        }
    } else {
        readIndexLayersFromXML(
            config_tree,
            "MaterialReferenceSid",
            voxel_num,
            voxel_per_layer_num,
            material_reference_sid,
            true
        );
        readIndexLayersFromXML(
            config_tree, "SegmentBid", voxel_num, voxel_per_layer_num, segment_bid, true
        );
        readIndexLayersFromXML(
            config_tree,
            "ExpansionSignalSid",
            voxel_num,
            voxel_per_layer_num,
            expansion_signal_sid,
            false
        );
        readEnumLayersFromXML(
            config_tree,
            "ExpansionControlMode",
            voxel_num,
            voxel_per_layer_num,
            expansion_control_mode,
            false
        );
        readIndexLayersFromXML(config_tree, "Link", voxel_num, voxel_per_layer_num, link, false);
    }
}

void RS_StructureTendonConfig::_validate() const {
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureTendonConfig, material_reference_sid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureTendonConfig, a_body_sid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureTendonConfig, a_voxel_x, RS_NULL_SIGNED_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureTendonConfig, a_voxel_y, RS_NULL_SIGNED_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureTendonConfig, a_voxel_z, RS_NULL_SIGNED_INDEX);

    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureTendonConfig, b_body_sid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureTendonConfig, b_voxel_x, RS_NULL_SIGNED_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureTendonConfig, b_voxel_y, RS_NULL_SIGNED_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureTendonConfig, b_voxel_z, RS_NULL_SIGNED_INDEX);
}

void RS_StructureTendonConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Structures(Structure[S]).Tendons(Tendon[T])
    material_reference_sid = config_tree.get<RIndex>("MaterialReferenceSid");

    a_body_sid = config_tree.get<RIndex>("ABodySid");
    a_voxel_x = config_tree.get<uint16_t>("AVoxelX");
    a_voxel_y = config_tree.get<uint16_t>("AVoxelY");
    a_voxel_z = config_tree.get<uint16_t>("AVoxelZ");
    int a_voxel_face_tmp = config_tree.get<int>("AVoxelFace");
    if (a_voxel_face_tmp >= 6 || a_voxel_face_tmp < 0) {
        RS_THROW(RS_InvalidConfigurationException("Unknown voxel face {}", a_voxel_face_tmp));
    }
    a_voxel_face = static_cast<RSE_LinkDirection>(a_voxel_face_tmp);

    b_body_sid = config_tree.get<RIndex>("BBodySid");
    b_voxel_x = config_tree.get<uint16_t>("BVoxelX");
    b_voxel_y = config_tree.get<uint16_t>("BVoxelY");
    b_voxel_z = config_tree.get<uint16_t>("BVoxelZ");
    int b_voxel_face_tmp = config_tree.get<int>("BVoxelFace");
    if (b_voxel_face_tmp >= 6 || b_voxel_face_tmp < 0) {
        RS_THROW(RS_InvalidConfigurationException("Unknown voxel face {}", b_voxel_face_tmp));
    }
    b_voxel_face = static_cast<RSE_LinkDirection>(b_voxel_face_tmp);
}

void RS_StructureConstraintConfig::_validate() const {
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureConstraintConfig, a_body_sid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureConstraintConfig, a_segment_bid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureConstraintConfig, b_body_sid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_StructureConstraintConfig, b_segment_bid, RS_NULL_INDEX);
    // Fixed joint: no additional validation needed
    if (type == RSE_HINGE_JOINT) {
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, hinge_control_mode, 0);
        RS_CONFIG_EXPECT_LE(RS_StructureConstraintConfig, hinge_control_mode, 3);
        RS_CONFIG_EXPECT_CLOSE(
            RS_StructureConstraintConfig, hinge_relative_axis.length(), 1.0f, 1e-3
        );
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, hinge_min, -RS_PI);
        RS_CONFIG_EXPECT_LE(RS_StructureConstraintConfig, hinge_max, RS_PI);
        RS_CONFIG_EXPECT_GT(RS_StructureConstraintConfig, hinge_max, hinge_min);
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, hinge_position_control_kp, 0);
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, hinge_position_control_kd, 0);
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, hinge_max_speed, 0);
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, hinge_max_torque, 0);
    }
    if (type == RSE_PRISMATIC_JOINT) {
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, prismatic_control_mode, 0);
        RS_CONFIG_EXPECT_LE(RS_StructureConstraintConfig, prismatic_control_mode, 3);
        RS_CONFIG_EXPECT_CLOSE(
            RS_StructureConstraintConfig, prismatic_relative_axis.length(), 1.0f, 1e-3
        );
        RS_CONFIG_EXPECT_GT(RS_StructureConstraintConfig, prismatic_max, prismatic_min);
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, prismatic_position_control_kp, 0);
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, prismatic_position_control_kd, 0);
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, prismatic_max_speed, 0);
        RS_CONFIG_EXPECT_GE(RS_StructureConstraintConfig, prismatic_max_force, 0);
    }
}

void RS_StructureConstraintConfig::_read(const RS_ConfigPropertyTree &constraint_tree) {
    // root of tree: RSC.Structures.Structure[S].Constraints.Constraint[C]
    auto constraint_type = constraint_tree.get<string>("Type");
    a_body_sid = constraint_tree.get<RIndex>("ABodySid");
    a_segment_bid = constraint_tree.get<RIndex>("ASegmentBid");
    b_body_sid = constraint_tree.get<RIndex>("BBodySid");
    b_segment_bid = constraint_tree.get<RIndex>("BSegmentBid");
    relative_anchor = readVec3rf(constraint_tree.get<string>("RelativeAnchor"));
    if (constraint_type == "BallAndSocketJoint") {
        type = RSE_BALL_AND_SOCKET_JOINT;
    } else if (constraint_type == "FixedJoint") {
        type = RSE_FIXED_JOINT;
    } else if (constraint_type == "HingeJoint") {
        type = RSE_HINGE_JOINT;
        hinge_control_mode = constraint_tree.get<int>(
            "HingeControlMode", RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_BASIC_POSITION
        );
        hinge_relative_axis =
            readVec3rf(constraint_tree.get<string>("HingeRelativeAxis")).normalized();
        hinge_min = constraint_tree.get<RFloat>("HingeMin", -RS_PI);
        hinge_max = constraint_tree.get<RFloat>("HingeMax", RS_PI);
        hinge_position_control_kp = constraint_tree.get<RFloat>("HingePositionControlKp", 0);
        hinge_position_control_kd = constraint_tree.get<RFloat>("HingePositionControlKd", 0);
        hinge_max_speed = constraint_tree.get<RFloat>("HingeMaxSpeed", 10);
        hinge_max_torque = constraint_tree.get<RFloat>("HingeMaxTorque", 0);
        hinge_signal_sid = constraint_tree.get<RIndex>("HingeSignalSid", RS_NULL_INDEX);
    } else if (constraint_type == "PrismaticJoint") {
        type = RSE_PRISMATIC_JOINT;
        prismatic_control_mode = constraint_tree.get<int>(
            "PrismaticControlMode", RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_BASIC_POSITION
        );
        prismatic_relative_axis =
            readVec3rf(constraint_tree.get<string>("PrismaticRelativeAxis")).normalized();
        prismatic_min = constraint_tree.get<RFloat>("PrismaticMin", -1.0);
        prismatic_max = constraint_tree.get<RFloat>("PrismaticMax", 1.0);
        prismatic_position_control_kp =
            constraint_tree.get<RFloat>("PrismaticPositionControlKp", 0);
        prismatic_position_control_kd =
            constraint_tree.get<RFloat>("PrismaticPositionControlKd", 0);
        prismatic_max_speed = constraint_tree.get<RFloat>("PrismaticMaxSpeed", 10);
        prismatic_max_force = constraint_tree.get<RFloat>("PrismaticMaxForce", 0);
        prismatic_signal_sid = constraint_tree.get<RIndex>("PrismaticSignalSid", RS_NULL_INDEX);
    } else {
        RS_THROW(RS_InvalidConfigurationException("Unknown constraint type {}", constraint_type));
    }
}

void RS_StructureConfig::_validate() const {
    RS_CONFIG_EXPECT_GT(RS_StructureConfig, voxel_size, 0);
    RS_CONFIG_EXPECT_GE(RS_StructureConfig, expansion_signal_num, 0);
    RS_CONFIG_EXPECT_GE(RS_StructureConfig, hinge_signal_num, 0);
    RS_CONFIG_EXPECT_GE(RS_StructureConfig, prismatic_signal_num, 0);
    for (auto &body : bodies) {
        body.validate();
    }
    for (auto &tendon : tendons) {
        tendon.validate();
    }
    for (auto &constraint : constraints) {
        constraint.validate();
    }

    _validateSIDMatchOrder();
    _validateStructureMaterialSIDExist();
    _validateStructureExpansionSignalSIDExist();
    _validateStructureHingeSignalSIDExist();
    _validateStructurePrismaticSignalSIDExist();
    _validateStructureExpansionControlModeSize();
    _validateStructureLinkExistInAdjacentVoxels();
    _validateTendon();
    _validateConstraint();
}

void RS_StructureConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Structures(Structure[S])
    name = config_tree.get("Name", name);
    is_fixed = config_tree.get("IsFixed", is_fixed);
    voxel_size = config_tree.get<RFloat>("VoxelSize");
    expansion_signal_num = config_tree.get("ExpansionSignalNum", expansion_signal_num);
    hinge_signal_num = config_tree.get("HingeSignalNum", hinge_signal_num);
    prismatic_signal_num = config_tree.get("PrismaticSignalNum", prismatic_signal_num);
    origin_position = readVec3rf(config_tree.get<string>("OriginPosition"));
    orientation = readQuat3rf(config_tree.get<string>("Orientation"));
    for (auto &str : split(config_tree.get<string>("MaterialReferences"), ",")) {
        material_references.emplace_back(strip(str));
    }

    for (auto &child : config_tree.getChild("Bodies").getChildrenValues()) {
        auto body = RS_StructureBodyConfig();
        body.read(child, false);
        bodies.emplace_back(body);
    }
    if (config_tree.isChildExist("Tendons")) {
        for (auto &child : config_tree.getChild("Tendons").getChildrenValues()) {
            auto tendon = RS_StructureTendonConfig();
            tendon.read(child, false);
            tendons.emplace_back(tendon);
        }
    }
    if (config_tree.isChildExist("Constraints")) {
        for (auto &child : config_tree.getChild("Constraints").getChildrenValues()) {
            auto constraint = RS_StructureConstraintConfig();
            constraint.read(child, false);
            constraints.emplace_back(constraint);
        }
    }
}

void RS_StructureConfig::_validateSIDMatchOrder() const {
    for (auto &&[idx, body_config] : enumerate(bodies)) {
        if (body_config.body_sid != idx) {
            RS_THROW(RS_InvalidConfigurationException(
                "Body SID {:d} doesn't match body order index {:d}", body_config.body_sid, idx
            ));
        }
    }
}

void RS_StructureConfig::_validateStructureMaterialSIDExist() const {
    for (auto &body_config : bodies) {
        for (auto m : body_config.material_reference_sid) {
            if (m != RS_NULL_INDEX && m >= material_references.size()) {
                RS_THROW(RS_InvalidConfigurationException(
                    "Material SID {:d} referenced in body with SID {:d} doesn't exist",
                    m,
                    body_config.body_sid
                ));
            }
        }
    }
}

void RS_StructureConfig::_validateStructureExpansionSignalSIDExist() const {
    for (auto &body_config : bodies) {
        for (auto e : body_config.expansion_signal_sid) {
            if (e != RS_NULL_INDEX && e >= expansion_signal_num) {
                RS_THROW(RS_InvalidConfigurationException(
                    "Expansion signal SID {:d} referenced in body with SID {:d} doesn't exist",
                    e,
                    body_config.body_sid
                ));
            }
        }
    }
}

void RS_StructureConfig::_validateStructureHingeSignalSIDExist() const {
    for (auto &&[idx, constraint_config] : enumerate(constraints)) {
        RIndex r = constraint_config.hinge_signal_sid;
        if (r != RS_NULL_INDEX && r >= hinge_signal_num) {
            RS_THROW(RS_InvalidConfigurationException(
                "Hinge signal SID {:d} referenced in constraint with SID {:d} doesn't exist",
                r,
                idx
            ));
        }
    }
}

void RS_StructureConfig::_validateStructurePrismaticSignalSIDExist() const {
    for (auto &&[idx, constraint_config] : enumerate(constraints)) {
        RIndex r = constraint_config.prismatic_signal_sid;
        if (r != RS_NULL_INDEX && r >= prismatic_signal_num) {
            RS_THROW(RS_InvalidConfigurationException(
                "Prismatic signal SID {:d} referenced in constraint with SID {:d} doesn't exist",
                r,
                idx
            ));
        }
    }
}

void RS_StructureConfig::_validateStructureExpansionControlModeSize() const {
    for (auto &body_config : bodies) {
        if (body_config.expansion_control_mode.empty()) continue;
        if (body_config.expansion_signal_sid.empty()) {
            RS_THROW(RS_InvalidConfigurationException(
                "Body with SID {:d} has expansion_control_mode but no expansion_signal_sid",
                body_config.body_sid
            ));
        }
        if (body_config.expansion_control_mode.size() != body_config.expansion_signal_sid.size()) {
            RS_THROW(RS_InvalidConfigurationException(
                "Body with SID {:d} expansion_control_mode size {} does not match "
                "expansion_signal_sid size {}",
                body_config.body_sid,
                body_config.expansion_control_mode.size(),
                body_config.expansion_signal_sid.size()
            ));
        }
    }
}

void RS_StructureConfig::_validateStructureLinkExistInAdjacentVoxels() const {
    for (auto &body_config : bodies) {
        if (body_config.link.empty()) continue;
        RSize layer_size = body_config.y_voxels * body_config.x_voxels;
        for (int z = 0; z < body_config.z_voxels; z++) {
            for (int y = 0; y < body_config.y_voxels; y++) {
                for (int x = 0; x < body_config.x_voxels; x++) {
                    RSize idx = z * layer_size + y * body_config.x_voxels + x;
                    for (int l = 0; l < 6; l++) {
                        bool has_link = body_config.link[idx] & (1 << l);
                        // not having link is much less common
                        if (!has_link) {
                            int neighbor_x =
                                x + linkDirectionToXOffset(static_cast<RSE_LinkDirection>(l));
                            int neighbor_y =
                                y + linkDirectionToYOffset(static_cast<RSE_LinkDirection>(l));
                            int neighbor_z =
                                z + linkDirectionToZOffset(static_cast<RSE_LinkDirection>(l));
                            if (neighbor_x >= 0 && neighbor_x < body_config.x_voxels &&
                                neighbor_y >= 0 && neighbor_y < body_config.y_voxels &&
                                neighbor_z >= 0 && neighbor_z < body_config.z_voxels) {
                                RSize neighbor_idx = neighbor_z * layer_size +
                                                     neighbor_y * body_config.x_voxels + neighbor_x;

                                bool neighbor_has_link =
                                    body_config.link[neighbor_idx] &
                                    (1 << oppositeLinkDirection(static_cast<RSE_LinkDirection>(l)));
                                if (neighbor_has_link) {
                                    RS_THROW(RS_InvalidConfigurationException(
                                        "Voxel {:d} at layer {:d} has inconsistent link "
                                        "with neighbor voxel {:d} at layer {:d}, in body "
                                        "with SID {:d}",
                                        y * body_config.x_voxels + x,
                                        z,
                                        neighbor_y * body_config.x_voxels + neighbor_x,
                                        neighbor_z,
                                        body_config.body_sid
                                    ));
                                } else if (body_config.segment_bid[idx] ==
                                           body_config.segment_bid[neighbor_idx]) {
                                    RS_THROW(RS_InvalidConfigurationException(
                                        "Voxel {:d} at layer {:d} is not connected "
                                        "with neighbor voxel {:d} at layer {:d}, "
                                        "but belong to the same segment in body "
                                        "with SID {:d}",
                                        y * body_config.x_voxels + x,
                                        z,
                                        neighbor_y * body_config.x_voxels + neighbor_x,
                                        neighbor_z,
                                        body_config.body_sid
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void RS_StructureConfig::_validateTendon() const {
    for (auto &&[idx, tendon_config] : enumerate(tendons)) {
        if (tendon_config.a_body_sid > bodies.size()) {
            RS_THROW(RS_InvalidConfigurationException(
                "Body A with SID {:d} referenced in tendon {:d} doesn't exist",
                tendon_config.a_body_sid,
                idx
            ));
        }
        if (tendon_config.b_body_sid > bodies.size()) {
            RS_THROW(RS_InvalidConfigurationException(
                "Body B with SID {:d} referenced in tendon {:d} doesn't exist",
                tendon_config.b_body_sid,
                idx
            ));
        }
        auto &body_a_config = bodies[tendon_config.a_body_sid];
        if (tendon_config.a_voxel_x < 0 || tendon_config.a_voxel_x >= body_a_config.x_voxels ||
            tendon_config.a_voxel_y < 0 || tendon_config.a_voxel_y >= body_a_config.y_voxels ||
            tendon_config.a_voxel_z < 0 || tendon_config.a_voxel_z >= body_a_config.z_voxels) {
            RS_THROW(RS_InvalidConfigurationException(
                "Body A voxel [{},{},{}] referenced in tendon {:d} exceeds body "
                "boundary definition or has negative index",
                tendon_config.a_voxel_x,
                tendon_config.a_voxel_y,
                tendon_config.a_voxel_z,
                idx
            ));
        }
        auto &body_b_config = bodies[tendon_config.b_body_sid];
        if (tendon_config.b_voxel_x < 0 || tendon_config.b_voxel_x >= body_b_config.x_voxels ||
            tendon_config.b_voxel_y < 0 || tendon_config.b_voxel_y >= body_b_config.y_voxels ||
            tendon_config.b_voxel_z < 0 || tendon_config.b_voxel_z >= body_b_config.z_voxels) {
            RS_THROW(RS_InvalidConfigurationException(
                "Body B voxel [{},{},{}] referenced in tendon {:d} exceeds body "
                "boundary definition or has negative index",
                tendon_config.b_voxel_x,
                tendon_config.b_voxel_y,
                tendon_config.b_voxel_z,
                idx
            ));
        }
        // Verify that connected voxels are non-null and faces are free

        // For Body A:
        RSize layer_size_a = body_a_config.x_voxels * body_a_config.y_voxels;
        RSize idx_a = tendon_config.a_voxel_z * layer_size_a +
                      tendon_config.a_voxel_y * body_a_config.x_voxels + tendon_config.a_voxel_x;
        // Check that the voxel is not empty.
        if (body_a_config.material_reference_sid[idx_a] == RS_NULL_INDEX) {
            RS_THROW(RS_InvalidConfigurationException(
                "Voxel [{},{},{}] in body A referenced in tendon {:d} is null",
                tendon_config.a_voxel_x,
                tendon_config.a_voxel_y,
                tendon_config.a_voxel_z,
                idx
            ));
        }
        // Check that the specified face is free.
        if (!body_a_config.link.empty()) {
            // When an explicit link vector is provided, the bit for the face should not be set.
            if (body_a_config.link[idx_a] & (1 << tendon_config.a_voxel_face)) {
                RS_THROW(RS_InvalidConfigurationException(
                    "Voxel [{},{},{}] in body A referenced in tendon {} has face {} connected",
                    tendon_config.a_voxel_x,
                    tendon_config.a_voxel_y,
                    tendon_config.a_voxel_z,
                    idx,
                    enumToString(tendon_config.a_voxel_face)
                ));
            }
        } else {
            // Otherwise, infer connectivity by checking if there is an adjacent voxel.
            int neighbor_ax =
                tendon_config.a_voxel_x + linkDirectionToXOffset(tendon_config.a_voxel_face);
            int neighbor_ay =
                tendon_config.a_voxel_y + linkDirectionToYOffset(tendon_config.a_voxel_face);
            int neighbor_az =
                tendon_config.a_voxel_z + linkDirectionToZOffset(tendon_config.a_voxel_face);
            if (neighbor_ax >= 0 && neighbor_ax < body_a_config.x_voxels && neighbor_ay >= 0 &&
                neighbor_ay < body_a_config.y_voxels && neighbor_az >= 0 &&
                neighbor_az < body_a_config.z_voxels) {
                RSize neighbor_idx =
                    neighbor_az * layer_size_a + neighbor_ay * body_a_config.x_voxels + neighbor_ax;
                if (body_a_config.material_reference_sid[neighbor_idx] != RS_NULL_INDEX) {
                    RS_THROW(RS_InvalidConfigurationException(
                        "Voxel [{},{},{}] in body A referenced in tendon {} has an adjacent "
                        "voxel at face {}",
                        tendon_config.a_voxel_x,
                        tendon_config.a_voxel_y,
                        tendon_config.a_voxel_z,
                        idx,
                        enumToString(tendon_config.a_voxel_face)
                    ));
                }
            }
        }

        // For Body B: (Apply similar checks as for Body A)
        RSize layer_size_b = body_b_config.x_voxels * body_b_config.y_voxels;
        RSize idx_b = tendon_config.b_voxel_z * layer_size_b +
                      tendon_config.b_voxel_y * body_b_config.x_voxels + tendon_config.b_voxel_x;
        if (body_b_config.material_reference_sid[idx_b] == RS_NULL_INDEX) {
            RS_THROW(RS_InvalidConfigurationException(
                "Voxel [{},{},{}] in body B referenced in tendon {} is null",
                tendon_config.b_voxel_x,
                tendon_config.b_voxel_y,
                tendon_config.b_voxel_z,
                idx
            ));
        }
        if (!body_b_config.link.empty()) {
            if (body_b_config.link[idx_b] & (1 << tendon_config.b_voxel_face)) {
                RS_THROW(RS_InvalidConfigurationException(
                    "Voxel [{},{},{}] in body B referenced in tendon {} has face {} connected",
                    tendon_config.b_voxel_x,
                    tendon_config.b_voxel_y,
                    tendon_config.b_voxel_z,
                    idx,
                    enumToString(tendon_config.b_voxel_face)
                ));
            }
        } else {
            int neighbor_bx =
                tendon_config.b_voxel_x + linkDirectionToXOffset(tendon_config.b_voxel_face);
            int neighbor_by =
                tendon_config.b_voxel_y + linkDirectionToYOffset(tendon_config.b_voxel_face);
            int neighbor_bz =
                tendon_config.b_voxel_z + linkDirectionToZOffset(tendon_config.b_voxel_face);
            if (neighbor_bx >= 0 && neighbor_bx < body_b_config.x_voxels && neighbor_by >= 0 &&
                neighbor_by < body_b_config.y_voxels && neighbor_bz >= 0 &&
                neighbor_bz < body_b_config.z_voxels) {
                RSize neighbor_idx =
                    neighbor_bz * layer_size_b + neighbor_by * body_b_config.x_voxels + neighbor_bx;
                if (body_b_config.material_reference_sid[neighbor_idx] != RS_NULL_INDEX) {
                    RS_THROW(RS_InvalidConfigurationException(
                        "Voxel [{},{},{}] in body B referenced in tendon {} has an adjacent "
                        "voxel at face {}",
                        tendon_config.b_voxel_x,
                        tendon_config.b_voxel_y,
                        tendon_config.b_voxel_z,
                        idx,
                        enumToString(tendon_config.b_voxel_face)
                    ));
                }
            }
        }
    }
}

void RS_StructureConfig::_validateConstraint() const {
    for (auto &&[idx, constraint_config] : enumerate(constraints)) {
        map<RIndex, set<RIndex>> body_segment_ids;
        for (auto &&body_config : bodies) {
            auto &bsi = body_segment_ids[body_config.body_sid];
            for (auto seg_id : body_config.segment_bid) {
                bsi.insert(seg_id);
            }
        }
        // Validate constraints
        if (constraint_config.a_body_sid > bodies.size()) {
            RS_THROW(RS_InvalidConfigurationException(
                "Body A with SID {:d} referenced in constraint {:d} doesn't exist",
                constraint_config.a_body_sid,
                idx
            ));
        }
        if (constraint_config.b_body_sid > bodies.size()) {
            RS_THROW(RS_InvalidConfigurationException(
                "Body B with SID {:d} referenced in constraint {:d} doesn't exist",
                constraint_config.b_body_sid,
                idx
            ));
        }
        auto &a_bsi = body_segment_ids[constraint_config.a_body_sid];
        auto &b_bsi = body_segment_ids[constraint_config.b_body_sid];
        if (a_bsi.find(constraint_config.a_segment_bid) == a_bsi.end()) {
            RS_THROW(RS_InvalidConfigurationException(
                "Segment with BID {:d} from Body A with SID {:d} referenced in "
                "constraint {:d} doesn't exist",
                constraint_config.a_segment_bid,
                constraint_config.a_body_sid,
                idx
            ));
        }
        if (b_bsi.find(constraint_config.b_segment_bid) == b_bsi.end()) {
            RS_THROW(RS_InvalidConfigurationException(
                "Segment with BID {:d} from Body B with SID {:d} referenced in "
                "constraint {:d} doesn't exist",
                constraint_config.b_segment_bid,
                constraint_config.b_body_sid,
                idx
            ));
        }
        if (constraint_config.type == RSE_HINGE_JOINT) {
            if (constraint_config.hinge_signal_sid != RS_NULL_INDEX &&
                constraint_config.hinge_signal_sid > hinge_signal_num) {
                RS_THROW(RS_InvalidConfigurationException(
                    "Hinge signal SID {:d} referenced in constraint {:d} doesn't exist",
                    constraint_config.hinge_signal_sid,
                    idx
                ));
            }
            if (constraint_config.hinge_max <= constraint_config.hinge_min) {
                RS_THROW(RS_InvalidConfigurationException(
                    "Hinge min value must be smaller than "
                    "hinge max value in constraint {:d}",
                    idx
                ));
            }
        }
    }
}

}  // namespace rise