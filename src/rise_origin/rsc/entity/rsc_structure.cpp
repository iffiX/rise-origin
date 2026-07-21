// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_structure.h"

#include <map>
#include <set>
#include <unordered_map>

#include "rise_origin/device/rs_dev_util.h"
#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rise_origin/utils/string/rs_string.h"

using namespace std;

namespace rise {

namespace rsc {

void RSC_StructureBase::_read(const RS_RSCFileTreeReader &tree) { (void)tree; }

void RSC_StructureBase::_write(RS_RSCFileTreeWriter &tree) const { (void)tree; }

void RSC_StructureBase::_deserialize(RS_RSCArchive &archive) {
    RSC_EntityBase::_deserialize(archive);
    archive.deserialize("origin_position", origin_position);
    archive.deserialize("orientation", orientation);
    archive.deserialize("material_references", material_references);
    archive.deserialize("material_optical_references", material_optical_references);
}

void RSC_StructureBase::_serialize(RS_RSCArchive &archive) const {
    RSC_EntityBase::_serialize(archive);
    archive.serialize("origin_position", origin_position);
    archive.serialize("orientation", orientation);
    archive.serialize("material_references", material_references);
    archive.serialize("material_optical_references", material_optical_references);
}

bool RSC_StructureVoxelBody::operator==(const RSC_StructureVoxelBody &other) const {
    return RSC_Base::operator==(other) &&
           relative_origin_position == other.relative_origin_position &&
           relative_orientation == other.relative_orientation && x_voxels == other.x_voxels &&
           y_voxels == other.y_voxels && z_voxels == other.z_voxels &&
           material_reference_sid == other.material_reference_sid &&
           material_optical_reference_mid == other.material_optical_reference_mid &&
           segment_bid == other.segment_bid && expansion_signal_sid == other.expansion_signal_sid &&
           expansion_control_mode == other.expansion_control_mode && link == other.link &&
           voxel_size == other.voxel_size;
}

void RSC_StructureVoxelBody::_validate(RS_RSCValidator &validator) const {
    if (validator.hasValidator("RSC_StructureVoxelBody::_validate", this)) return;
    validator.submitValidator("RSC_StructureVoxelBody::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_StructureVoxelBody, x_voxels, 0);
        RSC_EXPECT_GT(RSC_StructureVoxelBody, y_voxels, 0);
        RSC_EXPECT_GT(RSC_StructureVoxelBody, z_voxels, 0);

        RSize voxel_num = x_voxels * y_voxels * z_voxels;
        if (material_reference_sid.size() != voxel_num) {
            RS_THROW(RS_InvalidCharacterizationException(
                "RSC_StructureVoxelBody material_reference_sid size mismatch"
            ));
        }
        if (!material_optical_reference_mid.empty() &&
            material_optical_reference_mid.size() != voxel_num) {
            RS_THROW(RS_InvalidCharacterizationException(
                "RSC_StructureVoxelBody material_optical_reference_mid must be empty or match "
                "voxel "
                "count"
            ));
        }
        if (segment_bid.size() != voxel_num) {
            RS_THROW(RS_InvalidCharacterizationException(
                "RSC_StructureVoxelBody segment_bid size mismatch"
            ));
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
    });
    _validateNullSegment(validator);
    _validateLinkAdjacency(validator);
    _validateExpansionControlModeValidity(validator);
}

void RSC_StructureVoxelBody::_read(const RS_RSCFileTreeReader &tree) {
    relative_origin_position = readVec3rf(tree, "relative_origin_position");
    relative_orientation = readQuat3rf(tree, "relative_orientation").normalized();

    x_voxels = tree.get<RSize>("x_voxels");
    y_voxels = tree.get<RSize>("y_voxels");
    z_voxels = tree.get<RSize>("z_voxels");

    RSize voxel_num = x_voxels * y_voxels * z_voxels;

    _readVoxelArray(tree, "material_reference_sid", voxel_num, material_reference_sid);
    _readVoxelArray(
        tree, "material_optical_reference_mid", voxel_num, material_optical_reference_mid
    );
    _readVoxelArray(tree, "segment_bid", voxel_num, segment_bid);
    _readVoxelArray(tree, "expansion_signal_sid", voxel_num, expansion_signal_sid);
    _readVoxelArray(tree, "expansion_control_mode", voxel_num, expansion_control_mode);
    _readVoxelArray(tree, "link", voxel_num, link);

    voxel_size = tree.get<RFloat>("voxel_size", 0);
}

void RSC_StructureVoxelBody::_write(RS_RSCFileTreeWriter &tree) const {
    writeVec3rf(tree, "relative_origin_position", relative_origin_position);
    writeQuat3rf(tree, "relative_orientation", relative_orientation);

    tree.add("x_voxels", x_voxels);
    tree.add("y_voxels", y_voxels);
    tree.add("z_voxels", z_voxels);

    _writeVoxelArray(tree, "material_reference_sid", material_reference_sid);
    _writeVoxelArray(tree, "material_optical_reference_mid", material_optical_reference_mid);
    _writeVoxelArray(tree, "segment_bid", segment_bid);
    _writeVoxelArray(tree, "expansion_signal_sid", expansion_signal_sid);
    _writeVoxelArray(tree, "expansion_control_mode", expansion_control_mode);
    _writeVoxelArray(tree, "link", link);

    tree.add("voxel_size", voxel_size);
}

void RSC_StructureVoxelBody::_deserialize(RS_RSCArchive &archive) {
    archive.deserialize("relative_origin_position", relative_origin_position);
    archive.deserialize("relative_orientation", relative_orientation);
    archive.deserialize("x_voxels", x_voxels);
    archive.deserialize("y_voxels", y_voxels);
    archive.deserialize("z_voxels", z_voxels);
    archive.deserialize("material_reference_sid", material_reference_sid);
    archive.deserialize("material_optical_reference_mid", material_optical_reference_mid);
    archive.deserialize("segment_bid", segment_bid);
    archive.deserialize("expansion_signal_sid", expansion_signal_sid);
    archive.deserialize("expansion_control_mode", expansion_control_mode);
    archive.deserialize("link", link);
    archive.deserialize("voxel_size", voxel_size);
}

void RSC_StructureVoxelBody::_serialize(RS_RSCArchive &archive) const {
    archive.serialize("relative_origin_position", relative_origin_position);
    archive.serialize("relative_orientation", relative_orientation);
    archive.serialize("x_voxels", x_voxels);
    archive.serialize("y_voxels", y_voxels);
    archive.serialize("z_voxels", z_voxels);
    archive.serialize("material_reference_sid", material_reference_sid);
    archive.serialize("material_optical_reference_mid", material_optical_reference_mid);
    archive.serialize("segment_bid", segment_bid);
    archive.serialize("expansion_signal_sid", expansion_signal_sid);
    archive.serialize("expansion_control_mode", expansion_control_mode);
    archive.serialize("link", link);
    archive.serialize("voxel_size", voxel_size);
}

template <typename T>
void RSC_StructureVoxelBody::_readVoxelArray(
    const RS_RSCFileTreeReader &tree,
    const std::string &key,
    RSize voxel_num,
    std::vector<T> &out
) {
    if (!tree.hasChild(key)) {
        out.clear();
        return;
    }
    out.resize(voxel_num);
    tree.getBulk(key, out.data(), {voxel_num});
}

template <typename T>
void RSC_StructureVoxelBody::_writeVoxelArray(
    RS_RSCFileTreeWriter &tree,
    const std::string &key,
    const std::vector<T> &data
) {
    if (!data.empty()) tree.addBulk(key, data.data(), {data.size()});
}

void RSC_StructureVoxelBody::_validateNullSegment(RS_RSCValidator &validator) const {
    RSize voxel_num = x_voxels * y_voxels * z_voxels;
    if (voxel_num == 0 || material_reference_sid.empty()) return;
    RSize thread_num = validator.getThreadPool().size();
    RSize block_size =
        RS_MAX(RS_RSC_STRUCTURE_VOXEL_VALIDATION_BLOCK_SIZE, RS_CEIL_DIV(voxel_num, thread_num));
    RSize num_blocks = RS_CEIL_DIV(voxel_num, block_size);
    auto futures = std::make_shared<RS_MultiFuture<std::pair<bool, std::string>>>(
        validator.getThreadPool().submitTaskSequence(
            0,
            num_blocks,
            [this, voxel_num, block_size](RIndex block) -> std::pair<bool, std::string> {
                if (material_reference_sid.size() != voxel_num || segment_bid.size() != voxel_num) {
                    return {true, {}};
                }
                RSize lo = block * block_size;
                RSize hi = RS_MIN(lo + block_size, voxel_num);
                for (RSize i = lo; i < hi; i++) {
                    if (material_reference_sid[i] == RS_NULL_INDEX &&
                        segment_bid[i] != RS_NULL_INDEX) {
                        return {
                            false,
                            format(
                                "Voxel {:d} is empty (null material) but has non-null segment "
                                "BID {:d}",
                                i,
                                segment_bid[i]
                            )
                        };
                    }
                }
                return {true, {}};
            }
        )
    );
    validator.submitValidator("RSC_StructureVoxelBody::_validateNullSegment", this, [futures] {
        for (auto &[is_valid, message] : futures->get()) {
            if (!is_valid) RS_THROW(RS_InvalidCharacterizationException(message));
        }
    });
}

void RSC_StructureVoxelBody::_validateLinkAdjacency(RS_RSCValidator &validator) const {
    if (link.empty()) return;
    RSize voxel_num = x_voxels * y_voxels * z_voxels;
    RSize thread_num = validator.getThreadPool().size();
    RSize block_size =
        RS_MAX(RS_RSC_STRUCTURE_VOXEL_VALIDATION_BLOCK_SIZE, RS_CEIL_DIV(voxel_num, thread_num));
    RSize num_blocks = RS_CEIL_DIV(voxel_num, block_size);
    auto futures = std::make_shared<RS_MultiFuture<std::pair<bool, std::string>>>(
        validator.getThreadPool().submitTaskSequence(
            0,
            num_blocks,
            [this, voxel_num, block_size](RIndex block) -> std::pair<bool, std::string> {
                if (link.size() != voxel_num || segment_bid.size() != voxel_num) {
                    return {true, {}};
                }
                static constexpr int link_to_x_offset[] = {1, -1, 0, 0, 0, 0};
                static constexpr int link_to_y_offset[] = {0, 0, 1, -1, 0, 0};
                static constexpr int link_to_z_offset[] = {0, 0, 0, 0, 1, -1};
                static constexpr int opposite_link_direction[] = {1, 0, 3, 2, 5, 4};
                RSize layer_size = y_voxels * x_voxels;
                device::RS_DivMod<RSize> xy_divmod(layer_size);
                device::RS_DivMod<RSize> x_divmod(x_voxels);
                RSize lo = block * block_size;
                RSize hi = RS_MIN(lo + block_size, voxel_num);
                for (RSize idx = lo; idx < hi; idx++) {
                    RSize z, rem;
                    xy_divmod.divmod(idx, z, rem);
                    RSize y, x;
                    x_divmod.divmod(rem, y, x);
                    for (int l = 0; l < 6; l++) {
                        bool has_link = link[idx] & (1 << l);
                        if (has_link) continue;
                        int neighbor_x = (int)x + link_to_x_offset[l];
                        int neighbor_y = (int)y + link_to_y_offset[l];
                        int neighbor_z = (int)z + link_to_z_offset[l];
                        if (neighbor_x >= 0 && neighbor_x < (int)x_voxels && neighbor_y >= 0 &&
                            neighbor_y < (int)y_voxels && neighbor_z >= 0 &&
                            neighbor_z < (int)z_voxels) {
                            RSize neighbor_idx = (RSize)neighbor_z * layer_size +
                                                 (RSize)neighbor_y * x_voxels + (RSize)neighbor_x;
                            bool neighbor_has_link =
                                link[neighbor_idx] & (1 << opposite_link_direction[l]);
                            if (neighbor_has_link) {
                                return {
                                    false,
                                    format(
                                        "Voxel {:d} at layer {:d} has inconsistent link with "
                                        "neighbor voxel {:d} at layer {:d}",
                                        y * x_voxels + x,
                                        z,
                                        (RSize)neighbor_y * x_voxels + (RSize)neighbor_x,
                                        neighbor_z
                                    )
                                };
                            } else if (segment_bid[idx] == segment_bid[neighbor_idx]) {
                                return {
                                    false,
                                    format(
                                        "Voxel {:d} at layer {:d} is not connected with neighbor "
                                        "voxel {:d} at layer {:d}, but belong to the same segment",
                                        y * x_voxels + x,
                                        z,
                                        (RSize)neighbor_y * x_voxels + (RSize)neighbor_x,
                                        neighbor_z
                                    )
                                };
                            }
                        }
                    }
                }
                return {true, {}};
            }
        )
    );
    validator.submitValidator("RSC_StructureVoxelBody::_validateLinkAdjacency", this, [futures] {
        for (auto &[is_valid, message] : futures->get()) {
            if (!is_valid) RS_THROW(RS_InvalidCharacterizationException(message));
        }
    });
}

void RSC_StructureVoxelBody::_validateExpansionControlModeValidity(
    RS_RSCValidator &validator
) const {
    if (expansion_control_mode.empty()) return;
    RSize mode_num = expansion_control_mode.size();
    RSize thread_num = validator.getThreadPool().size();
    RSize block_size =
        RS_MAX(RS_RSC_STRUCTURE_VOXEL_VALIDATION_BLOCK_SIZE, RS_CEIL_DIV(mode_num, thread_num));
    RSize num_blocks = RS_CEIL_DIV(mode_num, block_size);
    auto futures = std::make_shared<RS_MultiFuture<std::pair<bool, std::string>>>(
        validator.getThreadPool().submitTaskSequence(
            0,
            num_blocks,
            [this, mode_num, block_size](RIndex block) -> std::pair<bool, std::string> {
                RSize lo = block * block_size;
                RSize hi = RS_MIN(lo + block_size, mode_num);
                for (RSize i = lo; i < hi; i++) {
                    if (!isValidEnum(expansion_control_mode[i])) {
                        return {
                            false,
                            format(
                                "Voxel {:d} has invalid expansion_control_mode value {:d}",
                                i,
                                static_cast<int64_t>(expansion_control_mode[i])
                            )
                        };
                    }
                }
                return {true, {}};
            }
        )
    );
    validator.submitValidator(
        "RSC_StructureVoxelBody::_validateExpansionControlModeValidity", this, [futures] {
            for (auto &[is_valid, message] : futures->get()) {
                if (!is_valid) RS_THROW(RS_InvalidCharacterizationException(message));
            }
        }
    );
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

void RSC_StructureVoxelConstraint::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_StructureVoxelConstraint::_validate", this, [this] {
        RSC_EXPECT_NOT_NULL(RSC_StructureVoxelConstraint, a_body_sid, RS_NULL_INDEX);
        RSC_EXPECT_NOT_NULL(RSC_StructureVoxelConstraint, a_segment_bid, RS_NULL_INDEX);
        RSC_EXPECT_NOT_NULL(RSC_StructureVoxelConstraint, b_body_sid, RS_NULL_INDEX);
        RSC_EXPECT_NOT_NULL(RSC_StructureVoxelConstraint, b_segment_bid, RS_NULL_INDEX);
        if (type == RSE_STRUCTURE_VOXEL_HINGE_JOINT) {
            RSC_EXPECT_GE(RSC_StructureVoxelConstraint, hinge_control_mode, 0);
            RSC_EXPECT_LE(RSC_StructureVoxelConstraint, hinge_control_mode, 3);
            RSC_EXPECT_CLOSE(
                RSC_StructureVoxelConstraint, hinge_relative_axis.length(), 1.0f, 1e-3
            );
            RSC_EXPECT_GE(RSC_StructureVoxelConstraint, hinge_min, -RFLOAT(RS_PI));
            RSC_EXPECT_LE(RSC_StructureVoxelConstraint, hinge_max, RFLOAT(RS_PI));
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
    });
}

void RSC_StructureVoxelConstraint::_read(const RS_RSCFileTreeReader &tree) {
    type = readEnum<RSE_StructureVoxelConstraintType>(tree, "type");
    is_collision_enabled = tree.get<bool>("is_collision_enabled", is_collision_enabled);
    a_body_sid = tree.get<RIndex>("a_body_sid");
    a_segment_bid = tree.get<RIndex>("a_segment_bid");
    b_body_sid = tree.get<RIndex>("b_body_sid");
    b_segment_bid = tree.get<RIndex>("b_segment_bid");
    relative_anchor = readVec3rf(tree, "relative_anchor");
    if (type == RSE_STRUCTURE_VOXEL_HINGE_JOINT) {
        hinge_control_mode =
            readEnum<RSE_StructureVoxelHingeControlMode>(tree, "hinge_control_mode");
        hinge_relative_axis = readVec3rf(tree, "hinge_relative_axis").normalized();
        hinge_min = tree.get<RFloat>("hinge_min", -RFLOAT(RS_PI));
        hinge_max = tree.get<RFloat>("hinge_max", RFLOAT(RS_PI));
        hinge_position_control_kp = tree.get<RFloat>("hinge_position_control_kp", 0);
        hinge_position_control_kd = tree.get<RFloat>("hinge_position_control_kd", 0);
        hinge_max_speed = tree.get<RFloat>("hinge_max_speed", 10);
        hinge_max_torque = tree.get<RFloat>("hinge_max_torque", 0);
        hinge_signal_sid = tree.get<RIndex>("hinge_signal_sid", RS_NULL_INDEX);
    } else if (type == RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT) {
        prismatic_control_mode =
            readEnum<RSE_StructureVoxelPrismaticControlMode>(tree, "prismatic_control_mode");
        prismatic_relative_axis = readVec3rf(tree, "prismatic_relative_axis").normalized();
        prismatic_min = tree.get<RFloat>("prismatic_min", -1.0);
        prismatic_max = tree.get<RFloat>("prismatic_max", 1.0);
        prismatic_position_control_kp = tree.get<RFloat>("prismatic_position_control_kp", 0);
        prismatic_position_control_kd = tree.get<RFloat>("prismatic_position_control_kd", 0);
        prismatic_max_speed = tree.get<RFloat>("prismatic_max_speed", 10);
        prismatic_max_force = tree.get<RFloat>("prismatic_max_force", 0);
        prismatic_signal_sid = tree.get<RIndex>("prismatic_signal_sid", RS_NULL_INDEX);
    }
}

void RSC_StructureVoxelConstraint::_write(RS_RSCFileTreeWriter &tree) const {
    writeEnum(tree, "type", type);
    tree.add("is_collision_enabled", is_collision_enabled);
    tree.add("a_body_sid", a_body_sid);
    tree.add("a_segment_bid", a_segment_bid);
    tree.add("b_body_sid", b_body_sid);
    tree.add("b_segment_bid", b_segment_bid);
    writeVec3rf(tree, "relative_anchor", relative_anchor);
    if (type == RSE_STRUCTURE_VOXEL_HINGE_JOINT) {
        writeEnum(tree, "hinge_control_mode", hinge_control_mode);
        writeVec3rf(tree, "hinge_relative_axis", hinge_relative_axis);
        tree.add("hinge_min", hinge_min);
        tree.add("hinge_max", hinge_max);
        tree.add("hinge_position_control_kp", hinge_position_control_kp);
        tree.add("hinge_position_control_kd", hinge_position_control_kd);
        tree.add("hinge_max_speed", hinge_max_speed);
        tree.add("hinge_max_torque", hinge_max_torque);
        tree.add("hinge_signal_sid", hinge_signal_sid);
    } else if (type == RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT) {
        writeEnum(tree, "prismatic_control_mode", prismatic_control_mode);
        writeVec3rf(tree, "prismatic_relative_axis", prismatic_relative_axis);
        tree.add("prismatic_min", prismatic_min);
        tree.add("prismatic_max", prismatic_max);
        tree.add("prismatic_position_control_kp", prismatic_position_control_kp);
        tree.add("prismatic_position_control_kd", prismatic_position_control_kd);
        tree.add("prismatic_max_speed", prismatic_max_speed);
        tree.add("prismatic_max_force", prismatic_max_force);
        tree.add("prismatic_signal_sid", prismatic_signal_sid);
    }
}

void RSC_StructureVoxelConstraint::_deserialize(RS_RSCArchive &archive) {
    archive.deserialize("type", type);
    archive.deserialize("is_collision_enabled", is_collision_enabled);
    archive.deserialize("a_body_sid", a_body_sid);
    archive.deserialize("a_segment_bid", a_segment_bid);
    archive.deserialize("b_body_sid", b_body_sid);
    archive.deserialize("b_segment_bid", b_segment_bid);
    archive.deserialize("relative_anchor", relative_anchor);
    archive.deserialize("hinge_control_mode", hinge_control_mode);
    archive.deserialize("hinge_relative_axis", hinge_relative_axis);
    archive.deserialize("hinge_min", hinge_min);
    archive.deserialize("hinge_max", hinge_max);
    archive.deserialize("hinge_position_control_kp", hinge_position_control_kp);
    archive.deserialize("hinge_position_control_kd", hinge_position_control_kd);
    archive.deserialize("hinge_max_speed", hinge_max_speed);
    archive.deserialize("hinge_max_torque", hinge_max_torque);
    archive.deserialize("hinge_signal_sid", hinge_signal_sid);
    archive.deserialize("prismatic_control_mode", prismatic_control_mode);
    archive.deserialize("prismatic_relative_axis", prismatic_relative_axis);
    archive.deserialize("prismatic_min", prismatic_min);
    archive.deserialize("prismatic_max", prismatic_max);
    archive.deserialize("prismatic_position_control_kp", prismatic_position_control_kp);
    archive.deserialize("prismatic_position_control_kd", prismatic_position_control_kd);
    archive.deserialize("prismatic_max_speed", prismatic_max_speed);
    archive.deserialize("prismatic_max_force", prismatic_max_force);
    archive.deserialize("prismatic_signal_sid", prismatic_signal_sid);
}

void RSC_StructureVoxelConstraint::_serialize(RS_RSCArchive &archive) const {
    archive.serialize("type", type);
    archive.serialize("is_collision_enabled", is_collision_enabled);
    archive.serialize("a_body_sid", a_body_sid);
    archive.serialize("a_segment_bid", a_segment_bid);
    archive.serialize("b_body_sid", b_body_sid);
    archive.serialize("b_segment_bid", b_segment_bid);
    archive.serialize("relative_anchor", relative_anchor);
    archive.serialize("hinge_control_mode", hinge_control_mode);
    archive.serialize("hinge_relative_axis", hinge_relative_axis);
    archive.serialize("hinge_min", hinge_min);
    archive.serialize("hinge_max", hinge_max);
    archive.serialize("hinge_position_control_kp", hinge_position_control_kp);
    archive.serialize("hinge_position_control_kd", hinge_position_control_kd);
    archive.serialize("hinge_max_speed", hinge_max_speed);
    archive.serialize("hinge_max_torque", hinge_max_torque);
    archive.serialize("hinge_signal_sid", hinge_signal_sid);
    archive.serialize("prismatic_control_mode", prismatic_control_mode);
    archive.serialize("prismatic_relative_axis", prismatic_relative_axis);
    archive.serialize("prismatic_min", prismatic_min);
    archive.serialize("prismatic_max", prismatic_max);
    archive.serialize("prismatic_position_control_kp", prismatic_position_control_kp);
    archive.serialize("prismatic_position_control_kd", prismatic_position_control_kd);
    archive.serialize("prismatic_max_speed", prismatic_max_speed);
    archive.serialize("prismatic_max_force", prismatic_max_force);
    archive.serialize("prismatic_signal_sid", prismatic_signal_sid);
}

bool RSC_StructureVoxelRecord::operator==(const RSC_StructureVoxelRecord &other) const {
    return RSC_RecordBase::operator==(other) && voxel_record_range == other.voxel_record_range &&
           link_record_range == other.link_record_range &&
           rigid_body_record_range == other.rigid_body_record_range &&
           joint_record_range == other.joint_record_range &&
           voxel_sample_num == other.voxel_sample_num && link_sample_num == other.link_sample_num;
}

void RSC_StructureVoxelRecord::_validate(RS_RSCValidator &validator) const {
    RSC_RecordBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_StructureVoxelRecord::_validate", this, [this] {
        RSC_EXPECT_EQ(RSC_StructureVoxelRecord, name, "default");
    });
}

void RSC_StructureVoxelRecord::_read(const RS_RSCFileTreeReader &tree) {
    RSC_RecordBase::_read(tree);
    name = "default";
    voxel_record_range = readEnum<RSE_StructureVoxelRecordVoxelRange>(tree, "voxel_record_range");
    link_record_range = readEnum<RSE_StructureVoxelRecordLinkRange>(tree, "link_record_range");
    rigid_body_record_range =
        readEnum<RSE_StructureVoxelRecordRigidBodyRange>(tree, "rigid_body_record_range");
    joint_record_range = readEnum<RSE_StructureVoxelRecordJointRange>(tree, "joint_record_range");
    voxel_sample_num = tree.get<RSize>("voxel_sample_num", voxel_sample_num);
    link_sample_num = tree.get<RSize>("link_sample_num", link_sample_num);
}

void RSC_StructureVoxelRecord::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_RecordBase::_write(tree);
    writeEnum(tree, "voxel_record_range", voxel_record_range);
    writeEnum(tree, "link_record_range", link_record_range);
    writeEnum(tree, "rigid_body_record_range", rigid_body_record_range);
    writeEnum(tree, "joint_record_range", joint_record_range);
    tree.add("voxel_sample_num", voxel_sample_num);
    tree.add("link_sample_num", link_sample_num);
}

void RSC_StructureVoxelRecord::_deserialize(RS_RSCArchive &archive) {
    RSC_RecordBase::_deserialize(archive);
    archive.deserialize("voxel_record_range", voxel_record_range);
    archive.deserialize("link_record_range", link_record_range);
    archive.deserialize("rigid_body_record_range", rigid_body_record_range);
    archive.deserialize("joint_record_range", joint_record_range);
    archive.deserialize("voxel_sample_num", voxel_sample_num);
    archive.deserialize("link_sample_num", link_sample_num);
}

void RSC_StructureVoxelRecord::_serialize(RS_RSCArchive &archive) const {
    RSC_RecordBase::_serialize(archive);
    archive.serialize("voxel_record_range", voxel_record_range);
    archive.serialize("link_record_range", link_record_range);
    archive.serialize("rigid_body_record_range", rigid_body_record_range);
    archive.serialize("joint_record_range", joint_record_range);
    archive.serialize("voxel_sample_num", voxel_sample_num);
    archive.serialize("link_sample_num", link_sample_num);
}

bool RSC_StructureVoxelSignal::operator==(const RSC_StructureVoxelSignal &other) const {
    return RSC_SignalBase::operator==(other) &&
           expansion_signal_num == other.expansion_signal_num &&
           hinge_signal_num == other.hinge_signal_num &&
           prismatic_signal_num == other.prismatic_signal_num;
}

void RSC_StructureVoxelSignal::_validate(RS_RSCValidator &validator) const {
    RSC_SignalBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_StructureVoxelSignal::_validate", this, [this] {
        RSC_EXPECT_EQ(RSC_StructureVoxelSignal, name, "default");
    });
}

void RSC_StructureVoxelSignal::_read(const RS_RSCFileTreeReader &tree) {
    RSC_SignalBase::_read(tree);
    name = "default";
    expansion_signal_num = tree.get("expansion_signal_num", expansion_signal_num);
    hinge_signal_num = tree.get("hinge_signal_num", hinge_signal_num);
    prismatic_signal_num = tree.get("prismatic_signal_num", prismatic_signal_num);
}

void RSC_StructureVoxelSignal::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_SignalBase::_write(tree);
    tree.add("expansion_signal_num", expansion_signal_num);
    tree.add("hinge_signal_num", hinge_signal_num);
    tree.add("prismatic_signal_num", prismatic_signal_num);
}

void RSC_StructureVoxelSignal::_deserialize(RS_RSCArchive &archive) {
    RSC_SignalBase::_deserialize(archive);
    archive.deserialize("expansion_signal_num", expansion_signal_num);
    archive.deserialize("hinge_signal_num", hinge_signal_num);
    archive.deserialize("prismatic_signal_num", prismatic_signal_num);
}

void RSC_StructureVoxelSignal::_serialize(RS_RSCArchive &archive) const {
    RSC_SignalBase::_serialize(archive);
    archive.serialize("expansion_signal_num", expansion_signal_num);
    archive.serialize("hinge_signal_num", hinge_signal_num);
    archive.serialize("prismatic_signal_num", prismatic_signal_num);
}

bool RSC_StructureVoxel::operator==(const RSC_StructureVoxel &other) const {
    return RSC_StructureBase::operator==(other) && is_fixed == other.is_fixed &&
           bodies == other.bodies && constraints == other.constraints && record == other.record &&
           signal == other.signal;
}

void RSC_StructureVoxel::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    for (auto &body : bodies) body.validate(validator);
    for (auto &constraint : constraints) constraint.validate(validator);

    _validateMaterialReferences(validator);
    _validateExpansionSignals(validator);
    _validateConstraints(validator);
    _validateConstraintSignals(validator);
}

void RSC_StructureVoxel::_validateMaterialReferences(RS_RSCValidator &validator) const {
    RSize material_limit = material_references.size();
    RSize thread_num = validator.getThreadPool().size();
    std::string job =
        format("RSC_StructureVoxel::_validateMaterialReferences:limit={:d}", material_limit);
    std::vector<RSize> optical_counts;
    optical_counts.reserve(material_optical_references.size());
    for (const auto &material_opticals : material_optical_references) {
        optical_counts.push_back(material_opticals.size());
    }
    for (auto &&[body_sid, body_rsc] : enumerate(bodies)) {
        const RSC_StructureVoxelBody *bp = &body_rsc;
        RSize voxel_num = body_rsc.material_reference_sid.size();
        if (voxel_num == 0 || validator.hasValidator(job, bp)) continue;

        RSize block_size = RS_MAX(
            RS_RSC_STRUCTURE_VOXEL_VALIDATION_BLOCK_SIZE, RS_CEIL_DIV(voxel_num, thread_num)
        );
        RSize num_blocks = RS_CEIL_DIV(voxel_num, block_size);
        auto futures = std::make_shared<RS_MultiFuture<std::pair<bool, std::string>>>(
            validator.getThreadPool().submitTaskSequence(
                0,
                num_blocks,
                [bp, body_sid, material_limit, voxel_num, block_size, optical_counts](RIndex block)
                    -> std::pair<bool, std::string> {
                    RSize lo = block * block_size;
                    RSize hi = RS_MIN(lo + block_size, voxel_num);
                    for (RSize i = lo; i < hi; i++) {
                        RIndex m = bp->material_reference_sid[i];
                        if (m != RS_NULL_INDEX && m >= material_limit) {
                            return {
                                false,
                                format(
                                    "Material SID {:d} referenced in body with SID {:d} doesn't "
                                    "exist",
                                    m,
                                    body_sid
                                )
                            };
                        }
                        // The optical mid, when given, indexes the voxel's material's optical list.
                        if (bp->material_optical_reference_mid.empty()) continue;
                        RIndex o = bp->material_optical_reference_mid[i];
                        if (o == RS_NULL_INDEX || m == RS_NULL_INDEX ||
                            m >= optical_counts.size()) {
                            continue;
                        }
                        // A material with no declared optical list still admits mid 0 (its
                        // default).
                        RSize optical_limit = optical_counts[m] > 0 ? optical_counts[m] : 1;
                        if (o >= optical_limit) {
                            return {
                                false,
                                format(
                                    "Material optical MID {:d} referenced in body with SID {:d} "
                                    "doesn't exist for its material",
                                    o,
                                    body_sid
                                )
                            };
                        }
                    }
                    return {true, {}};
                }
            )
        );
        // Use body pointer as key and dedup with limit as key so structures sharing
        // same body does not need duplicate validation.
        validator.submitValidator(job, bp, [this, futures] {
            if (!material_optical_references.empty() &&
                material_optical_references.size() != material_references.size()) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "RSC_StructureVoxel material_optical_references size must be empty or match "
                    "material_references size"
                ));
            }
            for (auto &[is_valid, message] : futures->get()) {
                if (!is_valid) RS_THROW(RS_InvalidCharacterizationException(message));
            }
        });
    }
}

void RSC_StructureVoxel::_validateExpansionSignals(RS_RSCValidator &validator) const {
    RSize limit = signal->expansion_signal_num;
    RSize thread_num = validator.getThreadPool().size();
    std::string job = format("RSC_StructureVoxel::_validateExpansionSignals:limit={:d}", limit);
    for (auto &&[body_sid, body_rsc] : enumerate(bodies)) {
        const RSC_StructureVoxelBody *bp = &body_rsc;
        RSize voxel_num = body_rsc.expansion_signal_sid.size();
        if (voxel_num == 0 || validator.hasValidator(job, bp)) continue;

        RSize block_size = RS_MAX(
            RS_RSC_STRUCTURE_VOXEL_VALIDATION_BLOCK_SIZE, RS_CEIL_DIV(voxel_num, thread_num)
        );
        RSize num_blocks = RS_CEIL_DIV(voxel_num, block_size);
        auto futures = std::make_shared<RS_MultiFuture<std::pair<bool, std::string>>>(
            validator.getThreadPool().submitTaskSequence(
                0,
                num_blocks,
                [bp, body_sid, limit, voxel_num, block_size](RIndex block)
                    -> std::pair<bool, std::string> {
                    RSize lo = block * block_size;
                    RSize hi = RS_MIN(lo + block_size, voxel_num);
                    for (RSize i = lo; i < hi; i++) {
                        RIndex e = bp->expansion_signal_sid[i];
                        if (e != RS_NULL_INDEX && e >= limit) {
                            return {
                                false,
                                format(
                                    "Expansion signal SID {:d} referenced in body with SID {:d} "
                                    "doesn't exist",
                                    e,
                                    body_sid
                                )
                            };
                        }
                    }
                    return {true, {}};
                }
            )
        );
        // Use body pointer as key and dedup with limit as key so structures sharing
        // same body does not need duplicate validation.
        validator.submitValidator(job, bp, [futures] {
            for (auto &[is_valid, message] : futures->get()) {
                if (!is_valid) RS_THROW(RS_InvalidCharacterizationException(message));
            }
        });
    }
}

void RSC_StructureVoxel::_validateConstraints(RS_RSCValidator &validator) const {
    // The check is fully determined by the shared body and constraint identities, so depend on
    // them all so reused structures validate once.
    std::string job = "RSC_StructureVoxel::_validateConstraints";
    std::vector<const RSC_Base *> dependencies;
    dependencies.reserve(bodies.size() + constraints.size());
    for (auto &body : bodies) dependencies.push_back(&body);
    for (auto &constraint : constraints) dependencies.push_back(&constraint);
    if (validator.hasValidator(job, dependencies)) return;

    RSize thread_num = validator.getThreadPool().size();
    auto body_segment_id_blocks = std::make_shared<std::vector<RS_MultiFuture<std::set<RIndex>>>>();
    auto block_body_sids = std::make_shared<std::vector<RSize>>();
    for (auto &&[body_sid, body_rsc] : enumerate(bodies)) {
        const RSC_StructureVoxelBody *bp = &body_rsc;
        RSize voxel_num = body_rsc.segment_bid.size();

        if (voxel_num == 0) continue;

        RSize block_size = RS_MAX(
            RS_RSC_STRUCTURE_VOXEL_VALIDATION_BLOCK_SIZE, RS_CEIL_DIV(voxel_num, thread_num)
        );
        RSize num_blocks = RS_CEIL_DIV(voxel_num, block_size);
        body_segment_id_blocks->push_back(validator.getThreadPool().submitTaskSequence(
            0, num_blocks, [bp, voxel_num, block_size](RIndex block) -> std::set<RIndex> {
                std::set<RIndex> ids;
                RSize lo = block * block_size;
                RSize hi = RS_MIN(lo + block_size, voxel_num);
                for (RSize i = lo; i < hi; i++) ids.insert(bp->segment_bid[i]);
                return ids;
            }
        ));
        block_body_sids->push_back(body_sid);
    }
    validator.submitValidator(job, dependencies, [this, body_segment_id_blocks, block_body_sids] {
        std::vector<std::set<RIndex>> segment_ids(bodies.size());
        for (RSize i = 0; i < body_segment_id_blocks->size(); i++) {
            RSize body_sid = (*block_body_sids)[i];
            for (auto &partial : (*body_segment_id_blocks)[i].get()) {
                segment_ids[body_sid].insert(partial.begin(), partial.end());
            }
        }
        for (auto &&[idx, constraint_rsc] : enumerate(constraints)) {
            if (constraint_rsc.a_body_sid >= bodies.size()) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Body A with SID {:d} referenced in constraint {:d} doesn't exist",
                    constraint_rsc.a_body_sid,
                    idx
                ));
            }
            if (constraint_rsc.b_body_sid >= bodies.size()) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Body B with SID {:d} referenced in constraint {:d} doesn't exist",
                    constraint_rsc.b_body_sid,
                    idx
                ));
            }
            auto &a_bsi = segment_ids[constraint_rsc.a_body_sid];
            auto &b_bsi = segment_ids[constraint_rsc.b_body_sid];
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
                        "Hinge min value must be smaller than hinge max value in constraint "
                        "{:d}",
                        idx
                    ));
                }
            } else if (constraint_rsc.type == RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT) {
                if (constraint_rsc.prismatic_max <= constraint_rsc.prismatic_min) {
                    RS_THROW(RS_InvalidCharacterizationException(
                        "Prismatic min value must be smaller than prismatic max value in "
                        "constraint {:d}",
                        idx
                    ));
                }
            }
        }
    });
}

void RSC_StructureVoxel::_validateConstraintSignals(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists(
        "RSC_StructureVoxel::_validateConstraintSignals", this, [this] {
            for (auto &&[idx, c] : enumerate(constraints)) {
                if (c.hinge_signal_sid != RS_NULL_INDEX &&
                    c.hinge_signal_sid >= signal->hinge_signal_num) {
                    RS_THROW(RS_InvalidCharacterizationException(
                        "Hinge signal SID {:d} referenced in constraint with SID {:d} doesn't "
                        "exist",
                        c.hinge_signal_sid,
                        idx
                    ));
                }
                if (c.prismatic_signal_sid != RS_NULL_INDEX &&
                    c.prismatic_signal_sid >= signal->prismatic_signal_num) {
                    RS_THROW(RS_InvalidCharacterizationException(
                        "Prismatic signal SID {:d} referenced in constraint with SID {:d} doesn't "
                        "exist",
                        c.prismatic_signal_sid,
                        idx
                    ));
                }
            }
        }
    );
}

void RSC_StructureVoxel::_read(const RS_RSCFileTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    is_fixed = tree.get("is_fixed", is_fixed);
    origin_position = readVec3rf(tree, "origin_position");
    orientation = readQuat3rf(tree, "orientation");
    material_references = tree.get<std::vector<std::string>>("material_references");
    if (tree.hasChild("material_optical_references")) {
        material_optical_references =
            tree.get<std::vector<std::vector<std::string>>>("material_optical_references");
    }

    bodies.read(tree, "bodies");
    constraints.read(tree, "constraints");
    if (tree.hasChild("record")) {
        record.read(tree, "record");
    }
    if (tree.hasChild("signal")) {
        signal.read(tree, "signal");
    }
}

void RSC_StructureVoxel::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    tree.add("is_fixed", is_fixed);
    writeVec3rf(tree, "origin_position", origin_position);
    writeQuat3rf(tree, "orientation", orientation);
    tree.add("material_references", material_references);
    if (!material_optical_references.empty()) {
        tree.add("material_optical_references", material_optical_references);
    }

    bodies.write(tree, "bodies");
    constraints.write(tree, "constraints");
    record.write(tree, "record");
    signal.write(tree, "signal");
}

void RSC_StructureVoxel::_deserialize(RS_RSCArchive &archive) {
    RSC_StructureBase::_deserialize(archive);
    archive.deserialize("is_fixed", is_fixed);
    archive.deserializeShared("bodies", bodies);
    archive.deserializeShared("constraints", constraints);
    archive.deserializeShared("record", record);
    archive.deserializeShared("signal", signal);
}

void RSC_StructureVoxel::_serialize(RS_RSCArchive &archive) const {
    RSC_StructureBase::_serialize(archive);
    archive.serialize("is_fixed", is_fixed);
    archive.serializeShared("bodies", bodies);
    archive.serializeShared("constraints", constraints);
    archive.serializeShared("record", record);
    archive.serializeShared("signal", signal);
}

void RSC_StructureVoxel::_deepCopyShared(MemoRef memo) {
    bodies = bodies.deepCopy(memo);
    constraints = constraints.deepCopy(memo);
    record = record.deepCopy(memo);
    signal = signal.deepCopy(memo);
}

}  // namespace rsc

}  // namespace rise
