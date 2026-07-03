// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_ray_sensor.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/utils/rs_conf.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_RaySensorRecord::operator==(const RSC_RaySensorRecord &other) const {
    return RSC_RecordBase::operator==(other) && has_hit == other.has_hit &&
           has_entity_info == other.has_entity_info &&
           has_entity_detail == other.has_entity_detail && has_material == other.has_material &&
           has_normal == other.has_normal && has_velocity == other.has_velocity;
}

void RSC_RaySensorRecord::_validate(RS_RSCValidator &validator) const {
    RSC_RecordBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_RaySensorRecord::_validate", this, [this] {
        RSC_EXPECT_EQ(RSC_RaySensorRecord, name, "default");
    });
}

void RSC_RaySensorRecord::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_RecordBase::_read(tree);
    name = "default";
    has_hit = tree.get("has_hit", has_hit);
    has_entity_info = tree.get("has_entity_info", has_entity_info);
    has_entity_detail = tree.get("has_entity_detail", has_entity_detail);
    has_material = tree.get("has_material", has_material);
    has_normal = tree.get("has_normal", has_normal);
    has_velocity = tree.get("has_velocity", has_velocity);
}

void RSC_RaySensorRecord::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_RecordBase::_write(tree);
    tree.add("has_hit", has_hit);
    tree.add("has_entity_info", has_entity_info);
    tree.add("has_entity_detail", has_entity_detail);
    tree.add("has_material", has_material);
    tree.add("has_normal", has_normal);
    tree.add("has_velocity", has_velocity);
}

bool RSC_RaySensorSignal::operator==(const RSC_RaySensorSignal &other) const {
    return RSC_SignalBase::operator==(other) && position_signal_num == other.position_signal_num &&
           orientation_signal_num == other.orientation_signal_num;
}

void RSC_RaySensorSignal::_validate(RS_RSCValidator &validator) const {
    RSC_SignalBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_RaySensorSignal::_validate", this, [this] {
        RSC_EXPECT_EQ(RSC_RaySensorSignal, name, "default");
    });
}

void RSC_RaySensorSignal::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_SignalBase::_read(tree);
    name = "default";
    position_signal_num = tree.get<RSize>("position_signal_num", position_signal_num);
    orientation_signal_num = tree.get<RSize>("orientation_signal_num", orientation_signal_num);
}

void RSC_RaySensorSignal::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_SignalBase::_write(tree);
    tree.add("position_signal_num", position_signal_num);
    tree.add("orientation_signal_num", orientation_signal_num);
}

void RSC_RaySensorBase::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_RaySensorBase::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_RaySensorBase, ray_directions.size(), 0);
        RSC_EXPECT_GT(RSC_RaySensorBase, max_range, 0);
    });
}

void RSC_RaySensorBase::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    max_range = tree.get<RFloat>("max_range", max_range);
    ray_directions.clear();
    if (tree.isChildExist("ray_directions")) {
        auto rows = tree.get<std::vector<std::vector<RFloat>>>("ray_directions");
        ray_directions.reserve(rows.size());
        for (const auto &row : rows) {
            if (row.size() != 3) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "ray_directions row must hold 3 values for RVec3rf"
                ));
            }
            ray_directions.push_back({row[0], row[1], row[2]});
        }
    }
    if (tree.isChildExist("record")) {
        record.read(tree.getChild("record"));
    }
    if (tree.isChildExist("signal")) {
        signal.read(tree.getChild("signal"));
    }
}

void RSC_RaySensorBase::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    std::vector<std::vector<RFloat>> rows;
    rows.reserve(ray_directions.size());
    for (const auto &d : ray_directions) {
        rows.push_back({d.x, d.y, d.z});
    }
    tree.add("ray_directions", rows);
    tree.add("max_range", max_range);
    record.write(tree.addChild("record"));
    signal.write(tree.addChild("signal"));
}

bool RSC_RaySensorFree::operator==(const RSC_RaySensorFree &other) const {
    return RSC_RaySensorBase::operator==(other) && origin_position == other.origin_position &&
           orientation == other.orientation;
}

void RSC_RaySensorFree::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_RaySensorBase::_read(tree);
    origin_position = readVec3rf(tree, "origin_position");
    orientation = readQuat3rf(tree, "orientation");
}

void RSC_RaySensorFree::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_RaySensorBase::_write(tree);
    writeVec3rf(tree, "origin_position", origin_position);
    writeQuat3rf(tree, "orientation", orientation);
}

bool RSC_RaySensorRigidBodyLinked::operator==(const RSC_RaySensorRigidBodyLinked &other) const {
    return RSC_RaySensorBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid &&
           relative_anchor == other.relative_anchor &&
           relative_orientation == other.relative_orientation;
}

void RSC_RaySensorRigidBodyLinked::_validate(RS_RSCValidator &validator) const {
    RSC_RaySensorBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_RaySensorRigidBodyLinked::_validate", this, [this] {
        RSC_EXPECT_NOT_NULL(RSC_RaySensorRigidBodyLinked, body_sid, RS_NULL_INDEX);
    });
}

void RSC_RaySensorRigidBodyLinked::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_RaySensorBase::_read(tree);
    structure_name = tree.get<string>("structure_name");
    body_sid = tree.get<RIndex>("body_sid");
    segment_bid = tree.get<RIndex>("segment_bid", segment_bid);
    relative_anchor = readVec3rf(tree, "relative_anchor");
    relative_orientation = readQuat3rf(tree, "relative_orientation");
}

void RSC_RaySensorRigidBodyLinked::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_RaySensorBase::_write(tree);
    tree.add("structure_name", structure_name);
    tree.add("body_sid", body_sid);
    tree.add("segment_bid", segment_bid);
    writeVec3rf(tree, "relative_anchor", relative_anchor);
    writeQuat3rf(tree, "relative_orientation", relative_orientation);
}

}  // namespace rsc

}  // namespace rise
