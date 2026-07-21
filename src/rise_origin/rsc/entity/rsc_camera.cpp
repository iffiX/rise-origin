// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_camera.h"

#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/utils/rs_conf.h"

using namespace std;

namespace rise {

namespace rsc {

void RSC_CameraBase::_deserialize(RS_RSCArchive &archive) { RSC_EntityBase::_deserialize(archive); }

void RSC_CameraBase::_serialize(RS_RSCArchive &archive) const {
    RSC_EntityBase::_serialize(archive);
}

bool RSC_CameraPerspectiveRecord::operator==(const RSC_CameraPerspectiveRecord &other) const {
    return RSC_RecordBase::operator==(other) && has_rgb == other.has_rgb &&
           has_depth == other.has_depth && has_normal == other.has_normal &&
           has_motion == other.has_motion && has_entity_info == other.has_entity_info &&
           is_depth_hyperbolic == other.is_depth_hyperbolic &&
           hyperbolic_near_depth == other.hyperbolic_near_depth &&
           hyperbolic_far_depth == other.hyperbolic_far_depth;
}

void RSC_CameraPerspectiveRecord::_validate(RS_RSCValidator &validator) const {
    RSC_RecordBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CameraPerspectiveRecord::_validate", this, [this] {
        RSC_EXPECT_EQ(RSC_CameraPerspectiveRecord, name, "default");
        if (is_depth_hyperbolic) {
            RSC_EXPECT_GT(RSC_CameraPerspectiveRecord, hyperbolic_near_depth, 0.0);
            RSC_EXPECT_GT(RSC_CameraPerspectiveRecord, hyperbolic_far_depth, hyperbolic_near_depth);
        }
    });
}

void RSC_CameraPerspectiveRecord::_read(const RS_RSCFileTreeReader &tree) {
    RSC_RecordBase::_read(tree);
    name = "default";
    has_rgb = tree.get("has_rgb", has_rgb);
    has_depth = tree.get("has_depth", has_depth);
    has_normal = tree.get("has_normal", has_normal);
    has_motion = tree.get("has_motion", has_motion);
    has_entity_info = tree.get("has_entity_info", has_entity_info);
    is_depth_hyperbolic = tree.get("is_depth_hyperbolic", is_depth_hyperbolic);
    hyperbolic_near_depth = tree.get("hyperbolic_near_depth", hyperbolic_near_depth);
    hyperbolic_far_depth = tree.get("hyperbolic_far_depth", hyperbolic_far_depth);
}

void RSC_CameraPerspectiveRecord::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_RecordBase::_write(tree);
    tree.add("has_rgb", has_rgb);
    tree.add("has_depth", has_depth);
    tree.add("has_normal", has_normal);
    tree.add("has_motion", has_motion);
    tree.add("has_entity_info", has_entity_info);
    tree.add("is_depth_hyperbolic", is_depth_hyperbolic);
    tree.add("hyperbolic_near_depth", hyperbolic_near_depth);
    tree.add("hyperbolic_far_depth", hyperbolic_far_depth);
}

void RSC_CameraPerspectiveRecord::_deserialize(RS_RSCArchive &archive) {
    RSC_RecordBase::_deserialize(archive);
    archive.deserialize("has_rgb", has_rgb);
    archive.deserialize("has_depth", has_depth);
    archive.deserialize("has_normal", has_normal);
    archive.deserialize("has_motion", has_motion);
    archive.deserialize("has_entity_info", has_entity_info);
    archive.deserialize("is_depth_hyperbolic", is_depth_hyperbolic);
    archive.deserialize("hyperbolic_near_depth", hyperbolic_near_depth);
    archive.deserialize("hyperbolic_far_depth", hyperbolic_far_depth);
}

void RSC_CameraPerspectiveRecord::_serialize(RS_RSCArchive &archive) const {
    RSC_RecordBase::_serialize(archive);
    archive.serialize("has_rgb", has_rgb);
    archive.serialize("has_depth", has_depth);
    archive.serialize("has_normal", has_normal);
    archive.serialize("has_motion", has_motion);
    archive.serialize("has_entity_info", has_entity_info);
    archive.serialize("is_depth_hyperbolic", is_depth_hyperbolic);
    archive.serialize("hyperbolic_near_depth", hyperbolic_near_depth);
    archive.serialize("hyperbolic_far_depth", hyperbolic_far_depth);
}

bool RSC_CameraPerspectiveSignal::operator==(const RSC_CameraPerspectiveSignal &other) const {
    return RSC_SignalBase::operator==(other) && position_signal_num == other.position_signal_num &&
           orientation_signal_num == other.orientation_signal_num &&
           optics_signal_num == other.optics_signal_num;
}

void RSC_CameraPerspectiveSignal::_validate(RS_RSCValidator &validator) const {
    RSC_SignalBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CameraPerspectiveSignal::_validate", this, [this] {
        RSC_EXPECT_EQ(RSC_CameraPerspectiveSignal, name, "default");
    });
}

void RSC_CameraPerspectiveSignal::_read(const RS_RSCFileTreeReader &tree) {
    RSC_SignalBase::_read(tree);
    name = "default";
    position_signal_num = tree.get<RSize>("position_signal_num", position_signal_num);
    orientation_signal_num = tree.get<RSize>("orientation_signal_num", orientation_signal_num);
    optics_signal_num = tree.get<RSize>("optics_signal_num", optics_signal_num);
}

void RSC_CameraPerspectiveSignal::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_SignalBase::_write(tree);
    tree.add("position_signal_num", position_signal_num);
    tree.add("orientation_signal_num", orientation_signal_num);
    tree.add("optics_signal_num", optics_signal_num);
}

void RSC_CameraPerspectiveSignal::_deserialize(RS_RSCArchive &archive) {
    RSC_SignalBase::_deserialize(archive);
    archive.deserialize("position_signal_num", position_signal_num);
    archive.deserialize("orientation_signal_num", orientation_signal_num);
    archive.deserialize("optics_signal_num", optics_signal_num);
}

void RSC_CameraPerspectiveSignal::_serialize(RS_RSCArchive &archive) const {
    RSC_SignalBase::_serialize(archive);
    archive.serialize("position_signal_num", position_signal_num);
    archive.serialize("orientation_signal_num", orientation_signal_num);
    archive.serialize("optics_signal_num", optics_signal_num);
}

void RSC_CameraPerspectiveBase::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CameraPerspectiveBase::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_CameraPerspectiveBase, image_width, 0);
        RSC_EXPECT_GT(RSC_CameraPerspectiveBase, image_height, 0);
        RSC_EXPECT_GT(RSC_CameraPerspectiveBase, samples_per_pixel, 0);
        RSC_EXPECT_GT(RSC_CameraPerspectiveBase, max_depth, 0);
        RSC_EXPECT_LE(RSC_CameraPerspectiveBase, max_depth, RS_WORLD_VISION_TRACE_MAX_DEPTH);
        RSC_EXPECT_GT(RSC_CameraPerspectiveBase, focus_distance, 0);
        RSC_EXPECT_GT(RSC_CameraPerspectiveBase, field_of_view, 0);
        RSC_EXPECT_GE(RSC_CameraPerspectiveBase, defocus_angle, 0);
    });
}

void RSC_CameraPerspectiveBase::_read(const RS_RSCFileTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    image_width = tree.get<RSize>("image_width", image_width);
    image_height = tree.get<RSize>("image_height", image_height);
    samples_per_pixel = tree.get<RSize>("samples_per_pixel", samples_per_pixel);
    max_depth = tree.get<RSize>("max_depth", max_depth);
    field_of_view = tree.get<RFloat>("field_of_view", field_of_view);
    defocus_angle = tree.get<RFloat>("defocus_angle", defocus_angle);
    focus_distance = tree.get<RFloat>("focus_distance", focus_distance);
    if (tree.hasChild("record")) {
        record.read(tree, "record");
    }
    if (tree.hasChild("signal")) {
        signal.read(tree, "signal");
    }
}

void RSC_CameraPerspectiveBase::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    tree.add("image_width", image_width);
    tree.add("image_height", image_height);
    tree.add("samples_per_pixel", samples_per_pixel);
    tree.add("max_depth", max_depth);
    tree.add("field_of_view", field_of_view);
    tree.add("defocus_angle", defocus_angle);
    tree.add("focus_distance", focus_distance);
    record.write(tree, "record");
    signal.write(tree, "signal");
}

void RSC_CameraPerspectiveBase::_deserialize(RS_RSCArchive &archive) {
    RSC_CameraBase::_deserialize(archive);
    archive.deserialize("image_width", image_width);
    archive.deserialize("image_height", image_height);
    archive.deserialize("samples_per_pixel", samples_per_pixel);
    archive.deserialize("max_depth", max_depth);
    archive.deserialize("field_of_view", field_of_view);
    archive.deserialize("defocus_angle", defocus_angle);
    archive.deserialize("focus_distance", focus_distance);
    archive.deserializeShared("record", record);
    archive.deserializeShared("signal", signal);
}

void RSC_CameraPerspectiveBase::_serialize(RS_RSCArchive &archive) const {
    RSC_CameraBase::_serialize(archive);
    archive.serialize("image_width", image_width);
    archive.serialize("image_height", image_height);
    archive.serialize("samples_per_pixel", samples_per_pixel);
    archive.serialize("max_depth", max_depth);
    archive.serialize("field_of_view", field_of_view);
    archive.serialize("defocus_angle", defocus_angle);
    archive.serialize("focus_distance", focus_distance);
    archive.serializeShared("record", record);
    archive.serializeShared("signal", signal);
}

void RSC_CameraPerspectiveBase::_deepCopyShared(MemoRef memo) {
    record = record.deepCopy(memo);
    signal = signal.deepCopy(memo);
}

bool RSC_CameraPerspectiveFree::operator==(const RSC_CameraPerspectiveFree &other) const {
    return RSC_CameraPerspectiveBase::operator==(other) &&
           origin_position == other.origin_position && orientation == other.orientation;
}

void RSC_CameraPerspectiveFree::_read(const RS_RSCFileTreeReader &tree) {
    RSC_CameraPerspectiveBase::_read(tree);
    origin_position = readVec3rf(tree, "origin_position");
    orientation = readQuat3rf(tree, "orientation");
}

void RSC_CameraPerspectiveFree::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_CameraPerspectiveBase::_write(tree);
    writeVec3rf(tree, "origin_position", origin_position);
    writeQuat3rf(tree, "orientation", orientation);
}

void RSC_CameraPerspectiveFree::_deserialize(RS_RSCArchive &archive) {
    RSC_CameraPerspectiveBase::_deserialize(archive);
    archive.deserialize("origin_position", origin_position);
    archive.deserialize("orientation", orientation);
}

void RSC_CameraPerspectiveFree::_serialize(RS_RSCArchive &archive) const {
    RSC_CameraPerspectiveBase::_serialize(archive);
    archive.serialize("origin_position", origin_position);
    archive.serialize("orientation", orientation);
}

bool RSC_CameraPerspectiveRigidBodyLinked::operator==(
    const RSC_CameraPerspectiveRigidBodyLinked &other
) const {
    return RSC_CameraPerspectiveBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid &&
           relative_anchor == other.relative_anchor &&
           relative_orientation == other.relative_orientation;
}

void RSC_CameraPerspectiveRigidBodyLinked::_validate(RS_RSCValidator &validator) const {
    RSC_CameraPerspectiveBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_CameraPerspectiveRigidBodyLinked::_validate", this, [this] {
            RSC_EXPECT_NOT_NULL(RSC_CameraPerspectiveRigidBodyLinked, body_sid, RS_NULL_INDEX);
        }
    );
}

void RSC_CameraPerspectiveRigidBodyLinked::_read(const RS_RSCFileTreeReader &tree) {
    RSC_CameraPerspectiveBase::_read(tree);
    structure_name = tree.get<string>("structure_name");
    body_sid = tree.get<RIndex>("body_sid");
    segment_bid = tree.get<RIndex>("segment_bid", segment_bid);
    relative_anchor = readVec3rf(tree, "relative_anchor");
    relative_orientation = readQuat3rf(tree, "relative_orientation");
}

void RSC_CameraPerspectiveRigidBodyLinked::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_CameraPerspectiveBase::_write(tree);
    tree.add("structure_name", structure_name);
    tree.add("body_sid", body_sid);
    tree.add("segment_bid", segment_bid);
    writeVec3rf(tree, "relative_anchor", relative_anchor);
    writeQuat3rf(tree, "relative_orientation", relative_orientation);
}

void RSC_CameraPerspectiveRigidBodyLinked::_deserialize(RS_RSCArchive &archive) {
    RSC_CameraPerspectiveBase::_deserialize(archive);
    archive.deserialize("structure_name", structure_name);
    archive.deserialize("body_sid", body_sid);
    archive.deserialize("segment_bid", segment_bid);
    archive.deserialize("relative_anchor", relative_anchor);
    archive.deserialize("relative_orientation", relative_orientation);
}

void RSC_CameraPerspectiveRigidBodyLinked::_serialize(RS_RSCArchive &archive) const {
    RSC_CameraPerspectiveBase::_serialize(archive);
    archive.serialize("structure_name", structure_name);
    archive.serialize("body_sid", body_sid);
    archive.serialize("segment_bid", segment_bid);
    archive.serialize("relative_anchor", relative_anchor);
    archive.serialize("relative_orientation", relative_orientation);
}

bool RSC_CameraStereoRecord::operator==(const RSC_CameraStereoRecord &other) const {
    return RSC_RecordBase::operator==(other) && has_rgb == other.has_rgb &&
           has_depth == other.has_depth && has_normal == other.has_normal &&
           has_motion == other.has_motion && has_entity_info == other.has_entity_info &&
           is_depth_hyperbolic == other.is_depth_hyperbolic &&
           hyperbolic_near_depth == other.hyperbolic_near_depth &&
           hyperbolic_far_depth == other.hyperbolic_far_depth;
}

void RSC_CameraStereoRecord::_validate(RS_RSCValidator &validator) const {
    RSC_RecordBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CameraStereoRecord::_validate", this, [this] {
        RSC_EXPECT_EQ(RSC_CameraStereoRecord, name, "default");
        if (is_depth_hyperbolic) {
            RSC_EXPECT_GT(RSC_CameraStereoRecord, hyperbolic_near_depth, 0.0);
            RSC_EXPECT_GT(RSC_CameraStereoRecord, hyperbolic_far_depth, hyperbolic_near_depth);
        }
    });
}

void RSC_CameraStereoRecord::_read(const RS_RSCFileTreeReader &tree) {
    RSC_RecordBase::_read(tree);
    name = "default";
    has_rgb = tree.get("has_rgb", has_rgb);
    has_depth = tree.get("has_depth", has_depth);
    has_normal = tree.get("has_normal", has_normal);
    has_motion = tree.get("has_motion", has_motion);
    has_entity_info = tree.get("has_entity_info", has_entity_info);
    is_depth_hyperbolic = tree.get("is_depth_hyperbolic", is_depth_hyperbolic);
    hyperbolic_near_depth = tree.get("hyperbolic_near_depth", hyperbolic_near_depth);
    hyperbolic_far_depth = tree.get("hyperbolic_far_depth", hyperbolic_far_depth);
}

void RSC_CameraStereoRecord::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_RecordBase::_write(tree);
    tree.add("has_rgb", has_rgb);
    tree.add("has_depth", has_depth);
    tree.add("has_normal", has_normal);
    tree.add("has_motion", has_motion);
    tree.add("has_entity_info", has_entity_info);
    tree.add("is_depth_hyperbolic", is_depth_hyperbolic);
    tree.add("hyperbolic_near_depth", hyperbolic_near_depth);
    tree.add("hyperbolic_far_depth", hyperbolic_far_depth);
}

void RSC_CameraStereoRecord::_deserialize(RS_RSCArchive &archive) {
    RSC_RecordBase::_deserialize(archive);
    archive.deserialize("has_rgb", has_rgb);
    archive.deserialize("has_depth", has_depth);
    archive.deserialize("has_normal", has_normal);
    archive.deserialize("has_motion", has_motion);
    archive.deserialize("has_entity_info", has_entity_info);
    archive.deserialize("is_depth_hyperbolic", is_depth_hyperbolic);
    archive.deserialize("hyperbolic_near_depth", hyperbolic_near_depth);
    archive.deserialize("hyperbolic_far_depth", hyperbolic_far_depth);
}

void RSC_CameraStereoRecord::_serialize(RS_RSCArchive &archive) const {
    RSC_RecordBase::_serialize(archive);
    archive.serialize("has_rgb", has_rgb);
    archive.serialize("has_depth", has_depth);
    archive.serialize("has_normal", has_normal);
    archive.serialize("has_motion", has_motion);
    archive.serialize("has_entity_info", has_entity_info);
    archive.serialize("is_depth_hyperbolic", is_depth_hyperbolic);
    archive.serialize("hyperbolic_near_depth", hyperbolic_near_depth);
    archive.serialize("hyperbolic_far_depth", hyperbolic_far_depth);
}

bool RSC_CameraStereoSignal::operator==(const RSC_CameraStereoSignal &other) const {
    return RSC_SignalBase::operator==(other);
}

void RSC_CameraStereoSignal::_validate(RS_RSCValidator &validator) const {
    RSC_SignalBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CameraStereoSignal::_validate", this, [this] {
        RSC_EXPECT_EQ(RSC_CameraStereoSignal, name, "default");
    });
}

void RSC_CameraStereoSignal::_read(const RS_RSCFileTreeReader &tree) {
    RSC_SignalBase::_read(tree);
    name = "default";
}

void RSC_CameraStereoSignal::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_SignalBase::_write(tree);
}

void RSC_CameraStereoSignal::_deserialize(RS_RSCArchive &archive) {
    RSC_SignalBase::_deserialize(archive);
}

void RSC_CameraStereoSignal::_serialize(RS_RSCArchive &archive) const {
    RSC_SignalBase::_serialize(archive);
}

void RSC_CameraStereoBase::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CameraStereoBase::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_CameraStereoBase, image_width, 0);
        RSC_EXPECT_GT(RSC_CameraStereoBase, image_height, 0);
        RSC_EXPECT_GT(RSC_CameraStereoBase, samples_per_pixel, 0);
        RSC_EXPECT_GT(RSC_CameraStereoBase, max_depth, 0);
        RSC_EXPECT_LE(RSC_CameraStereoBase, max_depth, RS_WORLD_VISION_TRACE_MAX_DEPTH);
        RSC_EXPECT_GT(RSC_CameraStereoBase, interpupillary_distance, 0);
        RSC_EXPECT_GE(RSC_CameraStereoBase, convergence_distance, 0);
        RSC_EXPECT_GT(RSC_CameraStereoBase, angle_right, angle_left);
        RSC_EXPECT_GT(RSC_CameraStereoBase, angle_up, angle_down);
    });
}

void RSC_CameraStereoBase::_read(const RS_RSCFileTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    image_width = tree.get<RSize>("image_width", image_width);
    image_height = tree.get<RSize>("image_height", image_height);
    samples_per_pixel = tree.get<RSize>("samples_per_pixel", samples_per_pixel);
    max_depth = tree.get<RSize>("max_depth", max_depth);
    interpupillary_distance = tree.get<RFloat>("interpupillary_distance", interpupillary_distance);
    convergence_distance = tree.get<RFloat>("convergence_distance", convergence_distance);
    angle_left = tree.get<RFloat>("angle_left", angle_left);
    angle_right = tree.get<RFloat>("angle_right", angle_right);
    angle_up = tree.get<RFloat>("angle_up", angle_up);
    angle_down = tree.get<RFloat>("angle_down", angle_down);
    if (tree.hasChild("record")) {
        record.read(tree, "record");
    }
    if (tree.hasChild("signal")) {
        signal.read(tree, "signal");
    }
}

void RSC_CameraStereoBase::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    tree.add("image_width", image_width);
    tree.add("image_height", image_height);
    tree.add("samples_per_pixel", samples_per_pixel);
    tree.add("max_depth", max_depth);
    tree.add("interpupillary_distance", interpupillary_distance);
    tree.add("convergence_distance", convergence_distance);
    tree.add("angle_left", angle_left);
    tree.add("angle_right", angle_right);
    tree.add("angle_up", angle_up);
    tree.add("angle_down", angle_down);
    record.write(tree, "record");
    signal.write(tree, "signal");
}

void RSC_CameraStereoBase::_deserialize(RS_RSCArchive &archive) {
    RSC_CameraBase::_deserialize(archive);
    archive.deserialize("image_width", image_width);
    archive.deserialize("image_height", image_height);
    archive.deserialize("samples_per_pixel", samples_per_pixel);
    archive.deserialize("max_depth", max_depth);
    archive.deserialize("interpupillary_distance", interpupillary_distance);
    archive.deserialize("convergence_distance", convergence_distance);
    archive.deserialize("angle_left", angle_left);
    archive.deserialize("angle_right", angle_right);
    archive.deserialize("angle_up", angle_up);
    archive.deserialize("angle_down", angle_down);
    archive.deserializeShared("record", record);
    archive.deserializeShared("signal", signal);
}

void RSC_CameraStereoBase::_serialize(RS_RSCArchive &archive) const {
    RSC_CameraBase::_serialize(archive);
    archive.serialize("image_width", image_width);
    archive.serialize("image_height", image_height);
    archive.serialize("samples_per_pixel", samples_per_pixel);
    archive.serialize("max_depth", max_depth);
    archive.serialize("interpupillary_distance", interpupillary_distance);
    archive.serialize("convergence_distance", convergence_distance);
    archive.serialize("angle_left", angle_left);
    archive.serialize("angle_right", angle_right);
    archive.serialize("angle_up", angle_up);
    archive.serialize("angle_down", angle_down);
    archive.serializeShared("record", record);
    archive.serializeShared("signal", signal);
}

void RSC_CameraStereoBase::_deepCopyShared(MemoRef memo) {
    record = record.deepCopy(memo);
    signal = signal.deepCopy(memo);
}

bool RSC_CameraStereoFree::operator==(const RSC_CameraStereoFree &other) const {
    return RSC_CameraStereoBase::operator==(other) && origin_position == other.origin_position &&
           orientation == other.orientation;
}

void RSC_CameraStereoFree::_read(const RS_RSCFileTreeReader &tree) {
    RSC_CameraStereoBase::_read(tree);
    origin_position = readVec3rf(tree, "origin_position");
    orientation = readQuat3rf(tree, "orientation");
}

void RSC_CameraStereoFree::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_CameraStereoBase::_write(tree);
    writeVec3rf(tree, "origin_position", origin_position);
    writeQuat3rf(tree, "orientation", orientation);
}

void RSC_CameraStereoFree::_deserialize(RS_RSCArchive &archive) {
    RSC_CameraStereoBase::_deserialize(archive);
    archive.deserialize("origin_position", origin_position);
    archive.deserialize("orientation", orientation);
}

void RSC_CameraStereoFree::_serialize(RS_RSCArchive &archive) const {
    RSC_CameraStereoBase::_serialize(archive);
    archive.serialize("origin_position", origin_position);
    archive.serialize("orientation", orientation);
}

bool RSC_CameraStereoRigidBodyLinked::operator==(
    const RSC_CameraStereoRigidBodyLinked &other
) const {
    return RSC_CameraStereoBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid &&
           relative_anchor == other.relative_anchor &&
           relative_orientation == other.relative_orientation;
}

void RSC_CameraStereoRigidBodyLinked::_validate(RS_RSCValidator &validator) const {
    RSC_CameraStereoBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_CameraStereoRigidBodyLinked::_validate", this, [this] {
            RSC_EXPECT_NOT_NULL(RSC_CameraStereoRigidBodyLinked, body_sid, RS_NULL_INDEX);
        }
    );
}

void RSC_CameraStereoRigidBodyLinked::_read(const RS_RSCFileTreeReader &tree) {
    RSC_CameraStereoBase::_read(tree);
    structure_name = tree.get<string>("structure_name");
    body_sid = tree.get<RIndex>("body_sid");
    segment_bid = tree.get<RIndex>("segment_bid", segment_bid);
    relative_anchor = readVec3rf(tree, "relative_anchor");
    relative_orientation = readQuat3rf(tree, "relative_orientation");
}

void RSC_CameraStereoRigidBodyLinked::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_CameraStereoBase::_write(tree);
    tree.add("structure_name", structure_name);
    tree.add("body_sid", body_sid);
    tree.add("segment_bid", segment_bid);
    writeVec3rf(tree, "relative_anchor", relative_anchor);
    writeQuat3rf(tree, "relative_orientation", relative_orientation);
}

void RSC_CameraStereoRigidBodyLinked::_deserialize(RS_RSCArchive &archive) {
    RSC_CameraStereoBase::_deserialize(archive);
    archive.deserialize("structure_name", structure_name);
    archive.deserialize("body_sid", body_sid);
    archive.deserialize("segment_bid", segment_bid);
    archive.deserialize("relative_anchor", relative_anchor);
    archive.deserialize("relative_orientation", relative_orientation);
}

void RSC_CameraStereoRigidBodyLinked::_serialize(RS_RSCArchive &archive) const {
    RSC_CameraStereoBase::_serialize(archive);
    archive.serialize("structure_name", structure_name);
    archive.serialize("body_sid", body_sid);
    archive.serialize("segment_bid", segment_bid);
    archive.serialize("relative_anchor", relative_anchor);
    archive.serialize("relative_orientation", relative_orientation);
}

}  // namespace rsc

}  // namespace rise
