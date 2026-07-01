// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_camera.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/utils/rs_conf.h"

using namespace std;

namespace rise {

namespace rsc {

static void
_readPerspectiveCameraBase(RSC_CameraPerspectiveBase &cam, const RS_RSCPropertyTreeReader &tree) {
    cam.image_width = tree.get<RSize>("image_width", cam.image_width);
    cam.image_height = tree.get<RSize>("image_height", cam.image_height);
    cam.samples_per_pixel = tree.get<RSize>("samples_per_pixel", cam.samples_per_pixel);
    cam.max_depth = tree.get<RSize>("max_depth", cam.max_depth);
    cam.field_of_view = tree.get<RFloat>("field_of_view", cam.field_of_view);
    cam.defocus_angle = tree.get<RFloat>("defocus_angle", cam.defocus_angle);
    cam.focus_distance = tree.get<RFloat>("focus_distance", cam.focus_distance);
    if (tree.isChildExist("record")) {
        cam.record.read(tree.getChild("record"));
    }
    if (tree.isChildExist("signal")) {
        cam.signal.read(tree.getChild("signal"));
    }
}

static void
_writePerspectiveCameraBase(const RSC_CameraPerspectiveBase &cam, RS_RSCPropertyTreeWriter &tree) {
    tree.add("image_width", cam.image_width);
    tree.add("image_height", cam.image_height);
    tree.add("samples_per_pixel", cam.samples_per_pixel);
    tree.add("max_depth", cam.max_depth);
    tree.add("field_of_view", cam.field_of_view);
    tree.add("defocus_angle", cam.defocus_angle);
    tree.add("focus_distance", cam.focus_distance);
    cam.record.write(tree.addChild("record"));
    cam.signal.write(tree.addChild("signal"));
}

static void _validatePerspectiveCameraBase(const RSC_CameraPerspectiveBase &cam) {
    RSC_EXPECT_GT(RSC_CameraPerspectiveBase, cam.image_width, 0);
    RSC_EXPECT_GT(RSC_CameraPerspectiveBase, cam.image_height, 0);
    RSC_EXPECT_GT(RSC_CameraPerspectiveBase, cam.samples_per_pixel, 0);
    RSC_EXPECT_GT(RSC_CameraPerspectiveBase, cam.max_depth, 0);
    RSC_EXPECT_LE(RSC_CameraPerspectiveBase, cam.max_depth, RS_WORLD_VISION_TRACE_MAX_DEPTH);
    RSC_EXPECT_GT(RSC_CameraPerspectiveBase, cam.focus_distance, 0);
    RSC_EXPECT_GT(RSC_CameraPerspectiveBase, cam.field_of_view, 0);
    RSC_EXPECT_GE(RSC_CameraPerspectiveBase, cam.defocus_angle, 0);
}

static void _readStereoCameraBase(RSC_CameraStereoBase &cam, const RS_RSCPropertyTreeReader &tree) {
    cam.image_width = tree.get<RSize>("image_width", cam.image_width);
    cam.image_height = tree.get<RSize>("image_height", cam.image_height);
    cam.samples_per_pixel = tree.get<RSize>("samples_per_pixel", cam.samples_per_pixel);
    cam.max_depth = tree.get<RSize>("max_depth", cam.max_depth);
    cam.interpupillary_distance =
        tree.get<RFloat>("interpupillary_distance", cam.interpupillary_distance);
    cam.convergence_distance = tree.get<RFloat>("convergence_distance", cam.convergence_distance);
    cam.angle_left = tree.get<RFloat>("angle_left", cam.angle_left);
    cam.angle_right = tree.get<RFloat>("angle_right", cam.angle_right);
    cam.angle_up = tree.get<RFloat>("angle_up", cam.angle_up);
    cam.angle_down = tree.get<RFloat>("angle_down", cam.angle_down);
    if (tree.isChildExist("record")) {
        cam.record.read(tree.getChild("record"));
    }
    if (tree.isChildExist("signal")) {
        cam.signal.read(tree.getChild("signal"));
    }
}

static void
_writeStereoCameraBase(const RSC_CameraStereoBase &cam, RS_RSCPropertyTreeWriter &tree) {
    tree.add("image_width", cam.image_width);
    tree.add("image_height", cam.image_height);
    tree.add("samples_per_pixel", cam.samples_per_pixel);
    tree.add("max_depth", cam.max_depth);
    tree.add("interpupillary_distance", cam.interpupillary_distance);
    tree.add("convergence_distance", cam.convergence_distance);
    tree.add("angle_left", cam.angle_left);
    tree.add("angle_right", cam.angle_right);
    tree.add("angle_up", cam.angle_up);
    tree.add("angle_down", cam.angle_down);
    cam.record.write(tree.addChild("record"));
    cam.signal.write(tree.addChild("signal"));
}

static void _validateStereoCameraBase(const RSC_CameraStereoBase &cam) {
    RSC_EXPECT_GT(RSC_CameraStereoBase, cam.image_width, 0);
    RSC_EXPECT_GT(RSC_CameraStereoBase, cam.image_height, 0);
    RSC_EXPECT_GT(RSC_CameraStereoBase, cam.samples_per_pixel, 0);
    RSC_EXPECT_GT(RSC_CameraStereoBase, cam.max_depth, 0);
    RSC_EXPECT_LE(RSC_CameraStereoBase, cam.max_depth, RS_WORLD_VISION_TRACE_MAX_DEPTH);
    RSC_EXPECT_GT(RSC_CameraStereoBase, cam.interpupillary_distance, 0);
    RSC_EXPECT_GE(RSC_CameraStereoBase, cam.convergence_distance, 0);
    RSC_EXPECT_GT(RSC_CameraStereoBase, cam.angle_right, cam.angle_left);
    RSC_EXPECT_GT(RSC_CameraStereoBase, cam.angle_up, cam.angle_down);
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

void RSC_CameraPerspectiveRecord::_read(const RS_RSCPropertyTreeReader &tree) {
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

void RSC_CameraPerspectiveRecord::_write(RS_RSCPropertyTreeWriter &tree) const {
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

void RSC_CameraPerspectiveSignal::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_SignalBase::_read(tree);
    name = "default";
    position_signal_num = tree.get<RSize>("position_signal_num", position_signal_num);
    orientation_signal_num = tree.get<RSize>("orientation_signal_num", orientation_signal_num);
    optics_signal_num = tree.get<RSize>("optics_signal_num", optics_signal_num);
}

void RSC_CameraPerspectiveSignal::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_SignalBase::_write(tree);
    tree.add("position_signal_num", position_signal_num);
    tree.add("orientation_signal_num", orientation_signal_num);
    tree.add("optics_signal_num", optics_signal_num);
}

bool RSC_CameraPerspectiveFree::operator==(const RSC_CameraPerspectiveFree &other) const {
    return RSC_CameraPerspectiveBase::operator==(other) &&
           origin_position == other.origin_position && orientation == other.orientation;
}

void RSC_CameraPerspectiveFree::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    _readPerspectiveCameraBase(*this, tree);
    origin_position = readVec3rf(tree, "origin_position");
    orientation = readQuat3rf(tree, "orientation");
}

void RSC_CameraPerspectiveFree::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    _writePerspectiveCameraBase(*this, tree);
    writeVec3rf(tree, "origin_position", origin_position);
    writeQuat3rf(tree, "orientation", orientation);
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
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_CameraPerspectiveRigidBodyLinked::_validate", this, [this] {
            _validatePerspectiveCameraBase(*this);
            RSC_EXPECT_NOT_NULL(RSC_CameraPerspectiveRigidBodyLinked, body_sid, RS_NULL_INDEX);
        }
    );
}

void RSC_CameraPerspectiveRigidBodyLinked::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    _readPerspectiveCameraBase(*this, tree);
    structure_name = tree.get<string>("structure_name");
    body_sid = tree.get<RIndex>("body_sid");
    segment_bid = tree.get<RIndex>("segment_bid", segment_bid);
    relative_anchor = readVec3rf(tree, "relative_anchor");
    relative_orientation = readQuat3rf(tree, "relative_orientation");
}

void RSC_CameraPerspectiveRigidBodyLinked::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    _writePerspectiveCameraBase(*this, tree);
    tree.add("structure_name", structure_name);
    tree.add("body_sid", body_sid);
    tree.add("segment_bid", segment_bid);
    writeVec3rf(tree, "relative_anchor", relative_anchor);
    writeQuat3rf(tree, "relative_orientation", relative_orientation);
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

void RSC_CameraStereoRecord::_read(const RS_RSCPropertyTreeReader &tree) {
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

void RSC_CameraStereoRecord::_write(RS_RSCPropertyTreeWriter &tree) const {
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

bool RSC_CameraStereoSignal::operator==(const RSC_CameraStereoSignal &other) const {
    return RSC_SignalBase::operator==(other);
}

void RSC_CameraStereoSignal::_validate(RS_RSCValidator &validator) const {
    RSC_SignalBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CameraStereoSignal::_validate", this, [this] {
        RSC_EXPECT_EQ(RSC_CameraStereoSignal, name, "default");
    });
}

void RSC_CameraStereoSignal::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_SignalBase::_read(tree);
    name = "default";
}

void RSC_CameraStereoSignal::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_SignalBase::_write(tree);
}

bool RSC_CameraStereoFree::operator==(const RSC_CameraStereoFree &other) const {
    return RSC_CameraStereoBase::operator==(other) && origin_position == other.origin_position &&
           orientation == other.orientation;
}

void RSC_CameraStereoFree::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    _readStereoCameraBase(*this, tree);
    origin_position = readVec3rf(tree, "origin_position");
    orientation = readQuat3rf(tree, "orientation");
}

void RSC_CameraStereoFree::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    _writeStereoCameraBase(*this, tree);
    writeVec3rf(tree, "origin_position", origin_position);
    writeQuat3rf(tree, "orientation", orientation);
}

bool RSC_CameraStereoRigidBodyLinked::operator==(const RSC_CameraStereoRigidBodyLinked &other
) const {
    return RSC_CameraStereoBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid &&
           relative_anchor == other.relative_anchor &&
           relative_orientation == other.relative_orientation;
}

void RSC_CameraStereoRigidBodyLinked::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_CameraStereoRigidBodyLinked::_validate", this, [this] {
            _validateStereoCameraBase(*this);
            RSC_EXPECT_NOT_NULL(RSC_CameraStereoRigidBodyLinked, body_sid, RS_NULL_INDEX);
        }
    );
}

void RSC_CameraStereoRigidBodyLinked::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    _readStereoCameraBase(*this, tree);
    structure_name = tree.get<string>("structure_name");
    body_sid = tree.get<RIndex>("body_sid");
    segment_bid = tree.get<RIndex>("segment_bid", segment_bid);
    relative_anchor = readVec3rf(tree, "relative_anchor");
    relative_orientation = readQuat3rf(tree, "relative_orientation");
}

void RSC_CameraStereoRigidBodyLinked::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    _writeStereoCameraBase(*this, tree);
    tree.add("structure_name", structure_name);
    tree.add("body_sid", body_sid);
    tree.add("segment_bid", segment_bid);
    writeVec3rf(tree, "relative_anchor", relative_anchor);
    writeQuat3rf(tree, "relative_orientation", relative_orientation);
}

}  // namespace rsc

}  // namespace rise
