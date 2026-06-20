// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_camera.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/utils/rs_conf.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_CameraPerspectiveRecord::operator==(const RSC_CameraPerspectiveRecord &other) const {
    return RSC_RecordBase::operator==(other) && has_rgb == other.has_rgb &&
           has_depth == other.has_depth && has_entity_info == other.has_entity_info;
}

void RSC_CameraPerspectiveRecord::_validate() const {
    RSC_RecordBase::_validate();
    RSC_EXPECT_EQ(RSC_CameraPerspectiveRecord, name, "default");
}

void RSC_CameraPerspectiveRecord::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_RecordBase::_read(rsc_tree);
    name = "default";
    has_rgb = rsc_tree.get("HasRGB", has_rgb);
    has_depth = rsc_tree.get("HasDepth", has_depth);
    has_entity_info = rsc_tree.get("HasEntityInfo", has_entity_info);
}

bool RSC_CameraPerspectiveSignal::operator==(const RSC_CameraPerspectiveSignal &other) const {
    return RSC_SignalBase::operator==(other) && position_signal_num == other.position_signal_num &&
           orientation_signal_num == other.orientation_signal_num &&
           optics_signal_num == other.optics_signal_num;
}

void RSC_CameraPerspectiveSignal::_validate() const {
    RSC_SignalBase::_validate();
    RSC_EXPECT_EQ(RSC_CameraPerspectiveSignal, name, "default");
}

void RSC_CameraPerspectiveSignal::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_SignalBase::_read(rsc_tree);
    name = "default";
    position_signal_num = rsc_tree.get<RSize>("PositionSignalNum", position_signal_num);
    orientation_signal_num = rsc_tree.get<RSize>("OrientationSignalNum", orientation_signal_num);
    optics_signal_num = rsc_tree.get<RSize>("OpticsSignalNum", optics_signal_num);
}

static void _readPerspectiveCameraBase(
    RSC_CameraPerspectiveBase &cam,
    const RS_CharacterizationPropertyTree &rsc_tree
) {
    cam.image_width = rsc_tree.get<RSize>("ImageWidth", cam.image_width);
    cam.image_height = rsc_tree.get<RSize>("ImageHeight", cam.image_height);
    cam.samples_per_pixel = rsc_tree.get<RSize>("SamplesPerPixel", cam.samples_per_pixel);
    cam.max_depth = rsc_tree.get<RSize>("MaxDepth", cam.max_depth);
    cam.field_of_view = rsc_tree.get<RFloat>("FieldOfView", cam.field_of_view);
    cam.defocus_angle = rsc_tree.get<RFloat>("DefocusAngle", cam.defocus_angle);
    cam.focus_distance = rsc_tree.get<RFloat>("FocusDistance", cam.focus_distance);
    if (rsc_tree.isChildExist("Record")) {
        cam.record.read(rsc_tree.getChild("Record"), false);
    }
    if (rsc_tree.isChildExist("Signal")) {
        cam.signal.read(rsc_tree.getChild("Signal"), false);
    }
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

bool RSC_CameraPerspectiveFree::operator==(const RSC_CameraPerspectiveFree &other) const {
    return RSC_CameraPerspectiveBase::operator==(other) &&
           origin_position == other.origin_position && orientation == other.orientation;
}

void RSC_CameraPerspectiveFree::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_EntityBase::_read(rsc_tree);
    _readPerspectiveCameraBase(*this, rsc_tree);
    origin_position = readVec3rf(rsc_tree.get<string>("OriginPosition"));
    orientation = readQuat3rf(rsc_tree.get<string>("Orientation"));
}

bool RSC_CameraPerspectiveRigidBodyLinked::operator==(
    const RSC_CameraPerspectiveRigidBodyLinked &other
) const {
    return RSC_CameraPerspectiveBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid &&
           relative_anchor == other.relative_anchor &&
           relative_orientation == other.relative_orientation;
}

void RSC_CameraPerspectiveRigidBodyLinked::_validate() const {
    RSC_EntityBase::_validate();
    _validatePerspectiveCameraBase(*this);
    RSC_EXPECT_NOT_NULL(RSC_CameraPerspectiveRigidBodyLinked, body_sid, RS_NULL_INDEX);
}

void RSC_CameraPerspectiveRigidBodyLinked::_read(
    const RS_CharacterizationPropertyTree &rsc_tree
) {
    RSC_EntityBase::_read(rsc_tree);
    _readPerspectiveCameraBase(*this, rsc_tree);
    structure_name = rsc_tree.get<string>("StructureName");
    body_sid = rsc_tree.get<RIndex>("BodySID");
    segment_bid = rsc_tree.get<RIndex>("SegmentBID", segment_bid);
    relative_anchor = readVec3rf(rsc_tree.get<string>("RelativeAnchor"));
    relative_orientation = readQuat3rf(rsc_tree.get<string>("RelativeOrientation"));
}

}  // namespace rsc

}  // namespace rise
