// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_camera_config.h"

#include "rs_rsc_util.h"

using namespace std;

namespace rise {

namespace perspective_camera_config {

void RS_PerspectiveCameraConfig::_validate() const {
    RS_CONFIG_EXPECT_GT(RS_PerspectiveCameraConfig, image_width, 0);
    RS_CONFIG_EXPECT_GT(RS_PerspectiveCameraConfig, image_height, 0);
    RS_CONFIG_EXPECT_GT(RS_PerspectiveCameraConfig, samples_per_pixel, 0);
    RS_CONFIG_EXPECT_GT(RS_PerspectiveCameraConfig, max_depth, 0);
    RS_CONFIG_EXPECT_GT(RS_PerspectiveCameraConfig, focus_distance, 0);
    RS_CONFIG_EXPECT_GT(RS_PerspectiveCameraConfig, field_of_view, 0);
    RS_CONFIG_EXPECT_GE(RS_PerspectiveCameraConfig, defocus_angle, 0);
}

void RS_PerspectiveCameraConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    image_width = config_tree.get<RSize>("ImageWidth", image_width);
    image_height = config_tree.get<RSize>("ImageHeight", image_height);
    samples_per_pixel = config_tree.get<RSize>("SamplesPerPixel", samples_per_pixel);
    max_depth = config_tree.get<RSize>("MaxDepth", max_depth);
    field_of_view = config_tree.get<RFloat>("FieldOfView", field_of_view);
    defocus_angle = config_tree.get<RFloat>("DefocusAngle", defocus_angle);
    focus_distance = config_tree.get<RFloat>("FocusDistance", focus_distance);
}

}  // namespace perspective_camera_config

void RS_FreePerspectiveCameraConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    RS_PerspectiveCameraConfig::_read(config_tree);
    origin_position = readVec3rf(config_tree.get<string>("OriginPosition"));
    orientation = readQuat3rf(config_tree.get<string>("Orientation"));
}

void RS_RigidBodyLinkedPerspectiveCameraConfig::_validate() const {
    RS_PerspectiveCameraConfig::_validate();
    RS_CONFIG_EXPECT_NOT_NULL(RS_RigidBodyLinkedPerspectiveCameraConfig, body_sid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(
        RS_RigidBodyLinkedPerspectiveCameraConfig, segment_bid, RS_NULL_INDEX
    );
}

void RS_RigidBodyLinkedPerspectiveCameraConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    RS_PerspectiveCameraConfig::_read(config_tree);
    structure_name = config_tree.get<string>("StructureName");
    body_sid = config_tree.get<RIndex>("BodySid");
    segment_bid = config_tree.get<RIndex>("SegmentBid");
    relative_anchor = readVec3rf(config_tree.get<string>("RelativeAnchor"));
    relative_orientation = readQuat3rf(config_tree.get<string>("RelativeOrientation"));
}

void RS_CameraConfig::_validate() const {
    if (type == RSE_CAMERA_FREE_PERSPECTIVE) {
        RS_CONFIG_EXPECT_ANY_SET(RS_CameraConfig, config, "RS_FreePerspectiveCameraConfig");
        config.as<RS_FreePerspectiveCameraConfig>().validate();
    } else if (type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE) {
        RS_CONFIG_EXPECT_ANY_SET(
            RS_CameraConfig, config, "RS_RigidBodyLinkedPerspectiveCameraConfig"
        );
        config.as<RS_RigidBodyLinkedPerspectiveCameraConfig>().validate();
    }
}

void RS_CameraConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    name = config_tree.get<string>("Name", name);
    if (!config_tree.isChildExist("Type"))
        RS_THROW(RS_InvalidConfigurationException("Invalid camera config, missing key: Type"));

    auto type_str = config_tree.get<string>("Type");
    if (type_str == "FreePerspective") {
        type = RSE_CAMERA_FREE_PERSPECTIVE;
        config.as<RS_FreePerspectiveCameraConfig>().read(config_tree, false);
    } else if (type_str == "RigidBodyLinkedPerspective") {
        type = RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE;
        config.as<RS_RigidBodyLinkedPerspectiveCameraConfig>().read(config_tree, false);
    } else {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid camera type: {}, available types are: FreePerspective, "
            "RigidBodyLinkedPerspective",
            config_tree.get<string>("Type")
        ));
    }
}

}  // namespace rise