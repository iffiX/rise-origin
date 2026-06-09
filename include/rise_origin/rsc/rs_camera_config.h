// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CAMERA_CONFIG_H
#define RS_CAMERA_CONFIG_H

#include "rise_origin/utils/rs_any.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_quat.h"
#include "rise_origin/utils/rs_vec.h"
#include "rs_base_config.h"

namespace rise {

namespace perspective_camera_config {

class RS_PerspectiveCameraConfig : public RS_BaseConfig {
   public:
    RSize image_width = 0;
    RSize image_height = 0;
    RSize samples_per_pixel = 0;
    RSize max_depth = 0;

    RFloat field_of_view = 0;
    RFloat defocus_angle = 0;
    RFloat focus_distance = 0;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("image_width", image_width);
        s("image_height", image_height);
        s("samples_per_pixel", samples_per_pixel);
        s("max_depth", max_depth);
        s("field_of_view", field_of_view);
        s("defocus_angle", defocus_angle);
        s("focus_distance", focus_distance);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

}  // namespace perspective_camera_config

RS_ENUM(
    RSE_CameraType,
    unsigned int,
    (RSE_CAMERA_NONE, 0),
    (RSE_CAMERA_FREE_PERSPECTIVE, 1),
    (RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE, 2)
)

class RS_FreePerspectiveCameraConfig
    : public perspective_camera_config::RS_PerspectiveCameraConfig {
   public:
    RVec3rf origin_position = RVec3rf::zero();
    RQuat3rf orientation = RQuat3rf::unit();

    template <typename Serializer>
    void serialize(Serializer &s) {
        RS_PerspectiveCameraConfig::serialize(s);
        s("origin_position", origin_position);
        s("orientation", orientation);
    }

   protected:
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_RigidBodyLinkedPerspectiveCameraConfig
    : public perspective_camera_config::RS_PerspectiveCameraConfig {
   public:
    std::string structure_name;

    RIndex body_sid = RS_NULL_INDEX;

    RIndex segment_bid = RS_NULL_INDEX;

    /// Structure scope anchor position.
    RVec3rf relative_anchor = RVec3rf::zero();

    /// Structure scope orientation.
    RQuat3rf relative_orientation = RQuat3rf::unit();

    template <typename Serializer>
    void serialize(Serializer &s) {
        RS_PerspectiveCameraConfig::serialize(s);
        s("structure_name", structure_name);
        s("body_sid", body_sid);
        s("segment_bid", segment_bid);
        s("relative_anchor", relative_anchor);
        s("relative_orientation", relative_orientation);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_CameraConfig : public RS_BaseConfig {
   public:
    /// Name of the camera.
    std::string name;
    RSE_CameraType type = RSE_CAMERA_NONE;
    RS_Any config;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("name", name);
        s("type", type);
        if (type == RSE_CAMERA_FREE_PERSPECTIVE) {
            s("config", config.as<RS_FreePerspectiveCameraConfig>());
        } else if (type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE) {
            s("config", config.as<RS_RigidBodyLinkedPerspectiveCameraConfig>());
        }
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

}  // namespace rise

#endif  // RS_CAMERA_CONFIG_H
