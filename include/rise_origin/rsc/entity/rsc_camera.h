// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_CAMERA_H
#define RSC_CAMERA_H

#include <string>

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
 * Camera Embedded Types
 *****************************************************************/

/**
 * @brief Record for perspective camera.
 */
class RSC_CameraPerspectiveRecord : public RSC_RecordBase {
   public:
    RSC_CameraPerspectiveRecord() { name = "default"; }

    /// Whether to record RGB image.
    bool has_rgb = true;

    /// Whether to record depth image.
    bool has_depth = false;

    /// Whether to record entity information.
    bool has_entity_info = false;

    bool operator==(const RSC_CameraPerspectiveRecord& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_RecordBase::serialize(s);
        s("has_rgb", has_rgb);
        s("has_depth", has_depth);
        s("has_entity_info", has_entity_info);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Signal for perspective camera.
 */
class RSC_CameraPerspectiveSignal : public RSC_SignalBase {
   public:
    RSC_CameraPerspectiveSignal() { name = "default"; }

    /// Number of position signals.
    RSize position_signal_num = 1;

    /// Number of orientation signals.
    RSize orientation_signal_num = 1;

    /// Number of optics signals.
    RSize optics_signal_num = 1;

    bool operator==(const RSC_CameraPerspectiveSignal& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_SignalBase::serialize(s);
        s("position_signal_num", position_signal_num);
        s("orientation_signal_num", orientation_signal_num);
        s("optics_signal_num", optics_signal_num);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/*****************************************************************
 * Camera Types
 *****************************************************************/

class RSC_CameraBase : public RSC_EntityBase {
   public:
    bool operator==(const RSC_CameraBase& other) const { return RSC_EntityBase::operator==(other); }

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_EntityBase::serialize(s);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override { (void)validator; }
    void _read(const RS_RSCPropertyTreeReader& tree) override { (void)tree; }
    void _write(RS_RSCPropertyTreeWriter& tree) const override { (void)tree; }
};

/**
 * @brief A perspective camera.
 */
class RSC_CameraPerspectiveBase : public RSC_CameraBase {
   public:
    /// The width of the image.
    RSize image_width = 0;

    /// The height of the image.
    RSize image_height = 0;

    /// The number of samples per pixel.
    RSize samples_per_pixel = 0;

    /// The maximum depth of the image path tracing.
    RSize max_depth = 0;

    /// The field of view of the camera.
    RFloat field_of_view = 0;

    /// The defocus angle of the camera.
    RFloat defocus_angle = 0;

    /// The focus distance of the camera.
    RFloat focus_distance = 0;

    /// Camera record
    RSC_CameraPerspectiveRecord record;

    /// Camera signal.
    RSC_CameraPerspectiveSignal signal;

    bool operator==(const RSC_CameraPerspectiveBase& other) const {
        return RSC_CameraBase::operator==(other) && image_width == other.image_width &&
               image_height == other.image_height && samples_per_pixel == other.samples_per_pixel &&
               max_depth == other.max_depth && field_of_view == other.field_of_view &&
               defocus_angle == other.defocus_angle && focus_distance == other.focus_distance &&
               record == other.record && signal == other.signal;
    }

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_CameraBase::serialize(s);
        s("image_width", image_width);
        s("image_height", image_height);
        s("samples_per_pixel", samples_per_pixel);
        s("max_depth", max_depth);
        s("field_of_view", field_of_view);
        s("defocus_angle", defocus_angle);
        s("focus_distance", focus_distance);
        s("record", record);
        s("signal", signal);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override { (void)validator; }
    void _read(const RS_RSCPropertyTreeReader& tree) override { (void)tree; }
    void _write(RS_RSCPropertyTreeWriter& tree) const override { (void)tree; }
};

/**
 * @brief A free perspective camera that can be moved and rotated freely.
 */
class RSC_CameraPerspectiveFree : public RSC_CameraPerspectiveBase {
   public:
    /// The origin position of the camera.
    RVec3rf origin_position = RVec3rf::zero();

    /// The orientation of the camera.
    RQuat3rf orientation = RQuat3rf::unit();

    bool operator==(const RSC_CameraPerspectiveFree& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_CameraPerspectiveBase::serialize(s);
        s("origin_position", origin_position);
        s("orientation", orientation);
    }

   protected:
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A perspective camera that is rigidly linked to a rigid body.
 */
class RSC_CameraPerspectiveRigidBodyLinked : public RSC_CameraPerspectiveBase {
   public:
    /// The name of the structure that the camera is linked to.
    std::string structure_name;

    /// The index of the body that the camera is linked to.
    RIndex body_sid = RS_NULL_INDEX;

    /// The index of the segment that the camera is linked to.
    /// Only valid if the structure is a voxel structure.
    RIndex segment_bid = RS_NULL_INDEX;

    /// The relative anchor point of the camera in structure coordinate system.
    RVec3rf relative_anchor = RVec3rf::zero();

    /// The relative orientation of the camera in structure coordinate system.
    RQuat3rf relative_orientation = RQuat3rf::unit();

    bool operator==(const RSC_CameraPerspectiveRigidBodyLinked& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_CameraPerspectiveBase::serialize(s);
        s("structure_name", structure_name);
        s("body_sid", body_sid);
        s("segment_bid", segment_bid);
        s("relative_anchor", relative_anchor);
        s("relative_orientation", relative_orientation);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_CAMERA_H
