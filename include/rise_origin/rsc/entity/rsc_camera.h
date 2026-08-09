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

class RSC_CameraBase : public RSC_EntityBase {
   public:
    /// Scene radiance the sensor saturates at, mapping to full scale output
    /// (unit: W/(m^2*sr)).
    RFloat saturation_radiance = 1.0;

    bool operator==(const RSC_CameraBase& other) const {
        return RSC_EntityBase::operator==(other) &&
               saturation_radiance == other.saturation_radiance;
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
};

/*****************************************************************
 * Perspective camera
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

    /// Whether to record world-space normal image.
    bool has_normal = false;

    /// Whether to record screen-space motion image.
    bool has_motion = false;

    /// Whether to record entity information.
    bool has_entity_info = false;

    /// Whether to encode depth as hyperbolic device depth instead of metric meters.
    bool is_depth_hyperbolic = false;

    /// Near plane distance for hyperbolic depth encoding.
    RFloat hyperbolic_near_depth = 0.1;

    /// Far plane distance for hyperbolic depth encoding.
    RFloat hyperbolic_far_depth = 1000.0;

    bool operator==(const RSC_CameraPerspectiveRecord& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
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

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A perspective camera.
 */
class RSC_CameraPerspectiveBase : public RSC_CameraBase {
   public:
    /// The width of the image.
    RSize image_width = 1280;

    /// The height of the image.
    RSize image_height = 720;

    /// The number of samples per pixel.
    RSize samples_per_pixel = 4;

    /// The maximum depth of the image path tracing.
    RSize max_depth = 3;

    /// The field of view of the camera.
    RFloat field_of_view = 60.0;

    /// The defocus angle of the camera.
    RFloat defocus_angle = 0.0;

    /// The focus distance of the camera.
    RFloat focus_distance = 1.0;

    /// Camera record
    RSCRef<RSC_CameraPerspectiveRecord> record = RSC_CameraPerspectiveRecord();

    /// Camera signal.
    RSCRef<RSC_CameraPerspectiveSignal> signal = RSC_CameraPerspectiveSignal();

    bool operator==(const RSC_CameraPerspectiveBase& other) const {
        return RSC_CameraBase::operator==(other) && image_width == other.image_width &&
               image_height == other.image_height && samples_per_pixel == other.samples_per_pixel &&
               max_depth == other.max_depth && field_of_view == other.field_of_view &&
               defocus_angle == other.defocus_angle && focus_distance == other.focus_distance &&
               record == other.record && signal == other.signal;
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    void _deepCopyShared(MemoRef memo) override;
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

   protected:
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
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

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/*****************************************************************
 * Stereo camera
 *****************************************************************/
/**
 * @brief Record for stereo camera.
 */
class RSC_CameraStereoRecord : public RSC_RecordBase {
   public:
    RSC_CameraStereoRecord() { name = "default"; }

    /// Whether to record RGB image.
    bool has_rgb = true;

    /// Whether to record depth image.
    bool has_depth = true;

    /// Whether to record world-space normal image.
    bool has_normal = false;

    /// Whether to record screen-space motion image.
    bool has_motion = false;

    /// Whether to record entity information.
    bool has_entity_info = false;

    /// Whether to encode depth as hyperbolic device depth instead of metric meters.
    bool is_depth_hyperbolic = false;

    /// Near plane distance for hyperbolic depth encoding.
    RFloat hyperbolic_near_depth = 0.1;

    /// Far plane distance for hyperbolic depth encoding.
    RFloat hyperbolic_far_depth = 1000.0;

    bool operator==(const RSC_CameraStereoRecord& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Signal for stereo camera.
 */
class RSC_CameraStereoSignal : public RSC_SignalBase {
   public:
    RSC_CameraStereoSignal() { name = "default"; }

    bool operator==(const RSC_CameraStereoSignal& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A stereo camera that renders both eyes with asymmetric frusta.
 * @details The eyes are separated by interpupillary_distance and may toe in by
 * convergence_distance. The default per-eye frustum is the four asymmetric angles
 * angle_left/right/up/down in radians; the live per-eye pose and frustum arrive each
 * step through the stereo camera signal.
 */
class RSC_CameraStereoBase : public RSC_CameraBase {
   public:
    /// The per-eye width of the image.
    RSize image_width = 1280;

    /// The per-eye height of the image.
    RSize image_height = 720;

    /// The number of samples per pixel.
    RSize samples_per_pixel = 4;

    /// The maximum depth of the image path tracing.
    RSize max_depth = 3;

    /// The distance between the two eyes in meters.
    RFloat interpupillary_distance = 0.063;

    /// The toe-in convergence distance in meters; 0 keeps the eyes parallel.
    RFloat convergence_distance = 0.0;

    /// The left edge angle of the default eye frustum in radians.
    RFloat angle_left = -0.7853981633974483;

    /// The right edge angle of the default eye frustum in radians.
    RFloat angle_right = 0.7853981633974483;

    /// The top edge angle of the default eye frustum in radians.
    RFloat angle_up = 0.7853981633974483;

    /// The bottom edge angle of the default eye frustum in radians.
    RFloat angle_down = -0.7853981633974483;

    /// Camera record.
    RSCRef<RSC_CameraStereoRecord> record = RSC_CameraStereoRecord();

    /// Camera signal.
    RSCRef<RSC_CameraStereoSignal> signal = RSC_CameraStereoSignal();

    bool operator==(const RSC_CameraStereoBase& other) const {
        return RSC_CameraBase::operator==(other) && image_width == other.image_width &&
               image_height == other.image_height && samples_per_pixel == other.samples_per_pixel &&
               max_depth == other.max_depth &&
               interpupillary_distance == other.interpupillary_distance &&
               convergence_distance == other.convergence_distance &&
               angle_left == other.angle_left && angle_right == other.angle_right &&
               angle_up == other.angle_up && angle_down == other.angle_down &&
               record == other.record && signal == other.signal;
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    void _deepCopyShared(MemoRef memo) override;
};

/**
 * @brief A free stereo camera that can be moved and rotated freely.
 */
class RSC_CameraStereoFree : public RSC_CameraStereoBase {
   public:
    /// The origin position of the head.
    RVec3rf origin_position = RVec3rf::zero();

    /// The orientation of the head.
    RQuat3rf orientation = RQuat3rf::unit();

    bool operator==(const RSC_CameraStereoFree& other) const;

   protected:
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A stereo camera that is rigidly linked to a rigid body.
 */
class RSC_CameraStereoRigidBodyLinked : public RSC_CameraStereoBase {
   public:
    RSC_CameraStereoRigidBodyLinked() {
        record->has_depth = true;
        record->has_entity_info = false;
    }

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

    bool operator==(const RSC_CameraStereoRigidBodyLinked& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_CAMERA_H
