// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_RAY_SENSOR_H
#define RSC_RAY_SENSOR_H

#include <string>
#include <vector>

#include "rise_origin/rsc/record/rsc_record_base.h"
#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/rsc/signal/rsc_signal_base.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_entity_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Record for ray sensor.
 */
class RSC_RaySensorRecord : public RSC_RecordBase {
   public:
    RSC_RaySensorRecord() { name = "default"; }

    /// Whether to record hit flag, distance and hit position.
    bool has_hit = true;

    /// Whether to record hit entity global id and type.
    bool has_entity_info = false;

    /// Whether to record hit body, segment and voxel id.
    bool has_entity_detail = false;

    /// Whether to record hit material id and emission.
    bool has_material = false;

    /// Whether to record world-space hit normal.
    bool has_normal = false;

    /// Whether to record hit velocity, relative velocity and radial velocity.
    bool has_velocity = false;

    bool operator==(const RSC_RaySensorRecord& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_RecordBase::serialize(s);
        s("has_hit", has_hit);
        s("has_entity_info", has_entity_info);
        s("has_entity_detail", has_entity_detail);
        s("has_material", has_material);
        s("has_normal", has_normal);
        s("has_velocity", has_velocity);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Signal for ray sensor.
 */
class RSC_RaySensorSignal : public RSC_SignalBase {
   public:
    RSC_RaySensorSignal() { name = "default"; }

    /// Number of position signals.
    RSize position_signal_num = 1;

    /// Number of orientation signals.
    RSize orientation_signal_num = 1;

    bool operator==(const RSC_RaySensorSignal& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_SignalBase::serialize(s);
        s("position_signal_num", position_signal_num);
        s("orientation_signal_num", orientation_signal_num);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A ray sensor that casts a fixed set of rays and reports the closest hit per ray.
 */
class RSC_RaySensorBase : public RSC_EntityBase {
   public:
    /// Ray directions relative to the sensor origin.
    std::vector<RVec3rf> ray_directions;

    /// Maximum ray distance in meters.
    RFloat max_range = 100.0;

    /// Ray sensor record.
    RSC_RaySensorRecord record;

    /// Ray sensor signal.
    RSC_RaySensorSignal signal;

    bool operator==(const RSC_RaySensorBase& other) const {
        return RSC_EntityBase::operator==(other) && ray_directions == other.ray_directions &&
               max_range == other.max_range && record == other.record && signal == other.signal;
    }

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_EntityBase::serialize(s);
        s("ray_directions", ray_directions);
        s("max_range", max_range);
        s("record", record);
        s("signal", signal);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
};

/**
 * @brief A free ray sensor that can be moved and rotated freely.
 */
class RSC_RaySensorFree : public RSC_RaySensorBase {
   public:
    /// The origin position of the sensor.
    RVec3rf origin_position = RVec3rf::zero();

    /// The orientation of the sensor.
    RQuat3rf orientation = RQuat3rf::unit();

    bool operator==(const RSC_RaySensorFree& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_RaySensorBase::serialize(s);
        s("origin_position", origin_position);
        s("orientation", orientation);
    }

   protected:
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A ray sensor that is rigidly linked to a rigid body.
 */
class RSC_RaySensorRigidBodyLinked : public RSC_RaySensorBase {
   public:
    /// The name of the structure that the sensor is linked to.
    std::string structure_name;

    /// The index of the body that the sensor is linked to.
    RIndex body_sid = RS_NULL_INDEX;

    /// The index of the segment that the sensor is linked to.
    /// Only valid if the structure is a voxel structure.
    RIndex segment_bid = RS_NULL_INDEX;

    /// The relative anchor point of the sensor in structure coordinate system.
    RVec3rf relative_anchor = RVec3rf::zero();

    /// The relative orientation of the sensor in structure coordinate system.
    RQuat3rf relative_orientation = RQuat3rf::unit();

    bool operator==(const RSC_RaySensorRigidBodyLinked& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_RaySensorBase::serialize(s);
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

#endif  // RSC_RAY_SENSOR_H
