// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RAY_SENSOR_M_H
#define RS_RAY_SENSOR_M_H

#include <refl.hpp>

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_conf.h"

namespace rise {

namespace logic {

RS_ENUM(
    RSE_RaySensorType,
    unsigned int,
    (RSE_RAY_SENSOR_FREE, 0),
    (RSE_RAY_SENSOR_RIGID_BODY_LINKED, 1)
)

union RS_RaySensorStateUnion {
    RState ray_sensor_state_packed;
    struct RS_RaySensorState {
        RSE_RaySensorType type : 1;
        RState has_hit : 1;
        RState has_entity_info : 1;
        RState has_entity_detail : 1;
        RState has_material : 1;
        RState has_normal : 1;
        RState has_velocity : 1;
    } ray_sensor_state;
};

template <RSE_FloatType FloatMemAccuracy>
class RS_RaySensorMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;
    using Quat3M = RS_Quat3DMem<FloatMemAccuracy>;
    using PositionVec = RS_Vec3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;

    /**
     * Pre-set attributes
     */
    /// Global id of the ray sensor entity.
    RIndex gid = RS_NULL_INDEX;

    /// Structure id of the rigid body the sensor rides. (rigid-body-linked only)
    RIndex rigid_body_sid = RS_NULL_INDEX;

    /// Number of rays.
    RSize ray_num = 0;

    /// Maximum ray distance in meters.
    FloatM max_range = 0;

    /// Handle of the sensor's pose signal entry. (free only)
    RHandle sensor_signal_handle = RS_NULL_HANDLE;

    /// Mount position relative to the rigid body. (rigid-body-linked only)
    PositionVec rigid_body_anchor_rb_position = PositionVec::zero();

    /// Mount orientation relative to the rigid body. (rigid-body-linked only)
    Quat3M rigid_body_anchor_rb_orientation = Quat3M::unit();

    /**
     * Post-set attributes
     */
    /// Handle of the resolved rigid body. (rigid-body-linked only)
    RHandle rigid_body = RS_NULL_HANDLE;

    /// Packed ray sensor state flags, see RS_RaySensorStateUnion.
    RState state = 0;

    /// Current world position.
    PositionVec position = PositionVec::zero();

    /// Current world orientation.
    Quat3M orientation = Quat3M::unit();

    /// World position at the last pose change.
    PositionVec prev_position = PositionVec::zero();

    /// World time of the last pose change.
    RBigFloat last_update_time = 0;

    /// World velocity of the sensor origin.
    PositionVec velocity = PositionVec::zero();
};

}  // namespace logic

}  // namespace rise

#define REFLECT_RAY_SENSOR_MEM(ray_sensor_mem_type) \
    REFL_AUTO(                                      \
        type(ray_sensor_mem_type),                  \
        field(gid),                                 \
        field(rigid_body_sid),                      \
        field(ray_num),                             \
        field(max_range),                           \
        field(sensor_signal_handle),                \
        field(rigid_body_anchor_rb_position),       \
        field(rigid_body_anchor_rb_orientation),    \
        field(rigid_body),                          \
        field(state),                               \
        field(position),                            \
        field(orientation),                         \
        field(prev_position),                       \
        field(last_update_time),                    \
        field(velocity)                             \
    )

REFLECT_RAY_SENSOR_MEM(rise::logic::RS_RaySensorMem<rise::RSE_FLOAT_32>)
REFLECT_RAY_SENSOR_MEM(rise::logic::RS_RaySensorMem<rise::RSE_FLOAT_64>)

#endif  // RS_RAY_SENSOR_M_H
