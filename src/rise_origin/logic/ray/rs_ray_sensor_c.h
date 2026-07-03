// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RAY_SENSOR_C_H
#define RS_RAY_SENSOR_C_H

#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/ray/rs_ray_sensor_m.h"
#include "rise_origin/logic/signal/rs_ray_sensor_signal_m.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_conf.h"

namespace rise {

namespace logic {

class RSCF_ResolveRigidBodyLinkedRaySensorDependency
    : public device::RS_ComputeFunction<RSCF_ResolveRigidBodyLinkedRaySensorDependency> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Kernel* k, RHandle ray_sensor, RSize rigid_body_segment_offset) {
        auto& ctx = k->ctx;
        RIndex rigid_body_sid = RAY_SENSOR_G(rigid_body_sid);
        RHandle rigid_body = rigid_body_sid + rigid_body_segment_offset;

        RAY_SENSOR_S(rigid_body, rigid_body);

        // Use double for universal precision
        RVec3df origin = RB_G(origin);
        RQuat3df orientation =
            RQuat3df(RB_G(orientation)) * RQuat3df(RAY_SENSOR_G(rigid_body_anchor_rb_orientation));
        RVec3df position =
            origin + orientation.rotateVec3D(RVec3df(RAY_SENSOR_G(rigid_body_anchor_rb_position)));
        RAY_SENSOR_S(position, position);
        RAY_SENSOR_S(orientation, orientation);
    }
};

template <RSE_FloatType FloatComputeAccuracy = RSE_FLOAT_32>
class RS_RaySensorCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;

    class RSCF_UpdateRaySensor : public device::RS_ComputeFunction<RSCF_UpdateRaySensor> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle ray_sensor) {
            if (k == nullptr || ray_sensor == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;

            RS_RaySensorStateUnion state_union{RAY_SENSOR_G(state)};
            bool is_rigid = state_union.ray_sensor_state.type == RSE_RAY_SENSOR_RIGID_BODY_LINKED;

            Vec3C position;
            Quat3C orientation;
            if (is_rigid) {
                RHandle rigid_body = RAY_SENSOR_G(rigid_body);
                if (rigid_body == RS_NULL_HANDLE) return;
                Quat3C rigid_orientation = RB_G(rigid_body, orientation);
                orientation =
                    rigid_orientation * Quat3C(RAY_SENSOR_G(rigid_body_anchor_rb_orientation));
                position = Vec3C(RB_G(rigid_body, origin)) +
                           rigid_orientation.rotateVec3D(
                               Vec3C(RAY_SENSOR_G(rigid_body_anchor_rb_position))
                           );
            } else {
                RHandle sensor_signal_handle = RAY_SENSOR_G(sensor_signal_handle);
                if (sensor_signal_handle == RS_NULL_HANDLE) return;
                auto& signal = ctx.ray_sensor_signals[sensor_signal_handle];
                position = Vec3C(signal.signal.position_signals.position[0]);
                orientation = Quat3C(signal.signal.orientation_signals.orientation[0]).normalized();
            }

            // Self velocity from finite difference across pose changes. All rays share one
            // origin, so only its linear velocity matters.
            Vec3C prev = Vec3C(RAY_SENSOR_G(prev_position));
            if (position.x != prev.x || position.y != prev.y || position.z != prev.z) {
                FloatC now = FloatC(k->getDeviceTime());
                FloatC dt = now - FloatC(RAY_SENSOR_G(last_update_time));
                if (dt > FloatC(1e-6)) {
                    RAY_SENSOR_S(velocity, (position - prev) / dt);
                }
                RAY_SENSOR_S(prev_position, position);
                RAY_SENSOR_S(last_update_time, now);
            }

            RAY_SENSOR_S(position, position);
            RAY_SENSOR_S(orientation, orientation);
        }
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_RAY_SENSOR_C_H
