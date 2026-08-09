// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CAMERA_RECORD_C_H
#define RS_CAMERA_RECORD_C_H

#include "rise_origin/core/rs_layout.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_value.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/device/rs_constant_memory.h"
#include "rise_origin/logic/record/rs_camera_record_m.h"
#include "rise_origin/logic/vision/rs_camera_m.h"

namespace rise {

namespace logic {

class RSCF_StoreCameraRGBRecord : public device::RS_ComputeFunction<RSCF_StoreCameraRGBRecord> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_CameraRecordMem &record, RHandle camera, RHandle pixel) {
        if (k != nullptr && camera != RS_NULL_HANDLE && pixel != RS_NULL_HANDLE &&
            record.is_valid) {
            auto &ctx = k->ctx;
            auto &r = record.rgb;

            // Maps canvas radiance onto the full scale the sensor expects.
            float saturation_radiance = float(CA_G(saturation_radiance));
            RVec3f pixel_color = ctx.camera_current_rgb_canvas[camera][pixel];
            if (saturation_radiance > 0.0f) pixel_color = pixel_color / saturation_radiance;
            r.pixel[pixel] = RColor3f(pixel_color).linearToSrgb().clip().operator RColor3b();
        }
    }
};

class RSCF_StoreCameraDepthRecord : public device::RS_ComputeFunction<RSCF_StoreCameraDepthRecord> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_CameraRecordMem &record, RHandle camera, RHandle pixel) {
        if (k != nullptr && camera != RS_NULL_HANDLE && pixel != RS_NULL_HANDLE &&
            record.is_valid) {
            auto &ctx = k->ctx;

            float metric_depth = ctx.camera_aux_depth_z[camera][pixel];
            RS_CameraStateUnion state_union{CA_G(state)};
            if (state_union.camera_state.is_depth_hyperbolic) {
                float near_z = float(CA_G(hyperbolic_near_depth));
                float far_z = float(CA_G(hyperbolic_far_depth));
                float encoded;
                if (metric_depth <= 0.0f) {
                    // Cosmic background hit at the far plane.
                    encoded = 1.0f;
                } else {
                    float z = RS_BOUND(metric_depth, near_z, far_z);
                    encoded = far_z / (far_z - near_z) * (1.0f - near_z / z);
                }
                record.depth.pixel[pixel] = encoded;
            } else {
                record.depth.pixel[pixel] = metric_depth;
            }
        }
    }
};

class RSCF_StoreCameraNormalRecord
    : public device::RS_ComputeFunction<RSCF_StoreCameraNormalRecord> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_CameraRecordMem &record, RHandle camera, RHandle pixel) {
        if (k != nullptr && camera != RS_NULL_HANDLE && pixel != RS_NULL_HANDLE &&
            record.is_valid) {
            auto &ctx = k->ctx;
            record.normal.pixel[pixel] = ctx.camera_aux_normal[camera][pixel];
        }
    }
};

class RSCF_StoreCameraMotionRecord
    : public device::RS_ComputeFunction<RSCF_StoreCameraMotionRecord> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_CameraRecordMem &record, RHandle camera, RHandle pixel) {
        if (k != nullptr && camera != RS_NULL_HANDLE && pixel != RS_NULL_HANDLE &&
            record.is_valid) {
            auto &ctx = k->ctx;
            record.motion.pixel[pixel] = ctx.camera_aux_screen_motion[camera][pixel];
        }
    }
};

class RSCF_StoreCameraEntityInfoRecord
    : public device::RS_ComputeFunction<RSCF_StoreCameraEntityInfoRecord> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_CameraRecordMem &record, RHandle camera, RHandle pixel) {
        if (k != nullptr && camera != RS_NULL_HANDLE && pixel != RS_NULL_HANDLE &&
            record.is_valid) {
            auto &ctx = k->ctx;
            auto &r = record.entity_info;

            RIndex gid = ctx.camera_aux_object_key[camera][pixel].entity_gid;
            r.entity_gid[pixel] = gid;
            if (gid == RS_WORLD_RSC_COSMIC_BACKGROUND_GID) {
                r.entity_type[pixel] = RSE_WORLD_ENTITY_TYPE_COSMIC_BACKGROUND;
            } else if (gid == RS_WORLD_RSC_FLOOR_GID) {
                r.entity_type[pixel] = RSE_WORLD_ENTITY_TYPE_FLOOR;
            } else {
                r.entity_type[pixel] = RSE_WORLD_ENTITY_TYPE_STRUCTURE_VOXEL;
            }
        }
    }
};

class RSCF_StoreCameraStateRecord : public device::RS_ComputeFunction<RSCF_StoreCameraStateRecord> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_CameraRecordMem &record, RHandle camera, RIndex record_index) {
        if (k != nullptr && camera != RS_NULL_HANDLE && record.is_valid) {
            auto &ctx = k->ctx;
            auto &r = record.camera_state;

            r.position[record_index] = CA_G(position);
            r.orientation[record_index] = CA_G(orientation);
            r.focus_distance[record_index] = CA_G(focus_distance);
            r.field_of_view[record_index] = CA_G(field_of_view);
        }
    }
};

class RSCF_UpdateCameraRootRecord
    : public device::RS_ComputeFunction<RSCF_UpdateCameraRootRecord> {
   public:
    template <typename Kernel, typename Buffer, typename... Sizes>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k,
        Buffer &buffer,
        RS_CameraRecordMem &record,
        bool is_active,
        Sizes... record_sizes) {
        // A frozen camera stores nothing this step, so its record carries no reading at all.
        if (!is_active) {
            record.is_valid = false;
            return;
        }
        RBigSize record_data_size = record.getRequiredMemorySize(record_sizes...);
        RS_MemoryPtr chunk = buffer.pushDevice(record_data_size);
        if (chunk == nullptr) {
            record.is_valid = false;
        } else {
            record.initFromMemory(chunk.ptr(), record_sizes...);
            *record.time = k->getDeviceTime();
            *record.step = k->getDeviceStep();
        }
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_CAMERA_RECORD_C_H
