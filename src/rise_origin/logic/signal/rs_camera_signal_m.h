// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CAMERA_SIGNAL_M_H
#define RS_CAMERA_SIGNAL_M_H

#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

RS_ENUM(
    RSE_CameraSignalType,
    unsigned int,
    (RSE_CAMERA_SIGNAL_PERSPECTIVE, 0),
    (RSE_CAMERA_SIGNAL_STEREO, 1)
)

/**
 * @brief A camera signal slot that is either a perspective or a stereo world signal.
 * @details Both perspective and stereo cameras publish one entry into the shared camera signal
 * array; the type tag selects which arm a sub-camera reads. A stereo camera's two sub-cameras
 * share one stereo entry and pick their eye by the camera mem's sub_camera_index.
 */
struct RS_CameraSignalMem {
    RSE_CameraSignalType type = RSE_CAMERA_SIGNAL_PERSPECTIVE;

    union {
        RS_WorldCameraPerspectiveSignal perspective;
        RS_WorldCameraStereoSignal stereo;
    };

    HOST_DEVICE_INLINE_QUALIFIER RS_CameraSignalMem()
        : type(RSE_CAMERA_SIGNAL_PERSPECTIVE), perspective() {}

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER RS_WorldRootSignal& root() {
        if (type == RSE_CAMERA_SIGNAL_STEREO) {
            return static_cast<RS_WorldRootSignal&>(stereo);
        }
        return static_cast<RS_WorldRootSignal&>(perspective);
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_CAMERA_SIGNAL_M_H
