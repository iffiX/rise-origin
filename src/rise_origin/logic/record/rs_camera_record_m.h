// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CAMERA_RECORD_M_H
#define RS_CAMERA_RECORD_M_H

#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * @brief Unified camera record. Internal device-side type instantiating every camera sub-record
 * stream so one device vector serves both mono and stereo cameras.
 * @details A stereo camera renders two sub-cameras, each producing its own unified record; the host
 * stereo record is then assembled by pointing its left_/right_ fields at the two sub-records' fields.
 * A manager passes object_num = 0 for any stream it does not emit. Canonical frame layout:
 * root | rgb | depth | normal | motion | entity_info | camera_state.
 */
class RS_CameraRecordMem : public RS_WorldCameraRecordBase {
   public:
    /// Pixel records, the size of each frame is camera resolution (width * height).
    RGB rgb;

    /// Depth records, the size of each frame is camera resolution (width * height).
    Depth depth;

    /// Normal records, the size of each frame is camera resolution (width * height).
    Normal normal;

    /// Motion records, the size of each frame is camera resolution (width * height).
    Motion motion;

    /// Entity info records, the size of each frame is camera resolution (width * height).
    EntityInfo entity_info;

    /// Camera state records, the size of each frame is 1.
    State camera_state;

    RS_CameraRecordMem() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize rgb_object_num,
        RBigSize depth_object_num,
        RBigSize normal_object_num,
        RBigSize motion_object_num,
        RBigSize entity_info_object_num,
        RBigSize camera_state_object_num = 1
    ) {
        RBigSize base_size = RS_WorldRootRecord::getRequiredMemorySize();
        RBigSize rgb_size = RGB::getRequiredMemorySize(rgb_object_num);
        RBigSize depth_size = Depth::getRequiredMemorySize(depth_object_num);
        RBigSize normal_size = Normal::getRequiredMemorySize(normal_object_num);
        RBigSize motion_size = Motion::getRequiredMemorySize(motion_object_num);
        RBigSize entity_info_size = EntityInfo::getRequiredMemorySize(entity_info_object_num);
        RBigSize camera_state_size = State::getRequiredMemorySize(camera_state_object_num);
        return base_size + rgb_size + depth_size + normal_size + motion_size + entity_info_size +
               camera_state_size;
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(
        void* memory,
        RBigSize rgb_object_num,
        RBigSize depth_object_num,
        RBigSize normal_object_num,
        RBigSize motion_object_num,
        RBigSize entity_info_object_num,
        RBigSize camera_state_object_num = 1
    ) {
        is_valid = true;

        memory = RS_WorldRootRecord::initFromMemory(memory);
        memory = rgb.initFromMemory(memory, rgb_object_num);
        memory = depth.initFromMemory(memory, depth_object_num);
        memory = normal.initFromMemory(memory, normal_object_num);
        memory = motion.initFromMemory(memory, motion_object_num);
        memory = entity_info.initFromMemory(memory, entity_info_object_num);
        memory = camera_state.initFromMemory(memory, camera_state_object_num);

        return memory;
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_CAMERA_RECORD_M_H
