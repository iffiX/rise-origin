// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RAY_SENSOR_RECORD_M_H
#define RS_RAY_SENSOR_RECORD_M_H

#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * @brief Device-side ray sensor record instantiating every sub-record stream.
 * @details A manager passes object_num = 0 for any stream a sensor does not emit. Canonical frame
 * layout: root | hit | entity_info | entity_detail | material | normal | velocity | sensor_state.
 */
class RS_RaySensorRecordMem : public RS_WorldRaySensorRecordBase {
   public:
    /// Hit records, the size of each frame is ray num.
    Hit hit;

    /// Entity info records, the size of each frame is ray num.
    EntityInfo entity_info;

    /// Entity detail records, the size of each frame is ray num.
    EntityDetail entity_detail;

    /// Material records, the size of each frame is ray num.
    Material material;

    /// Normal records, the size of each frame is ray num.
    Normal normal;

    /// Velocity records, the size of each frame is ray num.
    Velocity velocity;

    /// Sensor state records, the size of each frame is 1.
    State sensor_state;

    RS_RaySensorRecordMem() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize hit_object_num,
        RBigSize entity_info_object_num,
        RBigSize entity_detail_object_num,
        RBigSize material_object_num,
        RBigSize normal_object_num,
        RBigSize velocity_object_num,
        RBigSize sensor_state_object_num = 1
    ) {
        return RS_WorldRootRecord::getRequiredMemorySize() +
               Hit::getRequiredMemorySize(hit_object_num) +
               EntityInfo::getRequiredMemorySize(entity_info_object_num) +
               EntityDetail::getRequiredMemorySize(entity_detail_object_num) +
               Material::getRequiredMemorySize(material_object_num) +
               Normal::getRequiredMemorySize(normal_object_num) +
               Velocity::getRequiredMemorySize(velocity_object_num) +
               State::getRequiredMemorySize(sensor_state_object_num);
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(
        void* memory,
        RBigSize hit_object_num,
        RBigSize entity_info_object_num,
        RBigSize entity_detail_object_num,
        RBigSize material_object_num,
        RBigSize normal_object_num,
        RBigSize velocity_object_num,
        RBigSize sensor_state_object_num = 1
    ) {
        is_valid = true;
        memory = RS_WorldRootRecord::initFromMemory(memory);
        memory = hit.initFromMemory(memory, hit_object_num);
        memory = entity_info.initFromMemory(memory, entity_info_object_num);
        memory = entity_detail.initFromMemory(memory, entity_detail_object_num);
        memory = material.initFromMemory(memory, material_object_num);
        memory = normal.initFromMemory(memory, normal_object_num);
        memory = velocity.initFromMemory(memory, velocity_object_num);
        memory = sensor_state.initFromMemory(memory, sensor_state_object_num);
        return memory;
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_RAY_SENSOR_RECORD_M_H
