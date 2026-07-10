// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_DEF_H
#define RS_WORLD_DEF_H

#include "rise_origin/utils/rs_def.h"

namespace rise {

/**
 * @brief Lifecycle status of a world.
 * @details Values at or above RSE_WORLD_MAX_NORMAL_STATE, plus RSE_WORLD_END_SUCCESS and
 * RSE_WORLD_END_TERMINATED, are terminal ends as reported by isWorldEnd.
 */
RS_ENUM(
    RSE_WorldStatus,
    unsigned int,
    (RSE_WORLD_END_SUCCESS, 0),
    (RSE_WORLD_END_TERMINATED, 1),
    (RSE_WORLD_END_FAILED_WITH_INVALID_CONFIG, 100),
    (RSE_WORLD_END_FAILED_WITH_SIMULATION_EXCEPTION, 101),
    (RSE_WORLD_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION, 102),
    (RSE_WORLD_UNINITIALIZED, 10),
    (RSE_WORLD_INITIALIZING, 11),
    (RSE_WORLD_RUNNING, 12),
    (RSE_WORLD_PAUSED, 13),
    (RSE_WORLD_TERMINATING, 14),
    (RSE_WORLD_MAX_NORMAL_STATE, 99)
)

/**
 * @brief Kind of a world entity.
 */
RS_ENUM(
    RSE_WorldEntityType,
    unsigned int,
    (RSE_WORLD_ENTITY_TYPE_COSMIC_BACKGROUND, 0),
    (RSE_WORLD_ENTITY_TYPE_FLOOR, 1),
    (RSE_WORLD_ENTITY_TYPE_STRUCTURE_VOXEL, 2)
)

inline bool isWorldEnd(const RSE_WorldStatus &status) {
    return status == RSE_WORLD_END_SUCCESS || status == RSE_WORLD_END_TERMINATED ||
           status > RSE_WORLD_MAX_NORMAL_STATE;
}

}  // namespace rise

#endif  // RS_WORLD_DEF_H
