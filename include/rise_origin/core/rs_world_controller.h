// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_CONTROLLER_H
#define RS_WORLD_CONTROLLER_H

#include <string>
#include <vector>

#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/utils/data/rs_variant.h"

namespace rise {

/**
 * @brief Variant over every runtime record a controller may observe.
 * @details A record is a host-side snapshot of entity state staged for a
 * controller (structure voxel state, camera frame, etc.).
 */
using RS_WorldRecordVariant = RS_Variant<
    RS_WorldStructureVoxelRecord,
    RS_WorldCameraPerspectiveRecord,
    RS_WorldCameraStereoRecord,
    RS_WorldRaySensorRecord,
    RS_WorldIntersectRegionRecord>;

/**
 * @brief Variant over every runtime signal a controller may drive.
 * @details A signal is a host-side writable value pushed to the device each
 * control step (expansion / hinge / prismatic, camera pose, force, torque).
 */
using RS_WorldSignalVariant = RS_Variant<
    RS_WorldStructureVoxelSignal,
    RS_WorldCameraPerspectiveSignal,
    RS_WorldCameraStereoSignal,
    RS_WorldRaySensorSignal,
    RS_WorldForceSignal,
    RS_WorldTorqueSignal,
    RS_WorldTransformSignal>;

/**
 * @brief A controller: a general, named I/O container.
 * @details An I/O mechanism that reads the state (records) and controls the
 * behavior (signals) of one or more entities. It is generic across domains and
 * usable in various cases: passed through the in-process world callback, driven
 * over a network, etc. The kernel manager aggregates every record and signal
 * bound to this controller's name into the two vectors.
 */
class RS_WorldController {
   public:
    std::string name;
    std::vector<RS_WorldRecordVariant> records;
    std::vector<RS_WorldSignalVariant> signals;
};

}  // namespace rise

#endif  // RS_WORLD_CONTROLLER_H
