// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_RECORDER_H
#define RS_SIMULATION_RECORDER_H

#include <string>
#include <vector>

#include "rise_origin/core/rs_world_record.h"

namespace rise {

/**
 * RS_SimulationRecorder is the base class for performing recording actions
 * on a simulation. Several typical use cases include:
 * 1. Record frame information for visualization
 * 2. Process frame information and record results for analysis or event detection
 *
 * Usually, complete frame information is not viable for storing when there are millions
 * to billions of voxels, therefore only a part of all information is stored.
 */
class RS_SimulationRecorder {
   public:
    virtual ~RS_SimulationRecorder() = default;

    /// Recorder name.
    virtual std::string getName() const = 0;

    virtual void start(){};

    virtual void syncStructureVoxelRecords(
        const std::string &name,
        const std::vector<RS_WorldStructureVoxelRecord> &records
    ){};

    virtual void syncCameraPerspectiveRecords(
        const std::string &name,
        const std::vector<RS_WorldCameraPerspectiveRecord> &records
    ){};

    virtual void syncCameraStereoRecords(
        const std::string &name,
        const std::vector<RS_WorldCameraStereoRecord> &records
    ){};

    virtual void syncRaySensorRecords(
        const std::string &name,
        const std::vector<RS_WorldRaySensorRecord> &records
    ){};

    virtual void syncIntersectRegionRecords(
        const std::string &name,
        const std::vector<RS_WorldIntersectRegionRecord> &records
    ){};

    /// Flush + close the underlying storage.
    virtual void finish(){};
};

}  // namespace rise

#endif  // RS_SIMULATION_RECORDER_H
