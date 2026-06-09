// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_RECORDER_H
#define RS_SIMULATION_RECORDER_H

#include <vector>

#include "rise_origin/core/rs_simulation_record.h"
#include "rise_origin/rsc/rs_recorder_config.h"

namespace rise {

/**
 * RS_SimulationRecorder is the base class for performing recording actions
 * on a simulation. Several typical use cases include:
 * 1. Record *necessary* (i.e. only surface voxels) frame information for visualization
 * 2. Process frame information and record results for analysis or event detection
 *
 * Usually, complete frame information is not viable for storing when there are millions
 * to billions of voxels, therefore only a part of all information is stored.
 */
class RS_SimulationRecorder {
   public:
    virtual ~RS_SimulationRecorder() = default;
    /// The put function stores the corresponding row to the underlying storage
    /// indexed by \p step.
    /// Note: Most record formats only allows to create a new row with a higher
    /// step counter
    /// Note: \p append is set to true, the \p time_point must match
    /// exactly to ensure successful appending of \p voxels, \p links and \p rigid_bodies
    virtual void start() {};
    virtual void syncStructureRecords(
        const std::string &name,
        const std::vector<RS_SimulationStructureRecord> &records
    ) {};
    virtual void syncCameraRecords(
        const std::string &name,
        const std::vector<RS_SimulationCameraRecord> &records
    ) {};

    /// finish writing to the underlying storage
    virtual void finish() {};
};

}  // namespace rise

#endif  // RS_SIMULATION_RECORDER_H
