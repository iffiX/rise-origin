// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_CONTROLLER_H
#define RS_SIMULATION_CONTROLLER_H

#include <string>
#include <vector>

#include "rise_origin/core/rs_simulation_record.h"
#include "rise_origin/core/rs_simulation_signal.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

RS_ENUM(
    RSE_SimulationControllerType,
    unsigned int,
    (RSE_SIMULATION_STRUCTURE_CONTROLLER, 0),
    (RSE_SIMULATION_CAMERA_CONTROLLER, 1),
    (RSE_SIMULATION_FORCE_CONTROLLER, 2),
    (RSE_SIMULATION_TORQUE_CONTROLLER, 3)
);

class RS_SimulationStructureController {
   public:
    std::string name;
    RS_SimulationStructureRecord structure_record;
    std::vector<RS_SimulationCameraRecord> camera_records;
    RS_SimulationStructureSignal signal;
};

class RS_SimulationCameraController {
   public:
    std::string name;
    RS_SimulationCameraRecord camera_record;
    RS_SimulationCameraSignal signal;
};

class RS_SimulationForceController {
   public:
    std::string name;
    RS_SimulationForceSignal signal;
};

class RS_SimulationTorqueController {
   public:
    std::string name;
    RS_SimulationTorqueSignal signal;
};

}  // namespace rise

#endif  // RS_SIMULATION_CONTROLLER_H
