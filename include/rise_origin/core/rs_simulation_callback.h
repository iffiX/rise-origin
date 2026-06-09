// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_CALLBACK_H
#define RS_SIMULATION_CALLBACK_H

#include <functional>
#include <tuple>
#include <vector>

#include "rise_origin/core/rs_simulation_controller.h"
#include "rise_origin/core/rs_simulation_handle.h"
#include "rise_origin/utils/rs_any.h"

namespace rise {

/**
 * @brief Type alias for simulation callback function.
 *
 * This callback function is called to control a simulation.
 *
 * @param id Unique id of the simulation.
 * @param controllers Controllers of the simulation.
 */
using RS_SimulationCallback =
    std::function<void(RIndex, std::vector<std::pair<RSE_SimulationControllerType, RS_Any>>)>;

}  // namespace rise

#endif  // RS_SIMULATION_CALLBACK_H
