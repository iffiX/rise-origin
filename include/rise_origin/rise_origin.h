// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RISE_ORIGIN_H
#define RISE_ORIGIN_H

#if defined(__GNUC__) && (__GNUC__ < 11)
#error "Minimum GCC version required is 11"
#endif

#include <memory>

#include "rise_origin/core/rs_simulation_callback.h"
#include "rise_origin/core/rs_simulation_handle.h"
#include "rise_origin/rsc/rs_config.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_device.h"

using namespace rise;

class RiseOriginImpl;

/**
 * @brief The RiseOrigin simulator
 */
class RiseOrigin {
   public:
    /**
     * @brief Constructs a local RiseOrigin simulator instance with the specified device.
     *
     * @param device Device identifiers used by this simulator process.
     */
    explicit RiseOrigin(const RS_DeviceLocation &device = {RSE_DEVICE_TYPE_CPU, 0});

    /**
     * @brief Runs a simulation using a configuration provided as a vector of strings.
     *
     * Execution settings (update intervals, record buffer size, max time, save flag,
     * compute/memory accuracy, log level) are read from the configuration's
     * @ref RS_SimulationConfig `runtime_*` fields.
     *
     * @param config Vector of strings representing the simulation configuration.
     * @param id Unique unsigned integer id of the kernel.
     * @param callback Simulation callback for control & modification of the simulation.
     * @return Shared pointer to the simulation handle.
     */
    std::shared_ptr<RS_SimulationHandle> runSim(
        std::vector<std::string> config, RIndex id, RS_SimulationCallback callback = {}
    );

    /**
     * @brief Runs a simulation using a configuration provided as an RS_Config object.
     *
     * Execution settings are read from the configuration's @ref RS_SimulationConfig
     * `runtime_*` fields.
     *
     * @param config RS_Config object representing the simulation configuration.
     * @param id Unique unsigned integer id of the kernel.
     * @param callback Simulation callback for control & modification of the simulation.
     * @return Shared pointer to the simulation handle.
     */
    std::shared_ptr<RS_SimulationHandle> runSim(
        RS_Config config, RIndex id, RS_SimulationCallback callback = {}
    );

   private:
    std::shared_ptr<RiseOriginImpl> _rise_impl = nullptr;
};

#endif
