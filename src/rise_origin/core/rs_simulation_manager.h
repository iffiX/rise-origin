// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_MANAGER
#define RS_SIMULATION_MANAGER

#include <utility>
#include <vector>

#include "rise_origin/core/rs_simulation.h"
#include "rise_origin/core/rs_world_callback.h"
#include "rise_origin/core/rs_world_local_handle.h"
#include "rise_origin/rsc/rsc_world.h"
#include "rise_origin/utils/rs_logging.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/concurrency/rs_thread_pool.h"

namespace rise {

class RS_SimulationExecutor;

class RS_SimulationManager {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    RS_DeviceLocation device{};
    RSize record_buffer_size = 0;
    std::unique_ptr<std::thread> executor_thread{nullptr};
    RS_ThreadPool thread_pool;
    std::shared_ptr<RS_Simulation> simulation;
    std::shared_ptr<RS_MemoryProvider> host_provider;
    std::shared_ptr<RS_MemoryProvider> device_provider;
    std::shared_ptr<RS_WorldCallback> callback;

    /*****************************************************************
     * Methods
     *****************************************************************/
    // Note: any sink to be used in the simulation needs to be added before calling init()
    // Note: the "root" spdlog logger should be registered before calling init()
    /**
     * @brief Create a new instance of the simulation manager.
     * @param device Device of the simulation manager.
     */
    explicit RS_SimulationManager(const RS_DeviceLocation &device = {RSE_DEVICE_TYPE_CPU, 0});

    // Copy constructor is deleted
    RS_SimulationManager(const RS_SimulationManager &) = delete;

    // init() -> run() -> free() (blocking)
    void init();
    void free();

    /**
     * @brief Run a world.
     * @param world The world characterization defining the simulation.
     * @return The world handle and the rebindable world callback the engine drives each step.
     * @note Execution settings (time stepping, record buffer size, max time, save flag,
     *       compute/memory accuracy) are read from world.simulation.
     * @note world must be validated before used by RS_SimulationManager.
     */
    std::pair<RS_WorldLocalHandle, std::shared_ptr<RS_WorldCallback>> run(rsc::RSC_World world);

   private:
    // Initialized in init()
    std::shared_ptr<logging::logger> _sm_logger;

    std::string _getLogFormatter(const std::string &world_name = "");

    bool _initSim(const std::shared_ptr<RS_Simulation> &sim, const rsc::RSC_World &world);
    void _runSim(const rsc::RSC_World &world, RS_SimulationExecutor &exec, RFloat max_time);

    void _syncSimRecorders(const std::shared_ptr<RS_Simulation> &sim);
    void _flushSimRecorders(const std::shared_ptr<RS_Simulation> &sim);
    void _finishSimRecorders(const std::shared_ptr<RS_Simulation> &sim);
};

}  // namespace rise

#endif  // RS_SIMULATION_MANAGER
