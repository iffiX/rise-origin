// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_MANAGER
#define RS_SIMULATION_MANAGER

#include <vector>

#include "rise_origin/core/rs_simulation.h"
#include "rise_origin/core/rs_simulation_callback.h"
#include "rise_origin/core/rs_simulation_local_handle.h"
#include "rise_origin/rsc/rs_config.h"
#include "rise_origin/utils/rs_logging.h"
#include "rise_origin/utils/rs_memory.h"
#include "rise_origin/utils/rs_thread_pool.h"

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

    // init() -> runSim() -> free() (blocking)
    void init();
    void free();

    /**
     * @brief Run a simulation
     * @param config Config for defining the kernel.
     * @param id Unique unsigned integer id of the kernel.
     * @param callback Simulation callback for control & modification of the simulation.
     * @note Execution settings (update intervals, record buffer size, max time, save
     *       flag, compute/memory accuracy) are read from config.simulation_config's
     *       runtime_* fields.
     * @note Config must be validated before used by RS_SimulationManager
     */
    RS_SimulationLocalHandle runSim(
        RS_Config config, RIndex id, RS_SimulationCallback callback = {}
    );

   private:
    // Initialized in init()
    std::shared_ptr<logging::logger> _sm_logger;

    std::string _getLogFormatter(RIndex simulation_index = RS_NULL_INDEX);

    bool _initSim(const std::shared_ptr<RS_Simulation> &sim, const RS_Config &config);
    void _runSim(const RS_Config &config, RS_SimulationExecutor &exec, RFloat max_time);

    void _addSimRecorder(
        const std::shared_ptr<RS_Simulation> &sim,
        const RS_RecorderConfig &recorder_config
    );
    void _syncSimRecorders(const std::shared_ptr<RS_Simulation> &sim);
    void _flushSimRecorders(const std::shared_ptr<RS_Simulation> &sim);
    void _finishSimRecorders(const std::shared_ptr<RS_Simulation> &sim);
};

}  // namespace rise

#endif  // RS_SIMULATION_MANAGER
