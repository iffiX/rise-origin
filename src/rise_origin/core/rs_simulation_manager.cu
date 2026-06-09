// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <execution>
#include <filesystem>
#include <memory>
#include <thread>
#include <utility>

#include "rise_origin/core/recorder/rs_simulation_hdf5_recorder.h"
#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/core/rs_simulation_executor.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_format.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rs_simulation_manager.h"

using namespace std;
using namespace rise::device;
namespace fs = std::filesystem;

namespace rise {

RS_SimulationManager::RS_SimulationManager(const RS_DeviceLocation &device) : device(device) {
    std::string error;
    if (!isDeviceValid(device, error)) {
        RS_THROW(RS_InvalidArgumentException("Device is not valid: " + error));
    }
}

void RS_SimulationManager::init() {
    if (logging::get("root")) _sm_logger = logging::get("root")->clone("simulation manager");
    host_provider = getMemoryProvider({RSE_DEVICE_TYPE_CPU, 0});
    device_provider = getMemoryProvider(device);
    if (_sm_logger) _sm_logger->info(_getLogFormatter(), format("simulation manager initialized."));
}

void RS_SimulationManager::free() {
    if (executor_thread) {
        executor_thread->join();
        simulation = nullptr;
        host_provider = nullptr;
        device_provider = nullptr;
        RS_LOG(_sm_logger, info, _getLogFormatter(), format("simulation manager freed."));
    }
}

RS_SimulationLocalHandle
RS_SimulationManager::runSim(RS_Config config, RIndex id, RS_SimulationCallback callback) {
    if (executor_thread)
        RS_THROW(
            RS_HostExecutionException("A simulation is already run by the simulation manager.")
        );

    // Execution settings live in the config's runtime_* fields. The executor caches
    // the per-step ones in its init(); here we only need the manager-level ones.
    const auto &sim_config = config.simulation_config;
    RFloat max_time = sim_config.runtime_max_time;

    simulation = make_shared<RS_Simulation>();
    simulation->kernel_manager = make_shared<RS_KernelManager>();
    simulation->setStatusAndStatusInfo(RSE_SIMULATION_UNINITIALIZED, "");
    simulation->simulation_memory_accuracy = sim_config.runtime_simulation_memory_accuracy;
    simulation->id = id;

    this->record_buffer_size =
        sim_config.runtime_save_record ? RS_MAX(1, sim_config.runtime_record_buffer_size) : 0;

    RS_LOG(
        _sm_logger,
        debug,
        _getLogFormatter(),
        format(
            "Runtime configs:\n"
            "dt_update_interval: {}\n"
            "rigidbody_update_interval: {}\n"
            "contact_update_interval: {}\n"
            "mechanical_signal_update_interval: {}\n"
            "divergence_check_interval: {}\n"
            "record_buffer_size: {}\n"
            "max_time: {}\n"
            "save_record: {}\n"
            "simulation_compute_accuracy: {}\n"
            "simulation_memory_accuracy: {}",
            sim_config.runtime_dt_update_interval,
            sim_config.runtime_rigidbody_update_interval,
            sim_config.runtime_contact_update_interval,
            sim_config.runtime_mechanical_signal_update_interval,
            sim_config.runtime_divergence_check_interval,
            this->record_buffer_size,
            max_time,
            sim_config.runtime_save_record,
            enumToString(sim_config.runtime_simulation_compute_accuracy),
            enumToString(sim_config.runtime_simulation_memory_accuracy)
        )
    );

    RS_SimulationLocalHandle handle(simulation);

    executor_thread = make_unique<thread>(
        [this, config = std::move(config), callback = std::move(callback), max_time]() {
            auto exec = RS_SimulationExecutor(device_provider.get(), host_provider.get(), callback);
            _runSim(config, exec, max_time);
        }
    );

    return handle;
}

string RS_SimulationManager::_getLogFormatter(RIndex simulation_id) {
    if (simulation_id != RS_NULL_INDEX) {
        return format("(SimManager) Simulation {} {{}}", simulation_id);
    } else
        return format("(SimManager) {{}}");
}

bool RS_SimulationManager::_initSim(
    const std::shared_ptr<RS_Simulation> &sim,
    const RS_Config &config
) {
    sim->setStatusAndStatusInfo(RSE_SIMULATION_INITIALIZING, "");

    if (config.is_valid) {
        try {
            sim->kernel_manager->init(
                host_provider.get(), device_provider.get(), this->record_buffer_size
            );

            // Pause command execution until we have fully pre-initialized with provided RS_Config
            sim->kernel_manager->pauseCommandExecution();

            sim->kernel_manager->initKernel(sim->id, sim->simulation_memory_accuracy);
            sim->kernel_manager->setSimulationConfig(config.simulation_config, true);
            sim->kernel_manager->setWorld(config.world_config, true);
            sim->kernel_manager->setFloor(config.floor_config, true);
            for (auto &material_config : config.material_configs)
                sim->kernel_manager->addMaterial(material_config, true);
            for (auto &structure_config : config.structure_configs)
                sim->kernel_manager->addStructure(structure_config, true);

            // Records depend on structures, so add them after structures.
            for (auto &record_config : config.record_configs)
                sim->kernel_manager->addRecord(record_config, true);

            // Cameras may depend on structures, so add them after structures.
            for (auto &camera_config : config.camera_configs)
                sim->kernel_manager->addCamera(camera_config, true);

            // Signals depend on structures, so add them after structures.
            for (auto &signal_config : config.signal_configs)
                sim->kernel_manager->addSignal(signal_config, true);

            sim->kernel_manager->resumeCommandExecution();
        } catch (RS_BaseException &e) {
            sim->setStatusAndStatusInfo(RSE_SIMULATION_END_FAILED_WITH_INVALID_CONFIG, e.what());
            sim->kernel_manager->terminateKernel(true);
            sim->kernel_manager->freeKernel();
            sim->kernel_manager->free();
            return false;
        } catch (...) {
            sim->setStatusAndStatusInfo(
                RSE_SIMULATION_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION, "unknown error"
            );
            sim->kernel_manager->terminateKernel(true);
            sim->kernel_manager->freeKernel();
            sim->kernel_manager->free();
            return false;
        }
    } else {
        sim->setStatusAndStatusInfo(
            RSE_SIMULATION_END_FAILED_WITH_INVALID_CONFIG, config.invalid_reason
        );
        return false;
    }
    if (sim->getStatusAndStatusInfo().first != RSE_SIMULATION_END_FAILED_WITH_INVALID_CONFIG) {
        if (record_buffer_size != 0) {
            for (auto &recorder_config : config.recorder_configs)
                _addSimRecorder(sim, recorder_config);
        }
    }
    return true;
}

void RS_SimulationManager::_runSim(
    const RS_Config &config,
    RS_SimulationExecutor &exec,
    RFloat max_time
) {
    auto &sim = simulation;

    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "begin simulation initialization");
    bool is_inited = _initSim(sim, config);
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "end simulation initialization");

    if (!is_inited) {
        auto [status, status_info] = sim->getStatusAndStatusInfo();
        RS_LOG(
            _sm_logger,
            error,
            _getLogFormatter(sim->id),
            format(
                "produced following error during initialization. "
                "The simulation will be skipped. \nError: {}.",
                status_info
            )
        );
        return;
    }

    auto &kernel = sim->kernel_manager->kernel;
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "begin executor initialization.");
    exec.init(sim->id, &kernel, sim->kernel_manager.get());
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "end executor initialization.");

    sim->setStatusAndStatusInfo(RSE_SIMULATION_RUNNING, "");

    RSE_SimulationStatus end_status = RSE_SIMULATION_END_SUCCESS;
    while (kernel.getHostTime() < max_time) {
        if (kernel.isPaused()) {
            if (sim->status != RSE_SIMULATION_PAUSED) {
                sim->setStatusAndStatusInfo(RSE_SIMULATION_PAUSED, "");
                RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "paused.");
            }
            this_thread::sleep_for(chrono::milliseconds(10));
            continue;
        }
        if (kernel.isStopConditionMet() || kernel.isTerminated()) {
            sim->setStatusAndStatusInfo(RSE_SIMULATION_TERMINATING, "");
            RS_LOG(
                _sm_logger,
                info,
                _getLogFormatter(sim->id),
                format("ended at {}s.", kernel.getHostTime())
            );

            if (!kernel.isStopConditionMet()) {
                end_status = RSE_SIMULATION_END_TERMINATED;
            } else {
                end_status = RSE_SIMULATION_END_SUCCESS;
            }

            _flushSimRecorders(sim);
            break;
        }

        if (!exec.step()) {
            end_status = RSE_SIMULATION_END_FAILED_WITH_SIMULATION_EXCEPTION;
            sim->setStatusAndStatusInfo(RSE_SIMULATION_TERMINATING, "");
            RS_LOG(_sm_logger, warn, _getLogFormatter(sim->id), "has invalid solution.");

            _flushSimRecorders(sim);
            break;
        }

        // Save records
        if (record_buffer_size != 0 &&
            kernel.getExecutionCountOfEvent("record") == record_buffer_size) {
            kernel.clearExecutionCountOfEvent("record");
            _syncSimRecorders(sim);
        }
    }
    sim->kernel_manager->terminateKernel(true);
    if (!sim->recorders.empty()) {
        if (sim->recorder_sync_future.valid()) sim->recorder_sync_future.wait();
        RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "finishing recorders");
        _finishSimRecorders(sim);
        sim->recorders.clear();
        RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "recorders finished");
    }

    exec.free();
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "executor freed\n");
    sim->kernel_manager->freeKernel();
    sim->kernel_manager->free();
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "kernel freed");
    sim->setStatusAndStatusInfo(end_status, "");
}

void RS_SimulationManager::_addSimRecorder(
    const std::shared_ptr<RS_Simulation> &sim,
    const RS_RecorderConfig &recorder_config
) {
    if (recorder_config.type == RSE_RECORDER_HDF5) {
        auto &config = recorder_config.config.as<RS_HDF5RecorderConfig>();
        auto parent_dir = fs::path(config.path).parent_path();
        if (!parent_dir.empty() && !fs::exists(parent_dir)) fs::create_directories(parent_dir);
        sim->recorders.push_back(std::move(make_shared<RS_SimulationHDF5Recorder>(config.path)));
    }
}

void RS_SimulationManager::_syncSimRecorders(const std::shared_ptr<RS_Simulation> &sim) {
    // Wait for remaining recorder finish their work
    sim->waitForRecorderSync();

    // Copy records from device to host
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "record sync begin");
    sim->kernel_manager->syncRecorderRecords();

    // Put records into recorders
    sim->setRecorderSyncFuture(thread_pool.submitTask([&]() {
        RS_KernelManagerGuard guard(sim->kernel_manager.get());

        sim->kernel_manager->waitForRecorderRecordsSync();

        auto structure_records = sim->kernel_manager->getRecorderStructureRecords();
        std::for_each(
            std::execution::par,
            sim->recorders.begin(),
            sim->recorders.end(),
            [&](const auto &recorder) {
                for (auto &&[name, records] : structure_records)
                    recorder->syncStructureRecords(name, records);
            }
        );

        auto camera_records = sim->kernel_manager->getRecorderCameraRecords();
        std::for_each(
            std::execution::par,
            sim->recorders.begin(),
            sim->recorders.end(),
            [&](const auto &recorder) {
                for (auto &&[name, records] : camera_records)
                    recorder->syncCameraRecords(name, records);
            }
        );
        RS_LOG(_sm_logger, debug, _getLogFormatter(sim->id), "record sync is finished");
    }));
}

void RS_SimulationManager::_flushSimRecorders(const std::shared_ptr<RS_Simulation> &sim) {
    // Flush twice to ensure all records are flushed
    _syncSimRecorders(sim);
    _syncSimRecorders(sim);
    sim->waitForRecorderSync();
}

void RS_SimulationManager::_finishSimRecorders(const std::shared_ptr<RS_Simulation> &sim) {
    thread_pool.submitLoop(
                   0, sim->recorders.size(), [&](RIndex i) { sim->recorders[i]->finish(); }
    ).wait();
}

}  // namespace rise
