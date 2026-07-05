// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <memory>
#include <thread>
#include <utility>

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/core/rs_simulation_executor.h"
#include "rise_origin/core/rs_world_local_callback.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rise_origin/utils/string/rs_format.h"
#include "rs_simulation_manager.h"

using namespace std;
using namespace rise::device;
using namespace rise::rsc;

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
    if (_sm_logger) RS_LOG(_sm_logger, info, "Simulation manager initialized.");
}

void RS_SimulationManager::free() {
    if (executor_thread) {
        executor_thread->join();
        simulation = nullptr;
        host_provider = nullptr;
        device_provider = nullptr;
        RS_LOG(_sm_logger, info, "Simulation manager freed.");
    }
}

std::pair<RS_WorldLocalHandle, std::shared_ptr<RS_WorldCallback>> RS_SimulationManager::run(
    RSC_World world
) {
    if (executor_thread)
        RS_THROW(
            RS_HostExecutionException("A simulation is already run by the simulation manager.")
        );

    // Execution settings live in the world's simulation sub-characterization. The executor caches
    // the per-step ones in its init(); here we only need the manager-level ones.
    const RSC_Simulation &sim_rsc = world.simulation;
    RFloat max_time = sim_rsc.runtime.max_time;

    simulation = make_shared<RS_Simulation>();
    simulation->kernel_manager = make_shared<RS_KernelManager>();
    simulation->setStatusAndStatusInfo(RSE_WORLD_UNINITIALIZED, "");
    simulation->name = world.name;
    simulation->simulation_memory_accuracy = sim_rsc.runtime.memory_accuracy;

    this->record_buffer_size =
        sim_rsc.runtime.is_save_record ? RS_MAX(1, sim_rsc.runtime.record_buffer_size) : 0;

    RS_LOG(
        _sm_logger,
        debug,
        _getLogFormatter(world.name),
        format(
            "Runtime settings:\n"
            "record_buffer_size: {}\n"
            "max_time: {}\n"
            "is_save_record: {}\n"
            "compute_accuracy: {}\n"
            "memory_accuracy: {}",
            this->record_buffer_size,
            max_time,
            sim_rsc.runtime.is_save_record,
            enumToString(sim_rsc.runtime.compute_accuracy),
            enumToString(sim_rsc.runtime.memory_accuracy)
        )
    );

    RS_WorldLocalHandle handle(simulation);
    callback = make_shared<RS_WorldLocalCallback>(world.name);

    executor_thread = make_unique<thread>([this, world = std::move(world), max_time]() {
        auto exec =
            RS_SimulationExecutor(device_provider.get(), host_provider.get(), callback.get());
        _runSim(world, exec, max_time);
    });

    return {handle, callback};
}

string RS_SimulationManager::_getLogFormatter(const std::string &world_name) {
    if (!world_name.empty()) {
        return format("World {} {{}}", world_name);
    } else
        return format("World {{}}");
}

bool RS_SimulationManager::_initSim(
    const std::shared_ptr<RS_Simulation> &sim,
    const RSC_World &world
) {
    sim->setStatusAndStatusInfo(RSE_WORLD_INITIALIZING, "");

    try {
        RS_Timer timer;
        sim->validator.validate(world);
        RS_LOG(
            _sm_logger,
            debug,
            _getLogFormatter(sim->name),
            format("World validation takes {}", timer.elapsedTimeString())
        );
    } catch (RS_BaseException &e) {
        sim->setStatusAndStatusInfo(RSE_WORLD_END_FAILED_WITH_INVALID_CONFIG, e.what());
        return false;
    }

    try {
        sim->kernel_manager->init(
            host_provider.get(),
            device_provider.get(),
            this->record_buffer_size,
            [sim_ptr = sim.get()](RSE_WorldMessageLevel level, const std::string &text) {
                sim_ptr->postMessage(level, text);
            }
        );

        // Pause command execution until we have fully pre-loaded with the world.
        sim->kernel_manager->pauseCommandExecution();

        sim->kernel_manager->initKernel(sim->simulation_memory_accuracy);
        sim->kernel_manager->setSimulationRSC(world.simulation, true);

        // Order matters — providers before consumers:
        //   resources (textures) -> essences (materials reference textures)
        //   essences -> entities (structures/floor reference materials)
        //   entities -> records / controllers / signals (which reference entities)
        for (auto &resource : world.resources) {
            sim->kernel_manager->addResource(resource, true);
        }
        for (auto &essence : world.essences) {
            sim->kernel_manager->addEssence(essence, true);
        }
        for (auto &entity : world.entities) {
            sim->kernel_manager->addEntity(entity, true);
        }
        for (auto &record : world.records) {
            sim->kernel_manager->addRecord(record, true);
        }
        for (auto &signal : world.signals) {
            sim->kernel_manager->addSignal(signal, true);
        }

        sim->kernel_manager->resumeCommandExecution();
    } catch (RS_BaseException &e) {
        sim->setStatusAndStatusInfo(RSE_WORLD_END_FAILED_WITH_INVALID_CONFIG, e.what());
        sim->kernel_manager->terminateKernel(true);
        sim->kernel_manager->freeKernel();
        sim->kernel_manager->free();
        return false;
    } catch (...) {
        sim->setStatusAndStatusInfo(
            RSE_WORLD_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION, "unknown error"
        );
        sim->kernel_manager->terminateKernel(true);
        sim->kernel_manager->freeKernel();
        sim->kernel_manager->free();
        return false;
    }

    for (auto &controller : world.controllers) {
        sim->addController(controller);
    }

    if (record_buffer_size != 0) {
        for (auto &recorder : world.recorders) {
            sim->addRecorder(recorder);
        }
    }
    return true;
}

void RS_SimulationManager::_runSim(
    const RSC_World &world,
    RS_SimulationExecutor &exec,
    RFloat max_time
) {
    auto &sim = simulation;

    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "begin simulation initialization.");
    bool is_inited = _initSim(sim, world);
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "end simulation initialization.");

    if (!is_inited) {
        auto [status, status_info] = sim->getStatusAndStatusInfo();
        RS_LOG(
            _sm_logger,
            error,
            _getLogFormatter(sim->name),
            format(
                "produced following error during initialization. "
                "The simulation will be skipped. \nError: {}.",
                status_info
            )
        );
        return;
    }

    auto &kernel = sim->kernel_manager->kernel;
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "begin executor initialization.");
    exec.init(sim.get());
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "end executor initialization.");

    sim->setStatusAndStatusInfo(RSE_WORLD_RUNNING, "");

    RSE_WorldStatus end_status = RSE_WORLD_END_SUCCESS;
    while (kernel.getHostTime() < max_time) {
        if (kernel.isStopConditionMet() || kernel.isTerminated()) {
            sim->setStatusAndStatusInfo(RSE_WORLD_TERMINATING, "");
            RS_LOG(
                _sm_logger,
                info,
                _getLogFormatter(sim->name),
                format("ended at {}s.", kernel.getHostTime())
            );

            if (!kernel.isStopConditionMet()) {
                end_status = RSE_WORLD_END_TERMINATED;
            } else {
                end_status = RSE_WORLD_END_SUCCESS;
            }

            _flushSimRecorders(sim);
            break;
        }
        if (kernel.isPaused()) {
            if (sim->status != RSE_WORLD_PAUSED) {
                sim->setStatusAndStatusInfo(RSE_WORLD_PAUSED, "");
                RS_LOG(_sm_logger, info, _getLogFormatter(sim->name), "paused.");
            }
            exec.stepPaused();
            this_thread::sleep_for(chrono::milliseconds(10));
            continue;
        } else if (sim->status == RSE_WORLD_PAUSED) {
            sim->setStatusAndStatusInfo(RSE_WORLD_RUNNING, "");
            RS_LOG(_sm_logger, info, _getLogFormatter(sim->name), "resumed.");
        }

        if (!exec.step()) {
            end_status = RSE_WORLD_END_FAILED_WITH_SIMULATION_EXCEPTION;
            sim->setStatusAndStatusInfo(RSE_WORLD_TERMINATING, "");
            RS_LOG(_sm_logger, warn, _getLogFormatter(sim->name), "has invalid solution.");

            _flushSimRecorders(sim);
            break;
        }

        if (record_buffer_size != 0 &&
            kernel.getExecutionCountOfEvent("record") == record_buffer_size) {
            kernel.clearExecutionCountOfEvent("record");
            _syncSimRecorders(sim);
        }
    }
    sim->kernel_manager->terminateKernel(true);

    sim->waitForRecorderSync();
    _finishSimRecorders(sim);

    exec.free();
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "executor freed.");
    sim->kernel_manager->freeKernel();
    sim->kernel_manager->free();
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "kernel manager freed.");
    sim->setStatusAndStatusInfo(end_status, "");
}

void RS_SimulationManager::_syncSimRecorders(const std::shared_ptr<RS_Simulation> &sim) {
    sim->waitForRecorderSync();

    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "record sync begin.");
    sim->kernel_manager->syncRecorderRecords();

    sim->setRecorderSyncFuture(thread_pool.submitTask([this, sim]() {
        RS_KernelManagerCommandExecutionGuard guard(sim->kernel_manager.get());
        sim->kernel_manager->waitForRecorderRecordsSync();

        // Snapshot recorder shared_ptrs under lock so the parallel loop can run without holding
        // the mutex (recorder lifetime kept alive by the snapshot).
        std::vector<std::shared_ptr<RS_SimulationRecorder>> recorders;
        {
            std::lock_guard lock(sim->recorders_mutex);
            recorders.reserve(sim->recorders.size());
            for (auto &[_, entry] : sim->recorders) recorders.push_back(entry.second);
        }
        if (recorders.empty()) return;

        auto structure_voxel_records = sim->kernel_manager->getRecorderStructureVoxelRecords();
        thread_pool
            .submitLoop(
                0,
                recorders.size(),
                [&](RIndex i) {
                    const auto &recorder = recorders[i];
                    auto it = structure_voxel_records.find(recorder->getName());
                    if (it == structure_voxel_records.end()) return;
                    for (auto &&[structure_name, records] : it->second.structure_records)
                        recorder->syncStructureVoxelRecords(structure_name, records);
                }
            )
            .wait();

        auto camera_perspective_records =
            sim->kernel_manager->getRecorderCameraPerspectiveRecords();
        thread_pool
            .submitLoop(
                0,
                recorders.size(),
                [&](RIndex i) {
                    const auto &recorder = recorders[i];
                    auto it = camera_perspective_records.find(recorder->getName());
                    if (it == camera_perspective_records.end()) return;
                    for (auto &&[camera_name, records] : it->second.camera_records)
                        recorder->syncCameraPerspectiveRecords(camera_name, records);
                }
            )
            .wait();

        auto camera_stereo_records = sim->kernel_manager->getRecorderCameraStereoRecords();
        thread_pool
            .submitLoop(
                0,
                recorders.size(),
                [&](RIndex i) {
                    const auto &recorder = recorders[i];
                    auto it = camera_stereo_records.find(recorder->getName());
                    if (it == camera_stereo_records.end()) return;
                    for (auto &&[camera_name, records] : it->second.camera_records)
                        recorder->syncCameraStereoRecords(camera_name, records);
                }
            )
            .wait();

        auto ray_sensor_records = sim->kernel_manager->getRecorderRaySensorRecords();
        thread_pool
            .submitLoop(
                0,
                recorders.size(),
                [&](RIndex i) {
                    const auto &recorder = recorders[i];
                    auto it = ray_sensor_records.find(recorder->getName());
                    if (it == ray_sensor_records.end()) return;
                    for (auto &&[sensor_name, records] : it->second.ray_sensor_records)
                        recorder->syncRaySensorRecords(sensor_name, records);
                }
            )
            .wait();

        auto intersect_region_records = sim->kernel_manager->getRecorderIntersectRegionRecords();
        thread_pool
            .submitLoop(
                0,
                recorders.size(),
                [&](RIndex i) {
                    const auto &recorder = recorders[i];
                    auto it = intersect_region_records.find(recorder->getName());
                    if (it == intersect_region_records.end()) return;
                    for (auto &&[region_name, records] : it->second.intersect_region_records)
                        recorder->syncIntersectRegionRecords(region_name, records);
                }
            )
            .wait();
        RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "record sync finished.");
    }));
}

void RS_SimulationManager::_flushSimRecorders(const std::shared_ptr<RS_Simulation> &sim) {
    // Flush twice to ensure all buffered records are drained.
    _syncSimRecorders(sim);
    _syncSimRecorders(sim);
    sim->waitForRecorderSync();
}

void RS_SimulationManager::_finishSimRecorders(const std::shared_ptr<RS_Simulation> &sim) {
    std::vector<std::shared_ptr<RS_SimulationRecorder>> recorders;
    {
        std::lock_guard lock(sim->recorders_mutex);
        recorders.reserve(sim->recorders.size());
        for (auto &[_, entry] : sim->recorders) recorders.push_back(entry.second);
        sim->recorders.clear();
    }
    if (recorders.empty()) return;
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "finalizing recorders.");
    thread_pool.submitLoop(0, recorders.size(), [&](RIndex i) { recorders[i]->finish(); }).wait();
    RS_LOG(_sm_logger, debug, _getLogFormatter(sim->name), "recorders finalized.");
}

}  // namespace rise
