// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <filesystem>

#include "rise_origin/core/recorder/rs_simulation_hdf5_recorder.h"
#include "rise_origin/utils/rs_exception.h"
#include "rs_simulation.h"

namespace fs = std::filesystem;

namespace rise {

void RS_Simulation::addRecorder(const rsc::RSC_RecorderVariant &recorder) {
    recorder.visit([&](auto &cfg) {
        using T = std::decay_t<decltype(cfg)>;
        cfg.validate();

        std::shared_ptr<RS_SimulationRecorder> instance;
        if constexpr (std::is_same_v<T, rsc::RSC_RecorderHDF5>) {
            auto parent_dir = fs::path(cfg.path).parent_path();
            if (!parent_dir.empty() && !fs::exists(parent_dir)) {
                fs::create_directories(parent_dir);
            }
            instance = std::make_shared<RS_SimulationHDF5Recorder>(cfg.name, cfg.path);
        } else {
            RS_THROW(RS_ImplementationException("Unsupported recorder variant for '{}'", cfg.name));
        }

        instance->start();

        std::lock_guard lock(recorders_mutex);
        if (recorders.count(cfg.name)) {
            RS_THROW(RS_DuplicateException("Recorder name '{}' already in use", cfg.name));
        }
        recorders.emplace(cfg.name, std::make_pair(recorder, std::move(instance)));
    });
}

void RS_Simulation::removeRecorder(const std::string &name) {
    // Drain any in-flight sync before tearing down a recorder.
    waitForRecorderSync();

    std::shared_ptr<RS_SimulationRecorder> to_finish;
    {
        std::lock_guard lock(recorders_mutex);
        auto it = recorders.find(name);
        if (it == recorders.end()) RS_THROW(RS_NotFoundException("Recorder '{}' not found", name));
        to_finish = std::move(it->second.second);
        recorders.erase(it);
    }
    to_finish->finish();
}

rsc::RSC_RecorderVariant RS_Simulation::findRecorder(const std::string &name) {
    std::lock_guard lock(recorders_mutex);
    auto it = recorders.find(name);
    if (it == recorders.end()) RS_THROW(RS_NotFoundException("Recorder '{}' not found", name));
    return it->second.first;
}

void RS_Simulation::addController(const rsc::RSC_ControllerVariant &controller) {
    controller.visit([&](auto &cfg) {
        cfg.validate();
        std::lock_guard lock(controllers_mutex);
        if (controllers.count(cfg.name)) {
            RS_THROW(RS_DuplicateException("Controller name '{}' already in use", cfg.name));
        }
        controllers.emplace(cfg.name, controller);
    });
}

void RS_Simulation::removeController(const std::string &name) {
    std::lock_guard lock(controllers_mutex);
    if (controllers.erase(name) == 0) {
        RS_THROW(RS_NotFoundException("Controller '{}' not found", name));
    }
}

rsc::RSC_ControllerVariant RS_Simulation::findController(const std::string &name) {
    std::lock_guard lock(controllers_mutex);
    auto it = controllers.find(name);
    if (it == controllers.end()) RS_THROW(RS_NotFoundException("Controller '{}' not found", name));
    return it->second;
}

std::unordered_set<std::string> RS_Simulation::getControllerNames() {
    std::lock_guard lock(controllers_mutex);
    std::unordered_set<std::string> names;
    names.reserve(controllers.size());
    for (auto &[name, _] : controllers) names.insert(name);
    return names;
}

}  // namespace rise
