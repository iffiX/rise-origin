// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rise_origin.h"

#include <future>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "core/rs_simulation_local_handle.h"
#include "core/rs_simulation_manager.h"
#include "device/platform/rs_platform.h"
#include "rise_origin/rsc/rs_config.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_logging.h"

using namespace std;
using namespace rise;
using namespace rise::device;

class RiseOriginImpl {
   public:
    RS_DeviceLocation local_device{};
    std::mutex sim_manager_mutex{};
    RS_SimulationManager *sim_manager = nullptr;
    std::unique_ptr<std::thread, void (*)(std::thread *)> sim_manager_free_thread{
        nullptr,
        [](std::thread *t) {
            if (t == nullptr) return;  // already joined()
            t->join();
            delete t;
        }
    };

    RiseOriginImpl() = default;

    explicit RiseOriginImpl(const RS_DeviceLocation &device) : local_device(device) {}

    [[nodiscard]] std::shared_ptr<RS_SimulationHandle> runSim(
        RS_Config &&config, RIndex id, RS_SimulationCallback &&callback
    ) {
        const auto &sim_config = config.simulation_config;

        sim_manager_mutex.lock();

        if (!logging::get("root")) {
            auto logger = std::make_shared<logging::logger>("root");
            register_logger(logger);
            logger->sinks().emplace_back(make_shared<logging::sinks::stdout_color_sink_mt>());
        }
        logging::get("root")->set_level(parseLogLevel(sim_config.runtime_log_level));
        sim_manager = new RS_SimulationManager(local_device);

        sim_manager->init();

        auto local_handle = sim_manager->runSim(std::move(config), id, std::move(callback));

        auto sim_handle = std::make_shared<RS_SimulationLocalHandle>(local_handle);

        sim_manager_free_thread.reset(new std::thread([&]() {
            sim_manager->free();
            delete sim_manager;
            logging::get("root")->debug("Simulation manager instance freed");
            sim_manager = nullptr;
            sim_manager_mutex.unlock();
        }));

        return sim_handle;
    }

    static RS_Config parseConfig(const vector<string> &config) {
        return std::move(RS_Config(config, false));
    }

    static logging::level::level_enum parseLogLevel(const string &log_level_str) {
        if (log_level_str == "err") return logging::level::err;
        if (log_level_str == "warn") return logging::level::warn;
        if (log_level_str == "info") return logging::level::info;
        if (log_level_str == "debug") return logging::level::debug;
        if (log_level_str == "trace") return logging::level::trace;
        RS_THROW(RS_InvalidArgumentException(
            "Unknown log level type: {}, "
            "Supported are: err, warn, info, debug, trace",
            log_level_str
        ));
    }
};

RiseOrigin::RiseOrigin(const RS_DeviceLocation &device)
    : _rise_impl(make_shared<RiseOriginImpl>(device)) {}

std::shared_ptr<RS_SimulationHandle> RiseOrigin::runSim(
    std::vector<std::string> config, RIndex id, RS_SimulationCallback callback
) {
    auto parsed_config = RiseOriginImpl::parseConfig(config);
    return _rise_impl->runSim(std::move(parsed_config), id, std::move(callback));
}

std::shared_ptr<RS_SimulationHandle> RiseOrigin::runSim(
    RS_Config config, RIndex id, RS_SimulationCallback callback
) {
    config.validate(false);
    return _rise_impl->runSim(std::move(config), id, std::move(callback));
}
