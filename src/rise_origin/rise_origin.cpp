// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rise_origin.h"

#include <future>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "core/rs_simulation_manager.h"
#include "core/rs_world_local_handle.h"
#include "device/platform/rs_platform.h"
#include "rise_origin/rsc/rsc_world.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_logging.h"

using namespace std;
using namespace rise;
using namespace rise::rsc;
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

    [[nodiscard]] std::shared_ptr<RS_WorldHandle>
    hello(RSC_World &&world, RS_WorldCallback &&callback) {
        sim_manager_mutex.lock();

        auto log_level = parseLogLevel(world.simulation.runtime.log_level);
        if (!logging::get("root")) {
            auto logger = std::make_shared<logging::logger>("root");
            register_logger(logger);
            logger->sinks().emplace_back(make_shared<logging::sinks::stdout_color_sink_mt>());
        }
        logging::get("root")->set_level(log_level);
        sim_manager = new RS_SimulationManager(local_device);

        sim_manager->init();

        auto local_handle = sim_manager->run(std::move(world), std::move(callback));

        auto handle = std::make_shared<RS_WorldLocalHandle>(local_handle);

        sim_manager_free_thread.reset(new std::thread([&]() {
            sim_manager->free();
            delete sim_manager;
            logging::get("root")->debug("Simulation manager instance freed");
            sim_manager = nullptr;
            sim_manager_mutex.unlock();
        }));

        return handle;
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

std::shared_ptr<RS_WorldHandle>
RiseOrigin::hello(const RSC_World &world, RS_WorldCallback callback) {
    RSC_World local = world;
    local.validate(false);
    return _rise_impl->hello(std::move(local), std::move(callback));
}
