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
    std::shared_ptr<RS_WorldHandle> world_handle{};

    RiseOriginImpl() = default;

    explicit RiseOriginImpl(const RS_DeviceLocation &device) : local_device(device) {}

    ~RiseOriginImpl() {
        // Destroying the engine joins the free thread, which waits for the world to end. A world
        // still running here usually means the engine was destroyed too early, so warn first.
        if (world_handle && !world_handle->isEnd()) {
            if (auto logger = logging::get("root")) {
                logger->warn(
                    "RiseOrigin destroyed while world '{}' is still running. Keep the RiseOrigin "
                    "instance alive until its world ends; avoid using it as a temporary like "
                    "RiseOrigin(\"cpu\").hello(world).",
                    world_handle->name()
                );
                logger->flush();
            }
        }
    }

    [[nodiscard]] std::pair<std::shared_ptr<RS_WorldHandle>, std::shared_ptr<RS_WorldCallback>>
    hello(RSC_World &&world) {
        sim_manager_mutex.lock();

        auto log_level = parseLogLevel(world.simulation->runtime.log_level);
        if (!logging::get("root")) {
            auto logger = std::make_shared<logging::logger>("root");
            register_logger(logger);
            logger->sinks().emplace_back(make_shared<logging::sinks::stdout_color_sink_mt>());
        }
        logging::get("root")->set_level(log_level);
        _logGreeting(logging::get("root"), world, local_device);
        sim_manager = new RS_SimulationManager(local_device);

        sim_manager->init();

        auto [local_handle, callback] = sim_manager->run(std::move(world));

        auto handle = std::make_shared<RS_WorldLocalHandle>(local_handle);
        world_handle = handle;

        sim_manager_free_thread.reset(new std::thread([&]() {
            sim_manager->free();
            delete sim_manager;
            logging::get("root")->debug("Simulation manager instance freed");
            sim_manager = nullptr;
            sim_manager_mutex.unlock();
        }));

        return {handle, callback};
    }

    // Draw a login-style greeting once when a world is welcomed locally. It goes through the
    // logger so it inherits the same sink and color handling, with a raw message pattern and a
    // forced visible level so the cube and the hello signature always show.
    static void _logGreeting(
        const std::shared_ptr<logging::logger> &logger,
        const RSC_World &world,
        const RS_DeviceLocation &device
    ) {
        if (!logger) return;

        const std::string blue = "\033[38;5;75m";
        const std::string bold = "\033[1;97m";
        const std::string dim = "\033[38;5;244m";
        const std::string sign = "\033[1;96m";
        const std::string reset = "\033[0m";

        const std::string device_name =
            std::string(device.type == RSE_DEVICE_TYPE_CUDA ? "cuda" : "cpu") + ":" +
            std::to_string(static_cast<long long>(device.id));
        const std::string world_name = world.name.empty() ? std::string("world") : world.name;

        const std::vector<std::string> banner = {
            "",
            blue + "      ____" + reset,
            blue + "     /   /|" + reset + "    " + dim + "welcome to " + reset + bold +
                "rise-origin" + reset + dim + "  v" RS_ENGINE_VERSION + reset,
            blue + "    /___/ |" + reset + "    " + dim + "device" + reset + " : " + device_name,
            blue + "    |   | /" + reset + "    " + dim + "world " + reset + " : " + world_name,
            blue + "    |___|/ " + reset + "    " + sign + ">>> hello(\"" + world_name + "\")" +
                reset,
            "",
        };

        const auto previous_level = logger->level();
        logger->set_pattern("%v");
        logger->set_level(logging::level::info);
        for (const auto &line : banner) logger->info(line);
        logger->set_level(previous_level);
        logger->set_pattern("%+");
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

std::pair<std::shared_ptr<RS_WorldHandle>, std::shared_ptr<RS_WorldCallback>> RiseOrigin::hello(
    const RSC_World &world
) {
    RSC_World local = world;
    return _rise_impl->hello(std::move(local));
}

std::pair<std::shared_ptr<RS_WorldHandle>, std::shared_ptr<RS_WorldCallback>>
RiseOrigin::connect(const std::string & /*url*/) {
    RS_THROW(
        RS_NotImplementedException("RiseOrigin::connect (remote transport) is not yet implemented")
    );
    return {};
}
