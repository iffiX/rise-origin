// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "init_core.h"
#include "init_def.h"
#include "init_rsc.h"
#include "rise_origin/rise_origin.h"
#include "rise_origin/utils/string/rs_string.h"
namespace py = pybind11;

class RSPY_RiseOrigin {
   public:
    explicit RSPY_RiseOrigin(const RSPY_DeviceLocation &device = {RSE_DEVICE_TYPE_CPU, 0})
        : _rise_origin(device.loc) {
        _cleanup_thread = std::make_unique<std::thread>(&RSPY_RiseOrigin::_cleanupThreadFunc, this);
    }

    explicit RSPY_RiseOrigin(const std::string &device_str = "") {
        if (device_str.empty()) {
            print("Warning: Default to execute with local CPU\n");
            _rise_origin = RiseOrigin(RS_DeviceLocation{RSE_DEVICE_TYPE_CPU, 0});
        } else {
            _rise_origin = RiseOrigin(RSPY_DeviceLocation(device_str).loc);
        }

        _cleanup_thread = std::make_unique<std::thread>(&RSPY_RiseOrigin::_cleanupThreadFunc, this);
    }

    ~RSPY_RiseOrigin() {
        // release the gil to make sure the clean-up thread can work properly
        py::gil_scoped_release release;

        // Tearing down joins the clean-up thread, which waits for every world to end. A world still
        // running here usually means the instance was used as a temporary, so warn before blocking.
        {
            std::lock_guard sim_lock(_simulation_mutex);
            for (auto &[_, handle] : _job_handles) {
                if (handle->isEnd()) continue;
                print(
                    "Warning: RiseOrigin destroyed while world '{}' is still running. Keep the "
                    "RiseOrigin instance alive until its world ends; avoid using it as a temporary "
                    "like rs.RiseOrigin(\"cpu\").hello(world).\n",
                    handle->name()
                );
                std::fflush(stdout);
            }
        }

        {
            std::lock_guard lock(_cleanup_mutex);
            _shutdown = true;
        }
        _cleanup_cv.notify_one();

        if (_cleanup_thread && _cleanup_thread->joinable()) {
            _cleanup_thread->join();
        }
    }

    std::pair<RSPY_WorldHandle, RSPY_WorldCallback> hello(const rsc::RSC_World &world) {
        std::lock_guard lock(_simulation_mutex);

        size_t current_job_idx = _job_counter++;

        auto [handle, callback] = _rise_origin.hello(world);

        _job_handles[current_job_idx] = handle;

        _cleanup_queue.push(current_job_idx);
        _cleanup_cv.notify_one();

        return {RSPY_WorldHandle(handle), RSPY_WorldCallback(callback)};
    }

   private:
    RiseOrigin _rise_origin;
    std::mutex _simulation_mutex;
    std::map<size_t, std::shared_ptr<RS_WorldHandle>> _job_handles;
    std::atomic<size_t> _job_counter{0};
    std::unique_ptr<std::thread> _cleanup_thread;
    std::mutex _cleanup_mutex;
    std::condition_variable _cleanup_cv;
    std::queue<size_t> _cleanup_queue;
    bool _shutdown = false;

    void _cleanupThreadFunc() {
        while (true) {
            size_t job_idx;
            {
                std::unique_lock lock(_cleanup_mutex);

                if (_shutdown && _cleanup_queue.empty()) {
                    std::lock_guard sim_lock(_simulation_mutex);
                    if (_job_handles.empty()) {
                        break;
                    }
                }

                if (_cleanup_queue.empty()) {
                    _cleanup_cv.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                        return _shutdown || !_cleanup_queue.empty();
                    });
                    continue;
                }

                job_idx = _cleanup_queue.front();
                _cleanup_queue.pop();
            }

            std::shared_ptr<RS_WorldHandle> handle;
            {
                std::lock_guard sim_lock(_simulation_mutex);
                handle = _job_handles.at(job_idx);
            }

            handle->waitForEnd();

            {
                std::lock_guard sim_lock(_simulation_mutex);
                py::gil_scoped_acquire gil;
                _job_handles.erase(job_idx);
            }
        }
    }
};

PYBIND11_MODULE(_core, m) {
    initRSPYDefinitions(m);
    initRSPYRSC(m);
    initRSPYCore(m);

    py::class_<RSPY_RiseOrigin>(
        m,
        "RiseOrigin",
        R"doc(
        The RiseOrigin.

        Usage::

            world = RSC_World()
            r = RiseOrigin()
            handle, callback = r.hello(world)
            # World is yours
    )doc"
    )
        .def(
            py::init<const RSPY_DeviceLocation &>(),
            py::arg("device") = RSPY_DeviceLocation(),
            R"doc(
            Construct a local RiseOrigin instance on the specified device.

            Args:
                device: Device identifiers used by this process.
        )doc"
        )
        .def(
            py::init<const std::string &>(),
            py::arg("device_str") = "",
            R"doc(
            Construct a local RiseOrigin instance from a device string.

            Args:
                device_str: Device string, such as "cpu" or "cuda:0". Defaults to the
                            local CPU when empty.
        )doc"
        )
        .def(
            "hello",
            &RSPY_RiseOrigin::hello,
            py::arg("world"),
            R"doc(
            Say hello to the world.

            Args:
                world: World characterization describing all entities, essences, resources,
                       etc that should exist before time starts.

            Returns:
                A tuple of the world handle and the rebindable world callback.
        )doc"
        );
}
