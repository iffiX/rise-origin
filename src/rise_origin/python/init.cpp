// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <any>
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
#include "rise_origin/utils/rs_string.h"
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
        {
            std::lock_guard lock(_cleanup_mutex);
            _shutdown = true;
        }
        _cleanup_cv.notify_one();

        if (_cleanup_thread && _cleanup_thread->joinable()) {
            _cleanup_thread->join();
        }
    }

    RSPY_SimulationHandle
    run_sim(py::object py_config, RIndex id, py::object py_callback = py::none()) {
        std::lock_guard lock(_simulation_mutex);

        size_t current_job_idx = _job_counter++;

        if (!py_callback.is_none()) {
            _callbacks[current_job_idx] = py_callback;
        }

        RS_SimulationCallback cxx_cb;
        if (!py_callback.is_none()) {
            cxx_cb = [this, current_job_idx](
                         RIndex sim_id,
                         std::vector<std::pair<RSE_SimulationControllerType, RS_Any>> controllers
                     ) {
                py::gil_scoped_acquire gil;

                py::object callback_obj;
                {
                    std::lock_guard lock(this->_simulation_mutex);
                    auto it = this->_callbacks.find(current_job_idx);
                    if (it != this->_callbacks.end()) {
                        callback_obj = it->second;
                    } else {
                        print("Error: the handles are cleaned up, skipping callback\n");
                        return;
                    }
                }

                py::list controllers_list;

                for (const auto &[type, controller] : controllers) {
                    if (type == RSE_SIMULATION_STRUCTURE_CONTROLLER) {
                        auto &struct_controller = controller.as<RS_SimulationStructureController>();
                        controllers_list.append(
                            py::make_tuple(
                                type, RSPY_SimulationStructureController(struct_controller)
                            )
                        );
                    } else if (type == RSE_SIMULATION_CAMERA_CONTROLLER) {
                        auto &camera_controller = controller.as<RS_SimulationCameraController>();
                        controllers_list.append(
                            py::make_tuple(type, RSPY_SimulationCameraController(camera_controller))
                        );
                    } else if (type == RSE_SIMULATION_FORCE_CONTROLLER) {
                        auto &force_controller = controller.as<RS_SimulationForceController>();
                        controllers_list.append(
                            py::make_tuple(type, RSPY_SimulationForceController(force_controller))
                        );
                    } else if (type == RSE_SIMULATION_TORQUE_CONTROLLER) {
                        auto &torque_controller = controller.as<RS_SimulationTorqueController>();
                        controllers_list.append(
                            py::make_tuple(type, RSPY_SimulationTorqueController(torque_controller))
                        );
                    }
                }

                try {
                    py::cast<py::function>(callback_obj)(sim_id, controllers_list);
                } catch (const py::error_already_set &e) {
                    print("Exception in simulation callback: {}", e.what());
                }
            };
        }

        std::shared_ptr<RS_SimulationHandle> handle;
        if (py::isinstance<RS_Config>(py_config)) {
            handle = _rise_origin.runSim(py::cast<RS_Config>(py_config), id, cxx_cb);
        } else {
            std::vector<std::string> config;
            for (auto s : py::cast<py::iterable>(py_config)) {
                config.push_back(py::cast<std::string>(s));
            }
            handle = _rise_origin.runSim(config, id, cxx_cb);
        }

        _job_handles[current_job_idx] = handle;

        _cleanup_queue.push(current_job_idx);
        _cleanup_cv.notify_one();

        return RSPY_SimulationHandle(handle);
    }

   private:
    RiseOrigin _rise_origin;
    std::mutex _simulation_mutex;
    std::unordered_map<size_t, py::object> _callbacks;
    std::map<size_t, std::shared_ptr<RS_SimulationHandle>> _job_handles;
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

            std::shared_ptr<RS_SimulationHandle> handle;
            {
                std::lock_guard sim_lock(_simulation_mutex);
                handle = _job_handles.at(job_idx);
            }

            handle->waitForEnd();

            {
                std::lock_guard sim_lock(_simulation_mutex);
                py::gil_scoped_acquire gil;
                _callbacks.erase(job_idx);
                _job_handles.erase(job_idx);
            }
        }
    }
};

PYBIND11_MODULE(_core, m) {
    initRSPYDefinitions(m);
    initRSPYRSC(m);
    initRSPYCore(m);

    py::class_<RSPY_RiseOrigin>(m, "RiseOrigin")
        .def(py::init<const RSPY_DeviceLocation &>(), py::arg("device") = RSPY_DeviceLocation())
        .def(py::init<const std::string &>(), py::arg("device_str") = "")
        .def(
            "run_sim",
            &RSPY_RiseOrigin::run_sim,
            py::arg("config"),
            py::arg("id"),
            py::arg("callback") = py::none()
        );
}
