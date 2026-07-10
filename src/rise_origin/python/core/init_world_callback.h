// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_CORE_INIT_WORLD_CALLBACK_H
#define RSPY_CORE_INIT_WORLD_CALLBACK_H

#include "core/init_world_controller.h"
#include "init_common.h"

/*****************************************************************
 * World callback wrapper.
 *****************************************************************/

class RSPY_WorldCallback {
   public:
    std::shared_ptr<RS_WorldCallback> callback;
    explicit RSPY_WorldCallback(std::shared_ptr<RS_WorldCallback> callback)
        : callback(std::move(callback)) {}

    [[nodiscard]] std::string get_world_name() const { return callback->getWorldName(); }

    void set_callback(py::function py_callback) {
        // The engine drops the last reference to the wrapping function on its own thread, so the
        // captured Python object must be released under the GIL; a custom deleter does that. The
        // same holder backs get_callback, so it stays the function Python set.
        _py_callback = std::shared_ptr<py::object>(
            new py::object(std::move(py_callback)), [](py::object *object) {
                py::gil_scoped_acquire gil;
                delete object;
            }
        );
        std::shared_ptr<py::object> py_object = _py_callback;
        callback->setCallback(
            [py_object](RBigFloat time, std::vector<RS_WorldController> controllers) {
                py::gil_scoped_acquire gil;
                py::list controllers_list;
                for (auto &controller : controllers) {
                    controllers_list.append(RSPY_WorldController(controller));
                }
                try {
                    py::cast<py::function> (*py_object)(time, controllers_list);
                } catch (const py::error_already_set &error) {
                    py::print("Exception in world callback:", error.what());
                }
            }
        );
    }

    void reset_callback() {
        _py_callback.reset();
        callback->resetCallback();
    }

    [[nodiscard]] py::object get_callback() const {
        return _py_callback ? *_py_callback : py::none();
    }

   private:
    std::shared_ptr<py::object> _py_callback;
};

/// Bind the rebindable world callback.
void initRSPYWorldCallback(py::module &m);

#endif  // RSPY_CORE_INIT_WORLD_CALLBACK_H
