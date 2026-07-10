// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "core/init_world_callback.h"

void initRSPYWorldCallback(py::module &m) {
    /*****************************************************************
     * World callback
     *****************************************************************/

    py::class_<RSPY_WorldCallback>(
        m, "RS_WorldCallback", "The rebindable per-control-step callback for a world."
    )
        .def(
            "get_world_name",
            &RSPY_WorldCallback::get_world_name,
            R"doc(
            Returns the name of the world this callback is bound to.
        )doc"
        )
        .def(
            "set_callback",
            &RSPY_WorldCallback::set_callback,
            py::arg("callback"),
            R"doc(
            Sets the function the engine invokes each control step.

            Args:
                callback: A callable invoked as callback(time, controllers).
        )doc"
        )
        .def(
            "reset_callback",
            &RSPY_WorldCallback::reset_callback,
            R"doc(
            Clears the callback function so the engine invokes nothing.
        )doc"
        )
        .def(
            "get_callback",
            &RSPY_WorldCallback::get_callback,
            R"doc(
            Returns the currently set callback function, or None if none is set.
        )doc"
        );
}
