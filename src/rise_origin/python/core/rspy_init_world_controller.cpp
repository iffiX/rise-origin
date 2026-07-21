// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "core/rspy_init_world_controller.h"

void initRSPYWorldController(py::module &m) {
    /*****************************************************************
     * Controller
     *****************************************************************/

    py::class_<RSPY_WorldController>(
        m,
        "RS_WorldController",
        "A controller: a general, named I/O container reading records and driving signals."
    )
        .def("name", &RSPY_WorldController::name, "Controller name.")
        .def("records", &RSPY_WorldController::records, "Runtime records the controller observes.")
        .def("signals", &RSPY_WorldController::signals, "Runtime signals the controller drives.");
}
