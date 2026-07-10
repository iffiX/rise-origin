// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "core/init_world_controller.h"

void initRSPYWorldController(py::module &m) {
    /*****************************************************************
     * Controller
     *****************************************************************/

    py::class_<RSPY_WorldController>(m, "RS_WorldController")
        .def("name", &RSPY_WorldController::name)
        .def("records", &RSPY_WorldController::records)
        .def("signals", &RSPY_WorldController::signals);
}
