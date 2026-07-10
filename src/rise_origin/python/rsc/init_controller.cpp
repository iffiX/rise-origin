// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/init_controller.h"

void initRSPYController(py::module &m) {
    /***************************************************************
     * Controller
     ***************************************************************/
    rspyRSC<RSC_ControllerCallback, RSC_ControllerBase>(m, "RSC_ControllerCallback")
        .def(py::init<>());
}
