// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/rspy_init_signal.h"

void initRSPYSignal(py::module &m) {
    /***************************************************************
     * Signal
     ***************************************************************/
    rspyRSC<RSC_Force, RSC_SignalBase>(m, "RSC_Force")
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_Force::structure_name)
        .def_readwrite("voxel_range", &RSC_Force::voxel_range)
        .def_readwrite("body_sid", &RSC_Force::body_sid)
        .def_readwrite("segment_bid", &RSC_Force::segment_bid);

    rspyRSC<RSC_Torque, RSC_SignalBase>(m, "RSC_Torque")
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_Torque::structure_name)
        .def_readwrite("body_sid", &RSC_Torque::body_sid)
        .def_readwrite("segment_bid", &RSC_Torque::segment_bid);

    rspyRSC<RSC_Transform, RSC_SignalBase>(m, "RSC_Transform")
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_Transform::structure_name);
}
