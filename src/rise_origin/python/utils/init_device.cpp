// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "utils/init_device.h"

void initRSPYDevice(py::module &m) {
    /*****************************************************************
     * Enums and device location
     *****************************************************************/

    py::enum_<RSE_DeviceType>(m, "RSE_DeviceType")
        .value("RSE_DEVICE_TYPE_UNKNOWN", RSE_DEVICE_TYPE_UNKNOWN)
        .value("RSE_DEVICE_TYPE_CPU", RSE_DEVICE_TYPE_CPU)
        .value("RSE_DEVICE_TYPE_CUDA", RSE_DEVICE_TYPE_CUDA)
        .export_values();

    py::class_<RSPY_DeviceLocation>(m, "RS_DeviceLocation")
        .def(py::init<>())
        .def(py::init<const std::string &>())
        .def(py::init<RSE_DeviceType, RIndex>())
        .def(py::init<const RS_DeviceLocation &>())
        .def("type", &RSPY_DeviceLocation::type)
        .def("id", &RSPY_DeviceLocation::id)
        .def("__str__", &RSPY_DeviceLocation::str);
}
