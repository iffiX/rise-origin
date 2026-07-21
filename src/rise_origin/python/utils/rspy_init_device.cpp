// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "utils/rspy_init_device.h"

void initRSPYDevice(py::module &m) {
    /*****************************************************************
     * Enums and device location
     *****************************************************************/

    py::enum_<RSE_DeviceType>(m, "RSE_DeviceType", "Kind of a compute device.")
        .value("RSE_DEVICE_TYPE_UNKNOWN", RSE_DEVICE_TYPE_UNKNOWN)
        .value("RSE_DEVICE_TYPE_CPU", RSE_DEVICE_TYPE_CPU)
        .value("RSE_DEVICE_TYPE_CUDA", RSE_DEVICE_TYPE_CUDA)
        .export_values();

    py::class_<RSPY_DeviceLocation>(
        m, "RS_DeviceLocation", "A compute device, given by its type and index."
    )
        .def(py::init<>(), "Default to CPU device 0.")
        .def(
            py::init<const std::string &>(),
            py::arg("device_str"),
            "Construct from \"cpu\" or \"cuda:<id>\"."
        )
        .def(
            py::init<RSE_DeviceType, RIndex>(),
            py::arg("type"),
            py::arg("id"),
            "Construct from a device type and index."
        )
        .def(
            py::init<const RS_DeviceLocation &>(),
            py::arg("location"),
            "Construct from a core device location."
        )
        .def("type", &RSPY_DeviceLocation::type, "The device type (CPU or CUDA).")
        .def("id", &RSPY_DeviceLocation::id, "The device index.")
        .def("__str__", &RSPY_DeviceLocation::str);
}
