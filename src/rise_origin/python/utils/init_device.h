// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_UTILS_INIT_DEVICE_H
#define RSPY_UTILS_INIT_DEVICE_H

#include "init_common.h"

class RSPY_DeviceLocation {
   public:
    RS_DeviceLocation loc;

    RSPY_DeviceLocation() {
        loc.type = RSE_DEVICE_TYPE_CPU;
        loc.id = 0;
    }

    /// Construct from "cpu" or "cuda:<id>".
    explicit RSPY_DeviceLocation(const std::string &device_str) {
        if (device_str == "cpu") {
            loc.type = RSE_DEVICE_TYPE_CPU;
            loc.id = 0;
        } else if (startsWith(device_str, "cuda:")) {
            loc.type = RSE_DEVICE_TYPE_CUDA;
            if (!isNumber(device_str.substr(5))) {
                throw RS_InvalidArgumentException("Invalid device str: {}", device_str);
            }
            loc.id = std::stoi(device_str.substr(5));
        } else {
            throw RS_InvalidArgumentException("Invalid device str: {}", device_str);
        }
    }

    explicit RSPY_DeviceLocation(const RS_DeviceLocation &d) : loc(d) {}

    RSPY_DeviceLocation(RSE_DeviceType t, RIndex i) {
        loc.type = t;
        loc.id = i;
    }

    [[nodiscard]] RSE_DeviceType type() const { return loc.type; }
    [[nodiscard]] RIndex id() const { return loc.id; }
    [[nodiscard]] std::string str() const {
        if (loc.type == RSE_DEVICE_TYPE_CPU) return "cpu";
        if (loc.type == RSE_DEVICE_TYPE_CUDA) return "cuda:" + std::to_string(loc.id);
        return "unknown";
    }
};

/// Bind the device type enum and the RS_DeviceLocation wrapper.
void initRSPYDevice(py::module &m);

#endif  // RSPY_UTILS_INIT_DEVICE_H
