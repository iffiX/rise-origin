// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_DEVICE_H
#define RS_DEVICE_H

#include "rs_def.h"

namespace rise {

RS_ENUM(
    RSE_DeviceType,
    RIndex,
    (RSE_DEVICE_TYPE_CPU, 0),
    (RSE_DEVICE_TYPE_CUDA, 1),
    (RSE_DEVICE_TYPE_UNKNOWN, 100)
);

struct RS_DeviceLocation {
    RSE_DeviceType type = RSE_DEVICE_TYPE_UNKNOWN;
    RIndex id = RS_NULL_INDEX;

    RS_DeviceLocation() = default;

    HOST_DEVICE_INLINE_QUALIFIER RS_DeviceLocation(RSE_DeviceType type, RIndex id)
        : type(type), id(id) {}

    HOST_DEVICE_INLINE_QUALIFIER constexpr bool operator==(const RS_DeviceLocation& other) const {
        return this->type == other.type && this->id == other.id;
    }

    HOST_DEVICE_INLINE_QUALIFIER constexpr bool operator!=(const RS_DeviceLocation& other) const {
        return this->type != other.type || this->id != other.id;
    }

    HOST_DEVICE_INLINE_QUALIFIER constexpr bool isSameDeviceType(const RSE_DeviceType& type) const {
        return this->type == type;
    }
};

}  // namespace rise

#endif  // RS_DEVICE_H
