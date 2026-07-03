// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RAY_SENSOR_SIGNAL_M_H
#define RS_RAY_SENSOR_SIGNAL_M_H

#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * @brief A ray sensor signal slot holding one world pose signal.
 * @details A free ray sensor publishes one entry into the shared ray sensor signal array; the
 * update kernel reads the position and orientation from it each step.
 */
struct RS_RaySensorSignalMem {
    RS_WorldRaySensorSignal signal;

    HOST_DEVICE_INLINE_QUALIFIER RS_RaySensorSignalMem() : signal() {}

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER RS_WorldRootSignal& root() {
        return static_cast<RS_WorldRootSignal&>(signal);
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_RAY_SENSOR_SIGNAL_M_H
