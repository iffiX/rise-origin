// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_CORE_INIT_WORLD_CONTROLLER_H
#define RSPY_CORE_INIT_WORLD_CONTROLLER_H

#include "core/init_world_record.h"
#include "core/init_world_signal.h"
#include "init_common.h"

/*****************************************************************
 * Controller wrapper.
 *****************************************************************/

class RSPY_WorldController {
   public:
    RS_WorldController controller;
    explicit RSPY_WorldController(const RS_WorldController &ctrl) : controller(ctrl) {}

    [[nodiscard]] std::string name() const { return controller.name; }

    [[nodiscard]] py::list records() const {
        py::list result;
        for (const auto &record : controller.records) {
            record.visit([&](const auto &held) {
                using T = std::decay_t<decltype(held)>;
                if constexpr (std::is_same_v<T, RS_WorldStructureVoxelRecord>) {
                    result.append(py::cast(RSPY_WorldStructureVoxelRecord(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldCameraPerspectiveRecord>) {
                    result.append(py::cast(RSPY_WorldCameraPerspectiveRecord(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldCameraStereoRecord>) {
                    result.append(py::cast(RSPY_WorldCameraStereoRecord(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldRaySensorRecord>) {
                    result.append(py::cast(RSPY_WorldRaySensorRecord(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldIntersectRegionRecord>) {
                    result.append(py::cast(RSPY_WorldIntersectRegionRecord(held)));
                }
            });
        }
        return result;
    }

    [[nodiscard]] py::list signals() const {
        py::list result;
        for (const auto &signal : controller.signals) {
            signal.visit([&](const auto &held) {
                using T = std::decay_t<decltype(held)>;
                if constexpr (std::is_same_v<T, RS_WorldStructureVoxelSignal>) {
                    result.append(py::cast(RSPY_WorldStructureVoxelSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldCameraPerspectiveSignal>) {
                    result.append(py::cast(RSPY_WorldCameraPerspectiveSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldCameraStereoSignal>) {
                    result.append(py::cast(RSPY_WorldCameraStereoSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldRaySensorSignal>) {
                    result.append(py::cast(RSPY_WorldRaySensorSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldForceSignal>) {
                    result.append(py::cast(RSPY_WorldForceSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldTorqueSignal>) {
                    result.append(py::cast(RSPY_WorldTorqueSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldTransformSignal>) {
                    result.append(py::cast(RSPY_WorldTransformSignal(held)));
                }
            });
        }
        return result;
    }
};

/// Bind the runtime world controller view.
void initRSPYWorldController(py::module &m);

#endif  // RSPY_CORE_INIT_WORLD_CONTROLLER_H
