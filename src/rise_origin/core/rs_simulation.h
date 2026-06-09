// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_H
#define RS_SIMULATION_H

#include <condition_variable>
#include <memory>
#include <mutex>

#include "rise_origin/core/recorder/rs_simulation_recorder.h"
#include "rise_origin/core/rs_simulation_def.h"
#include "rise_origin/utils/rs_float.h"

namespace rise {

class RS_KernelManager;

class RS_Simulation {
   public:
    std::mutex status_mutex;
    std::condition_variable status_cv;
    RSE_SimulationStatus status{RSE_SIMULATION_UNINITIALIZED};
    std::string status_info;
    RIndex id = RS_NULL_INDEX;
    RSE_FloatType simulation_memory_accuracy = RSE_FLOAT_32;

    std::shared_ptr<RS_KernelManager> kernel_manager;
    std::future<void> recorder_sync_future{};
    std::vector<std::shared_ptr<RS_SimulationRecorder>> recorders;

    RS_Simulation() = default;

    [[nodiscard]] std::pair<RSE_SimulationStatus, std::string> getStatusAndStatusInfo() {
        std::lock_guard lock(status_mutex);
        return std::make_pair(status, status_info);
    }

    void setStatusAndStatusInfo(
        const RSE_SimulationStatus new_status,
        const std::string &new_status_info
    ) {
        std::lock_guard lock(status_mutex);
        status = new_status;
        status_info = new_status_info;
        status_cv.notify_all();
    }

    bool waitForStatus(const RSE_SimulationStatus &expected_status) {
        std::unique_lock lock(status_mutex);
        status_cv.wait(lock, [this, expected_status] {
            return status == expected_status || isSimulationEnd(status);
        });
        return expected_status == status;
    }

    void waitForEnd() {
        std::unique_lock lock(status_mutex);
        status_cv.wait(lock, [this] { return isSimulationEnd(status); });
    }

    void setRecorderSyncFuture(std::future<void> &&new_future) {
        recorder_sync_future = std::move(new_future);
    }

    void waitForRecorderSync() const {
        if (recorder_sync_future.valid()) recorder_sync_future.wait();
    }
};

}  // namespace rise

#endif  // RS_SIMULATION_H
