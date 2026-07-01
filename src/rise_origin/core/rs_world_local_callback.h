// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_LOCAL_CALLBACK_H
#define RS_WORLD_LOCAL_CALLBACK_H

#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "rise_origin/core/rs_world_callback.h"

namespace rise {

/**
 * @brief In-process RS_WorldCallback driving the local world's per-step callback.
 * @details Holds the rebindable function behind a mutex so the engine thread can invoke it while a
 * user or session rebinds it. operator() snapshots the function under the lock and runs the snapshot
 * unlocked, so a rebind during a step takes effect on the next step instead of blocking the engine.
 */
class RS_WorldLocalCallback : public RS_WorldCallback {
   public:
    explicit RS_WorldLocalCallback(std::string world_name) : _world_name(std::move(world_name)) {}

    [[nodiscard]] std::string getWorldName() const override { return _world_name; }

    void setCallback(Callback callback) override {
        std::lock_guard lock(_callback_mutex);
        _callback = std::move(callback);
    }

    void resetCallback() override {
        std::lock_guard lock(_callback_mutex);
        _callback = nullptr;
    }

    [[nodiscard]] Callback getCallback() const override {
        std::lock_guard lock(_callback_mutex);
        return _callback;
    }

    void operator()(RBigFloat time, std::vector<RS_WorldController> controllers) const override {
        Callback callback;
        {
            std::lock_guard lock(_callback_mutex);
            callback = _callback;
        }
        if (callback) callback(time, std::move(controllers));
    }

   private:
    mutable std::mutex _callback_mutex;
    Callback _callback;
    std::string _world_name;
};

}  // namespace rise

#endif  // RS_WORLD_LOCAL_CALLBACK_H
