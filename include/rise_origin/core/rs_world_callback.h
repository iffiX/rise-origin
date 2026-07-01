// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_CALLBACK_H
#define RS_WORLD_CALLBACK_H

#include <functional>
#include <string>
#include <vector>

#include "rise_origin/core/rs_world_controller.h"
#include "rise_origin/utils/math/rs_float.h"

namespace rise {

/**
 * @brief Rebindable holder for the function the engine invokes once per control step.
 * @details The engine owns the holder and calls it every step; the user (or a session) can
 * rebind the held function on a live world through setCallback/resetCallback.
 */
class RS_WorldCallback {
   public:
    /**
     * @brief Callback function type.
     * @param time Current world time (s).
     * @param controllers Controllers of the world.
     */
    using Callback =
        std::function<void(RBigFloat time, std::vector<RS_WorldController> controllers)>;

    virtual ~RS_WorldCallback() = default;

    /**
     * @brief Get the name of the world this callback is bound to.
     */
    [[nodiscard]] virtual std::string getWorldName() const = 0;

    /**
     * @brief Set the held callback function.
     */
    virtual void setCallback(Callback callback) = 0;

    /**
     * @brief Reset the held callback function to none.
     */
    virtual void resetCallback() = 0;

    /**
     * @brief Get the currently held callback function (empty if none set).
     */
    [[nodiscard]] virtual Callback getCallback() const = 0;

    /**
     * @brief Invoke the currently held callback function, doing nothing if none is set.
     * @details Used by the engine to drive the callback each control step.
     */
    virtual void operator()(RBigFloat time, std::vector<RS_WorldController> controllers) const = 0;
};

}  // namespace rise

#endif  // RS_WORLD_CALLBACK_H
