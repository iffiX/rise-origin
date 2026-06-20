// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_CALLBACK_H
#define RS_WORLD_CALLBACK_H

#include <functional>
#include <string>
#include <vector>

#include "rise_origin/core/rs_world_controller.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/utils/math/rs_float.h"

namespace rise {

/**
 * @brief Type alias for world callback function.
 *
 * This callback function is called to control a world.
 *
 * @param name Unique name of the world.
 * @param time Current world time (s).
 * @param controllers Controllers of the world.
 */
using RS_WorldCallback = std::function<void(
    const std::string &,
    RBigFloat,
    std::vector<RS_WorldController>
)>;

}  // namespace rise

#endif  // RS_WORLD_CALLBACK_H
