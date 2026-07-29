// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_SESSION_COMPONENT_SHARED_DATA_H
#define RS_WORLD_SESSION_COMPONENT_SHARED_DATA_H

#include <string>
#include <vector>

#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

/**
 * @brief Broker channel carrying where the user is and what the user aims at each control step as
 * an RS_WorldSessionUserView.
 */
inline const std::string RS_WORLD_SESSION_CHANNEL_USER_VIEW{"/user/view"};

/**
 * @brief Aim name for where the user looks.
 */
inline const std::string RS_WORLD_SESSION_USER_AIM_VISION{"vision"};

/**
 * @brief Aim name for where the user's left hand points.
 */
inline const std::string RS_WORLD_SESSION_USER_AIM_LEFT_HAND{"left_hand"};

/**
 * @brief Aim name for where the user's right hand points.
 */
inline const std::string RS_WORLD_SESSION_USER_AIM_RIGHT_HAND{"right_hand"};

/**
 * @brief One ray the user aims, in world space.
 * @details The origin is where the ray starts, the direction is the unit ray direction, and the
 * position is where the ray ends: a surface when a ray sensor backs the aim, otherwise the far
 * depth. is_valid is false when the aim is untracked, in which case the vectors are stale and
 * should be ignored.
 */
struct RS_WorldSessionUserAim {
    std::string name;
    bool is_valid = false;
    RVec3rf origin;
    RVec3rf direction;
    RVec3rf position;
};

/**
 * @brief One component's view of the user: where the user is and every ray the user aims.
 * @details The name is the publishing component's name, so several components may each carry their
 * own view of the user on the channel at once.
 */
struct RS_WorldSessionUserView {
    std::string name;
    RVec3rf position;
    std::vector<RS_WorldSessionUserAim> aims;
};

}  // namespace rise

#endif  // RS_WORLD_SESSION_COMPONENT_SHARED_DATA_H
