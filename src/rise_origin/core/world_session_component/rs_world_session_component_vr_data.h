// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_SESSION_COMPONENT_VR_DATA_H
#define RS_WORLD_SESSION_COMPONENT_VR_DATA_H

#include <string>

#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

/**
 * @brief Broker channel carrying the VR user's world position each control step as an
 * RS_WorldSessionVRUserPosition.
 */
inline const std::string RS_VR_CHANNEL_USER_POSITION{"/vr/user_position"};

/**
 * @brief Broker channel carrying both hands' aim rays each control step as an
 * RS_WorldSessionVRUserHandAims.
 */
inline const std::string RS_VR_CHANNEL_USER_HAND_AIMS{"/vr/user_hand_aims"};

/**
 * @brief The VR user's world-space position, taken as the midpoint between the eyes.
 */
struct RS_WorldSessionVRUserPosition {
    RVec3rf position;
};

/**
 * @brief One hand's aim ray in world space.
 * @details The origin is where the ray starts, the direction is the unit ray direction, and the
 * position is where the ray lands. is_valid is false when the hand is untracked, in which case the
 * vectors are stale and should be ignored.
 */
struct RS_WorldSessionVRHandAim {
    bool is_valid = false;
    RVec3rf aim_origin;
    RVec3rf aim_direction;
    RVec3rf aim_position;
};

/**
 * @brief Both hands' aim rays.
 */
struct RS_WorldSessionVRUserHandAims {
    RS_WorldSessionVRHandAim left;
    RS_WorldSessionVRHandAim right;
};

}  // namespace rise

#endif  // RS_WORLD_SESSION_COMPONENT_VR_DATA_H
