// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_DEF_H
#define RS_SIMULATION_DEF_H

#include "rise_origin/utils/rs_def.h"

namespace rise {

RS_ENUM(
    RSE_SimulationStatus,
    unsigned int,
    (RSE_SIMULATION_END_SUCCESS, 0),
    (RSE_SIMULATION_END_TERMINATED, 1),
    (RSE_SIMULATION_END_FAILED_WITH_INVALID_CONFIG, 100),
    (RSE_SIMULATION_END_FAILED_WITH_SIMULATION_EXCEPTION, 101),
    (RSE_SIMULATION_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION, 102),
    (RSE_SIMULATION_UNINITIALIZED, 10),
    (RSE_SIMULATION_INITIALIZING, 11),
    (RSE_SIMULATION_RUNNING, 12),
    (RSE_SIMULATION_PAUSED, 13),
    (RSE_SIMULATION_TERMINATING, 14),
    (RSE_SIMULATION_MAX_NORMAL_STATE, 99)
)

inline bool isSimulationEnd(const RSE_SimulationStatus &status) {
    return status == RSE_SIMULATION_END_SUCCESS || status == RSE_SIMULATION_END_TERMINATED ||
           status > RSE_SIMULATION_MAX_NORMAL_STATE;
}

RS_ENUM(
    RSE_LinkAxis,
    unsigned int,
    (RSE_X_AXIS, 0),  //!< X Axis
    (RSE_Y_AXIS, 1),  //!< Y Axis
    (RSE_Z_AXIS, 2),  //!< Z Axis
    (RSE_NO_AXIS, 3)  //!< For tendons
)

//! Defines the direction of a link relative to a given voxel.
RS_ENUM(
    RSE_LinkDirection,
    unsigned int,
    (RSE_X_POS, 0),        //!< Positive X direction
    (RSE_X_NEG, 1),        //!< Negative X direction
    (RSE_Y_POS, 2),        //!< Positive Y direction
    (RSE_Y_NEG, 3),        //!< Negative Y direction
    (RSE_Z_POS, 4),        //!< Positive Z direction
    (RSE_Z_NEG, 5),        //!< Negative Z direction
    (RSE_NO_DIRECTION, 6)  //!< For tendons
)

/// Returns the voxel X index offset from a link
inline HOST_DEVICE_QUALIFIER int linkDirectionToXOffset(RSE_LinkDirection direction) {
    return (direction == RSE_X_NEG) ? -1 : ((direction == RSE_X_POS) ? 1 : 0);
}

/// Returns the voxel Y index offset from a link
inline HOST_DEVICE_QUALIFIER int linkDirectionToYOffset(RSE_LinkDirection direction) {
    return (direction == RSE_Y_NEG) ? -1 : ((direction == RSE_Y_POS) ? 1 : 0);
}

/// Returns the voxel Z index offset from a link
inline HOST_DEVICE_QUALIFIER int linkDirectionToZOffset(RSE_LinkDirection direction) {
    return (direction == RSE_Z_NEG) ? -1 : ((direction == RSE_Z_POS) ? 1 : 0);
}

/// Returns the link axis of the specified link direction.
inline HOST_DEVICE_QUALIFIER RSE_LinkAxis linkDirectionToAxis(RSE_LinkDirection direction) {
    return static_cast<RSE_LinkAxis>(static_cast<int>(direction) >> 1);
}

/// Returns the link direction of the specified link axis and sign.
inline HOST_DEVICE_QUALIFIER RSE_LinkDirection
linkAxisToDirection(RSE_LinkAxis axis, bool positiveDirection) {
    return static_cast<RSE_LinkDirection>(2 * static_cast<int>(axis) + (positiveDirection ? 0 : 1));
}

/// Returns true if the specified link direction is negative.
inline HOST_DEVICE_QUALIFIER bool isLinkDirectionNegative(RSE_LinkDirection direction) {
    return direction & 1;
}

/// Returns true if the specified link direction is positive.
inline HOST_DEVICE_QUALIFIER bool isLinkDirectionisPositive(RSE_LinkDirection direction) {
    return (direction & 1) == 0;
}

/// Returns the opposite (negated) link direction of the specified
/// direction.
inline HOST_DEVICE_QUALIFIER RSE_LinkDirection oppositeLinkDirection(RSE_LinkDirection direction) {
    return static_cast<RSE_LinkDirection>(direction - (direction & 1) + ((direction + 1) & 1));
}

}  // namespace rise

#endif  // RS_SIMULATION_DEF_H
