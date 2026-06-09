// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_TORQUE_M_H
#define RS_TORQUE_M_H

#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * @brief Rigid body torque memory class for storing torque signal information.
 *
 * This class represents a torque signal applied to a rigid body. Torque is only
 * defined for rigid bodies (it is undefined for individual soft voxels), so the
 * entire torque is applied to one rigid body.
 */
class RS_RigidTorqueMem {
   public:
    /// Handle pointing to the offset in the torque signal array (3 RFloat values: x, y, z)
    RHandle torque_signal = RS_NULL_HANDLE;

    /// Structure global ID that this torque belongs to
    RIndex structure_gid = RS_NULL_INDEX;

    /// Rigid body structure scope ID (filter)
    RIndex rigid_body_sid = RS_NULL_INDEX;

    /// Last device time when torque signal was applied
    RBigFloat last_torque_signal_apply_time = 0.0;

    RS_RigidTorqueMem() = default;

    RS_RigidTorqueMem(RHandle torque_signal, RIndex structure_gid, RIndex rigid_body_sid)
        : torque_signal(torque_signal),
          structure_gid(structure_gid),
          rigid_body_sid(rigid_body_sid),
          last_torque_signal_apply_time(0.0) {}
};

}  // namespace logic

}  // namespace rise

#endif  // RS_TORQUE_M_H
