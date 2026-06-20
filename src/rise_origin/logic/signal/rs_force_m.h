// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_FORCE_M_H
#define RS_FORCE_M_H

#include "rise_origin/utils/math/rs_bound.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * @brief Rigid body force memory class for storing force signal information.
 *
 * This class represents a force signal applied to a rigid body.
 * The entire force is applied to one rigid body (unit_portion = 1.0).
 */
class RS_RigidForceMem {
   public:
    /// Handle pointing to the offset in the force signal array (3 RFloat values: x, y, z)
    RHandle force_signal = RS_NULL_HANDLE;

    /// How much portion of the force this rigid body should take (always 1.0 for rigid bodies)
    RFloat unit_portion = 1.0;

    /// Structure global ID that this force belongs to
    RIndex structure_gid = RS_NULL_INDEX;

    /// Rigid body structure scope ID (filter)
    RIndex rigid_body_sid = RS_NULL_INDEX;

    /// Last device time when force signal was applied
    RBigFloat last_force_signal_apply_time = 0.0;

    RS_RigidForceMem() = default;

    RS_RigidForceMem(
        RHandle force_signal,
        RFloat unit_portion,
        RIndex structure_gid,
        RIndex rigid_body_sid
    )
        : force_signal(force_signal),
          unit_portion(unit_portion),
          structure_gid(structure_gid),
          rigid_body_sid(rigid_body_sid),
          last_force_signal_apply_time(0.0) {}
};

/**
 * @brief Soft voxel force memory class for storing force signal information.
 *
 * This class represents a force signal applied to soft voxels within a specified range.
 * The force is distributed across multiple voxels based on the unit_portion value.
 */
class RS_SoftForceMem {
   public:
    /// Handle pointing to the offset in the force signal array (3 RFloat values: x, y, z)
    RHandle force_signal = RS_NULL_HANDLE;

    /// How much portion of the force each voxel should take (1.0 / num_voxels_in_range)
    RFloat unit_portion = 0.0;

    /// Structure global ID that this force belongs to
    RIndex structure_gid = RS_NULL_INDEX;

    /// Voxel range (bound) to apply the force to (filter)
    RBound3ri voxel_range;

    /// Body structure scope ID (filter)
    RIndex body_sid = RS_NULL_INDEX;

    /// Segment body scope ID (filter)
    RIndex segment_bid = RS_NULL_INDEX;

    /// Last device time when force signal was applied
    RBigFloat last_force_signal_apply_time = 0.0;

    RS_SoftForceMem() = default;

    RS_SoftForceMem(
        RHandle force_signal,
        RFloat unit_portion,
        RIndex structure_gid,
        const RBound3ri &voxel_range,
        RIndex body_sid,
        RIndex segment_bid
    )
        : force_signal(force_signal),
          unit_portion(unit_portion),
          structure_gid(structure_gid),
          voxel_range(voxel_range),
          body_sid(body_sid),
          segment_bid(segment_bid),
          last_force_signal_apply_time(0.0) {}
};

}  // namespace logic

}  // namespace rise

#endif  // RS_FORCE_M_H
