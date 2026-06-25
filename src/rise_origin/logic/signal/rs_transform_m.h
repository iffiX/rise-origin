// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_TRANSFORM_M_H
#define RS_TRANSFORM_M_H

#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * @brief Transform memory class for storing transform signal information.
 *
 * This class represents a transform signal applied once to a whole structure.
 * It applies to every voxel and rigid body of the structure regardless of
 * material, then the engine resets the signal to identity.
 */
class RS_TransformMem {
   public:
    /// Handle pointing to the offset in the transform signal array
    RHandle transform_signal = RS_NULL_HANDLE;

    /// Structure global ID that this transform belongs to
    RIndex structure_gid = RS_NULL_INDEX;

    RS_TransformMem() = default;

    RS_TransformMem(RHandle transform_signal, RIndex structure_gid)
        : transform_signal(transform_signal), structure_gid(structure_gid) {}
};

}  // namespace logic

}  // namespace rise

#endif  // RS_TRANSFORM_M_H
