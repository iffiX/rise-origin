// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VISION_FLOOR_M_H
#define RS_VISION_FLOOR_M_H

#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

RS_ENUM(
    RSE_VisionFloorDataType,
    unsigned int,
    (RSE_VISION_FLOOR_DATA_NONE, 0),
    (RSE_VISION_FLOOR_DATA_FLAT, 1),
    (RSE_VISION_FLOOR_DATA_ARRAY, 2)
)

/**
 * @brief Kernel-side floor state.
 */
class RS_VisionFloorMem {
   public:
    /// Floor geometry kind.
    RSE_VisionFloorDataType data_type = RSE_VISION_FLOOR_DATA_NONE;

    /// Vision material handle for rendering.
    RHandle vision_material = RS_NULL_HANDLE;

    /// Meters per UV tile (flat floor).
    RFloat uv_scale = 1;
};

}  // namespace logic

}  // namespace rise

#endif  // RS_VISION_FLOOR_M_H
