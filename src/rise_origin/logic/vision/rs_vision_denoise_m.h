// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VISION_DENOISE_M_H
#define RS_VISION_DENOISE_M_H

#include "rise_origin/utils/rs_def.h"

namespace rise {

RS_ENUM(
    RSE_VisionDenoiseType,
    unsigned int,
    (RSE_VISION_DENOISE_NONE, 0),
    (RSE_VISION_DENOISE_ATROUS, 1),
    (RSE_VISION_DENOISE_SVGF, 2)
)

}  // namespace rise

#endif  // RS_VISION_DENOISE_M_H
