// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VISION_COSMIC_BACKGROUND_M_H
#define RS_VISION_COSMIC_BACKGROUND_M_H

#include "rise_origin/logic/vision/rs_vision_texture_m.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

namespace logic {

/**
 * @brief Cosmic background render kind. 
 */
RS_ENUM(
    RSE_VisionCosmicBackgroundType,
    unsigned int,
    (RSE_VISION_COSMIC_BACKGROUND_UNIFORM, 0),
    (RSE_VISION_COSMIC_BACKGROUND_GRADIENT, 1),
    (RSE_VISION_COSMIC_BACKGROUND_HOSEK_WILKIE, 2)
)

union RS_VisionCosmicBackgroundStateUnion {
    RState cosmic_background_state_packed;
    struct RS_VisionCosmicBackgroundState {
        RSE_VisionCosmicBackgroundType type : 4;
        RSE_VisionSampleInputType color_input_type : 3;
    } cosmic_background_state;
};

/**
 * @brief Kernel-side cosmic background state.
 */
class RS_VisionCosmicBackgroundMem {
   public:
    /// Packed background type + color sample-input discriminator.
    RState state = 0;

    /// Uniform background color input.
    RS_VisionSampleInputMem<RColor3f> color;

    /// Gradient zenith color.
    RColor3f zenith_color = RColor3f::zero();

    /// Gradient horizon color.
    RColor3f horizon_color = RColor3f::zero();

    /// Sun direction in world space.
    RVec3f sun_direction = RVec3f(0, 0, 1);

    /// Sun disk color.
    RColor3f sun_color = RColor3f::zero();

    /// Sun disk angular radius in radians.
    RFloat sun_angular_radius = 0;

    /// Scalar multiplier applied to sampled radiance.
    RFloat intensity = 1;

    /// Rotation around the world up axis (radians), applied to the equirect UV.
    RFloat rotation_rad = 0;

    /// Hosek-Wilkie cooked state: 3 channels x 9 coefficients per channel.
    RFloat hw_config[27] = {0};

    /// Hosek-Wilkie per-channel radiance multipliers.
    RFloat hw_radiance[3] = {0, 0, 0};
};

}  // namespace logic

}  // namespace rise

#endif  // RS_VISION_COSMIC_BACKGROUND_M_H
