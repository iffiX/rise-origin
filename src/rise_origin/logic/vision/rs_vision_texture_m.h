// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VISION_TEXTURE_M_H
#define RS_VISION_TEXTURE_M_H

#include <refl.hpp>

#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

RS_ENUM(
    RSE_VisionSampleInputType,
    unsigned int,
    (RSE_VISION_SAMPLE_INPUT_CONSTANT, 0),
    (RSE_VISION_SAMPLE_INPUT_GRID, 5)
)

/**
 * @brief Tagged-union payload: either a constant value or a texture handle.
 * The discriminator is NOT stored here — it's packed as a bit in the owning
 * mem's state field alongside the other per-channel flags.
 */
template <typename T>
union RS_VisionSampleInputMem {
    T constant;
    RHandle texture;

    HOST_DEVICE_INLINE_QUALIFIER RS_VisionSampleInputMem() : constant() {}
    HOST_DEVICE_INLINE_QUALIFIER RS_VisionSampleInputMem(const T& c) : constant(c) {}
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_VisionSampleInputMem(RHandle h) : texture(h) {}
};

/**
 * @brief Procedural grid texture: a fill color with regularly spaced lines.
 *
 * A GRID vision sample input holds a handle into the context grid table; sampling
 * compares the sample coordinate (world position when is_meters, else interpolated UV)
 * against the line spacing to pick line_color vs fill_color.
 */
class RS_TextureGridMem {
   public:
    /// Background fill color.
    RColor3f fill_color = RColor3f::zero();

    /// Grid line color.
    RColor3f line_color = RColor3f::zero();

    /// When true the grid is laid out in world meters; otherwise in UV space.
    bool is_meters = true;

    /// Line spacing along X (meters when is_meters, else UV units).
    RFloat x_spacing = 1;

    /// Line spacing along Y (meters when is_meters, else UV units).
    RFloat y_spacing = 1;

    /// Full line width (same units as the spacing).
    RFloat line_width = 0.05;
};

}  // namespace logic

}  // namespace rise

REFL_AUTO(
    type(rise::logic::RS_TextureGridMem),
    field(fill_color),
    field(line_color),
    field(is_meters),
    field(x_spacing),
    field(y_spacing),
    field(line_width)
)

#endif  // RS_VISION_TEXTURE_M_H
