// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VISION_MATERIAL_M_H
#define RS_VISION_MATERIAL_M_H

#include <refl.hpp>

#include "rise_origin/utils/rs_color.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_float.h"

namespace rise {

RS_ENUM(
    RSE_VisionMaterialType,
    unsigned int,
    (RSE_VISION_MATERIAL_OPAQUE, 0),
    (RSE_VISION_MATERIAL_TRANSPARENT, 1),
    (RSE_VISION_MATERIAL_EMISSIVE, 2)
)

namespace logic {

union RS_VisionMaterialStateUnion {
    RState vision_material_state_packed;
    struct RS_VisionMaterialState {
        RSE_VisionMaterialType render_type : 4;
    } vision_material_state;
};

template <RSE_FloatType FloatMemAccuracy>
class RS_VisionMaterialMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;

    /// Global ID of the source material.
    RIndex material_gid = RS_NULL_INDEX;

    /// Packed render type.
    RState state = 0;

    /// Per-type color:
    /// base color (OPAQUE),
    /// per-channel attenuation (TRANSPARENT),
    /// emission color (EMISSIVE).
    RColor3f color = RColor3f::zero();

    /// Index of refraction; only meaningful for TRANSPARENT.
    FloatM ior = FloatM(1.0);

    /// Emission scalar multiplier; only meaningful for EMISSIVE.
    FloatM intensity = 0;

    /// Surface roughness in [0, 1].
    FloatM roughness = FloatM(1.0);

    /// Metalness in [0, 1]; > 0 picks the metallic lobe on OPAQUE.
    FloatM metalness = FloatM(0);
};

}  // namespace logic

}  // namespace rise

#define REFLECT_VISION_MATERIAL_MEM(vision_material_mem_type) \
    REFL_AUTO(                                                \
        type(vision_material_mem_type),                       \
        field(material_gid),                                  \
        field(state),                                         \
        field(color),                                         \
        field(ior),                                           \
        field(intensity),                                     \
        field(roughness),                                     \
        field(metalness)                                      \
    );

REFLECT_VISION_MATERIAL_MEM(rise::logic::RS_VisionMaterialMem<rise::RSE_FLOAT_32>);
REFLECT_VISION_MATERIAL_MEM(rise::logic::RS_VisionMaterialMem<rise::RSE_FLOAT_64>);

#endif  // RS_VISION_MATERIAL_M_H
