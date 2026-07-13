// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VISION_MATERIAL_M_H
#define RS_VISION_MATERIAL_M_H

#include <refl.hpp>

#include "rise_origin/logic/vision/rs_vision_texture_m.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/rs_def.h"

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
        RSE_VisionSampleInputType albedo_input_type : 3;
        RSE_VisionSampleInputType roughness_metalness_input_type : 3;
        RSE_VisionSampleInputType emission_input_type : 3;
    } vision_material_state;
};

/// Packed material identity of an optical: material gid in the high bits, optical mid in the low
/// bits, so one RIndex read yields both. Widths track RS_WORLD_RSC_MATERIAL_MAX_NUM (22 bits) and
/// RS_WORLD_RSC_MATERIAL_MAX_OPTICAL_PROPERTY_NUM (10 bits).
union RS_VisionMaterialKeyUnion {
    RIndex vision_material_key_packed;
    struct RS_VisionMaterialKey {
        RIndex optical_mid : 10;
        RIndex material_gid : 22;
    } vision_material_key;
};

template <RSE_FloatType FloatMemAccuracy>
class RS_VisionMaterialMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;

    /// Packed source material gid and optical mid, see RS_VisionMaterialKeyUnion.
    RIndex vision_material_key = 0;

    /// Packed render type + per-channel is_texture discriminators.
    RState state = 0;

    /// Albedo / base color
    RS_VisionSampleInputMem<RColor3f> albedo;

    /// Roughness / metalness
    RS_VisionSampleInputMem<RColor3f> roughness_metalness;

    /// Emission color
    RS_VisionSampleInputMem<RColor3f> emission_color;

    /// Index of refraction (transparent materials).
    FloatM ior = FloatM(1.5);

    /// Emission intensity scalar multiplier.
    FloatM emission_intensity = 0;
};

}  // namespace logic

}  // namespace rise

#define REFLECT_VISION_MATERIAL_MEM(vision_material_mem_type) \
    REFL_AUTO(                                                \
        type(vision_material_mem_type),                       \
        field(vision_material_key),                           \
        field(state),                                         \
        field(albedo),                                        \
        field(roughness_metalness),                           \
        field(emission_color),                                \
        field(ior),                                           \
        field(emission_intensity)                             \
    );

REFLECT_VISION_MATERIAL_MEM(rise::logic::RS_VisionMaterialMem<rise::RSE_FLOAT_32>);
REFLECT_VISION_MATERIAL_MEM(rise::logic::RS_VisionMaterialMem<rise::RSE_FLOAT_64>);

#endif  // RS_VISION_MATERIAL_M_H
