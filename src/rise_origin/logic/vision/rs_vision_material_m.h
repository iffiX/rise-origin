// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VISION_MATERIAL_M_H
#define RS_VISION_MATERIAL_M_H

#include <refl.hpp>

#include "rise_origin/logic/vision/rs_vision_texture_m.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/math/rs_float.h"

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

template <RSE_FloatType FloatMemAccuracy>
class RS_VisionMaterialMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;

    /// Global ID of the source material
    RIndex material_gid = RS_NULL_INDEX;

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
        field(material_gid),                                  \
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
