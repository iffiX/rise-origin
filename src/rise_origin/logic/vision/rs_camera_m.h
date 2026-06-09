// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CAMERA_M_H
#define RS_CAMERA_M_H

#include <refl.hpp>

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/logic/vision/rs_vision_material_m.h"
#include "rise_origin/rsc/rs_camera_config.h"
#include "rise_origin/utils/rs_color.h"

/**
 * @brief Surface kind at the end of the PTUD specular chain. 
 * @note PTUD: Path Tracing Until Diffuse.
 * @details
 *   SURFACE  - chain landed on a diffuse / glossy surface (common ReSTIR case)
 *   SKY      - chain ray escaped into the world background (sky miss)
 *   EMISSIVE - chain ray hit an emissive voxel directly (radiance out)
 */
RS_ENUM(
    RSE_VisionRestirChainTerminalKind,
    unsigned int,
    (RSE_VISION_RESTIR_CHAIN_TERMINAL_SURFACE, 0),
    (RSE_VISION_RESTIR_CHAIN_TERMINAL_SKY, 1),
    (RSE_VISION_RESTIR_CHAIN_TERMINAL_EMISSIVE, 2)
)

namespace rise {

namespace logic {

union RS_CameraStateUnion {
    RState camera_state_packed;
    struct RS_CameraState {
        RSE_CameraType type : 3;
    } camera_state;
};

template <RSE_FloatType FloatMemAccuracy>
class RS_CameraMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;
    using Vec3M = RS_Vec3DMem<FloatMemAccuracy>;
    using Quat3M = RS_Quat3DMem<FloatMemAccuracy>;
    using PositionVec = RS_Vec3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;

    /**
     * Pre-set attributes
     */
    RIndex gid = RS_NULL_INDEX;

    RIndex rigid_body_sid = RS_NULL_INDEX;

    RSize camera_control_signal_offset = RS_NULL_INDEX;

    FloatM aspect_ratio = 0;

    RSize image_width = 0;

    RSize image_height = 0;

    RSize samples_per_pixel = 0;

    RSize max_depth = 0;

    FloatM focus_distance = 0;

    FloatM field_of_view = 0;

    FloatM defocus_angle = 0;

    FloatM defocus_radius = 0;

    FloatM pixel_width = 0;

    FloatM pixel_height = 0;

    PositionVec rigid_body_anchor_rb_position = PositionVec::zero();

    Quat3M rigid_body_anchor_rb_orientation = Quat3M::unit();

    /**
     * Post-set attributes
     */
    RHandle rigid_body = RS_NULL_HANDLE;

    RState state = 0;

    PositionVec position = PositionVec::zero();

    Quat3M orientation = Quat3M::unit();

    PositionVec prev_position = PositionVec::zero();

    Quat3M prev_orientation = Quat3M::unit();

    RBigFloat last_update_time = 0;

    RBigSize frame_count = 0;
};

/// Packed identity of the hit object at a pixel: structure_gid + material_gid.
/// structure_and_kind: top 2 bits = kind tag, bottom 30 bits = structure_gid.
/// material_gid: material gid for voxel hits; 0 for floor; ~0u for miss.
struct RS_CameraPixelObjectKey {
    RIndex structure_and_kind = 0;
    RIndex material_gid = 0;

    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_CameraPixelObjectKey& other) const {
        return structure_and_kind == other.structure_and_kind &&
               material_gid == other.material_gid;
    }

    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_CameraPixelObjectKey& other) const {
        return !(*this == other);
    }

    /// True when this key marks a miss / sky pixel.
    HOST_DEVICE_INLINE_QUALIFIER bool isMissed() const {
        return structure_and_kind == ~RIndex(0) && material_gid == ~RIndex(0);
    }
};

/// Kind tag stored in the top 2 bits of structure_and_kind.
constexpr RIndex RS_CAMERA_PIXEL_OBJECT_KIND_SHIFT = 30;
constexpr RIndex RS_CAMERA_PIXEL_OBJECT_KIND_VOXEL = 0;
constexpr RIndex RS_CAMERA_PIXEL_OBJECT_KIND_FLOOR = 1;
constexpr RIndex RS_CAMERA_PIXEL_OBJECT_KIND_MISS = 3;
constexpr RIndex RS_CAMERA_PIXEL_OBJECT_GID_MASK =
    (RIndex(1) << RS_CAMERA_PIXEL_OBJECT_KIND_SHIFT) - 1;

/// Sentinel for miss/sky pixels (all bits set in both fields).
constexpr RS_CameraPixelObjectKey RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL = {
    ~RIndex(0), ~RIndex(0)
};

/// Per-pixel direct-illumination reservoir for ReSTIR DI.
struct RS_ReservoirDI {
    /// Identity of the pixel's hit surface; mismatch across frames resets the reservoir.
    RS_CameraPixelObjectKey object_key = RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL;

    /// 0 = sun; k >= 1 = structure_emissive_voxel_handles[k - 1].
    RIndex light_index = 0;

    /// Sampled point on the light at survivor time; unused for the sun.
    RVec3f light_position = RVec3f::zero();

    /// World-space shading point at survivor time.
    RVec3f prev_hit_position = RVec3f::zero();

    /// World-space surface normal at survivor time.
    RVec3f prev_normal = RVec3f::zero();

    /// Per-channel shadow-walk attenuation to the survivor light.
    RColor3f transmission = RColor3f::zero();

    /// Survivor's p_hat under current shading.
    float target_pdf = 0;

    /// Accumulated Bitterli WRS resampling weight.
    float weight_sum = 0;

    /// Temporal sample count; was M in Bitterli 2020.
    RSize sample_count = 0;
};

/// Per-pixel one-bounce indirect-illumination reservoir for ReSTIR GI.
struct RS_ReservoirGI {
    /// Identity of the endpoint shading surface; mismatch resets the reservoir.
    RS_CameraPixelObjectKey object_key = RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL;

    /// World-space endpoint shading point at survivor time.
    RVec3f prev_hit_position = RVec3f::zero();

    /// World-space endpoint surface normal at survivor time.
    RVec3f prev_normal = RVec3f::zero();

    /// BRDF-sampled bounce direction at the endpoint at survivor time.
    RVec3f bounce_direction = RVec3f::zero();

    /// World-space closest hit of the bounce ray; zero on sky miss.
    RVec3f bounce_hit_position = RVec3f::zero();

    /// Surface normal at bounce_hit_position; zero on sky miss.
    RVec3f bounce_hit_normal = RVec3f::zero();

    /// Incoming radiance at the bounce hit gathered toward the endpoint.
    RColor3f l_incoming = RColor3f::zero();

    /// Survivor's p_hat under current shading.
    float target_pdf = 0;

    /// Accumulated Bitterli WRS resampling weight.
    float weight_sum = 0;

    /// Temporal sample count; was M in Bitterli 2020.
    RSize sample_count = 0;

    /// Bias-correction Z (Ouyang 2021 Alg 4). 0 means "use sample_count".
    RSize Z_count = 0;
};

/// Index 0 of the light list is reserved for the sun; emissive voxels start at 1.
constexpr RIndex RS_VISION_SUN_LIGHT_INDEX = 0;

/// Per-camera RGB canvas; ping-pong pair lives on RS_Context.
using RS_CameraRGBCanvas = device::RS_Array<RVec3f, 2>;

/// General info-pass AOV buffers for denoising and ReSTIR.
using RS_CameraAuxNormal = device::RS_Array<RVec3f, 2>;
using RS_CameraAuxDepthZ = device::RS_Array<float, 2>;
using RS_CameraAuxHitPosition = device::RS_Array<RVec3f, 2>;
using RS_CameraAuxObjectKey = device::RS_Array<RS_CameraPixelObjectKey, 2>;
using RS_CameraAuxAlbedo = device::RS_Array<RVec3f, 2>;
using RS_CameraAuxRoughness = device::RS_Array<float, 2>;
using RS_CameraAuxMetalness = device::RS_Array<float, 2>;
using RS_CameraAuxIntensity = device::RS_Array<float, 2>;
using RS_CameraAuxMaterialType = device::RS_Array<RSE_VisionMaterialType, 2>;
using RS_CameraAuxScreenMotion = device::RS_Array<RVec2f, 2>;
using RS_CameraAuxWorldVelocity = device::RS_Array<RVec3f, 2>;

/// PTUD chain bookkeeping AOVs: throughput accumulates Beer-Lambert + Fresnel
/// along the chain; terminal_kind tags how the chain ended (see
/// RSE_VisionRestirChainTerminalKind).
using RS_CameraAuxChainThroughput   = device::RS_Array<RVec3f, 2>;
using RS_CameraAuxChainTerminalKind = device::RS_Array<RSE_VisionRestirChainTerminalKind, 2>;

using RS_CameraReservoirDI = device::RS_Array<RS_ReservoirDI, 2>;
using RS_CameraReservoirGI = device::RS_Array<RS_ReservoirGI, 2>;

/// SVGF prev-frame aux snapshots (disocclusion gate next frame).
using RS_CameraPrevAuxNormal    = device::RS_Array<RVec3f, 2>;
using RS_CameraPrevAuxDepthZ    = device::RS_Array<float, 2>;
using RS_CameraPrevAuxObjectKey = device::RS_Array<RS_CameraPixelObjectKey, 2>;

/// SVGF temporal history: color, luminance moments (m1, m2), length, variance.
using RS_CameraDenoiserColorHistory  = device::RS_Array<RVec3f, 2>;
using RS_CameraDenoiserMomentHistory = device::RS_Array<RVec2f, 2>;
using RS_CameraDenoiserHistoryLength = device::RS_Array<RFloat, 2>;
using RS_CameraDenoiserVariance      = device::RS_Array<RFloat, 2>;

}  // namespace logic

}  // namespace rise

#define REFLECT_CAMERA_MEM(camera_mem_type)   \
    REFL_AUTO(                                \
        type(camera_mem_type),                \
        field(gid),                           \
        field(rigid_body_sid),                \
        field(camera_control_signal_offset),  \
        field(aspect_ratio),                  \
        field(image_width),                   \
        field(image_height),                  \
        field(samples_per_pixel),             \
        field(max_depth),                     \
        field(focus_distance),                \
        field(field_of_view),                 \
        field(defocus_angle),                 \
        field(defocus_radius),                \
        field(pixel_width),                   \
        field(pixel_height),                  \
        field(rigid_body_anchor_rb_position), \
        field(rigid_body),                    \
        field(state),                         \
        field(position),                      \
        field(orientation),                   \
        field(prev_position),                 \
        field(prev_orientation),              \
        field(last_update_time),              \
        field(frame_count)                    \
    )

REFLECT_CAMERA_MEM(rise::logic::RS_CameraMem<rise::RSE_FLOAT_32>)
REFLECT_CAMERA_MEM(rise::logic::RS_CameraMem<rise::RSE_FLOAT_64>)

#endif