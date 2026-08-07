// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CAMERA_M_H
#define RS_CAMERA_M_H

#include <refl.hpp>

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/logic/vision/rs_vision_material_m.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/rs_conf.h"

namespace rise {

namespace logic {

/**
 * @brief Surface type at the end of the PTUD specular chain.
 * @note PTUD: Path Tracing Until Diffuse.
 * @details
 *   SURFACE           - chain landed on a diffuse / glossy surface (common ReSTIR case)
 *   COSMIC_BACKGROUND - chain ray escaped into the world background
 *   EMISSIVE          - chain ray hit an emissive voxel directly (radiance out)
 */
RS_ENUM(
    RSE_VisionRestirChainTerminalType,
    unsigned int,
    (RSE_VISION_RESTIR_CHAIN_TERMINAL_SURFACE, 0),
    (RSE_VISION_RESTIR_CHAIN_TERMINAL_COSMIC_BACKGROUND, 1),
    (RSE_VISION_RESTIR_CHAIN_TERMINAL_EMISSIVE, 2)
)

RS_ENUM(
    RSE_CameraType,
    unsigned int,
    (RSE_CAMERA_FREE_PERSPECTIVE, 0),
    (RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE, 1),
    (RSE_CAMERA_FREE_STEREO, 2),
    (RSE_CAMERA_RIGID_BODY_LINKED_STEREO, 3)
)

union RS_CameraStateUnion {
    RState camera_state_packed;
    struct RS_CameraState {
        RSE_CameraType type : 3;
        RState has_rgb : 1;
        RState has_depth : 1;
        RState has_normal : 1;
        RState has_motion : 1;
        RState has_entity_info : 1;
        RState is_depth_hyperbolic : 1;
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
    /// Global id of the camera entity.
    RIndex gid = RS_NULL_INDEX;

    /// Structure id of the rigid body the camera rides. (rigid-body-linked only)
    RIndex rigid_body_sid = RS_NULL_INDEX;

    /// Which sub-camera this mem renders; 0 for mono, 0 is the left eye and 1 the right. (stereo)
    RIndex sub_camera_index = 0;

    /// Handle of the camera's pose signal entry.
    RHandle camera_signal_handle = RS_NULL_HANDLE;

    /// Image height divided by image width.
    FloatM aspect_ratio = 0;

    /// Image width in pixels. (per eye for stereo)
    RSize image_width = 0;

    /// Image height in pixels.
    RSize image_height = 0;

    /// Path traced samples per pixel.
    RSize samples_per_pixel = 0;

    /// Maximum ray bounce depth.
    RSize max_depth = 0;

    /// Focus plane distance in meters.
    FloatM focus_distance = 0;

    /// Vertical field of view in degrees. (perspective only)
    FloatM field_of_view = 0;

    /// Defocus cone angle in degrees for depth of field.
    FloatM defocus_angle = 0;

    /// Lens radius derived from the defocus angle.
    FloatM defocus_radius = 0;

    /// Near plane distance for hyperbolic depth encoding.
    FloatM hyperbolic_near_depth = 0;

    /// Far plane distance for hyperbolic depth encoding.
    FloatM hyperbolic_far_depth = 0;

    /// World space width of one pixel on the viewport.
    FloatM pixel_width = 0;

    /// World space height of one pixel on the viewport.
    FloatM pixel_height = 0;

    /// Mount position relative to the rigid body. (rigid-body-linked only)
    PositionVec rigid_body_anchor_rb_position = PositionVec::zero();

    /// Mount orientation relative to the rigid body. (rigid-body-linked only)
    Quat3M rigid_body_anchor_rb_orientation = Quat3M::unit();

    /// Distance between the two eyes in meters. (stereo only)
    FloatM interpupillary_distance = 0;

    /// Toe-in convergence distance in meters; 0 keeps the eyes parallel. (stereo only)
    FloatM convergence_distance = 0;

    /// Left edge angle of the eye frustum in radians. (stereo only)
    FloatM angle_left = 0;

    /// Right edge angle of the eye frustum in radians. (stereo only)
    FloatM angle_right = 0;

    /// Top edge angle of the eye frustum in radians. (stereo only)
    FloatM angle_up = 0;

    /// Bottom edge angle of the eye frustum in radians. (stereo only)
    FloatM angle_down = 0;

    /**
     * Post-set attributes
     */
    /// Handle of the resolved rigid body. (rigid-body-linked only)
    RHandle rigid_body = RS_NULL_HANDLE;

    /// Packed camera state flags, see RS_CameraStateUnion.
    RState state = 0;

    /// Current world position.
    PositionVec position = PositionVec::zero();

    /// Current world orientation.
    Quat3M orientation = Quat3M::unit();

    /// Previous frame world position, for screen motion.
    PositionVec prev_position = PositionVec::zero();

    /// Previous frame world orientation, for screen motion.
    Quat3M prev_orientation = Quat3M::unit();

    /// World time of the last pose update.
    RBigFloat last_update_time = 0;

    /// World time passed between the last two pose updates.
    RBigFloat last_update_passed_time = 0;

    /// Number of frames rendered so far.
    RBigSize frame_count = 0;
};

/// Identity of the hit object at a pixel: entity_gid + packed vision material key.
/// entity_gid: cosmic background = 0, floor = 1, structure voxel = its entity gid.
/// vision_material_key: packed material gid and optical mid for voxel hits; 0 for floor;
/// sentinel cosmic-background miss. See RS_VisionMaterialKeyUnion.
struct RS_CameraPixelObjectKey {
    RIndex entity_gid = RS_WORLD_RSC_COSMIC_BACKGROUND_GID;
    RIndex vision_material_key = 0;

    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_CameraPixelObjectKey& other) const {
        return entity_gid == other.entity_gid && vision_material_key == other.vision_material_key;
    }

    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_CameraPixelObjectKey& other) const {
        return !(*this == other);
    }

    /// True when this key marks a cosmic-background pixel.
    HOST_DEVICE_INLINE_QUALIFIER bool isCosmicBackground() const {
        return entity_gid == RS_WORLD_RSC_COSMIC_BACKGROUND_GID;
    }
};

/// Sentinel for cosmic-background pixels: the cosmic background entity with no material.
constexpr RS_CameraPixelObjectKey RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL = {
    RS_WORLD_RSC_COSMIC_BACKGROUND_GID,
    ~RIndex(0)
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

    /// World-space closest hit of the bounce ray; zero on cosmic-background miss.
    RVec3f bounce_hit_position = RVec3f::zero();

    /// Surface normal at bounce_hit_position; zero on cosmic-background miss.
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
/// Note: do not use oct normal for normal because decoding is heavy.
using RS_CameraAuxNormal = device::RS_Array<RVec3f, 2>;
using RS_CameraAuxDepthZ = device::RS_Array<float, 2>;
using RS_CameraAuxHitPosition = device::RS_Array<RVec3f, 2>;
using RS_CameraAuxObjectKey = device::RS_Array<RS_CameraPixelObjectKey, 2>;
using RS_CameraAuxAlbedo = device::RS_Array<RS_Vec3DMem<RSE_FLOAT_16>, 2>;
using RS_CameraAuxRoughness = device::RS_Array<float, 2>;
using RS_CameraAuxMetalness = device::RS_Array<float, 2>;
using RS_CameraAuxIntensity = device::RS_Array<float, 2>;
using RS_CameraAuxMaterialType = device::RS_Array<RSE_VisionMaterialType, 2>;
using RS_CameraAuxScreenMotion = device::RS_Array<RVec2f, 2>;
using RS_CameraAuxWorldVelocity = device::RS_Array<RVec3f, 2>;

/// PTUD chain bookkeeping AOVs: throughput accumulates Beer-Lambert + Fresnel
/// along the chain; terminal_type tags how the chain ended (see
/// RSE_VisionRestirChainTerminalType).
using RS_CameraAuxChainThroughput = device::RS_Array<RVec3f, 2>;
using RS_CameraAuxChainTerminalType = device::RS_Array<RSE_VisionRestirChainTerminalType, 2>;

using RS_CameraReservoirDI = device::RS_Array<RS_ReservoirDI, 2>;
using RS_CameraReservoirGI = device::RS_Array<RS_ReservoirGI, 2>;

/// SVGF prev-frame aux snapshots (disocclusion gate next frame).
using RS_CameraPrevAuxNormal = device::RS_Array<RVec3f, 2>;
using RS_CameraPrevAuxDepthZ = device::RS_Array<float, 2>;
using RS_CameraPrevAuxObjectKey = device::RS_Array<RS_CameraPixelObjectKey, 2>;

/// SVGF temporal history: color, luminance moments (m1, m2), length, variance.
using RS_CameraDenoiserRGBHistory = device::RS_Array<RS_Vec3DMem<RSE_FLOAT_16>, 2>;
using RS_CameraDenoiserMomentHistory = device::RS_Array<RVec2f, 2>;
using RS_CameraDenoiserHistoryLength = device::RS_Array<RFloat, 2>;
using RS_CameraDenoiserVariance = device::RS_Array<RFloat, 2>;

}  // namespace logic

}  // namespace rise

#define REFLECT_CAMERA_MEM(camera_mem_type)   \
    REFL_AUTO(                                \
        type(camera_mem_type),                \
        field(gid),                           \
        field(rigid_body_sid),                \
        field(sub_camera_index),              \
        field(camera_signal_handle),          \
        field(aspect_ratio),                  \
        field(image_width),                   \
        field(image_height),                  \
        field(samples_per_pixel),             \
        field(max_depth),                     \
        field(focus_distance),                \
        field(field_of_view),                 \
        field(defocus_angle),                 \
        field(defocus_radius),                \
        field(hyperbolic_near_depth),         \
        field(hyperbolic_far_depth),          \
        field(pixel_width),                   \
        field(pixel_height),                  \
        field(rigid_body_anchor_rb_position), \
        field(interpupillary_distance),       \
        field(convergence_distance),          \
        field(angle_left),                    \
        field(angle_right),                   \
        field(angle_up),                      \
        field(angle_down),                    \
        field(rigid_body),                    \
        field(state),                         \
        field(position),                      \
        field(orientation),                   \
        field(prev_position),                 \
        field(prev_orientation),              \
        field(last_update_time),              \
        field(last_update_passed_time),       \
        field(frame_count)                    \
    )

REFLECT_CAMERA_MEM(rise::logic::RS_CameraMem<rise::RSE_FLOAT_32>)
REFLECT_CAMERA_MEM(rise::logic::RS_CameraMem<rise::RSE_FLOAT_64>)

#endif