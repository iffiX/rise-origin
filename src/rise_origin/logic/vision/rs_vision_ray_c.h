// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VISION_RAY_C_H
#define RS_VISION_RAY_C_H

#include "rise_origin/device/algorithm/rs_random.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"
#include "rise_origin/logic/vision/rs_camera_m.h"
#include "rise_origin/logic/vision/rs_vision_cosmic_background_m.h"
#include "rise_origin/logic/vision/rs_vision_floor_m.h"
#include "rise_origin/logic/vision/rs_vision_material_m.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_triangle.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_conf.h"

namespace rise {

namespace logic {

/**
 * @brief Camera ray-based path tracer.
 *
 * @details One megakernel per camera per frame. Each thread integrates one camera ray to a
 * fixed `max_depth` using a simple Mitsuba-style dielectric / Lambertian / metallic BSDF set
 * ported from the early-rise reference. Cross-surface refraction is tracked via Path B: a
 * single `current_medium_ior` scalar on the per-thread depth loop, updated only when the
 * Schlick + Snell pick lands on refraction. Self-intersection avoidance uses the Wachter and
 * Binder integer-offset method (Ray Tracing Gems vol. 1 ch. 6, 2019) instead of a fixed
 * surface epsilon. The cosmic background is dispatched by its type: a uniform color or
 * equirectangular grid texture, a hand-tunable two-tone gradient with sun disk, or the
 * Hosek-Wilkie 2012 / 2013 analytic clear-sky model. The Hosek-Wilkie 27-coefficient state
 * is cooked once per camera-setup on the host and read by the kernel as a flat float array.
 *
 *     init camera RGB ray -> for d in 0..max_depth:
 *         _findHit (voxel BVH + floor BVH closest hit)
 *         _shadeRGBOnceBasic or _shadeRGBOnceRestir (dispatch by vision_is_restir_enabled)
 *
 * The "info pass" mentioned in the long-form design notes is a future step gated on the
 * camera_aux_* AOV buffers; only the RGB megakernel is shipped here.
 *
 * Symbols, used consistently below:
 *   theta    angle of the ray to world +Z (zenith); used by the sky model
 *   gamma    angle between ray direction and sun direction
 *   eta      ratio ior_from / ior_to feeding Snell's law
 *   F        Schlick Fresnel reflectance scalar
 *   throughput  cumulative BSDF weight along the path
 */
template <RSE_FloatType FloatComputeAccuracy = RSE_FLOAT_32>
class RS_VisionRayCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Vec2C = RS_Vec2D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;

    using RS_FI = RS_FloatIntrinsics<FloatC>;

    // Scene-independent constants.
    static constexpr FloatC RAY_T_MIN = FloatC(1e-4);
    static constexpr FloatC RR_MAX_SURVIVAL_PROB = FloatC(0.95);
    static constexpr FloatC RR_MIN_PROB = FloatC(1e-4);
    static constexpr FloatC ONB_AXIS_ALIGN_THRESHOLD = FloatC(0.999);
    /// Sentinel for "previous bounce was a delta (mirror/glass/sentinel)";
    /// any partner pdf is finite, so the power heuristic returns 1.
    static constexpr FloatC MIS_DELTA_BRDF_PDF = FloatC(1e30);

    /// Grazing-angle reject for sample reconnection; 0.1 ~ 84° from normal.
    static constexpr FloatC RESTIR_RECONNECT_COS_PHI_MIN = FloatC(0.1);
    /// Max source/destination distance ratio for sample reconnection.
    static constexpr FloatC RESTIR_RECONNECT_DIST_RATIO_MAX = FloatC(10);
    /// Reconnection Jacobian must lie in [min, max] or candidate is rejected.
    static constexpr FloatC RESTIR_RECONNECT_JACOBIAN_MIN = FloatC(0.1);
    static constexpr FloatC RESTIR_RECONNECT_JACOBIAN_MAX = FloatC(10);

    /// Filament smooth-surface roughness floor before squaring to GGX alpha.
    static constexpr FloatC PBR_MIN_ROUGHNESS = FloatC(0.045);
    /// Veach 9.2 spec-lobe selection probability clamps so neither lobe starves.
    static constexpr FloatC PBR_SPEC_PROB_MIN = FloatC(0.05);
    static constexpr FloatC PBR_SPEC_PROB_MAX = FloatC(0.95);
    /// Default IOR for shading sites that do not carry one (ReSTIR endpoints).
    static constexpr FloatC PBR_DEFAULT_IOR = FloatC(1.5);

    /// Stack-allocated camera ray.
    struct Ray {
        Vec3C ray_origin;
        Vec3C ray_direction;

        Ray() = default;
        HOST_DEVICE_INLINE_QUALIFIER Ray(const Vec3C& origin, const Vec3C& direction)
            : ray_origin(origin), ray_direction(direction) {}

        HOST_DEVICE_INLINE_QUALIFIER Vec3C at(FloatC t) const {
            return ray_origin + t * ray_direction;
        }
    };

    union HitStateUnion {
        RState hit_state_packed;
        struct HitState {
            RState vision_material_type : 2;
            RState is_front_face : 1;
            RState is_refraction : 1;
            RState is_floor_hit : 1;
            RState has_face_normal : 1;
            RState face : 3;
            RState face_role : 2;
        } hit_state;

        HOST_DEVICE_INLINE_QUALIFIER HitStateUnion() : hit_state_packed(0) {}
        HOST_DEVICE_INLINE_QUALIFIER explicit HitStateUnion(RState packed_state)
            : hit_state_packed(packed_state) {}
    };

    /// Sampled light against a shading point. `emission` is surface radiance L
    /// in W/m^2/sr. Voxels are area-sampled uniformly over the 1-3 front-facing
    /// cube faces for efficiency, giving the `cos_emit * area_visible / dist^2`
    /// area-light geometry factor. Sun is parallel (`dist = RS_MAX_FLOAT`,
    /// `cos_emit = area_visible = 1`; the is_sun branch skips the geometry
    /// multiply).
    struct LightSample {
        Vec3C dir = Vec3C::zero();
        Vec3C position = Vec3C::zero();
        FloatC dist = FloatC(0);
        Vec3C emission = Vec3C::zero();
        FloatC cos_emit = FloatC(0);
        FloatC area_visible = FloatC(0);
        bool is_sun = false;
    };

    /// Per-hit record carrying geometry + material parameters resolved at the hit.
    struct HitRecord {
        Vec3C p = Vec3C::zero();
        Vec3C normal = Vec3C(0, 0, 1);
        /// Smoothed normal for shading and refraction, kept on the geometric side.
        Vec3C shading_normal = Vec3C(0, 0, 1);
        Vec2C uv = Vec2C::zero();
        /// Where the crossing sits on the hit face, in meters.
        Vec2C face_pos = Vec2C::zero();
        FloatC t = FloatC(0);
        Vec3C color = Vec3C::zero();
        Ray scattered;

        Vec3C albedo = Vec3C::zero();
        FloatC roughness = FloatC(1);
        FloatC metalness = FloatC(0);
        FloatC ior = FloatC(1);
        FloatC intensity = FloatC(0);

        RHandle voxel = RS_NULL_HANDLE;
        RHandle vision_material = RS_NULL_HANDLE;
        RIndex structure_gid = RS_NULL_INDEX;
        RIndex vision_material_key = 0;

        HitStateUnion state;
    };

    /**
     * @brief Basic-mode info pass: one deterministic primary ray per pixel,
     *        writes the per-camera AOV buffers consumed by the denoiser edge
     *        stops and reprojection.
     *
     * @details No jitter and no defocus disk so the buffers are stable across
     *          frames. Runs once per camera per frame before the RGB pass.
     *          Miss pixels write the miss sentinel into object_key and zero
     *          into the other AOVs.
     *
     * @param k       simulation kernel.
     * @param camera  target camera handle.
     * @param pixel   flat (i * W + j) pixel index.
     */
    class RSCF_TraceCameraInfoRaysBasic
        : public device::RS_ComputeFunction<RSCF_TraceCameraInfoRaysBasic> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;

            RSize image_width = CA_G(image_width);
            RSize i = RSize(pixel) / image_width;
            RSize j = RSize(pixel) - i * image_width;

            Vec3C ray_origin, ray_direction;
            _initCameraInfoRay(k, camera, pixel, image_width, ray_origin, ray_direction);
            Ray ray(ray_origin, ray_direction);

            HitRecord rec;
            bool any_hit = _findHit(k, ray, rec);

            if (!any_hit) {
                ctx.camera_aux_normal[camera][{i, j}] = RVec3f::zero();
                ctx.camera_aux_depth_z[camera][{i, j}] = float(0);
                ctx.camera_aux_hit_position[camera][{i, j}] = RVec3f::zero();
                ctx.camera_aux_object_key[camera][{i, j}] =
                    RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL;
                ctx.camera_aux_roughness[camera][{i, j}] = float(1);
                ctx.camera_aux_screen_motion[camera][{i, j}] = RVec2f(0.0f, 0.0f);
                return;
            }

            // Perpendicular depth along the camera lookat: stable across small
            // camera motion at the same world point.
            Vec3C cam_position = CA_G(position);
            Quat3C cam_orientation = CA_G(orientation);
            Vec3C lookat = cam_orientation.toOrthoAxisX();
            FloatC perp_depth = (rec.p - cam_position).dot(lookat);

            ctx.camera_aux_normal[camera][{i, j}] =
                RVec3f(float(rec.normal.x), float(rec.normal.y), float(rec.normal.z));
            ctx.camera_aux_depth_z[camera][{i, j}] = float(perp_depth);
            ctx.camera_aux_hit_position[camera][{i, j}] =
                RVec3f(float(rec.p.x), float(rec.p.y), float(rec.p.z));
            ctx.camera_aux_object_key[camera][{i, j}] = _packObjectKey(rec);
            ctx.camera_aux_roughness[camera][{i, j}] = float(rec.roughness);

            Vec3C primary_prev_hit_position;
            _computePrevHitPosition(k, rec.voxel, rec.p, primary_prev_hit_position);
            RVec2f primary_screen_motion(0.0f, 0.0f);
            _computeScreenMotion(k, camera, primary_prev_hit_position, i, j, primary_screen_motion);
            ctx.camera_aux_screen_motion[camera][{i, j}] = primary_screen_motion;
        }
    };

    /**
     * @brief Basic-mode RGB pass: one stratified ray per (pixel, sample),
     *        depth loop with `_shadeRGBOnceBasic` (BRDF scatter + 1-sample NEE
     *        at each opaque bounce), firefly clamp, and atomicAdd into
     *        camera_current_rgb_canvas.
     *
     * @details Resolves the per-thread (pixel, sample) from ray_id, generates
     *          the camera ray with Halton-jittered subpixel offset and a
     *          defocus-disk origin, then runs the depth loop. Russian-roulette
     *          terminates after `vision_trace_russian_roulette_start_depth`
     *          steps. Contribution per step is scaled by 1/spp before the
     *          canvas accumulate.
     *
     * @param k        simulation kernel.
     * @param camera   target camera handle; canvas atomicAdd destination.
     * @param ray_id   unique per-thread id; decodes into (pixel_id, sample_id).
     */
    class RSCF_TraceCameraRGBRaysBasic
        : public device::RS_ComputeFunction<RSCF_TraceCameraRGBRaysBasic> {
       public:
        using device::RS_ComputeFunction<RSCF_TraceCameraRGBRaysBasic>::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle ray_id) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;

            RSize image_width = CA_G(image_width);
            RSize image_height = CA_G(image_height);
            RSize samples_per_pixel = CA_G(samples_per_pixel);
            RSize max_depth = CA_G(max_depth);
            FloatC inv_spp = FloatC(1) / FloatC(samples_per_pixel);

            device::RS_RandomState random_state;
            random_state.init(
                k->seed,
                static_cast<uint64_t>(ray_id),
                static_cast<uint64_t>(CA_G(frame_count)) ^ uint64_t(0x6F92D5B13AC8E74FULL)
            );

            RIndex pixel_id;
            Vec3C ray_origin, ray_direction;
            _initCameraRGBRay(
                k,
                camera,
                ray_id,
                image_width,
                image_height,
                samples_per_pixel,
                ray_origin,
                ray_direction,
                pixel_id
            );
            RIndex i = pixel_id / image_width;
            RIndex j = pixel_id - i * image_width;

            Ray current_ray(ray_origin, ray_direction);
            Vec3C throughput = Vec3C::one();
            Vec3C color_accum = Vec3C::zero();
            FloatC current_medium_ior = FloatC(1);
            // Sentinel = no previous bounce; sky-on-miss at depth 0 weights 1.
            FloatC last_bounce_brdf_pdf = MIS_DELTA_BRDF_PDF;

            for (RSize depth = 0; depth < max_depth; ++depth) {
                HitRecord rec;
                bool any_hit = _findHit(k, current_ray, rec);

                Vec3C contribution;
                bool continue_tracing;
                _shadeRGBOnceBasic(
                    k,
                    depth,
                    any_hit,
                    random_state,
                    current_ray,
                    rec,
                    throughput,
                    current_medium_ior,
                    last_bounce_brdf_pdf,
                    contribution,
                    continue_tracing
                );
                color_accum = color_accum + contribution;
                if (!continue_tracing) break;
            }

            device::deviceAtomicAdd(
                &ctx.camera_current_rgb_canvas[camera][{i, j}], RVec3f(color_accum * inv_spp)
            );
        }
    };

    /**
     * @brief ReSTIR-mode info pass: writes the same primary AOVs as the basic
     *        pass, then additionally walks the deterministic specular chain
     *        from the primary hit (PTUD via `_shadeInfoOnceRestir`) to record
     *        the chain endpoint and its throughput.
     *
     * @details Adds three buckets beyond the basic pass: endpoint shading
     *          AOVs (hit_position, normal, object_key, albedo, roughness,
     *          metalness, material_type, screen_motion),
     *          chain_throughput, and chain_terminal_type. OPAQUE primaries
     *          short-circuit with endpoint = primary and throughput = (1,1,1);
     *          specular primaries follow the chain to the diffuse surface,
     *          cosmic background, or emissive it eventually reaches.
     *
     * @param k       simulation kernel.
     * @param camera  target camera handle.
     * @param pixel   flat (i * W + j) pixel index.
     */
    class RSCF_TraceCameraInfoRaysRestir
        : public device::RS_ComputeFunction<RSCF_TraceCameraInfoRaysRestir> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;

            RSize image_width = CA_G(image_width);
            RSize i = RSize(pixel) / image_width;
            RSize j = RSize(pixel) - i * image_width;

            Vec3C ray_origin, ray_direction;
            _initCameraInfoRay(k, camera, pixel, image_width, ray_origin, ray_direction);
            Ray ray(ray_origin, ray_direction);

            HitRecord rec;
            bool any_hit = _findHit(k, ray, rec);

            // Primary AOVs (shared with basic-mode info pass).
            Vec3C cam_position = CA_G(position);
            Quat3C cam_orientation = CA_G(orientation);
            Vec3C lookat = cam_orientation.toOrthoAxisX();
            if (!any_hit) {
                ctx.camera_aux_normal[camera][{i, j}] = RVec3f::zero();
                ctx.camera_aux_depth_z[camera][{i, j}] = float(0);
                ctx.camera_aux_hit_position[camera][{i, j}] = RVec3f::zero();
                ctx.camera_aux_object_key[camera][{i, j}] =
                    RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL;
                ctx.camera_aux_roughness[camera][{i, j}] = float(1);
                ctx.camera_aux_screen_motion[camera][{i, j}] = RVec2f(0.0f, 0.0f);
            } else {
                FloatC perp_depth = (rec.p - cam_position).dot(lookat);
                ctx.camera_aux_normal[camera][{i, j}] =
                    RVec3f(float(rec.normal.x), float(rec.normal.y), float(rec.normal.z));
                ctx.camera_aux_depth_z[camera][{i, j}] = float(perp_depth);
                ctx.camera_aux_hit_position[camera][{i, j}] =
                    RVec3f(float(rec.p.x), float(rec.p.y), float(rec.p.z));
                ctx.camera_aux_object_key[camera][{i, j}] = _packObjectKey(rec);
                ctx.camera_aux_roughness[camera][{i, j}] = float(rec.roughness);

                Vec3C primary_prev_hit_position;
                _computePrevHitPosition(k, rec.voxel, rec.p, primary_prev_hit_position);
                RVec2f primary_screen_motion(0.0f, 0.0f);
                _computeScreenMotion(
                    k, camera, primary_prev_hit_position, i, j, primary_screen_motion
                );
                ctx.camera_aux_screen_motion[camera][{i, j}] = primary_screen_motion;
            }

            // Specular chain → endpoint + chain bookkeeping AOVs.
            device::RS_RandomState chain_rs;
            chain_rs.init(
                k->seed,
                static_cast<uint64_t>(pixel),
                static_cast<uint64_t>(CA_G(frame_count)) ^ uint64_t(0xB74C1E8F5A29D603ULL)
            );

            HitRecord endpoint_rec;
            Vec3C chain_throughput;
            Vec3C chain_reflect_radiance;
            RSE_VisionRestirChainTerminalType terminal_type;
            _shadeInfoOnceRestir(
                k,
                ray,
                any_hit,
                rec,
                chain_rs,
                endpoint_rec,
                chain_throughput,
                chain_reflect_radiance,
                terminal_type
            );

            ctx.camera_aux_chain_throughput[camera][{i, j}] = RVec3f(
                float(chain_throughput.x), float(chain_throughput.y), float(chain_throughput.z)
            );
            ctx.camera_aux_chain_terminal_type[camera][{i, j}] = terminal_type;

            // Off-chain reflect contributions atomic-added to canvas; the RGB
            // pass adds the on-chain reservoir term on top.
            if (_luminance(chain_reflect_radiance) > FloatC(0)) {
                device::deviceAtomicAdd(
                    &ctx.camera_current_rgb_canvas[camera][{i, j}], RVec3f(chain_reflect_radiance)
                );
            }

            if (terminal_type == RSE_VISION_RESTIR_CHAIN_TERMINAL_COSMIC_BACKGROUND) {
                // endpoint_rec.p holds the terminal ray direction.
                ctx.camera_aux_endpoint_hit_position[camera][{i, j}] = RVec3f(
                    float(endpoint_rec.p.x), float(endpoint_rec.p.y), float(endpoint_rec.p.z)
                );
                ctx.camera_aux_endpoint_normal[camera][{i, j}] = RVec3f::zero();
                ctx.camera_aux_endpoint_object_key[camera][{i, j}] =
                    RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL;
                ctx.camera_aux_endpoint_albedo[camera][{i, j}] = RVec3f::zero();
                ctx.camera_aux_endpoint_roughness[camera][{i, j}] = float(1);
                ctx.camera_aux_endpoint_metalness[camera][{i, j}] = float(0);
                ctx.camera_aux_endpoint_intensity[camera][{i, j}] = float(0);
                ctx.camera_aux_endpoint_material_type[camera][{i, j}] = RSE_VISION_MATERIAL_OPAQUE;
                ctx.camera_aux_endpoint_screen_motion[camera][{i, j}] = RVec2f(0.0f, 0.0f);
                return;
            }

            auto& ehs = endpoint_rec.state.hit_state;
            RSE_VisionMaterialType endpoint_material_type =
                static_cast<RSE_VisionMaterialType>(ehs.vision_material_type);
            ctx.camera_aux_endpoint_hit_position[camera][{i, j}] =
                RVec3f(float(endpoint_rec.p.x), float(endpoint_rec.p.y), float(endpoint_rec.p.z));
            ctx.camera_aux_endpoint_normal[camera][{i, j}] = RVec3f(
                float(endpoint_rec.normal.x),
                float(endpoint_rec.normal.y),
                float(endpoint_rec.normal.z)
            );
            ctx.camera_aux_endpoint_object_key[camera][{i, j}] = _packObjectKey(endpoint_rec);
            ctx.camera_aux_endpoint_albedo[camera][{i, j}] = RVec3f(
                float(endpoint_rec.albedo.x),
                float(endpoint_rec.albedo.y),
                float(endpoint_rec.albedo.z)
            );
            ctx.camera_aux_endpoint_roughness[camera][{i, j}] = float(endpoint_rec.roughness);
            ctx.camera_aux_endpoint_metalness[camera][{i, j}] = float(endpoint_rec.metalness);
            ctx.camera_aux_endpoint_intensity[camera][{i, j}] = float(endpoint_rec.intensity);
            ctx.camera_aux_endpoint_material_type[camera][{i, j}] = endpoint_material_type;

            Vec3C endpoint_prev_hit_position;
            _computePrevHitPosition(
                k, endpoint_rec.voxel, endpoint_rec.p, endpoint_prev_hit_position
            );
            RVec2f endpoint_screen_motion(0.0f, 0.0f);
            _computeScreenMotion(
                k, camera, endpoint_prev_hit_position, i, j, endpoint_screen_motion
            );
            ctx.camera_aux_endpoint_screen_motion[camera][{i, j}] = endpoint_screen_motion;
        }
    };

    /**
     * @brief ReSTIR-mode RGB pass: a single `_shadeRGBOnceRestir` call per
     *        pixel that reads the endpoint AOVs + chain throughput + DI/GI
     *        reservoirs written by the preceding info / resample / spatial
     *        kernels.
     *
     * @details No depth loop: the info pass walked the PTUD specular chain to
     *          the endpoint, the reservoirs already encode direct + one-bounce
     *          indirect, so a single read suffices. Launched per pixel (over
     *          `camera_info_ray_layout`); per-frame ReSTIR candidate budget is
     *          driven by `samples_per_pixel` in the resample kernels.
     *
     * @param k       simulation kernel.
     * @param camera  target camera handle; canvas atomicAdd destination.
     * @param pixel   flat (i * W + j) pixel index.
     */
    class RSCF_TraceCameraRGBRaysRestir
        : public device::RS_ComputeFunction<RSCF_TraceCameraRGBRaysRestir> {
       public:
        using device::RS_ComputeFunction<RSCF_TraceCameraRGBRaysRestir>::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;

            RSize image_width = CA_G(image_width);
            RIndex i = RSize(pixel) / image_width;
            RIndex j = RSize(pixel) - i * image_width;

            Vec3C contribution;
            _shadeRGBOnceRestir(k, camera, i, j, contribution);

            device::deviceAtomicAdd(
                &ctx.camera_current_rgb_canvas[camera][{i, j}], RVec3f(contribution)
            );
        }
    };

    /**
     * @brief Per-emissive-voxel kernel that writes the ReSTIR DI light-weight at
     *        slot `1 + emissive_idx`. Slot 0 is the sun and is written from the
     *        host before this launch.
     *
     * @details Resolves the voxel handle through `ctx.structure_emissive_voxel_handles`,
     *          reads the vision material's albedo and intensity, and stores the luminance
     *          of `intensity` into `light_weight`. Then inclusive scan into `light_cdf`
     *          runs after this kernel.
     *
     * @param k             simulation kernel.
     * @param emissive_idx  flat index into `ctx.structure_emissive_voxel_handles`.
     */
    class RSCF_WriteEmissiveLightWeights
        : public device::RS_ComputeFunction<RSCF_WriteEmissiveLightWeights> {
       public:
        using device::RS_ComputeFunction<RSCF_WriteEmissiveLightWeights>::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k) {
            auto& ctx = k->ctx;
            RIndex i = index1D();
            if (RSize(i) >= ctx.structure_emissive_voxel_handles.size()) return;
            RHandle voxel = ctx.structure_emissive_voxel_handles[i];
            RHandle vision_material = V_G(voxel, vision_material);
            // Approximation using just intensity, for a better estimate, needs to
            // consider emission color (for texture probably average) and voxel size.
            FloatC intensity = FloatC(VSM_G(vision_material, emission_intensity));
            ctx.light_weight[RSize(i) + 1] = float(intensity);
        }
    };

    /**
     * @brief Per-pixel ReSTIR DI resampler. Generates initial candidates from
     *        the light CDF, merges the prior frame's reservoir with M-clamp
     *        decay and multi-field disocclusion gate, shadow-walks the
     *        survivor, and writes the updated reservoir.
     */
    class RSCF_ResampleReservoirDI : public device::RS_ComputeFunction<RSCF_ResampleReservoirDI> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;

            RSize image_width = CA_G(image_width);
            RIndex ii = RSize(pixel) / image_width;
            RIndex jj = RSize(pixel) - ii * image_width;

            // ReSTIR keys on the chain endpoint (== primary for OPAQUE primaries).
            // Miss-out: no surface at the endpoint either, clear the reservoir.
            auto object_key = ctx.camera_aux_endpoint_object_key[camera][{ii, jj}];
            if (object_key.isCosmicBackground()) {
                ctx.camera_current_reservoir_di[camera][{ii, jj}] = RS_ReservoirDI{};
                return;
            }

            Vec3C hit_position = Vec3C(ctx.camera_aux_endpoint_hit_position[camera][{ii, jj}]);
            Vec3C normal = Vec3C(ctx.camera_aux_endpoint_normal[camera][{ii, jj}]);
            Vec3C albedo = Vec3C(ctx.camera_aux_endpoint_albedo[camera][{ii, jj}]);
            FloatC roughness = FloatC(ctx.camera_aux_endpoint_roughness[camera][{ii, jj}]);
            FloatC metalness = FloatC(ctx.camera_aux_endpoint_metalness[camera][{ii, jj}]);
            RSE_VisionMaterialType material_type =
                ctx.camera_aux_endpoint_material_type[camera][{ii, jj}];
            Vec3C camera_position = Vec3C(CA_G(position));
            Vec3C view_dir = (camera_position - hit_position).normalized();

            device::RS_RandomState rs;
            rs.init(
                k->seed,
                static_cast<uint64_t>(pixel),
                static_cast<uint64_t>(CA_G(frame_count)) ^ uint64_t(0x13EF6B90A4D87C52ULL)
            );

            // M_init initial candidates: per-camera samples_per_pixel.
            RS_ReservoirDI reservoir{};
            RSize initial_candidate_count = CA_G(samples_per_pixel);
            for (RSize candidate_idx = 0; candidate_idx < initial_candidate_count;
                 ++candidate_idx) {
                FloatC rand_u = rs.template rand<FloatC>();
                RIndex light_index = _sampleLight(k, rand_u);
                FloatC source_pdf = _lightSourcePdf(k, light_index);
                if (source_pdf <= FloatC(0)) continue;

                LightSample light_sample = _sampleLightSample(k, light_index, hit_position, rs);
                Vec3C brdf_cos = _brdfCos(
                    material_type,
                    albedo,
                    metalness,
                    roughness,
                    PBR_DEFAULT_IOR,
                    normal,
                    view_dir,
                    light_sample.dir
                );
                Vec3C contribution =
                    brdf_cos.elemMul(light_sample.emission) * _lightGeometryFactor(light_sample);
                if (_luminance(contribution) <= FloatC(RS_EPS)) continue;

                // Per-candidate visibility: fold the shadow walk into target_pdf so
                // the WRS can only ever pick survivors that actually contribute.
                // Without this, sample_weight = tp / source_pdf counts the candidate
                // at its unshadowed brightness; an occluded survivor then carries a
                // huge restir_weight = sum_all_tp / tp_survivor into the depth-0
                // read and the visible-survivor frames overshoot to sum_all_tp
                // instead of NEE's sum_visible_tp.
                FloatC max_distance_c = light_sample.is_sun ? FloatC(RS_MAX_FLOAT)
                                                            : light_sample.dist - FloatC(RS_EPS);
                RHandle light_handle_c =
                    (light_index == RS_VISION_SUN_LIGHT_INDEX)
                        ? RS_NULL_HANDLE
                        : ctx.structure_emissive_voxel_handles[light_index - 1];
                Vec3C origin_c = _offsetRayOrigin(hit_position, normal);
                RColor3f transmission_c = _shadowWalkWithTransmission(
                    k, origin_c, light_sample.dir, max_distance_c, light_handle_c
                );
                Vec3C transmission_c_vec = Vec3C(transmission_c.toVec3D());
                Vec3C contribution_vis = contribution.elemMul(transmission_c_vec);
                FloatC target_pdf = _luminance(contribution_vis);
                if (target_pdf <= FloatC(0)) continue;
                FloatC sample_weight = target_pdf / source_pdf;
                RVec3f light_position(
                    float(light_sample.position.x),
                    float(light_sample.position.y),
                    float(light_sample.position.z)
                );
                _reservoirUpdateDI(
                    sample_weight,
                    light_index,
                    light_position,
                    transmission_c,
                    target_pdf,
                    rs,
                    reservoir
                );
            }
            // Bitterli 2020 Alg 5: M counts every candidate examined, including
            // those skipped above for being below-horizon, dim, or occluded.
            // _reservoirUpdateDI only increments sample_count when reached, so
            // we restore the correct denominator here. Without this, fresh-only
            // resamples on scenes with many small voxel lights end up with
            // sample_count ≪ spp, inflating restir_weight = weight_sum /
            // (sample_count × target_pdf) by spp / sample_count_observed —
            // the visible white spike on diffuse surfaces during camera
            // motion when the temporal gate fails.
            reservoir.sample_count = initial_candidate_count;

            // Reproject through last frame's camera + surface velocity so that
            // camera or object motion still aligns to the right pixel before
            // gating. Clamp with signed math to keep negative reprojections
            // from wrapping when cast to unsigned RIndex.
            RSize image_height = CA_G(image_height);
            RVec2f screen_motion = ctx.camera_aux_endpoint_screen_motion[camera][{ii, jj}];
            int ii_prev_signed =
                int(RS_FI::rs_round(FloatC(ii) - FloatC(screen_motion.y) * FloatC(image_height)));
            int jj_prev_signed =
                int(RS_FI::rs_round(FloatC(jj) - FloatC(screen_motion.x) * FloatC(image_width)));
            ii_prev_signed = RS_MAX(0, RS_MIN(ii_prev_signed, int(image_height) - 1));
            jj_prev_signed = RS_MAX(0, RS_MIN(jj_prev_signed, int(image_width) - 1));
            RS_ReservoirDI reservoir_prev =
                ctx.camera_prev_reservoir_di[camera]
                                            [{RIndex(ii_prev_signed), RIndex(jj_prev_signed)}];

            // Multi-field temporal gate. Position threshold rejects same-key
            // surfaces that translated; normal threshold rejects rotations.
            FloatC disocclusion_position_threshold =
                FloatC(k->vision_restir_disocclusion_position_threshold);
            FloatC normal_cos_threshold = FloatC(k->vision_restir_disocclusion_normal_threshold);
            bool is_prev_valid =
                !k->vision_is_restir_reservoir_stale && reservoir_prev.sample_count > 0 &&
                reservoir_prev.object_key == object_key &&
                (Vec3C(reservoir_prev.prev_hit_position) - hit_position).length() <
                    disocclusion_position_threshold &&
                Vec3C(reservoir_prev.prev_normal).dot(normal) > normal_cos_threshold;

            if (is_prev_valid) {
                LightSample light_sample_prev = _evalStoredLightSample(
                    k, reservoir_prev.light_index, hit_position, reservoir_prev.light_position
                );
                Vec3C brdf_cos_prev = _brdfCos(
                    material_type,
                    albedo,
                    metalness,
                    roughness,
                    PBR_DEFAULT_IOR,
                    normal,
                    view_dir,
                    light_sample_prev.dir
                );
                // p̂ must be evaluated consistently with storage: the stored
                // reservoir_prev.target_pdf included the candidate's shadow-walked
                // transmission, so the re-eval must too. Reusing the stored
                // transmission (biased Algorithm 6) keeps the ratio
                // target_pdf_prev_under_curr / reservoir_prev.target_pdf bounded
                // by the BRDF / geometry change between frames rather than
                // exploding by 1/transmission_storage on partial-shadow survivors.
                Vec3C contribution_prev = brdf_cos_prev.elemMul(light_sample_prev.emission) *
                                          _lightGeometryFactor(light_sample_prev);
                Vec3C contribution_prev_vis =
                    contribution_prev.elemMul(Vec3C(reservoir_prev.transmission.toVec3D()));
                FloatC target_pdf_prev_under_curr = _luminance(contribution_prev_vis);

                // Re-score the prior sample under current shading and merge in.
                FloatC sample_weight_prev = target_pdf_prev_under_curr *
                                            FloatC(reservoir_prev.weight_sum) /
                                            (FloatC(reservoir_prev.target_pdf) + FloatC(RS_EPS));
                _reservoirUpdateDI(
                    sample_weight_prev,
                    reservoir_prev.light_index,
                    reservoir_prev.light_position,
                    reservoir_prev.transmission,
                    target_pdf_prev_under_curr,
                    rs,
                    reservoir
                );
                // Combined sample count is this frame plus prior frame.
                reservoir.sample_count =
                    initial_candidate_count + RSize(reservoir_prev.sample_count);

                // Cap retained history so stale samples can't dominate; rescale
                // weight_sum to keep the per-sample average unchanged.
                RSize m_cap =
                    RS_MAX(RSize(1), RSize(k->vision_restir_di_m_cap) * initial_candidate_count);
                if (reservoir.sample_count > m_cap) {
                    FloatC scale = FloatC(m_cap) / FloatC(reservoir.sample_count);
                    reservoir.weight_sum = float(FloatC(reservoir.weight_sum) * scale);
                    reservoir.sample_count = m_cap;
                }
            }

            // Visibility on the survivor.
            if (reservoir.sample_count > 0 && reservoir.target_pdf > FloatC(0)) {
                LightSample light_sample_survivor = _evalStoredLightSample(
                    k, reservoir.light_index, hit_position, reservoir.light_position
                );
                FloatC max_distance = light_sample_survivor.is_sun
                                          ? FloatC(RS_MAX_FLOAT)
                                          : light_sample_survivor.dist - FloatC(RS_EPS);
                RHandle light_handle =
                    (reservoir.light_index == RS_VISION_SUN_LIGHT_INDEX)
                        ? RS_NULL_HANDLE
                        : ctx.structure_emissive_voxel_handles[reservoir.light_index - 1];
                Vec3C origin = _offsetRayOrigin(hit_position, normal);
                reservoir.transmission = _shadowWalkWithTransmission(
                    k, origin, light_sample_survivor.dir, max_distance, light_handle
                );
            } else {
                reservoir.transmission = RColor3f::zero();
            }

            // Identity and disocclusion bookkeeping for next frame.
            reservoir.object_key = object_key;
            reservoir.prev_hit_position =
                RVec3f(float(hit_position.x), float(hit_position.y), float(hit_position.z));
            reservoir.prev_normal = RVec3f(float(normal.x), float(normal.y), float(normal.z));
            ctx.camera_current_reservoir_di[camera][{ii, jj}] = reservoir;
        }
    };

    /**
     * @brief Per-pixel biased spatial reuse pass over the ReSTIR DI reservoir.
     *
     * @details Reads our pixel's reservoir from `read_reservoir`, samples K
     *          neighbors inside `vision_restir_di_spatial_radius`, gates them
     *          by object key + depth ratio + normal similarity, re-evaluates
     *          each surviving neighbor's stored light sample at our shading
     *          point, and folds it into a fresh Bitterli WRS update. Visibility
     *          is taken from the neighbor's previously-walked transmission
     *          (biased Algorithm 6); no shadow rays are spawned here. Writes
     *          the merged reservoir to `write_reservoir`. Launched twice per
     *          frame between resample and the RGB pass; the executor flips
     *          `write_reservoir` and `read_reservoir` between iterations via
     *          host-side `exchange`.
     *
     * @note    Biased Algorithm 6 (Bitterli 2020 §4.4): the survivor inherits
     *          the neighbor's previously-walked transmission instead of being
     *          re-shadowed at our shading point. The unbiased variant would
     *          shadow-walk each neighbor at our pixel — K extra rays per pass.
     *          The bias shows as darkening near shadow edges where neighbors
     *          across the edge have different occlusion. The disocclusion gate
     *          (object_key + depth + normal) keeps that error small for
     *          neighbors with similar geometry.
     *
     * @param k                simulation kernel.
     * @param camera           target camera handle.
     * @param pixel            flat (i * W + j) pixel index.
     * @param read_reservoir   reservoir buffer the merge reads from.
     * @param write_reservoir  reservoir buffer the merge writes to.
     * @param iter_index       0-based pass index, used to differentiate the
     *                         RNG seed across passes so neighbors decorrelate.
     */
    class RSCF_SpatialReuseReservoirDI
        : public device::RS_ComputeFunction<RSCF_SpatialReuseReservoirDI> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;

            RSize image_width = CA_G(image_width);
            RSize image_height = CA_G(image_height);
            RIndex ii = RSize(pixel) / image_width;
            RIndex jj = RSize(pixel) - ii * image_width;

            // ReSTIR shading state comes from the chain endpoint; the depth
            // similarity gate uses the camera-facing primary surface so two
            // pixels seeing the same glass surface still pair up.
            auto object_key = ctx.camera_aux_endpoint_object_key[camera][{ii, jj}];
            if (object_key.isCosmicBackground()) {
                ctx.camera_spatial_reservoir_di[camera][{ii, jj}] = RS_ReservoirDI{};
                return;
            }

            Vec3C hit_position = Vec3C(ctx.camera_aux_endpoint_hit_position[camera][{ii, jj}]);
            Vec3C normal = Vec3C(ctx.camera_aux_endpoint_normal[camera][{ii, jj}]);
            Vec3C albedo = Vec3C(ctx.camera_aux_endpoint_albedo[camera][{ii, jj}]);
            FloatC roughness = FloatC(ctx.camera_aux_endpoint_roughness[camera][{ii, jj}]);
            FloatC metalness = FloatC(ctx.camera_aux_endpoint_metalness[camera][{ii, jj}]);
            FloatC depth_curr = FloatC(ctx.camera_aux_depth_z[camera][{ii, jj}]);
            RSE_VisionMaterialType material_type =
                ctx.camera_aux_endpoint_material_type[camera][{ii, jj}];
            Vec3C camera_position = Vec3C(CA_G(position));
            Vec3C view_dir = (camera_position - hit_position).normalized();

            // Seed from this frame's temporal output; neighbor reads use the
            // same buffer so the spatial pass never reuses its own output.
            RS_ReservoirDI reservoir = ctx.camera_current_reservoir_di[camera][{ii, jj}];

            device::RS_RandomState rs;
            rs.init(
                k->seed,
                static_cast<uint64_t>(pixel),
                static_cast<uint64_t>(CA_G(frame_count)) ^ uint64_t(0xD8A4F92C7E6135B1ULL)
            );

            FloatC spatial_radius = FloatC(k->vision_restir_di_spatial_radius);
            RSize neighbor_count = k->vision_restir_di_spatial_neighbors;
            RSize history_ratio = RSize(k->vision_restir_di_m_cap);
            RSize candidate_count = CA_G(samples_per_pixel);
            RSize sample_count_cap = history_ratio * candidate_count;
            FloatC NORMAL_COS_THRESHOLD = FloatC(k->vision_restir_disocclusion_normal_threshold);
            constexpr FloatC DEPTH_REL_THRESHOLD = FloatC(0.1);

            for (RSize n = 0; n < neighbor_count; ++n) {
                FloatC rand_u = rs.template rand<FloatC>();
                FloatC rand_v = rs.template rand<FloatC>();
                FloatC dx = (rand_u * FloatC(2) - FloatC(1)) * spatial_radius;
                FloatC dy = (rand_v * FloatC(2) - FloatC(1)) * spatial_radius;
                int ii_n_signed = int(RS_FI::rs_round(FloatC(ii) + dy));
                int jj_n_signed = int(RS_FI::rs_round(FloatC(jj) + dx));
                ii_n_signed = RS_MAX(0, RS_MIN(ii_n_signed, int(image_height) - 1));
                jj_n_signed = RS_MAX(0, RS_MIN(jj_n_signed, int(image_width) - 1));
                RIndex ii_n = RIndex(ii_n_signed);
                RIndex jj_n = RIndex(jj_n_signed);
                if (ii_n == ii && jj_n == jj) continue;

                auto neighbor_key = ctx.camera_aux_endpoint_object_key[camera][{ii_n, jj_n}];
                if (neighbor_key.isCosmicBackground()) continue;
                if (neighbor_key != object_key) continue;

                FloatC depth_neighbor = FloatC(ctx.camera_aux_depth_z[camera][{ii_n, jj_n}]);
                FloatC depth_ratio_denom = RS_MAX(RS_ABS(depth_curr), FloatC(RS_EPS));
                if (RS_ABS(depth_neighbor - depth_curr) > DEPTH_REL_THRESHOLD * depth_ratio_denom) {
                    continue;
                }
                Vec3C normal_neighbor = Vec3C(ctx.camera_aux_endpoint_normal[camera][{ii_n, jj_n}]);
                if (normal_neighbor.dot(normal) < NORMAL_COS_THRESHOLD) continue;

                RS_ReservoirDI reservoir_neighbor =
                    ctx.camera_current_reservoir_di[camera][{ii_n, jj_n}];
                if (reservoir_neighbor.sample_count == 0) continue;
                if (FloatC(reservoir_neighbor.target_pdf) <= FloatC(0)) continue;

                LightSample light_sample_neighbor = _evalStoredLightSample(
                    k,
                    reservoir_neighbor.light_index,
                    hit_position,
                    reservoir_neighbor.light_position
                );
                Vec3C brdf_cos = _brdfCos(
                    material_type,
                    albedo,
                    metalness,
                    roughness,
                    PBR_DEFAULT_IOR,
                    normal,
                    view_dir,
                    light_sample_neighbor.dir
                );
                // p̂ must include visibility for consistency with the neighbor's
                // stored target_pdf; biased Algorithm 6 reuses the neighbor's
                // shadow-walked transmission rather than re-walking at our
                // shading point. Without this multiply, the ratio
                // target_pdf_at_curr / reservoir_neighbor.target_pdf blows up
                // when the neighbor's survivor was in partial shadow, dragging
                // an inflated sample_weight into our weight_sum.
                Vec3C contribution = brdf_cos.elemMul(light_sample_neighbor.emission) *
                                     _lightGeometryFactor(light_sample_neighbor);
                Vec3C contribution_vis =
                    contribution.elemMul(Vec3C(reservoir_neighbor.transmission.toVec3D()));
                FloatC target_pdf_at_curr = _luminance(contribution_vis);
                if (target_pdf_at_curr <= FloatC(0)) continue;

                // Re-score the neighbor's sample under our shading and merge.
                FloatC sample_weight = target_pdf_at_curr * FloatC(reservoir_neighbor.weight_sum) /
                                       (FloatC(reservoir_neighbor.target_pdf) + FloatC(RS_EPS));
                _reservoirUpdateDI(
                    sample_weight,
                    reservoir_neighbor.light_index,
                    reservoir_neighbor.light_position,
                    reservoir_neighbor.transmission,
                    target_pdf_at_curr,
                    rs,
                    reservoir
                );
                reservoir.sample_count =
                    reservoir.sample_count - 1 + RSize(reservoir_neighbor.sample_count);
            }

            // Cap accumulated samples; rescale weight to keep its mean.
            RSize m_cap_spatial = RS_MAX(RSize(1), sample_count_cap * (neighbor_count + RSize(1)));
            if (reservoir.sample_count > m_cap_spatial) {
                FloatC scale = FloatC(m_cap_spatial) / FloatC(reservoir.sample_count);
                reservoir.weight_sum = float(FloatC(reservoir.weight_sum) * scale);
                reservoir.sample_count = m_cap_spatial;
            }

            // Write to the spatial buffer; the temporal buffer stays untouched
            // so next frame's merge keeps reading from this frame's resample.
            reservoir.object_key = object_key;
            reservoir.prev_hit_position =
                RVec3f(float(hit_position.x), float(hit_position.y), float(hit_position.z));
            reservoir.prev_normal = RVec3f(float(normal.x), float(normal.y), float(normal.z));
            ctx.camera_spatial_reservoir_di[camera][{ii, jj}] = reservoir;
        }
    };

    /**
     * @brief Per-pixel ReSTIR GI resampler over the chain endpoint.
     *
     * @details For each candidate, BRDF-samples a bounce direction at the
     *          endpoint shading point, traces the bounce ray, and gathers
     *          `l_incoming`: sky radiance on a miss (MIS-weighted against
     *          NEE-sun), direct emission on an emissive hit, or sub-NEE
     *          radiance on an opaque hit gated by
     *          `vision_restir_gi_is_bounce_nee_enabled`. The candidate's
     *          target_pdf is `luminance(brdf_cos · l_incoming)`. Then merges
     *          the reprojected prior-frame reservoir with M-clamp decay using
     *          the shared disocclusion gate of object_key + endpoint position
     *          threshold + normal threshold. Visibility is implicit in the
     *          bounce-ray closest-hit so no survivor shadow walk is needed.
     *          Misses on the endpoint, and specular or emissive endpoints
     *          handled separately by the chain, clear the GI reservoir.
     *
     * @param k       simulation kernel.
     * @param camera  target camera handle.
     * @param pixel   flat (i * W + j) pixel index.
     */
    class RSCF_ResampleReservoirGI : public device::RS_ComputeFunction<RSCF_ResampleReservoirGI> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;

            RSize image_width = CA_G(image_width);
            RIndex ii = RSize(pixel) / image_width;
            RIndex jj = RSize(pixel) - ii * image_width;

            // Miss-out: no endpoint surface, clear the GI reservoir.
            auto object_key = ctx.camera_aux_endpoint_object_key[camera][{ii, jj}];
            if (object_key.isCosmicBackground()) {
                ctx.camera_current_reservoir_gi[camera][{ii, jj}] = RS_ReservoirGI{};
                return;
            }
            // GI only meaningful on opaque diffuse endpoints; specular endpoints
            // would have stayed in the chain and emissives have no further bounce.
            auto material_type = ctx.camera_aux_endpoint_material_type[camera][{ii, jj}];
            if (material_type != RSE_VISION_MATERIAL_OPAQUE) {
                ctx.camera_current_reservoir_gi[camera][{ii, jj}] = RS_ReservoirGI{};
                return;
            }

            Vec3C hit_position = Vec3C(ctx.camera_aux_endpoint_hit_position[camera][{ii, jj}]);
            Vec3C normal = Vec3C(ctx.camera_aux_endpoint_normal[camera][{ii, jj}]);
            Vec3C albedo = Vec3C(ctx.camera_aux_endpoint_albedo[camera][{ii, jj}]);
            FloatC roughness = FloatC(ctx.camera_aux_endpoint_roughness[camera][{ii, jj}]);
            FloatC metalness = FloatC(ctx.camera_aux_endpoint_metalness[camera][{ii, jj}]);
            // Endpoints reached through a specular chain have a "true" view
            // direction equal to the chain ray's last direction; we approximate
            // with the camera-based view direction. Lambertian endpoints are
            // view-invariant so the approximation only affects rough-metallic.
            Vec3C camera_position = Vec3C(CA_G(position));
            Vec3C view_dir = (camera_position - hit_position).normalized();

            device::RS_RandomState rs;
            rs.init(
                k->seed ^ uint64_t(0x6122),
                static_cast<uint64_t>(pixel),
                static_cast<uint64_t>(CA_G(frame_count))
            );

            // M_init initial candidates: per-camera samples_per_pixel.
            RS_ReservoirGI reservoir{};
            RSize initial_candidate_count = CA_G(samples_per_pixel);
            for (RSize candidate_idx = 0; candidate_idx < initial_candidate_count;
                 ++candidate_idx) {
                // BRDF-sample a bounce direction via the unified two-lobe scatter.
                HitRecord scatter_rec;
                scatter_rec.p = hit_position;
                scatter_rec.normal = normal;
                scatter_rec.albedo = albedo;
                scatter_rec.roughness = roughness;
                scatter_rec.metalness = metalness;
                scatter_rec.ior = PBR_DEFAULT_IOR;
                Ray incoming(_offsetRayOrigin(hit_position, normal), -view_dir);
                _scatterOpaque(incoming, rs, scatter_rec);
                Vec3C bounce_dir = scatter_rec.scattered.ray_direction;
                FloatC brdf_pdf = _brdfPdf(
                    material_type,
                    albedo,
                    metalness,
                    roughness,
                    PBR_DEFAULT_IOR,
                    normal,
                    view_dir,
                    bounce_dir
                );
                if (brdf_pdf <= FloatC(RS_EPS)) continue;

                // Trace the bounce ray.
                Ray bounce_ray(_offsetRayOrigin(hit_position, normal), bounce_dir);
                HitRecord bounce_rec;
                bool bounce_any_hit = _findHit(k, bounce_ray, bounce_rec);

                Vec3C l_incoming = Vec3C::zero();
                Vec3C bounce_hit_position = Vec3C::zero();
                Vec3C bounce_hit_normal = Vec3C::zero();
                FloatC mis_brdf = FloatC(1);

                if (!bounce_any_hit) {
                    // BRDF into sky: full sky background + sun disc with MIS
                    // weight subtracted (mirrors `_shadeRGBOnceBasic` sky-miss
                    // path so the sun is correctly captured without double
                    // counting against NEE's sun pick).
                    Vec3C sky = _findHitBackground(k, bounce_dir);
                    Vec3C sun_disk = _evalCosmicBackgroundSunDisk(k, bounce_dir);
                    if (_luminance(sun_disk) > FloatC(0)) {
                        FloatC p_nee = _evalNeePdfSun(k, bounce_dir);
                        FloatC sun_mis_brdf = _powerHeuristicMIS(brdf_pdf, p_nee);
                        sky = sky + sun_disk * (sun_mis_brdf - FloatC(1));
                    }
                    l_incoming = sky;
                } else {
                    auto& bhs = bounce_rec.state.hit_state;
                    RSE_VisionMaterialType bounce_mt =
                        static_cast<RSE_VisionMaterialType>(bhs.vision_material_type);
                    bounce_hit_position = bounce_rec.p;
                    bounce_hit_normal = bounce_rec.normal;

                    if (bounce_mt == RSE_VISION_MATERIAL_EMISSIVE) {
                        l_incoming = bounce_rec.albedo * bounce_rec.intensity;
                        FloatC p_nee_voxel =
                            _evalNeePdfVoxel(k, hit_position, bounce_dir, bounce_rec.voxel);
                        if (p_nee_voxel > FloatC(0)) {
                            mis_brdf = _powerHeuristicMIS(brdf_pdf, p_nee_voxel);
                        }
                    } else if (bounce_mt == RSE_VISION_MATERIAL_OPAQUE) {
                        if (k->vision_restir_gi_is_bounce_nee_enabled) {
                            l_incoming = _evalNeeAtBounce(
                                k,
                                bounce_rec.p,
                                bounce_rec.normal,
                                -bounce_dir,
                                bounce_mt,
                                bounce_rec.albedo,
                                bounce_rec.metalness,
                                bounce_rec.roughness,
                                rs
                            );
                        }
                        l_incoming = l_incoming + _gatherIndirectPath(
                                                      k,
                                                      bounce_rec.p,
                                                      bounce_rec.normal,
                                                      bounce_rec.albedo,
                                                      bounce_rec.metalness,
                                                      bounce_rec.roughness,
                                                      bounce_dir,
                                                      CA_G(max_depth),
                                                      rs
                                                  );
                    }
                    // TRANSPARENT bounce leaves l_incoming = 0.
                }

                Vec3C l_weighted = l_incoming * mis_brdf;
                Vec3C brdf_cos = _brdfCos(
                    material_type,
                    albedo,
                    metalness,
                    roughness,
                    PBR_DEFAULT_IOR,
                    normal,
                    view_dir,
                    bounce_dir
                );
                Vec3C contribution = brdf_cos.elemMul(l_weighted);
                FloatC target_pdf = _luminance(contribution);
                if (target_pdf <= FloatC(RS_EPS)) continue;

                FloatC sample_weight = target_pdf / brdf_pdf;
                RVec3f bounce_direction_storage(
                    float(bounce_dir.x), float(bounce_dir.y), float(bounce_dir.z)
                );
                RVec3f bounce_hit_position_storage(
                    float(bounce_hit_position.x),
                    float(bounce_hit_position.y),
                    float(bounce_hit_position.z)
                );
                RVec3f bounce_hit_normal_storage(
                    float(bounce_hit_normal.x),
                    float(bounce_hit_normal.y),
                    float(bounce_hit_normal.z)
                );
                RColor3f l_incoming_storage(
                    float(l_weighted.x), float(l_weighted.y), float(l_weighted.z)
                );
                _reservoirUpdateGI(
                    sample_weight,
                    bounce_direction_storage,
                    bounce_hit_position_storage,
                    bounce_hit_normal_storage,
                    l_incoming_storage,
                    target_pdf,
                    rs,
                    reservoir
                );
            }
            // M counts every candidate examined (Bitterli 2020 Alg 5);
            // _reservoirUpdateGI only fires for non-skipped candidates, so
            // restore the correct denominator here. See DI resample for the
            // detailed reasoning — same firefly mechanism on fresh GI resets
            // when bounce candidates are skipped for brdf_pdf or target_pdf
            // below RS_EPS.
            reservoir.sample_count = initial_candidate_count;

            // Temporal merge via endpoint reprojection.
            RSize image_height = CA_G(image_height);
            RVec2f screen_motion = ctx.camera_aux_endpoint_screen_motion[camera][{ii, jj}];
            int ii_prev_signed =
                int(RS_FI::rs_round(FloatC(ii) - FloatC(screen_motion.y) * FloatC(image_height)));
            int jj_prev_signed =
                int(RS_FI::rs_round(FloatC(jj) - FloatC(screen_motion.x) * FloatC(image_width)));
            ii_prev_signed = RS_MAX(0, RS_MIN(ii_prev_signed, int(image_height) - 1));
            jj_prev_signed = RS_MAX(0, RS_MIN(jj_prev_signed, int(image_width) - 1));
            RS_ReservoirGI reservoir_prev =
                ctx.camera_prev_reservoir_gi[camera]
                                            [{RIndex(ii_prev_signed), RIndex(jj_prev_signed)}];

            FloatC disocclusion_position_threshold =
                FloatC(k->vision_restir_disocclusion_position_threshold);
            FloatC normal_cos_threshold = FloatC(k->vision_restir_disocclusion_normal_threshold);
            // Drop the stored sample every N frames if its geometry went stale.
            RSize validation_period = k->vision_restir_gi_validation_period;
            bool is_validation_frame = validation_period > 0 &&
                                       (k->getDeviceStep() % validation_period == 0);
            bool is_sample_geometrically_valid =
                !is_validation_frame ||
                _validateReservoirGISample(
                    k, hit_position, normal, reservoir_prev, disocclusion_position_threshold
                );
            bool is_prev_valid =
                is_sample_geometrically_valid && reservoir_prev.sample_count > 0 &&
                reservoir_prev.object_key == object_key &&
                (Vec3C(reservoir_prev.prev_hit_position) - hit_position).length() <
                    disocclusion_position_threshold &&
                Vec3C(reservoir_prev.prev_normal).dot(normal) > normal_cos_threshold;

            if (is_prev_valid) {
                Vec3C bounce_dir_prev = Vec3C(reservoir_prev.bounce_direction);
                Vec3C l_incoming_prev = Vec3C(reservoir_prev.l_incoming.toVec3D());
                Vec3C brdf_cos_prev = _brdfCos(
                    material_type,
                    albedo,
                    metalness,
                    roughness,
                    PBR_DEFAULT_IOR,
                    normal,
                    view_dir,
                    bounce_dir_prev
                );
                Vec3C contribution_prev = brdf_cos_prev.elemMul(l_incoming_prev);
                FloatC target_pdf_prev_under_curr = _luminance(contribution_prev);

                // Re-score the prior sample under current shading and merge in.
                FloatC sample_weight_prev = target_pdf_prev_under_curr *
                                            FloatC(reservoir_prev.weight_sum) /
                                            (FloatC(reservoir_prev.target_pdf) + FloatC(RS_EPS));

                _reservoirUpdateGI(
                    sample_weight_prev,
                    reservoir_prev.bounce_direction,
                    reservoir_prev.bounce_hit_position,
                    reservoir_prev.bounce_hit_normal,
                    reservoir_prev.l_incoming,
                    target_pdf_prev_under_curr,
                    rs,
                    reservoir
                );
                // Combined sample count is this frame plus prior frame.
                reservoir.sample_count =
                    initial_candidate_count + RSize(reservoir_prev.sample_count);
                // Cap retained history; rescale weight to keep its mean.
                RSize m_cap =
                    RS_MAX(RSize(1), RSize(k->vision_restir_gi_m_cap) * initial_candidate_count);
                if (reservoir.sample_count > m_cap) {
                    FloatC scale = FloatC(m_cap) / FloatC(reservoir.sample_count);
                    reservoir.weight_sum = float(FloatC(reservoir.weight_sum) * scale);
                    reservoir.sample_count = m_cap;
                }
            }

            reservoir.object_key = object_key;
            reservoir.prev_hit_position =
                RVec3f(float(hit_position.x), float(hit_position.y), float(hit_position.z));
            reservoir.prev_normal = RVec3f(float(normal.x), float(normal.y), float(normal.z));
            // Pre-spatial output: contributing-source count equals sample count.
            reservoir.Z_count = reservoir.sample_count;
            ctx.camera_current_reservoir_gi[camera][{ii, jj}] = reservoir;
        }
    };

    /**
     * @brief Per-pixel biased spatial reuse pass over the ReSTIR GI reservoir.
     *
     * @details Reads our pixel's reservoir from `read_reservoir`, samples K
     *          neighbors inside `vision_restir_gi_spatial_radius`, gates them
     *          by object key + depth ratio + normal similarity, re-evaluates
     *          each surviving neighbor's stored bounce direction + l_incoming
     *          at our endpoint to compute a fresh `target_pdf`, and folds the
     *          neighbor's reservoir into a Bitterli WRS update with M-clamp
     *          decay. No new bounce rays are spawned: the neighbor's bounce
     *          evidence is replayed at our shading point. Visibility is
     *          inherited from the original closest-hit (biased Algorithm 6).
     *          Writes the merged reservoir to `write_reservoir`. Launched
     *          twice per frame between resample and the RGB pass; the
     *          executor flips `write_reservoir` and `read_reservoir` between
     *          iterations via host-side `exchange`.
     *
     * @param k                simulation kernel.
     * @param camera           target camera handle.
     * @param pixel            flat (i * W + j) pixel index.
     * @param read_reservoir   reservoir buffer the merge reads from.
     * @param write_reservoir  reservoir buffer the merge writes to.
     * @param iter_index       0-based pass index, used to differentiate the
     *                         RNG seed across passes so neighbors decorrelate.
     */
    class RSCF_SpatialReuseReservoirGI
        : public device::RS_ComputeFunction<RSCF_SpatialReuseReservoirGI> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;

            RSize image_width = CA_G(image_width);
            RSize image_height = CA_G(image_height);
            RIndex ii = RSize(pixel) / image_width;
            RIndex jj = RSize(pixel) - ii * image_width;

            auto object_key = ctx.camera_aux_endpoint_object_key[camera][{ii, jj}];
            if (object_key.isCosmicBackground()) {
                ctx.camera_spatial_reservoir_gi[camera][{ii, jj}] = RS_ReservoirGI{};
                return;
            }
            auto material_type = ctx.camera_aux_endpoint_material_type[camera][{ii, jj}];
            if (material_type != RSE_VISION_MATERIAL_OPAQUE) {
                ctx.camera_spatial_reservoir_gi[camera][{ii, jj}] = RS_ReservoirGI{};
                return;
            }

            Vec3C hit_position = Vec3C(ctx.camera_aux_endpoint_hit_position[camera][{ii, jj}]);
            Vec3C normal = Vec3C(ctx.camera_aux_endpoint_normal[camera][{ii, jj}]);
            Vec3C albedo = Vec3C(ctx.camera_aux_endpoint_albedo[camera][{ii, jj}]);
            FloatC roughness = FloatC(ctx.camera_aux_endpoint_roughness[camera][{ii, jj}]);
            FloatC metalness = FloatC(ctx.camera_aux_endpoint_metalness[camera][{ii, jj}]);
            FloatC depth_curr = FloatC(ctx.camera_aux_depth_z[camera][{ii, jj}]);
            Vec3C camera_position = Vec3C(CA_G(position));
            Vec3C view_dir = (camera_position - hit_position).normalized();

            // Seed from this frame's temporal output; neighbor reads use the
            // same buffer so the spatial pass never reuses its own output.
            RS_ReservoirGI reservoir = ctx.camera_current_reservoir_gi[camera][{ii, jj}];

            // RNG seeded with the iteration index per loop iteration instead
            // of the device step so each pixel picks the same neighbor set
            // every frame; reservoir converges instead of flickering.
            device::RS_RandomState rs;

            FloatC spatial_radius = FloatC(k->vision_restir_gi_spatial_radius);
            RSize neighbor_count = k->vision_restir_gi_spatial_neighbors;
            RSize history_ratio = RSize(k->vision_restir_gi_m_cap);
            RSize candidate_count = CA_G(samples_per_pixel);
            RSize sample_count_cap = history_ratio * candidate_count;
            FloatC NORMAL_COS_THRESHOLD = FloatC(k->vision_restir_disocclusion_normal_threshold);
            constexpr FloatC DEPTH_REL_THRESHOLD = FloatC(0.1);

            // Remember each accepted neighbor so the post-loop pass can
            // re-score whether its pixel could have produced the survivor.
            constexpr RSize MAX_VISITED_NEIGHBORS = 16;
            RIndex visited_ii[MAX_VISITED_NEIGHBORS];
            RIndex visited_jj[MAX_VISITED_NEIGHBORS];
            RSize visited_m[MAX_VISITED_NEIGHBORS];
            RSize visited_count = 0;
            RSize self_m = reservoir.sample_count;

            for (RSize n = 0; n < neighbor_count; ++n) {
                rs.init(
                    k->seed + uint64_t(0x6213),
                    static_cast<uint64_t>(pixel),
                    static_cast<uint64_t>(n)
                );
                FloatC rand_u = rs.template rand<FloatC>();
                FloatC rand_v = rs.template rand<FloatC>();
                FloatC dx = (rand_u * FloatC(2) - FloatC(1)) * spatial_radius;
                FloatC dy = (rand_v * FloatC(2) - FloatC(1)) * spatial_radius;
                int ii_n_signed = int(RS_FI::rs_round(FloatC(ii) + dy));
                int jj_n_signed = int(RS_FI::rs_round(FloatC(jj) + dx));
                ii_n_signed = RS_MAX(0, RS_MIN(ii_n_signed, int(image_height) - 1));
                jj_n_signed = RS_MAX(0, RS_MIN(jj_n_signed, int(image_width) - 1));
                RIndex ii_n = RIndex(ii_n_signed);
                RIndex jj_n = RIndex(jj_n_signed);
                if (ii_n == ii && jj_n == jj) continue;

                auto neighbor_key = ctx.camera_aux_endpoint_object_key[camera][{ii_n, jj_n}];
                if (neighbor_key.isCosmicBackground()) continue;
                if (neighbor_key != object_key) continue;

                FloatC depth_neighbor = FloatC(ctx.camera_aux_depth_z[camera][{ii_n, jj_n}]);
                FloatC depth_ratio_denom = RS_MAX(RS_ABS(depth_curr), FloatC(RS_EPS));
                if (RS_ABS(depth_neighbor - depth_curr) > DEPTH_REL_THRESHOLD * depth_ratio_denom) {
                    continue;
                }
                Vec3C normal_neighbor = Vec3C(ctx.camera_aux_endpoint_normal[camera][{ii_n, jj_n}]);
                if (normal_neighbor.dot(normal) < NORMAL_COS_THRESHOLD) continue;

                RS_ReservoirGI reservoir_neighbor =
                    ctx.camera_current_reservoir_gi[camera][{ii_n, jj_n}];
                if (reservoir_neighbor.sample_count == 0) continue;
                if (FloatC(reservoir_neighbor.target_pdf) <= FloatC(0)) continue;

                // Rebuild the bounce direction from this pixel to the
                // neighbor's stored sample point so the spatial shift in
                // origin is accounted for. Sky-miss samples have no point;
                // reuse the stored direction and shadow walk to infinity.
                Vec3C sample_point = Vec3C(reservoir_neighbor.bounce_hit_position);
                Vec3C bounce_dir;
                FloatC shadow_max_dist;
                bool is_sky_bounce = sample_point.x == FloatC(0) && sample_point.y == FloatC(0) &&
                                     sample_point.z == FloatC(0);
                if (is_sky_bounce) {
                    bounce_dir = Vec3C(reservoir_neighbor.bounce_direction);
                    shadow_max_dist = FloatC(RS_MAX_FLOAT);
                } else {
                    Vec3C diff = sample_point - hit_position;
                    FloatC dist = diff.length();
                    if (dist < FloatC(RS_EPS)) continue;
                    bounce_dir = diff * (FloatC(1) / dist);
                    shadow_max_dist = dist - FloatC(RS_EPS);
                }

                Vec3C l_incoming = Vec3C(reservoir_neighbor.l_incoming.toVec3D());
                Vec3C brdf_cos = _brdfCos(
                    material_type,
                    albedo,
                    metalness,
                    roughness,
                    PBR_DEFAULT_IOR,
                    normal,
                    view_dir,
                    bounce_dir
                );
                Vec3C contribution = brdf_cos.elemMul(l_incoming);
                FloatC target_pdf_at_curr = _luminance(contribution);
                if (target_pdf_at_curr <= FloatC(0)) continue;

                Vec3C shadow_origin = _offsetRayOrigin(hit_position, normal);
                RColor3f trans = _shadowWalkWithTransmission(
                    k, shadow_origin, bounce_dir, shadow_max_dist, RS_NULL_HANDLE
                );
                if (_luminance(Vec3C(trans.toVec3D())) <= FloatC(RS_EPS)) continue;

                // Correct the source-sampling density for the shift in
                // origin via the geometric Jacobian. Sky-miss samples are
                // direction-only, so the Jacobian is 1 and the step is skipped.
                // Reject grazing reconnections, extreme distance ratios, and
                // out-of-band Jacobians before applying the correction so
                // that bad candidates contribute nothing rather than spiking.
                if (!is_sky_bounce) {
                    Vec3C neighbor_hit_position =
                        Vec3C(ctx.camera_aux_endpoint_hit_position[camera][{ii_n, jj_n}]);
                    Vec3C bhn = Vec3C(reservoir_neighbor.bounce_hit_normal);
                    Vec3C from_neighbor = neighbor_hit_position - sample_point;
                    FloatC dist_neighbor = from_neighbor.length();
                    if (dist_neighbor < FloatC(RS_EPS)) continue;
                    FloatC cos_phi_neighbor =
                        RS_ABS(bhn.dot(from_neighbor * (FloatC(1) / dist_neighbor)));
                    if (cos_phi_neighbor < RESTIR_RECONNECT_COS_PHI_MIN) continue;

                    Vec3C from_curr = hit_position - sample_point;
                    FloatC dist_curr = from_curr.length();
                    if (dist_curr < FloatC(RS_EPS)) continue;
                    FloatC cos_phi_curr = RS_ABS(bhn.dot(from_curr * (FloatC(1) / dist_curr)));
                    if (cos_phi_curr < RESTIR_RECONNECT_COS_PHI_MIN) continue;

                    FloatC dist_ratio = dist_curr / dist_neighbor;
                    if (dist_ratio > RESTIR_RECONNECT_DIST_RATIO_MAX) continue;
                    if (dist_ratio < FloatC(1) / RESTIR_RECONNECT_DIST_RATIO_MAX) continue;

                    FloatC jacobian = (cos_phi_curr / cos_phi_neighbor) *
                                      (dist_neighbor * dist_neighbor) / (dist_curr * dist_curr);
                    if (jacobian < RESTIR_RECONNECT_JACOBIAN_MIN) continue;
                    if (jacobian > RESTIR_RECONNECT_JACOBIAN_MAX) continue;

                    target_pdf_at_curr = target_pdf_at_curr / jacobian;
                }

                // Re-score the neighbor's sample under our shading and merge.
                FloatC sample_weight = target_pdf_at_curr * FloatC(reservoir_neighbor.weight_sum) /
                                       (FloatC(reservoir_neighbor.target_pdf) + FloatC(RS_EPS));
                // Store the recomputed direction so later reads of this
                // reservoir see a direction consistent with the sample point.
                RVec3f bounce_dir_storage(
                    float(bounce_dir.x), float(bounce_dir.y), float(bounce_dir.z)
                );
                _reservoirUpdateGI(
                    sample_weight,
                    bounce_dir_storage,
                    reservoir_neighbor.bounce_hit_position,
                    reservoir_neighbor.bounce_hit_normal,
                    reservoir_neighbor.l_incoming,
                    target_pdf_at_curr,
                    rs,
                    reservoir
                );
                reservoir.sample_count =
                    reservoir.sample_count - 1 + RSize(reservoir_neighbor.sample_count);
                if (visited_count < MAX_VISITED_NEIGHBORS) {
                    visited_ii[visited_count] = ii_n;
                    visited_jj[visited_count] = jj_n;
                    visited_m[visited_count] = reservoir_neighbor.sample_count;
                    ++visited_count;
                }
            }

            reservoir.object_key = object_key;
            reservoir.prev_hit_position =
                RVec3f(float(hit_position.x), float(hit_position.y), float(hit_position.z));
            reservoir.prev_normal = RVec3f(float(normal.x), float(normal.y), float(normal.z));

            // Cap accumulated samples; rescale weight to keep its mean.
            RSize m_cap_spatial = RS_MAX(RSize(1), sample_count_cap * (neighbor_count + RSize(1)));
            if (reservoir.sample_count > m_cap_spatial) {
                FloatC scale = FloatC(m_cap_spatial) / FloatC(reservoir.sample_count);
                reservoir.weight_sum = float(FloatC(reservoir.weight_sum) * scale);
                reservoir.sample_count = m_cap_spatial;
            }

            // Sum the sample counts of every source pixel whose own bounce
            // hemisphere contains the survivor direction, including self.
            // This is the normalizer for the unbiased reservoir estimator.
            RSize Z = (self_m > 0) ? self_m : RSize(0);
            Vec3C survivor_sample_point = Vec3C(reservoir.bounce_hit_position);
            Vec3C survivor_bounce_dir = Vec3C(reservoir.bounce_direction);
            bool survivor_is_sky = survivor_sample_point.x == FloatC(0) &&
                                   survivor_sample_point.y == FloatC(0) &&
                                   survivor_sample_point.z == FloatC(0);
            for (RSize v = 0; v < visited_count; ++v) {
                Vec3C n_normal =
                    Vec3C(ctx.camera_aux_endpoint_normal[camera][{visited_ii[v], visited_jj[v]}]);
                Vec3C dir_at_n;
                if (survivor_is_sky) {
                    dir_at_n = survivor_bounce_dir;
                } else {
                    Vec3C n_pos = Vec3C(
                        ctx.camera_aux_endpoint_hit_position[camera][{visited_ii[v], visited_jj[v]}]
                    );
                    Vec3C diff = survivor_sample_point - n_pos;
                    FloatC dist = diff.length();
                    if (dist < FloatC(RS_EPS)) continue;
                    dir_at_n = diff * (FloatC(1) / dist);
                }
                if (n_normal.dot(dir_at_n) > FloatC(0)) Z += visited_m[v];
            }
            reservoir.Z_count = Z;
            ctx.camera_spatial_reservoir_gi[camera][{ii, jj}] = reservoir;
        }
    };

   private:
    /*****************************************************************
     * Camera / ray-gen math
     *****************************************************************/

    /**
     * @brief Halton low-discrepancy sample for sub-pixel jitter + defocus disk seeds.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _haltonSample(RIndex i, RIndex base) {
        FloatC f = FloatC(1);
        FloatC r = FloatC(0);
        RIndex idx = i + RIndex(1);
        while (idx > 0) {
            f = f / FloatC(base);
            r = r + f * FloatC(idx % base);
            idx = idx / base;
        }
        return r;
    }

    /**
     * @brief Polar-warp two uniforms to a point in the unit disk.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec2C _generateUnitDiskPoint(FloatC u, FloatC v) {
        FloatC r = RS_FI::rs_sqrt(u);
        FloatC theta = RS_2PI * v;
        FloatC sin_theta, cos_theta;
        RS_FI::rs_sincos(theta, &sin_theta, &cos_theta);
        return Vec2C(r * cos_theta, r * sin_theta);
    }

    /**
     * @brief Build the deterministic info ray for (camera, pixel).
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _initCameraInfoRay(
        Kernel* k,
        RHandle camera,
        RHandle pixel,
        RSize image_width,
        Vec3C& ray_origin_out,
        Vec3C& ray_direction_out
    ) {
        auto& ctx = k->ctx;
        RSize image_height = CA_G(image_height);

        Vec3C center = CA_G(position);
        Quat3C orientation = CA_G(orientation);
        Vec3C unit_lookat = orientation.toOrthoAxisX();
        Vec3C unit_up = orientation.toOrthoAxisZ();
        Vec3C unit_right = -orientation.toOrthoAxisY();

        RIndex i = RSize(pixel) / image_width;
        RIndex j = RSize(pixel) - i * image_width;

        RS_CameraStateUnion state_union{CA_G(state)};
        if (state_union.camera_state.type == RSE_CAMERA_FREE_STEREO ||
            state_union.camera_state.type == RSE_CAMERA_RIGID_BODY_LINKED_STEREO) {
            FloatC tan_left = RS_FI::rs_tan(FloatC(CA_G(angle_left)));
            FloatC tan_right = RS_FI::rs_tan(FloatC(CA_G(angle_right)));
            FloatC tan_up = RS_FI::rs_tan(FloatC(CA_G(angle_up)));
            FloatC tan_down = RS_FI::rs_tan(FloatC(CA_G(angle_down)));
            FloatC pixel_delta_u =
                (static_cast<FloatC>(j) + FloatC(0.5)) / static_cast<FloatC>(image_width);
            FloatC pixel_delta_v =
                (static_cast<FloatC>(i) + FloatC(0.5)) / static_cast<FloatC>(image_height);
            FloatC x_tan = tan_left + pixel_delta_u * (tan_right - tan_left);
            FloatC y_tan = tan_up + pixel_delta_v * (tan_down - tan_up);
            ray_origin_out = center;
            ray_direction_out = (unit_lookat + unit_right * x_tan + unit_up * y_tan).normalized();
        } else {
            Vec3C pixel_delta_u = unit_right * CA_G(pixel_width);
            Vec3C pixel_delta_v = -unit_up * CA_G(pixel_height);
            Vec3C viewport_upper_left =
                center + (CA_G(focus_distance) * unit_lookat) -
                ((FloatC(0.5) * static_cast<FloatC>(image_width)) * pixel_delta_u) -
                ((FloatC(0.5) * static_cast<FloatC>(image_height)) * pixel_delta_v);
            Vec3C pixel_00_loc =
                viewport_upper_left + (FloatC(0.5) * pixel_delta_u) + (FloatC(0.5) * pixel_delta_v);
            Vec3C pixel_target = pixel_00_loc + pixel_delta_u * static_cast<FloatC>(j) +
                                 pixel_delta_v * static_cast<FloatC>(i);
            ray_origin_out = center;
            ray_direction_out = (pixel_target - center).normalized();
        }
    }

    /**
     * @brief Build a Halton-jittered camera ray with defocus-disk origin offset.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _initCameraRGBRay(
        Kernel* k,
        RHandle camera,
        RHandle ray_id,
        RSize image_width,
        RSize image_height,
        RSize samples_per_pixel,
        Vec3C& ray_origin_out,
        Vec3C& ray_direction_out,
        RIndex& pixel_id_out
    ) {
        auto& ctx = k->ctx;

        Vec3C center = CA_G(position);
        Quat3C orientation = CA_G(orientation);
        Vec3C unit_lookat = orientation.toOrthoAxisX();
        Vec3C unit_up = orientation.toOrthoAxisZ();
        Vec3C unit_right = -orientation.toOrthoAxisY();

        pixel_id_out = RIndex(ray_id) / samples_per_pixel;
        RIndex i = pixel_id_out / image_width;
        RIndex j = pixel_id_out - i * image_width;
        RIndex local_ray_id = RIndex(ray_id) - pixel_id_out * samples_per_pixel;

        FloatC offset_x = _haltonSample(local_ray_id, 2) - FloatC(0.5);
        FloatC offset_y = _haltonSample(local_ray_id, 3) - FloatC(0.5);

        RS_CameraStateUnion state_union{CA_G(state)};
        if (state_union.camera_state.type == RSE_CAMERA_FREE_STEREO ||
            state_union.camera_state.type == RSE_CAMERA_RIGID_BODY_LINKED_STEREO) {
            FloatC tan_left = RS_FI::rs_tan(FloatC(CA_G(angle_left)));
            FloatC tan_right = RS_FI::rs_tan(FloatC(CA_G(angle_right)));
            FloatC tan_up = RS_FI::rs_tan(FloatC(CA_G(angle_up)));
            FloatC tan_down = RS_FI::rs_tan(FloatC(CA_G(angle_down)));
            FloatC pixel_delta_u = (static_cast<FloatC>(j) + FloatC(0.5) + offset_x) /
                                   static_cast<FloatC>(image_width);
            FloatC pixel_delta_v = (static_cast<FloatC>(i) + FloatC(0.5) + offset_y) /
                                   static_cast<FloatC>(image_height);
            FloatC x_tan = tan_left + pixel_delta_u * (tan_right - tan_left);
            FloatC y_tan = tan_up + pixel_delta_v * (tan_down - tan_up);
            ray_origin_out = center;
            ray_direction_out = (unit_lookat + unit_right * x_tan + unit_up * y_tan).normalized();
        } else {
            FloatC defocus_radius = CA_G(defocus_radius);
            Vec3C pixel_delta_u = unit_right * CA_G(pixel_width);
            Vec3C pixel_delta_v = -unit_up * CA_G(pixel_height);
            Vec3C viewport_upper_left =
                center + (CA_G(focus_distance) * unit_lookat) -
                ((FloatC(0.5) * static_cast<FloatC>(image_width)) * pixel_delta_u) -
                ((FloatC(0.5) * static_cast<FloatC>(image_height)) * pixel_delta_v);
            Vec3C pixel_00_loc =
                viewport_upper_left + (FloatC(0.5) * pixel_delta_u) + (FloatC(0.5) * pixel_delta_v);
            Vec3C defocus_disk_u = unit_right * defocus_radius;
            Vec3C defocus_disk_v = unit_up * defocus_radius;
            Vec3C pixel_target = pixel_00_loc + (pixel_delta_u * (FloatC(j) + offset_x)) +
                                 (pixel_delta_v * (FloatC(i) + offset_y));

            Vec2C disk = _generateUnitDiskPoint(
                _haltonSample(local_ray_id, 5), _haltonSample(local_ray_id, 7)
            );
            ray_origin_out = center + (disk.x * defocus_disk_u) + (disk.y * defocus_disk_v);
            ray_direction_out = (pixel_target - ray_origin_out).normalized();
        }
    }

    /*****************************************************************
     * BVH / intersection
     *****************************************************************/

    /**
     * @brief Slab AABB / ray intersection. Returns true on hit, fills t_enter and t_exit.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _isRayIntersectingAABB(
        const Ray& ray,
        const Vec3C& dir_reciprocal,
        const Vec3C& aabb_min,
        const Vec3C& aabb_max,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_enter_out,
        FloatC& t_exit_out
    ) {
        Vec3C t0 = (aabb_min - ray.ray_origin).elemMul(dir_reciprocal);
        Vec3C t1 = (aabb_max - ray.ray_origin).elemMul(dir_reciprocal);
        Vec3C t_min_vec = t0.min(t1);
        Vec3C t_max_vec = t0.max(t1);
        t_enter_out = t_min_vec.max();
        t_exit_out = t_max_vec.min();
        return t_exit_out >= t_enter_out && t_exit_out >= t_min && t_enter_out <= t_max;
    }

    /**
     * @brief Nearest crossing of a ray with one deformed quad, tested as two triangles.
     *
     * @details The corners arrive in the winding _voxelFaceCornerIndices produces, so the cross
     * product of the first and last edge is the outward geometric normal. A deformed quad need
     * not be planar, so both triangles are tested and the nearer crossing wins.
     *
     * @param ray        the ray being traced.
     * @param corner_0   first quad corner in winding order.
     * @param corner_1   second quad corner.
     * @param corner_2   third quad corner, opposite the first.
     * @param corner_3   fourth quad corner.
     * @param t_min    nearest accepted distance along the ray.
     * @param t_max    furthest accepted distance along the ray.
     * @param t_out    distance to the crossing.
     * @param uv_out   where the crossing sits on the quad, 0 at corner_0 and 1 at the far edge.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _isRayIntersectingQuad(
        const Ray& ray,
        const Vec3C& corner_0,
        const Vec3C& corner_1,
        const Vec3C& corner_2,
        const Vec3C& corner_3,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        Vec2C& uv_out
    ) {
        // A quad point is corner_0 + u * (corner_1 - corner_0) + v * (corner_3 - corner_0), and
        // corner_2 is the far corner, so each triangle's barycentrics fold into that pair.
        FloatC u, v, t_cross;
        FloatC t_near = t_max;
        bool has_hit = false;
        if (RS_Triangle<FloatC>::rayIntersect(
                ray.ray_origin,
                ray.ray_direction,
                corner_0,
                corner_1,
                corner_2,
                t_min,
                t_near,
                t_cross,
                u,
                v
            )) {
            t_near = t_cross;
            uv_out = Vec2C(u + v, v);
            has_hit = true;
        }
        if (RS_Triangle<FloatC>::rayIntersect(
                ray.ray_origin,
                ray.ray_direction,
                corner_0,
                corner_2,
                corner_3,
                t_min,
                t_near,
                t_cross,
                u,
                v
            )) {
            t_near = t_cross;
            uv_out = Vec2C(u, u + v);
            has_hit = true;
        }
        if (!has_hit) return false;
        t_out = t_near;
        return true;
    }

    /**
     * @brief Slab test of a ray already in a box's own frame, the box centered on the origin.
     *
     * @details Reports both crossings and the axis each one lies on, so the caller can turn
     * them into faces. The ray starts within the box when the entry distance falls short of
     * t_min.
     *
     * @param local_origin    ray origin in the box frame.
     * @param local_direction ray direction in the box frame.
     * @param half_size       half the box edge length.
     * @param t_min           nearest accepted distance along the ray.
     * @param t_enter_out     distance to the crossing where the ray goes in.
     * @param enter_axis_out  axis that crossing lies on, 0 for x, 1 for y and 2 for z.
     * @param t_exit_out      distance to the crossing where the ray comes out.
     * @param exit_axis_out   axis that crossing lies on.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _isRayIntersectingBox(
        const Vec3C& local_origin,
        const Vec3C& local_direction,
        FloatC half_size,
        FloatC t_min,
        FloatC& t_enter_out,
        int& enter_axis_out,
        FloatC& t_exit_out,
        int& exit_axis_out
    ) {
        Vec3C direction_reciprocal = local_direction.reciprocalFast();
        Vec3C t0 = (Vec3C::full(-half_size) - local_origin).elemMul(direction_reciprocal);
        Vec3C t1 = (Vec3C::full(half_size) - local_origin).elemMul(direction_reciprocal);
        Vec3C t_min_vec = t0.min(t1);
        Vec3C t_max_vec = t0.max(t1);
        t_enter_out = t_min_vec.max();
        t_exit_out = t_max_vec.min();
        if (t_exit_out < t_enter_out || t_exit_out < t_min) return false;
        enter_axis_out = t_min_vec.argmax();
        exit_axis_out = t_max_vec.argmin();
        return true;
    }

    /**
     * @brief First real boundary face along the ray inside a voxel hexahedron.
     *
     * @details The leaf carries eight corners that deformation can pull out of any box shape,
     * so each face is crossed as a quad rather than a slab. Only faces the roles expose are
     * tested, which is what makes this cheap: an interior face can never carry a boundary, an
     * entry is reported for any exposed face, and an exit only where the face meets air. The
     * nearest qualifying crossing wins, which for a convex voxel is the entry when the ray
     * starts outside and the exit when it starts within.
     *
     * @param leaf         ray-BVH leaf holding the corners, roles and face normals.
     * @param ray          the ray being traced.
     * @param t_min        nearest accepted distance along the ray.
     * @param t_max        furthest accepted distance along the ray.
     * @param t_out        distance to the crossing.
     * @param face_out     crossed face index in RSE_LinkDirection order.
     * @param is_front_out whether the ray meets the face from outside the voxel.
     * @param role_out     role of the crossed face.
     * @param normal_out   outward shading normal of the crossed face, unit length.
     * @param uv_out       where the crossing sits on the face, 0 to 1 along each face edge.
     * @param face_pos_out that same place in meters, measured from the face's first corner.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _voxelHexahedronFirstBoundary(
        RState metadata,
        const RS_VoxelRayBVHObjectIdentifier& leaf,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        int& face_out,
        bool& is_front_out,
        RSE_VoxelFaceRole& role_out,
        Vec3C& normal_out,
        Vec2C& uv_out,
        Vec2C& face_pos_out
    ) {
        Vec3C center = Vec3C(leaf.position);

        FloatC t_near = t_max;
        bool has_hit = false;
        Vec3C best_normal, best_edge_u, best_edge_v;
        for (int face = 0; face < 6; face++) {
            RSE_VoxelFaceRole role = getRayLeafFaceRole(metadata, face);
            if (role == RSE_VOXEL_FACE_INTERIOR) continue;

            int index_0, index_1, index_2, index_3;
            getVoxelFaceCornerIndices(
                face >> 1, (face & 1) == 0, index_0, index_1, index_2, index_3
            );
            const auto& corner = leaf.shape.hexahedron.corner_offset;
            Vec3C corner_0 = center + corner[index_0].template toVec3D<FloatC>();
            Vec3C corner_1 = center + corner[index_1].template toVec3D<FloatC>();
            Vec3C corner_2 = center + corner[index_2].template toVec3D<FloatC>();
            Vec3C corner_3 = center + corner[index_3].template toVec3D<FloatC>();

            // The diagonals give the normal of the whole face, so either triangle of a face
            // bent out of plane reads the same way.
            Vec3C normal = (corner_2 - corner_0).cross(corner_3 - corner_1);
            bool is_front = ray.ray_direction.dot(normal) < FloatC(0);

            // One interface belongs to two voxels, and the voxel being entered reports it, so
            // a face met from within is skipped unless it opens onto air, where there is no
            // other voxel to report it.
            if (!is_front && role != RSE_VOXEL_FACE_AIR) continue;

            FloatC t_cross;
            Vec2C uv;
            if (!_isRayIntersectingQuad(
                    ray, corner_0, corner_1, corner_2, corner_3, t_min, t_near, t_cross, uv
                ))
                continue;

            t_near = t_cross;
            face_out = face;
            role_out = role;
            is_front_out = is_front;
            uv_out = uv;
            best_normal = normal;
            best_edge_u = corner_1 - corner_0;
            best_edge_v = corner_3 - corner_0;
            has_hit = true;
        }
        if (!has_hit) return false;

        // Only the winning face pays for a root.
        t_out = t_near;
        normal_out = best_normal.normalized();
        face_pos_out = Vec2C(uv_out.x * best_edge_u.length(), uv_out.y * best_edge_v.length());
        return true;
    }

    /**
     * @brief First real boundary face along the ray inside a voxel box.
     *
     * @details A non-deformable voxel keeps its box, so one slab test covers all six faces and
     * only the two it crosses need a role. The entry face comes first when the ray starts
     * outside, then the exit face, and the first that is a real boundary wins.
     *
     * @param leaf         ray-BVH leaf holding the box and the roles.
     * @param ray          the ray being traced.
     * @param t_min        nearest accepted distance along the ray.
     * @param t_max        furthest accepted distance along the ray.
     * @param t_out        distance to the crossing.
     * @param face_out     crossed face index in RSE_LinkDirection order.
     * @param is_front_out whether the ray meets the face from outside the voxel.
     * @param role_out     role of the crossed face.
     * @param normal_out   outward normal of the crossed face, unit length.
     * @param uv_out       where the crossing sits on the face, 0 to 1 along each face edge.
     * @param face_pos_out that same place in meters, measured from the face's first corner.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _voxelBoxFirstBoundary(
        RState metadata,
        const RS_VoxelRayBVHObjectIdentifier& leaf,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        int& face_out,
        bool& is_front_out,
        RSE_VoxelFaceRole& role_out,
        Vec3C& normal_out,
        Vec2C& uv_out,
        Vec2C& face_pos_out
    ) {
        Quat3C orientation = Quat3C(leaf.shape.box.orientation);
        FloatC size = FloatC(leaf.shape.box.size);

        FloatC half_size = FloatC(0.5) * size;
        Vec3C local_origin = orientation.rotateVec3DInv(ray.ray_origin - Vec3C(leaf.position));
        Vec3C local_direction = orientation.rotateVec3DInv(ray.ray_direction);

        FloatC t_enter, t_exit;
        int enter_axis, exit_axis;
        if (!_isRayIntersectingBox(
                local_origin,
                local_direction,
                half_size,
                t_min,
                t_enter,
                enter_axis,
                t_exit,
                exit_axis
            ))
            return false;

        for (int pass = 0; pass < 2; ++pass) {
            bool is_entry = pass == 0;
            if (is_entry && t_enter < t_min) continue;
            FloatC t_cross = is_entry ? t_enter : t_exit;
            if (t_cross > t_max) return false;

            int axis = is_entry ? enter_axis : exit_axis;
            Vec3C local_hit = local_origin + local_direction * t_cross;
            bool is_positive = local_hit[axis] > FloatC(0);
            int face = 2 * axis + (is_positive ? 0 : 1);
            RSE_VoxelFaceRole role = getRayLeafFaceRole(metadata, face);
            if (role == RSE_VOXEL_FACE_INTERIOR) continue;

            // One interface belongs to two voxels, and the voxel being entered reports it, so
            // a face met from within is skipped unless it opens onto air, where there is no
            // other voxel to report it.
            if (!is_entry && role != RSE_VOXEL_FACE_AIR) continue;

            // Same face edge order the corner indices give, so the two shapes agree on uv.
            int u_axis = (axis + 1) % 3;
            int v_axis = (axis + 2) % 3;
            if (!is_positive) {
                int swap = u_axis;
                u_axis = v_axis;
                v_axis = swap;
            }
            FloatC side = is_positive ? FloatC(1) : FloatC(-1);
            Vec3C local_normal(
                axis == 0 ? side : FloatC(0),
                axis == 1 ? side : FloatC(0),
                axis == 2 ? side : FloatC(0)
            );

            t_out = t_cross;
            face_out = face;
            role_out = role;
            normal_out = orientation.rotateVec3D(local_normal);
            is_front_out = is_entry;
            uv_out = Vec2C(
                (local_hit[u_axis] + half_size) / size, (local_hit[v_axis] + half_size) / size
            );
            face_pos_out = uv_out * size;
            return true;
        }
        return false;
    }

    /** @brief First real boundary face along the ray, taking whichever shape the leaf holds. */
    HOST_DEVICE_INLINE_QUALIFIER static bool _voxelFirstBoundary(
        const RS_VoxelRayBVHObjectIdentifier& leaf,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        int& face_out,
        bool& is_front_out,
        RSE_VoxelFaceRole& role_out,
        Vec3C& normal_out,
        Vec2C& uv_out,
        Vec2C& face_pos_out
    ) {
        RState metadata = leaf.object_metadata;
        RS_VoxelRayBVHLeafMetadataUnion m_union{metadata};
        if (m_union.meta.is_box)
            return _voxelBoxFirstBoundary(
                metadata,
                leaf,
                ray,
                t_min,
                t_max,
                t_out,
                face_out,
                is_front_out,
                role_out,
                normal_out,
                uv_out,
                face_pos_out
            );
        return _voxelHexahedronFirstBoundary(
            metadata,
            leaf,
            ray,
            t_min,
            t_max,
            t_out,
            face_out,
            is_front_out,
            role_out,
            normal_out,
            uv_out,
            face_pos_out
        );
    }

    /**
     * @brief Closest-hit voxel BVH walk: test-before-push, closer-first ordered descent.
     *
     * @details The node being visited is carried in `idx` rather than being pushed and
     * popped again. A node whose children both survive the test descends into the nearer
     * one and pushes only the farther one; a node with one surviving child descends
     * straight into it and touches the stack not at all. The stack is thread local memory
     * whatever the register budget, because a run time index cannot address a register, so
     * every push and pop that goes away is a load or a store removed from the walk.
     */
    template <typename Kernel, typename BVHAccessor>
    HOST_DEVICE_INLINE_QUALIFIER static void _findHitVoxelInBVH(
        Kernel* k,
        BVHAccessor root,
        const Ray& ray,
        FloatC t_min,
        FloatC& t_closest_in_out,
        HitRecord& rec_in_out
    ) {
        if (root.current() == RS_NULL_INDEX) return;
        // Every bound in this walk is tested against the same ray, so the slab test's
        // reciprocal is formed once here rather than three times per node visited.
        Vec3C dir_reciprocal = ray.ray_direction.reciprocalFast();
        RIndex stack[RS_RUNTIME_BVH_STACK_SIZE];
        int stack_top = 0;
        RIndex idx = root.current();
        while (true) {
            if (idx == RS_NULL_INDEX) {
                if (stack_top == 0) break;
                idx = stack[--stack_top];
                continue;
            }
            auto node = root;
            node.moveTo(idx);
            // Nothing left to descend into unless a child survives the test below.
            idx = RS_NULL_INDEX;
            if (node.isLeaf()) {
                FloatC t_enter, t_exit;
                auto bnd = node.bound();
                if (!_isRayIntersectingAABB(
                        ray,
                        dir_reciprocal,
                        Vec3C(bnd.minBound()),
                        Vec3C(bnd.maxBound()),
                        t_min,
                        t_closest_in_out,
                        t_enter,
                        t_exit
                    ))
                    continue;
                auto obj_ptr = node.object();
                if (obj_ptr == nullptr) continue;
                FloatC t_hit;
                int face;
                bool is_front;
                RSE_VoxelFaceRole role;
                Vec3C normal;
                Vec2C uv, face_pos;
                if (!_voxelFirstBoundary(
                        *obj_ptr,
                        ray,
                        t_min,
                        t_closest_in_out,
                        t_hit,
                        face,
                        is_front,
                        role,
                        normal,
                        uv,
                        face_pos
                    ))
                    continue;
                t_closest_in_out = t_hit;
                rec_in_out.t = t_hit;
                rec_in_out.normal = is_front ? normal : -normal;
                rec_in_out.uv = uv;
                rec_in_out.face_pos = face_pos;
                rec_in_out.voxel = obj_ptr->object_handle;
                rec_in_out.vision_material = obj_ptr->vision_material;
                rec_in_out.structure_gid = obj_ptr->structure_gid;
                rec_in_out.state.hit_state.has_face_normal =
                    RS_VoxelRayBVHLeafMetadataUnion{obj_ptr->object_metadata}
                        .meta.is_smooth_surface;
                rec_in_out.state.hit_state.face = face;
                rec_in_out.state.hit_state.face_role = role;
                rec_in_out.state.hit_state.is_front_face = is_front;
                continue;
            }

            auto internal_node = node.internalNode();
            FloatC t_left_enter, t_right_enter, t_exit_unused;
            bool is_left_hit = internal_node.left != RS_NULL_INDEX &&
                               _isRayIntersectingAABB(
                                   ray,
                                   dir_reciprocal,
                                   Vec3C(internal_node.left_bound.minBound()),
                                   Vec3C(internal_node.left_bound.maxBound()),
                                   t_min,
                                   t_closest_in_out,
                                   t_left_enter,
                                   t_exit_unused
                               );
            bool is_right_hit = internal_node.right != RS_NULL_INDEX &&
                                _isRayIntersectingAABB(
                                    ray,
                                    dir_reciprocal,
                                    Vec3C(internal_node.right_bound.minBound()),
                                    Vec3C(internal_node.right_bound.maxBound()),
                                    t_min,
                                    t_closest_in_out,
                                    t_right_enter,
                                    t_exit_unused
                                );
            if (is_left_hit && is_right_hit) {
                bool is_left_nearer = t_left_enter < t_right_enter;
                idx = is_left_nearer ? internal_node.left : internal_node.right;
                RIndex far_node = is_left_nearer ? internal_node.right : internal_node.left;
                if (stack_top < RS_RUNTIME_BVH_STACK_SIZE) stack[stack_top++] = far_node;
            } else if (is_left_hit) {
                idx = internal_node.left;
            } else if (is_right_hit) {
                idx = internal_node.right;
            }
        }
    }

    /**
     * @brief Closest-hit across the static + dynamic voxel BVHs.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool
    _findHitVoxel(Kernel* k, const Ray& ray, FloatC t_min, FloatC t_max, HitRecord& rec_in_out) {
        auto& ctx = k->ctx;
        FloatC t_closest = t_max;
        rec_in_out.voxel = RS_NULL_HANDLE;
        _findHitVoxelInBVH(k, ctx.static_voxel_ray_bvh.root(), ray, t_min, t_closest, rec_in_out);
        _findHitVoxelInBVH(k, ctx.dynamic_voxel_ray_bvh.root(), ray, t_min, t_closest, rec_in_out);
        return rec_in_out.voxel != RS_NULL_HANDLE;
    }

    /**
     * @brief Floor triangle narrow-phase: Moller-Trumbore via RS_Triangle.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _isRayIntersectingFloorTriangle(
        Kernel* k,
        const Ray& ray,
        RHandle triangle,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        Vec3C& normal_out
    ) {
        auto& ctx = k->ctx;
        auto& tri = ctx.mesh_triangles.get({triangle});
        Vec3C v0(tri.vertex_0);
        Vec3C v1(tri.vertex_1);
        Vec3C v2(tri.vertex_2);
        FloatC u, v;
        if (!RS_Triangle<FloatC>::rayIntersect(
                ray.ray_origin, ray.ray_direction, v0, v1, v2, t_min, t_max, t_out, u, v
            ))
            return false;
        normal_out = Vec3C(tri.face_normal);
        return true;
    }

    /**
     * @brief Closest-hit floor triangle BVH walk.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _findHitFloorTriangle(
        Kernel* k,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        RHandle& triangle_out,
        FloatC& t_out,
        Vec3C& normal_out
    ) {
        auto& ctx = k->ctx;
        auto root = ctx.floor_triangle_bvh.root();
        if (root.current() == RS_NULL_INDEX) return false;

        FloatC t_closest = t_max;
        bool found = false;
        // Every bound in this walk is tested against the same ray, so the slab test's
        // reciprocal is formed once here rather than three times per node visited.
        Vec3C dir_reciprocal = ray.ray_direction.reciprocalFast();
        RIndex stack[RS_RUNTIME_BVH_STACK_SIZE];
        int stack_top = 0;
        RIndex idx = root.current();
        while (true) {
            if (idx == RS_NULL_INDEX) {
                if (stack_top == 0) break;
                idx = stack[--stack_top];
                continue;
            }
            auto node = root;
            node.moveTo(idx);
            // Nothing left to descend into unless a child survives the test below.
            idx = RS_NULL_INDEX;
            FloatC t_enter, t_exit;
            auto bnd = node.bound();
            if (!_isRayIntersectingAABB(
                    ray,
                    dir_reciprocal,
                    Vec3C(bnd.minBound()),
                    Vec3C(bnd.maxBound()),
                    t_min,
                    t_closest,
                    t_enter,
                    t_exit
                ))
                continue;
            if (node.isLeaf()) {
                auto obj_ptr = node.object();
                if (obj_ptr == nullptr) continue;
                RHandle tri = obj_ptr->object_handle;
                FloatC t_hit;
                Vec3C tri_normal;
                if (_isRayIntersectingFloorTriangle(
                        k, ray, tri, t_min, t_closest, t_hit, tri_normal
                    ) &&
                    t_hit < t_closest) {
                    t_closest = t_hit;
                    triangle_out = tri;
                    normal_out = tri_normal;
                    found = true;
                }
                continue;
            }
            RIndex l = node.left();
            RIndex r = node.right();
            auto ln = root;
            ln.moveTo(l);
            auto rn = root;
            rn.moveTo(r);
            FloatC t_left_enter, t_right_enter, t_exit_unused;
            bool is_left_hit = ln.current() != RS_NULL_INDEX && _isRayIntersectingAABB(
                                                                    ray,
                                                                    dir_reciprocal,
                                                                    Vec3C(ln.bound().minBound()),
                                                                    Vec3C(ln.bound().maxBound()),
                                                                    t_min,
                                                                    t_closest,
                                                                    t_left_enter,
                                                                    t_exit_unused
                                                                );
            bool is_right_hit = rn.current() != RS_NULL_INDEX && _isRayIntersectingAABB(
                                                                     ray,
                                                                     dir_reciprocal,
                                                                     Vec3C(rn.bound().minBound()),
                                                                     Vec3C(rn.bound().maxBound()),
                                                                     t_min,
                                                                     t_closest,
                                                                     t_right_enter,
                                                                     t_exit_unused
                                                                 );
            if (is_left_hit && is_right_hit) {
                bool is_left_nearer = t_left_enter < t_right_enter;
                idx = is_left_nearer ? l : r;
                RIndex far_node = is_left_nearer ? r : l;
                if (stack_top < RS_RUNTIME_BVH_STACK_SIZE) stack[stack_top++] = far_node;
            } else if (is_left_hit) {
                idx = l;
            } else if (is_right_hit) {
                idx = r;
            }
        }
        t_out = t_closest;
        return found;
    }

    /**
     * @brief Closest-hit floor BVH walk for periodic ARRAY floors.
     *
     * @details Treats the heightfield as a periodic 3D tiling of canonical-tile-shaped
     * boxes `[(i-0.5)x_size, (i+0.5)x_size] x [(j-0.5)y_size, (j+0.5)y_size] x
     * [height_min, height_max]`. The ray is first clipped to the global z-slab
     * `[height_min, height_max]` (infinite in xy). Then a 2D DDA over tile indices
     * (i, j) advances along the ray; each tile shifts the ray into canonical
     * coordinates and runs a bounded `_findHitFloorTriangle` for the (t_curr, t_exit)
     * sub-range. The first hit wins.
     *
     * @details Non-repeating axes: when `is_x_repeat == false`, only `tile_i == 0`
     * runs the per-tile BVH query; other tiles are skipped (the floor doesn't extend
     * there). Same for y. With both axes off, this reduces to a single bounded BVH
     * call on canonical, matching the pre-§2g behavior bit-exactly.
     *
     * @note Translation in xy commutes with t along the ray, so the returned t_out is
     * the same in world space as in canonical space. World hit reconstruction at the
     * caller still uses the original (unshifted) ray.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _findHitFloorTriangleTiled(
        Kernel* k,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        RHandle& triangle_out,
        FloatC& t_out,
        Vec3C& normal_out
    ) {
        auto& f = k->ctx.floor;

        // Non-repeating in both axes: single canonical query, same as pre-§2g.
        if (!f.is_x_repeat && !f.is_y_repeat) {
            return _findHitFloorTriangle(k, ray, t_min, t_max, triangle_out, t_out, normal_out);
        }

        FloatC x_size = FloatC(f.x_size);
        FloatC y_size = FloatC(f.y_size);
        FloatC half_x = x_size * FloatC(0.5);
        FloatC half_y = y_size * FloatC(0.5);

        // Step 1: clip ray's parameter range to the finite-z slab. Pad by RS_EPS
        // on each end so triangles flush with height_min / height_max are not
        // clipped by float round-off in the slab plane intersection.
        FloatC z_lo = FloatC(f.height_min) - FloatC(RS_EPS);
        FloatC z_hi = FloatC(f.height_max) + FloatC(RS_EPS);
        FloatC t_slab_enter = t_min;
        FloatC t_slab_exit = t_max;
        FloatC dz = ray.ray_direction.z;
        if (RS_ABS(dz) > FloatC(RS_EPS)) {
            FloatC t_a = (z_lo - ray.ray_origin.z) / dz;
            FloatC t_b = (z_hi - ray.ray_origin.z) / dz;
            FloatC t_enter_z = RS_MIN(t_a, t_b);
            FloatC t_exit_z = RS_MAX(t_a, t_b);
            t_slab_enter = RS_MAX(t_slab_enter, t_enter_z);
            t_slab_exit = RS_MIN(t_slab_exit, t_exit_z);
        } else {
            // Horizontal ray: must already lie inside the z-slab.
            if (ray.ray_origin.z < z_lo || ray.ray_origin.z > z_hi) return false;
        }
        if (t_slab_enter >= t_slab_exit) return false;

        // Step 2: 2D DDA over tile (i, j) between t_slab_enter and t_slab_exit.
        // Note we are not walking on the array itself, we walk on the tile grid.
        FloatC t_curr = t_slab_enter;
        for (int iter = 0;
             iter < RS_WORLD_VISION_TRACE_MAX_ARRAY_FLOOR_TILE_NUM && t_curr < t_slab_exit;
             ++iter) {
            // Determine current tile from an interior point just past t_curr.
            FloatC t_probe = t_curr + FloatC(RS_EPS);
            Vec3C p = ray.ray_origin + ray.ray_direction * t_probe;
            FloatC fi = RS_FI::rs_floor((p.x + half_x) / x_size);
            FloatC fj = RS_FI::rs_floor((p.y + half_y) / y_size);

            // Find t_exit: when does the ray exit this tile's xy footprint?
            FloatC t_exit = t_slab_exit;
            if (RS_ABS(ray.ray_direction.x) > FloatC(RS_EPS)) {
                FloatC tile_x_far = (ray.ray_direction.x > FloatC(0)) ? (fi + FloatC(0.5)) * x_size
                                                                      : (fi - FloatC(0.5)) * x_size;
                FloatC t_xx = (tile_x_far - ray.ray_origin.x) / ray.ray_direction.x;
                if (t_xx < t_exit) t_exit = t_xx;
            }
            if (RS_ABS(ray.ray_direction.y) > FloatC(RS_EPS)) {
                FloatC tile_y_far = (ray.ray_direction.y > FloatC(0)) ? (fj + FloatC(0.5)) * y_size
                                                                      : (fj - FloatC(0.5)) * y_size;
                FloatC t_yy = (tile_y_far - ray.ray_origin.y) / ray.ray_direction.y;
                if (t_yy < t_exit) t_exit = t_yy;
            }

            // Valid tile? Non-repeating axes mean only tile 0 holds geometry.
            bool tile_valid = true;
            if (!f.is_x_repeat && fi != FloatC(0)) tile_valid = false;
            if (!f.is_y_repeat && fj != FloatC(0)) tile_valid = false;

            if (tile_valid && t_exit > t_curr) {
                // Shift ray xy into canonical and run a bounded BVH query.
                Ray shifted(
                    Vec3C(
                        ray.ray_origin.x - fi * x_size,
                        ray.ray_origin.y - fj * y_size,
                        ray.ray_origin.z
                    ),
                    ray.ray_direction
                );
                RHandle tri;
                FloatC t_hit;
                Vec3C n;
                if (_findHitFloorTriangle(k, shifted, t_curr, t_exit, tri, t_hit, n)) {
                    triangle_out = tri;
                    t_out = t_hit;
                    normal_out = n;
                    return true;
                }
            }

            // Advance to the neighboring tile.
            t_curr = t_exit + FloatC(RS_EPS);
        }
        return false;
    }

    /**
     * @brief Closest-hit aggregator over voxel BVHs + floor: writes the full HitRecord.
     *
     * @details Voxel walk runs unbounded; floor walk is bounded by the voxel t to skip
     * triangles already occluded. The winning hit populates rec.{p, normal, t,
     * is_front_face, is_floor_hit, voxel} and dispatches to the material reader.
     *
     * @param k       simulation kernel.
     * @param ray     camera ray in world coordinates.
     * @param rec_out output hit record; untouched on a miss.
     * @return true on hit, false on miss.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool
    _findHit(Kernel* k, const Ray& ray, HitRecord& rec_out) {
        // Voxel BVH closest hit. Interior faces are skipped inside the walk, so this
        // returns the nearest real boundary face the ray crosses, already resolved into rec_out.
        bool voxel_hit = _findHitVoxel(k, ray, RAY_T_MIN, RS_NumTraits<FloatC>::highest(), rec_out);
        FloatC t_voxel = voxel_hit ? rec_out.t : RS_NumTraits<FloatC>::highest();

        // Floor BVH closest hit, bounded by voxel t. Flat floor is the analytic z=0 plane.
        RHandle floor_triangle = RS_NULL_HANDLE;
        FloatC t_floor = voxel_hit ? t_voxel : RS_NumTraits<FloatC>::highest();
        Vec3C floor_normal(0, 0, 1);
        bool floor_hit = false;
        if (k->ctx.floor_vision_mem.data_type == RSE_VISION_FLOOR_DATA_ARRAY) {
            floor_hit = _findHitFloorTriangleTiled(
                k, ray, RAY_T_MIN, t_floor, floor_triangle, t_floor, floor_normal
            );
        } else if (k->ctx.floor_vision_mem.data_type == RSE_VISION_FLOOR_DATA_FLAT) {
            FloatC dz = ray.ray_direction.z;
            if (RS_ABS(dz) > FloatC(RS_EPS)) {
                FloatC t_ground = (FloatC(k->ctx.floor.height) - ray.ray_origin.z) / dz;
                if (t_ground > RAY_T_MIN && t_ground < t_floor) {
                    t_floor = t_ground;
                    floor_normal = Vec3C(0, 0, 1);
                    floor_hit = true;
                }
            }
        }

        if (!voxel_hit && !floor_hit) return false;

        bool floor_wins = floor_hit && (!voxel_hit || t_floor < t_voxel);
        if (floor_wins) {
            rec_out.state.hit_state.is_floor_hit = true;
            rec_out.voxel = RS_NULL_HANDLE;
            rec_out.t = t_floor;
            rec_out.p = ray.at(t_floor);
            bool is_front = ray.ray_direction.dot(floor_normal) < FloatC(0);
            rec_out.normal = is_front ? floor_normal : -floor_normal;
            rec_out.state.hit_state.is_front_face = is_front;
            _readFloorHit(k, rec_out);
        } else {
            rec_out.state.hit_state.is_floor_hit = false;
            rec_out.p = ray.at(rec_out.t);
            _readVoxelHit(k, rec_out);
        }
        return true;
    }

    /*****************************************************************
     * Material-property resolution
     *****************************************************************/

    /**
     * @brief Sample a procedural grid texture at a shading coordinate, picking line_color vs
     *        fill_color (the metric position when is_meters, else the normalized UV).
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _sampleGridTexture(Kernel* k, RHandle h, const Vec2C& pos, const Vec2C& uv) {
        const auto& grid = k->ctx.texture_grids[h];
        FloatC cx = grid.is_meters ? pos.x : uv.x;
        FloatC cy = grid.is_meters ? pos.y : uv.y;
        FloatC half_width = FloatC(0.5) * FloatC(grid.line_width);
        FloatC dx =
            RS_ABS(cx - RS_FI::rs_round(cx / FloatC(grid.x_spacing)) * FloatC(grid.x_spacing));
        FloatC dy =
            RS_ABS(cy - RS_FI::rs_round(cy / FloatC(grid.y_spacing)) * FloatC(grid.y_spacing));
        return (dx <= half_width || dy <= half_width) ? Vec3C(grid.line_color.toVec3D())
                                                      : Vec3C(grid.fill_color.toVec3D());
    }

    /**
     * @brief Resolve the albedo color for `vision_material` at the shading coordinate. Samples
     *        the bound grid texture when grid-driven, otherwise returns the constant albedo.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _resolveAlbedo(
        Kernel* k,
        RHandle vision_material,
        const RS_VisionMaterialStateUnion& vsm_state,
        const Vec2C& pos,
        const Vec2C& uv,
        Vec3C& albedo_out
    ) {
        auto& ctx = k->ctx;
        switch (vsm_state.vision_material_state.albedo_input_type) {
            case RSE_VISION_SAMPLE_INPUT_GRID:
                albedo_out = _sampleGridTexture(k, VSM_G(vision_material, albedo).texture, pos, uv);
                break;
            default:
                albedo_out = VSM_G(vision_material, albedo).constant.toVec3D();
                break;
        }
    }

    /**
     * @brief Resolve the emission color for `vision_material` at the shading coordinate. The
     *        caller multiplies by emission_intensity to produce final radiance.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _resolveEmissionColor(
        Kernel* k,
        RHandle vision_material,
        const RS_VisionMaterialStateUnion& vsm_state,
        const Vec2C& pos,
        const Vec2C& uv,
        Vec3C& emission_color_out
    ) {
        auto& ctx = k->ctx;
        switch (vsm_state.vision_material_state.emission_input_type) {
            case RSE_VISION_SAMPLE_INPUT_GRID:
                emission_color_out =
                    _sampleGridTexture(k, VSM_G(vision_material, emission_color).texture, pos, uv);
                break;
            default:
                emission_color_out = VSM_G(vision_material, emission_color).constant.toVec3D();
                break;
        }
    }

    /**
     * @brief Resolve the (roughness, metalness) pair for `vision_material` at the shading
     *        coordinate. Grid-driven samples pack roughness in R and metalness in G; otherwise
     *        returns the constant roughness/metalness.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _resolveRoughnessMetalness(
        Kernel* k,
        RHandle vision_material,
        const RS_VisionMaterialStateUnion& vsm_state,
        const Vec2C& pos,
        const Vec2C& uv,
        FloatC& roughness_out,
        FloatC& metalness_out
    ) {
        auto& ctx = k->ctx;
        switch (vsm_state.vision_material_state.roughness_metalness_input_type) {
            case RSE_VISION_SAMPLE_INPUT_GRID: {
                Vec3C rm = _sampleGridTexture(
                    k, VSM_G(vision_material, roughness_metalness).texture, pos, uv
                );
                roughness_out = rm.x;
                metalness_out = rm.y;
                break;
            }
            default: {
                auto rm = VSM_G(vision_material, roughness_metalness).constant;
                roughness_out = FloatC(rm.r);
                metalness_out = FloatC(rm.g);
                break;
            }
        }
    }

    /**
     * @brief Resolve the hit voxel's material params into the hit record. Mirrors _readFloorHit,
     *        sampling at the face UV the BVH walk already measured; face_pos is that same place
     *        in meters, so a grid texture set in meters keeps its real spacing.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _readVoxelHit(Kernel* k, HitRecord& rec_in_out) {
        auto& ctx = k->ctx;

        rec_in_out.shading_normal = rec_in_out.normal;
        if (rec_in_out.state.hit_state.has_face_normal) {
            RHandle voxel = rec_in_out.voxel;
            RHandle voxel_face_normal = V_G(voxel_face_normal);
            Quat3C orientation = V_G(orientation);
            Vec3C smooth = orientation.rotateVec3D(Vec3C(
                VFN_G(face_normal)[rec_in_out.state.hit_state.face].template toVec3D<FloatC>()
            ));
            // The baked normal points out of the material, so on an exit it faces away from the
            // side the geometry picked. Turn it onto that side rather than drop it, or only
            // entry crossings would bend and a lens could not hold a focus.
            rec_in_out.shading_normal = smooth.dot(rec_in_out.normal) < FloatC(0) ? -smooth
                                                                                  : smooth;
        }

        RHandle vision_material = rec_in_out.vision_material;
        RS_VisionMaterialStateUnion mat_state{VSM_G(vision_material, state)};
        const RSE_VisionMaterialType render_type = mat_state.vision_material_state.render_type;

        const Vec2C uv = rec_in_out.uv;
        const Vec2C pos = rec_in_out.face_pos;

        Vec3C albedo, emission_color;
        FloatC roughness, metalness;
        _resolveAlbedo(k, vision_material, mat_state, pos, uv, albedo);
        _resolveRoughnessMetalness(k, vision_material, mat_state, pos, uv, roughness, metalness);
        _resolveEmissionColor(k, vision_material, mat_state, pos, uv, emission_color);

        // Emissive surfaces emit emission_color * emission_intensity downstream (rec.albedo *
        // rec.intensity); route the emission color through the albedo slot in that case.
        rec_in_out.albedo = (render_type == RSE_VISION_MATERIAL_EMISSIVE) ? emission_color : albedo;
        rec_in_out.roughness = roughness;
        rec_in_out.metalness = metalness;
        rec_in_out.ior = FloatC(VSM_G(vision_material, ior));
        rec_in_out.intensity = FloatC(VSM_G(vision_material, emission_intensity));
        rec_in_out.vision_material_key = VSM_G(vision_material, vision_material_key);
        rec_in_out.state.hit_state.vision_material_type = render_type;
    }

    /**
     * @brief Resolve the floor's referenced material into the hit record. Mirrors
     *        _loadVoxelMaterial, sampling at the floor UV: flat floors tile per uv_scale,
     *        array floors span the whole heightmap; pos is the world XY in meters.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _readFloorHit(Kernel* k, HitRecord& rec_in_out) {
        auto& ctx = k->ctx;
        rec_in_out.shading_normal = rec_in_out.normal;
        const auto& fl = k->ctx.floor_vision_mem;
        RHandle vision_material = fl.vision_material;
        RS_VisionMaterialStateUnion mat_state{VSM_G(vision_material, state)};
        const RSE_VisionMaterialType render_type = mat_state.vision_material_state.render_type;

        const Vec2C pos = Vec2C(rec_in_out.p.x, rec_in_out.p.y);
        Vec2C uv;
        if (fl.data_type == RSE_VISION_FLOOR_DATA_ARRAY) {
            const auto& fa = k->ctx.floor;
            uv = Vec2C(
                rec_in_out.p.x / FloatC(fa.x_size) + FloatC(0.5),
                rec_in_out.p.y / FloatC(fa.y_size) + FloatC(0.5)
            );
        } else {
            uv = pos / FloatC(fl.uv_scale);
        }
        rec_in_out.uv = uv;
        rec_in_out.face_pos = pos;

        Vec3C albedo, emission_color;
        FloatC roughness, metalness;
        _resolveAlbedo(k, vision_material, mat_state, pos, uv, albedo);
        _resolveRoughnessMetalness(k, vision_material, mat_state, pos, uv, roughness, metalness);
        _resolveEmissionColor(k, vision_material, mat_state, pos, uv, emission_color);

        rec_in_out.albedo = (render_type == RSE_VISION_MATERIAL_EMISSIVE) ? emission_color : albedo;
        rec_in_out.roughness = roughness;
        rec_in_out.metalness = metalness;
        rec_in_out.ior = FloatC(VSM_G(vision_material, ior));
        rec_in_out.intensity = FloatC(VSM_G(vision_material, emission_intensity));
        rec_in_out.vision_material_key = VSM_G(vision_material, vision_material_key);
        rec_in_out.state.hit_state.vision_material_type = render_type;
    }

    /*****************************************************************
     * Material scattering
     *****************************************************************/

    /**
     * @brief Two-lobe opaque scatter: Lambertian diffuse + GGX-VNDF specular,
     *        blended by metalness and dielectric Fresnel. Variance-optimal
     *        lobe selection by directional albedo (Veach 1997 §9.2). Sets
     *        `rec.color = f * NoL / pdf` so callers multiply it into their
     *        running throughput unchanged.
     */
    HOST_DEVICE_INLINE_QUALIFIER static void _scatterOpaque(
        const Ray& incoming_ray,
        device::RS_RandomState& rs_in_out,
        HitRecord& rec_in_out
    ) {
        FloatC roughness = RS_MAX(rec_in_out.roughness, PBR_MIN_ROUGHNESS);
        FloatC alpha = roughness * roughness;
        FloatC metalness = rec_in_out.metalness;
        FloatC ior = rec_in_out.ior;
        Vec3C albedo = rec_in_out.albedo;
        Vec3C n = rec_in_out.shading_normal;

        Vec3C V_world = -incoming_ray.ray_direction.normalized();
        FloatC NoV = RS_MAX(FloatC(RS_EPS), n.dot(V_world));

        Vec3C u_axis, v_axis;
        _buildONB(n, u_axis, v_axis);
        Vec3C V_local(V_world.dot(u_axis), V_world.dot(v_axis), V_world.dot(n));
        if (V_local.z < FloatC(RS_EPS)) V_local.z = FloatC(RS_EPS);
        V_local = V_local.normalized();

        FloatC spec_prob = _specProbOpaque(albedo, metalness, ior, NoV);

        FloatC xi = rs_in_out.template rand<FloatC>();
        FloatC r1 = rs_in_out.template rand<FloatC>();
        FloatC r2 = rs_in_out.template rand<FloatC>();

        Vec3C L_local;
        if (xi < spec_prob) {
            Vec3C H_local = _sampleGGXVNDF(V_local, alpha, r1, r2);
            FloatC VoH = RS_MAX(FloatC(RS_EPS), V_local.dot(H_local));
            L_local = H_local * (FloatC(2) * VoH) - V_local;
        } else {
            FloatC phi = RS_2PI * r1;
            FloatC sin_theta = RS_FI::rs_sqrt(r2);
            FloatC cos_theta = RS_FI::rs_sqrt(FloatC(1) - r2);
            FloatC sin_phi, cos_phi;
            RS_FI::rs_sincos(phi, &sin_phi, &cos_phi);
            L_local = Vec3C(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
        }

        if (L_local.z <= FloatC(0)) {
            // Invalid sample: scattered direction below the hemisphere. Point
            // it INTO the surface so downstream `_brdfPdf` returns 0 (NoL < 0)
            // and the caller's `brdf_pdf <= eps` guard skips the candidate.
            // Pointing along +n would yield NoL = 1 — a silent perpendicular
            // bounce ray that produces an orthographic light "projection".
            rec_in_out.color = Vec3C::zero();
            rec_in_out.scattered = Ray(_offsetRayOrigin(rec_in_out.p, -n), -n);
            return;
        }

        Vec3C H = (V_local + L_local).normalized();
        FloatC NoL = L_local.z;
        FloatC NoH = RS_MAX(FloatC(0), H.z);
        FloatC VoH = RS_MAX(FloatC(RS_EPS), V_local.dot(H));

        FloatC D = _ggxD(NoH, alpha);
        FloatC G2 = _ggxG2HeightCorrelated(NoV, NoL, alpha);
        FloatC G1_V = _ggxG1(NoV, alpha);
        Vec3C F_h = _fresnelOpaqueAtH(albedo, metalness, ior, VoH);

        Vec3C f_spec = F_h * (D * G2 / (FloatC(4) * NoV * NoL));
        Vec3C f_diff = albedo * ((FloatC(1) - metalness) / FloatC(RS_PI));
        Vec3C f_total = f_spec + f_diff;

        FloatC pdf_spec = D * G1_V / (FloatC(4) * NoV);
        FloatC pdf_diff = NoL / FloatC(RS_PI);
        FloatC pdf_combined = spec_prob * pdf_spec + (FloatC(1) - spec_prob) * pdf_diff;

        Vec3C L_world = (u_axis * L_local.x + v_axis * L_local.y + n * L_local.z).normalized();
        rec_in_out.color = f_total * (NoL / RS_MAX(pdf_combined, FloatC(RS_EPS)));
        rec_in_out.scattered = Ray(_offsetRayOrigin(rec_in_out.p, n), L_world);
    }

    /**
     * @brief Deterministic mirror reflection with Schlick metallic Fresnel.
     *        Used by PTUD's specular chain walker where stochastic VNDF
     *        sampling would break the chain's deterministic premise.
     */
    HOST_DEVICE_INLINE_QUALIFIER static void
    _scatterMirrorDelta(const Ray& incoming_ray, HitRecord& rec_in_out) {
        Vec3C unit_dir = incoming_ray.ray_direction.normalized();
        Vec3C n = rec_in_out.shading_normal;
        FloatC NoV = RS_MAX(FloatC(0), -unit_dir.dot(n));
        Vec3C reflected = unit_dir - n * (FloatC(2) * unit_dir.dot(n));
        rec_in_out.color = _fresnelSchlick(NoV, rec_in_out.albedo);
        rec_in_out.scattered =
            Ray(_offsetRayOrigin(rec_in_out.p, rec_in_out.normal), reflected.normalized());
    }

    /**
     * @brief Stochastic dielectric scatter at an IOR boundary: Fresnel-importance
     *        sampling between reflect and refract, with TIR fallback to mirror.
     *        Used by the basic-mode depth loop where per-sample stochastic
     *        decisions average across `samples_per_pixel` rays. The deterministic
     *        PTUD chain uses `_scatterTransparentWithIORDeterministic` instead.
     *
     * @details `rec.color = albedo` for both branches — importance-sampled
     *          Fresnel weights drop out of the delta-lobe estimator, leaving
     *          Beer-Lambert albedo tint on both reflect and refract. Roughness
     *          perturbs the chosen direction inside a cone for rough glass /
     *          water. `is_refraction` is set so the caller can update its
     *          Path B medium tracker.
     */
    HOST_DEVICE_INLINE_QUALIFIER static void _scatterTransparentWithIOR(
        const Ray& incoming_ray,
        FloatC ior_from,
        FloatC ior_to,
        device::RS_RandomState& rs_in_out,
        HitRecord& rec_in_out
    ) {
        Vec3C unit_dir = incoming_ray.ray_direction.normalized();
        Vec3C n = rec_in_out.shading_normal;
        FloatC cos_theta = RS_MAX(FloatC(0), RS_MIN(-unit_dir.dot(n), FloatC(1)));
        FloatC eta = ior_from / ior_to;
        FloatC sin2_t = eta * eta * (FloatC(1) - cos_theta * cos_theta);
        bool cannot_refract = sin2_t > FloatC(1);
        FloatC F = cannot_refract ? FloatC(1) : _fresnelDielectric(cos_theta, ior_to / ior_from);
        bool is_reflect = cannot_refract || (rs_in_out.template rand<FloatC>() < F);

        Vec3C direction;
        if (is_reflect) {
            direction = (unit_dir - n * (FloatC(2) * unit_dir.dot(n))).normalized();
        } else {
            Vec3C r_perp = (unit_dir + n * cos_theta) * eta;
            FloatC r_par_z = -RS_FI::rs_sqrt(RS_MAX(FloatC(0), FloatC(1) - r_perp.dot(r_perp)));
            direction = (r_perp + n * r_par_z).normalized();
        }

        if (rec_in_out.roughness > FloatC(RS_EPS)) {
            Vec3C u_axis, v_axis;
            _buildONB(direction, u_axis, v_axis);
            FloatC r1 = rs_in_out.template rand<FloatC>();
            FloatC r2 = rs_in_out.template rand<FloatC>();
            FloatC phi = RS_2PI * r1;
            FloatC sin_pt = RS_FI::rs_sqrt(r2) * rec_in_out.roughness;
            FloatC cos_pt = RS_FI::rs_sqrt(RS_MAX(FloatC(0), FloatC(1) - sin_pt * sin_pt));
            FloatC sin_phi, cos_phi;
            RS_FI::rs_sincos(phi, &sin_phi, &cos_phi);
            direction =
                (u_axis * (cos_phi * sin_pt) + v_axis * (sin_phi * sin_pt) + direction * cos_pt)
                    .normalized();
        }

        Vec3C offset_normal = is_reflect ? rec_in_out.normal : -rec_in_out.normal;
        rec_in_out.scattered = Ray(_offsetRayOrigin(rec_in_out.p, offset_normal), direction);
        rec_in_out.state.hit_state.is_refraction = !is_reflect;
        rec_in_out.color = rec_in_out.albedo;
    }

    /**
     * @brief Deterministic refract-side dielectric scatter at an IOR boundary,
     *        with TIR fallback to mirror reflection. Analogous in shape to
     *        `_scatterMirrorDelta`: one ray in, one scattered ray + Beer-Lambert
     *        / Fresnel weight out. The reflect side of the interface is handled
     *        separately by `_gatherReflectChain` (mini-PTUD on the reflect
     *        direction). Used by the PTUD chain walkers where stochastic
     *        per-frame choices would break determinism and cause flicker.
     *
     * @details `rec.color = albedo * (1 - F)` on refract, `(1,1,1)` on TIR.
     *          Roughness perturbs the chosen direction inside a cone.
     */
    HOST_DEVICE_INLINE_QUALIFIER static void _scatterTransparentWithIORDeterministic(
        const Ray& incoming_ray,
        FloatC ior_from,
        FloatC ior_to,
        device::RS_RandomState& rs_in_out,
        HitRecord& rec_in_out
    ) {
        Vec3C unit_dir = incoming_ray.ray_direction.normalized();
        Vec3C n = rec_in_out.shading_normal;
        FloatC cos_theta = RS_MAX(FloatC(0), RS_MIN(-unit_dir.dot(n), FloatC(1)));
        FloatC eta = ior_from / ior_to;
        FloatC sin2_t = eta * eta * (FloatC(1) - cos_theta * cos_theta);
        bool is_tir = sin2_t > FloatC(1);

        Vec3C direction;
        if (is_tir) {
            direction = (unit_dir - n * (FloatC(2) * unit_dir.dot(n))).normalized();
        } else {
            Vec3C r_perp = (unit_dir + n * cos_theta) * eta;
            FloatC r_par_z = -RS_FI::rs_sqrt(RS_MAX(FloatC(0), FloatC(1) - r_perp.dot(r_perp)));
            direction = (r_perp + n * r_par_z).normalized();
        }

        if (rec_in_out.roughness > FloatC(RS_EPS)) {
            Vec3C u_axis, v_axis;
            _buildONB(direction, u_axis, v_axis);
            FloatC r1 = rs_in_out.template rand<FloatC>();
            FloatC r2 = rs_in_out.template rand<FloatC>();
            FloatC phi = RS_2PI * r1;
            FloatC sin_pt = RS_FI::rs_sqrt(r2) * rec_in_out.roughness;
            FloatC cos_pt = RS_FI::rs_sqrt(RS_MAX(FloatC(0), FloatC(1) - sin_pt * sin_pt));
            FloatC sin_phi, cos_phi;
            RS_FI::rs_sincos(phi, &sin_phi, &cos_phi);
            direction =
                (u_axis * (cos_phi * sin_pt) + v_axis * (sin_phi * sin_pt) + direction * cos_pt)
                    .normalized();
        }

        if (is_tir) {
            rec_in_out.scattered =
                Ray(_offsetRayOrigin(rec_in_out.p, rec_in_out.normal), direction);
            rec_in_out.color = Vec3C::one();
            rec_in_out.state.hit_state.is_refraction = false;
            return;
        }

        FloatC F = _fresnelDielectric(cos_theta, ior_to / ior_from);
        rec_in_out.scattered = Ray(_offsetRayOrigin(rec_in_out.p, -rec_in_out.normal), direction);
        rec_in_out.color = rec_in_out.albedo * (FloatC(1) - F);
        rec_in_out.state.hit_state.is_refraction = true;
    }

    /*****************************************************************
     * Cosmic background models
     *****************************************************************/

    /**
     * @brief Equirectangular UV for a world-space direction, rotated about the
     *        world up axis by `rotation_rad`. u from the azimuth, v from the
     *        elevation; matches the texture lookup convention.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec2C
    _cosmicBackgroundDirToUv(const Vec3C& dir, FloatC rotation_rad) {
        Vec3C d = dir.normalized();
        FloatC sin_r, cos_r;
        RS_FI::rs_sincos(rotation_rad, &sin_r, &cos_r);
        FloatC dx = cos_r * d.x + sin_r * d.y;
        FloatC dy = -sin_r * d.x + cos_r * d.y;
        FloatC u = FloatC(0.5) + RS_FI::rs_atan2(dy, dx) / FloatC(RS_2PI);
        FloatC z = RS_MAX(FloatC(-1), RS_MIN(FloatC(1), d.z));
        FloatC v = FloatC(0.5) + RS_FI::rs_asin(z) / FloatC(RS_PI);
        return Vec2C(u, v);
    }

    /**
     * @brief Uniform background: flat color (CONSTANT) or the procedural grid
     *        sampled at the equirectangular UV (GRID), scaled by intensity.
     *        Has no sun disk.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _evalCosmicBackgroundUniform(Kernel* k, const Vec3C& dir) {
        const auto& cb = k->ctx.cosmic_background_vision_mem;
        RS_VisionCosmicBackgroundStateUnion st{cb.state};
        if (st.cosmic_background_state.color_input_type == RSE_VISION_SAMPLE_INPUT_GRID) {
            Vec2C uv = _cosmicBackgroundDirToUv(dir, FloatC(cb.rotation_rad));
            return _sampleGridTexture(k, cb.color.texture, uv, uv) * FloatC(cb.intensity);
        }
        return Vec3C(cb.color.constant.toVec3D()) * FloatC(cb.intensity);
    }

    /**
     * @brief Two-tone gradient sky, sun disk excluded. Zenith and horizon
     *        colors blend by `1 - |dir.z|`; the sun is added separately by
     *        `_evalCosmicBackgroundSunDisk` so MIS can weight it independently.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _evalCosmicBackgroundGradient(Kernel* k, const Vec3C& dir) {
        const auto& cb = k->ctx.cosmic_background_vision_mem;
        FloatC horizon_w = FloatC(1) - RS_ABS(dir.z);
        Vec3C zenith = Vec3C(cb.zenith_color.toVec3D());
        Vec3C horizon = Vec3C(cb.horizon_color.toVec3D());
        return zenith * (FloatC(1) - horizon_w) + horizon * horizon_w;
    }

    /**
     * @brief Hosek-Wilkie analytic clear-sky lookup, sun disk excluded.
     *        Verbatim `ArHosekSkyModel_GetRadianceInternal` over cooked
     *        `hw_config` / `hw_radiance`; below-horizon returns zero.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _evalCosmicBackgroundHosekWilkie(Kernel* k, const Vec3C& dir) {
        const auto& cb = k->ctx.cosmic_background_vision_mem;
        if (dir.z <= FloatC(0)) return Vec3C::zero();
        Vec3C sun_dir = Vec3C(cb.sun_direction);
        FloatC cos_theta = dir.z;
        FloatC cos_gamma = dir.dot(sun_dir);
        if (cos_gamma > FloatC(1)) cos_gamma = FloatC(1);
        if (cos_gamma < FloatC(-1)) cos_gamma = FloatC(-1);

        Vec3C sky_color;
        for (int ch = 0; ch < 3; ++ch) {
            FloatC c0 = FloatC(cb.hw_config[ch * 9 + 0]);
            FloatC c1 = FloatC(cb.hw_config[ch * 9 + 1]);
            FloatC c2 = FloatC(cb.hw_config[ch * 9 + 2]);
            FloatC c3 = FloatC(cb.hw_config[ch * 9 + 3]);
            FloatC c4 = FloatC(cb.hw_config[ch * 9 + 4]);
            FloatC c5 = FloatC(cb.hw_config[ch * 9 + 5]);
            FloatC c6 = FloatC(cb.hw_config[ch * 9 + 6]);
            FloatC c7 = FloatC(cb.hw_config[ch * 9 + 7]);
            FloatC c8 = FloatC(cb.hw_config[ch * 9 + 8]);
            FloatC gamma = RS_FI::rs_acos(cos_gamma);
            FloatC expM = RS_FI::rs_exp_fast(c4 * gamma);
            FloatC rayM = cos_gamma * cos_gamma;
            FloatC mie_denom = FloatC(1) + c8 * c8 - FloatC(2) * c8 * cos_gamma;
            FloatC mieM = (FloatC(1) + rayM) / (mie_denom * RS_FI::rs_sqrt(mie_denom));
            FloatC zenith_term = RS_FI::rs_sqrt(cos_theta);
            FloatC val = (FloatC(1) + c0 * RS_FI::rs_exp_fast(c1 / (cos_theta + FloatC(0.01)))) *
                         (c2 + c3 * expM + c5 * rayM + c6 * mieM + c7 * zenith_term);
            sky_color[ch] = val * FloatC(cb.hw_radiance[ch]);
        }
        sky_color *= FloatC(cb.intensity);
        return sky_color;
    }

    /**
     * @brief True when `dir` points into the sun disk's angular cone.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool
    _isInCosmicBackgroundSunCone(Kernel* k, const Vec3C& dir) {
        const auto& cb = k->ctx.cosmic_background_vision_mem;
        Vec3C sun_dir = Vec3C(cb.sun_direction);
        FloatC cos_radius = RS_FI::rs_cos(FloatC(cb.sun_angular_radius));
        return dir.dot(sun_dir) >= cos_radius;
    }

    /**
     * @brief Sun disk radiance under the hard-cutoff disk model.
     *        `sun_color` inside the cone, zero outside; shared by both
     *        the cosmic background path and the NEE sun emission.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _evalCosmicBackgroundSunDisk(Kernel* k, const Vec3C& dir) {
        if (!_isInCosmicBackgroundSunCone(k, dir)) return Vec3C::zero();
        return Vec3C(k->ctx.cosmic_background_vision_mem.sun_color.toVec3D());
    }

    /**
     * @brief Background dispatch by cosmic background type, sky plus sun disk.
     *        MIS-weighting callers should split via `_evalCosmicBackgroundGradient` /
     *        `_evalCosmicBackgroundHosekWilkie` and `_evalCosmicBackgroundSunDisk` directly.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _findHitBackground(Kernel* k, const Vec3C& dir) {
        RS_VisionCosmicBackgroundStateUnion st{k->ctx.cosmic_background_vision_mem.state};
        Vec3C base;
        if (st.cosmic_background_state.type == RSE_VISION_COSMIC_BACKGROUND_HOSEK_WILKIE) {
            base = _evalCosmicBackgroundHosekWilkie(k, dir);
        } else if (st.cosmic_background_state.type == RSE_VISION_COSMIC_BACKGROUND_GRADIENT) {
            base = _evalCosmicBackgroundGradient(k, dir);
        } else {
            return _evalCosmicBackgroundUniform(k, dir);
        }
        return base + _evalCosmicBackgroundSunDisk(k, dir);
    }

    /*****************************************************************
     * Light traversal
     *****************************************************************/

    /**
     * @brief True iff the surface continues the deterministic specular chain
     *        used by PTUD's restir info pass. TRANSPARENT always; OPAQUE only
     *        when it is close to a perfect mirror (high metalness, low roughness).
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool
    _isSpecularSurface(RSE_VisionMaterialType material_type, FloatC roughness, FloatC metalness) {
        if (material_type == RSE_VISION_MATERIAL_TRANSPARENT) return true;
        if (material_type == RSE_VISION_MATERIAL_OPAQUE && metalness > FloatC(0.5) &&
            roughness < FloatC(0.05))
            return true;
        return false;
    }

    /**
     * @brief Rec.709 linear luminance of an RGB vector.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _luminance(const Vec3C& rgb) {
        return RColor3f(rgb).luminance();
    }

    /**
     * @brief Veach 1997 power heuristic (beta = 2). Returns 1 when `pdf_a`
     *        is the delta sentinel `MIS_DELTA_BRDF_PDF` so mirror / glass
     *        bounces preserve their full contribution.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _powerHeuristicMIS(FloatC pdf_a, FloatC pdf_b) {
        if (pdf_a >= MIS_DELTA_BRDF_PDF) return FloatC(1);
        FloatC a2 = pdf_a * pdf_a;
        FloatC b2 = pdf_b * pdf_b;
        return a2 / RS_MAX(a2 + b2, FloatC(RS_EPS));
    }

    /**
     * @brief BRDF * cos(theta_i) at the shading point. Lambertian for OPAQUE
     *        non-metal; delta surfaces return zero so the reservoir produces
     *        a zero direct estimate at mirrors, glass, and emitters.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _brdfCos(
        RSE_VisionMaterialType material_type,
        const Vec3C& albedo,
        FloatC metalness,
        FloatC roughness,
        FloatC ior,
        const Vec3C& normal,
        const Vec3C& view_dir,
        const Vec3C& light_dir
    ) {
        if (material_type != RSE_VISION_MATERIAL_OPAQUE) return Vec3C::zero();
        FloatC NoL = normal.dot(light_dir);
        if (NoL <= FloatC(0)) return Vec3C::zero();
        FloatC NoV = RS_MAX(FloatC(RS_EPS), normal.dot(view_dir));
        Vec3C H = (view_dir + light_dir).normalized();
        FloatC NoH = RS_MAX(FloatC(0), normal.dot(H));
        FloatC VoH = RS_MAX(FloatC(RS_EPS), view_dir.dot(H));
        FloatC r = RS_MAX(roughness, PBR_MIN_ROUGHNESS);
        FloatC alpha = r * r;
        FloatC D = _ggxD(NoH, alpha);
        FloatC G2 = _ggxG2HeightCorrelated(NoV, NoL, alpha);
        Vec3C F_h = _fresnelOpaqueAtH(albedo, metalness, ior, VoH);
        Vec3C f_spec = F_h * (D * G2 / (FloatC(4) * NoV * NoL));
        Vec3C f_diff = albedo * ((FloatC(1) - metalness) / FloatC(RS_PI));
        return (f_spec + f_diff) * NoL;
    }

    /**
     * @brief Probability density of the BRDF sampler producing `light_dir`.
     *        Lambertian cosine-weighted on the hemisphere; delta surfaces
     *        return the sentinel `MIS_DELTA_BRDF_PDF` so power heuristic
     *        keeps them weighted at 1.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _brdfPdf(
        RSE_VisionMaterialType material_type,
        const Vec3C& albedo,
        FloatC metalness,
        FloatC roughness,
        FloatC ior,
        const Vec3C& normal,
        const Vec3C& view_dir,
        const Vec3C& light_dir
    ) {
        if (material_type != RSE_VISION_MATERIAL_OPAQUE) return MIS_DELTA_BRDF_PDF;
        FloatC NoL = normal.dot(light_dir);
        if (NoL <= FloatC(0)) return FloatC(0);
        FloatC NoV = RS_MAX(FloatC(RS_EPS), normal.dot(view_dir));
        Vec3C H = (view_dir + light_dir).normalized();
        FloatC NoH = RS_MAX(FloatC(0), normal.dot(H));
        FloatC r = RS_MAX(roughness, PBR_MIN_ROUGHNESS);
        FloatC alpha = r * r;
        FloatC D = _ggxD(NoH, alpha);
        FloatC G1_V = _ggxG1(NoV, alpha);
        FloatC pdf_spec = D * G1_V / (FloatC(4) * NoV);
        FloatC pdf_diff = NoL / FloatC(RS_PI);
        FloatC spec_prob = _specProbOpaque(albedo, metalness, ior, NoV);
        return spec_prob * pdf_spec + (FloatC(1) - spec_prob) * pdf_diff;
    }

    /**
     * @brief Count voxel cube faces front-facing toward a receiver and return
     *        their total area. Front-facing iff the receiver lies outside the
     *        face's axis-slab; 1-3 normally, 0 only if the receiver is inside
     *        the voxel.
     */
    HOST_DEVICE_INLINE_QUALIFIER static void _voxelFrontFaces(
        const Vec3C& voxel_center,
        FloatC voxel_size,
        const Vec3C& receiver,
        int& n_front_out,
        FloatC& area_visible_out
    ) {
        FloatC half = voxel_size * FloatC(0.5);
        int n = 0;
        if (receiver.x < voxel_center.x - half || receiver.x > voxel_center.x + half) ++n;
        if (receiver.y < voxel_center.y - half || receiver.y > voxel_center.y + half) ++n;
        if (receiver.z < voxel_center.z - half || receiver.z > voxel_center.z + half) ++n;
        n_front_out = n;
        area_visible_out = FloatC(n) * voxel_size * voxel_size;
    }

    /**
     * @brief Recover the cube face a stored sample point lies on from its
     *        offset to the voxel center. Returns the outward face normal along
     *        +/-X, +/-Y, or +/-Z.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _voxelFaceNormalFromPoint(const Vec3C& sample_point, const Vec3C& voxel_center) {
        Vec3C offset = sample_point - voxel_center;
        FloatC abs_x = RS_ABS(offset.x);
        FloatC abs_y = RS_ABS(offset.y);
        FloatC abs_z = RS_ABS(offset.z);
        if (abs_x >= abs_y && abs_x >= abs_z) {
            return Vec3C((offset.x >= FloatC(0)) ? FloatC(1) : FloatC(-1), 0, 0);
        }
        if (abs_y >= abs_z) {
            return Vec3C(0, (offset.y >= FloatC(0)) ? FloatC(1) : FloatC(-1), 0);
        }
        return Vec3C(0, 0, (offset.z >= FloatC(0)) ? FloatC(1) : FloatC(-1));
    }

    /**
     * @brief Area-light estimator geometry factor `cos_emit * area / dist^2`.
     *        Sun returns 1 (parallel light has no inverse-square or emit-cos).
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _lightGeometryFactor(const LightSample& ls) {
        if (ls.is_sun) return FloatC(1);
        return ls.cos_emit * ls.area_visible / (ls.dist * ls.dist);
    }

    /**
     * @brief Bitterli weighted-reservoir-sampling streaming update. Carries
     *        a candidate transmission so spatial-reuse merges adopt the
     *        winning neighbor's shadow walk without a fresh trace.
     */
    HOST_DEVICE_INLINE_QUALIFIER static void _reservoirUpdateDI(
        FloatC sample_weight,
        RIndex light_index,
        const RVec3f& light_position,
        const RColor3f& transmission,
        FloatC target_pdf,
        device::RS_RandomState& rs_in_out,
        RS_ReservoirDI& reservoir_in_out
    ) {
        reservoir_in_out.sample_count = reservoir_in_out.sample_count + 1;
        reservoir_in_out.weight_sum = float(FloatC(reservoir_in_out.weight_sum) + sample_weight);
        FloatC rand_u = rs_in_out.template rand<FloatC>();
        if (rand_u * FloatC(reservoir_in_out.weight_sum) < sample_weight) {
            reservoir_in_out.light_index = light_index;
            reservoir_in_out.light_position = light_position;
            reservoir_in_out.transmission = transmission;
            reservoir_in_out.target_pdf = float(target_pdf);
        }
    }

    /**
     * @brief Bitterli WRS streaming update on a GI reservoir. Survives the
     *        bounce direction, bounce-ray hit, and gathered radiance.
     */
    HOST_DEVICE_INLINE_QUALIFIER static void _reservoirUpdateGI(
        FloatC sample_weight,
        const RVec3f& bounce_direction,
        const RVec3f& bounce_hit_position,
        const RVec3f& bounce_hit_normal,
        const RColor3f& l_incoming,
        FloatC target_pdf,
        device::RS_RandomState& rs_in_out,
        RS_ReservoirGI& reservoir_in_out
    ) {
        reservoir_in_out.sample_count = reservoir_in_out.sample_count + 1;
        reservoir_in_out.weight_sum = float(FloatC(reservoir_in_out.weight_sum) + sample_weight);
        FloatC rand_u = rs_in_out.template rand<FloatC>();
        if (rand_u * FloatC(reservoir_in_out.weight_sum) < sample_weight) {
            reservoir_in_out.bounce_direction = bounce_direction;
            reservoir_in_out.bounce_hit_position = bounce_hit_position;
            reservoir_in_out.bounce_hit_normal = bounce_hit_normal;
            reservoir_in_out.l_incoming = l_incoming;
            reservoir_in_out.target_pdf = float(target_pdf);
        }
    }

    /**
     * @brief CDF-pick a light index proportional to `light_weight`. Returns
     *        the smallest `i` with `light_cdf[i] >= u * light_cdf.back()`
     *        via inclusive-CDF binary search.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static RIndex _sampleLight(Kernel* k, FloatC u) {
        auto& ctx = k->ctx;
        RSize n = ctx.light_cdf.size();
        if (n == 0) return RIndex(0);
        FloatC total = FloatC(ctx.light_cdf[n - 1]);
        if (total <= FloatC(0)) return RIndex(0);
        FloatC target = u * total;
        RSize lo = 0;
        RSize hi = n;
        while (lo < hi) {
            RSize mid = (lo + hi) >> 1;
            if (FloatC(ctx.light_cdf[mid]) < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo >= n) lo = n - 1;
        return RIndex(lo);
    }

    /**
     * @brief Probability of drawing `light_index` from the source CDF.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _lightSourcePdf(Kernel* k, RIndex light_index) {
        auto& ctx = k->ctx;
        RSize n = ctx.light_cdf.size();
        if (n == 0 || RSize(light_index) >= n) return FloatC(0);
        FloatC total = FloatC(ctx.light_cdf[n - 1]);
        if (total <= FloatC(0)) return FloatC(0);
        FloatC weight = FloatC(ctx.light_weight[RSize(light_index)]);
        return weight / total;
    }

    /**
     * @brief Resolve an emissive-voxel light index into its world center, cube
     *        edge, and surface radiance L = chromaticity * intensity (W/m^2/sr).
     *        Caller does the per-shading-point area sampling.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _readVoxelLight(
        Kernel* k,
        RIndex light_index,
        Vec3C& position_out,
        FloatC& voxel_size_out,
        Vec3C& emission_out
    ) {
        auto& ctx = k->ctx;
        RHandle voxel = ctx.structure_emissive_voxel_handles[light_index - 1];
        position_out = Vec3C(V_G(voxel, position));
        RHandle vision_material = V_G(voxel, vision_material);
        RS_VisionMaterialStateUnion mat_state{VSM_G(vision_material, state)};
        Vec3C emission_color;
        _resolveEmissionColor(
            k, vision_material, mat_state, Vec2C::zero(), Vec2C::zero(), emission_color
        );
        FloatC intensity = FloatC(VSM_G(vision_material, emission_intensity));
        emission_out = emission_color * intensity;
        RHandle voxel_material = V_G(voxel, voxel_material);
        voxel_size_out = FloatC(VM_G(voxel_material, voxel_size));
    }

    /**
     * @brief Sample a direction uniformly inside the sun's angular disc.
     *        Uniform-on-cone around the cosmic background sun direction; degenerate cone
     *        collapses to the sun axis exactly.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _sampleSunConeDirection(Kernel* k, device::RS_RandomState& rs_in_out) {
        Vec3C sun_dir = Vec3C(k->ctx.cosmic_background_vision_mem.sun_direction);
        FloatC half_angle = FloatC(k->ctx.cosmic_background_vision_mem.sun_angular_radius);
        FloatC cos_half = RS_FI::rs_cos(half_angle);
        FloatC one_minus_cos = FloatC(1) - cos_half;
        if (one_minus_cos <= FloatC(RS_EPS)) return sun_dir;
        FloatC u1 = rs_in_out.template rand<FloatC>();
        FloatC u2 = rs_in_out.template rand<FloatC>();
        FloatC cos_alpha = FloatC(1) - u1 * one_minus_cos;
        FloatC sin_alpha = RS_FI::rs_sqrt(RS_MAX(FloatC(1) - cos_alpha * cos_alpha, FloatC(0)));
        FloatC phi = FloatC(2) * FloatC(RS_PI) * u2;
        FloatC sin_phi = RS_FI::rs_sin(phi);
        FloatC cos_phi = RS_FI::rs_cos(phi);
        Vec3C onb_u, onb_v;
        _buildONB(sun_dir, onb_u, onb_v);
        return (onb_u * (sin_alpha * cos_phi) + onb_v * (sin_alpha * sin_phi) + sun_dir * cos_alpha)
            .normalized();
    }

    /**
     * @brief Fresh light sample at `hit_pos`. Sun cone-samples a direction;
     *        voxel uniformly picks one of the 1-3 front-facing cube faces and
     *        then a uniform 2D point on it. Uniform face pick is a lofi /
     *        efficiency choice — variance-optimal would weight by projected
     *        area `cos_emit * s^2`; the diversity here is bounded (1-3) so the
     *        loss is small and the code stays straight.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static LightSample _sampleLightSample(
        Kernel* k,
        RIndex light_index,
        const Vec3C& hit_pos,
        device::RS_RandomState& rs_in_out
    ) {
        LightSample ls;
        if (light_index == RS_VISION_SUN_LIGHT_INDEX) {
            Vec3C sun_dir = _sampleSunConeDirection(k, rs_in_out);
            ls.dir = sun_dir;
            ls.dist = FloatC(RS_MAX_FLOAT);
            ls.position = Vec3C::zero();
            ls.emission = Vec3C(k->ctx.cosmic_background_vision_mem.sun_color.toVec3D());
            ls.cos_emit = FloatC(1);
            ls.area_visible = FloatC(1);
            ls.is_sun = true;
            return ls;
        }
        Vec3C voxel_center, voxel_emission;
        FloatC voxel_size;
        _readVoxelLight(k, light_index, voxel_center, voxel_size, voxel_emission);

        int n_front;
        FloatC area_visible;
        _voxelFrontFaces(voxel_center, voxel_size, hit_pos, n_front, area_visible);
        if (n_front == 0) {
            // Receiver inside the voxel: degenerate, zero-area sample.
            ls.position = voxel_center;
            ls.dir = Vec3C(0, 0, 1);
            ls.dist = FloatC(RS_EPS);
            ls.emission = voxel_emission;
            ls.cos_emit = FloatC(0);
            ls.area_visible = FloatC(0);
            ls.is_sun = false;
            return ls;
        }

        // Pick one of the front-facing axes uniformly.
        FloatC half = voxel_size * FloatC(0.5);
        int signs[3] = {0, 0, 0};
        if (hit_pos.x < voxel_center.x - half)
            signs[0] = -1;
        else if (hit_pos.x > voxel_center.x + half)
            signs[0] = +1;
        if (hit_pos.y < voxel_center.y - half)
            signs[1] = -1;
        else if (hit_pos.y > voxel_center.y + half)
            signs[1] = +1;
        if (hit_pos.z < voxel_center.z - half)
            signs[2] = -1;
        else if (hit_pos.z > voxel_center.z + half)
            signs[2] = +1;

        FloatC u_face = rs_in_out.template rand<FloatC>();
        int pick = int(u_face * FloatC(n_front));
        if (pick >= n_front) pick = n_front - 1;
        int axis = 0;
        for (int a = 0, seen = 0; a < 3; ++a) {
            if (signs[a] != 0) {
                if (seen == pick) {
                    axis = a;
                    break;
                }
                ++seen;
            }
        }
        FloatC face_sign = FloatC(signs[axis]);
        Vec3C face_normal(
            (axis == 0) ? face_sign : FloatC(0),
            (axis == 1) ? face_sign : FloatC(0),
            (axis == 2) ? face_sign : FloatC(0)
        );

        // Uniform 2D point on the chosen face.
        FloatC u = rs_in_out.template rand<FloatC>() - FloatC(0.5);
        FloatC v = rs_in_out.template rand<FloatC>() - FloatC(0.5);
        Vec3C tangent_u, tangent_v;
        if (axis == 0) {
            tangent_u = Vec3C(0, 1, 0);
            tangent_v = Vec3C(0, 0, 1);
        } else if (axis == 1) {
            tangent_u = Vec3C(1, 0, 0);
            tangent_v = Vec3C(0, 0, 1);
        } else {
            tangent_u = Vec3C(1, 0, 0);
            tangent_v = Vec3C(0, 1, 0);
        }
        Vec3C sample_point = voxel_center + face_normal * half + tangent_u * (u * voxel_size) +
                             tangent_v * (v * voxel_size);

        Vec3C diff = sample_point - hit_pos;
        FloatC d = diff.length();
        ls.position = sample_point;
        ls.dir = (d > FloatC(RS_EPS)) ? diff * (FloatC(1) / d) : face_normal * FloatC(-1);
        ls.dist = d;
        ls.emission = voxel_emission;
        ls.cos_emit = RS_MAX(-face_normal.dot(ls.dir), FloatC(0));
        ls.area_visible = area_visible;
        ls.is_sun = false;
        return ls;
    }

    /**
     * @brief Re-evaluate a stored light sample at a new shading point. Sun
     *        always resolves to the cosmic background sun direction; voxel recovers the face
     *        the stored point sits on, then recomputes cos_emit / area_visible
     *        for the new receiver.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static LightSample _evalStoredLightSample(
        Kernel* k,
        RIndex light_index,
        const Vec3C& hit_pos,
        const RVec3f& stored_position
    ) {
        LightSample ls;
        if (light_index == RS_VISION_SUN_LIGHT_INDEX) {
            ls.dir = Vec3C(k->ctx.cosmic_background_vision_mem.sun_direction);
            ls.dist = FloatC(RS_MAX_FLOAT);
            ls.position = Vec3C::zero();
            ls.emission = Vec3C(k->ctx.cosmic_background_vision_mem.sun_color.toVec3D());
            ls.cos_emit = FloatC(1);
            ls.area_visible = FloatC(1);
            ls.is_sun = true;
            return ls;
        }
        Vec3C voxel_center, voxel_emission;
        FloatC voxel_size;
        _readVoxelLight(k, light_index, voxel_center, voxel_size, voxel_emission);

        Vec3C sample_point = Vec3C(stored_position);
        Vec3C face_normal = _voxelFaceNormalFromPoint(sample_point, voxel_center);

        Vec3C diff = sample_point - hit_pos;
        FloatC d = diff.length();
        ls.position = sample_point;
        ls.dir = (d > FloatC(RS_EPS)) ? diff * (FloatC(1) / d) : face_normal * FloatC(-1);
        ls.dist = d;
        ls.emission = voxel_emission;
        ls.cos_emit = RS_MAX(-face_normal.dot(ls.dir), FloatC(0));

        int n_front;
        FloatC area_visible;
        _voxelFrontFaces(voxel_center, voxel_size, hit_pos, n_front, area_visible);
        ls.area_visible = (n_front > 0) ? area_visible : FloatC(0);
        ls.is_sun = false;
        return ls;
    }

    /**
     * @brief NEE solid-angle pdf for the sun pick at `dir`: returns
     *        `p_pick_sun / omega_sun_cone` inside the cone, zero outside.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _evalNeePdfSun(Kernel* k, const Vec3C& dir) {
        if (!_isInCosmicBackgroundSunCone(k, dir)) return FloatC(0);
        FloatC half_angle = FloatC(k->ctx.cosmic_background_vision_mem.sun_angular_radius);
        FloatC cos_half = RS_FI::rs_cos(half_angle);
        FloatC one_minus_cos = FloatC(1) - cos_half;
        if (one_minus_cos <= FloatC(RS_EPS)) return FloatC(0);
        FloatC pick = _lightSourcePdf(k, RS_VISION_SUN_LIGHT_INDEX);
        FloatC omega = FloatC(2) * FloatC(RS_PI) * one_minus_cos;
        return pick / omega;
    }

    /**
     * @brief NEE solid-angle pdf for the specific voxel `bounce_hit_voxel`
     *        as seen from `hit_position`. Returns zero when the voxel is not
     *        in the emissive light list or geometrically unreachable.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _evalNeePdfVoxel(
        Kernel* k,
        const Vec3C& hit_position,
        const Vec3C& dir,
        RHandle bounce_hit_voxel
    ) {
        if (bounce_hit_voxel == RS_NULL_HANDLE) return FloatC(0);
        auto& ctx = k->ctx;
        if (ctx.light_cdf.size() == 0) return FloatC(0);
        FloatC total = FloatC(ctx.light_cdf[ctx.light_cdf.size() - 1]);
        if (total <= FloatC(0)) return FloatC(0);

        RHandle vision_material = V_G(bounce_hit_voxel, vision_material);
        FloatC weight_v = FloatC(VSM_G(vision_material, emission_intensity));
        if (weight_v <= FloatC(0)) return FloatC(0);
        FloatC p_pick = weight_v / total;

        Vec3C voxel_center = Vec3C(V_G(bounce_hit_voxel, position));
        RHandle voxel_material = V_G(bounce_hit_voxel, voxel_material);
        FloatC voxel_size = FloatC(VM_G(voxel_material, voxel_size));
        int n_front;
        FloatC area_visible;
        _voxelFrontFaces(voxel_center, voxel_size, hit_position, n_front, area_visible);
        if (n_front == 0 || area_visible <= FloatC(0)) return FloatC(0);

        FloatC abs_x = RS_ABS(dir.x);
        FloatC abs_y = RS_ABS(dir.y);
        FloatC abs_z = RS_ABS(dir.z);
        Vec3C face_normal = Vec3C::zero();
        if (abs_x >= abs_y && abs_x >= abs_z) {
            face_normal.x = (dir.x >= FloatC(0)) ? FloatC(-1) : FloatC(1);
        } else if (abs_y >= abs_z) {
            face_normal.y = (dir.y >= FloatC(0)) ? FloatC(-1) : FloatC(1);
        } else {
            face_normal.z = (dir.z >= FloatC(0)) ? FloatC(-1) : FloatC(1);
        }
        FloatC cos_emit = RS_MAX(-face_normal.dot(dir), FloatC(0));
        if (cos_emit <= FloatC(RS_EPS)) return FloatC(0);

        Vec3C diff = voxel_center - hit_position;
        FloatC d = diff.length();
        if (d < FloatC(RS_EPS)) return FloatC(0);
        return p_pick * d * d / (area_visible * cos_emit);
    }

    /**
     * @brief NEE solid-angle pdf of the sample `ls` conditional on its
     *        `light_index` being picked. Sun: `p_pick_sun / Omega_disc` inside
     *        the cone, zero outside. Voxel area-sample: convert the surface
     *        pdf `1 / area_visible` to a solid-angle pdf via `dist^2 / cos_emit`
     *        — this is finite, so MIS against the BRDF lobe does real work
     *        instead of always returning 1.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static FloatC
    _evalNeePdf(Kernel* k, RIndex light_index, const LightSample& ls) {
        if (light_index == RS_VISION_SUN_LIGHT_INDEX) {
            if (!_isInCosmicBackgroundSunCone(k, ls.dir)) return FloatC(0);
            FloatC half_angle = FloatC(k->ctx.cosmic_background_vision_mem.sun_angular_radius);
            FloatC cos_half = RS_FI::rs_cos(half_angle);
            FloatC one_minus_cos = FloatC(1) - cos_half;
            if (one_minus_cos <= FloatC(RS_EPS)) return FloatC(0);
            FloatC pick = _lightSourcePdf(k, RS_VISION_SUN_LIGHT_INDEX);
            FloatC omega = FloatC(2) * FloatC(RS_PI) * one_minus_cos;
            return pick / omega;
        }
        if (ls.area_visible <= FloatC(0) || ls.cos_emit <= FloatC(0)) return FloatC(0);
        FloatC pick = _lightSourcePdf(k, light_index);
        return pick * ls.dist * ls.dist / (ls.area_visible * ls.cos_emit);
    }

    /**
     * @brief Marches `_findHit` along the ray, multiplying the running transmission
     *        by each transparent hit's albedo until the ray reaches `target_handle`,
     *        crosses `max_dist`, hits an opaque surface, or its luminance drops below
     *        `SHADOW_EPS`. Returns the surviving per-channel transmission.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static RColor3f _shadowWalkWithTransmission(
        Kernel* k,
        const Vec3C& origin,
        const Vec3C& dir,
        FloatC max_dist,
        RHandle target_handle
    ) {
        constexpr FloatC SHADOW_EPS = FloatC(1e-4);
        constexpr int MAX_SHADOW_STEPS = 16;
        Vec3C transmission = Vec3C::one();
        Ray ray(origin, dir);
        FloatC t_total = FloatC(0);
        for (int step = 0; step < MAX_SHADOW_STEPS; ++step) {
            if (t_total >= max_dist) break;
            HitRecord rec;
            bool any_hit = _findHit(k, ray, rec);
            if (!any_hit) break;
            FloatC remaining = max_dist - t_total;
            if (rec.t >= remaining) break;
            RSE_VisionMaterialType mt =
                static_cast<RSE_VisionMaterialType>(rec.state.hit_state.vision_material_type);
            if (mt == RSE_VISION_MATERIAL_EMISSIVE) {
                if (rec.voxel == target_handle) break;
                return RColor3f::zero();
            }
            if (mt == RSE_VISION_MATERIAL_TRANSPARENT) {
                transmission = transmission.elemMul(rec.albedo);
                if (_luminance(transmission) < SHADOW_EPS) return RColor3f::zero();
                ray = Ray(_offsetRayOrigin(rec.p, -rec.normal), dir);
                t_total = t_total + rec.t;
                continue;
            }
            return RColor3f::zero();
        }
        return RColor3f(float(transmission.x), float(transmission.y), float(transmission.z));
    }

    /**
     * @brief One-sample next event estimation at a bounce vertex. Returns the
     *        per-channel radiance contribution; caller scales by throughput
     *        and applies the firefly clamp.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _evalNeeAtBounce(
        Kernel* k,
        const Vec3C& hit_position,
        const Vec3C& normal,
        const Vec3C& view_dir,
        RSE_VisionMaterialType material_type,
        const Vec3C& albedo,
        FloatC metalness,
        FloatC roughness,
        device::RS_RandomState& rs_in_out
    ) {
        auto& ctx = k->ctx;
        if (ctx.light_cdf.size() == 0) return Vec3C::zero();

        FloatC rand_u = rs_in_out.template rand<FloatC>();
        RIndex light_index = _sampleLight(k, rand_u);
        FloatC source_pdf = _lightSourcePdf(k, light_index);
        if (source_pdf <= FloatC(0)) return Vec3C::zero();

        LightSample light_sample = _sampleLightSample(k, light_index, hit_position, rs_in_out);
        Vec3C brdf_cos = _brdfCos(
            material_type,
            albedo,
            metalness,
            roughness,
            PBR_DEFAULT_IOR,
            normal,
            view_dir,
            light_sample.dir
        );
        Vec3C contribution =
            brdf_cos.elemMul(light_sample.emission) * _lightGeometryFactor(light_sample);
        if (_luminance(contribution) <= FloatC(RS_EPS)) return Vec3C::zero();

        FloatC max_distance = light_sample.is_sun ? FloatC(RS_MAX_FLOAT)
                                                  : light_sample.dist - FloatC(RS_EPS);
        RHandle light_handle = (light_index == RS_VISION_SUN_LIGHT_INDEX)
                                   ? RS_NULL_HANDLE
                                   : ctx.structure_emissive_voxel_handles[light_index - 1];
        Vec3C origin = _offsetRayOrigin(hit_position, normal);
        RColor3f transmission =
            _shadowWalkWithTransmission(k, origin, light_sample.dir, max_distance, light_handle);
        FloatC p_nee_picked = _evalNeePdf(k, light_index, light_sample);
        FloatC p_brdf_picked = _brdfPdf(
            material_type,
            albedo,
            metalness,
            roughness,
            PBR_DEFAULT_IOR,
            normal,
            view_dir,
            light_sample.dir
        );
        FloatC mis_nee = _powerHeuristicMIS(p_nee_picked, p_brdf_picked);
        return contribution.elemMul(Vec3C(transmission.toVec3D())) * (mis_nee / source_pdf);
    }

    /*****************************************************************
     * Shade
     *****************************************************************/

    /**
     * @brief Indirect-only firefly clamp: caps per-sample contribution after the primary hit.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _fireflyClamp(Kernel* k, const Vec3C& contrib, RSize depth) {
        if (depth == 0) return contrib;
        FloatC cap = FloatC(k->vision_trace_firefly_cap);
        FloatC m = RS_MAX(contrib.x, RS_MAX(contrib.y, contrib.z));
        if (m > cap) return contrib * (cap / m);
        return contrib;
    }

    /**
     * @brief Gather (depth-1) additional BRDF-sampled bounces of indirect
     *        radiance starting from a Lambertian first-bounce hit, returning
     *        the contribution that adds onto the direct NEE at that hit.
     *        Implements the multi-bounce extension to ReSTIR GI's initial
     *        sampling described in Ouyang 2021 Sec 5.
     *
     * @details Iterative path tracer (NEE + BRDF MIS at each interior vertex,
     *          BRDF-MIS-weighted sky/emissive contributions at the path end).
     *          The Lambertian estimator collapses brdf*cos/pdf to `albedo`, so
     *          the throughput accumulates as a product of surface albedos.
     *          `depth = 1` reproduces the legacy single-bounce path; `depth =
     *          2` is the paper's two-bounce GI; deeper extends naturally.
     *
     * @param k             simulation kernel.
     * @param p             first-bounce hit position.
     * @param normal        first-bounce surface normal.
     * @param albedo        first-bounce surface albedo.
     * @param metalness     first-bounce material metalness.
     * @param roughness     first-bounce material roughness.
     * @param incoming_dir  world-space traveling direction of the ray that
     *                      arrived at `p` (so `V = -incoming_dir`). Must NOT
     *                      be the surface normal — passing the normal collapses
     *                      the spec lobe to a perpendicular reflection.
     * @param path_depth    maximum number of bounces in the path.
     * @param rs_in_out     random state advanced by BRDF samples + NEEs.
     * @return indirect radiance summed across depth-2..depth-N bounces.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _gatherIndirectPath(
        Kernel* k,
        const Vec3C& p,
        const Vec3C& normal,
        const Vec3C& albedo,
        FloatC metalness,
        FloatC roughness,
        const Vec3C& incoming_dir,
        RSize path_depth,
        device::RS_RandomState& rs_in_out
    ) {
        if (path_depth < RSize(2)) return Vec3C::zero();

        Vec3C cur_p = p;
        Vec3C cur_normal = normal;
        Vec3C cur_albedo = albedo;
        FloatC cur_metalness = metalness;
        FloatC cur_roughness = roughness;
        RSE_VisionMaterialType cur_mt = RSE_VISION_MATERIAL_OPAQUE;
        Vec3C throughput = Vec3C::one();
        Vec3C indirect = Vec3C::zero();

        // Traveling direction of the ray that arrived at the current vertex;
        // updated to the outgoing scatter direction after each bounce.
        Vec3C cur_incoming_dir = incoming_dir;
        for (RSize d = RSize(1); d < path_depth; ++d) {
            HitRecord scatter_rec;
            scatter_rec.p = cur_p;
            scatter_rec.normal = cur_normal;
            scatter_rec.albedo = cur_albedo;
            scatter_rec.roughness = cur_roughness;
            scatter_rec.metalness = cur_metalness;
            scatter_rec.ior = PBR_DEFAULT_IOR;
            Ray incoming_ray(_offsetRayOrigin(cur_p, cur_normal), cur_incoming_dir);
            _scatterOpaque(incoming_ray, rs_in_out, scatter_rec);
            Vec3C bounce_dir = scatter_rec.scattered.ray_direction;
            FloatC brdf_pdf = _brdfPdf(
                cur_mt,
                cur_albedo,
                cur_metalness,
                cur_roughness,
                PBR_DEFAULT_IOR,
                cur_normal,
                -cur_incoming_dir,
                bounce_dir
            );
            if (brdf_pdf <= FloatC(RS_EPS)) break;

            // _scatterOpaque sets rec.color = f * NoL / pdf — the path throughput
            // contribution from this vertex. Multiply into running throughput.
            throughput = throughput.elemMul(scatter_rec.color);
            cur_incoming_dir = bounce_dir;

            // Trace next bounce.
            Ray bounce_ray(_offsetRayOrigin(cur_p, cur_normal), bounce_dir);
            HitRecord bounce_rec;
            bool any_hit = _findHit(k, bounce_ray, bounce_rec);

            if (!any_hit) {
                // BRDF-into-sky: MIS against sun-cone NEE pdf so the sun disc
                // does not get double-counted when this direction lands in it.
                Vec3C sky = _findHitBackground(k, bounce_dir);
                Vec3C sun_disk = _evalCosmicBackgroundSunDisk(k, bounce_dir);
                if (_luminance(sun_disk) > FloatC(0)) {
                    FloatC p_nee = _evalNeePdfSun(k, bounce_dir);
                    FloatC mis_brdf = _powerHeuristicMIS(brdf_pdf, p_nee);
                    sky = sky + sun_disk * (mis_brdf - FloatC(1));
                }
                indirect = indirect + throughput.elemMul(sky);
                break;
            }
            auto& bhs = bounce_rec.state.hit_state;
            RSE_VisionMaterialType bounce_mt =
                static_cast<RSE_VisionMaterialType>(bhs.vision_material_type);
            if (bounce_mt == RSE_VISION_MATERIAL_EMISSIVE) {
                // BRDF-into-emissive: MIS against the voxel-pick NEE pdf.
                Vec3C emission = bounce_rec.albedo * bounce_rec.intensity;
                FloatC p_nee = _evalNeePdfVoxel(k, cur_p, bounce_dir, bounce_rec.voxel);
                FloatC mis_brdf = p_nee > FloatC(0) ? _powerHeuristicMIS(brdf_pdf, p_nee)
                                                    : FloatC(1);
                indirect = indirect + throughput.elemMul(emission * mis_brdf);
                break;
            }
            if (bounce_mt != RSE_VISION_MATERIAL_OPAQUE) {
                // Transparent vertex skipped (no inner chain in this version).
                break;
            }

            // NEE at this opaque vertex contributes for the current depth.
            if (k->vision_restir_gi_is_bounce_nee_enabled) {
                Vec3C nee = _evalNeeAtBounce(
                    k,
                    bounce_rec.p,
                    bounce_rec.normal,
                    -bounce_dir,
                    bounce_mt,
                    bounce_rec.albedo,
                    bounce_rec.metalness,
                    bounce_rec.roughness,
                    rs_in_out
                );
                indirect = indirect + throughput.elemMul(nee);
            }

            // Advance the path; loop continues for deeper bounces.
            cur_p = bounce_rec.p;
            cur_normal = bounce_rec.normal;
            cur_albedo = bounce_rec.albedo;
            cur_metalness = bounce_rec.metalness;
            cur_roughness = bounce_rec.roughness;
            cur_mt = bounce_mt;
        }
        return indirect;
    }

    /**
     * @brief Walk a deterministic specular sub-chain starting from a glass
     *        surface's reflect direction (mini-PTUD on the reflect side while
     *        the main chain follows refract). Terminates at cosmic background / emissive /
     *        diffuse / chain budget exhaustion. Returns
     *        `F * sub_throughput * L_endpoint` — the radiance arriving back at
     *        the glass surface, already Fresnel-weighted, ready for the caller
     *        to multiply by its own running chain throughput and accumulate.
     * @note This is not a "standard" practice, its for ReSTIR glass rendering.
     *
     * @details Endpoint shading is one-shot NEE at a diffuse surface, raw
     *          emission at an emissive, or sky background on miss. No GI
     *          reservoir is used on this side. Inner glass surfaces along
     *          the sub-chain take refract only (no recursive fan) — a small
     *          bias on glass-behind-glass reflections, traded for bounded
     *          linear cost.
     *
     * @param k                   simulation kernel.
     * @param glass_p             glass-surface hit position.
     * @param glass_normal        glass-surface outward normal.
     * @param glass_incoming_dir  traveling direction of the main-chain ray
     *                            arriving at the glass surface (so
     *                            `V = -glass_incoming_dir`).
     * @param glass_roughness     glass-surface roughness, perturbs the entry
     *                            reflect direction inside a cone.
     * @param ior_from            medium IOR on the main chain's incoming side.
     * @param ior_to              medium IOR on the refracted side (used by
     *                            Fresnel only; the reflect ray itself stays
     *                            in `ior_from`).
     * @param rs_in_out           RNG state advanced by entry-perturb + sub-chain.
     * @return Fresnel-weighted radiance arriving at the glass via the reflect
     *         direction, or zero on TIR-with-F=0 / budget exhaustion.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _gatherReflectChain(
        Kernel* k,
        const Vec3C& glass_p,
        const Vec3C& glass_normal,
        const Vec3C& glass_incoming_dir,
        FloatC glass_roughness,
        FloatC ior_from,
        FloatC ior_to,
        device::RS_RandomState& rs_in_out
    ) {
        Vec3C unit_dir = glass_incoming_dir.normalized();
        FloatC cos_theta = RS_MAX(FloatC(0), RS_MIN(-unit_dir.dot(glass_normal), FloatC(1)));
        FloatC eta = ior_from / ior_to;
        FloatC sin2_t = eta * eta * (FloatC(1) - cos_theta * cos_theta);
        bool cannot_refract = sin2_t > FloatC(1);
        FloatC F = cannot_refract ? FloatC(1) : _fresnelDielectric(cos_theta, ior_to / ior_from);
        if (F <= FloatC(RS_EPS)) return Vec3C::zero();

        Vec3C reflect_dir =
            (unit_dir - glass_normal * (FloatC(2) * unit_dir.dot(glass_normal))).normalized();
        if (glass_roughness > FloatC(RS_EPS)) {
            Vec3C u_axis, v_axis;
            _buildONB(reflect_dir, u_axis, v_axis);
            FloatC r1 = rs_in_out.template rand<FloatC>();
            FloatC r2 = rs_in_out.template rand<FloatC>();
            FloatC phi = RS_2PI * r1;
            FloatC sin_pt = RS_FI::rs_sqrt(r2) * glass_roughness;
            FloatC cos_pt = RS_FI::rs_sqrt(RS_MAX(FloatC(0), FloatC(1) - sin_pt * sin_pt));
            FloatC sin_phi, cos_phi;
            RS_FI::rs_sincos(phi, &sin_phi, &cos_phi);
            reflect_dir =
                (u_axis * (cos_phi * sin_pt) + v_axis * (sin_phi * sin_pt) + reflect_dir * cos_pt)
                    .normalized();
        }
        Ray cur_ray(_offsetRayOrigin(glass_p, glass_normal), reflect_dir);
        Vec3C sub_throughput = Vec3C::one();
        FloatC sub_medium_ior = ior_from;
        RSize max_chain = k->vision_restir_max_specular_chain;

        for (RSize step = 0; step < max_chain; ++step) {
            HitRecord sub_rec;
            bool sub_hit = _findHit(k, cur_ray, sub_rec);

            if (!sub_hit) {
                Vec3C sky = _findHitBackground(k, cur_ray.ray_direction);
                return sub_throughput.elemMul(sky) * F;
            }

            auto& shs = sub_rec.state.hit_state;
            RSE_VisionMaterialType smt =
                static_cast<RSE_VisionMaterialType>(shs.vision_material_type);

            if (smt == RSE_VISION_MATERIAL_EMISSIVE) {
                Vec3C emission = sub_rec.albedo * sub_rec.intensity;
                return sub_throughput.elemMul(emission) * F;
            }

            if (!_isSpecularSurface(smt, sub_rec.roughness, sub_rec.metalness)) {
                Vec3C view_dir = -cur_ray.ray_direction.normalized();
                Vec3C nee = _evalNeeAtBounce(
                    k,
                    sub_rec.p,
                    sub_rec.normal,
                    view_dir,
                    smt,
                    sub_rec.albedo,
                    sub_rec.metalness,
                    sub_rec.roughness,
                    rs_in_out
                );
                return sub_throughput.elemMul(nee) * F;
            }

            // Specular continuation: scatter + accumulate throughput.
            if (smt == RSE_VISION_MATERIAL_TRANSPARENT) {
                FloatC ior_to_sub = shs.is_front_face ? sub_rec.ior : FloatC(1);
                _scatterTransparentWithIORDeterministic(
                    cur_ray, sub_medium_ior, ior_to_sub, rs_in_out, sub_rec
                );
                if (shs.is_refraction) sub_medium_ior = ior_to_sub;
            } else {
                _scatterMirrorDelta(cur_ray, sub_rec);
            }
            sub_throughput = sub_throughput.elemMul(sub_rec.color);
            cur_ray = sub_rec.scattered;
        }
        return Vec3C::zero();
    }

    /**
     * @brief Re-trace the stored sample's bounce ray from the current endpoint
     *        and report whether the hit geometry still matches what the
     *        reservoir stored. Used by sample validation (Ouyang 2021 Sec 4.3).
     *
     * @details Sky bounces are valid iff the re-traced ray also misses.
     *          Surface bounces are valid iff the new hit position is within
     *          `position_threshold` of the stored bounce hit. Does NOT
     *          re-evaluate radiance; only catches geometric staleness.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _validateReservoirGISample(
        Kernel* k,
        const Vec3C& endpoint_position,
        const Vec3C& endpoint_normal,
        const RS_ReservoirGI& reservoir,
        FloatC position_threshold
    ) {
        if (reservoir.sample_count == 0) return true;
        Vec3C stored_dir = Vec3C(reservoir.bounce_direction);
        Vec3C stored_hit = Vec3C(reservoir.bounce_hit_position);
        bool stored_is_sky = stored_hit.x == FloatC(0) && stored_hit.y == FloatC(0) &&
                             stored_hit.z == FloatC(0);

        Ray validate_ray(_offsetRayOrigin(endpoint_position, endpoint_normal), stored_dir);
        HitRecord validate_rec;
        bool validate_any_hit = _findHit(k, validate_ray, validate_rec);
        if (stored_is_sky) return !validate_any_hit;
        if (!validate_any_hit) return false;
        return (stored_hit - validate_rec.p).length() < position_threshold;
    }

    /**
     * @brief Convert a ReSTIR DI reservoir into the per-pixel direct-lighting
     *        contribution at the shading point: re-evaluates the stored light
     *        sample under current shading, multiplies in the WRS resampling
     *        weight, survivor's transmission, and NEE-side MIS weight. Returns
     *        zero if the reservoir is empty or degenerate. Caller multiplies
     *        by throughput.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _evalReservoirDI(
        Kernel* k,
        const RS_ReservoirDI& reservoir,
        RSE_VisionMaterialType material_type,
        const Vec3C& hit_position,
        const Vec3C& normal,
        const Vec3C& albedo,
        FloatC metalness,
        FloatC roughness,
        const Vec3C& view_dir
    ) {
        if (reservoir.sample_count == 0) return Vec3C::zero();
        if (FloatC(reservoir.target_pdf) <= FloatC(RS_EPS)) return Vec3C::zero();

        LightSample light_sample = _evalStoredLightSample(
            k, reservoir.light_index, hit_position, reservoir.light_position
        );
        Vec3C brdf_cos = _brdfCos(
            material_type,
            albedo,
            metalness,
            roughness,
            PBR_DEFAULT_IOR,
            normal,
            view_dir,
            light_sample.dir
        );
        Vec3C contribution =
            brdf_cos.elemMul(light_sample.emission) * _lightGeometryFactor(light_sample);
        Vec3C transmission = Vec3C(reservoir.transmission.toVec3D());
        Vec3C contribution_vis = contribution.elemMul(transmission);
        // Re-evaluate p̂(y) at read with current shading + stored transmission,
        // matching the contribution_vis we hand back. Using the cached
        // reservoir.target_pdf instead would leave a residual bias factor of
        // (current p̂) / (storage p̂) on temporal- or spatial-fold survivors —
        // visible as the 1/transmission spike when a partial-shadow stored
        // survivor is read at a fully-visible current pixel. With this re-eval
        // the final contribution collapses to (unit color direction of
        // contribution_vis) × weight_sum / sample_count × MIS, bounded by the
        // mean per-sample weight.
        FloatC target_pdf_at_read = _luminance(contribution_vis);
        if (target_pdf_at_read <= FloatC(RS_EPS)) return Vec3C::zero();
        FloatC restir_weight =
            FloatC(reservoir.weight_sum) / (FloatC(reservoir.sample_count) * target_pdf_at_read);
        FloatC p_nee_survivor = _evalNeePdf(k, reservoir.light_index, light_sample);
        FloatC p_brdf_survivor = _brdfPdf(
            material_type,
            albedo,
            metalness,
            roughness,
            PBR_DEFAULT_IOR,
            normal,
            view_dir,
            light_sample.dir
        );
        FloatC mis_nee = _powerHeuristicMIS(p_nee_survivor, p_brdf_survivor);
        return contribution_vis * (restir_weight * mis_nee);
    }

    /**
     * @brief Per-pixel one-bounce indirect contribution from a GI reservoir.
     *        Re-evaluates `brdf_cos · l_incoming` at the stored bounce
     *        direction and folds in the WRS resampling weight.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _evalReservoirGI(
        Kernel* /*k*/,
        const RS_ReservoirGI& reservoir,
        RSE_VisionMaterialType material_type,
        const Vec3C& normal,
        const Vec3C& albedo,
        FloatC metalness,
        FloatC roughness,
        const Vec3C& view_dir
    ) {
        if (reservoir.sample_count == 0) return Vec3C::zero();
        if (FloatC(reservoir.target_pdf) <= FloatC(RS_EPS)) return Vec3C::zero();

        Vec3C bounce_dir = Vec3C(reservoir.bounce_direction);
        Vec3C brdf_cos = _brdfCos(
            material_type,
            albedo,
            metalness,
            roughness,
            PBR_DEFAULT_IOR,
            normal,
            view_dir,
            bounce_dir
        );
        Vec3C l_incoming = Vec3C(reservoir.l_incoming.toVec3D());
        Vec3C contribution = brdf_cos.elemMul(l_incoming);
        // Re-evaluate p̂(y) at read time with current shading; using the cached
        // reservoir.target_pdf would leave a (current p̂)/(storage p̂) bias on
        // temporal- and spatial-fold survivors whose brdf_cos at storage
        // differed from ours. Mirrors the same fix in _evalReservoirDI.
        FloatC target_pdf_at_read = _luminance(contribution);
        if (target_pdf_at_read <= FloatC(RS_EPS)) return Vec3C::zero();
        // Stored contributing-source count when set, else raw sample count.
        RSize normalizer = (reservoir.Z_count > 0) ? reservoir.Z_count : reservoir.sample_count;
        FloatC restir_weight =
            FloatC(reservoir.weight_sum) / (FloatC(normalizer) * target_pdf_at_read);
        return contribution * restir_weight;
    }

    /**
     * @brief Basic path-tracer one-depth-step shader. No reservoir read; one
     *        NEE sample per opaque bounce vertex via `_evalNeeAtBounce`. Used
     *        when `vision_is_restir_enabled == false`.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _shadeRGBOnceBasic(
        Kernel* k,
        RSize depth,
        bool any_hit,
        device::RS_RandomState& rs_in_out,
        Ray& current_ray_in_out,
        HitRecord& rec_in_out,
        Vec3C& throughput_in_out,
        FloatC& current_medium_in_out,
        FloatC& last_bounce_brdf_pdf_in_out,
        Vec3C& contribution_out,
        bool& is_continue_out
    ) {
        contribution_out = Vec3C::zero();
        is_continue_out = false;
        if (k == nullptr) return;

        if (!any_hit) {
            Vec3C dir = current_ray_in_out.ray_direction;
            Vec3C sky = _findHitBackground(k, dir);
            // MIS-weight the sun-disc contribution only; smooth sky has no NEE
            // partner and passes through with weight 1.
            if (last_bounce_brdf_pdf_in_out < MIS_DELTA_BRDF_PDF) {
                Vec3C sun_disk = _evalCosmicBackgroundSunDisk(k, dir);
                if (_luminance(sun_disk) > FloatC(0)) {
                    FloatC p_nee = _evalNeePdfSun(k, dir);
                    FloatC mis_brdf = _powerHeuristicMIS(last_bounce_brdf_pdf_in_out, p_nee);
                    sky = sky + sun_disk * (mis_brdf - FloatC(1));
                }
            }
            contribution_out = _fireflyClamp(k, throughput_in_out.elemMul(sky), depth);
            return;
        }

        bool continue_tracing = true;
        auto& hs = rec_in_out.state.hit_state;
        RSE_VisionMaterialType material_type =
            static_cast<RSE_VisionMaterialType>(hs.vision_material_type);

        if (material_type == RSE_VISION_MATERIAL_OPAQUE) {
            Vec3C view_dir = -current_ray_in_out.ray_direction;
            Vec3C nee = _evalNeeAtBounce(
                k,
                rec_in_out.p,
                rec_in_out.normal,
                view_dir,
                material_type,
                rec_in_out.albedo,
                rec_in_out.metalness,
                rec_in_out.roughness,
                rs_in_out
            );
            if (_luminance(nee) > FloatC(0)) {
                contribution_out =
                    contribution_out + _fireflyClamp(k, throughput_in_out.elemMul(nee), depth);
            }
        }

        if (material_type == RSE_VISION_MATERIAL_EMISSIVE) {
            contribution_out = _fireflyClamp(
                k, throughput_in_out.elemMul(rec_in_out.albedo * rec_in_out.intensity), depth
            );
            continue_tracing = false;
        } else if (material_type == RSE_VISION_MATERIAL_TRANSPARENT) {
            FloatC ior_to = hs.is_front_face ? rec_in_out.ior : FloatC(1);
            _scatterTransparentWithIOR(
                current_ray_in_out, current_medium_in_out, ior_to, rs_in_out, rec_in_out
            );
            if (hs.is_refraction) current_medium_in_out = ior_to;
        } else {
            _scatterOpaque(current_ray_in_out, rs_in_out, rec_in_out);
        }

        if (!continue_tracing) return;

        throughput_in_out = throughput_in_out.elemMul(rec_in_out.color);

        if (depth >= k->vision_trace_russian_roulette_start_depth) {
            FloatC p_cont = RS_MIN(
                RR_MAX_SURVIVAL_PROB,
                RS_MAX(throughput_in_out.x, RS_MAX(throughput_in_out.y, throughput_in_out.z))
            );
            if (rs_in_out.template rand<FloatC>() > p_cont) return;
            throughput_in_out = throughput_in_out * (FloatC(1) / RS_MAX(p_cont, RR_MIN_PROB));
        }

        // BRDF pdf at the scattered direction, for the next iteration's MIS
        // partner on bounce-into-sky / bounce-into-emissive. Mirror / glass /
        // emissive surfaces leave the sentinel in place so the next bounce
        // sees the previous step as a delta.
        Vec3C view_dir = -current_ray_in_out.ray_direction;
        last_bounce_brdf_pdf_in_out = _brdfPdf(
            material_type,
            rec_in_out.albedo,
            rec_in_out.metalness,
            rec_in_out.roughness,
            rec_in_out.ior,
            rec_in_out.normal,
            view_dir,
            rec_in_out.scattered.ray_direction
        );

        current_ray_in_out = rec_in_out.scattered;
        is_continue_out = true;
    }

    /**
     * @brief Walk the deterministic specular chain (PTUD) from a primary hit
     *        until it lands on diffuse, cosmic background, or emissive (or runs out of the
     *        budget), accumulating Beer-Lambert / Fresnel throughput along the
     *        way. Caller hands in the primary HitRecord; gets back the
     *        endpoint shading record + accumulated throughput + terminal type
     *        + off-chain reflect radiance.
     *
     * @details For non-specular primaries (or a miss) the function returns
     *          immediately with endpoint = primary, throughput = (1,1,1).
     *          When the chain ends in COSMIC_BACKGROUND, `endpoint_rec_out.p` holds the
     *          terminal ray direction (not a world point); callers writing
     *          it into the endpoint_hit_position AOV treat it as a direction
     *          based on the terminal_type tag.
     *
     *          At each glass surface along the chain, the main chain follows
     *          refract deterministically; the reflect side is handled by
     *          `_gatherReflectChain` and its Fresnel-weighted contribution is
     *          summed into `chain_reflect_radiance_out`. Caller atomic-adds
     *          this off-chain radiance to the canvas; the on-chain endpoint
     *          contribution comes later from the reservoir read in the RGB
     *          pass.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _shadeInfoOnceRestir(
        Kernel* k,
        const Ray& primary_ray,
        bool primary_any_hit,
        const HitRecord& rec,
        device::RS_RandomState& rs_in_out,
        HitRecord& endpoint_rec_out,
        Vec3C& chain_throughput_out,
        Vec3C& chain_reflect_radiance_out,
        RSE_VisionRestirChainTerminalType& terminal_type_out
    ) {
        chain_throughput_out = Vec3C::one();
        chain_reflect_radiance_out = Vec3C::zero();
        if (!primary_any_hit) {
            endpoint_rec_out = HitRecord{};
            endpoint_rec_out.p = primary_ray.ray_direction;
            terminal_type_out = RSE_VISION_RESTIR_CHAIN_TERMINAL_COSMIC_BACKGROUND;
            return;
        }

        HitRecord cur_rec = rec;
        Ray cur_ray = primary_ray;
        FloatC medium_ior = FloatC(1);
        RSize max_chain = k->vision_restir_max_specular_chain;
        for (RSize chain_depth = 0; chain_depth < max_chain; ++chain_depth) {
            auto& hs = cur_rec.state.hit_state;
            RSE_VisionMaterialType mt =
                static_cast<RSE_VisionMaterialType>(hs.vision_material_type);

            if (mt == RSE_VISION_MATERIAL_EMISSIVE) {
                endpoint_rec_out = cur_rec;
                terminal_type_out = RSE_VISION_RESTIR_CHAIN_TERMINAL_EMISSIVE;
                return;
            }
            if (!_isSpecularSurface(mt, cur_rec.roughness, cur_rec.metalness)) {
                endpoint_rec_out = cur_rec;
                terminal_type_out = RSE_VISION_RESTIR_CHAIN_TERMINAL_SURFACE;
                return;
            }

            // Specular hit: scatter + accumulate Beer-Lambert / Fresnel throughput.
            if (mt == RSE_VISION_MATERIAL_TRANSPARENT) {
                FloatC ior_to = hs.is_front_face ? cur_rec.ior : FloatC(1);
                // Reflect side: mini-PTUD off the glass, Fresnel-weighted, added
                // to off-chain radiance. Refract side continues the main chain.
                Vec3C reflect_contrib = _gatherReflectChain(
                    k,
                    cur_rec.p,
                    cur_rec.normal,
                    cur_ray.ray_direction,
                    cur_rec.roughness,
                    medium_ior,
                    ior_to,
                    rs_in_out
                );
                chain_reflect_radiance_out =
                    chain_reflect_radiance_out + chain_throughput_out.elemMul(reflect_contrib);
                _scatterTransparentWithIORDeterministic(
                    cur_ray, medium_ior, ior_to, rs_in_out, cur_rec
                );
                if (hs.is_refraction) medium_ior = ior_to;
            } else {
                _scatterMirrorDelta(cur_ray, cur_rec);
            }
            chain_throughput_out = chain_throughput_out.elemMul(cur_rec.color);
            cur_ray = cur_rec.scattered;

            if (!_findHit(k, cur_ray, cur_rec)) {
                endpoint_rec_out = HitRecord{};
                endpoint_rec_out.p = cur_ray.ray_direction;
                terminal_type_out = RSE_VISION_RESTIR_CHAIN_TERMINAL_COSMIC_BACKGROUND;
                return;
            }
        }
        // Budget exhausted: treat the last hit as a surface endpoint (slight
        // darkening on hall-of-mirrors corners).
        endpoint_rec_out = cur_rec;
        terminal_type_out = RSE_VISION_RESTIR_CHAIN_TERMINAL_SURFACE;
    }

    /**
     * @brief ReSTIR-mode flat shader: dispatches on the chain terminal type
     *        and reads endpoint AOVs + reservoirs. Used when
     *        `vision_is_restir_enabled == true`.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    _shadeRGBOnceRestir(Kernel* k, RHandle camera, RIndex ii, RIndex jj, Vec3C& contribution_out) {
        contribution_out = Vec3C::zero();
        if (k == nullptr) return;
        auto& ctx = k->ctx;

        RSE_VisionRestirChainTerminalType terminal_type =
            ctx.camera_aux_chain_terminal_type[camera][{ii, jj}];
        Vec3C throughput = Vec3C(ctx.camera_aux_chain_throughput[camera][{ii, jj}]);

        if (terminal_type == RSE_VISION_RESTIR_CHAIN_TERMINAL_COSMIC_BACKGROUND) {
            Vec3C sky_dir = Vec3C(ctx.camera_aux_endpoint_hit_position[camera][{ii, jj}]);
            contribution_out = throughput.elemMul(_findHitBackground(k, sky_dir));
            return;
        }

        if (terminal_type == RSE_VISION_RESTIR_CHAIN_TERMINAL_EMISSIVE) {
            Vec3C albedo = Vec3C(ctx.camera_aux_endpoint_albedo[camera][{ii, jj}]);
            FloatC intensity = FloatC(ctx.camera_aux_endpoint_intensity[camera][{ii, jj}]);
            contribution_out = throughput.elemMul(albedo * intensity);
            return;
        }

        // SURFACE terminal: DI + GI reservoirs evaluated at the endpoint.
        Vec3C hit_position = Vec3C(ctx.camera_aux_endpoint_hit_position[camera][{ii, jj}]);
        Vec3C normal = Vec3C(ctx.camera_aux_endpoint_normal[camera][{ii, jj}]);
        Vec3C albedo = Vec3C(ctx.camera_aux_endpoint_albedo[camera][{ii, jj}]);
        FloatC roughness = FloatC(ctx.camera_aux_endpoint_roughness[camera][{ii, jj}]);
        FloatC metalness = FloatC(ctx.camera_aux_endpoint_metalness[camera][{ii, jj}]);
        RSE_VisionMaterialType material_type =
            ctx.camera_aux_endpoint_material_type[camera][{ii, jj}];

        // Endpoint view dir approximated as camera-relative (matches resample).
        Vec3C camera_position = Vec3C(CA_G(position));
        Vec3C view_dir = (camera_position - hit_position).normalized();

        // Shading reads the spatial buffer, the final per-pixel reservoir.
        const RS_ReservoirDI& di_reservoir = ctx.camera_spatial_reservoir_di[camera][{ii, jj}];
        const RS_ReservoirGI& gi_reservoir = ctx.camera_spatial_reservoir_gi[camera][{ii, jj}];
        Vec3C direct = _evalReservoirDI(
            k,
            di_reservoir,
            material_type,
            hit_position,
            normal,
            albedo,
            metalness,
            roughness,
            view_dir
        );
        Vec3C indirect = _evalReservoirGI(
            k, gi_reservoir, material_type, normal, albedo, metalness, roughness, view_dir
        );
        // Cap multi-bounce escape spikes at concave corners.
        indirect = _fireflyClamp(k, indirect, RSize(1));
        contribution_out = throughput.elemMul(direct + indirect);
    }

    /*****************************************************************
     * Motion / temporal helpers
     *****************************************************************/

    /**
     * @brief Project a world point through any camera pose + intrinsics into
     *        fractional pixel coordinates. Returns false if the point is
     *        behind the camera.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _projectToPixel(
        Kernel* k,
        RHandle camera,
        const Vec3C& world_point,
        const Vec3C& camera_position,
        const Quat3C& camera_orientation,
        FloatC& i_out,
        FloatC& j_out
    ) {
        auto& ctx = k->ctx;
        constexpr FloatC PROJECTION_FORWARD_EPS = FloatC(1e-4);
        RSize image_width = CA_G(image_width);
        RSize image_height = CA_G(image_height);

        Vec3C unit_lookat = camera_orientation.toOrthoAxisX();
        Vec3C unit_up = camera_orientation.toOrthoAxisZ();
        Vec3C unit_right = -camera_orientation.toOrthoAxisY();

        Vec3C offset = world_point - camera_position;
        FloatC forward = offset.dot(unit_lookat);
        if (forward <= PROJECTION_FORWARD_EPS) return false;

        RS_CameraStateUnion state_union{CA_G(state)};
        FloatC pixel_delta_u = FloatC(0);
        FloatC pixel_delta_v = FloatC(0);
        if (state_union.camera_state.type == RSE_CAMERA_FREE_STEREO ||
            state_union.camera_state.type == RSE_CAMERA_RIGID_BODY_LINKED_STEREO) {
            FloatC tan_left = RS_FI::rs_tan(FloatC(CA_G(angle_left)));
            FloatC tan_right = RS_FI::rs_tan(FloatC(CA_G(angle_right)));
            FloatC tan_up = RS_FI::rs_tan(FloatC(CA_G(angle_up)));
            FloatC tan_down = RS_FI::rs_tan(FloatC(CA_G(angle_down)));
            FloatC x_tan = offset.dot(unit_right) / forward;
            FloatC y_tan = offset.dot(unit_up) / forward;
            pixel_delta_u = (x_tan - tan_left) / (tan_right - tan_left);
            pixel_delta_v = (y_tan - tan_up) / (tan_down - tan_up);
        } else {
            FloatC focus_distance = FloatC(CA_G(focus_distance));
            FloatC pixel_width = FloatC(CA_G(pixel_width));
            FloatC pixel_height = FloatC(CA_G(pixel_height));
            Vec3C focus_offset = offset * (focus_distance / forward);
            pixel_delta_u =
                focus_offset.dot(unit_right) / (pixel_width * static_cast<FloatC>(image_width)) +
                FloatC(0.5);
            pixel_delta_v =
                -focus_offset.dot(unit_up) / (pixel_height * static_cast<FloatC>(image_height)) +
                FloatC(0.5);
        }
        j_out = pixel_delta_u * static_cast<FloatC>(image_width) - FloatC(0.5);
        i_out = pixel_delta_v * static_cast<FloatC>(image_height) - FloatC(0.5);
        return true;
    }

    /**
     * @brief World point the hit surface sat at on the last vision update.
     *        Carries the hit through the voxel's recorded pose, so a body that
     *        rotated or was thrown about between renders still lands right. A
     *        hit on the floor or the cosmic background has no voxel and stays
     *        where it is, which is correct for geometry that cannot move.
     * @details The voxel's expansion and distortion are not recorded, so a hit
     *          inside a cell that deformed is off by a fraction of the voxel.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _computePrevHitPosition(
        Kernel* k,
        RHandle hit_voxel,
        const Vec3C& hit_position,
        Vec3C& prev_hit_position_out
    ) {
        auto& ctx = k->ctx;
        if (hit_voxel == RS_NULL_HANDLE) {
            prev_hit_position_out = hit_position;
            return;
        }
        RS_VoxelStateUnion state_union{V_G(hit_voxel, state)};
        if (state_union.voxel_state.is_static) {
            prev_hit_position_out = hit_position;
            return;
        }
        Quat3C orientation = Quat3C(V_G(hit_voxel, orientation));
        Vec3C position = Vec3C(V_G(hit_voxel, position));
        Vec3C voxel_offset = orientation.rotateVec3DInv(hit_position - position);
        prev_hit_position_out = Vec3C(V_G(hit_voxel, prev_position)) +
                                Quat3C(V_G(hit_voxel, prev_orientation)).rotateVec3D(voxel_offset);
    }

    /**
     * @brief Screen-space motion vector at a pixel. Projects the world point
     *        the surface sat at one frame back through the previous camera
     *        pose; the resulting offset is normalized by image dimensions so
     *        the denoiser and reservoir reprojection apply the same convention.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _computeScreenMotion(
        Kernel* k,
        RHandle camera,
        const Vec3C& prev_hit_position,
        RSize ii_curr,
        RSize jj_curr,
        RVec2f& screen_motion_out
    ) {
        auto& ctx = k->ctx;
        Vec3C prev_camera_position = Vec3C(CA_G(prev_position));
        Quat3C prev_camera_orientation = Quat3C(CA_G(prev_orientation));

        FloatC ii_prev_f, jj_prev_f;
        bool is_in_frustum = _projectToPixel(
            k,
            camera,
            prev_hit_position,
            prev_camera_position,
            prev_camera_orientation,
            ii_prev_f,
            jj_prev_f
        );
        if (!is_in_frustum) {
            screen_motion_out = RVec2f(0.0f, 0.0f);
            return;
        }
        RSize image_width = CA_G(image_width);
        RSize image_height = CA_G(image_height);
        FloatC motion_x = (FloatC(jj_curr) - jj_prev_f) / FloatC(image_width);
        FloatC motion_y = (FloatC(ii_curr) - ii_prev_f) / FloatC(image_height);
        screen_motion_out = RVec2f(float(motion_x), float(motion_y));
    }

    /*****************************************************************
     * Pure-math primitives
     *****************************************************************/

    /**
     * @brief Wachter and Binder self-intersection offset (Ray Tracing Gems vol. 1 ch. 6).
     *
     * @details Pure IEEE-754 float32 integer-bit arithmetic; offsets the hit point along
     * the normal by a few ULPs scaled to the magnitude of each axis, switching to a
     * fixed additive offset near the origin where the integer trick underflows. Robust
     * across the whole float32 range.
     *
     * @param p hit point in world space.
     * @param n outward surface normal at the hit; reverse it to bias through the surface.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _offsetRayOrigin(const Vec3C& p, const Vec3C& n) {
        constexpr float ORIGIN_THRESHOLD = 1.0f / 32.0f;
        constexpr float FLOAT_SCALE = 1.0f / 65536.0f;
        constexpr float INT_SCALE = 256.0f;

        float px = float(p.x), py = float(p.y), pz = float(p.z);
        float nx = float(n.x), ny = float(n.y), nz = float(n.z);

        int of_ix = int(INT_SCALE * nx);
        int of_iy = int(INT_SCALE * ny);
        int of_iz = int(INT_SCALE * nz);

        using RS_FI = RS_FloatIntrinsics<float>;
        int px_bits = RS_FI::rs_float_as_int(px);
        int py_bits = RS_FI::rs_float_as_int(py);
        int pz_bits = RS_FI::rs_float_as_int(pz);
        float p_ix = RS_FI::rs_int_as_float(px_bits + (px < 0.0f ? -of_ix : of_ix));
        float p_iy = RS_FI::rs_int_as_float(py_bits + (py < 0.0f ? -of_iy : of_iy));
        float p_iz = RS_FI::rs_int_as_float(pz_bits + (pz < 0.0f ? -of_iz : of_iz));

        float abs_px = px < 0.0f ? -px : px;
        float abs_py = py < 0.0f ? -py : py;
        float abs_pz = pz < 0.0f ? -pz : pz;
        float ox = abs_px < ORIGIN_THRESHOLD ? px + FLOAT_SCALE * nx : p_ix;
        float oy = abs_py < ORIGIN_THRESHOLD ? py + FLOAT_SCALE * ny : p_iy;
        float oz = abs_pz < ORIGIN_THRESHOLD ? pz + FLOAT_SCALE * nz : p_iz;
        return Vec3C(FloatC(ox), FloatC(oy), FloatC(oz));
    }

    /**
     * @brief Build an orthonormal basis whose w-axis is the input vector n.
     */
    HOST_DEVICE_INLINE_QUALIFIER static void _buildONB(const Vec3C& n, Vec3C& u, Vec3C& v) {
        Vec3C seed = (RS_ABS(n.z) < ONB_AXIS_ALIGN_THRESHOLD) ? Vec3C(0, 0, 1) : Vec3C(1, 0, 0);
        v = n.cross(seed).normalized();
        u = v.cross(n);
    }

    /**
     * @brief x^5 via two multiplications.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _pow5(FloatC x) {
        FloatC x2 = x * x;
        return x2 * x2 * x;
    }

    /**
     * @brief GGX / Trowbridge-Reitz normal distribution function.
     *        alpha = roughness^2 (caller pre-clamps roughness to PBR_MIN_ROUGHNESS).
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _ggxD(FloatC NoH, FloatC alpha) {
        FloatC a2 = alpha * alpha;
        FloatC denom = NoH * NoH * (a2 - FloatC(1)) + FloatC(1);
        return a2 / (FloatC(RS_PI) * denom * denom + FloatC(RS_EPS));
    }

    /**
     * @brief Smith G1 for GGX. Used inside the VNDF pdf and the BRDF pdf.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _ggxG1(FloatC NoV, FloatC alpha) {
        FloatC a2 = alpha * alpha;
        FloatC tmp = a2 + (FloatC(1) - a2) * NoV * NoV;
        return FloatC(2) * NoV / (NoV + RS_FI::rs_sqrt(RS_MAX(FloatC(RS_EPS), tmp)));
    }

    /**
     * @brief Height-correlated Smith G2 for GGX (Heitz 2014 JCGT eq. 99).
     *        More accurate than separable G1(V)*G1(L) at grazing angles where
     *        masking and shadowing become correlated through surface height.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC
    _ggxG2HeightCorrelated(FloatC NoV, FloatC NoL, FloatC alpha) {
        FloatC a2 = alpha * alpha;
        FloatC lambda_v = NoL * RS_FI::rs_sqrt(a2 + (FloatC(1) - a2) * NoV * NoV);
        FloatC lambda_l = NoV * RS_FI::rs_sqrt(a2 + (FloatC(1) - a2) * NoL * NoL);
        return FloatC(2) * NoV * NoL / (lambda_v + lambda_l + FloatC(RS_EPS));
    }

    /**
     * @brief Sample the GGX visible-normal distribution (Heitz 2018 JCGT).
     *        Returns a half-vector in shading space given view direction Ve
     *        (z = surface normal). Importance-sampling the visible NDF removes
     *        the grazing-direction variance and the need for rejection passes
     *        used by older isotropic GGX sampling routines.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _sampleGGXVNDF(const Vec3C& Ve, FloatC alpha, FloatC r1, FloatC r2) {
        Vec3C Vh = Vec3C(alpha * Ve.x, alpha * Ve.y, Ve.z).normalized();
        FloatC lensq = Vh.x * Vh.x + Vh.y * Vh.y;
        Vec3C T1 = lensq > FloatC(0) ? Vec3C(-Vh.y, Vh.x, FloatC(0)) / RS_FI::rs_sqrt(lensq)
                                     : Vec3C(FloatC(1), FloatC(0), FloatC(0));
        Vec3C T2 = Vh.cross(T1);
        FloatC r = RS_FI::rs_sqrt(r1);
        FloatC phi = FloatC(2) * FloatC(RS_PI) * r2;
        FloatC sin_phi, cos_phi;
        RS_FI::rs_sincos(phi, &sin_phi, &cos_phi);
        FloatC t1 = r * cos_phi;
        FloatC t2 = r * sin_phi;
        FloatC s = FloatC(0.5) * (FloatC(1) + Vh.z);
        t2 = (FloatC(1) - s) * RS_FI::rs_sqrt(RS_MAX(FloatC(0), FloatC(1) - t1 * t1)) + s * t2;
        Vec3C Nh = T1 * t1 + T2 * t2 +
                   Vh * RS_FI::rs_sqrt(RS_MAX(FloatC(0), FloatC(1) - t1 * t1 - t2 * t2));
        Vec3C Ne(alpha * Nh.x, alpha * Nh.y, RS_MAX(FloatC(0), Nh.z));
        return Ne.normalized();
    }

    /**
     * @brief Exact unpolarised Fresnel reflectance for a dielectric interface
     *        (Walter et al. 2007 eq. 22). Returns 1 on total internal reflection.
     *
     * @param cos_i  Cosine between the incoming direction and the (micro)
     *               surface normal — non-negative on the incoming hemisphere.
     * @param eta_r  Relative IOR `eta_t / eta_i` at this boundary.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _fresnelDielectric(FloatC cos_i, FloatC eta_r) {
        FloatC c = cos_i > FloatC(0) ? cos_i : -cos_i;
        FloatC g_sq = eta_r * eta_r - FloatC(1) + c * c;
        if (g_sq <= FloatC(0)) return FloatC(1);
        FloatC g = RS_FI::rs_sqrt(g_sq);
        FloatC g_minus_c = g - c;
        FloatC g_plus_c = g + c;
        FloatC term1 = g_minus_c / g_plus_c;
        term1 = FloatC(0.5) * term1 * term1;
        FloatC inner = (c * g_plus_c - FloatC(1)) / (c * g_minus_c + FloatC(1));
        return term1 * (FloatC(1) + inner * inner);
    }

    /**
     * @brief Schlick approximation of metallic Fresnel; vec3 F0 = albedo.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _fresnelSchlick(FloatC cos_i, const Vec3C& F0) {
        FloatC c = cos_i > FloatC(0) ? cos_i : -cos_i;
        return F0 + (Vec3C::one() - F0) * _pow5(FloatC(1) - c);
    }

    /**
     * @brief Per-direction Fresnel at the half-vector for the two-lobe opaque
     *        BSDF: dielectric + metal blended by metalness.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _fresnelOpaqueAtH(const Vec3C& albedo, FloatC metalness, FloatC ior, FloatC VoH) {
        FloatC F_h_diel = _fresnelDielectric(VoH, ior);
        Vec3C F_h_metal = _fresnelSchlick(VoH, albedo);
        return Vec3C(F_h_diel, F_h_diel, F_h_diel) * (FloatC(1) - metalness) +
               F_h_metal * metalness;
    }

    /**
     * @brief Veach 1997 §9.2 variance-optimal lobe-selection probability for
     *        the spec lobe in the two-lobe opaque BSDF. Compares directional
     *        albedos of spec (Fresnel at NoV) and diffuse ((1 - metalness) ×
     *        albedo) under CIE Y luminance. Clamped so neither lobe starves.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC
    _specProbOpaque(const Vec3C& albedo, FloatC metalness, FloatC ior, FloatC NoV) {
        FloatC F_NoV_diel = _fresnelDielectric(NoV, ior);
        Vec3C F_NoV_metal = _fresnelSchlick(NoV, albedo);
        Vec3C F_NoV = Vec3C(F_NoV_diel, F_NoV_diel, F_NoV_diel) * (FloatC(1) - metalness) +
                      F_NoV_metal * metalness;
        Vec3C rho_diff = albedo * (FloatC(1) - metalness);
        FloatC L_spec = RColor3f(F_NoV).luminance();
        FloatC L_diff = RColor3f(rho_diff).luminance();
        FloatC p = L_spec / RS_MAX(L_spec + L_diff, FloatC(RS_EPS));
        return RS_MAX(PBR_SPEC_PROB_MIN, RS_MIN(PBR_SPEC_PROB_MAX, p));
    }

    /*****************************************************************
     * Miscellaneous
     *****************************************************************/

    /**
     * @brief Pack the per-pixel object-identity key used for temporal reprojection and ReSTIR
     *        reservoir matching. Voxel: the structure entity gid with its vision material key.
     *        Floor: the reserved floor gid.
     */
    HOST_DEVICE_INLINE_QUALIFIER static RS_CameraPixelObjectKey _packObjectKey(
        const HitRecord& rec
    ) {
        RS_CameraPixelObjectKey key{};
        if (rec.state.hit_state.is_floor_hit) {
            key.entity_gid = RS_WORLD_RSC_FLOOR_GID;
            key.vision_material_key = 0;
        } else {
            key.entity_gid = rec.structure_gid;
            key.vision_material_key = rec.vision_material_key;
        }
        return key;
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_VISION_RAY_C_H
