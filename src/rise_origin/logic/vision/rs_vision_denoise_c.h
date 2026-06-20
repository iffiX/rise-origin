// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VISION_DENOISE_C_H
#define RS_VISION_DENOISE_C_H

#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/vision/rs_camera_m.h"
#include "rise_origin/logic/vision/rs_vision_denoise_m.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

namespace logic {

/**
 * @brief Per-camera denoiser kernels. Run after RSCF_TraceCameraRGBRays has
 *        accumulated noisy radiance into the read canvas; the executor wires
 *        ping-pong between camera_current_rgb_canvas and camera_next_rgb_canvas
 *        for the iteration count.
 *
 * @note Aux buffers (normal / depth_z / object_key / roughness) are written
 *       by RSCF_TraceCameraInfoRays once per camera per frame before the RGB
 *       pass, so every read here is on a stable snapshot.
 */
template <RSE_FloatType FloatComputeAccuracy = RSE_FLOAT_32>
class RS_VisionDenoiseCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;

    /**
     * @brief Edge-stopping a-trous wavelet pass with constant sigma_color.
     *        5x5 B-spline kernel with edge stops on (normal dot, depth gauss,
     *        luminance L1), iteration step doubling between passes. Per-pixel
     *        pass count is capped by ceil(roughness^2 * total_passes), floored
     *        at vision_denoise_atrous_min_passes, so mirrors get a single 5x5 pass and
     *        fully diffuse pixels run the full iteration count.
     *
     * @details Dammertz, H., Sewtz, D., Hanika, J. & Lensch, H. 2010.
     *          Edge-Avoiding A-Trous Wavelet Transform for Fast Global
     *          Illumination Filtering. HPG 2010, 67-75.
     *          https://jo.dreggn.org/home/2010_atrous.pdf
     *          Section 3 defines the 5x5 B-spline kernel + edge stops.
     * @details Zhdan, D. 2020. Fast Denoising with Self-Stabilizing Recurrent
     *          Blurs. NVIDIA GTC 2020 (S22699). Slide p.46 gives the
     *          specular lobe half-angle r^2 / (1 + r^2), here applied as a
     *          per-pixel pass-count cap.
     *
     * @param k             simulation kernel.
     * @param camera        target camera handle.
     * @param pixel         flat (i * W + j) pixel index.
     * @param read_canvas   per-camera input canvas vector.
     * @param write_canvas  per-camera output canvas vector.
     * @param pass_index    0-based à-trous pass index; step = 1 << pass_index.
     */
    class RSCF_AtrousFilter : public device::RS_ComputeFunction<RSCF_AtrousFilter> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel* k,
            RHandle camera,
            RHandle pixel,
            device::RS_Vector<RS_CameraRGBCanvas> read_canvas,
            device::RS_Vector<RS_CameraRGBCanvas> write_canvas,
            RSize pass_index
        ) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;
            if (k->vision_denoise_type != RSE_VISION_DENOISE_ATROUS) return;
            RSize total_passes = k->vision_denoise_atrous_iterations;
            if (pass_index >= total_passes) return;
            RSize w = CA_G(image_width);
            RSize h = CA_G(image_height);
            RSize i = RSize(pixel) / w;
            RSize j = RSize(pixel) - i * w;

            using RS_FI = RS_FloatIntrinsics<FloatC>;
            FloatC roughness_p = FloatC(ctx.camera_aux_roughness[camera][{i, j}]);
            FloatC m = roughness_p * roughness_p;
            RSize max_pass_p = RS_MAX(
                RSize(k->vision_denoise_atrous_min_passes),
                RSize(RS_FI::rs_ceil(m * FloatC(total_passes)))
            );
            if (pass_index >= max_pass_p) {
                write_canvas[camera][{i, j}] = read_canvas[camera][{i, j}];
                return;
            }

            int step = 1 << int(pass_index);

            Vec3C center_color = Vec3C(read_canvas[camera][{i, j}]);
            Vec3C center_normal = Vec3C(ctx.camera_aux_normal[camera][{i, j}]);
            FloatC center_depth = FloatC(ctx.camera_aux_depth_z[camera][{i, j}]);
            RS_CameraPixelObjectKey center_object_key = ctx.camera_aux_object_key[camera][{i, j}];
            // Material-based edge stop: compare kind tag + material_gid only.
            // Same material across different structures still smooths together.
            RIndex center_kind =
                center_object_key.structure_and_kind >> RS_CAMERA_PIXEL_OBJECT_KIND_SHIFT;
            RIndex center_material_gid = center_object_key.material_gid;
            FloatC sigma_color = FloatC(k->vision_denoise_sigma_color);
            FloatC sigma_normal = FloatC(k->vision_denoise_sigma_normal);
            FloatC sigma_depth = FloatC(k->vision_denoise_sigma_depth);
            FloatC lum_c = center_color.x * FloatC(0.2126) + center_color.y * FloatC(0.7152) +
                           center_color.z * FloatC(0.0722);
            FloatC inv_sigma_l = FloatC(1) / (sigma_color + FloatC(RS_EPS));

            // 5x5 a-trous kernel (B-spline).
            FloatC kernel_w[5] = {
                FloatC(1.0 / 16.0),
                FloatC(4.0 / 16.0),
                FloatC(6.0 / 16.0),
                FloatC(4.0 / 16.0),
                FloatC(1.0 / 16.0)
            };
            Vec3C accum = Vec3C::zero();
            FloatC accum_w = FloatC(0);
            for (int di = -2; di <= 2; ++di) {
                for (int dj = -2; dj <= 2; ++dj) {
                    RSize ni = _clampIdx(int(i) + di * step, int(h));
                    RSize nj = _clampIdx(int(j) + dj * step, int(w));
                    Vec3C c = Vec3C(read_canvas[camera][{ni, nj}]);
                    Vec3C n = Vec3C(ctx.camera_aux_normal[camera][{ni, nj}]);
                    FloatC dep = FloatC(ctx.camera_aux_depth_z[camera][{ni, nj}]);
                    RS_CameraPixelObjectKey n_object_key =
                        ctx.camera_aux_object_key[camera][{ni, nj}];

                    // Hard binary edge stop on kind tag + material_gid.
                    RIndex n_kind = n_object_key.structure_and_kind >>
                                    RS_CAMERA_PIXEL_OBJECT_KIND_SHIFT;
                    if (n_kind != center_kind ||
                        n_object_key.material_gid != center_material_gid)
                        continue;

                    FloatC w_k = kernel_w[di + 2] * kernel_w[dj + 2];
                    FloatC w_n = RS_FI::rs_pow(
                        RS_MAX(FloatC(0), center_normal.dot(n)), sigma_normal
                    );
                    FloatC depth_diff = dep - center_depth;
                    FloatC w_d = RS_FI::rs_exp_fast(
                        -(depth_diff * depth_diff) /
                        (sigma_depth * sigma_depth + FloatC(RS_EPS))
                    );
                    FloatC lum_n =
                        c.x * FloatC(0.2126) + c.y * FloatC(0.7152) + c.z * FloatC(0.0722);
                    FloatC w_l = RS_FI::rs_exp_fast(-RS_ABS(lum_c - lum_n) * inv_sigma_l);
                    FloatC weight = w_k * w_n * w_d * w_l;
                    accum = accum + c * weight;
                    accum_w = accum_w + weight;
                }
            }
            Vec3C out = accum_w > FloatC(RS_EPS) ? accum / accum_w : center_color;
            write_canvas[camera][{i, j}] = RVec3f(float(out.x), float(out.y), float(out.z));
        }

    };

    /**
     * @brief SVGF temporal accumulation. Reproject the current pixel through
     *        its screen-motion AOV, gate against the previous frame's
     *        (object_key, normal, depth) snapshot, and EMA-blend color +
     *        luminance moments (m1, m2). Roughness^2 caps the effective
     *        history length so mirror-class pixels never carry the long-tail
     *        ghost from primary-surface motion vectors (Zhdan 2020 p.46).
     *
     * @details Schied et al. 2017 §4.1; Zhdan, NVIDIA GTC 2020 (S22699) p.46.
     *
     * @param k       simulation kernel.
     * @param camera  target camera handle.
     * @param pixel   flat (i * W + j) pixel index.
     */
    class RSCF_SVGFTemporalAccumulate
        : public device::RS_ComputeFunction<RSCF_SVGFTemporalAccumulate> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;
            if (k->vision_denoise_type != RSE_VISION_DENOISE_SVGF) return;
            RSize w = CA_G(image_width);
            RSize h = CA_G(image_height);
            RSize i = RSize(pixel) / w;
            RSize j = RSize(pixel) - i * w;

            Vec3C current = Vec3C(ctx.camera_current_rgb_canvas[camera][{i, j}]);
            Vec3C normal = Vec3C(ctx.camera_aux_normal[camera][{i, j}]);
            FloatC depth = FloatC(ctx.camera_aux_depth_z[camera][{i, j}]);
            RS_CameraPixelObjectKey object_key = ctx.camera_aux_object_key[camera][{i, j}];
            RVec2f screen_motion = ctx.camera_aux_screen_motion[camera][{i, j}];

            // Reproject via screen-space motion vector; round to nearest pixel.
            using RS_FI = RS_FloatIntrinsics<FloatC>;
            int ri = int(i) - int(RS_FI::rs_round(FloatC(screen_motion.y) * FloatC(h)));
            int rj = int(j) - int(RS_FI::rs_round(FloatC(screen_motion.x) * FloatC(w)));
            bool in_bounds = (ri >= 0 && ri < int(h) && rj >= 0 && rj < int(w));

            bool accept = false;
            Vec3C prev_color = Vec3C::zero();
            RVec2f prev_moments(0.0f, 0.0f);
            FloatC prev_len = FloatC(0);
            if (in_bounds) {
                RSize hri = RSize(ri);
                RSize hrj = RSize(rj);
                RS_CameraPixelObjectKey prev_key =
                    ctx.camera_prev_aux_object_key[camera][{hri, hrj}];
                Vec3C n_prev = Vec3C(ctx.camera_prev_aux_normal[camera][{hri, hrj}]);
                FloatC d_prev = FloatC(ctx.camera_prev_aux_depth_z[camera][{hri, hrj}]);
                bool id_ok = (prev_key == object_key);
                bool n_ok = n_prev.dot(normal) > FloatC(0.9);
                bool d_ok =
                    RS_ABS(d_prev - depth) < FloatC(0.1) * RS_MAX(d_prev, FloatC(1e-3));
                if (id_ok && n_ok && d_ok) {
                    accept = true;
                    prev_color =
                        Vec3C(ctx.camera_prev_denoiser_color_history[camera][{hri, hrj}]);
                    prev_moments =
                        ctx.camera_prev_denoiser_moment_history[camera][{hri, hrj}];
                    prev_len =
                        FloatC(ctx.camera_prev_denoiser_history_length[camera][{hri, hrj}]);
                }
            }

            FloatC alpha = FloatC(k->vision_denoise_temporal_alpha);
            FloatC alpha_m = FloatC(k->vision_denoise_temporal_alpha_moments);
            FloatC new_len = accept ? prev_len + FloatC(1) : FloatC(1);

            // Roughness^2 cap on the effective EMA length: mirror/glass converge
            // to capped_len ~= temporal_min_history (eff_alpha = 1), so a primary
            // motion vector that does not track the reflected content cannot
            // smear stale color forward in time.
            FloatC roughness_p = FloatC(ctx.camera_aux_roughness[camera][{i, j}]);
            FloatC m_rough = roughness_p * roughness_p;
            FloatC capped_len = RS_MAX(
                FloatC(k->vision_denoise_temporal_min_history), m_rough * new_len
            );

            FloatC eff_alpha = RS_MAX(alpha, FloatC(1) / capped_len);
            FloatC eff_alpha_m = RS_MAX(alpha_m, FloatC(1) / capped_len);

            Vec3C blended =
                accept ? prev_color * (FloatC(1) - eff_alpha) + current * eff_alpha : current;
            FloatC lum = current.x * FloatC(0.2126) + current.y * FloatC(0.7152) +
                         current.z * FloatC(0.0722);
            FloatC m1 = accept ? FloatC(prev_moments.x) * (FloatC(1) - eff_alpha_m) +
                                     lum * eff_alpha_m
                               : lum;
            FloatC m2 = accept ? FloatC(prev_moments.y) * (FloatC(1) - eff_alpha_m) +
                                     lum * lum * eff_alpha_m
                               : lum * lum;

            ctx.camera_denoiser_color_history[camera][{i, j}] =
                RVec3f(float(blended.x), float(blended.y), float(blended.z));
            ctx.camera_denoiser_moment_history[camera][{i, j}] =
                RVec2f(float(m1), float(m2));
            ctx.camera_denoiser_history_length[camera][{i, j}] = RFloat(new_len);
            ctx.camera_current_rgb_canvas[camera][{i, j}] =
                RVec3f(float(blended.x), float(blended.y), float(blended.z));
        }
    };

    /**
     * @brief Per-pixel luminance variance for the variance-guided a-trous.
     *        Uses temporal moments (var = m2 - m1^2) once the history length
     *        reaches 4 frames; before that, falls back to a 7x7 spatial
     *        luminance variance over the temporally blended canvas.
     *
     * @details Schied et al. 2017 §4.1 (spatial fallback at short history).
     */
    class RSCF_SVGFVarianceEstimate
        : public device::RS_ComputeFunction<RSCF_SVGFVarianceEstimate> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;
            if (k->vision_denoise_type != RSE_VISION_DENOISE_SVGF) return;
            RSize w = CA_G(image_width);
            RSize h = CA_G(image_height);
            RSize i = RSize(pixel) / w;
            RSize j = RSize(pixel) - i * w;

            FloatC hist_len = FloatC(ctx.camera_denoiser_history_length[camera][{i, j}]);
            FloatC variance;
            if (hist_len >= FloatC(4)) {
                RVec2f m = ctx.camera_denoiser_moment_history[camera][{i, j}];
                variance = RS_MAX(FloatC(0), FloatC(m.y) - FloatC(m.x) * FloatC(m.x));
            } else {
                // Spatial fallback: 7x7 luminance variance over the blended canvas.
                const int radius = 3;
                FloatC sum = FloatC(0);
                FloatC sum_sq = FloatC(0);
                FloatC n = FloatC(0);
                for (int di = -radius; di <= radius; ++di) {
                    for (int dj = -radius; dj <= radius; ++dj) {
                        RSize ni = _clampIdx(int(i) + di, int(h));
                        RSize nj = _clampIdx(int(j) + dj, int(w));
                        Vec3C c = Vec3C(ctx.camera_current_rgb_canvas[camera][{ni, nj}]);
                        FloatC lum = c.x * FloatC(0.2126) + c.y * FloatC(0.7152) +
                                     c.z * FloatC(0.0722);
                        sum = sum + lum;
                        sum_sq = sum_sq + lum * lum;
                        n = n + FloatC(1);
                    }
                }
                FloatC mean = sum / n;
                variance = RS_MAX(FloatC(0), sum_sq / n - mean * mean);
            }
            ctx.camera_denoiser_variance[camera][{i, j}] = RFloat(variance);
        }
    };

    /**
     * @brief Variance-guided a-trous wavelet pass. Same shape as
     *        RSCF_AtrousFilter (5x5 B-spline, doubling step, roughness^2
     *        pass-count cap, hard material-gid edge stop) but the
     *        luminance-similarity sigma is scaled by sqrt(variance) so
     *        confident pixels stay sharp while noisy ones smooth more.
     *        Reads from read_canvas, writes to write_canvas; the executor
     *        ping-pongs canvases between passes.
     *
     * @details Schied et al. 2017 §4.2.
     */
    class RSCF_SVGFAtrousFilter : public device::RS_ComputeFunction<RSCF_SVGFAtrousFilter> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel* k,
            RHandle camera,
            RHandle pixel,
            device::RS_Vector<RS_CameraRGBCanvas> read_canvas,
            device::RS_Vector<RS_CameraRGBCanvas> write_canvas,
            RSize pass_index
        ) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;
            if (k->vision_denoise_type != RSE_VISION_DENOISE_SVGF) return;
            RSize total_passes = k->vision_denoise_atrous_iterations;
            if (pass_index >= total_passes) return;
            RSize w = CA_G(image_width);
            RSize h = CA_G(image_height);
            RSize i = RSize(pixel) / w;
            RSize j = RSize(pixel) - i * w;

            using RS_FI = RS_FloatIntrinsics<FloatC>;
            FloatC roughness_p = FloatC(ctx.camera_aux_roughness[camera][{i, j}]);
            FloatC m = roughness_p * roughness_p;
            RSize max_pass_p = RS_MAX(
                RSize(k->vision_denoise_atrous_min_passes),
                RSize(RS_FI::rs_ceil(m * FloatC(total_passes)))
            );
            if (pass_index >= max_pass_p) {
                write_canvas[camera][{i, j}] = read_canvas[camera][{i, j}];
                return;
            }

            int step = 1 << int(pass_index);

            Vec3C center_color = Vec3C(read_canvas[camera][{i, j}]);
            Vec3C center_normal = Vec3C(ctx.camera_aux_normal[camera][{i, j}]);
            FloatC center_depth = FloatC(ctx.camera_aux_depth_z[camera][{i, j}]);
            RS_CameraPixelObjectKey center_object_key = ctx.camera_aux_object_key[camera][{i, j}];
            RIndex center_kind =
                center_object_key.structure_and_kind >> RS_CAMERA_PIXEL_OBJECT_KIND_SHIFT;
            RIndex center_material_gid = center_object_key.material_gid;
            FloatC variance = FloatC(ctx.camera_denoiser_variance[camera][{i, j}]);
            FloatC sigma_color = FloatC(k->vision_denoise_sigma_color);
            FloatC sigma_normal = FloatC(k->vision_denoise_sigma_normal);
            FloatC sigma_depth = FloatC(k->vision_denoise_sigma_depth);
            FloatC lum_c = center_color.x * FloatC(0.2126) + center_color.y * FloatC(0.7152) +
                           center_color.z * FloatC(0.0722);
            FloatC inv_sigma_l =
                FloatC(1) /
                (sigma_color * RS_FI::rs_sqrt(RS_MAX(variance, FloatC(RS_EPS))) + FloatC(RS_EPS));

            // 5x5 a-trous kernel (B-spline).
            FloatC kernel_w[5] = {
                FloatC(1.0 / 16.0),
                FloatC(4.0 / 16.0),
                FloatC(6.0 / 16.0),
                FloatC(4.0 / 16.0),
                FloatC(1.0 / 16.0)
            };
            Vec3C accum = Vec3C::zero();
            FloatC accum_w = FloatC(0);
            for (int di = -2; di <= 2; ++di) {
                for (int dj = -2; dj <= 2; ++dj) {
                    RSize ni = _clampIdx(int(i) + di * step, int(h));
                    RSize nj = _clampIdx(int(j) + dj * step, int(w));
                    Vec3C c = Vec3C(read_canvas[camera][{ni, nj}]);
                    Vec3C n = Vec3C(ctx.camera_aux_normal[camera][{ni, nj}]);
                    FloatC dep = FloatC(ctx.camera_aux_depth_z[camera][{ni, nj}]);
                    RS_CameraPixelObjectKey n_object_key =
                        ctx.camera_aux_object_key[camera][{ni, nj}];

                    RIndex n_kind = n_object_key.structure_and_kind >>
                                    RS_CAMERA_PIXEL_OBJECT_KIND_SHIFT;
                    if (n_kind != center_kind ||
                        n_object_key.material_gid != center_material_gid)
                        continue;

                    FloatC w_k = kernel_w[di + 2] * kernel_w[dj + 2];
                    FloatC w_n = RS_FI::rs_pow(
                        RS_MAX(FloatC(0), center_normal.dot(n)), sigma_normal
                    );
                    FloatC depth_diff = dep - center_depth;
                    FloatC w_d = RS_FI::rs_exp_fast(
                        -(depth_diff * depth_diff) /
                        (sigma_depth * sigma_depth + FloatC(RS_EPS))
                    );
                    FloatC lum_n =
                        c.x * FloatC(0.2126) + c.y * FloatC(0.7152) + c.z * FloatC(0.0722);
                    FloatC w_l = RS_FI::rs_exp_fast(-RS_ABS(lum_c - lum_n) * inv_sigma_l);
                    FloatC weight = w_k * w_n * w_d * w_l;
                    accum = accum + c * weight;
                    accum_w = accum_w + weight;
                }
            }
            Vec3C out = accum_w > FloatC(RS_EPS) ? accum / accum_w : center_color;
            write_canvas[camera][{i, j}] = RVec3f(float(out.x), float(out.y), float(out.z));
        }
    };

    /**
     * @brief End-of-frame snapshot of (normal, depth_z, object_key) into the
     *        prev-aux slots so next frame's RSCF_SVGFTemporalAccumulate can
     *        run its disocclusion gate.
     */
    class RSCF_SVGFCopyAux : public device::RS_ComputeFunction<RSCF_SVGFCopyAux> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k == nullptr || camera == RS_NULL_HANDLE) return;
            auto& ctx = k->ctx;
            if (k->vision_denoise_type != RSE_VISION_DENOISE_SVGF) return;
            RSize w = CA_G(image_width);
            RSize i = RSize(pixel) / w;
            RSize j = RSize(pixel) - i * w;
            ctx.camera_prev_aux_normal[camera][{i, j}] = ctx.camera_aux_normal[camera][{i, j}];
            ctx.camera_prev_aux_depth_z[camera][{i, j}] = ctx.camera_aux_depth_z[camera][{i, j}];
            ctx.camera_prev_aux_object_key[camera][{i, j}] =
                ctx.camera_aux_object_key[camera][{i, j}];
        }
    };

   private:
    /// Clamp-to-edge neighbor index; clamps in signed space so off-image
    /// taps replicate the near edge instead of wrapping to the far edge.
    HOST_DEVICE_INLINE_QUALIFIER static RSize _clampIdx(int v, int hi) {
        if (v < 0) return RSize(0);
        if (v >= hi) return RSize(hi - 1);
        return RSize(v);
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_VISION_DENOISE_C_H
