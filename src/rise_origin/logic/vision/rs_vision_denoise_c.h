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

    // Temporal reprojection disocclusion gate.
    static constexpr FloatC REPROJECT_NORMAL_COS_MIN = FloatC(0.9);
    static constexpr FloatC REPROJECT_DEPTH_REL_THRESHOLD = FloatC(0.1);
    static constexpr FloatC REPROJECT_DEPTH_FLOOR = FloatC(1e-3);
    // Spatial variance boost for the newest pixels with the least history.
    static constexpr FloatC VARIANCE_SHORT_HISTORY_BOOST = FloatC(4);

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

            Vec3C center_rgb = Vec3C(read_canvas[camera][{i, j}]);
            Vec3C center_normal = Vec3C(ctx.camera_aux_normal[camera][{i, j}]);
            FloatC center_depth = FloatC(ctx.camera_aux_depth_z[camera][{i, j}]);
            RS_CameraPixelObjectKey center_object_key = ctx.camera_aux_object_key[camera][{i, j}];
            RIndex center_kind =
                center_object_key.entity_gid >= RS_WORLD_RSC_ENTITY_RESERVED_GID_NUM
                    ? RIndex(0)
                    : center_object_key.entity_gid;
            RIndex center_vision_material_key = center_object_key.vision_material_key;
            FloatC sigma_color = FloatC(k->vision_denoise_sigma_color);
            FloatC sigma_normal = FloatC(k->vision_denoise_sigma_normal);
            FloatC sigma_depth = FloatC(k->vision_denoise_sigma_depth);
            FloatC lum_c = RColor3f(center_rgb).luminance();
            FloatC inv_sigma_l = FloatC(1) / (sigma_color + FloatC(RS_EPS));
            FloatC inv_sigma_depth2 =
                RS_FI::rs_divide_fast(FloatC(1), sigma_depth * sigma_depth + FloatC(RS_EPS));

            Vec3C accum = Vec3C::zero();
            FloatC accum_w = FloatC(0);
#pragma unroll
            for (int di = -2; di <= 2; ++di) {
#pragma unroll
                for (int dj = -2; dj <= 2; ++dj) {
                    RSize ni = _clampIdx(int(i) + di * step, int(h));
                    RSize nj = _clampIdx(int(j) + dj * step, int(w));
                    Vec3C rgb = Vec3C(read_canvas[camera][{ni, nj}]);
                    Vec3C n = Vec3C(ctx.camera_aux_normal[camera][{ni, nj}]);
                    FloatC dep = FloatC(ctx.camera_aux_depth_z[camera][{ni, nj}]);
                    RS_CameraPixelObjectKey n_object_key =
                        ctx.camera_aux_object_key[camera][{ni, nj}];

                    RIndex n_kind = n_object_key.entity_gid >= RS_WORLD_RSC_ENTITY_RESERVED_GID_NUM
                                        ? RIndex(0)
                                        : n_object_key.entity_gid;
                    // Hard binary edge stop on object category + vision material key.
                    if (n_kind != center_kind ||
                        n_object_key.vision_material_key != center_vision_material_key)
                        continue;

                    FloatC w_k = _bspline5(di) * _bspline5(dj);
                    FloatC w_n =
                        RS_FI::rs_pow_fast(RS_MAX(FloatC(0), center_normal.dot(n)), sigma_normal);
                    FloatC depth_diff = dep - center_depth;
                    FloatC w_d = RS_FI::rs_exp_fast(-(depth_diff * depth_diff) * inv_sigma_depth2);
                    FloatC lum_n = RColor3f(rgb).luminance();
                    FloatC w_l = RS_FI::rs_exp_fast(-RS_ABS(lum_c - lum_n) * inv_sigma_l);
                    FloatC weight = w_k * w_n * w_d * w_l;
                    accum = accum + rgb * weight;
                    accum_w = accum_w + weight;
                }
            }
            Vec3C out = accum_w > FloatC(RS_EPS) ? accum / accum_w : center_rgb;
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

            Vec3C current_rgb = Vec3C(ctx.camera_current_rgb_canvas[camera][{i, j}]);
            Vec3C normal = Vec3C(ctx.camera_aux_normal[camera][{i, j}]);
            FloatC depth = FloatC(ctx.camera_aux_depth_z[camera][{i, j}]);
            RS_CameraPixelObjectKey object_key = ctx.camera_aux_object_key[camera][{i, j}];
            RVec2f screen_motion = ctx.camera_aux_screen_motion[camera][{i, j}];

            // Reproject via screen-space motion vector with bilinear 2x2 taps so
            // sub-pixel head motion does not snap to a single stale pixel.
            using RS_FI = RS_FloatIntrinsics<FloatC>;
            FloatC fri = FloatC(i) - FloatC(screen_motion.y) * FloatC(h);
            FloatC frj = FloatC(j) - FloatC(screen_motion.x) * FloatC(w);
            int ri0 = int(RS_FI::rs_floor(fri));
            int rj0 = int(RS_FI::rs_floor(frj));
            FloatC ty = fri - FloatC(ri0);
            FloatC tx = frj - FloatC(rj0);

            // Accumulate only the bilinear taps that pass the disocclusion gate,
            // then renormalize over the accepted weight.
            bool accept = false;
            Vec3C prev_rgb = Vec3C::zero();
            RVec2f prev_moments(0.0f, 0.0f);
            FloatC prev_len = FloatC(0);
            FloatC prev_m1 = FloatC(0);
            FloatC prev_m2 = FloatC(0);
            FloatC sum_w = FloatC(0);
            for (int t = 0; t < 4; ++t) {
                int pri = ri0 + (t >> 1);
                int prj = rj0 + (t & 1);
                if (pri < 0 || pri >= int(h) || prj < 0 || prj >= int(w)) continue;
                RSize hri = RSize(pri);
                RSize hrj = RSize(prj);
                RS_CameraPixelObjectKey prev_key =
                    ctx.camera_prev_aux_object_key[camera][{hri, hrj}];
                Vec3C n_prev = Vec3C(ctx.camera_prev_aux_normal[camera][{hri, hrj}]);
                FloatC d_prev = FloatC(ctx.camera_prev_aux_depth_z[camera][{hri, hrj}]);
                bool id_ok = (prev_key == object_key);
                bool n_ok = n_prev.dot(normal) > REPROJECT_NORMAL_COS_MIN;
                bool d_ok = RS_ABS(d_prev - depth) <
                            REPROJECT_DEPTH_REL_THRESHOLD * RS_MAX(d_prev, REPROJECT_DEPTH_FLOOR);
                if (!(id_ok && n_ok && d_ok)) continue;
                // Bilinear tap weight, di on the row (t >> 1), dj on the column (t & 1):
                // t0: (1 - tx)(1 - ty)   t1: tx(1 - ty)   t2: (1 - tx)ty   t3: tx*ty
                FloatC tap_weight =
                    ((t & 1) ? tx : (FloatC(1) - tx)) * ((t >> 1) ? ty : (FloatC(1) - ty));
                prev_rgb =
                    prev_rgb +
                    Vec3C(ctx.camera_prev_denoiser_rgb_history[camera][{hri, hrj}]) * tap_weight;
                RVec2f pm = ctx.camera_prev_denoiser_moment_history[camera][{hri, hrj}];
                prev_m1 = prev_m1 + tap_weight * FloatC(pm.x);
                prev_m2 = prev_m2 + tap_weight * FloatC(pm.y);
                prev_len = prev_len +
                           tap_weight *
                               FloatC(ctx.camera_prev_denoiser_history_length[camera][{hri, hrj}]);
                sum_w = sum_w + tap_weight;
            }
            if (sum_w > FloatC(RS_EPS)) {
                accept = true;
                FloatC inv_sum_w = FloatC(1) / sum_w;
                prev_rgb = prev_rgb * inv_sum_w;
                prev_len = prev_len * inv_sum_w;
                prev_moments = RVec2f(float(prev_m1 * inv_sum_w), float(prev_m2 * inv_sum_w));
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
            FloatC capped_len =
                RS_MAX(FloatC(k->vision_denoise_temporal_min_history), m_rough * new_len);

            FloatC eff_alpha = RS_MAX(alpha, FloatC(1) / capped_len);
            FloatC eff_alpha_m = RS_MAX(alpha_m, FloatC(1) / capped_len);

            Vec3C blended = accept ? prev_rgb * (FloatC(1) - eff_alpha) + current_rgb * eff_alpha
                                   : current_rgb;
            FloatC lum = RColor3f(current_rgb).luminance();
            FloatC m1 = accept
                            ? FloatC(prev_moments.x) * (FloatC(1) - eff_alpha_m) + lum * eff_alpha_m
                            : lum;
            FloatC m2 = accept ? FloatC(prev_moments.y) * (FloatC(1) - eff_alpha_m) +
                                     lum * lum * eff_alpha_m
                               : lum * lum;

            ctx.camera_denoiser_rgb_history[camera][{i, j}] =
                RVec3f(float(blended.x), float(blended.y), float(blended.z));
            ctx.camera_denoiser_moment_history[camera][{i, j}] = RVec2f(float(m1), float(m2));
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
    class RSCF_SVGFVarianceEstimate : public device::RS_ComputeFunction<RSCF_SVGFVarianceEstimate> {
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
                FloatC sum = FloatC(0);
                FloatC sum_sq = FloatC(0);
                FloatC n = FloatC(0);
#pragma unroll
                for (int di = -3; di <= 3; ++di) {
#pragma unroll
                    for (int dj = -3; dj <= 3; ++dj) {
                        RSize ni = _clampIdx(int(i) + di, int(h));
                        RSize nj = _clampIdx(int(j) + dj, int(w));
                        auto rgb = RColor3f(ctx.camera_current_rgb_canvas[camera][{ni, nj}]);
                        FloatC lum = rgb.luminance();
                        sum = sum + lum;
                        sum_sq = sum_sq + lum * lum;
                        n = n + FloatC(1);
                    }
                }
                FloatC mean = sum / n;
                variance = RS_MAX(FloatC(0), sum_sq / n - mean * mean);
                // Scale the spatial estimate up by 4 / h: h=1 gives x4 for the
                // newest, noisiest pixel, h=2 gives x2, and h=4 gives x1 where it
                // meets the temporal path. The less history a pixel has, the more
                // it is distrusted and the harder it is filtered.
                variance = variance * (VARIANCE_SHORT_HISTORY_BOOST / RS_MAX(hist_len, FloatC(1)));
            }
            ctx.camera_denoiser_variance[camera][{i, j}] = RFloat(variance);
        }
    };

    /**
     * @brief Variance-guided a-trous wavelet pass. Same shape as
     *        RSCF_AtrousFilter but the luminance sigma is scaled by
     *        sqrt(variance) so confident pixels stay sharp. The luminance
     *        weight reads a 3x3 Gaussian prefiltered variance so a slowly
     *        varying penumbra is not over-sharpened into mist, and the variance
     *        is filtered with the squared edge-stop weight into write_variance
     *        so each pass narrows the next pass's sigma. The executor ping-pongs
     *        both canvas and variance between passes.
     *
     * @details Schied et al. 2017 §4.2. Prefilter and squared-weight variance
     *          propagation follow the Falcor SVGFAtrous reference.
     *
     * @note Albedo demodulation (blur radiance / albedo, then multiply albedo
     *       back) is deliberately not implemented. The divide and multiply invert
     *       only at the same pixel, but the a-trous blurs neighbors in between, so
     *       a pixel is multiplied by its own albedo after being divided partly by
     *       a neighbor's, and the two no longer cancel. On a saturated color, one
     *       channel close to 1 and the other close to 0, the divide by the near-0
     *       channel explodes into a spike that blurs out into a white grid glow.
     *
     * @param read_variance   per-camera input variance vector.
     * @param write_variance  per-camera output variance vector.
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
            device::RS_Vector<RS_CameraDenoiserVariance> read_variance,
            device::RS_Vector<RS_CameraDenoiserVariance> write_variance,
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
                write_variance[camera][{i, j}] = read_variance[camera][{i, j}];
                return;
            }

            int step = 1 << int(pass_index);

            Vec3C center_rgb = Vec3C(read_canvas[camera][{i, j}]);
            Vec3C center_normal = Vec3C(ctx.camera_aux_normal[camera][{i, j}]);
            FloatC center_depth = FloatC(ctx.camera_aux_depth_z[camera][{i, j}]);
            RS_CameraPixelObjectKey center_object_key = ctx.camera_aux_object_key[camera][{i, j}];
            RIndex center_kind =
                center_object_key.entity_gid >= RS_WORLD_RSC_ENTITY_RESERVED_GID_NUM
                    ? RIndex(0)
                    : center_object_key.entity_gid;
            RIndex center_vision_material_key = center_object_key.vision_material_key;
            FloatC sigma_color = FloatC(k->vision_denoise_sigma_color);
            FloatC sigma_normal = FloatC(k->vision_denoise_sigma_normal);
            FloatC sigma_depth = FloatC(k->vision_denoise_sigma_depth);
            FloatC lum_c = RColor3f(center_rgb).luminance();
            FloatC var_prefiltered = _prefilterVariance3x3(read_variance, camera, i, j, w, h);
            FloatC inv_sigma_l =
                FloatC(1) / (sigma_color * RS_FI::rs_sqrt(RS_MAX(var_prefiltered, FloatC(RS_EPS))) +
                             FloatC(RS_EPS));

            FloatC inv_sigma_depth2 =
                RS_FI::rs_divide_fast(FloatC(1), sigma_depth * sigma_depth + FloatC(RS_EPS));
            Vec3C accum = Vec3C::zero();
            FloatC accum_w = FloatC(0);
            FloatC accum_var = FloatC(0);
#pragma unroll
            for (int di = -2; di <= 2; ++di) {
#pragma unroll
                for (int dj = -2; dj <= 2; ++dj) {
                    RSize ni = _clampIdx(int(i) + di * step, int(h));
                    RSize nj = _clampIdx(int(j) + dj * step, int(w));
                    Vec3C rgb = Vec3C(read_canvas[camera][{ni, nj}]);
                    Vec3C n = Vec3C(ctx.camera_aux_normal[camera][{ni, nj}]);
                    FloatC dep = FloatC(ctx.camera_aux_depth_z[camera][{ni, nj}]);
                    RS_CameraPixelObjectKey n_object_key =
                        ctx.camera_aux_object_key[camera][{ni, nj}];

                    RIndex n_kind = n_object_key.entity_gid >= RS_WORLD_RSC_ENTITY_RESERVED_GID_NUM
                                        ? RIndex(0)
                                        : n_object_key.entity_gid;
                    // Hard binary edge stop on object category + vision material key.
                    if (n_kind != center_kind ||
                        n_object_key.vision_material_key != center_vision_material_key)
                        continue;

                    FloatC w_k = _bspline5(di) * _bspline5(dj);
                    FloatC w_n =
                        RS_FI::rs_pow_fast(RS_MAX(FloatC(0), center_normal.dot(n)), sigma_normal);
                    FloatC depth_diff = dep - center_depth;
                    FloatC w_d = RS_FI::rs_exp_fast(-(depth_diff * depth_diff) * inv_sigma_depth2);
                    FloatC lum_n = RColor3f(rgb).luminance();
                    FloatC w_l = RS_FI::rs_exp_fast(-RS_ABS(lum_c - lum_n) * inv_sigma_l);
                    FloatC weight = w_k * w_n * w_d * w_l;
                    accum = accum + rgb * weight;
                    accum_w = accum_w + weight;
                    // Variance filters with the squared weight so its scale tracks
                    // the color estimator's variance reduction across passes.
                    accum_var =
                        accum_var + weight * weight * FloatC(read_variance[camera][{ni, nj}]);
                }
            }
            Vec3C out = accum_w > FloatC(RS_EPS) ? accum / accum_w : center_rgb;
            FloatC out_var = accum_w > FloatC(RS_EPS) ? accum_var / (accum_w * accum_w)
                                                      : FloatC(read_variance[camera][{i, j}]);
            write_canvas[camera][{i, j}] = RVec3f(float(out.x), float(out.y), float(out.z));
            write_variance[camera][{i, j}] = RFloat(out_var);
        }

       private:
        /**
         * @brief 3x3 Gaussian over the variance buffer with the Falcor weights
         *        (center 1/4, edge 1/8, corner 1/16). Driving the luminance
         *        weight off a smoothed variance keeps a slowly varying penumbra
         *        from reading a near-zero sigma and being over-sharpened into
         *        mist.
         */
        HOST_DEVICE_INLINE_QUALIFIER static FloatC _prefilterVariance3x3(
            device::RS_Vector<RS_CameraDenoiserVariance> read_variance,
            RHandle camera,
            RSize i,
            RSize j,
            RSize w,
            RSize h
        ) {
            FloatC sum = FloatC(0);
            FloatC sum_w = FloatC(0);
#pragma unroll
            for (int di = -1; di <= 1; ++di) {
#pragma unroll
                for (int dj = -1; dj <= 1; ++dj) {
                    RSize ni = _clampIdx(int(i) + di, int(h));
                    RSize nj = _clampIdx(int(j) + dj, int(w));
                    FloatC kw = _gauss3x3(RS_ABS(di) + RS_ABS(dj));
                    sum = sum + kw * FloatC(read_variance[camera][{ni, nj}]);
                    sum_w = sum_w + kw;
                }
            }
            return sum / sum_w;
        }
    };

   private:
    /**
     * @brief Clamp-to-edge neighbor index; clamps in signed space so off-image
     *        taps replicate the near edge instead of wrapping to the far edge.
     */
    HOST_DEVICE_INLINE_QUALIFIER static RSize _clampIdx(int v, int hi) {
        if (v < 0) return RSize(0);
        if (v >= hi) return RSize(hi - 1);
        return RSize(v);
    }

    /**
     * @brief 5-tap B-spline weight for offset d in [-2, 2], values
     *        {1, 4, 6, 4, 1} / 16, computed with selects so no local-memory
     *        array is materialized for the a-trous kernel weights.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _bspline5(int d) {
        int a = RS_ABS(d);
        return a == 0 ? FloatC(6.0 / 16.0) : (a == 1 ? FloatC(4.0 / 16.0) : FloatC(1.0 / 16.0));
    }

    /**
     * @brief 3x3 Gaussian weight (Falcor) for combined offset d = |di| + |dj|,
     *        values 1/4 center, 1/8 edge, 1/16 corner, computed with selects so
     *        no local-memory array is materialized for the variance prefilter.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC _gauss3x3(int d) {
        return d == 0 ? FloatC(1.0 / 4.0) : (d == 1 ? FloatC(1.0 / 8.0) : FloatC(1.0 / 16.0));
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_VISION_DENOISE_C_H
