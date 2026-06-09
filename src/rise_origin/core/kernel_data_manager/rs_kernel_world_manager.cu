// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <algorithm>
#include <cmath>

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/utils/rs_color.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_kernel_world_manager.h"

using namespace std;

namespace rise {

void RS_KernelWorldManager::init(
    RS_KernelManager* kernel_manager, const RS_WorldConfig& world_config
) {
    _kernel_manager = kernel_manager;
    setWorld(world_config);
}

void RS_KernelWorldManager::free() {
    if (_hosek_state != nullptr) {
        arhosekskymodelstate_free(_hosek_state);
        _hosek_state = nullptr;
    }
}

void RS_KernelWorldManager::setWorld(const RS_WorldConfig& world_config) {
    // Drop any previously-cooked Hosek state so re-applying a new config never
    // leaks. Re-allocated below if the new config is Hosek-Wilkie with the sun
    // above the horizon.
    free();

    auto& kernel = _kernel_manager->kernel;
    kernel.world_render_type = world_config.render_config.type;

    if (world_config.render_config.type == RSE_WORLD_RENDER_CONSTANT_COLOR) {
        auto& constant_color_config =
            world_config.render_config.config.as<RS_ConstantColorWorldRenderConfig>();
        kernel.world_color = RColor3f(constant_color_config.color).srgbToLinear();
    } else if (world_config.render_config.type == RSE_WORLD_RENDER_GRADIENT) {
        auto& gradient_config =
            world_config.render_config.config.as<RS_GradientSkyWorldRenderConfig>();
        kernel.world_color = RColor3f(gradient_config.zenith_color).srgbToLinear();
        kernel.world_horizon_color = RColor3f(gradient_config.horizon_color).srgbToLinear();
        kernel.world_sun_color = RColor3f(gradient_config.sun_color).srgbToLinear();
        kernel.world_sun_direction = RVec3f(gradient_config.sun_direction).normalized();
        kernel.world_sun_angular_radius = gradient_config.sun_angular_radius;
        kernel.world_sky_intensity = RFloat(1);
    } else if (world_config.render_config.type == RSE_WORLD_RENDER_HOSEK_WILKIE) {
        auto& hw_config =
            world_config.render_config.config.as<RS_HosekWilkieSkyWorldRenderConfig>();
        RVec3f sun_dir = RVec3f(hw_config.sun_direction).normalized();
        kernel.world_sun_direction = sun_dir;
        kernel.world_sun_color = RColor3f(hw_config.sun_color).srgbToLinear();
        kernel.world_sun_angular_radius = hw_config.sun_angular_radius;
        kernel.world_sky_intensity = hw_config.intensity;

        // Average linear ground color into a single albedo for the model.
        RColor3f ground_linear = RColor3f(hw_config.ground_color).srgbToLinear();
        double albedo = std::clamp(double(ground_linear.luminance()), 0.0, 1.0);

        // Solar elevation = angle of sun_direction above the horizon (XY plane).
        // sun_direction points TOWARD the sun, so elevation = asin(z / |sun_dir|).
        double elevation = std::asin(std::clamp(double(sun_dir.z), -1.0, 1.0));
        double turbidity = std::clamp(double(hw_config.turbidity), 1.0, 10.0);

        if (elevation < 0) {
            // Sun below horizon: zero out cooked state; sky returns black.
            for (int i = 0; i < 27; ++i) kernel.world_hw_config[i] = 0;
            kernel.world_hw_radiance[0] = 0;
            kernel.world_hw_radiance[1] = 0;
            kernel.world_hw_radiance[2] = 0;
        } else {
            _hosek_state = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo, elevation);

            // The renderer's final stage is linearToSrgb().clip() with no exposure or
            // tonemap, so the kernel assumes scene radiance lies in ~[0, 1]. The Hosek-
            // Wilkie cook returns absolute physical radiance ~3-31 per channel for typical
            // turbidity / elevation, so without rescaling the sky saturates to white. Fix
            // is a scalar (Rec.709 luminance, not per-channel) hemisphere-max
            // normalization baked into _hosek_state->radiances[] at cook time —
            // preserves chromaticity, scales magnitude. Recomputed per cook because the
            // absolute scale shifts with turbidity / sun_elevation / albedo.
            double max_lum = 0.0;
            constexpr int THETA_STEPS = 90;
            constexpr int PHI_STEPS = 120;
            for (int it = 0; it <= THETA_STEPS; ++it) {
                // Stop just shy of the horizon: the model diverges as cos_theta -> 0.
                double theta = (RS_PI * 0.5) * (double(it) / THETA_STEPS) * (89.0 / 90.0);
                double ct = std::cos(theta);
                double st = std::sin(theta);
                for (int ip = 0; ip < PHI_STEPS; ++ip) {
                    double ph = RS_2PI * (double(ip) / PHI_STEPS);
                    double dx = st * std::cos(ph);
                    double dy = st * std::sin(ph);
                    double dz = ct;
                    double cg = std::clamp(
                        dx * double(sun_dir.x) + dy * double(sun_dir.y) +
                            dz * double(sun_dir.z),
                        -1.0,
                        1.0
                    );
                    RColor3f rgb(
                        float(_evalHosekChannel(_hosek_state, 0, ct, cg)),
                        float(_evalHosekChannel(_hosek_state, 1, ct, cg)),
                        float(_evalHosekChannel(_hosek_state, 2, ct, cg))
                    );
                    double lum = double(rgb.luminance());
                    if (lum > max_lum) max_lum = lum;
                }
            }
            constexpr double TARGET = 0.9;  // headroom below the linearToSrgb clip ceiling
            double norm = (max_lum > 1e-6) ? (TARGET / max_lum) : 1.0;

            for (int ch = 0; ch < 3; ++ch) {
                for (int i = 0; i < 9; ++i) {
                    kernel.world_hw_config[ch * 9 + i] = RFloat(_hosek_state->configs[ch][i]);
                }
                kernel.world_hw_radiance[ch] = RFloat(_hosek_state->radiances[ch] * norm);
            }
        }
    }

    _config = world_config;
}

const RS_WorldConfig& RS_KernelWorldManager::getConfig() const { return _config; }

double RS_KernelWorldManager::_evalHosekChannel(
    const ArHosekSkyModelState* state, int ch, double cos_theta, double cos_gamma
) {
    const double* c = state->configs[ch];
    double gamma = std::acos(cos_gamma);
    double expM = std::exp(c[4] * gamma);
    double rayM = cos_gamma * cos_gamma;
    double md = 1.0 + c[8] * c[8] - 2.0 * c[8] * cos_gamma;
    double mieM = (1.0 + rayM) / (md * std::sqrt(md));
    double zenith = std::sqrt(cos_theta);
    double val = (1.0 + c[0] * std::exp(c[1] / (cos_theta + 0.01))) *
                 (c[2] + c[3] * expM + c[5] * rayM + c[6] * mieM + c[7] * zenith);
    return val * state->radiances[ch];
}

}  // namespace rise
