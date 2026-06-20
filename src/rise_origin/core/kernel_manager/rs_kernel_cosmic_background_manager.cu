// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <algorithm>
#include <cmath>

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rs_kernel_cosmic_background_manager.h"
#include "rs_kernel_texture_manager.h"

using namespace std;
using namespace rise::rsc;

namespace rise {

/*****************************************************************
 * RS_KernelCosmicBackgroundManager
 *****************************************************************/
RS_KernelCosmicBackgroundManager::RS_KernelCosmicBackgroundManager(
    RS_KernelManager* kernel_manager, const SupportedRSC& cosmic_rsc
)
    : _kernel_manager(kernel_manager), _rsc(cosmic_rsc) {
    // Reference the Uniform background's texture so it cannot be removed while alive.
    if (_rsc.is<RSC_CosmicBackgroundUniform>()) {
        auto& uniform = _rsc.as<RSC_CosmicBackgroundUniform>();
        if (!uniform.texture_reference.empty()) {
            auto& tex_mgr = _kernel_manager->getKernelTextureManager();
            RSE_TextureStorageFormat storage_format =
                tex_mgr.getTextureStorageFormat(uniform.texture_reference);
            _texture_reference = RS_KernelTextureReference(
                _kernel_manager, uniform.texture_reference, RSE_TEXTURE_COLOR_SPACE_SRGB,
                storage_format
            );
        }
    }
}

void RS_KernelCosmicBackgroundManager::commit() {
    _kernel_manager->enterKernelDataCriticalZone();
    _updateKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;
}

void RS_KernelCosmicBackgroundManager::decommit() {
    if (_is_kernel_data_inited) {
        _kernel_manager->enterKernelDataCriticalZone();
        castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
            k_ptr->ctx.cosmic_background_vision_mem = logic::RS_VisionCosmicBackgroundMem{};
            _kernel_manager->markKernelModified();
        });
        _kernel_manager->exitKernelDataCriticalZone();
        _is_kernel_data_inited = false;
    }
}

void RS_KernelCosmicBackgroundManager::freeze() { _is_frozen = true; }

void RS_KernelCosmicBackgroundManager::unfreeze() { _is_frozen = false; }

bool RS_KernelCosmicBackgroundManager::isFrozen() const { return _is_frozen; }

const RS_KernelCosmicBackgroundManager::SupportedRSC&
RS_KernelCosmicBackgroundManager::getRSC() const {
    return _rsc;
}

void RS_KernelCosmicBackgroundManager::reference() { _use_count += 1; }

void RS_KernelCosmicBackgroundManager::dereference() {
    if (_use_count == 0) {
        RS_THROW(RS_HostExecutionException("Cosmic background use_count underflow on dereference"));
    }
    _use_count -= 1;
}

RSize RS_KernelCosmicBackgroundManager::getUseCount() const { return _use_count; }

void RS_KernelCosmicBackgroundManager::_updateKernelData() {
    if (_rsc.is<RSC_CosmicBackgroundUniform>()) {
        _updateKernelUniformData(_rsc.as<RSC_CosmicBackgroundUniform>());
    } else if (_rsc.is<RSC_CosmicBackgroundGradient>()) {
        _updateKernelGradientData(_rsc.as<RSC_CosmicBackgroundGradient>());
    } else {
        _updateKernelHosekWilkieData(_rsc.as<RSC_CosmicBackgroundHosekWilkie>());
    }
}

void RS_KernelCosmicBackgroundManager::_updateKernelUniformData(
    const RSC_CosmicBackgroundUniform& uniform
) {
    auto& tex_mgr = _kernel_manager->getKernelTextureManager();

    logic::RSE_VisionSampleInputType color_input_type = logic::RSE_VISION_SAMPLE_INPUT_CONSTANT;
    logic::RS_VisionSampleInputMem<RColor3f> color{RColor3f::zero()};

    if (!uniform.texture_reference.empty()) {
        // The texture is referenced by the constructor; resolve its value here.
        if (tex_mgr.isConstant(uniform.texture_reference)) {
            color = logic::RS_VisionSampleInputMem<RColor3f>(
                tex_mgr.getConstant(uniform.texture_reference)
            );
            color_input_type = logic::RSE_VISION_SAMPLE_INPUT_CONSTANT;
        } else {
            color = logic::RS_VisionSampleInputMem<RColor3f>(
                tex_mgr.findTextureHandle(uniform.texture_reference)
            );
            color_input_type = logic::RSE_VISION_SAMPLE_INPUT_GRID;
        }
    }

    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& cb = k_ptr->ctx.cosmic_background_vision_mem;
        logic::RS_VisionCosmicBackgroundStateUnion st{};
        st.cosmic_background_state.type = logic::RSE_VISION_COSMIC_BACKGROUND_UNIFORM;
        st.cosmic_background_state.color_input_type = color_input_type;
        cb.state = st.cosmic_background_state_packed;
        cb.color = color;
        cb.intensity = uniform.intensity;
        cb.rotation_rad = uniform.rotation_rad;
        // Uniform backgrounds carry no sun disk.
        cb.sun_color = RColor3f::zero();

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelCosmicBackgroundManager::_updateKernelGradientData(
    const RSC_CosmicBackgroundGradient& gradient
) {
    RColor3f zenith = RColor3f(gradient.zenith_color).srgbToLinear();
    RColor3f horizon = RColor3f(gradient.horizon_color).srgbToLinear();
    RColor3f sun_color = RColor3f(gradient.sun_color).srgbToLinear();
    RVec3f sun_dir = RVec3f(gradient.sun_direction).normalized();

    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& cb = k_ptr->ctx.cosmic_background_vision_mem;
        logic::RS_VisionCosmicBackgroundStateUnion st{};
        st.cosmic_background_state.type = logic::RSE_VISION_COSMIC_BACKGROUND_GRADIENT;
        cb.state = st.cosmic_background_state_packed;
        cb.zenith_color = zenith;
        cb.horizon_color = horizon;
        cb.sun_color = sun_color;
        cb.sun_direction = sun_dir;
        cb.sun_angular_radius = gradient.sun_angular_radius;
        cb.intensity = RFloat(1);

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelCosmicBackgroundManager::_updateKernelHosekWilkieData(
    const RSC_CosmicBackgroundHosekWilkie& hosek
) {
    RVec3f sun_dir = RVec3f(hosek.sun_direction).normalized();
    RColor3f sun_color = RColor3f(hosek.sun_color).srgbToLinear();

    // Average linear ground color into a single albedo for the model.
    RColor3f ground_linear = RColor3f(hosek.ground_color).srgbToLinear();
    double albedo = std::clamp(double(ground_linear.luminance()), 0.0, 1.0);

    // Solar elevation = angle of sun_direction above the horizon (XY plane).
    // sun_direction points TOWARD the sun, so elevation = asin(z / |sun_dir|).
    double elevation = std::asin(std::clamp(double(sun_dir.z), -1.0, 1.0));
    double turbidity = std::clamp(double(hosek.turbidity), 1.0, 10.0);

    RFloat hw_config[27] = {0};
    RFloat hw_radiance[3] = {0, 0, 0};

    if (elevation >= 0) {
        ArHosekSkyModelState* hosek_state =
            arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo, elevation);

        // The renderer's final stage is linearToSrgb().clip() with no exposure or
        // tonemap, so the kernel assumes scene radiance lies in ~[0, 1]. The Hosek-
        // Wilkie cook returns absolute physical radiance ~3-31 per channel for typical
        // turbidity / elevation, so without rescaling the sky saturates to white. Fix
        // is a scalar (Rec.709 luminance, not per-channel) hemisphere-max
        // normalization baked into hosek_state->radiances[] at cook time —
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
                    dx * double(sun_dir.x) + dy * double(sun_dir.y) + dz * double(sun_dir.z),
                    -1.0,
                    1.0
                );
                RColor3f rgb(
                    float(_evalHosekChannel(hosek_state, 0, ct, cg)),
                    float(_evalHosekChannel(hosek_state, 1, ct, cg)),
                    float(_evalHosekChannel(hosek_state, 2, ct, cg))
                );
                double lum = double(rgb.luminance());
                if (lum > max_lum) max_lum = lum;
            }
        }
        constexpr double TARGET = 0.9;  // headroom below the linearToSrgb clip ceiling
        double norm = (max_lum > 1e-6) ? (TARGET / max_lum) : 1.0;

        for (int ch = 0; ch < 3; ++ch) {
            for (int i = 0; i < 9; ++i) {
                hw_config[ch * 9 + i] = RFloat(hosek_state->configs[ch][i]);
            }
            hw_radiance[ch] = RFloat(hosek_state->radiances[ch] * norm);
        }

        arhosekskymodelstate_free(hosek_state);
    }

    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& cb = k_ptr->ctx.cosmic_background_vision_mem;
        logic::RS_VisionCosmicBackgroundStateUnion st{};
        st.cosmic_background_state.type = logic::RSE_VISION_COSMIC_BACKGROUND_HOSEK_WILKIE;
        cb.state = st.cosmic_background_state_packed;
        cb.sun_direction = sun_dir;
        cb.sun_color = sun_color;
        cb.sun_angular_radius = hosek.sun_angular_radius;
        cb.intensity = hosek.intensity;
        for (int i = 0; i < 27; ++i) cb.hw_config[i] = hw_config[i];
        for (int ch = 0; ch < 3; ++ch) cb.hw_radiance[ch] = hw_radiance[ch];

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

double RS_KernelCosmicBackgroundManager::_evalHosekChannel(
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

/*****************************************************************
 * RS_KernelCosmicBackgroundReference
 *****************************************************************/
RS_KernelCosmicBackgroundReference::RS_KernelCosmicBackgroundReference(
    RS_KernelManager* kernel_manager
)
    : _kernel_manager(kernel_manager) {
    auto& cosmic = _kernel_manager->getKernelCosmicBackgroundManager();
    if (!cosmic) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Cosmic background is not registered in the kernel manager"
        ));
    }
    cosmic->reference();
}

RS_KernelCosmicBackgroundReference::~RS_KernelCosmicBackgroundReference() { _release(); }

RS_KernelCosmicBackgroundReference::RS_KernelCosmicBackgroundReference(
    RS_KernelCosmicBackgroundReference&& other
) noexcept
    : _kernel_manager(other._kernel_manager) {
    other._kernel_manager = nullptr;
}

RS_KernelCosmicBackgroundReference& RS_KernelCosmicBackgroundReference::operator=(
    RS_KernelCosmicBackgroundReference&& other
) noexcept {
    if (this != &other) {
        _release();
        _kernel_manager = other._kernel_manager;
        other._kernel_manager = nullptr;
    }
    return *this;
}

void RS_KernelCosmicBackgroundReference::_release() {
    if (_kernel_manager == nullptr) return;
    if (auto& cosmic = _kernel_manager->getKernelCosmicBackgroundManager()) {
        cosmic->dereference();
    }
    _kernel_manager = nullptr;
}

}  // namespace rise
