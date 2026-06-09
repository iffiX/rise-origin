// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/logic/vision/rs_vision_material_m.h"
#include "rs_kernel_material_manager.h"

using namespace std;
using namespace rise::logic;

namespace rise {

void RS_KernelMaterialManager::init(
    RS_KernelManager* kernel_manager,
    const RS_MaterialConfig& config
) {
    _kernel_manager = kernel_manager;
    _config = config;
    _material_gid = kernel_manager->getMaterialNameToGid().at(config.name);

    _initKernelData();
}

void RS_KernelMaterialManager::update(const RS_MaterialConfig& config) {
    _config = config;
    _updateKernelData();
}

void RS_KernelMaterialManager::free() { _freeKernelData(); }

std::string RS_KernelMaterialManager::getName() const { return _config.name; }

const RS_MaterialConfig& RS_KernelMaterialManager::getConfig() const { return _config; }

void RS_KernelMaterialManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        lock_guard lock(k_ptr->getSyncMutex());

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Context data is modified, needs to update kernel
        k_ptr->markModified();

        auto& ctx = k_ptr->ctx;
        using ContextType = decltype(k_ptr->ctx);

        auto vision_material =
            _buildVisionMaterial<ContextType::FloatMemAccuracy>(_config.render_config);
        vision_material.material_gid = _material_gid;

        // Update host layout
        ctx.host_data->vision_material_layout.add(1, _material_gid);

        // Update device layout
        ctx.host_data->vision_material_layout.syncDeviceLayout(ctx.vision_material_layout);

        // Adapt device storage size
        ctx.vision_materials.resize(ctx.host_data->vision_material_layout.capacity());

        // Write data to device storage
        RSize vision_material_offset = ctx.host_data->vision_material_layout.segmentOffset(
            ctx.host_data->vision_material_layout.find(_material_gid)
        );
        ctx.vision_materials.writeSlice(&vision_material, 1, vision_material_offset);
    });
}

void RS_KernelMaterialManager::_updateKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        lock_guard lock(k_ptr->getSyncMutex());

        auto& ctx = k_ptr->ctx;
        using ContextType = decltype(k_ptr->ctx);

        auto vision_material =
            _buildVisionMaterial<ContextType::FloatMemAccuracy>(_config.render_config);
        vision_material.material_gid = _material_gid;

        RSize vision_material_offset = ctx.host_data->vision_material_layout.segmentOffset(
            ctx.host_data->vision_material_layout.find(_material_gid)
        );
        ctx.vision_materials.writeSlice(&vision_material, 1, vision_material_offset);
    });
}

void RS_KernelMaterialManager::_freeKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        lock_guard lock(k_ptr->getSyncMutex());

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Context data is modified, needs to update kernel
        k_ptr->markModified();

        auto& ctx = k_ptr->ctx;

        // Update host layout
        ctx.host_data->vision_material_layout.remove(
            ctx.host_data->vision_material_layout.find(_material_gid)
        );

        // Update device layout
        ctx.host_data->vision_material_layout.syncDeviceLayout(ctx.vision_material_layout);

        // Adapt device storage size
        ctx.vision_materials.resize(ctx.host_data->vision_material_layout.capacity());
    });
}

RSE_VisionMaterialType RS_KernelMaterialManager::_toVisionMaterialType(
    RSE_MaterialRenderType config_type
) {
    switch (config_type) {
        case RSE_MATERIAL_RENDER_OPAQUE:
            return RSE_VISION_MATERIAL_OPAQUE;
        case RSE_MATERIAL_RENDER_TRANSPARENT:
            return RSE_VISION_MATERIAL_TRANSPARENT;
        case RSE_MATERIAL_RENDER_EMISSIVE:
            return RSE_VISION_MATERIAL_EMISSIVE;
    }
    return RSE_VISION_MATERIAL_OPAQUE;
}

template <RSE_FloatType FloatMemAccuracy>
RS_VisionMaterialMem<FloatMemAccuracy> RS_KernelMaterialManager::_buildVisionMaterial(
    const RS_MaterialRenderConfig& cfg
) {
    RS_VisionMaterialMem<FloatMemAccuracy> mat;
    RS_VisionMaterialStateUnion state_union{0};
    state_union.vision_material_state.render_type = _toVisionMaterialType(cfg.type);
    mat.state = state_union.vision_material_state_packed;

    if (cfg.type == RSE_MATERIAL_RENDER_OPAQUE) {
        auto& sub = cfg.config.as<RS_OpaqueMaterialRenderConfig>();
        mat.color = RColor3f(sub.color).srgbToLinear();
        mat.roughness = sub.roughness;
        mat.metalness = sub.metalness;
    } else if (cfg.type == RSE_MATERIAL_RENDER_TRANSPARENT) {
        auto& sub = cfg.config.as<RS_TransparentMaterialRenderConfig>();
        mat.color = RColor3f(sub.color).srgbToLinear();
        mat.ior = sub.ior;
        mat.roughness = sub.roughness;
    } else if (cfg.type == RSE_MATERIAL_RENDER_EMISSIVE) {
        auto& sub = cfg.config.as<RS_EmissiveMaterialRenderConfig>();
        mat.color = RColor3f(sub.color).srgbToLinear();
        mat.intensity = sub.intensity;
    }
    return mat;
}

}  // namespace rise
