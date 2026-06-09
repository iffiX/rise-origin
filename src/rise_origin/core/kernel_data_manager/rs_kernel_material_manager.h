// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_MATERIAL_MANAGER_H
#define RS_KERNEL_MATERIAL_MANAGER_H

#include "rise_origin/logic/vision/rs_vision_material_m.h"
#include "rise_origin/rsc/rs_material_config.h"

namespace rise {

class RS_KernelManager;

class RS_KernelMaterialManager {
   public:
    RS_KernelMaterialManager() = default;

    void init(RS_KernelManager *kernel_manager, const RS_MaterialConfig &config);

    void update(const RS_MaterialConfig &config);

    void free();

    std::string getName() const;

    [[nodiscard]] const RS_MaterialConfig &getConfig() const;

   private:
    RS_KernelManager *_kernel_manager = nullptr;

    RS_MaterialConfig _config;

    RIndex _material_gid = RS_NULL_INDEX;

    void _initKernelData();

    void _updateKernelData();

    void _freeKernelData();

    static RSE_VisionMaterialType _toVisionMaterialType(RSE_MaterialRenderType config_type);

    template <RSE_FloatType FloatMemAccuracy>
    static logic::RS_VisionMaterialMem<FloatMemAccuracy> _buildVisionMaterial(
        const RS_MaterialRenderConfig &cfg
    );
};

}  // namespace rise

#endif  // RS_KERNEL_MATERIAL_MANAGER_H
