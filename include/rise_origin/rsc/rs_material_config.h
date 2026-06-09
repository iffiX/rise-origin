// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MATERIAL_CONFIG_H
#define RS_MATERIAL_CONFIG_H

#include <string>

#include "rise_origin/utils/rs_any.h"
#include "rise_origin/utils/rs_color.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_base_config.h"

namespace rise {

RS_ENUM(
    RSE_MaterialRenderType,
    unsigned int,
    (RSE_MATERIAL_RENDER_OPAQUE, 0),
    (RSE_MATERIAL_RENDER_TRANSPARENT, 1),
    (RSE_MATERIAL_RENDER_EMISSIVE, 2)
)

class RS_OpaqueMaterialRenderConfig : public RS_BaseConfig {
   public:
    /// Surface base color.
    RColor3b color = RColor3b::zero();

    /// Surface roughness in [0, 1].
    RFloat roughness = 1.0;

    /// Metalness in [0, 1]; > 0 picks the metallic lobe.
    RFloat metalness = 0.0;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("color", color);
        s("roughness", roughness);
        s("metalness", metalness);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_TransparentMaterialRenderConfig : public RS_BaseConfig {
   public:
    /// Per-channel attenuation in [0,1].
    RColor3b color = RColor3b::zero();

    /// Index of refraction; 1.0 = straight-through.
    RFloat ior = 1.0;

    /// Surface roughness in [0, 1]; perturbs refraction/reflection cone.
    RFloat roughness = 1.0;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("color", color);
        s("ior", ior);
        s("roughness", roughness);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_EmissiveMaterialRenderConfig : public RS_BaseConfig {
   public:
    /// Chromaticity of the emitted radiance (sRGB-decoded internally).
    RColor3b color = RColor3b::zero();

    /// Surface radiance in W/m^2/sr; emitted radiance = color * intensity.
    RFloat intensity = 0;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("color", color);
        s("intensity", intensity);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_MaterialRenderConfig : public RS_BaseConfig {
   public:
    RSE_MaterialRenderType type = RSE_MATERIAL_RENDER_OPAQUE;
    RS_Any config = RS_OpaqueMaterialRenderConfig{};

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("type", type);
        if (type == RSE_MATERIAL_RENDER_OPAQUE) {
            s("config", config.as<RS_OpaqueMaterialRenderConfig>());
        } else if (type == RSE_MATERIAL_RENDER_TRANSPARENT) {
            s("config", config.as<RS_TransparentMaterialRenderConfig>());
        } else if (type == RSE_MATERIAL_RENDER_EMISSIVE) {
            s("config", config.as<RS_EmissiveMaterialRenderConfig>());
        }
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_MechanicalMaterialConfig : public RS_BaseConfig {
   public:
    /// A fixed material forms an immovable global anchor. Can be rigid as well.
    bool is_fixed = false;

    /// A rigid material forms a rigid body that joints can constrain.
    bool is_rigid = false;

    /// Min expansion (max contraction) allowed by the material,
    /// 0 means no contraction is allowed and -1 means max
    /// contraction ratio is -100%.
    RFloat min_expansion = -1;

    /// Max expansion allowed by the material, 0 means no expansion
    /// is allowed and 1 means max expansion ratio is 100%.
    RFloat max_expansion = 1;

    /// Elastic modulus of the material (unit: Pa).
    RFloat elastic_modulus = 0;

    /// Maximum failure stress of the material (unit: N/m^2).
    RFloat fail_stress = RS_MAX_FLOAT;

    /// Density of the material (unit: kg/m^3).
    RFloat density = 0;

    /// Possion's ratio of the material.
    RFloat poissons_ratio = 0;

    /// Static friction coefficient of the material.
    RFloat u_static = 0;

    /// Dynamic friction coefficient of the material.
    RFloat u_dynamic = 0;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("is_fixed", is_fixed);
        s("is_rigid", is_rigid);
        s("min_expansion", min_expansion);
        s("max_expansion", max_expansion);
        s("elastic_modulus", elastic_modulus);
        s("fail_stress", fail_stress);
        s("density", density);
        s("poissons_ratio", poissons_ratio);
        s("u_static", u_static);
        s("u_dynamic", u_dynamic);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_MaterialConfig : public RS_BaseConfig {
   public:
    /// Name of the material.
    std::string name;

    RS_MaterialRenderConfig render_config;

    RS_MechanicalMaterialConfig mechanical_config;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("name", name);
        s("render_config", render_config);
        s("mechanical_config", mechanical_config);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

}  // namespace rise
#endif  // RS_MATERIAL_CONFIG_H
