// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_material_config.h"

#include "rise_origin/utils/rs_string.h"
#include "rs_rsc_util.h"

using namespace std;

namespace rise {

void RS_OpaqueMaterialRenderConfig::_validate() const {
    RS_CONFIG_EXPECT_GE(RS_OpaqueMaterialRenderConfig, roughness, 0);
    RS_CONFIG_EXPECT_LE(RS_OpaqueMaterialRenderConfig, roughness, 1);
    RS_CONFIG_EXPECT_GE(RS_OpaqueMaterialRenderConfig, metalness, 0);
    RS_CONFIG_EXPECT_LE(RS_OpaqueMaterialRenderConfig, metalness, 1);
}

void RS_OpaqueMaterialRenderConfig::_read(const RS_ConfigPropertyTree& config_tree) {
    color = readColor3c(config_tree.get<string>("Color"));
    roughness = config_tree.get("Roughness", roughness);
    metalness = config_tree.get("Metalness", metalness);
}

void RS_TransparentMaterialRenderConfig::_validate() const {
    RS_CONFIG_EXPECT_GT(RS_TransparentMaterialRenderConfig, ior, 0);
    RS_CONFIG_EXPECT_GE(RS_TransparentMaterialRenderConfig, roughness, 0);
    RS_CONFIG_EXPECT_LE(RS_TransparentMaterialRenderConfig, roughness, 1);
}

void RS_TransparentMaterialRenderConfig::_read(const RS_ConfigPropertyTree& config_tree) {
    color = readColor3c(config_tree.get<string>("Color"));
    ior = config_tree.get("Ior", ior);
    roughness = config_tree.get("Roughness", roughness);
}

void RS_EmissiveMaterialRenderConfig::_validate() const {
    RS_CONFIG_EXPECT_GE(RS_EmissiveMaterialRenderConfig, intensity, 0);
}

void RS_EmissiveMaterialRenderConfig::_read(const RS_ConfigPropertyTree& config_tree) {
    color = readColor3c(config_tree.get<string>("Color"));
    intensity = config_tree.get("Intensity", intensity);
}

void RS_MaterialRenderConfig::_validate() const {
    if (type == RSE_MATERIAL_RENDER_OPAQUE) {
        RS_CONFIG_EXPECT_ANY_SET(
            RS_MaterialRenderConfig, config, "RS_OpaqueMaterialRenderConfig"
        );
        config.as<RS_OpaqueMaterialRenderConfig>().validate();
    } else if (type == RSE_MATERIAL_RENDER_TRANSPARENT) {
        RS_CONFIG_EXPECT_ANY_SET(
            RS_MaterialRenderConfig, config, "RS_TransparentMaterialRenderConfig"
        );
        config.as<RS_TransparentMaterialRenderConfig>().validate();
    } else if (type == RSE_MATERIAL_RENDER_EMISSIVE) {
        RS_CONFIG_EXPECT_ANY_SET(RS_MaterialRenderConfig, config, "RS_EmissiveMaterialRenderConfig");
        config.as<RS_EmissiveMaterialRenderConfig>().validate();
    }
}

void RS_MaterialRenderConfig::_read(const RS_ConfigPropertyTree& config_tree) {
    auto render_type_str = config_tree.get<string>("Type");
    if (render_type_str == "Opaque") {
        type = RSE_MATERIAL_RENDER_OPAQUE;
        config.as<RS_OpaqueMaterialRenderConfig>().read(config_tree, false);
    } else if (render_type_str == "Transparent") {
        type = RSE_MATERIAL_RENDER_TRANSPARENT;
        config.as<RS_TransparentMaterialRenderConfig>().read(config_tree, false);
    } else if (render_type_str == "Emissive") {
        type = RSE_MATERIAL_RENDER_EMISSIVE;
        config.as<RS_EmissiveMaterialRenderConfig>().read(config_tree, false);
    } else {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid render material type: {}, available types are: Opaque, Transparent, "
            "Emissive",
            render_type_str
        ));
    }
}

void RS_MechanicalMaterialConfig::_validate() const {
    RS_CONFIG_EXPECT_GE(RS_MechanicalMaterialConfig, min_expansion, -1);
    RS_CONFIG_EXPECT_GE(RS_MechanicalMaterialConfig, max_expansion, min_expansion);
    RS_CONFIG_EXPECT_GT(RS_MechanicalMaterialConfig, elastic_modulus, 0);
    RS_CONFIG_EXPECT_GT(RS_MechanicalMaterialConfig, fail_stress, 0);
    RS_CONFIG_EXPECT_GT(RS_MechanicalMaterialConfig, density, 0);
    RS_CONFIG_EXPECT_GE(RS_MechanicalMaterialConfig, poissons_ratio, 0);
    RS_CONFIG_EXPECT_LE(RS_MechanicalMaterialConfig, poissons_ratio, 0.5);
    RS_CONFIG_EXPECT_GE(RS_MechanicalMaterialConfig, u_static, 0);
    RS_CONFIG_EXPECT_LE(RS_MechanicalMaterialConfig, u_static, 1);
    RS_CONFIG_EXPECT_GE(RS_MechanicalMaterialConfig, u_dynamic, 0);
    RS_CONFIG_EXPECT_LE(RS_MechanicalMaterialConfig, u_dynamic, 1);
}

void RS_MechanicalMaterialConfig::_read(const RS_ConfigPropertyTree& config_tree) {
    // root of tree: RSC.Materials(Material[M]).Mechanical
    is_fixed = config_tree.get("IsFixed", is_fixed);
    is_rigid = config_tree.get("IsRigid", is_rigid);
    min_expansion = config_tree.get("MinExpansion", min_expansion);
    max_expansion = config_tree.get("MaxExpansion", max_expansion);
    elastic_modulus = config_tree.get("ElasticModulus", elastic_modulus);
    fail_stress = config_tree.get("FailStress", fail_stress);
    density = config_tree.get("Density", density);
    poissons_ratio = config_tree.get("PoissonsRatio", poissons_ratio);
    u_static = config_tree.get("UStatic", u_static);
    u_dynamic = config_tree.get("UDynamic", u_dynamic);
}

void RS_MaterialConfig::_validate() const {
    render_config.validate();
    mechanical_config.validate();
}

void RS_MaterialConfig::_read(const RS_ConfigPropertyTree& config_tree) {
    // root of tree: RSC.Materials(Material[M])
    name = config_tree.get("Name", name);
    render_config.read(config_tree.getChild("Render"), false);
    mechanical_config.read(config_tree.getChild("Mechanical"), false);
}

}  // namespace rise
