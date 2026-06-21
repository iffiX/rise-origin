// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/essence/rsc_material_unary_property.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/utils/string/rs_string.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_MaterialPhysicalPropertyFixed::operator==(
    const RSC_MaterialPhysicalPropertyFixed& other
) const {
    return RSC_MaterialPhysicalPropertyBase::operator==(other);
}

void RSC_MaterialPhysicalPropertyFixed::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_EssenceBase::_read(rsc_tree);
    material_reference = rsc_tree.get("MaterialReference", material_reference);
}

bool RSC_MaterialPhysicalPropertySoft::operator==(
    const RSC_MaterialPhysicalPropertySoft& other
) const {
    return RSC_MaterialPhysicalPropertyBase::operator==(other) &&
           elastic_modulus == other.elastic_modulus && fail_stress == other.fail_stress &&
           density == other.density && poissons_ratio == other.poissons_ratio &&
           min_expansion == other.min_expansion && max_expansion == other.max_expansion;
}

void RSC_MaterialPhysicalPropertySoft::_validate() const {
    RSC_EssenceBase::_validate();
    RSC_EXPECT_GT(RSC_MaterialPhysicalPropertySoft, elastic_modulus, 0);
    RSC_EXPECT_GT(RSC_MaterialPhysicalPropertySoft, fail_stress, 0);
    RSC_EXPECT_GT(RSC_MaterialPhysicalPropertySoft, density, 0);
    RSC_EXPECT_GE(RSC_MaterialPhysicalPropertySoft, poissons_ratio, 0);
    RSC_EXPECT_LE(RSC_MaterialPhysicalPropertySoft, poissons_ratio, 0.5);
    RSC_EXPECT_GE(RSC_MaterialPhysicalPropertySoft, min_expansion, -1);
    RSC_EXPECT_GE(RSC_MaterialPhysicalPropertySoft, max_expansion, min_expansion);
}

void RSC_MaterialPhysicalPropertySoft::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_EssenceBase::_read(rsc_tree);
    material_reference = rsc_tree.get("MaterialReference", material_reference);
    elastic_modulus = rsc_tree.get("ElasticModulus", elastic_modulus);
    fail_stress = rsc_tree.get("FailStress", fail_stress);
    density = rsc_tree.get("Density", density);
    poissons_ratio = rsc_tree.get("PoissonsRatio", poissons_ratio);
    min_expansion = rsc_tree.get("MinExpansion", min_expansion);
    max_expansion = rsc_tree.get("MaxExpansion", max_expansion);
}

bool RSC_MaterialPhysicalPropertyRigid::operator==(
    const RSC_MaterialPhysicalPropertyRigid& other
) const {
    return RSC_MaterialPhysicalPropertyBase::operator==(other) && density == other.density;
}

void RSC_MaterialPhysicalPropertyRigid::_validate() const {
    RSC_EssenceBase::_validate();
    RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyRigid, density, 0);
}

void RSC_MaterialPhysicalPropertyRigid::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_EssenceBase::_read(rsc_tree);
    material_reference = rsc_tree.get("MaterialReference", material_reference);
    density = rsc_tree.get("Density", density);
}

bool RSC_MaterialPhysicalPropertyAtlas::operator==(
    const RSC_MaterialPhysicalPropertyAtlas& other
) const {
    return RSC_MaterialPhysicalPropertyBase::operator==(other) && mass == other.mass &&
           com_position == other.com_position && inertia == other.inertia;
}

void RSC_MaterialPhysicalPropertyAtlas::_validate() const {
    RSC_EssenceBase::_validate();
    RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyAtlas, mass, 0);
    RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyAtlas, inertia[0][0], 0);
    RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyAtlas, inertia[1][1], 0);
    RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyAtlas, inertia[2][2], 0);
}

void RSC_MaterialPhysicalPropertyAtlas::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_EssenceBase::_read(rsc_tree);
    material_reference = rsc_tree.get("MaterialReference", material_reference);
    mass = rsc_tree.get("Mass", mass);
    com_position = readVec3rf(rsc_tree.get("COMPosition", string("0,0,0")));
    inertia = readMat3rf(rsc_tree.get("Inertia", string("0,0,0,0,0,0,0,0,0")));
}

bool RSC_MaterialOpticalPropertySimpleMicrofacet::operator==(
    const RSC_MaterialOpticalPropertySimpleMicrofacet& other
) const {
    return RSC_MaterialOpticalPropertyBase::operator==(other) &&
           albedo_texture_reference == other.albedo_texture_reference &&
           roughness_metalness_texture_reference == other.roughness_metalness_texture_reference &&
           is_transparent == other.is_transparent && ior == other.ior &&
           emission_texture_reference == other.emission_texture_reference &&
           emission_intensity == other.emission_intensity;
}

void RSC_MaterialOpticalPropertySimpleMicrofacet::_read(
    const RS_CharacterizationPropertyTree& rsc_tree
) {
    RSC_EssenceBase::_read(rsc_tree);
    material_reference = rsc_tree.get("MaterialReference", material_reference);
    albedo_texture_reference = rsc_tree.get("AlbedoTextureReference", albedo_texture_reference);
    roughness_metalness_texture_reference =
        rsc_tree.get("RoughnessMetalnessTextureReference", roughness_metalness_texture_reference);
    is_transparent = rsc_tree.get("IsTransparent", is_transparent);
    ior = rsc_tree.get("IOR", ior);
    emission_texture_reference =
        rsc_tree.get("EmissionTextureReference", emission_texture_reference);
    emission_intensity = rsc_tree.get("EmissionIntensity", emission_intensity);
}

}  // namespace rsc

}  // namespace rise
