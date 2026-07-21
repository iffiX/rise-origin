// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/essence/rsc_material_unary_property.h"

#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/utils/string/rs_string.h"

using namespace std;

namespace rise {

namespace rsc {

void RSC_MaterialUnaryProperty::_deserialize(RS_RSCArchive& archive) {
    RSC_EssenceBase::_deserialize(archive);
    archive.deserialize("material_reference", material_reference);
}

void RSC_MaterialUnaryProperty::_serialize(RS_RSCArchive& archive) const {
    RSC_EssenceBase::_serialize(archive);
    archive.serialize("material_reference", material_reference);
}

void RSC_MaterialPhysicalPropertyBase::_deserialize(RS_RSCArchive& archive) {
    RSC_MaterialUnaryProperty::_deserialize(archive);
}

void RSC_MaterialPhysicalPropertyBase::_serialize(RS_RSCArchive& archive) const {
    RSC_MaterialUnaryProperty::_serialize(archive);
}

bool RSC_MaterialPhysicalPropertyFixed::operator==(
    const RSC_MaterialPhysicalPropertyFixed& other
) const {
    return RSC_MaterialPhysicalPropertyBase::operator==(other);
}

void RSC_MaterialPhysicalPropertyFixed::_read(const RS_RSCFileTreeReader& tree) {
    RSC_EssenceBase::_read(tree);
    material_reference = tree.get("material_reference", material_reference);
}

void RSC_MaterialPhysicalPropertyFixed::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_EssenceBase::_write(tree);
    tree.add("material_reference", material_reference);
}

void RSC_MaterialPhysicalPropertyFixed::_deserialize(RS_RSCArchive& archive) {
    RSC_MaterialPhysicalPropertyBase::_deserialize(archive);
}

void RSC_MaterialPhysicalPropertyFixed::_serialize(RS_RSCArchive& archive) const {
    RSC_MaterialPhysicalPropertyBase::_serialize(archive);
}

bool RSC_MaterialPhysicalPropertySoft::operator==(
    const RSC_MaterialPhysicalPropertySoft& other
) const {
    return RSC_MaterialPhysicalPropertyBase::operator==(other) &&
           elastic_modulus == other.elastic_modulus && fail_stress == other.fail_stress &&
           density == other.density && poissons_ratio == other.poissons_ratio &&
           min_expansion == other.min_expansion && max_expansion == other.max_expansion;
}

void RSC_MaterialPhysicalPropertySoft::_validate(RS_RSCValidator& validator) const {
    RSC_EssenceBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_MaterialPhysicalPropertySoft::_validate", this, [this] {
            RSC_EXPECT_GT(RSC_MaterialPhysicalPropertySoft, elastic_modulus, 0);
            RSC_EXPECT_GT(RSC_MaterialPhysicalPropertySoft, fail_stress, 0);
            RSC_EXPECT_GT(RSC_MaterialPhysicalPropertySoft, density, 0);
            RSC_EXPECT_GE(RSC_MaterialPhysicalPropertySoft, poissons_ratio, 0);
            RSC_EXPECT_LE(RSC_MaterialPhysicalPropertySoft, poissons_ratio, 0.5);
            RSC_EXPECT_GE(RSC_MaterialPhysicalPropertySoft, min_expansion, -1);
            RSC_EXPECT_GE(RSC_MaterialPhysicalPropertySoft, max_expansion, min_expansion);
        }
    );
}

void RSC_MaterialPhysicalPropertySoft::_read(const RS_RSCFileTreeReader& tree) {
    RSC_EssenceBase::_read(tree);
    material_reference = tree.get("material_reference", material_reference);
    elastic_modulus = tree.get("elastic_modulus", elastic_modulus);
    fail_stress = tree.get("fail_stress", fail_stress);
    density = tree.get("density", density);
    poissons_ratio = tree.get("poissons_ratio", poissons_ratio);
    min_expansion = tree.get("min_expansion", min_expansion);
    max_expansion = tree.get("max_expansion", max_expansion);
}

void RSC_MaterialPhysicalPropertySoft::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_EssenceBase::_write(tree);
    tree.add("material_reference", material_reference);
    tree.add("elastic_modulus", elastic_modulus);
    tree.add("fail_stress", fail_stress);
    tree.add("density", density);
    tree.add("poissons_ratio", poissons_ratio);
    tree.add("min_expansion", min_expansion);
    tree.add("max_expansion", max_expansion);
}

void RSC_MaterialPhysicalPropertySoft::_deserialize(RS_RSCArchive& archive) {
    RSC_MaterialPhysicalPropertyBase::_deserialize(archive);
    archive.deserialize("elastic_modulus", elastic_modulus);
    archive.deserialize("fail_stress", fail_stress);
    archive.deserialize("density", density);
    archive.deserialize("poissons_ratio", poissons_ratio);
    archive.deserialize("min_expansion", min_expansion);
    archive.deserialize("max_expansion", max_expansion);
}

void RSC_MaterialPhysicalPropertySoft::_serialize(RS_RSCArchive& archive) const {
    RSC_MaterialPhysicalPropertyBase::_serialize(archive);
    archive.serialize("elastic_modulus", elastic_modulus);
    archive.serialize("fail_stress", fail_stress);
    archive.serialize("density", density);
    archive.serialize("poissons_ratio", poissons_ratio);
    archive.serialize("min_expansion", min_expansion);
    archive.serialize("max_expansion", max_expansion);
}

bool RSC_MaterialPhysicalPropertyRigid::operator==(
    const RSC_MaterialPhysicalPropertyRigid& other
) const {
    return RSC_MaterialPhysicalPropertyBase::operator==(other) && density == other.density;
}

void RSC_MaterialPhysicalPropertyRigid::_validate(RS_RSCValidator& validator) const {
    RSC_EssenceBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_MaterialPhysicalPropertyRigid::_validate", this, [this] {
            RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyRigid, density, 0);
        }
    );
}

void RSC_MaterialPhysicalPropertyRigid::_read(const RS_RSCFileTreeReader& tree) {
    RSC_EssenceBase::_read(tree);
    material_reference = tree.get("material_reference", material_reference);
    density = tree.get("density", density);
}

void RSC_MaterialPhysicalPropertyRigid::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_EssenceBase::_write(tree);
    tree.add("material_reference", material_reference);
    tree.add("density", density);
}

void RSC_MaterialPhysicalPropertyRigid::_deserialize(RS_RSCArchive& archive) {
    RSC_MaterialPhysicalPropertyBase::_deserialize(archive);
    archive.deserialize("density", density);
}

void RSC_MaterialPhysicalPropertyRigid::_serialize(RS_RSCArchive& archive) const {
    RSC_MaterialPhysicalPropertyBase::_serialize(archive);
    archive.serialize("density", density);
}

bool RSC_MaterialPhysicalPropertyAtlas::operator==(
    const RSC_MaterialPhysicalPropertyAtlas& other
) const {
    return RSC_MaterialPhysicalPropertyBase::operator==(other) && mass == other.mass &&
           com_position == other.com_position && inertia == other.inertia;
}

void RSC_MaterialPhysicalPropertyAtlas::_validate(RS_RSCValidator& validator) const {
    RSC_EssenceBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_MaterialPhysicalPropertyAtlas::_validate", this, [this] {
            RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyAtlas, mass, 0);
            RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyAtlas, inertia[0][0], 0);
            RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyAtlas, inertia[1][1], 0);
            RSC_EXPECT_GT(RSC_MaterialPhysicalPropertyAtlas, inertia[2][2], 0);
        }
    );
}

void RSC_MaterialPhysicalPropertyAtlas::_read(const RS_RSCFileTreeReader& tree) {
    RSC_EssenceBase::_read(tree);
    material_reference = tree.get("material_reference", material_reference);
    mass = tree.get("mass", mass);
    com_position = readVec3rf(tree, "com_position", com_position);
    inertia = readMat3rf(tree, "inertia");
}

void RSC_MaterialPhysicalPropertyAtlas::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_EssenceBase::_write(tree);
    tree.add("material_reference", material_reference);
    tree.add("mass", mass);
    writeVec3rf(tree, "com_position", com_position);
    writeMat3rf(tree, "inertia", inertia);
}

void RSC_MaterialPhysicalPropertyAtlas::_deserialize(RS_RSCArchive& archive) {
    RSC_MaterialPhysicalPropertyBase::_deserialize(archive);
    archive.deserialize("mass", mass);
    archive.deserialize("com_position", com_position);
    archive.deserialize("inertia", inertia);
}

void RSC_MaterialPhysicalPropertyAtlas::_serialize(RS_RSCArchive& archive) const {
    RSC_MaterialPhysicalPropertyBase::_serialize(archive);
    archive.serialize("mass", mass);
    archive.serialize("com_position", com_position);
    archive.serialize("inertia", inertia);
}

void RSC_MaterialOpticalPropertyBase::_deserialize(RS_RSCArchive& archive) {
    RSC_MaterialUnaryProperty::_deserialize(archive);
}

void RSC_MaterialOpticalPropertyBase::_serialize(RS_RSCArchive& archive) const {
    RSC_MaterialUnaryProperty::_serialize(archive);
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

void RSC_MaterialOpticalPropertySimpleMicrofacet::_read(const RS_RSCFileTreeReader& tree) {
    RSC_EssenceBase::_read(tree);
    material_reference = tree.get("material_reference", material_reference);
    albedo_texture_reference = tree.get("albedo_texture_reference", albedo_texture_reference);
    roughness_metalness_texture_reference =
        tree.get("roughness_metalness_texture_reference", roughness_metalness_texture_reference);
    is_transparent = tree.get("is_transparent", is_transparent);
    ior = tree.get("ior", ior);
    emission_texture_reference = tree.get("emission_texture_reference", emission_texture_reference);
    emission_intensity = tree.get("emission_intensity", emission_intensity);
}

void RSC_MaterialOpticalPropertySimpleMicrofacet::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_EssenceBase::_write(tree);
    tree.add("material_reference", material_reference);
    tree.add("albedo_texture_reference", albedo_texture_reference);
    tree.add("roughness_metalness_texture_reference", roughness_metalness_texture_reference);
    tree.add("is_transparent", is_transparent);
    tree.add("ior", ior);
    tree.add("emission_texture_reference", emission_texture_reference);
    tree.add("emission_intensity", emission_intensity);
}

void RSC_MaterialOpticalPropertySimpleMicrofacet::_deserialize(RS_RSCArchive& archive) {
    RSC_MaterialOpticalPropertyBase::_deserialize(archive);
    archive.deserialize("albedo_texture_reference", albedo_texture_reference);
    archive.deserialize(
        "roughness_metalness_texture_reference", roughness_metalness_texture_reference
    );
    archive.deserialize("is_transparent", is_transparent);
    archive.deserialize("ior", ior);
    archive.deserialize("emission_texture_reference", emission_texture_reference);
    archive.deserialize("emission_intensity", emission_intensity);
}

void RSC_MaterialOpticalPropertySimpleMicrofacet::_serialize(RS_RSCArchive& archive) const {
    RSC_MaterialOpticalPropertyBase::_serialize(archive);
    archive.serialize("albedo_texture_reference", albedo_texture_reference);
    archive.serialize(
        "roughness_metalness_texture_reference", roughness_metalness_texture_reference
    );
    archive.serialize("is_transparent", is_transparent);
    archive.serialize("ior", ior);
    archive.serialize("emission_texture_reference", emission_texture_reference);
    archive.serialize("emission_intensity", emission_intensity);
}

}  // namespace rsc

}  // namespace rise
