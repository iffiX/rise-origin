// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_cosmic_background.h"

#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"

using namespace std;

namespace rise {

namespace rsc {

void RSC_CosmicBackgroundBase::_deserialize(RS_RSCArchive &archive) {
    RSC_EntityBase::_deserialize(archive);
}

void RSC_CosmicBackgroundBase::_serialize(RS_RSCArchive &archive) const {
    RSC_EntityBase::_serialize(archive);
}

bool RSC_CosmicBackgroundUniform::operator==(const RSC_CosmicBackgroundUniform &other) const {
    return RSC_CosmicBackgroundBase::operator==(other) &&
           texture_reference == other.texture_reference && radiance == other.radiance &&
           rotation_rad == other.rotation_rad;
}

void RSC_CosmicBackgroundUniform::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CosmicBackgroundUniform::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_CosmicBackgroundUniform, texture_reference);
        RSC_EXPECT_GE(RSC_CosmicBackgroundUniform, radiance, 0.0f);
    });
}

void RSC_CosmicBackgroundUniform::_read(const RS_RSCFileTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    texture_reference = tree.get<string>("texture_reference", texture_reference);
    radiance = tree.get("radiance", radiance);
    rotation_rad = tree.get("rotation_rad", rotation_rad);
}

void RSC_CosmicBackgroundUniform::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    tree.add("texture_reference", texture_reference);
    tree.add("radiance", radiance);
    tree.add("rotation_rad", rotation_rad);
}

void RSC_CosmicBackgroundUniform::_deserialize(RS_RSCArchive &archive) {
    RSC_CosmicBackgroundBase::_deserialize(archive);
    archive.deserialize("texture_reference", texture_reference);
    archive.deserialize("radiance", radiance);
    archive.deserialize("rotation_rad", rotation_rad);
}

void RSC_CosmicBackgroundUniform::_serialize(RS_RSCArchive &archive) const {
    RSC_CosmicBackgroundBase::_serialize(archive);
    archive.serialize("texture_reference", texture_reference);
    archive.serialize("radiance", radiance);
    archive.serialize("rotation_rad", rotation_rad);
}

bool RSC_CosmicBackgroundGradient::operator==(const RSC_CosmicBackgroundGradient &other) const {
    return RSC_CosmicBackgroundBase::operator==(other) && zenith_color == other.zenith_color &&
           horizon_color == other.horizon_color && sun_direction == other.sun_direction &&
           sun_color == other.sun_color && sun_irradiance == other.sun_irradiance &&
           sun_angular_radius == other.sun_angular_radius;
}

void RSC_CosmicBackgroundGradient::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CosmicBackgroundGradient::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_CosmicBackgroundGradient, sun_angular_radius, 0);
        RSC_EXPECT_GE(RSC_CosmicBackgroundGradient, sun_irradiance, 0);
    });
}

void RSC_CosmicBackgroundGradient::_read(const RS_RSCFileTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    zenith_color = readColor3b(tree, "zenith_color");
    horizon_color = readColor3b(tree, "horizon_color");
    sun_direction = readVec3rf(tree, "sun_direction");
    sun_color = readColor3b(tree, "sun_color");
    sun_irradiance = tree.get("sun_irradiance", sun_irradiance);
    sun_angular_radius = tree.get("sun_angular_radius", sun_angular_radius);
}

void RSC_CosmicBackgroundGradient::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    writeColor3b(tree, "zenith_color", zenith_color);
    writeColor3b(tree, "horizon_color", horizon_color);
    writeVec3rf(tree, "sun_direction", sun_direction);
    writeColor3b(tree, "sun_color", sun_color);
    tree.add("sun_irradiance", sun_irradiance);
    tree.add("sun_angular_radius", sun_angular_radius);
}

void RSC_CosmicBackgroundGradient::_deserialize(RS_RSCArchive &archive) {
    RSC_CosmicBackgroundBase::_deserialize(archive);
    archive.deserialize("zenith_color", zenith_color);
    archive.deserialize("horizon_color", horizon_color);
    archive.deserialize("sun_direction", sun_direction);
    archive.deserialize("sun_color", sun_color);
    archive.deserialize("sun_irradiance", sun_irradiance);
    archive.deserialize("sun_angular_radius", sun_angular_radius);
}

void RSC_CosmicBackgroundGradient::_serialize(RS_RSCArchive &archive) const {
    RSC_CosmicBackgroundBase::_serialize(archive);
    archive.serialize("zenith_color", zenith_color);
    archive.serialize("horizon_color", horizon_color);
    archive.serialize("sun_direction", sun_direction);
    archive.serialize("sun_color", sun_color);
    archive.serialize("sun_irradiance", sun_irradiance);
    archive.serialize("sun_angular_radius", sun_angular_radius);
}

bool RSC_CosmicBackgroundHosekWilkie::operator==(
    const RSC_CosmicBackgroundHosekWilkie &other
) const {
    return RSC_CosmicBackgroundBase::operator==(other) && sun_direction == other.sun_direction &&
           sun_color == other.sun_color && sun_irradiance == other.sun_irradiance &&
           sun_angular_radius == other.sun_angular_radius &&
           turbidity == other.turbidity && ground_color == other.ground_color &&
           radiance_scale == other.radiance_scale;
}

void RSC_CosmicBackgroundHosekWilkie::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_CosmicBackgroundHosekWilkie::_validate", this, [this] {
            RSC_EXPECT_GT(RSC_CosmicBackgroundHosekWilkie, sun_angular_radius, 0);
            RSC_EXPECT_GE(RSC_CosmicBackgroundHosekWilkie, sun_irradiance, 0);
            RSC_EXPECT_GE(RSC_CosmicBackgroundHosekWilkie, turbidity, 1);
            RSC_EXPECT_LE(RSC_CosmicBackgroundHosekWilkie, turbidity, 10);
            RSC_EXPECT_GE(RSC_CosmicBackgroundHosekWilkie, radiance_scale, 0.0f);
        }
    );
}

void RSC_CosmicBackgroundHosekWilkie::_read(const RS_RSCFileTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    sun_direction = readVec3rf(tree, "sun_direction");
    sun_color = readColor3b(tree, "sun_color");
    sun_irradiance = tree.get("sun_irradiance", sun_irradiance);
    sun_angular_radius = tree.get("sun_angular_radius", sun_angular_radius);
    turbidity = tree.get("turbidity", turbidity);
    ground_color = readColor3b(tree, "ground_color");
    radiance_scale = tree.get("radiance_scale", radiance_scale);
}

void RSC_CosmicBackgroundHosekWilkie::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    writeVec3rf(tree, "sun_direction", sun_direction);
    writeColor3b(tree, "sun_color", sun_color);
    tree.add("sun_irradiance", sun_irradiance);
    tree.add("sun_angular_radius", sun_angular_radius);
    tree.add("turbidity", turbidity);
    writeColor3b(tree, "ground_color", ground_color);
    tree.add("radiance_scale", radiance_scale);
}

void RSC_CosmicBackgroundHosekWilkie::_deserialize(RS_RSCArchive &archive) {
    RSC_CosmicBackgroundBase::_deserialize(archive);
    archive.deserialize("sun_direction", sun_direction);
    archive.deserialize("sun_color", sun_color);
    archive.deserialize("sun_irradiance", sun_irradiance);
    archive.deserialize("sun_angular_radius", sun_angular_radius);
    archive.deserialize("turbidity", turbidity);
    archive.deserialize("ground_color", ground_color);
    archive.deserialize("radiance_scale", radiance_scale);
}

void RSC_CosmicBackgroundHosekWilkie::_serialize(RS_RSCArchive &archive) const {
    RSC_CosmicBackgroundBase::_serialize(archive);
    archive.serialize("sun_direction", sun_direction);
    archive.serialize("sun_color", sun_color);
    archive.serialize("sun_irradiance", sun_irradiance);
    archive.serialize("sun_angular_radius", sun_angular_radius);
    archive.serialize("turbidity", turbidity);
    archive.serialize("ground_color", ground_color);
    archive.serialize("radiance_scale", radiance_scale);
}

}  // namespace rsc

}  // namespace rise
