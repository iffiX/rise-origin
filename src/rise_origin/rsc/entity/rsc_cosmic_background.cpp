// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_cosmic_background.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_CosmicBackgroundUniform::operator==(const RSC_CosmicBackgroundUniform &other) const {
    return RSC_CosmicBackgroundBase::operator==(other) &&
           texture_reference == other.texture_reference && intensity == other.intensity &&
           rotation_rad == other.rotation_rad;
}

void RSC_CosmicBackgroundUniform::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CosmicBackgroundUniform::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_CosmicBackgroundUniform, texture_reference);
        RSC_EXPECT_GE(RSC_CosmicBackgroundUniform, intensity, 0.0f);
    });
}

void RSC_CosmicBackgroundUniform::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    texture_reference = tree.get<string>("texture_reference", texture_reference);
    intensity = tree.get("intensity", intensity);
    rotation_rad = tree.get("rotation_rad", rotation_rad);
}

void RSC_CosmicBackgroundUniform::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    tree.add("texture_reference", texture_reference);
    tree.add("intensity", intensity);
    tree.add("rotation_rad", rotation_rad);
}

bool RSC_CosmicBackgroundGradient::operator==(const RSC_CosmicBackgroundGradient &other) const {
    return RSC_CosmicBackgroundBase::operator==(other) && zenith_color == other.zenith_color &&
           horizon_color == other.horizon_color && sun_direction == other.sun_direction &&
           sun_color == other.sun_color && sun_angular_radius == other.sun_angular_radius;
}

void RSC_CosmicBackgroundGradient::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_CosmicBackgroundGradient::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_CosmicBackgroundGradient, sun_angular_radius, 0);
    });
}

void RSC_CosmicBackgroundGradient::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    zenith_color = readColor3b(tree, "zenith_color");
    horizon_color = readColor3b(tree, "horizon_color");
    sun_direction = readVec3rf(tree, "sun_direction");
    sun_color = readColor3b(tree, "sun_color");
    sun_angular_radius = tree.get("sun_angular_radius", sun_angular_radius);
}

void RSC_CosmicBackgroundGradient::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    writeColor3b(tree, "zenith_color", zenith_color);
    writeColor3b(tree, "horizon_color", horizon_color);
    writeVec3rf(tree, "sun_direction", sun_direction);
    writeColor3b(tree, "sun_color", sun_color);
    tree.add("sun_angular_radius", sun_angular_radius);
}

bool RSC_CosmicBackgroundHosekWilkie::operator==(
    const RSC_CosmicBackgroundHosekWilkie &other
) const {
    return RSC_CosmicBackgroundBase::operator==(other) && sun_direction == other.sun_direction &&
           sun_color == other.sun_color && sun_angular_radius == other.sun_angular_radius &&
           turbidity == other.turbidity && ground_color == other.ground_color &&
           intensity == other.intensity;
}

void RSC_CosmicBackgroundHosekWilkie::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_CosmicBackgroundHosekWilkie::_validate", this, [this] {
            RSC_EXPECT_GT(RSC_CosmicBackgroundHosekWilkie, sun_angular_radius, 0);
            RSC_EXPECT_GE(RSC_CosmicBackgroundHosekWilkie, turbidity, 1);
            RSC_EXPECT_LE(RSC_CosmicBackgroundHosekWilkie, turbidity, 10);
            RSC_EXPECT_GE(RSC_CosmicBackgroundHosekWilkie, intensity, 0.0f);
        }
    );
}

void RSC_CosmicBackgroundHosekWilkie::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    sun_direction = readVec3rf(tree, "sun_direction");
    sun_color = readColor3b(tree, "sun_color");
    sun_angular_radius = tree.get("sun_angular_radius", sun_angular_radius);
    turbidity = tree.get("turbidity", turbidity);
    ground_color = readColor3b(tree, "ground_color");
    intensity = tree.get("intensity", intensity);
}

void RSC_CosmicBackgroundHosekWilkie::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    writeVec3rf(tree, "sun_direction", sun_direction);
    writeColor3b(tree, "sun_color", sun_color);
    tree.add("sun_angular_radius", sun_angular_radius);
    tree.add("turbidity", turbidity);
    writeColor3b(tree, "ground_color", ground_color);
    tree.add("intensity", intensity);
}

}  // namespace rsc

}  // namespace rise
