// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_cosmic_background.h"

#include "rise_origin/rsc/rs_rsc_util.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_CosmicBackgroundUniform::operator==(const RSC_CosmicBackgroundUniform &other) const {
    return RSC_CosmicBackgroundBase::operator==(other) &&
           texture_reference == other.texture_reference && intensity == other.intensity &&
           rotation_rad == other.rotation_rad;
}

void RSC_CosmicBackgroundUniform::_validate() const {
    RSC_EntityBase::_validate();
    RSC_EXPECT_NOT_EMPTY(RSC_CosmicBackgroundUniform, texture_reference);
    RSC_EXPECT_GE(RSC_CosmicBackgroundUniform, intensity, 0.0f);
}

void RSC_CosmicBackgroundUniform::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_EntityBase::_read(rsc_tree);
    texture_reference = rsc_tree.get<string>("TextureReference", texture_reference);
    intensity = rsc_tree.get("Intensity", intensity);
    rotation_rad = rsc_tree.get("Rotation", rotation_rad);
}

bool RSC_CosmicBackgroundGradient::operator==(const RSC_CosmicBackgroundGradient &other) const {
    return RSC_CosmicBackgroundBase::operator==(other) && zenith_color == other.zenith_color &&
           horizon_color == other.horizon_color && sun_direction == other.sun_direction &&
           sun_color == other.sun_color && sun_angular_radius == other.sun_angular_radius;
}

void RSC_CosmicBackgroundGradient::_validate() const {
    RSC_EntityBase::_validate();
    RSC_EXPECT_GT(RSC_CosmicBackgroundGradient, sun_angular_radius, 0);
}

void RSC_CosmicBackgroundGradient::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_EntityBase::_read(rsc_tree);
    zenith_color = readColor3b(rsc_tree.get<string>("ZenithColor"));
    horizon_color = readColor3b(rsc_tree.get<string>("HorizonColor"));
    sun_direction = readVec3rf(rsc_tree.get<string>("SunDirection"));
    sun_color = readColor3b(rsc_tree.get<string>("SunColor"));
    sun_angular_radius = rsc_tree.get("SunAngularRadius", sun_angular_radius);
}

bool RSC_CosmicBackgroundHosekWilkie::operator==(
    const RSC_CosmicBackgroundHosekWilkie &other
) const {
    return RSC_CosmicBackgroundBase::operator==(other) && sun_direction == other.sun_direction &&
           sun_color == other.sun_color && sun_angular_radius == other.sun_angular_radius &&
           turbidity == other.turbidity && ground_color == other.ground_color &&
           intensity == other.intensity;
}

void RSC_CosmicBackgroundHosekWilkie::_validate() const {
    RSC_EntityBase::_validate();
    RSC_EXPECT_GT(RSC_CosmicBackgroundHosekWilkie, sun_angular_radius, 0);
    RSC_EXPECT_GE(RSC_CosmicBackgroundHosekWilkie, turbidity, 1);
    RSC_EXPECT_LE(RSC_CosmicBackgroundHosekWilkie, turbidity, 10);
    RSC_EXPECT_GE(RSC_CosmicBackgroundHosekWilkie, intensity, 0.0f);
}

void RSC_CosmicBackgroundHosekWilkie::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_EntityBase::_read(rsc_tree);
    sun_direction = readVec3rf(rsc_tree.get<string>("SunDirection"));
    sun_color = readColor3b(rsc_tree.get<string>("SunColor"));
    sun_angular_radius = rsc_tree.get("SunAngularRadius", sun_angular_radius);
    turbidity = rsc_tree.get("Turbidity", turbidity);
    ground_color = readColor3b(rsc_tree.get<string>("GroundColor"));
    intensity = rsc_tree.get("Intensity", intensity);
}

}  // namespace rsc

}  // namespace rise
