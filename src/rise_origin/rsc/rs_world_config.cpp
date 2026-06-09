// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_world_config.h"

#include "rs_rsc_util.h"

using namespace std;

namespace rise {

void RS_ConstantColorWorldRenderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    color = readColor3c(config_tree.get<string>("Color"));
}

void RS_GradientSkyWorldRenderConfig::_validate() const {
    if (sun_direction.length2() == RFloat(0)) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid sky config, sun_direction must be non-zero"
        ));
    }
    RS_CONFIG_EXPECT_GT(RS_GradientSkyWorldRenderConfig, sun_angular_radius, 0);
    RS_CONFIG_EXPECT_LT(RS_GradientSkyWorldRenderConfig, sun_angular_radius, RFloat(RS_PI / 2));
}

void RS_GradientSkyWorldRenderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    zenith_color = readColor3c(config_tree.get<string>("ZenithColor"));
    horizon_color = readColor3c(config_tree.get<string>("HorizonColor"));
    sun_direction = readVec3rf(config_tree.get<string>("SunDirection"));
    sun_color = readColor3c(config_tree.get<string>("SunColor"));
    sun_angular_radius = config_tree.get("SunAngularRadius", sun_angular_radius);
}

void RS_HosekWilkieSkyWorldRenderConfig::_validate() const {
    if (sun_direction.length2() == RFloat(0)) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid sky config, sun_direction must be non-zero"
        ));
    }
    RS_CONFIG_EXPECT_GT(RS_HosekWilkieSkyWorldRenderConfig, sun_angular_radius, 0);
    RS_CONFIG_EXPECT_LT(RS_HosekWilkieSkyWorldRenderConfig, sun_angular_radius, RFloat(RS_PI / 2));
    RS_CONFIG_EXPECT_GE(RS_HosekWilkieSkyWorldRenderConfig, turbidity, 1);
    RS_CONFIG_EXPECT_LE(RS_HosekWilkieSkyWorldRenderConfig, turbidity, 10);
    RS_CONFIG_EXPECT_GE(RS_HosekWilkieSkyWorldRenderConfig, intensity, 0);
}

void RS_HosekWilkieSkyWorldRenderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    sun_direction = readVec3rf(config_tree.get<string>("SunDirection"));
    sun_color = readColor3c(config_tree.get<string>("SunColor"));
    sun_angular_radius = config_tree.get("SunAngularRadius", sun_angular_radius);
    turbidity = config_tree.get("Turbidity", turbidity);
    ground_color = readColor3c(config_tree.get<string>("GroundColor"));
    intensity = config_tree.get("Intensity", intensity);
}

void RS_WorldRenderConfig::_validate() const {
    if (type == RSE_WORLD_RENDER_CONSTANT_COLOR) {
        RS_CONFIG_EXPECT_ANY_SET(RS_WorldRenderConfig, config, "RS_ConstantColorWorldRenderConfig");
        config.as<RS_ConstantColorWorldRenderConfig>().validate();
    } else if (type == RSE_WORLD_RENDER_GRADIENT) {
        RS_CONFIG_EXPECT_ANY_SET(RS_WorldRenderConfig, config, "RS_GradientSkyWorldRenderConfig");
        config.as<RS_GradientSkyWorldRenderConfig>().validate();
    }
}

void RS_WorldRenderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    auto render_type_str = config_tree.get<string>("Type");
    if (render_type_str == "ConstantColor") {
        type = RSE_WORLD_RENDER_CONSTANT_COLOR;
        config.as<RS_ConstantColorWorldRenderConfig>().read(config_tree, false);
    } else if (render_type_str == "Gradient") {
        type = RSE_WORLD_RENDER_GRADIENT;
        config.as<RS_GradientSkyWorldRenderConfig>().read(config_tree, false);
    } else if (render_type_str == "HosekWilkie") {
        type = RSE_WORLD_RENDER_HOSEK_WILKIE;
        config.as<RS_HosekWilkieSkyWorldRenderConfig>().read(config_tree, false);
    } else {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid world render type: {}, available types are: ConstantColor, Gradient, "
            "HosekWilkie",
            render_type_str
        ));
    }
}

void RS_WorldConfig::_validate() const { render_config.validate(); }

void RS_WorldConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.World
    grav_acc = config_tree.get("GravAcc", grav_acc);
    render_config.read(config_tree.getChild("Render"), false);
}

}  // namespace rise