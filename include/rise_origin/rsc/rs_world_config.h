// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_CONFIG_H
#define RS_WORLD_CONFIG_H

#include "rise_origin/utils/rs_any.h"
#include "rise_origin/utils/rs_color.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_vec.h"
#include "rs_base_config.h"

namespace rise {

RS_ENUM(
    RSE_WorldRenderType,
    unsigned int,
    (RSE_WORLD_RENDER_CONSTANT_COLOR, 0),
    (RSE_WORLD_RENDER_GRADIENT, 1),
    (RSE_WORLD_RENDER_HOSEK_WILKIE, 2)
)

class RS_ConstantColorWorldRenderConfig : public RS_BaseConfig {
   public:
    /// Sky color.
    RColor3b color = RColor3b::zero();

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _read(const RS_ConfigPropertyTree &config_tree) override;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("color", color);
    }
};

class RS_GradientSkyWorldRenderConfig : public RS_BaseConfig {
   public:
    /// Zenith color; blended with horizon_color by |dir.z|.
    RColor3b zenith_color = RColor3b::zero();

    /// Horizon color; blended with zenith_color by |dir.z|.
    RColor3b horizon_color = RColor3b::zero();

    /// Sun direction in world space; must be non-zero.
    RVec3rf sun_direction = RVec3rf(0.3, 0.4, -1.0);

    /// Sun disk color along sun_direction.
    RColor3b sun_color = RColor3b::zero();

    /// Sun disk angular radius in radians, range (0, pi/2); larger = bigger sun.
    RFloat sun_angular_radius = 0.00465;

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _validate() const override;

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _read(const RS_ConfigPropertyTree &config_tree) override;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("zenith_color", zenith_color);
        s("horizon_color", horizon_color);
        s("sun_direction", sun_direction);
        s("sun_color", sun_color);
        s("sun_angular_radius", sun_angular_radius);
    }
};

class RS_HosekWilkieSkyWorldRenderConfig : public RS_BaseConfig {
   public:
    /// Sun direction in world space; must be non-zero. Pointing toward the sun.
    RVec3rf sun_direction = RVec3rf(0.3, 0.4, 0.866);

    /// Sun disk color along sun_direction.
    RColor3b sun_color = RColor3b::zero();

    /// Sun disk angular radius in radians, range (0, pi/2); larger = bigger sun.
    RFloat sun_angular_radius = 0.00465;

    /// Atmospheric turbidity in [1, 10]; 1 is pure Rayleigh, 10 is hazy.
    RFloat turbidity = 3.0;

    /// Ground color in [0, 1] per channel; feeds bounce light back into the sky.
    RColor3b ground_color = RColor3b::zero();

    /// Sky radiance multiplier; scales the final analytic radiance.
    RFloat intensity = 1.0;

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _validate() const override;

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _read(const RS_ConfigPropertyTree &config_tree) override;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("sun_direction", sun_direction);
        s("sun_color", sun_color);
        s("sun_angular_radius", sun_angular_radius);
        s("turbidity", turbidity);
        s("ground_color", ground_color);
        s("intensity", intensity);
    }
};

class RS_WorldRenderConfig : public RS_BaseConfig {
   public:
    RSE_WorldRenderType type = RSE_WORLD_RENDER_CONSTANT_COLOR;
    RS_Any config = RS_ConstantColorWorldRenderConfig{};

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _validate() const override;

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _read(const RS_ConfigPropertyTree &config_tree) override;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("type", type);
        if (type == RSE_WORLD_RENDER_CONSTANT_COLOR) {
            s("config", config.as<RS_ConstantColorWorldRenderConfig>());
        } else if (type == RSE_WORLD_RENDER_GRADIENT) {
            s("config", config.as<RS_GradientSkyWorldRenderConfig>());
        } else if (type == RSE_WORLD_RENDER_HOSEK_WILKIE) {
            s("config", config.as<RS_HosekWilkieSkyWorldRenderConfig>());
        }
    }
};

class RS_WorldConfig : public RS_BaseConfig {
   public:
    // In RSC.World.Gravity
    /// Gravity acceleration (unit: m/s^2).
    RFloat grav_acc = -9.81;

    RS_WorldRenderConfig render_config;

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _validate() const override;

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _read(const RS_ConfigPropertyTree &config_tree) override;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("grav_acc", grav_acc);
        s("render_config", render_config);
    }
};

}  // namespace rise

#endif  // RS_WORLD_CONFIG_H
