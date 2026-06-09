// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_FLOOR_CONFIG_H
#define RS_FLOOR_CONFIG_H

#include <vector>

#include "rise_origin/utils/rs_any.h"
#include "rise_origin/utils/rs_color.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_base_config.h"

namespace rise {

RS_ENUM(
    RSE_FloorDataType,
    unsigned int,
    (RSE_FLOOR_DATA_NONE, 0),
    (RSE_FLOOR_DATA_FLAT, 1),
    (RSE_FLOOR_DATA_ARRAY, 2)
)

class RS_ArrayFloorDataConfig : public RS_BaseConfig {
   public:
    /// Size of the elevated floor in meters in X direction.
    RFloat x_size = 0;
    /// Size of the elevated floor in meters in Y direction.
    RFloat y_size = 0;
    /// Number of values in X direction (columns).
    RSize x_values = 0;
    /// Number of values in Y direction (rows).
    RSize y_values = 0;
    /// Whether the floor repeats in X direction when out of the specified
    /// boundary
    bool is_x_repeat = false;
    /// Whether the floor repeats in Y direction when out of the specified
    /// boundary
    bool is_y_repeat = false;
    /// 2D height matrix of size [y_values, x_values], in meters along Z.
    std::vector<RFloat> height;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("x_size", x_size);
        s("y_size", y_size);
        s("x_values", x_values);
        s("y_values", y_values);
        s("height", height);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_FloorDataConfig : public RS_BaseConfig {
   public:
    /// Type of floor (none, flat, array).
    RSE_FloorDataType type = RSE_FLOOR_DATA_NONE;

    RS_Any config;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("type", type);
        if (type == RSE_FLOOR_DATA_ARRAY) {
            s("config", config.as<RS_ArrayFloorDataConfig>());
        }
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

RS_ENUM(
    RSE_FloorRenderType,
    unsigned int,
    (RSE_FLOOR_RENDER_OPAQUE, 0),
    (RSE_FLOOR_RENDER_OPAQUE_GRID, 1),
    (RSE_FLOOR_RENDER_TRANSPARENT, 2)
)

class RS_OpaqueFloorRenderConfig : public RS_BaseConfig {
   public:
    /// Floor surface color.
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

class RS_OpaqueGridFloorRenderConfig : public RS_BaseConfig {
   public:
    /// Spacing between grid lines in meters along X, unit: Meters.
    RFloat x_spacing = 1;

    /// Spacing between grid lines in meters along Y, unit: Meters.
    RFloat y_spacing = 1;
    
    /// Width of the grid lines in meters, unit: Meters.
    RFloat line_width = 0.05;
    
    /// Color of the cells between grid lines.
    RColor3b fill_color = RColor3b::zero();
    
    /// Color of the grid lines.
    RColor3b line_color = RColor3b::zero();

    /// Surface roughness in [0, 1].
    RFloat roughness = 1.0;

    /// Metalness in [0, 1]; > 0 picks the metallic lobe.
    RFloat metalness = 0.0;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("x_spacing", x_spacing);
        s("y_spacing", y_spacing);
        s("line_width", line_width);
        s("fill_color", fill_color);
        s("line_color", line_color);
        s("roughness", roughness);
        s("metalness", metalness);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_TransparentFloorRenderConfig : public RS_BaseConfig {
   public:
    /// Per-channel attenuation in [0,1].
    RColor3b color = RColor3b::zero();

    /// Index of refraction; 1.0 = straight-through.
    RFloat ior = 1.5;

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

class RS_FloorRenderConfig : public RS_BaseConfig {
   public:
    RSE_FloorRenderType type = RSE_FLOOR_RENDER_OPAQUE;
    RS_Any config = RS_OpaqueFloorRenderConfig{};

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("type", type);
        if (type == RSE_FLOOR_RENDER_OPAQUE) {
            s("config", config.as<RS_OpaqueFloorRenderConfig>());
        } else if (type == RSE_FLOOR_RENDER_OPAQUE_GRID) {
            s("config", config.as<RS_OpaqueGridFloorRenderConfig>());
        } else if (type == RSE_FLOOR_RENDER_TRANSPARENT) {
            s("config", config.as<RS_TransparentFloorRenderConfig>());
        }
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_FloorConfig : public RS_BaseConfig {
   public:
    RS_FloorDataConfig data_config;
    RS_FloorRenderConfig render_config;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("data_config", data_config);
        s("render_config", render_config);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

}  // namespace rise

#endif  // RS_FLOOR_CONFIG_H
