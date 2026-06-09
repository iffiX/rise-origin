// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_floor_config.h"

#include <highfive/highfive.hpp>

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/utils/rs_string.h"

using namespace std;
using namespace HighFive;

namespace rise {

void RS_ArrayFloorDataConfig::_validate() const {
    RS_CONFIG_EXPECT_GT(RS_ArrayFloorDataConfig, x_size, 0);
    RS_CONFIG_EXPECT_GT(RS_ArrayFloorDataConfig, y_size, 0);
    RS_CONFIG_EXPECT_GT(RS_ArrayFloorDataConfig, x_values, 0);
    RS_CONFIG_EXPECT_GT(RS_ArrayFloorDataConfig, y_values, 0);
    if (height.size() != x_values * y_values) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid floor data config, the size of height array "
            "must be equal to x_values * y_values"
        ));
    }
}

void RS_ArrayFloorDataConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Floor.Data
    if (!config_tree.isChildrenSameAsRequired(
            {"Type", "XSize", "YSize", "XValues", "YValues", "Height"}
        ) &&
        !config_tree.isChildrenSameAsRequired(
            {"Type", "XSize", "YSize", "XValues", "YValues", "Hdf5File"}
        )) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid simple floor config, required keys are: Type, XSize, YSize, XValues, "
            "YValues, "
            "Height (or Hdf5File), but found: {}",
            join(",", config_tree.getChildrenKeys())
        ));
    } else {
        x_size = config_tree.get("XSize", x_size);
        y_size = config_tree.get("YSize", y_size);
        x_values = config_tree.get("XValues", x_values);
        y_values = config_tree.get("YValues", y_values);
        RSize num = x_values * y_values;
        height.clear();
        height.reserve(num);
        if (!config_tree.isChildExist("Hdf5File")) {
            auto split_values = split(strip(config_tree.get<string>("Height")), ",");
            if (split_values.size() != num)
                RS_THROW(RS_InvalidConfigurationException(
                    "Floor elevation height array size is {}, required {} (XSize = {}, YSize = "
                    "{})",
                    split_values.size(),
                    num,
                    x_size,
                    y_size
                ));
            for (const auto &h_str : split_values) {
                height.push_back(static_cast<RFloat>(stod(h_str)));
            }
        } else {
            DataSetAccessProps access_props;
            access_props.add(HighFive::Caching(8192, 131072));
            File hdf5_file(config_tree.get<string>("Hdf5File"));
            DataSet dataset = hdf5_file.getDataSet(config_tree.get<string>("Height"), access_props);
            dataset.read(height);
            if (height.size() != num)
                RS_THROW(RS_InvalidConfigurationException(
                    "Floor elevation height HDF5 size is {}, required {} (XSize = {}, YSize = "
                    "{})",
                    height.size(),
                    num,
                    x_size,
                    y_size
                ));
        }
    }
}

void RS_FloorDataConfig::_validate() const {
    if (type == RSE_FLOOR_DATA_ARRAY) {
        RS_CONFIG_EXPECT_ANY_SET(RS_FloorDataConfig, config, "RS_ArrayFloorDataConfig");
        config.as<RS_ArrayFloorDataConfig>().validate();
    }
}

void RS_FloorDataConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Floor.Data
    if (!config_tree.isChildExist("Type"))
        RS_THROW(RS_InvalidConfigurationException("Invalid floor config, missing key: Type"));

    if (config_tree.get<string>("Type") == "None") {
        type = RSE_FLOOR_DATA_NONE;
    } else if (config_tree.get<string>("Type") == "Flat") {
        type = RSE_FLOOR_DATA_FLAT;
    } else if (config_tree.get<string>("Type") == "Array") {
        type = RSE_FLOOR_DATA_ARRAY;
        config.as<RS_ArrayFloorDataConfig>().read(config_tree, false);
    } else {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid floor data type: {}, available types are: None, Flat, Array",
            config_tree.get<string>("Type")
        ));
    }
}

void RS_OpaqueFloorRenderConfig::_validate() const {
    RS_CONFIG_EXPECT_GE(RS_OpaqueFloorRenderConfig, roughness, 0);
    RS_CONFIG_EXPECT_LE(RS_OpaqueFloorRenderConfig, roughness, 1);
    RS_CONFIG_EXPECT_GE(RS_OpaqueFloorRenderConfig, metalness, 0);
    RS_CONFIG_EXPECT_LE(RS_OpaqueFloorRenderConfig, metalness, 1);
}

void RS_OpaqueFloorRenderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Floor.Render.Opaque
    color = readColor3c(config_tree.get<string>("Color"));
    roughness = config_tree.get("Roughness", roughness);
    metalness = config_tree.get("Metalness", metalness);
}

void RS_OpaqueGridFloorRenderConfig::_validate() const {
    RS_CONFIG_EXPECT_GT(RS_OpaqueGridFloorRenderConfig, x_spacing, 0);
    RS_CONFIG_EXPECT_GT(RS_OpaqueGridFloorRenderConfig, y_spacing, 0);
    RS_CONFIG_EXPECT_GT(RS_OpaqueGridFloorRenderConfig, line_width, 0);
    RS_CONFIG_EXPECT_GE(RS_OpaqueGridFloorRenderConfig, roughness, 0);
    RS_CONFIG_EXPECT_LE(RS_OpaqueGridFloorRenderConfig, roughness, 1);
    RS_CONFIG_EXPECT_GE(RS_OpaqueGridFloorRenderConfig, metalness, 0);
    RS_CONFIG_EXPECT_LE(RS_OpaqueGridFloorRenderConfig, metalness, 1);
}

void RS_OpaqueGridFloorRenderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Floor.Render.OpaqueGrid
    x_spacing = config_tree.get("XSpacing", x_spacing);
    y_spacing = config_tree.get("YSpacing", y_spacing);
    line_width = config_tree.get("LineWidth", line_width);
    fill_color = readColor3c(config_tree.get<string>("FillColor"));
    line_color = readColor3c(config_tree.get<string>("LineColor"));
    roughness = config_tree.get("Roughness", roughness);
    metalness = config_tree.get("Metalness", metalness);
}

void RS_TransparentFloorRenderConfig::_validate() const {
    RS_CONFIG_EXPECT_GT(RS_TransparentFloorRenderConfig, ior, 0);
    RS_CONFIG_EXPECT_GE(RS_TransparentFloorRenderConfig, roughness, 0);
    RS_CONFIG_EXPECT_LE(RS_TransparentFloorRenderConfig, roughness, 1);
}

void RS_TransparentFloorRenderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Floor.Render.Transparent
    color = readColor3c(config_tree.get<string>("Color"));
    ior = config_tree.get("Ior", ior);
    roughness = config_tree.get("Roughness", roughness);
}

void RS_FloorRenderConfig::_validate() const {
    if (type == RSE_FLOOR_RENDER_OPAQUE) {
        RS_CONFIG_EXPECT_ANY_SET(RS_FloorRenderConfig, config, "RS_OpaqueFloorRenderConfig");
        config.as<RS_OpaqueFloorRenderConfig>().validate();
    } else if (type == RSE_FLOOR_RENDER_OPAQUE_GRID) {
        RS_CONFIG_EXPECT_ANY_SET(RS_FloorRenderConfig, config, "RS_OpaqueGridFloorRenderConfig");
        config.as<RS_OpaqueGridFloorRenderConfig>().validate();
    } else if (type == RSE_FLOOR_RENDER_TRANSPARENT) {
        RS_CONFIG_EXPECT_ANY_SET(RS_FloorRenderConfig, config, "RS_TransparentFloorRenderConfig");
        config.as<RS_TransparentFloorRenderConfig>().validate();
    }
}

void RS_FloorRenderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Floor.Render
    if (!config_tree.isChildExist("Type"))
        RS_THROW(
            RS_InvalidConfigurationException("Invalid floor render config, missing key: Type")
        );

    auto type_str = config_tree.get<string>("Type");
    if (type_str == "Opaque") {
        type = RSE_FLOOR_RENDER_OPAQUE;
        config.as<RS_OpaqueFloorRenderConfig>().read(config_tree, false);
    } else if (type_str == "OpaqueGrid") {
        type = RSE_FLOOR_RENDER_OPAQUE_GRID;
        config.as<RS_OpaqueGridFloorRenderConfig>().read(config_tree, false);
    } else if (type_str == "Transparent") {
        type = RSE_FLOOR_RENDER_TRANSPARENT;
        config.as<RS_TransparentFloorRenderConfig>().read(config_tree, false);
    } else {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid floor render type: {}, available types are: Opaque, OpaqueGrid, Transparent",
            type_str
        ));
    }
}

void RS_FloorConfig::_validate() const {
    data_config.validate();
    render_config.validate();
}

void RS_FloorConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Floor
    data_config.read(config_tree.getChild("Data"), false);
    render_config.read(config_tree.getChild("Render"), false);
}

}  // namespace rise