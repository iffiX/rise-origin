// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_recorder_config.h"

#include "rise_origin/utils/rs_string.h"

using namespace std;

namespace rise {

void RS_HDF5RecorderConfig::_validate() const {
    if (path.empty()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid RS_HDF5RecorderConfig config, path must not be empty"
        ));
    }
}

void RS_HDF5RecorderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Recorders(Recorder[R])
    path = config_tree.get<string>("Path");
}

void RS_RecorderConfig::_validate() const {
    if (type == RSE_RECORDER_HDF5) {
        RS_CONFIG_EXPECT_ANY_SET(RS_RecorderConfig, config, "RS_HDF5RecorderConfig");
        config.as<RS_HDF5RecorderConfig>().validate();
    }
}

void RS_RecorderConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Recorders(Recorder[R])

    if (!config_tree.isChildExist("Type")) {
        RS_THROW(RS_InvalidConfigurationException("Invalid recorder config, missing key: Type"));
    } else {
        if (config_tree.get<string>("Type") == "HDF5") {
            type = RSE_RECORDER_HDF5;
            config.as<RS_HDF5RecorderConfig>().read(config_tree, false);
        } else {
            RS_THROW(RS_InvalidConfigurationException(
                "Invalid recorder type: {}, available types are: HDF5",
                config_tree.get<string>("Type")
            ));
        }
    }
}

}  // namespace rise