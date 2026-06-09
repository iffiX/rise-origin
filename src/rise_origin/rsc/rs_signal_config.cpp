// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_signal_config.h"

#include "rise_origin/utils/rs_string.h"
#include "rs_rsc_util.h"

using namespace std;

namespace rise {

void RS_ForceSignalConfig::_validate() const {
    if (structure_name.empty()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid RS_ForceSignalConfig config, structure_name must not be empty"
        ));
    }
    if (!voxel_range.isValid()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid RS_ForceSignalConfig config, voxel_range must be valid"
        ));
    }
    RS_CONFIG_EXPECT_NOT_NULL(RS_ForceSignal, body_sid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_ForceSignal, segment_bid, RS_NULL_INDEX);
}

void RS_ForceSignalConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Signal(Signal[S])
    structure_name = config_tree.get<string>("StructureName");
    voxel_range = readBound3ri(config_tree.get<string>("VoxelRange"));
    body_sid = config_tree.get("BodySid", body_sid);
    segment_bid = config_tree.get("SegmentBid", segment_bid);
}

void RS_TorqueSignalConfig::_validate() const {
    if (structure_name.empty()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid RS_TorqueSignalConfig config, structure_name must not be empty"
        ));
    }
    RS_CONFIG_EXPECT_NOT_NULL(RS_TorqueSignal, body_sid, RS_NULL_INDEX);
    RS_CONFIG_EXPECT_NOT_NULL(RS_TorqueSignal, segment_bid, RS_NULL_INDEX);
}

void RS_TorqueSignalConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Signal(Signal[S])
    structure_name = config_tree.get<string>("StructureName");
    body_sid = config_tree.get("BodySid", body_sid);
    segment_bid = config_tree.get("SegmentBid", segment_bid);
}

void RS_SignalConfig::_validate() const {
    if (type == RSE_SIGNAL_FORCE) {
        RS_CONFIG_EXPECT_ANY_SET(RS_SignalConfig, config, "RS_ForceSignalConfig");
        config.as<RS_ForceSignalConfig>().validate();
    } else if (type == RSE_SIGNAL_TORQUE) {
        RS_CONFIG_EXPECT_ANY_SET(RS_SignalConfig, config, "RS_TorqueSignalConfig");
        config.as<RS_TorqueSignalConfig>().validate();
    }
}

void RS_SignalConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Signal(Signal[S])
    name = config_tree.get<string>("Name");
    if (!config_tree.isChildExist("Type")) {
        RS_THROW(RS_InvalidConfigurationException("Invalid signal config, missing key: Type"));
    } else {
        if (config_tree.get<string>("Type") == "Force") {
            type = RSE_SIGNAL_FORCE;
            config.as<RS_ForceSignalConfig>().read(config_tree, false);
        } else if (config_tree.get<string>("Type") == "Torque") {
            type = RSE_SIGNAL_TORQUE;
            config.as<RS_TorqueSignalConfig>().read(config_tree, false);
        } else {
            RS_THROW(RS_InvalidConfigurationException(
                "Invalid signal type: {}, available types are: Force, Torque",
                config_tree.get<string>("Type")
            ));
        }
    }
}

}  // namespace rise
