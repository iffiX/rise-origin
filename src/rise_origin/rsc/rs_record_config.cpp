// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_record_config.h"

#include "rise_origin/utils/rs_string.h"

using namespace std;

namespace rise {

void RS_StructureRecordConfig::_validate() const {
    if (structure_name.empty()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid RS_StructureRecordConfig config, structure_name must not be empty"
        ));
    }
}

void RS_StructureRecordConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Record(Record[R])
    structure_name = config_tree.get<string>("StructureName");
    voxel_record_range = config_tree.get("VoxelRecordRange", voxel_record_range);
    link_record_range = config_tree.get("LinkRecordRange", link_record_range);
    rigid_body_record_range = config_tree.get("RigidBodyRecordRange", rigid_body_record_range);
    joint_record_range = config_tree.get("JointRecordRange", joint_record_range);
    voxel_sample_num = config_tree.get("VoxelSampleNum", voxel_sample_num);
    link_sample_num = config_tree.get("LinkSampleNum", link_sample_num);
}

void RS_RecordConfig::_validate() const {
    if (type == RSE_RECORD_STRUCTURE) {
        RS_CONFIG_EXPECT_ANY_SET(RS_RecordConfig, config, "RS_StructureRecordConfig");
        config.as<RS_StructureRecordConfig>().validate();
    }
}

void RS_RecordConfig::_read(const RS_ConfigPropertyTree &config_tree) {
    // root of tree: RSC.Record(Record[R])
    name = config_tree.get<string>("Name");
    if (!config_tree.isChildExist("Type")) {
        RS_THROW(RS_InvalidConfigurationException("Invalid record config, missing key: Type"));
    } else {
        if (config_tree.get<string>("Type") == "Structure") {
            type = RSE_RECORD_STRUCTURE;
            config.as<RS_StructureRecordConfig>().read(config_tree, false);
        } else {
            RS_THROW(RS_InvalidConfigurationException(
                "Invalid recorder type: {}, available types are: Structure",
                config_tree.get<string>("Type")
            ));
        }
    }
}

}  // namespace rise