// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RECORD_CONFIG_H
#define RS_RECORD_CONFIG_H

#include <unordered_map>

#include "rise_origin/utils/rs_any.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_base_config.h"

namespace rise {

RS_ENUM(
    RSE_StructureVoxelRecordRange,
    unsigned int,
    (RSE_VOXEL_RECORD_RANGE_NONE, 0),
    (RSE_VOXEL_RECORD_RANGE_ALL, 1),
    (RSE_VOXEL_RECORD_RANGE_RANDOM, 2)
)

RS_ENUM(
    RSE_StructureLinkRecordRange,
    unsigned int,
    (RSE_LINK_RECORD_RANGE_NONE, 0),
    (RSE_LINK_RECORD_RANGE_ALL, 1),
    (RSE_LINK_RECORD_RANGE_RANDOM, 2)
)

RS_ENUM(
    RSE_StructureRigidBodyRecordRange,
    unsigned int,
    (RSE_RIGID_BODY_RECORD_RANGE_NONE, 0),
    (RSE_RIGID_BODY_RECORD_RANGE_ALL, 1)
)

RS_ENUM(
    RSE_StructureJointRecordRange,
    unsigned int,
    (RSE_JOINT_RECORD_RANGE_NONE, 0),
    (RSE_JOINT_RECORD_RANGE_ALL, 1)
)

class RS_StructureRecordConfig : public RS_BaseConfig {
   public:
    std::string structure_name;

    int voxel_record_range = RSE_VOXEL_RECORD_RANGE_NONE;

    int link_record_range = RSE_LINK_RECORD_RANGE_NONE;

    int rigid_body_record_range = RSE_RIGID_BODY_RECORD_RANGE_NONE;

    int joint_record_range = RSE_JOINT_RECORD_RANGE_NONE;

    RSize voxel_sample_num = 0;

    RSize link_sample_num = 0;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("structure_name", structure_name);
        s("voxel_record_range", voxel_record_range);
        s("link_record_range", link_record_range);
        s("joint_record_range", joint_record_range);
        s("rigid_body_record_range", rigid_body_record_range);
        s("voxel_sample_num", voxel_sample_num);
        s("link_sample_num", link_sample_num);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

RS_ENUM(RSE_RecordType, unsigned int, (RSE_RECORD_STRUCTURE, 0))

class RS_RecordConfig : public RS_BaseConfig {
   public:
    /// Name of the record.
    std::string name;

    RSE_RecordType type = RSE_RECORD_STRUCTURE;

    RS_Any config;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("name", name);
        s("type", type);
        if (type == RSE_RECORD_STRUCTURE) {
            s("config", config.as<RS_StructureRecordConfig>());
        }
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

}  // namespace rise
#endif  // RS_RECORD_CONFIG_H
