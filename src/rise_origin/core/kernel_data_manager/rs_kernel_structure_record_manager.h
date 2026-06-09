// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_RECORD_MANAGER_H
#define RS_KERNEL_RECORD_MANAGER_H

#include <vector>

#include "rise_origin/core/rs_simulation_record.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_buffer.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/rsc/rs_record_config.h"

namespace rise {

class RS_KernelManager;

class RS_KernelStructureRecordManager {
   public:
    RS_KernelStructureRecordManager() = default;

    void init(RS_KernelManager *kernel_manager, const RS_RecordConfig &config);

    void free();

    bool hasRecorderRecords();

    void syncRecorderRecords();

    void waitForRecorderRecordsSync();

    std::vector<RS_SimulationStructureRecord> getRecorderRecords();

    bool hasControllerRecord();

    void syncControllerRecord();

    RS_SimulationStructureRecord getControllerRecord();

    std::string getName() const;

    std::string getStructureName() const;

    RIndex getStructureGid() const;

    [[nodiscard]] const RS_RecordConfig &getConfig() const;

   private:
    RIndex _structure_gid = RS_NULL_INDEX;

    RS_KernelManager *_kernel_manager = nullptr;

    RS_RecordConfig _config;

    bool _is_kernel_data_inited = false;

    RSize _voxel_record_num = 0;

    RSize _link_record_num = 0;

    RSize _rigid_body_record_num = 0;

    RSize _joint_record_num = 0;

    device::RS_DoubleBuffer _recorder_buffer;

    device::RS_DirectBuffer _controller_buffer;

    device::RS_Array<RIndex, 1> _voxel_record_sid;

    device::RS_Array<RHandle, 1> _voxel_record_handles;

    device::RS_Array<RIndex, 1> _link_record_sid;

    device::RS_Array<RHandle, 1> _link_record_handles;

    device::RS_Array<RIndex, 1> _rigid_body_record_sid;

    device::RS_Array<RHandle, 1> _rigid_body_record_handles;

    device::RS_Array<RIndex, 1> _constraint_joint_record_sid;

    device::RS_Array<RHandle, 1> _constraint_joint_record_handles;

    std::vector<RIndex>
    _selectSID(const std::string &structure_name, RSize select_num, RSize range);

    void _initKernelData();

    void _freeKernelData();
};

}  // namespace rise

#endif  // RS_KERNEL_RECORD_MANAGER_H
