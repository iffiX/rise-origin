// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <random>

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/device/algorithm/rs_generate.h"
#include "rise_origin/logic/record/rs_basic_record_c.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rs_kernel_structure_manager.h"
#include "rs_kernel_structure_record_manager.h"

using namespace std;
using namespace rise::logic;
using namespace rise::device;

namespace rise {

void RS_KernelStructureRecordManager::init(
    RS_KernelManager* kernel_manager,
    const RS_RecordConfig& config
) {
    if (_structure_gid != RS_NULL_INDEX)
        RS_THROW(RS_HostExecutionException("Structure record manager cannot be re-initialized"));
    _kernel_manager = kernel_manager;
    _config = config;

    auto& sr_config = _config.config.as<RS_StructureRecordConfig>();

    // Check whether the target structure exists
    auto it = _kernel_manager->getStructureNameToGid().find(sr_config.structure_name);
    if (it == _kernel_manager->getStructureNameToGid().end())
        RS_THROW(RS_InvalidConfigurationException(
            "Structure {} is not registered in the kernel manager", sr_config.structure_name
        ));

    auto memory_provider = _kernel_manager->getDeviceProvider();
    auto buffer_frame_num = _kernel_manager->getKernelRecordBufferFrameNum();
    auto& structure_manager = _kernel_manager->getKernelStructureManagers().at(it->second);
    _structure_gid = it->second;
    _voxel_record_num = 0;
    _link_record_num = 0;
    _rigid_body_record_num = 0;
    _joint_record_num = 0;

    RSize voxel_num = structure_manager.getVoxelNum();
    RSize link_num = structure_manager.getLinkNum();
    RSize rigid_body_num = structure_manager.getRigidBodyNum();
    RSize joint_num = structure_manager.getJointNum();
    RSize constraint_joint_num = structure_manager.getConstraintJointNum();

    // Voxel
    if (sr_config.voxel_record_range == RSE_VOXEL_RECORD_RANGE_ALL) {
        castProviderPtrAndCall(memory_provider, [&](auto p_ptr) {
            _voxel_record_sid = RS_Array<RIndex, 1>(voxel_num, p_ptr);
            _voxel_record_handles = RS_Array<RHandle, 1>(voxel_num, p_ptr);
            generateIndex(_voxel_record_sid);
        });
    } else if (sr_config.voxel_record_range == RSE_VOXEL_RECORD_RANGE_RANDOM) {
        /// Create a random voxel sample sid array
        castProviderPtrAndCall(memory_provider, [&](auto p_ptr) {
            RSize voxel_sample_num = RS_MIN(voxel_num, sr_config.voxel_sample_num);
            auto random_voxel_sid =
                _selectSID(sr_config.structure_name, voxel_sample_num, voxel_num);
            _voxel_record_sid = RS_Array<RIndex, 1>(random_voxel_sid.size(), p_ptr);
            _voxel_record_handles = RS_Array<RHandle, 1>(random_voxel_sid.size(), p_ptr);
            _voxel_record_sid.write(random_voxel_sid.data());
        });
    }
    _voxel_record_num = _voxel_record_sid.size();

    // Link
    if (sr_config.link_record_range == RSE_LINK_RECORD_RANGE_ALL) {
        castProviderPtrAndCall(memory_provider, [&](auto p_ptr) {
            _link_record_sid = RS_Array<RIndex, 1>(link_num, p_ptr);
            _link_record_handles = RS_Array<RHandle, 1>(link_num, p_ptr);
            generateIndex(_link_record_sid);
        });
    } else if (sr_config.link_record_range == RSE_LINK_RECORD_RANGE_RANDOM) {
        /// Create a random link sample sid array
        castProviderPtrAndCall(memory_provider, [&](auto p_ptr) {
            RSize link_sample_num = RS_MIN(link_num, sr_config.link_sample_num);
            auto random_link_sid = _selectSID(sr_config.structure_name, link_sample_num, link_num);
            _link_record_sid = RS_Array<RIndex, 1>(random_link_sid.size(), p_ptr);
            _link_record_handles = RS_Array<RHandle, 1>(random_link_sid.size(), p_ptr);
            _link_record_sid.write(random_link_sid.data());
        });
    }
    _link_record_num = _link_record_sid.size();

    // Rigid body
    if (sr_config.rigid_body_record_range == RSE_RIGID_BODY_RECORD_RANGE_ALL) {
        _rigid_body_record_num = rigid_body_num;
        castProviderPtrAndCall(memory_provider, [&](auto p_ptr) {
            _rigid_body_record_sid = RS_Array<RIndex, 1>(rigid_body_num, p_ptr);
            _rigid_body_record_handles = RS_Array<RHandle, 1>(rigid_body_num, p_ptr);
            generateIndex(_rigid_body_record_sid);
        });
    }

    // Joint (Note: we reserve record space per joint, but handles are per constraint joint)
    if (sr_config.joint_record_range == RSE_JOINT_RECORD_RANGE_ALL) {
        _joint_record_num = joint_num;
        castProviderPtrAndCall(memory_provider, [&](auto p_ptr) {
            _constraint_joint_record_sid = RS_Array<RIndex, 1>(constraint_joint_num, p_ptr);
            _constraint_joint_record_handles = RS_Array<RHandle, 1>(constraint_joint_num, p_ptr);
            generateIndex(_constraint_joint_record_sid);
        });
    }

    // Create record buffers
    RSize frame_size = RS_SimulationStructureRecord::getRequiredMemorySize(
        _voxel_record_num, _link_record_num, _rigid_body_record_num, _joint_record_num
    );
    _recorder_buffer.init(buffer_frame_num, buffer_frame_num * frame_size, memory_provider);
    _controller_buffer.init(frame_size, memory_provider);

    _initKernelData();
}

void RS_KernelStructureRecordManager::free() {
    _freeKernelData();
    _voxel_record_sid.free();
    _voxel_record_handles.free();
    _link_record_sid.free();
    _link_record_handles.free();
    _rigid_body_record_sid.free();
    _rigid_body_record_handles.free();
    _constraint_joint_record_sid.free();
    _constraint_joint_record_handles.free();
    _recorder_buffer.free();
    _controller_buffer.free();
}

bool RS_KernelStructureRecordManager::hasRecorderRecords() {
    return _voxel_record_num + _link_record_num + _rigid_body_record_num + _joint_record_num > 0;
}

void RS_KernelStructureRecordManager::syncRecorderRecords() {
    _recorder_buffer.swapAndCopy(true, false, false);
}

void RS_KernelStructureRecordManager::waitForRecorderRecordsSync() {
    _recorder_buffer.waitForCopyFinish();
}

vector<RS_SimulationStructureRecord> RS_KernelStructureRecordManager::getRecorderRecords() {
    vector<RS_SimulationStructureRecord> records;
    for (RIndex i = 0, total = _recorder_buffer.getHostNum(); i < total; ++i) {
        RS_SimulationStructureRecord record;
        record.initFromMemory(
            _recorder_buffer.getHostChunk(i).ptr(),
            _voxel_record_num,
            _link_record_num,
            _rigid_body_record_num,
            _joint_record_num
        );
        record.structure_gid = _structure_gid;
        record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
        records.emplace_back(record);
    }
    return records;
}

bool RS_KernelStructureRecordManager::hasControllerRecord() {
    return _voxel_record_num + _link_record_num + _rigid_body_record_num + _joint_record_num > 0;
}

void RS_KernelStructureRecordManager::syncControllerRecord() { _controller_buffer.sync(true); }

RS_SimulationStructureRecord RS_KernelStructureRecordManager::getControllerRecord() {
    RS_SimulationStructureRecord record;
    record.initFromMemory(
        _controller_buffer.getHostChunk(0).ptr(),
        _voxel_record_num,
        _link_record_num,
        _rigid_body_record_num,
        _joint_record_num
    );
    record.structure_gid = _structure_gid;
    record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
    return record;
}

std::string RS_KernelStructureRecordManager::getName() const { return _config.name; }

std::string RS_KernelStructureRecordManager::getStructureName() const {
    return _config.config.as<RS_StructureRecordConfig>().structure_name;
}

RIndex RS_KernelStructureRecordManager::getStructureGid() const { return _structure_gid; }

const RS_RecordConfig& RS_KernelStructureRecordManager::getConfig() const { return _config; }

vector<RIndex> RS_KernelStructureRecordManager::_selectSID(
    const string& structure_name,
    RSize select_num,
    RSize range
) {
    vector<RIndex> all_sids(range);
    for (RIndex i = 0; i < range; ++i) {
        all_sids[i] = i;
    }

    // Create a seed from the structure name for reproducibility
    seed_seq seed(structure_name.begin(), structure_name.end());

    // Initialize the random number generator with the seed
    mt19937 generator(seed);
    shuffle(all_sids.begin(), all_sids.end(), generator);
    vector<RIndex> record_sid(all_sids.begin(), all_sids.begin() + select_num);

    return record_sid;
}

void RS_KernelStructureRecordManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        lock_guard record_lock(k_ptr->getSyncMutex());
        auto& ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Context data is modified, needs to update kernel
        k_ptr->markModified();

        // Update host layout
        ctx.host_data->structure_record_layout.add(1, _structure_gid);
        ctx.host_data->structure_voxel_record_layout.add(_voxel_record_num, _structure_gid);
        ctx.host_data->structure_link_record_layout.add(_link_record_num, _structure_gid);
        ctx.host_data->structure_rigid_body_record_layout.add(
            _rigid_body_record_num, _structure_gid
        );
        ctx.host_data->structure_joint_record_layout.add(_joint_record_num, _structure_gid);
        ctx.host_data->structure_constraint_joint_record_layout.add(
            _constraint_joint_record_sid.size(), _structure_gid
        );

        // Update device layout
        ctx.host_data->structure_voxel_record_layout.syncDeviceLayout(
            ctx.structure_voxel_record_layout
        );
        ctx.host_data->structure_link_record_layout.syncDeviceLayout(
            ctx.structure_link_record_layout
        );
        ctx.host_data->structure_rigid_body_record_layout.syncDeviceLayout(
            ctx.structure_rigid_body_record_layout
        );
        ctx.host_data->structure_joint_record_layout.syncDeviceLayout(
            ctx.structure_joint_record_layout
        );
        ctx.host_data->structure_constraint_joint_record_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_record_layout
        );

        // Resolve handles needed by the record
        castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
            p_ptr->template run<RSCF_ResolveRecordDependency>(
                {_voxel_record_sid.size()},
                ctx.host_data->structure_voxel_layout.segmentOffset(
                    ctx.host_data->structure_voxel_layout.find(_structure_gid)
                ),
                _voxel_record_sid,
                _voxel_record_handles
            );

            p_ptr->template run<RSCF_ResolveRecordDependency>(
                {_link_record_sid.size()},
                ctx.host_data->structure_link_layout.segmentOffset(
                    ctx.host_data->structure_link_layout.find(_structure_gid)
                ),
                _link_record_sid,
                _link_record_handles
            );

            p_ptr->template run<RSCF_ResolveRecordDependency>(
                {_rigid_body_record_sid.size()},
                ctx.host_data->structure_rigid_body_layout.segmentOffset(
                    ctx.host_data->structure_rigid_body_layout.find(_structure_gid)
                ),
                _rigid_body_record_sid,
                _rigid_body_record_handles
            );

            p_ptr->template run<RSCF_ResolveRecordDependency>(
                {_constraint_joint_record_sid.size()},
                ctx.host_data->structure_constraint_joint_layout.segmentOffset(
                    ctx.host_data->structure_constraint_joint_layout.find(_structure_gid)
                ),
                _constraint_joint_record_sid,
                _constraint_joint_record_handles
            );
        });

        // Add record, record buffer and handles to device data
        ctx.structure_records.append({});
        ctx.structure_recorder_buffers.append(_recorder_buffer);
        ctx.structure_controller_buffers.append(_controller_buffer);

        ctx.structure_voxel_record_handles.append(_voxel_record_handles);
        ctx.structure_link_record_handles.append(_link_record_handles);
        ctx.structure_rigid_body_record_handles.append(_rigid_body_record_handles);
        ctx.structure_constraint_joint_record_handles.append(_constraint_joint_record_handles);

        _is_kernel_data_inited = true;
    });
}

void RS_KernelStructureRecordManager::_freeKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        if (!_is_kernel_data_inited) return;
        lock_guard lock(k_ptr->getSyncMutex());
        auto& ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Context data is modified, needs to update kernel
        k_ptr->markModified();

        RIndex record_index = ctx.host_data->structure_record_layout.find(_structure_gid);

        // Update host layout
        ctx.host_data->structure_record_layout.remove(record_index);
        ctx.host_data->structure_voxel_record_layout.remove(record_index);
        ctx.host_data->structure_link_record_layout.remove(record_index);
        ctx.host_data->structure_rigid_body_record_layout.remove(record_index);
        ctx.host_data->structure_joint_record_layout.remove(record_index);
        ctx.host_data->structure_constraint_joint_record_layout.remove(record_index);

        // Update device layout
        ctx.host_data->structure_voxel_record_layout.syncDeviceLayout(
            ctx.structure_voxel_record_layout
        );
        ctx.host_data->structure_link_record_layout.syncDeviceLayout(
            ctx.structure_link_record_layout
        );
        ctx.host_data->structure_rigid_body_record_layout.syncDeviceLayout(
            ctx.structure_rigid_body_record_layout
        );
        ctx.host_data->structure_joint_record_layout.syncDeviceLayout(
            ctx.structure_joint_record_layout
        );
        ctx.host_data->structure_constraint_joint_record_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_record_layout
        );

        // Remove record, record buffer and handles from device data
        ctx.structure_records.erase(record_index);
        ctx.structure_recorder_buffers.erase(record_index);
        ctx.structure_controller_buffers.erase(record_index);
        ctx.structure_voxel_record_handles.erase(record_index);
        ctx.structure_link_record_handles.erase(record_index);
        ctx.structure_rigid_body_record_handles.erase(record_index);
        ctx.structure_constraint_joint_record_handles.erase(record_index);
    });
}

}  // namespace rise