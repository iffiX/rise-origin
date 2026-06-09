// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rs_kernel_force_manager.h"

using namespace std;

namespace rise {

RS_KernelForceManager::RS_KernelForceManager() {}

void RS_KernelForceManager::init(RS_KernelManager *kernel_manager, const RS_SignalConfig &config) {
    if (_structure_gid != RS_NULL_INDEX)
        RS_THROW(RS_HostExecutionException("Force manager cannot be re-initialized"));

    _kernel_manager = kernel_manager;
    _config = config;

    auto &force_signal = _config.config.as<RS_ForceSignalConfig>();

    // Check whether the target structure exists
    auto it = _kernel_manager->getStructureNameToGid().find(force_signal.structure_name);
    if (it == _kernel_manager->getStructureNameToGid().end())
        RS_THROW(RS_InvalidConfigurationException(
            "Structure {} is not registered in the kernel manager", force_signal.structure_name
        ));

    _structure_gid = it->second;

    // Get the structure manager
    auto &structure_manager = _kernel_manager->getKernelStructureManagers().at(_structure_gid);

    // Check segment type (throws if segment not found)
    _is_rigid = structure_manager.getSegmentType(force_signal.body_sid, force_signal.segment_bid);

    if (_is_rigid) {
        // For rigid body, find the rigid body SID
        _rigid_body_sid =
            structure_manager.findRigidBodySID(force_signal.body_sid, force_signal.segment_bid);
        if (_rigid_body_sid == RS_NULL_INDEX) {
            RS_THROW(RS_InvalidConfigurationException(
                "Rigid body with body_sid {} and segment_bid {} not found in structure {}",
                force_signal.body_sid,
                force_signal.segment_bid,
                force_signal.structure_name
            ));
        }
    } else {
        // For soft voxels, count voxels in range
        _soft_voxel_count = structure_manager.getSegmentVoxelNumInRange(
            force_signal.body_sid, force_signal.segment_bid, force_signal.voxel_range
        );
        if (_soft_voxel_count == 0) {
            RS_THROW(RS_InvalidConfigurationException(
                "No soft voxels found in range for body_sid {} segment_bid {} in structure {}",
                force_signal.body_sid,
                force_signal.segment_bid,
                force_signal.structure_name
            ));
        }
    }

    // Allocate signal buffer for force (3 * RFloat for x, y, z components)
    // Note: user interface buffers, use RFloat accuracy
    RS_MemoryProvider *memory_provider = _kernel_manager->getDeviceProvider();
    _host_force_signal_buffer = memory_provider->allocateHost(sizeof(RFloat) * 3, true);
    _host_force_signal_buffer.set(0);

    _initKernelData();
}

void RS_KernelForceManager::free() {
    _freeKernelData();
    _host_force_signal_buffer.free();
}

bool RS_KernelForceManager::hasControllerSignal() { return true; }

void RS_KernelForceManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
            // no need to lock kernel as we don't modify kernel data, just pointer contents
            auto &ctx = k_ptr->ctx;
            RS_MemoryPtr device_force_signals =
                ctx.force_signals.data + sizeof(RFloat) * _device_force_signal_offset;

            device_force_signals.copyFrom(
                _host_force_signal_buffer, _host_force_signal_buffer.size()
            );
        });
    }
}

RS_SimulationForceSignal RS_KernelForceManager::getControllerSignal() {
    if (hasControllerSignal()) {
        return RS_SimulationForceSignal(
            _host_force_signal_buffer.device(),
            _structure_gid,
            _host_force_signal_buffer.as<RFloat>()
        );
    }
    return {};
}

const RS_SignalConfig &RS_KernelForceManager::getConfig() const { return _config; }

std::string RS_KernelForceManager::getName() const { return _config.name; }

void RS_KernelForceManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        lock_guard lock(k_ptr->getSyncMutex());
        auto &ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Context data is modified, needs to update kernel
        k_ptr->markModified();

        // Update host layout
        ctx.host_data->force_signal_layout.add(sizeof(RFloat) * 3, _structure_gid);
        _device_force_signal_offset = ctx.host_data->force_signal_layout.segmentOffset(
            ctx.host_data->force_signal_layout.find(_structure_gid)
        );

        // Update device layout
        ctx.host_data->force_signal_layout.syncDeviceLayout(ctx.force_signal_layout);

        // Resize device signal buffer
        ctx.force_signals.resize(ctx.host_data->force_signal_layout.capacity());

        // Clear device signal buffer by copy zero cleared host force signal buffer
        RS_MemoryPtr device_force_signals =
            ctx.force_signals.data + sizeof(RFloat) * _device_force_signal_offset;
        device_force_signals.copyFrom(_host_force_signal_buffer, _host_force_signal_buffer.size());

        // Create force memory object and add to appropriate context vector
        auto &force_signal = _config.config.as<RS_ForceSignalConfig>();
        RHandle force_signal_handle =
            _device_force_signal_offset / 3;  // Handle is index into 3-component array

        if (_is_rigid) {
            // Create rigid force memory
            logic::RS_RigidForceMem rigid_force_mem(
                force_signal_handle,
                1.0,  // unit_portion = 1.0 for rigid bodies
                _structure_gid,
                _rigid_body_sid
            );

            // Add to host layout to track position in vector
            ctx.host_data->force_rigid_layout.add(1, _structure_gid);
            ctx.force_rigid.append(rigid_force_mem);
        } else {
            // Create soft force memory
            RFloat unit_portion = 1.0 / static_cast<RFloat>(_soft_voxel_count);
            logic::RS_SoftForceMem soft_force_mem(
                force_signal_handle,
                unit_portion,
                _structure_gid,
                force_signal.voxel_range,
                force_signal.body_sid,
                force_signal.segment_bid
            );

            // Add to host layout to track position in vector
            ctx.host_data->force_soft_layout.add(1, _structure_gid);
            ctx.force_soft.append(soft_force_mem);
        }

        _is_kernel_data_inited = true;
    });
}

void RS_KernelForceManager::_freeKernelData() {
    if (_is_kernel_data_inited) {
        castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
            lock_guard lock(k_ptr->getSyncMutex());
            auto &ctx = k_ptr->ctx;

            // Sync to make sure all operations in memory provider stream are finished
            _kernel_manager->getDeviceProvider()->sync();

            // Context data is modified, needs to update kernel
            k_ptr->markModified();

            // Remove force memory from appropriate vector and layout
            if (_is_rigid) {
                RIndex force_mem_index = ctx.host_data->force_rigid_layout.find(_structure_gid);
                // Remove from rigid force memory vector
                ctx.force_rigid.erase(force_mem_index);

                // Update host layout
                ctx.host_data->force_rigid_layout.remove(force_mem_index);
            } else {
                RIndex force_mem_index = ctx.host_data->force_soft_layout.find(_structure_gid);
                // Remove from soft force memory vector
                ctx.force_soft.erase(force_mem_index);

                // Update host layout
                ctx.host_data->force_soft_layout.remove(force_mem_index);
            }

            // Update force signal layout
            ctx.host_data->force_signal_layout.remove(
                ctx.host_data->force_signal_layout.find(_structure_gid)
            );

            // Update device layout
            ctx.host_data->force_signal_layout.syncDeviceLayout(ctx.force_signal_layout);

            // Resize device signal buffer
            ctx.force_signals.resize(ctx.host_data->force_signal_layout.capacity());
        });

        _is_kernel_data_inited = false;
    }
}

}  // namespace rise
