// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rs_kernel_torque_manager.h"

using namespace std;

namespace rise {

RS_KernelTorqueManager::RS_KernelTorqueManager() {}

void RS_KernelTorqueManager::init(RS_KernelManager *kernel_manager, const RS_SignalConfig &config) {
    if (_structure_gid != RS_NULL_INDEX)
        RS_THROW(RS_HostExecutionException("Torque manager cannot be re-initialized"));

    _kernel_manager = kernel_manager;
    _config = config;

    auto &torque_signal = _config.config.as<RS_TorqueSignalConfig>();

    // Check whether the target structure exists
    auto it = _kernel_manager->getStructureNameToGid().find(torque_signal.structure_name);
    if (it == _kernel_manager->getStructureNameToGid().end())
        RS_THROW(RS_InvalidConfigurationException(
            "Structure {} is not registered in the kernel manager", torque_signal.structure_name
        ));

    _structure_gid = it->second;

    // Get the structure manager
    auto &structure_manager = _kernel_manager->getKernelStructureManagers().at(_structure_gid);

    // Torque is only defined for rigid bodies (throws if segment not found)
    bool is_rigid =
        structure_manager.getSegmentType(torque_signal.body_sid, torque_signal.segment_bid);
    if (!is_rigid) {
        RS_THROW(RS_InvalidConfigurationException(
            "Torque signal requires a rigid body segment, but body_sid {} segment_bid {} in "
            "structure {} is soft",
            torque_signal.body_sid,
            torque_signal.segment_bid,
            torque_signal.structure_name
        ));
    }

    // Find the rigid body SID
    _rigid_body_sid =
        structure_manager.findRigidBodySID(torque_signal.body_sid, torque_signal.segment_bid);
    if (_rigid_body_sid == RS_NULL_INDEX) {
        RS_THROW(RS_InvalidConfigurationException(
            "Rigid body with body_sid {} and segment_bid {} not found in structure {}",
            torque_signal.body_sid,
            torque_signal.segment_bid,
            torque_signal.structure_name
        ));
    }

    // Allocate signal buffer for torque (3 * RFloat for x, y, z components)
    // Note: user interface buffers, use RFloat accuracy
    RS_MemoryProvider *memory_provider = _kernel_manager->getDeviceProvider();
    _host_torque_signal_buffer = memory_provider->allocateHost(sizeof(RFloat) * 3, true);
    _host_torque_signal_buffer.set(0);

    _initKernelData();
}

void RS_KernelTorqueManager::free() {
    _freeKernelData();
    _host_torque_signal_buffer.free();
}

bool RS_KernelTorqueManager::hasControllerSignal() { return true; }

void RS_KernelTorqueManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
            // no need to lock kernel as we don't modify kernel data, just pointer contents
            auto &ctx = k_ptr->ctx;
            RS_MemoryPtr device_torque_signals =
                ctx.torque_signals.data + sizeof(RFloat) * _device_torque_signal_offset;

            device_torque_signals.copyFrom(
                _host_torque_signal_buffer, _host_torque_signal_buffer.size()
            );
        });
    }
}

RS_SimulationTorqueSignal RS_KernelTorqueManager::getControllerSignal() {
    if (hasControllerSignal()) {
        return RS_SimulationTorqueSignal(
            _host_torque_signal_buffer.device(),
            _structure_gid,
            _host_torque_signal_buffer.as<RFloat>()
        );
    }
    return {};
}

const RS_SignalConfig &RS_KernelTorqueManager::getConfig() const { return _config; }

std::string RS_KernelTorqueManager::getName() const { return _config.name; }

void RS_KernelTorqueManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        lock_guard lock(k_ptr->getSyncMutex());
        auto &ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Context data is modified, needs to update kernel
        k_ptr->markModified();

        // Update host layout
        ctx.host_data->torque_signal_layout.add(sizeof(RFloat) * 3, _structure_gid);
        _device_torque_signal_offset = ctx.host_data->torque_signal_layout.segmentOffset(
            ctx.host_data->torque_signal_layout.find(_structure_gid)
        );

        // Update device layout
        ctx.host_data->torque_signal_layout.syncDeviceLayout(ctx.torque_signal_layout);

        // Resize device signal buffer
        ctx.torque_signals.resize(ctx.host_data->torque_signal_layout.capacity());

        // Clear device signal buffer by copy zero cleared host torque signal buffer
        RS_MemoryPtr device_torque_signals =
            ctx.torque_signals.data + sizeof(RFloat) * _device_torque_signal_offset;
        device_torque_signals.copyFrom(
            _host_torque_signal_buffer, _host_torque_signal_buffer.size()
        );

        // Create torque memory object and add to context vector
        RHandle torque_signal_handle =
            _device_torque_signal_offset / 3;  // Handle is index into 3-component array

        logic::RS_RigidTorqueMem rigid_torque_mem(
            torque_signal_handle, _structure_gid, _rigid_body_sid
        );

        // Add to host layout to track position in vector
        ctx.host_data->torque_rigid_layout.add(1, _structure_gid);
        ctx.torque_rigid.append(rigid_torque_mem);

        _is_kernel_data_inited = true;
    });
}

void RS_KernelTorqueManager::_freeKernelData() {
    if (_is_kernel_data_inited) {
        castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
            lock_guard lock(k_ptr->getSyncMutex());
            auto &ctx = k_ptr->ctx;

            // Sync to make sure all operations in memory provider stream are finished
            _kernel_manager->getDeviceProvider()->sync();

            // Context data is modified, needs to update kernel
            k_ptr->markModified();

            // Remove torque memory from vector and layout
            RIndex torque_mem_index = ctx.host_data->torque_rigid_layout.find(_structure_gid);
            ctx.torque_rigid.erase(torque_mem_index);
            ctx.host_data->torque_rigid_layout.remove(torque_mem_index);

            // Update torque signal layout
            ctx.host_data->torque_signal_layout.remove(
                ctx.host_data->torque_signal_layout.find(_structure_gid)
            );

            // Update device layout
            ctx.host_data->torque_signal_layout.syncDeviceLayout(ctx.torque_signal_layout);

            // Resize device signal buffer
            ctx.torque_signals.resize(ctx.host_data->torque_signal_layout.capacity());
        });

        _is_kernel_data_inited = false;
    }
}

}  // namespace rise
