// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rs_kernel_torque_manager.h"

using namespace std;

namespace rise {

RS_KernelTorqueManager::RS_KernelTorqueManager(
    RS_KernelManager *kernel_manager,
    const SupportedRSC &torque_rsc
)
    : _kernel_manager(kernel_manager), _rsc(torque_rsc) {
    const auto &torque = _rsc.as<rsc::RSC_Torque>();

    _scoped_provider =
        std::make_unique<RS_MemoryScopedProvider>(_kernel_manager->getDeviceProvider());

    // Resolve structure + segment context so commit() can proceed.
    auto &structure_managers = _kernel_manager->getKernelStructureVoxelManagers();
    auto it = structure_managers.find(torque.structure_name);
    if (it == structure_managers.end())
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure {} is not registered in the kernel manager", torque.structure_name
        ));
    auto &structure_manager = *it->second;

    // Torque applies to rigid or rigid-fixed bodies only.
    auto type = structure_manager.getSegmentPhysicalType(torque.body_sid, torque.segment_bid);
    if (type != RSE_STRUCTURE_VOXEL_MATERIAL_RIGID &&
        type != RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Torque signal requires a rigid body segment, but body_sid {} segment_bid {} in "
            "structure {} is not rigid",
            torque.body_sid,
            torque.segment_bid,
            torque.structure_name
        ));
    }

    _rigid_body_sid = structure_manager.findRigidBodySID(torque.body_sid, torque.segment_bid);
    if (_rigid_body_sid == RS_NULL_INDEX) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Rigid body with body_sid {} and segment_bid {} not found in structure {}",
            torque.body_sid,
            torque.segment_bid,
            torque.structure_name
        ));
    }

    // Reference the structure so it cannot be removed while this signal is bound.
    _structure_reference = RS_KernelStructureVoxelReference(_kernel_manager, torque.structure_name);

    // Allocate per-manager host and device buffers backing one unified torque
    // signal entry (one 3D vector).
    RBigSize total_size = RS_WorldTorqueSignal::getRequiredMemorySize(1);
    _host_signal_buffer = _scoped_provider->allocateHost(total_size, true);
    _device_signal_buffer = _scoped_provider->allocate(total_size, true);
    _host_signal_buffer.set(0);
    _device_signal_buffer.set(0);

    _host_signal.initFromMemory(_host_signal_buffer.ptr(), 1);
    _host_signal.structure_gid = _structure_reference.gid();
    _device_signal.initFromMemory(_device_signal_buffer.ptr(), 1);
    _device_signal.structure_gid = _structure_reference.gid();
    _device_signal.device = _device_signal_buffer.device();

    _signal_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_SIGNAL, torque.name);
}

void RS_KernelTorqueManager::commit() {
    _kernel_manager->enterKernelDataCriticalZone();
    _initKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;
}

void RS_KernelTorqueManager::decommit() {
    if (_is_kernel_data_inited) {
        _kernel_manager->enterKernelDataCriticalZone();
        _freeKernelData();
        _kernel_manager->exitKernelDataCriticalZone();
        _is_kernel_data_inited = false;
    }
}

bool RS_KernelTorqueManager::hasControllerSignal() { return _is_kernel_data_inited; }

void RS_KernelTorqueManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());
    }
}

RS_WorldTorqueSignal RS_KernelTorqueManager::getControllerSignal() {
    if (hasControllerSignal()) {
        RS_WorldTorqueSignal signal;
        signal.initFromMemory(_host_signal_buffer.ptr(), 1);
        signal.structure_gid = _structure_reference.gid();
        signal.device = _host_signal_buffer.device();
        return signal;
    }
    return {};
}

const RS_KernelTorqueManager::SupportedRSC &RS_KernelTorqueManager::getRSC() const { return _rsc; }

std::string RS_KernelTorqueManager::getSignalName() const {
    return _rsc.as<rsc::RSC_Torque>().name;
}

std::string RS_KernelTorqueManager::getStructureName() const {
    return _rsc.as<rsc::RSC_Torque>().structure_name;
}

RIndex RS_KernelTorqueManager::getSignalGid() const { return _signal_registration.gid(); }

RIndex RS_KernelTorqueManager::getStructureGid() const { return _structure_reference.gid(); }

void RS_KernelTorqueManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        // Update host + device layout
        ctx.host_data->torque_signal_layout.add(1, _structure_reference.gid());
        ctx.host_data->torque_signal_layout.syncDeviceLayout(ctx.torque_signal_layout);

        // Resize device signal vector and publish this manager's unified entry
        ctx.torque_signals.resize({ctx.host_data->torque_signal_layout.capacity()});
        _torque_signal = ctx.host_data->torque_signal_layout.segmentOffset(
            ctx.host_data->torque_signal_layout.find(_structure_reference.gid())
        );
        ctx.torque_signals.writeSlice(&_device_signal, 1, _torque_signal);

        // Create rigid torque memory object and add to context vector
        RHandle torque_signal_handle = _torque_signal;
        logic::RS_RigidTorqueMem rigid_torque_mem(
            torque_signal_handle, _structure_reference.gid(), _rigid_body_sid
        );

        // Add to host layout to track position in vector
        ctx.host_data->torque_rigid_layout.add(1, _structure_reference.gid());
        ctx.torque_rigid.append(rigid_torque_mem);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelTorqueManager::_freeKernelData() {
    if (_is_kernel_data_inited) {
        castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
            auto &ctx = k_ptr->ctx;

            // Remove torque memory from vector and layout
            RIndex torque_mem_index = ctx.host_data->torque_rigid_layout.find(_structure_reference.gid());
            ctx.torque_rigid.erase(torque_mem_index);
            ctx.host_data->torque_rigid_layout.remove(torque_mem_index);

            // Update torque signal layout
            ctx.host_data->torque_signal_layout.remove(
                ctx.host_data->torque_signal_layout.find(_structure_reference.gid())
            );

            // Update device layout
            ctx.host_data->torque_signal_layout.syncDeviceLayout(ctx.torque_signal_layout);

            // Resize device signal vector
            ctx.torque_signals.resize({ctx.host_data->torque_signal_layout.capacity()});

            // Context data is modified, needs to update kernel
            _kernel_manager->markKernelModified();
        });
    }
}

}  // namespace rise
