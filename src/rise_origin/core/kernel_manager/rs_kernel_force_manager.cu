// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rs_kernel_force_manager.h"

using namespace std;

namespace rise {

RS_KernelForceManager::RS_KernelForceManager(
    RS_KernelManager *kernel_manager,
    const SupportedRSC &force_rsc
)
    : _kernel_manager(kernel_manager), _rsc(force_rsc) {
    const auto &force = _rsc.as<rsc::RSC_Force>();

    _scoped_provider =
        std::make_unique<RS_MemoryScopedProvider>(_kernel_manager->getDeviceProvider());

    // Resolve structure + segment context so commit() can proceed.
    auto &structure_managers = _kernel_manager->getKernelStructureVoxelManagers();
    auto it = structure_managers.find(force.structure_name);
    if (it == structure_managers.end())
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure {} is not registered in the kernel manager", force.structure_name
        ));
    auto &structure_manager = *it->second;

    auto type = structure_manager.getSegmentPhysicalType(force.body_sid, force.segment_bid);
    _is_rigid = type == RSE_STRUCTURE_VOXEL_MATERIAL_RIGID ||
                type == RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED;

    if (_is_rigid) {
        _rigid_body_sid = structure_manager.findRigidBodySID(force.body_sid, force.segment_bid);
        if (_rigid_body_sid == RS_NULL_INDEX) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Rigid body with body_sid {} and segment_bid {} not found in structure {}",
                force.body_sid,
                force.segment_bid,
                force.structure_name
            ));
        }
    } else {
        _soft_voxel_count = structure_manager.getSegmentVoxelNumInRange(
            force.body_sid, force.segment_bid, force.voxel_range
        );
        if (_soft_voxel_count == 0) {
            RS_THROW(RS_InvalidCharacterizationException(
                "No soft voxels found in range for body_sid {} segment_bid {} in structure {}",
                force.body_sid,
                force.segment_bid,
                force.structure_name
            ));
        }
    }

    // Reference the structure so it cannot be removed while this signal is bound.
    _structure_reference = RS_KernelStructureVoxelReference(_kernel_manager, force.structure_name);

    // Allocate per-manager host and device buffers backing one unified force
    // signal entry (one 3D vector).
    RBigSize total_size = RS_WorldForceSignal::getRequiredMemorySize(1);
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
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_SIGNAL, force.name);
}

void RS_KernelForceManager::commit() {
    _kernel_manager->enterKernelDataCriticalZone();
    _initKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;
}

void RS_KernelForceManager::decommit() {
    if (_is_kernel_data_inited) {
        _kernel_manager->enterKernelDataCriticalZone();
        _freeKernelData();
        _kernel_manager->exitKernelDataCriticalZone();
        _is_kernel_data_inited = false;
    }
}

bool RS_KernelForceManager::hasControllerSignal() { return _is_kernel_data_inited; }

void RS_KernelForceManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());
    }
}

RS_WorldForceSignal RS_KernelForceManager::getControllerSignal() {
    if (hasControllerSignal()) {
        RS_WorldForceSignal signal;
        signal.initFromMemory(_host_signal_buffer.ptr(), 1);
        signal.structure_gid = _structure_reference.gid();
        signal.device = _host_signal_buffer.device();
        return signal;
    }
    return {};
}

const RS_KernelForceManager::SupportedRSC &RS_KernelForceManager::getRSC() const { return _rsc; }

std::string RS_KernelForceManager::getSignalName() const {
    return _rsc.as<rsc::RSC_Force>().name;
}

std::string RS_KernelForceManager::getStructureName() const {
    return _rsc.as<rsc::RSC_Force>().structure_name;
}

RIndex RS_KernelForceManager::getSignalGid() const { return _signal_registration.gid(); }

RIndex RS_KernelForceManager::getStructureGid() const { return _structure_reference.gid(); }

void RS_KernelForceManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        // Update host + device layout
        ctx.host_data->force_signal_layout.add(1, _structure_reference.gid());
        ctx.host_data->force_signal_layout.syncDeviceLayout(ctx.force_signal_layout);

        // Resize device signal vector and publish this manager's unified entry
        ctx.force_signals.resize({ctx.host_data->force_signal_layout.capacity()});
        _force_signal = ctx.host_data->force_signal_layout.segmentOffset(
            ctx.host_data->force_signal_layout.find(_structure_reference.gid())
        );
        ctx.force_signals.writeSlice(&_device_signal, 1, _force_signal);

        // Create force memory object and add to appropriate context vector
        RHandle force_signal_handle = _force_signal;

        if (_is_rigid) {
            // Create rigid force memory
            logic::RS_RigidForceMem rigid_force_mem(
                force_signal_handle,
                1.0,  // unit_portion = 1.0 for rigid bodies
                _structure_reference.gid(),
                _rigid_body_sid
            );

            // Add to host layout to track position in vector
            ctx.host_data->force_rigid_layout.add(1, _structure_reference.gid());
            ctx.force_rigid.append(rigid_force_mem);
        } else {
            // Create soft force memory
            const auto &force = _rsc.as<rsc::RSC_Force>();
            RFloat unit_portion = 1.0 / static_cast<RFloat>(_soft_voxel_count);
            logic::RS_SoftForceMem soft_force_mem(
                force_signal_handle,
                unit_portion,
                _structure_reference.gid(),
                force.voxel_range,
                force.body_sid,
                force.segment_bid
            );

            // Add to host layout to track position in vector
            ctx.host_data->force_soft_layout.add(1, _structure_reference.gid());
            ctx.force_soft.append(soft_force_mem);
        }

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelForceManager::_freeKernelData() {
    if (_is_kernel_data_inited) {
        castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
            auto &ctx = k_ptr->ctx;

            // Remove force memory from appropriate vector and layout
            if (_is_rigid) {
                RIndex force_mem_index = ctx.host_data->force_rigid_layout.find(_structure_reference.gid());
                // Remove from rigid force memory vector
                ctx.force_rigid.erase(force_mem_index);

                // Update host layout
                ctx.host_data->force_rigid_layout.remove(force_mem_index);
            } else {
                RIndex force_mem_index = ctx.host_data->force_soft_layout.find(_structure_reference.gid());
                // Remove from soft force memory vector
                ctx.force_soft.erase(force_mem_index);

                // Update host layout
                ctx.host_data->force_soft_layout.remove(force_mem_index);
            }

            // Update force signal layout
            ctx.host_data->force_signal_layout.remove(
                ctx.host_data->force_signal_layout.find(_structure_reference.gid())
            );

            // Update device layout
            ctx.host_data->force_signal_layout.syncDeviceLayout(ctx.force_signal_layout);

            // Resize device signal vector
            ctx.force_signals.resize({ctx.host_data->force_signal_layout.capacity()});

            // Context data is modified, needs to update kernel
            _kernel_manager->markKernelModified();
        });
    }
}

}  // namespace rise
