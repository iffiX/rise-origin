// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rs_kernel_transform_manager.h"

using namespace std;

namespace rise {

RS_KernelTransformManager::RS_KernelTransformManager(
    RS_KernelManager *kernel_manager,
    const SupportedRSC &transform_rsc
)
    : _kernel_manager(kernel_manager), _rsc(transform_rsc) {
    const auto &transform = _rsc.as<rsc::RSC_Transform>();

    _scoped_device_provider =
        std::make_unique<RS_MemoryScopedProvider>(_kernel_manager->getDeviceProvider());

    // Resolve structure context so commit() can proceed.
    auto &structure_managers = _kernel_manager->getKernelStructureVoxelManagers();
    auto it = structure_managers.find(transform.structure_name);
    if (it == structure_managers.end())
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure {} is not registered in the kernel manager", transform.structure_name
        ));

    // Reference the structure so it cannot be removed while this signal is bound.
    _structure_reference =
        RS_KernelStructureVoxelReference(_kernel_manager, transform.structure_name);

    // Allocate per-manager host and device buffers backing one unified transform
    // signal entry (one translation, rotation, and pivot).
    RBigSize total_size = RS_WorldTransformSignal::getRequiredMemorySize(1, 1);
    _host_signal_buffer = _scoped_device_provider->allocateHost(total_size, true);
    _device_signal_buffer = _scoped_device_provider->allocate(total_size, true);
    _host_signal_buffer.set(0);
    _device_signal_buffer.set(0);

    _host_signal.initFromMemory(_host_signal_buffer.ptr(), 1, 1);
    _host_signal.structure_gid = _structure_reference.gid();
    _device_signal.initFromMemory(_device_signal_buffer.ptr(), 1, 1);
    _device_signal.structure_gid = _structure_reference.gid();
    _device_signal.device = _device_signal_buffer.device();

    // A zeroed buffer holds a degenerate quaternion, so seed identity on both
    // host and device before the first apply.
    _writeIdentitySignal(_host_signal);
    _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());

    _signal_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_SIGNAL, transform.name);
}

void RS_KernelTransformManager::commit() {
    _kernel_manager->enterKernelDataCriticalZone();
    _initKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;
}

void RS_KernelTransformManager::decommit() {
    if (_is_kernel_data_inited) {
        _kernel_manager->enterKernelDataCriticalZone();
        _freeKernelData();
        _kernel_manager->exitKernelDataCriticalZone();
        _is_kernel_data_inited = false;
    }
}

bool RS_KernelTransformManager::hasControllerSignal() { return _is_kernel_data_inited; }

void RS_KernelTransformManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());

        // A non-identity transform moves voxels; if any are static, flag a static rebuild for the
        // executor to raise once the move is applied next step.
        if (!_isIdentitySignal(_host_signal)) {
            auto &structure_managers = _kernel_manager->getKernelStructureVoxelManagers();
            auto it = structure_managers.find(getStructureName());
            if (it != structure_managers.end() && it->second->hasStaticVoxels()) {
                _is_static_rebuild_pending = true;
            }
        }
    }
}

bool RS_KernelTransformManager::isStaticRebuildPending() const {
    return _is_static_rebuild_pending;
}

void RS_KernelTransformManager::clearStaticRebuildPending() {
    _is_static_rebuild_pending = false;

    // The transform applies once, so clear the host value now that it has been applied.
    if (hasControllerSignal()) _writeIdentitySignal(_host_signal);
}

RS_WorldTransformSignal RS_KernelTransformManager::getControllerSignal() {
    if (hasControllerSignal()) {
        RS_WorldTransformSignal signal;
        signal.initFromMemory(_host_signal_buffer.ptr(), 1, 1);
        signal.structure_gid = _structure_reference.gid();
        signal.device = _host_signal_buffer.device();
        return signal;
    }
    return {};
}

const RS_KernelTransformManager::SupportedRSC &RS_KernelTransformManager::getRSC() const {
    return _rsc;
}

std::string RS_KernelTransformManager::getSignalName() const {
    return _rsc.as<rsc::RSC_Transform>().name;
}

std::string RS_KernelTransformManager::getStructureName() const {
    return _rsc.as<rsc::RSC_Transform>().structure_name;
}

RIndex RS_KernelTransformManager::getSignalGid() const { return _signal_registration.gid(); }

RIndex RS_KernelTransformManager::getStructureGid() const { return _structure_reference.gid(); }

void RS_KernelTransformManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        // Update host + device layout
        ctx.host_data->transform_signal_layout.add(1, _structure_reference.gid());
        ctx.host_data->transform_signal_layout.syncDeviceLayout(ctx.transform_signal_layout);

        // Resize device signal vector and publish this manager's unified entry
        ctx.transform_signals.resize({ctx.host_data->transform_signal_layout.capacity()});
        _transform_signal = ctx.host_data->transform_signal_layout.segmentOffset(
            ctx.host_data->transform_signal_layout.find(_structure_reference.gid())
        );
        ctx.transform_signals.writeSlice(&_device_signal, 1, _transform_signal);

        // Create transform memory object and add to context vector
        RHandle transform_signal_handle = _transform_signal;
        logic::RS_TransformMem transform_mem(transform_signal_handle, _structure_reference.gid());

        // Add to host layout to track position in vector
        ctx.host_data->transform_layout.add(1, _structure_reference.gid());
        ctx.transform_mem.append(transform_mem);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelTransformManager::_freeKernelData() {
    if (_is_kernel_data_inited) {
        castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
            auto &ctx = k_ptr->ctx;

            // Remove transform memory from vector and layout
            RIndex transform_mem_index =
                ctx.host_data->transform_layout.find(_structure_reference.gid());
            ctx.transform_mem.erase(transform_mem_index);
            ctx.host_data->transform_layout.remove(transform_mem_index);

            // Update transform signal layout
            ctx.host_data->transform_signal_layout.remove(
                ctx.host_data->transform_signal_layout.find(_structure_reference.gid())
            );

            // Update device layout
            ctx.host_data->transform_signal_layout.syncDeviceLayout(ctx.transform_signal_layout);

            // Resize device signal vector
            ctx.transform_signals.resize({ctx.host_data->transform_signal_layout.capacity()});

            // Context data is modified, needs to update kernel
            _kernel_manager->markKernelModified();
        });
    }
}

bool RS_KernelTransformManager::_isIdentitySignal(const RS_WorldTransformSignal &signal) {
    return signal.translation_signals.translation[0] == RVec3rf::zero() &&
           signal.rotation_signals.rotation[0] == RQuat3rf::unit();
}

void RS_KernelTransformManager::_writeIdentitySignal(RS_WorldTransformSignal &signal) {
    signal.translation_signals.translation[0] = RVec3rf::zero();
    signal.rotation_signals.rotation[0] = RQuat3rf::unit();
    signal.rotation_signals.pivot[0] = RVec3rf::zero();
}

}  // namespace rise
