// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rs_kernel_spatial_manager.h"

using namespace std;
using namespace rise::device;

namespace rise {

void RS_KernelSpatialManager::init(RS_KernelManager *kernel_manager) {
    _kernel_manager = kernel_manager;
    _kernel_manager->enterKernelDataCriticalZone();
    _initKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
}

void RS_KernelSpatialManager::free() {
    if (!_kernel_manager) return;
    _kernel_manager->enterKernelDataCriticalZone();
    _freeKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
    _kernel_manager = nullptr;
    _is_voxel_partition_modified = true;
    _is_floor_triangle_modified = true;
}

void RS_KernelSpatialManager::markVoxelPartitionModified() {
    _is_voxel_partition_modified = true;
}

bool RS_KernelSpatialManager::isVoxelPartitionModified() const {
    return _is_voxel_partition_modified;
}

void RS_KernelSpatialManager::clearVoxelPartitionModified() {
    _is_voxel_partition_modified = false;
}

void RS_KernelSpatialManager::markFloorTriangleModified() { _is_floor_triangle_modified = true; }

bool RS_KernelSpatialManager::isFloorTriangleModified() const {
    return _is_floor_triangle_modified;
}

void RS_KernelSpatialManager::clearFloorTriangleModified() {
    _is_floor_triangle_modified = false;
}

void RS_KernelSpatialManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {

        auto mem_provider = _kernel_manager->getDeviceProvider();
        auto &ctx = k_ptr->ctx;
        ctx.static_voxel_bvh.init(mem_provider);
        ctx.dynamic_voxel_bvh.init(mem_provider);
        ctx.floor_triangle_bvh.init(mem_provider);
        ctx.intersect_region_bvh.init(mem_provider);
    });
}

void RS_KernelSpatialManager::_freeKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();

        ctx.static_voxel_bvh.free();
        ctx.dynamic_voxel_bvh.free();
        ctx.floor_triangle_bvh.free();
        ctx.intersect_region_bvh.free();
    });
}

}  // namespace rise
