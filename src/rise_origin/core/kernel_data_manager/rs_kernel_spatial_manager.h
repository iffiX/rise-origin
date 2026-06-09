// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_SPATIAL_MANAGER_H
#define RS_KERNEL_SPATIAL_MANAGER_H

namespace rise {

class RS_KernelManager;

class RS_KernelSpatialManager {
   public:
    RS_KernelSpatialManager() = default;

    void init(RS_KernelManager *kernel_manager);

    void free();

    void markStaticVoxelModified();
    bool isStaticVoxelModified() const;
    void clearStaticVoxelModified();

    void markFloorTriangleModified();
    bool isFloorTriangleModified() const;
    void clearFloorTriangleModified();

   private:
    RS_KernelManager *_kernel_manager = nullptr;

    bool _is_static_voxel_modified = true;
    bool _is_floor_triangle_modified = true;
};

}  // namespace rise

#endif  // RS_KERNEL_SPATIAL_MANAGER_H
