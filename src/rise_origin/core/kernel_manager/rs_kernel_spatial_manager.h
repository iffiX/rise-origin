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

    void markVoxelPartitionModified();
    bool isVoxelPartitionModified() const;
    void clearVoxelPartitionModified();

    void markRigidBodyPartitionModified();
    bool isRigidBodyPartitionModified() const;
    void clearRigidBodyPartitionModified();

    void markStaticVoxelMoved();
    bool isStaticVoxelMoved() const;
    void clearStaticVoxelMoved();

    void markFloorTriangleModified();
    bool isFloorTriangleModified() const;
    void clearFloorTriangleModified();

   private:
    RS_KernelManager *_kernel_manager = nullptr;

    bool _is_voxel_partition_modified = true;
    bool _is_rigid_body_partition_modified = true;
    bool _is_static_voxel_moved = false;
    bool _is_floor_triangle_modified = true;

    void _initKernelData();

    void _freeKernelData();
};

}  // namespace rise

#endif  // RS_KERNEL_SPATIAL_MANAGER_H
