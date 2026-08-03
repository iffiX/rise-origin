// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONF_H
#define RS_CONF_H

/*****************************************************************
 * Engine
 *****************************************************************/
#ifndef RS_ENGINE_VERSION
#define RS_ENGINE_VERSION "0.0.0"
#endif

/*****************************************************************
 * Debug
 *****************************************************************/
// #define RS_DEBUG_ENABLE_DEVICE_SOA true
// #define RS_DEBUG_ENABLE_DEVICE_HASHMAP true
// #define RS_DEBUG_ENABLE_LOGIC_VOXEL true
// #define RS_DEBUG_ENABLE_LOGIC_LINK true

/*****************************************************************
 * Device
 *****************************************************************/
#ifndef RS_DEVICE_MEMORY_ALIGN_SIZE
#define RS_DEVICE_MEMORY_ALIGN_SIZE 32
#endif
#ifndef RS_DEVICE_MAX_BLOCK_SIZE
#define RS_DEVICE_MAX_BLOCK_SIZE 1024
#endif
#ifndef RS_DEVICE_SHARED_MEMORY_MAX_SIZE
#define RS_DEVICE_SHARED_MEMORY_MAX_SIZE 49152
#endif
// Note: Should not use 65536 (full 64KB) for constant memory,
// as ptxas will reject it.
#ifndef RS_DEVICE_CONSTANT_MEMORY_MAX_SIZE
#define RS_DEVICE_CONSTANT_MEMORY_MAX_SIZE 62000
#endif
#ifndef RS_DEVICE_ENABLE_INTRINSICS
#define RS_DEVICE_ENABLE_INTRINSICS
#endif


#ifndef RS_DEVICE_CPU_FIBER_STACK_SIZE
#define RS_DEVICE_CPU_FIBER_STACK_SIZE 32768
#endif
#ifndef RS_DEVICE_CPU_THREAD_POOL_TASK_TO_THREAD_RATIO
#define RS_DEVICE_CPU_THREAD_POOL_TASK_TO_THREAD_RATIO 8
#endif
#ifndef RS_DEVICE_CPU_DEFAULT_BLOCK_SIZE
#define RS_DEVICE_CPU_DEFAULT_BLOCK_SIZE 256
#endif

#ifndef RS_DEVICE_CUDA_MAX_WARP_SIZE
#define RS_DEVICE_CUDA_MAX_WARP_SIZE 32
#endif
#ifndef RS_DEVICE_CUDA_DEFAULT_BLOCK_SIZE
#define RS_DEVICE_CUDA_DEFAULT_BLOCK_SIZE 256
#endif

/*****************************************************************
 * Runtime
 *****************************************************************/
// Depth of the per-thread BVH traversal stack.
#ifndef RS_RUNTIME_BVH_STACK_SIZE
#define RS_RUNTIME_BVH_STACK_SIZE 32
#endif

/*****************************************************************
 * RSC
 *****************************************************************/
// Current durable RSC file schema version this build writes.
#ifndef RS_RSC_VERSION
#define RS_RSC_VERSION "1.0.0"
#endif

// RSC characterization dialect this build reads and writes.
#ifndef RS_RSC_EDITION
#define RS_RSC_EDITION "origin"
#endif

#ifndef RS_RSC_STRUCTURE_VOXEL_VALIDATION_BLOCK_SIZE
#define RS_RSC_STRUCTURE_VOXEL_VALIDATION_BLOCK_SIZE 1024
#endif

/*****************************************************************
 * World
 *****************************************************************/
// Number of entries the world message log retains before evicting the oldest.
#ifndef RS_WORLD_MESSAGE_BUFFER_DEFAULT_SIZE
#define RS_WORLD_MESSAGE_BUFFER_DEFAULT_SIZE 4096
#endif

// World range in meters. Dynamic structures use the tighter range; static geometry the wider one,
// which must stay under the centimeter-unit Morton broadphase coverage of about 10485 m.
#ifndef RS_WORLD_MAX_DYNAMIC_RANGE
#define RS_WORLD_MAX_DYNAMIC_RANGE 1000.0
#endif
#ifndef RS_WORLD_MAX_STATIC_RANGE
#define RS_WORLD_MAX_STATIC_RANGE 10000.0
#endif

// Hard limit on algorithms.
#ifndef RS_WORLD_VISION_TRACE_MAX_ARRAY_FLOOR_TILE_NUM
#define RS_WORLD_VISION_TRACE_MAX_ARRAY_FLOOR_TILE_NUM 16
#endif
#ifndef RS_WORLD_VISION_TRACE_MAX_DEPTH
#define RS_WORLD_VISION_TRACE_MAX_DEPTH 64
#endif

// Reserved GIDs.
#ifndef RS_WORLD_RSC_ENTITY_RESERVED_GID_NUM
#define RS_WORLD_RSC_ENTITY_RESERVED_GID_NUM 1000
#endif
#ifndef RS_WORLD_RSC_COSMIC_BACKGROUND_GID
#define RS_WORLD_RSC_COSMIC_BACKGROUND_GID 0
#endif
#ifndef RS_WORLD_RSC_FLOOR_GID
#define RS_WORLD_RSC_FLOOR_GID 1
#endif

// Hard limit on optical properties per material, set by the 10-bit optical mid packed into the
// camera pixel object key alongside the material gid.
#ifndef RS_WORLD_RSC_MATERIAL_MAX_NUM
#define RS_WORLD_RSC_MATERIAL_MAX_NUM (1u << 22)
#endif

#ifndef RS_WORLD_RSC_MATERIAL_MAX_OPTICAL_PROPERTY_NUM
#define RS_WORLD_RSC_MATERIAL_MAX_OPTICAL_PROPERTY_NUM 1024
#endif

// Per-structure-voxel limits.
/// Hard limit on total bodies, set by the 20-bit body_sid contact filter field.
#ifndef RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_BODY_NUM
#define RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_BODY_NUM (1u << 20)
#endif
/// Hard limit on segment ids per body, set by the 16-bit segment_bid contact filter field.
#ifndef RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_BODY_SEGMENT_NUM
#define RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_BODY_SEGMENT_NUM (1u << 16)
#endif
#ifndef RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_VOXEL_NUM
#define RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_VOXEL_NUM (1u << 31)
#endif
#ifndef RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_LINK_NUM
#define RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_LINK_NUM (1u << 31)
#endif
/// Rigid body count ceiling in an unconstrained structure.
#ifndef RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_RIGID_BODY_UNCONSTRAINED_NUM
#define RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_RIGID_BODY_UNCONSTRAINED_NUM (1u << 20)
#endif
/// Rigid body count ceiling in a constrained structure, set by the 128-bit connected mask.
#ifndef RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_RIGID_BODY_CONSTRAINED_NUM
#define RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_RIGID_BODY_CONSTRAINED_NUM 128
#endif
#ifndef RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_CONSTRAINT_NUM
#define RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_CONSTRAINT_NUM 128
#endif

#endif
