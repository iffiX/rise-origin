// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONF_H
#define RS_CONF_H

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
 * World
 *****************************************************************/
// Reserved id for messages the world emits internally.
#ifndef RS_WORLD_MESSAGE_SYSTEM_ID
#define RS_WORLD_MESSAGE_SYSTEM_ID (0xFFFFFFFEu)
#endif

// Message ids wrap back to 0 once they pass this threshold; stays below the system id.
#ifndef RS_WORLD_MESSAGE_ID_ROTATE_THRESHOLD
#define RS_WORLD_MESSAGE_ID_ROTATE_THRESHOLD (0xFFFFFFFDu)
#endif

// World range in meters.
#ifndef RS_WORLD_MAX_RANGE
#define RS_WORLD_MAX_RANGE 1000.0
#endif

// Hard limit on algorithms.
#ifndef RS_WORLD_VISION_TRACE_MAX_ARRAY_FLOOR_TILE_NUM
#define RS_WORLD_VISION_TRACE_MAX_ARRAY_FLOOR_TILE_NUM 16
#endif
#ifndef RS_WORLD_VISION_TRACE_MAX_DEPTH
#define RS_WORLD_VISION_TRACE_MAX_DEPTH 64
#endif

// Per-structure-voxel limits.
/// Hard limit on total bodies, set by the 20-bit body_sid contact filter field.
#ifndef RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_BODY_NUM
#define RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_BODY_NUM (1u << 20)
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
