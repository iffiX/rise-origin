// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_CAMERA_UTILS_H
#define RS_KERNEL_CAMERA_UTILS_H

#include "rise_origin/utils/rs_def.h"

namespace rise {

/// Number of low bits in a camera layout segment gid reserved for the sub-camera index.
constexpr RIndex RS_CAMERA_SEGMENT_SUB_INDEX_BITS = 3;

/// Mask selecting the sub-camera index bits of a camera layout segment gid.
constexpr RIndex RS_CAMERA_SEGMENT_SUB_INDEX_MASK =
    (RIndex(1) << RS_CAMERA_SEGMENT_SUB_INDEX_BITS) - 1;

/**
 * @brief Encode a camera's entity gid and sub-camera index into a unique layout segment gid.
 * @details One camera entity may render several sub-cameras (a stereo camera renders a left and a
 * right eye). Each sub-camera owns its own segment in the shared camera layouts, so its segment gid
 * packs the entity gid in the high bits and the sub-camera index in the low bits. A mono camera
 * always uses sub-camera index 0. At most 8 sub-cameras per entity, and entity gids up to 2^29.
 */
HOST_DEVICE_INLINE_QUALIFIER RIndex makeCameraSegmentGid(RIndex entity_gid, RIndex camera_index) {
    return (entity_gid << RS_CAMERA_SEGMENT_SUB_INDEX_BITS) |
           (camera_index & RS_CAMERA_SEGMENT_SUB_INDEX_MASK);
}

/**
 * @brief Recover the entity gid from a camera layout segment gid.
 */
HOST_DEVICE_INLINE_QUALIFIER RIndex getCameraEntityGid(RIndex segment_gid) {
    return segment_gid >> RS_CAMERA_SEGMENT_SUB_INDEX_BITS;
}

/**
 * @brief Recover the sub-camera index from a camera layout segment gid.
 */
HOST_DEVICE_INLINE_QUALIFIER RIndex getCameraSubIndex(RIndex segment_gid) {
    return segment_gid & RS_CAMERA_SEGMENT_SUB_INDEX_MASK;
}

}  // namespace rise

#endif  // RS_KERNEL_CAMERA_UTILS_H
