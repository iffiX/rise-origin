// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VOXEL_FACE_NORMAL_M_H
#define RS_VOXEL_FACE_NORMAL_M_H

#include "rise_origin/utils/math/rs_normal.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * @brief Smoothed normals of one surface voxel's six faces, held in the voxel rest frame.
 */
class RS_VoxelFaceNormalMem {
   public:
    /// Structure gid of this face normal set.
    RIndex structure_gid = RS_NULL_INDEX;

    /// Structure scope id of this face normal set.
    RIndex sid = RS_NULL_INDEX;

    /// Smoothed normal per face in the voxel rest frame, ordered by RSE_LinkDirection.
    RS_NormalOctMem face_normal[6] = {};
};

}  // namespace logic

}  // namespace rise

#define REFLECT_VOXEL_FACE_NORMAL_MEM(voxel_face_normal_mem_type) \
    REFL_AUTO(                                                    \
        type(voxel_face_normal_mem_type),                         \
        field(structure_gid),                                     \
        field(sid),                                               \
        field(face_normal)                                        \
    )

REFLECT_VOXEL_FACE_NORMAL_MEM(rise::logic::RS_VoxelFaceNormalMem)

#endif  // RS_VOXEL_FACE_NORMAL_M_H
