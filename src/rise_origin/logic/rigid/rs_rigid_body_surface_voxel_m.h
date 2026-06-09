// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RIGID_BODY_SURFACE_VOXEL_M_H
#define RS_RIGID_BODY_SURFACE_VOXEL_M_H

#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * Memory layout for a "surface voxel" belonging to a RigidBody.
 *
 * \tparam FloatMemAccuracy One of RSE_FloatType (RSE_FLOAT_32, RSE_FLOAT_64)
 */
template <RSE_FloatType FloatMemAccuracy>
class RS_RigidBodySurfaceVoxelMem {
   public:
    using PositionVec = RS_Vec3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;

    /*****************************************************************
     * Pre-set attributes
     *****************************************************************/
    /// Structure gid of this rigid body surface voxel
    RIndex structure_gid = RS_NULL_INDEX;

    /// Voxel structure scope ID
    RIndex voxel_sid = RS_NULL_INDEX;

    /// RigidBody structure scope ID where this voxel is contained
    RIndex rigid_body_sid = RS_NULL_INDEX;

    /// Voxel position relative to rigid body origin
    PositionVec voxel_rb_position = PositionVec::zero();

    /*****************************************************************
     * Post-set attributes
     *****************************************************************/
    /// Voxel handle
    RHandle voxel = RS_NULL_HANDLE;

    /// RigidBody handle
    RHandle rigid_body = RS_NULL_HANDLE;
};

}  // namespace logic

}  // namespace rise

#define REFLECT_RS_RIGID_BODY_SURFACE_VOXEL_MEM(rigid_body_surface_voxel_mem_type) \
    REFL_AUTO(                                                                     \
        type(rigid_body_surface_voxel_mem_type),                                   \
        field(structure_gid),                                                      \
        field(voxel_sid),                                                          \
        field(rigid_body_sid),                                                     \
        field(voxel_rb_position),                                                  \
        field(voxel),                                                              \
        field(rigid_body)                                                          \
    )

REFLECT_RS_RIGID_BODY_SURFACE_VOXEL_MEM(
    rise::logic::RS_RigidBodySurfaceVoxelMem<rise::RSE_FLOAT_32>
)
REFLECT_RS_RIGID_BODY_SURFACE_VOXEL_MEM(
    rise::logic::RS_RigidBodySurfaceVoxelMem<rise::RSE_FLOAT_64>
)
#endif  // RS_RIGID_BODY_SURFACE_VOXEL_M_H