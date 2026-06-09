// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_LINK_M_H
#define RS_LINK_M_H

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_quat.h"
#include "rise_origin/utils/rs_vec.h"

namespace rise {

namespace logic {

union RS_LinkStateUnion {
    RState link_state_packed;
    struct LinkState {
        RState axis : 3;
        RState is_enabled : 1;
        RState is_tendon : 1;
        RState is_failed : 1;
        RState voxel_neg_rigid : 1;
        RState voxel_pos_rigid : 1;
    } link_state;
    HOST_DEVICE_QUALIFIER explicit RS_LinkStateUnion(unsigned int link_state_packed)
        : link_state_packed(link_state_packed) {}
};

template <RSE_FloatType FloatMemAccuracy>
class RS_LinkMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;
    using Vec3M = RS_Vec3DMem<FloatMemAccuracy>;

    /**
     * Pre-set attributes
     */
    /// Structure gid of this link
    RIndex structure_gid = RS_NULL_INDEX;

    /// Structure scope ID of this link
    RIndex sid = RS_NULL_INDEX;

    /// Structure scope ID of negative and positive voxels
    RIndex voxel_neg_sid = RS_NULL_INDEX, voxel_pos_sid = RS_NULL_INDEX;

    /// Structure scope ID of link material
    RIndex link_material_sid = RS_NULL_INDEX;

    /// Link state
    RState state = 0;

    FloatM strain_ratio = 0;

    /**
     * Post-set attributes
     */

    /// Handles of negative and positive voxels
    RHandle voxel_neg = RS_NULL_HANDLE, voxel_pos = RS_NULL_HANDLE;

    /// Handle of material
    RHandle link_material = RS_NULL_HANDLE;

    Vec3M force_neg = Vec3M::zero(), force_pos = Vec3M::zero();
    Vec3M moment_neg = Vec3M::zero(), moment_pos = Vec3M::zero();

    // link length, also it is the relative position of voxel_pos,
    // after link re-orientation
    FloatM length = 0;
    FloatM rest_length = 0;

    // axial strain
    FloatM strain = 0;
    FloatM stress = 0;
    FloatM transverse_area = 0;

    // Orientation angles of both voxels after link re-orientation
    Vec3M angle1v = Vec3M::zero(), angle2v = Vec3M::zero();
};

}  // namespace logic

}  // namespace rise

#define REFLECT_LINK_MEM(link_mem_type) \
    REFL_AUTO(                          \
        type(link_mem_type),            \
        field(structure_gid),           \
        field(sid),                     \
        field(voxel_neg_sid),           \
        field(voxel_pos_sid),           \
        field(link_material_sid),       \
        field(state),                   \
        field(strain_ratio),            \
        field(voxel_neg),               \
        field(voxel_pos),               \
        field(link_material),           \
        field(force_neg),               \
        field(force_pos),               \
        field(moment_neg),              \
        field(moment_pos),              \
        field(length),                  \
        field(rest_length),             \
        field(strain),                  \
        field(stress),                  \
        field(transverse_area),         \
        field(angle1v),                 \
        field(angle2v)                  \
    )

REFLECT_LINK_MEM(rise::logic::RS_LinkMem<rise::RSE_FLOAT_32>)
REFLECT_LINK_MEM(rise::logic::RS_LinkMem<rise::RSE_FLOAT_64>)
#endif  // RS_LINK_M_H
