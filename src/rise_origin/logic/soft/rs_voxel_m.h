// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VOXEL_M_H
#define RS_VOXEL_M_H

#include <cstdint>
#include <functional>

#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/device/data/rs_bvh.h"

namespace rise {

namespace logic {

RS_ENUM(
    RSE_VoxelFrictionState,
    RState,
    (RSE_VOXEL_NO_CONTACT_FRICTION, 0),  //!< Not in contact; no friction state.
    (RSE_VOXEL_STATIC_FRICTION, 1),      //!< In contact, sticking (static) friction.
    (RSE_VOXEL_DYNAMIC_FRICTION, 2)      //!< In contact, sliding (dynamic) friction.
)

RS_ENUM(
    RSE_VoxelFaceRole,
    RState,
    (RSE_VOXEL_FACE_INTERIOR, 0),  //!< Neighbor present, same vision material; ray passes through.
    (RSE_VOXEL_FACE_AIR, 1),       //!< No neighbor; material<->air interface.
    (RSE_VOXEL_FACE_OPAQUE_BOUNDARY,
     2),  //!< Neighbor present, different non-transparent material; ray stops.
    (RSE_VOXEL_FACE_TRANSPARENT_BOUNDARY,
     3)  //!< Neighbor present, different transparent material; ray refracts across.
)

RS_ENUM(
    RSE_VoxelExpansionControlMode,
    RState,
    (RSE_VOXEL_EXPANSION_CONTROL_BASIC_POSITION, 0),  //!< Signal is the target expansion ratio.
    (RSE_VOXEL_EXPANSION_CONTROL_RATE, 1)             //!< Signal is the expansion rate.
)

union RS_VoxelStateUnion {
    RState voxel_state_packed;
    struct RS_VoxelState {
        /// Friction regime recorded on the previous step.
        RState previous_friction_state : 2;
        /// Whether the voxel participates in the world.
        RState is_enabled : 1;
        /// Whether the voxel lies on its body surface.
        RState is_surface : 1;
        /// Whether the voxel is currently immobile, from fixed material or freezing.
        RState is_static : 1;
        /// Whether the voxel belongs to a rigid body.
        RState is_rigid : 1;
        /// Whether the voxel is permanently fixed by its material.
        RState is_fixed : 1;
        /// Whether the voxel has at least one soft link.
        RState has_link : 1;
        /// Whether the voxel carries an expansion signal.
        RState has_expansion : 1;
        /// How the expansion signal is interpreted.
        RState expansion_control_mode : 1;
        /// Whether contact slot 0 holds a voxel contact.
        RState has_contact_voxel_0 : 1;
        /// Whether the slot 0 contact voxel is rigid or static.
        RState is_contact_voxel_0_rigid_or_static : 1;
        /// Whether contact slot 1 holds a voxel contact.
        RState has_contact_voxel_1 : 1;
        /// Whether the slot 1 contact voxel is rigid or static.
        RState is_contact_voxel_1_rigid_or_static : 1;
        /// Whether the voxel is in contact with the floor.
        RState has_contact_floor : 1;
        /// Whether the owning structure has at least one constraint.
        RState is_structure_constrained : 1;
        /// Packed per-face role bits driving render and contact activity.
        RState face_role : 12;
    } voxel_state;
};

union RS_VoxelBVHLeafMetadataUnion {
    RState meta_packed;
    struct RS_VoxelBVHLeafMetadata {
        /// Whether the voxel participates in the world.
        RState is_enabled : 1;
        /// Whether the voxel lies on its body surface.
        RState is_surface : 1;
        /// Whether the voxel is currently immobile, from fixed material or freezing.
        RState is_static : 1;
        /// Whether the voxel belongs to a rigid body.
        RState is_rigid : 1;
        /// Whether the voxel emits light.
        RState is_emissive : 1;
        /// Whether the voxel is rendered as transparent.
        RState is_transparent : 1;
        /// Whether the voxel is included in render traversal.
        RState is_render_active : 1;
        /// Whether the voxel is included in contact traversal.
        RState is_contact_active : 1;
        /// Whether the owning structure has at least one constraint.
        RState is_structure_constrained : 1;
    } meta;
};

union RS_VoxelContactFilterKeyUnion {
    RBigIndex contact_filter_key_packed;
    struct RS_VoxelContactFilterKey {
        RBigIndex body_sid : 20;
        RBigIndex constrained_rigid_body_sid : 8;
        RBigIndex index_x_res : 4;
        RBigIndex index_y_res : 4;
        RBigIndex index_z_res : 4;
    } contact_filter_key;
};

static constexpr RBigIndex RS_VOXEL_CONTACT_FILTER_KEY_NULL_CONSTRAINED_RIGID_BODY_SID = 0xFF;

/// Voxel-BVH leaf: caches the contact broad-phase payload inline.
struct RS_VoxelBVHObjectIdentifier {
    /// Packed RS_VoxelBVHLeafMetadata (is_enabled / is_surface / ...).
    RState object_metadata = 0;
    /// Handle of the voxel this leaf wraps.
    RHandle object_handle = RS_NULL_HANDLE;
    /// Packed RS_VoxelContactFilterKey.
    RBigIndex contact_filter_key = 0;
    /// World-space sphere centre for the overlap test.
    RVec3f position = RVec3f::zero();
    /// Sphere radius (OBB half-diagonal) for the overlap test.
    float radius = 0.0f;
    /// Owning structure gid; gates the same-body / connected-body filters.
    RIndex structure_gid = RS_NULL_INDEX;

    bool operator==(const RS_VoxelBVHObjectIdentifier& other) const {
        return object_handle == other.object_handle;
    }
};

struct RS_VoxelBVHObjectIdentifierHostHasher {
    std::size_t operator()(const RS_VoxelBVHObjectIdentifier& identifier) const {
        return std::hash<RHandle>()(identifier.object_handle);
    }
};

using RS_VoxelBVH = device::RS_BVHBase<
    RBigIndex,
    logic::RS_VoxelBVHObjectIdentifier,
    logic::RS_VoxelBVHObjectIdentifierHostHasher>;

template <RSE_FloatType FloatMemAccuracy>
class RS_VoxelMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;
    using Vec3M = RS_Vec3DMem<FloatMemAccuracy>;
    using Quat3M = RS_Quat3DMem<FloatMemAccuracy>;
    using PositionVec = RS_Vec3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;

    /**
     * Pre-set attributes
     */
    /// Structure gid of this voxel.
    RIndex structure_gid = RS_NULL_INDEX;

    /// ID of this voxel in the structure.
    RIndex sid = RS_NULL_INDEX;

    /// Body index in the structure.
    RIndex body_sid = RS_NULL_INDEX;

    /// Body segment id in the structure.
    RIndex segment_bid = RS_NULL_INDEX;

    /// Rigid body segment id in the structure.
    RIndex rigid_body_sid = RS_NULL_INDEX;

    /// X, Y, Z index of this voxel in the body.
    RSignedIndex index_x = 0, index_y = 0, index_z = 0;

    /// Packed contact-filter key.
    RBigIndex contact_filter_key = 0;

    /// Material id of this voxel.
    RIndex voxel_material_sid = RS_NULL_INDEX;

    /// Expansion signal id.
    RIndex expansion_signal_sid = RS_NULL_INDEX;

    /// Structure signal handle.
    RHandle structure_signal = RS_NULL_HANDLE;

    /// Link ids in the 6 cardinal directions according to RSE_LinkDirection enumeration.
    RIndex link_sid[6] =
        {RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX};

    /// Neighbor voxel ids in the 6 cardinal directions according to RSE_LinkDirection enumeration.
    RIndex neighbor_voxel_sid[6] =
        {RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX};

    /**
     * Post-set attributes
     */
    /// Voxel state
    RState state = 0;

    /// Rigid body where this voxel belongs to
    RHandle rigid_body = RS_NULL_HANDLE;

    /// Material of this voxel
    RHandle voxel_material = RS_NULL_HANDLE;

    /// Material of this voxel for vision (Note, it is preset, an exception for the rule)
    RHandle vision_material = RS_NULL_HANDLE;

    /// Links in the 6 cardinal directions according to RSE_LinkDirection enumeration
    RHandle links[6] = {
        RS_NULL_HANDLE,
        RS_NULL_HANDLE,
        RS_NULL_HANDLE,
        RS_NULL_HANDLE,
        RS_NULL_HANDLE,
        RS_NULL_HANDLE
    };

    /// Neighbor voxels in the 6 cardinal directions according to RSE_LinkDirection enumeration
    RHandle neighbor_voxels[6] = {
        RS_NULL_HANDLE,
        RS_NULL_HANDLE,
        RS_NULL_HANDLE,
        RS_NULL_HANDLE,
        RS_NULL_HANDLE,
        RS_NULL_HANDLE
    };

    /// Contact voxel 0
    RHandle contact_voxel_0 = RS_NULL_HANDLE;

    /// Contact position of contact voxel 0
    PositionVec contact_position_0 = PositionVec::zero();

    /// Contact normal of contact voxel 0
    Vec3M contact_normal_0 = Vec3M::zero();

    /// Contact force of contact voxel 0, includes bounce force and friction force
    Vec3M contact_force_0 = Vec3M::zero();

    /// Contact depth of contact voxel 0
    FloatM contact_depth_0 = 0;

    /// Accumulated normal contact lambda for contact voxel 0
    FloatM contact_lambda_n_0 = 0;

    /// Accumulated tangent friction contact lambda for contact voxel 0
    Vec3M contact_lambda_t_0 = Vec3M::zero();

    /// Contact voxel 1
    RHandle contact_voxel_1 = RS_NULL_HANDLE;

    /// Contact position of contact voxel 1
    PositionVec contact_position_1 = PositionVec::zero();

    /// Contact normal of contact voxel 1
    Vec3M contact_normal_1 = Vec3M::zero();

    /// Contact force of contact voxel 1, includes bounce force and friction force
    Vec3M contact_force_1 = Vec3M::zero();

    /// Contact depth of contact voxel 1
    FloatM contact_depth_1 = 0;

    /// Accumulated normal contact lambda for contact voxel 1
    FloatM contact_lambda_n_1 = 0;

    /// Accumulated tangent friction contact lambda for contact voxel 1
    Vec3M contact_lambda_t_1 = Vec3M::zero();

    /// Accumulated normal contact lambda for the static floor contact
    FloatM contact_lambda_n_floor = 0;

    /// Accumulated tangent friction contact lambda for the static floor contact
    Vec3M contact_lambda_t_floor = Vec3M::zero();

    /// Global position of this voxel
    PositionVec position = PositionVec::zero();

    /// Global orientation of this voxel.
    Quat3M orientation = Quat3M::unit();

    /// External force applied to this voxel
    Vec3M ext_force = Vec3M::zero();

    /// External moment applied to this voxel
    Vec3M ext_moment = Vec3M::zero();

    /// Voxel size (with expansion)
    FloatM size = 0;

    /// Corner offsets of vertex NNN and PPP
    Vec3M nnn_offset = Vec3M::zero(), ppp_offset = Vec3M::zero();

    /// Half-diagonal of the OBB
    FloatM radius = 0;

    /// Current linear momentum
    Vec3M linear_momentum = Vec3M::zero();

    /// Current angular momentum
    Vec3M angular_momentum = Vec3M::zero();

    /// 0 is no expansion
    FloatM expansion = 0;

    /// Cached poissons strain
    Vec3M poissons_strain = Vec3M::zero();
};

}  // namespace logic

}  // namespace rise

#define REFLECT_VOXEL_MEM(voxel_mem_type) \
    REFL_AUTO(                            \
        type(voxel_mem_type),             \
        field(structure_gid),             \
        field(sid),                       \
        field(body_sid),                  \
        field(segment_bid),               \
        field(rigid_body_sid),            \
        field(index_x),                   \
        field(index_y),                   \
        field(index_z),                   \
        field(contact_filter_key),        \
        field(voxel_material_sid),        \
        field(expansion_signal_sid),      \
        field(structure_signal),          \
        field(link_sid),                  \
        field(neighbor_voxel_sid),        \
        field(state),                     \
        field(rigid_body),                \
        field(voxel_material),            \
        field(vision_material),           \
        field(links),                     \
        field(neighbor_voxels),           \
        field(contact_voxel_0),           \
        field(contact_position_0),        \
        field(contact_normal_0),          \
        field(contact_force_0),           \
        field(contact_depth_0),           \
        field(contact_lambda_n_0),        \
        field(contact_lambda_t_0),        \
        field(contact_voxel_1),           \
        field(contact_position_1),        \
        field(contact_normal_1),          \
        field(contact_force_1),           \
        field(contact_depth_1),           \
        field(contact_lambda_n_1),        \
        field(contact_lambda_t_1),        \
        field(contact_lambda_n_floor),    \
        field(contact_lambda_t_floor),    \
        field(position),                  \
        field(orientation),               \
        field(ext_force),                 \
        field(ext_moment),                \
        field(size),                      \
        field(nnn_offset),                \
        field(ppp_offset),                \
        field(radius),                    \
        field(linear_momentum),           \
        field(angular_momentum),          \
        field(expansion),                 \
        field(poissons_strain)            \
    )

REFLECT_VOXEL_MEM(rise::logic::RS_VoxelMem<rise::RSE_FLOAT_32>)
REFLECT_VOXEL_MEM(rise::logic::RS_VoxelMem<rise::RSE_FLOAT_64>)

#endif  // RS_VOXEL_M_H
