// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VOXEL_C_H
#define RS_VOXEL_C_H

#include "rise_origin/core/rs_kernel.h"
#include "rise_origin/device/rs_static_construct.h"
#include "rise_origin/logic/soft/rs_link_c.h"
#include "rise_origin/utils/rs_conf.h"

namespace rise {

namespace logic {

/** @brief Read the role of face `face` (0..5) from a packed voxel state. */
HOST_DEVICE_INLINE_QUALIFIER RSE_VoxelFaceRole getFaceRole(RState state, int face) {
    RS_VoxelStateUnion u{state};
    return static_cast<RSE_VoxelFaceRole>((u.voxel_state.face_role >> (2 * face)) & RState(0x3));
}

/** @brief Read the role of face `face` (0..5) from packed ray-BVH leaf metadata. */
HOST_DEVICE_INLINE_QUALIFIER RSE_VoxelFaceRole getRayLeafFaceRole(RState metadata, int face) {
    RS_VoxelRayBVHLeafMetadataUnion u{metadata};
    return static_cast<RSE_VoxelFaceRole>((u.meta.face_role >> (2 * face)) & RState(0x3));
}

/** @brief Write the role of face `face` (0..5) into a packed voxel state. */
HOST_DEVICE_INLINE_QUALIFIER void setFaceRole(RState &state, int face, RSE_VoxelFaceRole role) {
    RS_VoxelStateUnion u{state};
    RState fr = u.voxel_state.face_role;
    fr &= ~(RState(0x3) << (2 * face));
    fr |= (RState(role) << (2 * face));
    u.voxel_state.face_role = fr;
    state = u.voxel_state_packed;
}

/** @brief Whether any face is exposed to air; the voxel can contact another body. */
HOST_DEVICE_INLINE_QUALIFIER bool isContactActive(RState state) {
    for (int f = 0; f < 6; f++) {
        if (getFaceRole(state, f) == RSE_VOXEL_FACE_AIR) return true;
    }
    return false;
}

/** @brief Whether any face is not interior, so a bonded neighbor falling away exposes it. */
HOST_DEVICE_INLINE_QUALIFIER bool isRayActive(RState state) {
    for (int f = 0; f < 6; f++) {
        if (getFaceRole(state, f) != RSE_VOXEL_FACE_INTERIOR) return true;
    }
    return false;
}

/**
 * @brief Corner indices of one voxel face, wound so the quad normal points out of the voxel.
 *
 * @details A corner index carries one bit per axis, set on the positive side. The face fixes its
 * own axis bit and the other two run over the four combinations. Swapping the two running axes on
 * the negative face flips the winding, which turns that quad's normal outward again.
 *
 * @param axis         0 for x, 1 for y and 2 for z.
 * @param is_positive  whether this is the face on the positive side of that axis.
 * @param corner_0_out first corner index in winding order.
 * @param corner_1_out second corner index.
 * @param corner_2_out third corner index, opposite the first.
 * @param corner_3_out fourth corner index.
 */
HOST_DEVICE_INLINE_QUALIFIER void getVoxelFaceCornerIndices(
    int axis,
    bool is_positive,
    int &corner_0_out,
    int &corner_1_out,
    int &corner_2_out,
    int &corner_3_out
) {
    int u_axis = (axis + 1) % 3;
    int v_axis = (axis + 2) % 3;
    if (!is_positive) {
        int swap = u_axis;
        u_axis = v_axis;
        v_axis = swap;
    }
    int base = is_positive ? (1 << axis) : 0;
    corner_0_out = base;
    corner_1_out = base | (1 << u_axis);
    corner_2_out = base | (1 << u_axis) | (1 << v_axis);
    corner_3_out = base | (1 << v_axis);
}

template <RSE_FloatType FloatComputeAccuracy>
class RS_LinkCompute;

template <RSE_FloatType FloatComputeAccuracy>
class RS_RigidBodyCompute;

/**
 * @brief Disable every voxel in a structure's voxel segment by clearing its is_enabled flag.
 * @details A disabled voxel is skipped by the physics, contact and render passes while the rest
 * of its state is left untouched, so it can be re-enabled later.
 *
 * @param k                    simulation kernel.
 * @param voxel_segment_offset handle offset added to the 1D index to reach this segment's voxels.
 */
class RSCF_DisableVoxel : public device::RS_ComputeFunction<RSCF_DisableVoxel> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RSize voxel_segment_offset) {
        auto &ctx = k->ctx;
        RHandle voxel = index1D() + voxel_segment_offset;
        RS_VoxelStateUnion state_union{V_G(voxel, state)};
        state_union.voxel_state.is_enabled = false;
        V_S(state, state_union.voxel_state_packed);
    }
};

/**
 * @brief Set or clear the effective immobility of every voxel in a structure's voxel segment.
 * @details Freezing forces is_static true so the voxel stops integrating and joins the static
 * subset while staying collidable and rendered; unfreezing restores is_static to the voxel's
 * permanent is_fixed, so a mixed-material structure keeps its originally-fixed voxels immobile.
 *
 * @param k                   simulation kernel.
 * @param voxel_segment_offset handle offset added to the 1D index to reach this segment's voxels.
 * @param freeze              true to freeze the voxels, false to restore them.
 */
class RSCF_SetVoxelStaticOverSegment
    : public device::RS_ComputeFunction<RSCF_SetVoxelStaticOverSegment> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Kernel *k, RSize voxel_segment_offset, bool freeze) {
        auto &ctx = k->ctx;
        RHandle voxel = index1D() + voxel_segment_offset;
        RS_VoxelStateUnion state_union{V_G(voxel, state)};
        state_union.voxel_state.is_static = freeze ? true : state_union.voxel_state.is_fixed;
        V_S(state, state_union.voxel_state_packed);
    }
};

/**
 * @brief Resolve a voxel's segment-relative dependency ids into absolute handles.
 * @details Adds the matching segment offset to the stored voxel-material, link, neighbor-voxel
 * and rigid-body ids so the voxel holds live handles after its segment is placed; null ids are
 * left unresolved.
 *
 * @param k                             simulation kernel.
 * @param voxel_segment_offset          handle offset to this segment's voxels.
 * @param link_segment_offset           handle offset to this segment's links.
 * @param voxel_material_segment_offset handle offset to this segment's voxel materials.
 * @param rigid_body_segment_offset     handle offset to this segment's rigid bodies.
 */
class RSCF_ResolveVoxelDependency : public device::RS_ComputeFunction<RSCF_ResolveVoxelDependency> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel *k,
        RSize voxel_segment_offset,
        RSize link_segment_offset,
        RSize voxel_material_segment_offset,
        RSize rigid_body_segment_offset
    ) {
        RHandle voxel = index1D() + voxel_segment_offset;
        auto &ctx = k->ctx;
        if (RIndex voxel_material_sid = V_G(voxel_material_sid);
            voxel_material_sid != RS_NULL_INDEX) {
            RHandle voxel_material = voxel_material_sid + voxel_material_segment_offset;
            V_S(voxel_material, voxel_material);
        }
        auto link_sid = V_G(link_sid);
        auto links = V_G(links);
        for (int i = 0; i < 6; i++) {
            RIndex l_sid = link_sid[i];
            if (l_sid != RS_NULL_INDEX) {
                RHandle link = l_sid + link_segment_offset;
                links[i] = link;
            }
        }

        auto neighbor_voxel_sid = V_G(neighbor_voxel_sid);
        auto neighbor_voxels = V_G(neighbor_voxels);
        for (int i = 0; i < 6; i++) {
            RIndex n_sid = neighbor_voxel_sid[i];
            if (n_sid != RS_NULL_INDEX) {
                RHandle neighbor_voxel = n_sid + voxel_segment_offset;
                neighbor_voxels[i] = neighbor_voxel;
            }
        }

        if (RIndex rigid_body_sid = V_G(rigid_body_sid); rigid_body_sid != RS_NULL_INDEX) {
            RHandle rigid_body = rigid_body_sid + rigid_body_segment_offset;
            V_S(rigid_body, rigid_body);
        }
    }
};

/**
 * @brief Classify a voxel into the dynamic and static subsets.
 * @details The dynamic subset is every mobile voxel so physics never reads static state. The
 * static subset is every immobile voxel, with no exposed gate: the structure voxel manager
 * already keeps only exposed voxels when it emits a fixed body, and a static voxel's exposure
 * never changes at runtime, so the gate would be redundant for fixed and frozen-rigid bodies.
 * A frozen soft body adds its interior voxels to the static subset, an accepted cost.
 *
 * @param k             simulation kernel.
 * @param voxel         handle of the voxel to classify.
 * @param flag_index    index of this voxel within the flag arrays.
 * @param dynamic_flags per-voxel output, 1 when mobile.
 * @param static_flags  per-voxel output, 1 when immobile.
 */
class RSCF_GetVoxelMobilityFlags : public device::RS_ComputeFunction<RSCF_GetVoxelMobilityFlags> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        const Kernel *k,
        RHandle voxel,
        RIndex flag_index,
        device::RS_Vector<RIndex> dynamic_flags,
        device::RS_Vector<RIndex> static_flags
    ) {
        if (k == nullptr || voxel == RS_NULL_HANDLE) return;
        auto &ctx = k->ctx;
        auto &dynamic_flag = dynamic_flags[flag_index];
        auto &static_flag = static_flags[flag_index];
        RS_VoxelStateUnion s_union{V_G(state)};
        if (!s_union.voxel_state.is_enabled) {
            dynamic_flag = 0;
            static_flag = 0;
            return;
        }
        bool is_static = s_union.voxel_state.is_static;
        if (!is_static && s_union.voxel_state.is_rigid) {
            RHandle rigid_body = V_G(rigid_body);
            if (rigid_body != RS_NULL_HANDLE) {
                RS_RigidBodyStateUnion rb_union{RB_G(rigid_body, state)};
                is_static = rb_union.rigid_body_state.is_static;
            }
        }
        dynamic_flag = is_static ? 0 : 1;
        static_flag = is_static ? 1 : 0;
    }
};

/**
 * @brief Classify a voxel into the contact BVH subset.
 * @details Launched over a mobility subset, so the voxel is already known static or dynamic and
 * only the exposed gate remains, since only an air face can touch another body. Run on every
 * contact BVH rebuild so a freshly exposed voxel, such as a soft bond tearing open, joins it.
 *
 * @param k            simulation kernel.
 * @param voxel        handle of the voxel to classify.
 * @param flag_index   index of this voxel within the flag array.
 * @param active_flags per-voxel output, 1 when the voxel can carry a contact.
 */
class RSCF_GetVoxelContactActiveFlags
    : public device::RS_ComputeFunction<RSCF_GetVoxelContactActiveFlags> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RHandle voxel, RIndex flag_index, device::RS_Vector<RIndex> active_flags) {
        if (k == nullptr || voxel == RS_NULL_HANDLE) return;
        auto &ctx = k->ctx;
        RS_VoxelStateUnion s_union{V_G(state)};
        bool is_exposed = isContactActive(s_union.voxel_state_packed);
        active_flags[flag_index] = (s_union.voxel_state.is_enabled && is_exposed) ? 1 : 0;
    }
};

/**
 * @brief Classify a voxel into the ray BVH subset.
 * @details Launched over a mobility subset, so the voxel is already known static or dynamic and
 * only the exposed gate remains, since only an air or transparent face can be hit. Run on every
 * ray BVH rebuild so a freshly exposed voxel, such as a soft bond tearing open, joins it.
 *
 * @param k            simulation kernel.
 * @param voxel        handle of the voxel to classify.
 * @param flag_index   index of this voxel within the flag array.
 * @param active_flags per-voxel output, 1 when the voxel can be hit.
 */
class RSCF_GetVoxelRayActiveFlags : public device::RS_ComputeFunction<RSCF_GetVoxelRayActiveFlags> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RHandle voxel, RIndex flag_index, device::RS_Vector<RIndex> active_flags) {
        if (k == nullptr || voxel == RS_NULL_HANDLE) return;
        auto &ctx = k->ctx;
        RS_VoxelStateUnion s_union{V_G(state)};
        bool is_exposed = isRayActive(s_union.voxel_state_packed);
        active_flags[flag_index] = (s_union.voxel_state.is_enabled && is_exposed) ? 1 : 0;
    }
};

/**
 * @brief Scatter the flagged voxel handles into a packed list using the prefix-sum.
 * @details Used for both the static and dynamic subsets. A voxel whose flag is set writes its
 * handle into out_handles at prefix_sums[flag_index] - 1.
 *
 * @param k           simulation kernel.
 * @param voxel       handle of this voxel.
 * @param flag_index  index of this voxel within the flag and prefix arrays.
 * @param flags       per-voxel 0/1 membership flags for the subset being compacted.
 * @param prefix_sums inclusive prefix sum of flags giving each flagged voxel its output slot.
 * @param out_handles packed output list of the flagged voxel handles.
 */
class RSCF_CompactVoxelHandles : public device::RS_ComputeFunction<RSCF_CompactVoxelHandles> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        const Kernel *k,
        RHandle voxel,
        RIndex flag_index,
        device::RS_Vector<RIndex> flags,
        device::RS_Vector<RIndex> prefix_sums,
        device::RS_Vector<RHandle> out_handles
    ) {
        if (k == nullptr || voxel == RS_NULL_HANDLE) return;
        if (flags[flag_index]) {
            RIndex out = prefix_sums[flag_index] - 1;
            out_handles[out] = voxel;
        }
    }
};

template <RSE_FloatType FloatComputeAccuracy>
class RS_VoxelCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Mat3C = RS_Mat3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using FI = RS_FloatIntrinsics<FloatC>;
    using LinkCompute = RS_LinkCompute<FloatComputeAccuracy>;
    using RigidBodyCompute = RS_RigidBodyCompute<FloatComputeAccuracy>;

    /*****************************************************************
     * Device side compute functions for updating
     *****************************************************************/
    /**
     * @brief Advance one soft-body voxel by a single integration step.
     * @details For an enabled, non-static voxel: refreshes expansion and size, gathers the
     * external force and moment from its bonds, contacts, gravity and damping, resolves floor
     * friction, then semi-implicitly integrates linear and angular momentum to update position
     * and orientation and refresh the Poisson strain and corner offsets. A rigid voxel only
     * accumulates its external force and moment, since its pose is driven by the owning rigid
     * body.
     *
     * @param k     simulation kernel.
     * @param voxel handle of the voxel to update.
     */
    class RSCF_UpdateVoxel : public device::RS_ComputeFunction<RSCF_UpdateVoxel> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_UpdateVoxel>;
        using BaseType::sharedMemAt1D;

        struct SharedMemLayout {
            RHandle links_0;
            RHandle links_1;
            RHandle links_2;
            RHandle links_3;
            RHandle links_4;
            RHandle links_5;

            template <RIndex Link>
            static constexpr RHandle SharedMemLayout::*get() {
                if constexpr (Link == 0)
                    return &SharedMemLayout::links_0;
                else if constexpr (Link == 1)
                    return &SharedMemLayout::links_1;
                else if constexpr (Link == 2)
                    return &SharedMemLayout::links_2;
                else if constexpr (Link == 3)
                    return &SharedMemLayout::links_3;
                else if constexpr (Link == 4)
                    return &SharedMemLayout::links_4;
                else if constexpr (Link == 5)
                    return &SharedMemLayout::links_5;
            }
        };

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle voxel) {
            if (k != nullptr && voxel != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                FloatC dt =
                    k->getDeviceTimeSinceLastOperation(RSE_KERNEL_UPDATE_OPERATION_SOFT_BODY);

                RS_VoxelStateUnion state_union{V_G(state)};
                RS_VoxelNeighborStateUnion neighbor_state_union{V_G(neighbor_state)};

                if (state_union.voxel_state.is_static || !state_union.voxel_state.is_enabled)
                    return;

                RHandle voxel_material = V_G(voxel_material);
                FloatC voxel_size = VM_G(voxel_size);

                auto links = V_G(links);
                device::loop<0, 6>([&](auto i_const) {
                    constexpr RIndex i = i_const.value;
                    sharedMemAt1D(SharedMemLayout::template get<i>()) = links[i];
                });

                Vec3C ext_force, ext_moment;
                if (state_union.voxel_state.is_rigid) {
                    // position and orientation updated by rigid body surface voxel

                    // no need to update nnn/ppp offset or radius
                    // since they are the same as initial values
                    _updateExternalForceAndMoment(
                        k,
                        voxel,
                        voxel_material,
                        state_union,
                        neighbor_state_union,
                        ext_force,
                        ext_moment
                    );
                } else {
                    _updateExpansionAndSize(k, voxel, voxel_material, dt, state_union, voxel_size);

                    _updateExternalForceAndMoment(
                        k,
                        voxel,
                        voxel_material,
                        state_union,
                        neighbor_state_union,
                        ext_force,
                        ext_moment
                    );

                    Vec3C position = V_G(position);
                    Vec3C linear_momentum = V_G(linear_momentum);
                    Vec3C total_force = ext_force;
                    Vec3C normal_vec = Vec3C::zero();
                    FloatC mass_inverse = VM_G(mass_inverse);

                    _computeTotalForceWithFloorFriction(
                        k,
                        voxel,
                        voxel_material,
                        voxel_size,
                        dt,
                        mass_inverse,
                        position,
                        state_union,
                        linear_momentum,
                        total_force,
                        normal_vec
                    );

#ifdef RS_DEBUG_ENABLE_LOGIC_VOXEL
                    CUDA_PRINTF_ASSERT(!isnan(total_force.x) && abs(total_force.x) < 1e6);
                    CUDA_PRINTF_ASSERT(!isnan(total_force.y) && abs(total_force.y) < 1e6);
                    CUDA_PRINTF_ASSERT(!isnan(total_force.z) && abs(total_force.z) < 1e6);
#endif

                    /// Integration
                    linear_momentum += total_force * dt;

                    // Note: previous_friction state is "current" after being updated by
                    // _computeTotalForceWithFloorFriction
                    bool is_floor_static_friction =
                        state_union.voxel_state.previous_floor_friction_state ==
                        RSE_VOXEL_STATIC_FRICTION;
                    // if we're in a state of static friction, zero out all linear momentum
                    // in plane
                    if (is_floor_static_friction) {
                        linear_momentum = linear_momentum.project(normal_vec);
                    }

                    // movement of the voxel this timestep
                    Vec3C translate(linear_momentum * (dt * mass_inverse));
                    // if we're in a state of static friction, zero out all translation in
                    // plane
                    if (is_floor_static_friction) {
                        translate = translate.project(normal_vec);
                    }

                    V_S(position, Vec3C(V_G(position)) + translate);

                    // Rotation
                    Vec3C total_moment = ext_moment;
                    Vec3C angular_momentum = V_G(angular_momentum);
                    angular_momentum += total_moment * dt;

                    // update the orientation
                    Vec3C rotation_vec = dt * VM_G(moment_inertia_inverse) * angular_momentum;

                    Quat3C rotation;
                    rotation.fromRotationVector(rotation_vec);
                    V_S(orientation, rotation * Quat3C(V_G(orientation)));

                    V_S(angular_momentum, angular_momentum);
                    V_S(linear_momentum, linear_momentum);

#ifdef RS_DEBUG_ENABLE_LOGIC_VOXEL
                    CUDA_PRINTF_ASSERT(Vec3C(V_G(position)).isValid());
                    CUDA_PRINTF_ASSERT(Quat3C(V_G(orientation)).isValid());
#endif
                    _updatePoissonsStrainAndCornerOffset(
                        k, voxel, voxel_material, neighbor_state_union, voxel_size
                    );
                }

                // Update states, note both branches can tear a bond, so both need to write.
                V_S(state, state_union.voxel_state_packed);
                V_S(neighbor_state, neighbor_state_union.neighbor_state_packed);
            }
        }

       private:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateExpansionAndSize(
            Kernel *k,
            RHandle voxel,
            RHandle voxel_material,
            FloatC dt,
            RS_VoxelStateUnion &state_union_in_out,
            FloatC &size_in_out
        ) {
            auto &ctx = k->ctx;

            // Update expansion
            if (state_union_in_out.voxel_state.has_expansion) {
                FloatC old_expansion = V_G(expansion);
                FloatC signal = ctx.structure_signals[V_G(structure_signal)]
                                    .expansion_signals.expansion[V_G(expansion_signal_sid)];
                FloatC target = state_union_in_out.voxel_state.expansion_control_mode ==
                                        RSE_VOXEL_EXPANSION_CONTROL_RATE
                                    ? old_expansion + dt * signal
                                    : signal;
                FloatC new_expansion = RS_BOUND(
                    target, VM_G(voxel_material, min_expansion), VM_G(voxel_material, max_expansion)
                );
                if (new_expansion != old_expansion) {
                    V_S(expansion, new_expansion);
                }
                size_in_out = baseSize(size_in_out, new_expansion);
            }
            V_S(size, size_in_out);
        }

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateExternalForceAndMoment(
            Kernel *k,
            RHandle voxel,
            RHandle voxel_material,
            RS_VoxelStateUnion &state_union_in_out,
            RS_VoxelNeighborStateUnion &neighbor_state_union_in_out,
            Vec3C &ext_force_out,
            Vec3C &ext_moment_out
        ) {
            auto &ctx = k->ctx;

            // forces from internal bonds
            ext_force_out = Vec3C::zero();
            ext_moment_out = Vec3C::zero();

            bool has_link = false;
            bool has_null_link = false;
            RBigIndex torn_link_mask = 0;

            if (state_union_in_out.voxel_state.has_link) {
                device::loop<0, 2>([&](auto dir_const) {
                    device::loop<0, 6, 2>([&](auto axis_const) {
                        static constexpr RIndex i = dir_const.value + axis_const.value;
                        if (!(neighbor_state_union_in_out.neighbor_state.is_neighbor_linked &
                              (RState(1) << i))) {
                            has_null_link = true;
                            return;
                        }
                        RHandle link = sharedMemAt1D(SharedMemLayout::template get<i>());
                        RS_LinkStateUnion link_state_union{L_G(link, state)};
                        if (link_state_union.link_state.is_failed) {
                            neighbor_state_union_in_out.neighbor_state.is_neighbor_linked &=
                                ~(RState(1) << i);
                            has_null_link = true;
                            torn_link_mask |= RBigIndex(1) << i;
                            // A torn bond exposes this face to air on both sides.
                            setFaceRole(
                                state_union_in_out.voxel_state_packed, i, RSE_VOXEL_FACE_AIR
                            );
                            return;
                        }
                        has_link = true;
                        // For link, it always points from voxel_neg to voxel_pos
                        // and its direction is always in the positive direction
                        if (isLinkDirectionNegative(static_cast<RSE_LinkDirection>(i))) {
                            ext_force_out += Vec3C(L_G(link, force_pos));
                            ext_moment_out += Vec3C(L_G(link, moment_pos));
                        } else {
                            ext_force_out += Vec3C(L_G(link, force_neg));
                            ext_moment_out += Vec3C(L_G(link, moment_neg));
                        }
                    });
                });

                state_union_in_out.voxel_state.has_link = has_link;
                state_union_in_out.voxel_state.is_surface =
                    (!state_union_in_out.voxel_state.is_rigid && has_null_link) ||
                    state_union_in_out.voxel_state.is_surface;

                // A torn face is no longer held by a link, so the broad phase must stop
                // filtering that neighbor out.
                if (torn_link_mask != 0) {
                    RS_VoxelContactFilterKeyUnion filter_key_union{V_G(contact_filter_key)};
                    filter_key_union.contact_filter_key.connectivity &= ~torn_link_mask;
                    V_S(contact_filter_key, filter_key_union.contact_filter_key_packed);
                }
            }

            // from local to global coordinates
            RQuat3rf orientation = V_G(orientation);
            ext_force_out = orientation.rotateVec3D(ext_force_out);

            // Contact force
            if (state_union_in_out.voxel_state.has_contact_voxel_0 &&
                (!state_union_in_out.voxel_state.is_rigid ||
                 !state_union_in_out.voxel_state.is_contact_voxel_0_rigid_or_static)) {
                ext_force_out += Vec3C(V_G(contact_force_0));
            }
            if (state_union_in_out.voxel_state.has_contact_voxel_1 &&
                (!state_union_in_out.voxel_state.is_rigid ||
                 !state_union_in_out.voxel_state.is_contact_voxel_1_rigid_or_static)) {
                ext_force_out += Vec3C(V_G(contact_force_1));
            }

            if (!state_union_in_out.voxel_state.is_rigid) {
                // Other forces, for rigid voxels we apply to rigid bodies instead
                // 1. Global damping f-cv
                ext_force_out -= linearVelocity(
                                     Vec3C(V_G(linear_momentum)), VM_G(voxel_material, mass_inverse)
                                 ) *
                                 VM_G(voxel_material, global_damping_translate_c);

                // 2. Gravity, according to f=mg
                ext_force_out.z += VM_G(voxel_material, gravity);
            }

            V_S(ext_force, ext_force_out);

            ext_moment_out = orientation.rotateVec3D(ext_moment_out);

            if (!state_union_in_out.voxel_state.is_rigid) {
                // Other moments, for rigid voxels we apply to rigid bodies instead
                // 1. Global damping
                ext_moment_out -=
                    angularVelocity(
                        Vec3C(V_G(angular_momentum)), VM_G(voxel_material, moment_inertia_inverse)
                    ) *
                    VM_G(voxel_material, global_damping_rotate_c);
            }
            V_S(ext_moment, ext_moment_out);

#ifdef RS_DEBUG_ENABLE_LOGIC_VOXEL
            CUDA_PRINTF_ASSERT(!isnan(ext_force.x) && abs(ext_force.x) < 1e6);
            CUDA_PRINTF_ASSERT(!isnan(ext_force.y) && abs(ext_force.y) < 1e6);
            CUDA_PRINTF_ASSERT(!isnan(ext_force.z) && abs(ext_force.z) < 1e6);
            CUDA_PRINTF_ASSERT(!isnan(ext_moment.x) && abs(ext_moment.x) < 1e6);
            CUDA_PRINTF_ASSERT(!isnan(ext_moment.y) && abs(ext_moment.y) < 1e6);
            CUDA_PRINTF_ASSERT(!isnan(ext_moment.z) && abs(ext_moment.z) < 1e6);
#endif
        }

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _computeTotalForceWithFloorFriction(
            Kernel *k,
            RHandle voxel,
            RHandle voxel_material,
            FloatC voxel_size,
            FloatC dt,
            FloatC mass_inverse,
            const Vec3C &position,
            RS_VoxelStateUnion &state_union_in_out,
            Vec3C &linear_momentum_in_out,
            Vec3C &total_force_in_out,
            Vec3C &normal_vec_out
        ) {
            auto &ctx = k->ctx;

            // Translation
            unsigned int current_friction_state = RSE_VOXEL_NO_CONTACT_FRICTION;

            if (hasFloor(k)) {
                FloatC floor_penetration = V_G(voxel, contact_depth_floor);
                normal_vec_out = isFloorFlat(k)
                                     ? Vec3C(FloatC(0), FloatC(0), FloatC(1))
                                     : V_G(voxel, contact_normal_floor).template toVec3D<FloatC>();
                if (floor_penetration >= 0) {
                    Vec3C velocity = linearVelocity(linear_momentum_in_out, mass_inverse);
                    FloatC normal_force_norm = VM_G(penetration_stiffness) * floor_penetration;

                    // Voxel-floor friction from the material pair tables.
                    RIndex mat_gid_a = VM_G(material_gid);
                    RIndex floor_gid = ctx.floor.material_gid;
                    FloatC u_static = FloatC(
                        ctx.material_friction_static_table[{(RSize)mat_gid_a, (RSize)floor_gid}]
                    );
                    FloatC u_dynamic = FloatC(
                        ctx.material_friction_dynamic_table[{(RSize)mat_gid_a, (RSize)floor_gid}]
                    );

                    // in the normal direction: k*x-C*v - spring and damping
                    FloatC separating_speed = velocity.dot(normal_vec_out);
                    total_force_in_out +=
                        RS_MAX(
                            normal_force_norm -
                                VM_G(contact_damping_translate_c) * separating_speed,
                            FloatC(0)
                        ) *
                        normal_vec_out;
                    Vec3C plane_force =
                        total_force_in_out - total_force_in_out.project(normal_vec_out);
                    Vec3C plane_velocity = velocity - velocity.project(normal_vec_out);
                    if (state_union_in_out.voxel_state.previous_floor_friction_state !=
                        RSE_VOXEL_DYNAMIC_FRICTION) {
                        // If voxel is previously in static friction mode (no lateral
                        // motion) check whether the voxel will continue to stay in static
                        // friction state If voxel is previously in no contact mode, check
                        // which state the voxel will become

                        // use squares to avoid a square root
                        FloatC surface_force_sq = plane_force.length2();
                        FloatC friction_force_sq = u_static * normal_force_norm;
                        friction_force_sq = friction_force_sq * friction_force_sq;

                        current_friction_state = surface_force_sq < friction_force_sq
                                                     ? RSE_VOXEL_STATIC_FRICTION
                                                     : RSE_VOXEL_DYNAMIC_FRICTION;
                    } else {
                        // Check whether voxel will continue to stay in dynamic friction
                        // state Work of dynamic friction
                        FloatC friction_force = u_dynamic * normal_force_norm;
                        FloatC work = dt * friction_force * plane_velocity.length();

                        // horizontal kinetic energy
                        FloatC hKe = FloatC(0.5) * mass_inverse *
                                     (linear_momentum_in_out -
                                      linear_momentum_in_out.project(normal_vec_out))
                                         .length2();
                        if (hKe - work <= 0) {
                            // this checks for a change of direction
                            // according to the work-energy principle
                            current_friction_state = RSE_VOXEL_STATIC_FRICTION;
                        } else {
                            current_friction_state = RSE_VOXEL_DYNAMIC_FRICTION;
                        }
                    }

                    // resolve friction state transition
                    if (current_friction_state == RSE_VOXEL_DYNAMIC_FRICTION) {
                        FloatC plane_velocity_norm = plane_velocity.length();
                        if (plane_velocity_norm > 0) {
                            Vec3C friction_dir = -plane_velocity / plane_velocity_norm;
                            total_force_in_out += u_dynamic * normal_force_norm * friction_dir;
                        }
                    } else {
                        // current_friction_state = STATIC_FRICTION, since NO_CONTACT will
                        // never enter the branch
                        // zero out forces on the projection plane since they are balanced
                        // with static force
                        total_force_in_out = total_force_in_out.project(normal_vec_out);
                    }
                } else
                    current_friction_state = RSE_VOXEL_NO_CONTACT_FRICTION;
            }
            state_union_in_out.voxel_state.previous_floor_friction_state = current_friction_state;
        }

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updatePoissonsStrainAndCornerOffset(
            Kernel *k,
            RHandle voxel,
            RHandle voxel_material,
            const RS_VoxelNeighborStateUnion &neighbor_state_union,
            FloatC voxel_size
        ) {
            // if no connections in the positive and negative directions of a particular
            // axis, strain is zero.
            // if one connection in positive or negative direction of a particular
            // axis, strain is that strain - ?? and force or constraint?
            // if connections in both the positive and negative directions of a particular
            // axis, strain is the average.

            auto &ctx = k->ctx;

            int bond_axis_num[3] = {0, 0, 0};
            FloatC strains[6];

            device::loop<0, 2>([&](auto dir_const) {
                device::loop<0, 6, 2>([&](auto axis_const) {
                    static constexpr RIndex i = dir_const.value + axis_const.value;
                    if (!(neighbor_state_union.neighbor_state.is_neighbor_linked &
                          (RState(1) << i))) {
                        strains[i] = FloatC(0);
                        return;
                    }
                    RHandle link = sharedMemAt1D(SharedMemLayout::template get<i>());
                    bond_axis_num[i >> 1] += 1;
                    strains[i] =
                        LinkCompute::axialStrain(L_G(link, strain), L_G(link, strain_ratio), false);
                });
            });

            FloatC half_base_size = FloatC(0.5) * voxel_size;

            Vec3C nnn(
                -(half_base_size * strains[1] + half_base_size),
                -(half_base_size * strains[3] + half_base_size),
                -(half_base_size * strains[5] + half_base_size)
            );
            Vec3C ppp(
                half_base_size * strains[0] + half_base_size,
                half_base_size * strains[2] + half_base_size,
                half_base_size * strains[4] + half_base_size
            );
            V_S(nnn_offset, nnn);
            V_S(ppp_offset, ppp);
            V_S(radius, (ppp - nnn).length() * FloatC(0.5));

            // Per axis strain sum
            Vec3C axis_strain_sum(
                strains[0] + strains[1], strains[2] + strains[3], strains[4] + strains[5]
            );
            bool tension[3] = {false, false, false};

            // average if both sides pulling,
            // or just one side and a fixed or forced voxel... (not implemented)
            tension[0] = bond_axis_num[0] == 2;
            axis_strain_sum.x *= tension[0] ? FloatC(0.5) : FloatC(1);
            tension[1] = bond_axis_num[1] == 2;
            axis_strain_sum.y *= tension[1] ? FloatC(0.5) : FloatC(1);
            tension[2] = bond_axis_num[2] == 2;
            axis_strain_sum.z *= tension[2] ? FloatC(0.5) : FloatC(1);
            if (!(tension[0] && tension[1] && tension[2])) {
                // if at least one isn't in tension
                FloatC add = 0;
                add += axis_strain_sum.x * FloatC(tension[0]);
                add += axis_strain_sum.y * FloatC(tension[1]);
                add += axis_strain_sum.z * FloatC(tension[2]);
                auto value = FloatC(0);
                // For a base close to 0, pow will return nan
                // See https://forums.developer.nvidia.com/t/powf-x-y-gives-nan/28094
                if (abs(FloatC(1.0) + add) > 0.1) {
                    value = FI::rs_pow(FloatC(1.0) + add, -VM_G(nu)) - FloatC(1.0);
                }
                axis_strain_sum.x = tension[0] ? axis_strain_sum.x : value;
                axis_strain_sum.y = tension[1] ? axis_strain_sum.y : value;
                axis_strain_sum.z = tension[2] ? axis_strain_sum.z : value;
            }
            V_S(poissons_strain, axis_strain_sum);
#ifdef RS_DEBUG_ENABLE_LOGIC_VOXEL
            CUDA_PRINTF_ASSERT(!isnan(axis_strain_sum.x) && axis_strain_sum.x < 1e6);
            CUDA_PRINTF_ASSERT(!isnan(axis_strain_sum.y) && axis_strain_sum.y < 1e6);
            CUDA_PRINTF_ASSERT(!isnan(axis_strain_sum.z) && axis_strain_sum.z < 1e6);
#endif
        }
    };

    /**
     * @brief Fill one voxel contact-BVH leaf on the build path.
     * @details For each active voxel, composes the leaf identifier of object metadata and handle
     * into obj_ids and writes the loose world-AABB into leaf_bounds.
     *
     * @param k              simulation kernel.
     * @param active_handles packed list of the contact active voxel handles, one per leaf.
     * @param obj_ids        per-leaf output of composed contact-BVH leaf identifiers.
     * @param leaf_bounds    per-leaf output of loose world-space AABBs.
     */
    class RSCF_FillVoxelContactBVHLeafData
        : public device::RS_ComputeFunction<RSCF_FillVoxelContactBVHLeafData> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_FillVoxelContactBVHLeafData>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            device::RS_Vector<RHandle> active_handles,
            device::RS_Vector<RS_VoxelContactBVHObjectIdentifier> obj_ids,
            device::RS_Vector<RBound3f> leaf_bounds
        ) {
            RIndex leaf_index = index1D();
            if (leaf_index >= active_handles.size()) return;
            RHandle voxel = active_handles[leaf_index];
            _composeVoxelContactBVHLeafIdentifier(k, voxel, obj_ids[leaf_index]);
            leaf_bounds[leaf_index] = _computeVoxelContactLooseAABB(k, voxel);
        }
    };

    /**
     * @brief Fill one voxel ray-BVH leaf on the build path.
     * @details For each active voxel, composes the leaf identifier of metadata, corners and face
     * normals into obj_ids and writes the loose world-AABB of those corners into leaf_bounds.
     *
     * @param k              simulation kernel.
     * @param active_handles packed list of the ray active voxel handles, one per leaf.
     * @param obj_ids        per-leaf output of composed ray-BVH leaf identifiers.
     * @param leaf_bounds    per-leaf output of loose world-space AABBs.
     */
    class RSCF_FillVoxelRayBVHLeafData
        : public device::RS_ComputeFunction<RSCF_FillVoxelRayBVHLeafData> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_FillVoxelRayBVHLeafData>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            device::RS_Vector<RHandle> active_handles,
            device::RS_Vector<RS_VoxelRayBVHObjectIdentifier> obj_ids,
            device::RS_Vector<RBound3f> leaf_bounds
        ) {
            RIndex leaf_index = index1D();
            if (leaf_index >= active_handles.size()) return;
            RHandle voxel = active_handles[leaf_index];
            _composeVoxelRayBVHLeafIdentifier(k, voxel, obj_ids[leaf_index]);
            leaf_bounds[leaf_index] = _computeVoxelRayLooseAABB(obj_ids[leaf_index]);
        }
    };

    /**
     * @brief Refit one voxel contact-BVH leaf in place on the refit path.
     * @details Reads the voxel handle out of obj_ids and refreshes both the leaf identifier and
     * the loose world-AABB without reordering, so the existing Morton order is preserved.
     *
     * @param k           simulation kernel.
     * @param obj_ids     per-leaf contact-BVH leaf identifiers, read for the handle and rewritten.
     * @param leaf_bounds per-leaf output of refreshed loose world-space AABBs.
     */
    class RSCF_RefitVoxelContactBVHLeafData
        : public device::RS_ComputeFunction<RSCF_RefitVoxelContactBVHLeafData> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_RefitVoxelContactBVHLeafData>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            device::RS_Vector<RS_VoxelContactBVHObjectIdentifier> obj_ids,
            device::RS_Vector<RBound3f> leaf_bounds
        ) {
            RIndex leaf_index = index1D();
            RHandle voxel = obj_ids[leaf_index].object_handle;
            _composeVoxelContactBVHLeafIdentifier(k, voxel, obj_ids[leaf_index]);
            leaf_bounds[leaf_index] = _computeVoxelContactLooseAABB(k, voxel);
        }
    };

    /**
     * @brief Refit one voxel ray-BVH leaf in place on the refit path.
     * @details Reads the voxel handle out of obj_ids and refreshes both the leaf identifier and
     * the loose world-AABB without reordering, so the existing Morton order is preserved.
     *
     * @param k           simulation kernel.
     * @param obj_ids     per-leaf ray-BVH leaf identifiers, read for the handle and rewritten.
     * @param leaf_bounds per-leaf output of refreshed loose world-space AABBs.
     */
    class RSCF_RefitVoxelRayBVHLeafData
        : public device::RS_ComputeFunction<RSCF_RefitVoxelRayBVHLeafData> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_RefitVoxelRayBVHLeafData>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            device::RS_Vector<RS_VoxelRayBVHObjectIdentifier> obj_ids,
            device::RS_Vector<RBound3f> leaf_bounds
        ) {
            RIndex leaf_index = index1D();
            RHandle voxel = obj_ids[leaf_index].object_handle;
            _composeVoxelRayBVHLeafIdentifier(k, voxel, obj_ids[leaf_index]);
            leaf_bounds[leaf_index] = _computeVoxelRayLooseAABB(obj_ids[leaf_index]);
        }
    };

    /**
     * @brief Find the up to two nearest contact-candidate voxels for one voxel.
     * @details Queries both the dynamic and static voxel BVHs for nearby candidates, applies the
     * same-body and connected-body filtering, and stores the two largest sphere-overlap contacts
     * into contact_voxel_0 and contact_voxel_1. Launched over the Morton-sorted
     * ctx.dynamic_voxel_contact_bvh.object_identifiers so adjacent warp threads process spatially
     * adjacent voxels and share BVH node cache lines.
     *
     * @param k                         simulation kernel.
     * @param sorted_object_identifiers Morton-sorted voxel-BVH leaf identifiers, one per thread.
     */
    class RSCF_FindContactVoxels : public device::RS_ComputeFunction<RSCF_FindContactVoxels> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_FindContactVoxels>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            device::RS_Vector<RS_VoxelContactBVHObjectIdentifier> sorted_object_identifiers
        ) {
            if (k == nullptr) return;
            RIndex idx = index1D();
            if (idx >= sorted_object_identifiers.size()) return;

            // Metadata-only early-out: skip non-collision / disabled leaves without
            // touching the voxel SoA. is_static never enters here because the dynamic
            // BVH build excluded fixed and parent-fixed-rigid-body voxels.
            RS_VoxelContactBVHLeafMetadataUnion meta{
                sorted_object_identifiers[idx].object_metadata
            };
            if (!meta.meta.is_enabled || !meta.meta.is_contact_active) return;

            RHandle voxel = sorted_object_identifiers[idx].object_handle;
            if (voxel == RS_NULL_HANDLE) return;

            auto &ctx = k->ctx;
            RS_VoxelStateUnion state_union{V_G(state)};

            Vec3C position = V_G(position);
            FloatC radius = V_G(radius);

            RIndex self_structure_gid = V_G(structure_gid);
            RS_VoxelContactFilterKeyUnion self_filter_key{V_G(contact_filter_key)};

            // A non-soft voxel is held by its segment rather than by a link, so its
            // segment mates stay filtered out even with no link between them.
            bool is_self_non_soft = state_union.voxel_state.is_rigid ||
                                    state_union.voxel_state.is_fixed;

            // Load the connected-body mask as two register words. Only constrained
            // structures own a mask; everything else leaves both words zero.
            bool is_structure_constrained = meta.meta.is_structure_constrained;
            RBigIndex self_mask_0 = 0, self_mask_1 = 0;
            if (is_structure_constrained &&
                self_filter_key.contact_filter_key.constrained_rigid_body_sid !=
                    RS_VOXEL_CONTACT_FILTER_KEY_NULL_CONSTRAINED_RIGID_BODY_SID) {
                self_mask_0 = RB_G(V_G(rigid_body), connected_body_mask_0);
                self_mask_1 = RB_G(V_G(rigid_body), connected_body_mask_1);
            }

            // Expand by our sphere radius: BVH leaf AABB encloses the other voxel.
            RBound3f query_bound(
                position.x - radius,
                position.y - radius,
                position.z - radius,
                position.x + radius,
                position.y + radius,
                position.z + radius
            );

            RHandle contact_voxel_0 = RS_NULL_HANDLE;
            FloatC overlap_0 = 0.0;
            RHandle contact_voxel_1 = RS_NULL_HANDLE;
            FloatC overlap_1 = 0.0;

            // Walk dynamic BVH first, then static. Both passes share the two
            // contact slots through contact_voxel_0/1 + overlap_0/1.
            if (ctx.dynamic_voxel_contact_bvh.object_identifiers.size() > 0) {
                auto bvh_accessor = ctx.dynamic_voxel_contact_bvh.root();
                _traverseBVHForContactCandidates(
                    k,
                    voxel,
                    position,
                    radius,
                    self_structure_gid,
                    self_filter_key,
                    is_self_non_soft,
                    is_structure_constrained,
                    self_mask_0,
                    self_mask_1,
                    query_bound,
                    bvh_accessor,
                    contact_voxel_0,
                    overlap_0,
                    contact_voxel_1,
                    overlap_1
                );
            }
            if (ctx.static_voxel_contact_bvh.object_identifiers.size() > 0) {
                auto bvh_accessor = ctx.static_voxel_contact_bvh.root();
                _traverseBVHForContactCandidates(
                    k,
                    voxel,
                    position,
                    radius,
                    self_structure_gid,
                    self_filter_key,
                    is_self_non_soft,
                    is_structure_constrained,
                    self_mask_0,
                    self_mask_1,
                    query_bound,
                    bvh_accessor,
                    contact_voxel_0,
                    overlap_0,
                    contact_voxel_1,
                    overlap_1
                );
            }

            RHandle prev_contact_voxel_0 = V_G(contact_voxel_0);
            if (contact_voxel_0 != RS_NULL_HANDLE) {
                state_union.voxel_state.has_contact_voxel_0 = true;
                V_S(contact_voxel_0, contact_voxel_0);
            } else
                state_union.voxel_state.has_contact_voxel_0 = false;
            if (contact_voxel_0 != prev_contact_voxel_0) {
                V_S(contact_lambda_n_0, FloatC(0));
                V_S(contact_lambda_t_0, Vec3C::zero());
            }

            RHandle prev_contact_voxel_1 = V_G(contact_voxel_1);
            if (contact_voxel_1 != RS_NULL_HANDLE) {
                state_union.voxel_state.has_contact_voxel_1 = true;
                V_S(contact_voxel_1, contact_voxel_1);
            } else
                state_union.voxel_state.has_contact_voxel_1 = false;
            if (contact_voxel_1 != prev_contact_voxel_1) {
                V_S(contact_lambda_n_1, FloatC(0));
                V_S(contact_lambda_t_1, Vec3C::zero());
            }

            V_S(state, state_union.voxel_state_packed);
        }

       private:
        // Signed difference of two mod-16 residuals, result in the range -8..7.
        // The bitwise mask recovers the low nibble for a negative difference.
        HOST_DEVICE_INLINE_QUALIFIER static int _mod16Difference(int a, int b) {
            int d = (a - b) & 0xF;
            return d >= 8 ? d - 16 : d;
        }

        // Direction from this voxel towards a face adjacent neighbor, or RSE_NO_DIRECTION
        // for an edge or corner adjacent one. The differences run from the neighbor to
        // this voxel, so a positive difference points the negative way.
        HOST_DEVICE_INLINE_QUALIFIER static RSE_LinkDirection
        _getNeighborLinkDirection(int dx, int dy, int dz) {
            if (dy == 0 && dz == 0) {
                if (dx == 1) return RSE_X_NEG;
                if (dx == -1) return RSE_X_POS;
            } else if (dx == 0 && dz == 0) {
                if (dy == 1) return RSE_Y_NEG;
                if (dy == -1) return RSE_Y_POS;
            } else if (dx == 0 && dy == 0) {
                if (dz == 1) return RSE_Z_NEG;
                if (dz == -1) return RSE_Z_POS;
            }
            return RSE_NO_DIRECTION;
        }

        // Walk one voxel BVH starting from the root currently held by
        // bvh_accessor_in_out and apply same-body / connected-body filtering
        // to each candidate leaf, keeping the two largest sphere-overlap
        // contacts in (contact_voxel_0_in_out, contact_voxel_1_in_out).
        // Reused for the dynamic and static BVH passes; only the root differs.
        template <typename Kernel, typename BVHAccessor>
        HOST_DEVICE_INLINE_QUALIFIER static void _traverseBVHForContactCandidates(
            Kernel *k,
            RHandle voxel,
            const Vec3C &position,
            FloatC radius,
            RIndex self_structure_gid,
            const RS_VoxelContactFilterKeyUnion &self_filter_key,
            bool is_self_non_soft,
            bool is_structure_constrained,
            RBigIndex self_mask_0,
            RBigIndex self_mask_1,
            const RBound3f &query_bound,
            BVHAccessor &bvh_accessor_in_out,
            RHandle &contact_voxel_0_in_out,
            FloatC &overlap_0_in_out,
            RHandle &contact_voxel_1_in_out,
            FloatC &overlap_1_in_out
        ) {
            auto &ctx = k->ctx;

            RIndex stack[RS_RUNTIME_BVH_STACK_SIZE];
            RSize stack_top = 0;
            RIndex root_node = bvh_accessor_in_out.current();
            if (root_node != RS_NULL_INDEX) stack[stack_top++] = root_node;

            while (stack_top > 0) {
                RIndex node = stack[--stack_top];
                bvh_accessor_in_out.moveTo(node);

                if (bvh_accessor_in_out.isLeaf()) {
                    auto *obj = bvh_accessor_in_out.object();
                    if (obj == nullptr) continue;
                    RHandle other_voxel = obj->object_handle;
                    if (other_voxel == voxel || other_voxel == RS_NULL_HANDLE) continue;

                    // Same-structure check gates both filters; different-structure
                    // candidates skip the filters entirely and fall to overlap.
                    if (self_structure_gid == obj->structure_gid) {
                        RS_VoxelContactFilterKeyUnion other_filter_key{obj->contact_filter_key};

                        // Same body + Chebyshev-1 in voxel grid → face, edge or corner
                        // adjacent neighbor. Indices are stored as mod-16 residuals; the
                        // signed mod-16 difference recovers the true gap because sphere
                        // overlap bounds candidates to one grid unit per axis.
                        // A1: face adjacent, linked -> skip, the link carries the force
                        // A2: face adjacent, no link, one non-soft segment -> skip, the
                        //     segment holds both and a non-soft pair never gets a link
                        // A3: face adjacent, no link, anything else -> contact, the bond
                        //     was authored away or torn at runtime
                        // B1: edge or corner adjacent, same segment -> skip, degenerate touch
                        // B2: edge or corner adjacent, other segment -> contact, moves apart
                        if (self_filter_key.contact_filter_key.body_sid ==
                            other_filter_key.contact_filter_key.body_sid) {
                            int dx = _mod16Difference(
                                int(self_filter_key.contact_filter_key.index_x_res),
                                int(other_filter_key.contact_filter_key.index_x_res)
                            );
                            int dy = _mod16Difference(
                                int(self_filter_key.contact_filter_key.index_y_res),
                                int(other_filter_key.contact_filter_key.index_y_res)
                            );
                            int dz = _mod16Difference(
                                int(self_filter_key.contact_filter_key.index_z_res),
                                int(other_filter_key.contact_filter_key.index_z_res)
                            );
                            if (RS_ABS(dx) <= 1 && RS_ABS(dy) <= 1 && RS_ABS(dz) <= 1) {
                                RSE_LinkDirection direction = _getNeighborLinkDirection(dx, dy, dz);
                                bool is_same_segment =
                                    self_filter_key.contact_filter_key.segment_bid ==
                                    other_filter_key.contact_filter_key.segment_bid;
                                if (direction != RSE_NO_DIRECTION) {
                                    // A1 and A2 are handled here. Links are symmetric so
                                    // reading our own bit is enough, and a non-soft pair is
                                    // held by its segment instead of by a link.
                                    RS_VoxelContactBVHLeafMetadataUnion other_meta{
                                        obj->object_metadata
                                    };
                                    bool is_other_non_soft = other_meta.meta.is_rigid ||
                                                             other_meta.meta.is_static;
                                    if ((self_filter_key.contact_filter_key.connectivity >>
                                         direction) &
                                        RBigIndex(1))
                                        continue;
                                    if (is_same_segment && is_self_non_soft && is_other_non_soft)
                                        continue;
                                } else if (is_same_segment) {
                                    // B1 is handled here
                                    continue;
                                }
                                // A3 and B2 are handled here, they fall through to the
                                // sphere overlap test below
                            }
                        }

                        // Constraint-connected rigid-body pair, skip.
                        RBigIndex other_crb_sid =
                            other_filter_key.contact_filter_key.constrained_rigid_body_sid;
                        if (is_structure_constrained &&
                            other_crb_sid !=
                                RS_VOXEL_CONTACT_FILTER_KEY_NULL_CONSTRAINED_RIGID_BODY_SID) {
                            RBigIndex word = (other_crb_sid < 64) ? self_mask_0 : self_mask_1;
                            if ((word >> (other_crb_sid & 63)) & RBigIndex(1)) continue;
                        }
                    }

                    FloatC other_radius = FloatC(obj->radius);
                    FloatC distance = (position - Vec3C(obj->position)).length();
                    FloatC overlap = radius + other_radius - distance;

                    // Keep top-2 by overlap (slot 0 = largest, slot 1 = second).
                    if (overlap > overlap_0_in_out) {
                        overlap_1_in_out = overlap_0_in_out;
                        contact_voxel_1_in_out = contact_voxel_0_in_out;
                        overlap_0_in_out = overlap;
                        contact_voxel_0_in_out = other_voxel;
                    } else if (overlap > overlap_1_in_out) {
                        overlap_1_in_out = overlap;
                        contact_voxel_1_in_out = other_voxel;
                    }
                } else {
                    auto internal_node = bvh_accessor_in_out.internalNode();

                    if (internal_node.left != RS_NULL_INDEX &&
                        stack_top < RS_RUNTIME_BVH_STACK_SIZE &&
                        internal_node.left_bound.isIntersecting(query_bound)) {
                        stack[stack_top++] = internal_node.left;
                    }
                    if (internal_node.right != RS_NULL_INDEX &&
                        stack_top < RS_RUNTIME_BVH_STACK_SIZE &&
                        internal_node.right_bound.isIntersecting(query_bound)) {
                        stack[stack_top++] = internal_node.right;
                    }
                }
            }
        }
    };

    /**
     * @brief Resolve a voxel's contacts and write the resulting contact forces.
     * @details Flags floor contact for air-exposed rigid voxels, then for each of the up to two
     * broad-phase contact partners builds an oriented bounding box, runs a separating-axis test
     * and, on penetration, computes the penetration depth, contact normal and the normal plus
     * friction force stored back on the voxel. A voxel with no contact partner clears its stale
     * floor accumulator and returns early.
     *
     * Two limits are deliberate. A pair is solved from whichever side recorded it, so a voxel
     * whose partner dropped it takes the force alone and the partner takes none. And a voxel
     * pair carries dynamic friction only, since sticking is a constraint that has to be
     * solved, which a force based contact neither needs nor can afford. Both are accepted for
     * the fidelity they cost; the requirement here is only that a resting pair does not creep.
     *
     * @param k     simulation kernel.
     * @param voxel handle of the voxel to update.
     */
    class RSCF_UpdateVoxelContact : public device::RS_ComputeFunction<RSCF_UpdateVoxelContact> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle voxel) {
            if (k != nullptr && voxel != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                RS_VoxelStateUnion state_union{V_G(state)};
                if (!state_union.voxel_state.is_enabled) return;

                // Measure the floor once for every moving voxel and keep the depth, so the
                // paths that need it later read one number instead of the whole box again.
                // Flag the rigid air-exposed ones for the manifold, and clear the stale floor
                // accumulator.
                bool floor_contact = false;
                if (!state_union.voxel_state.is_static) {
                    Vec3C voxel_pos = V_G(position);
                    FloatC floor_height;
                    Vec3C floor_normal(FloatC(0), FloatC(0), FloatC(1));
                    FloatC floor_depth = FloatC(-1);
                    if (floorHeightAndNormal(k, voxel_pos, floor_height, floor_normal)) {
                        // A corner sits at most its own size from the center, so a voxel
                        // riding higher than that is clear and never needs the corner walk.
                        FloatC reach = VM_G(V_G(voxel_material), voxel_size);
                        floor_depth = reach - (voxel_pos.z - floor_height) * floor_normal.z;
                        if (floor_depth > FloatC(0))
                            floor_depth = floorPenetration(
                                V_G(orientation),
                                V_G(nnn_offset),
                                V_G(ppp_offset),
                                voxel_pos,
                                floor_normal,
                                floor_height
                            );
                    }
                    V_S(contact_depth_floor, floor_depth);
                    if (!isFloorFlat(k)) V_S(contact_normal_floor, floor_normal);
                    floor_contact = floor_depth > FloatC(0) && state_union.voxel_state.is_rigid &&
                                    isContactActive(state_union.voxel_state_packed);
                    if (!floor_contact) {
                        V_S(contact_lambda_n_floor, FloatC(0));
                        V_S(contact_lambda_t_floor, Vec3C::zero());
                    }
                }
                state_union.voxel_state.has_contact_floor = floor_contact ? 1 : 0;

                // The voxel must has at least one contact voxel
                if (!state_union.voxel_state.has_contact_voxel_0 &&
                    !state_union.voxel_state.has_contact_voxel_1) {
                    V_S(state, state_union.voxel_state_packed);
                    return;
                }

                // Build OBB once for the current voxel
                OBB obb_a = _buildOBB(k, voxel);

                // Friction may only slow a slide within this step, so it needs the same step
                // the integration will take.
                FloatC dt =
                    k->getDeviceTimeSinceLastOperation(RSE_KERNEL_UPDATE_OPERATION_SOFT_BODY);

                // Process contact #0
                if (state_union.voxel_state.has_contact_voxel_0)
                    _updateOneContact<0>(k, voxel, obb_a, V_G(contact_voxel_0), dt, state_union);

                // Process contact #1
                if (state_union.voxel_state.has_contact_voxel_1)
                    _updateOneContact<1>(k, voxel, obb_a, V_G(contact_voxel_1), dt, state_union);

                V_S(state, state_union.voxel_state_packed);
            }
        }

       private:
        /** @brief OBB struct for SAT */
        struct OBB {
            Vec3C center;    // world-space center
            Vec3C axis[3];   // orthonormal axes
            Vec3C half_ext;  // half extents along each axis
        };

        /** @brief Build an OBB from the voxel data */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static OBB _buildOBB(Kernel *k, RHandle v) {
            auto &ctx = k->ctx;
            Vec3C nnn = V_G(v, nnn_offset);
            Vec3C ppp = V_G(v, ppp_offset);
            Vec3C pos = V_G(v, position);
            Quat3C ori = V_G(v, orientation);

            // Compute orthonormal axes from quaternion
            Vec3C axis_x = ori.toOrthoAxisX();
            Vec3C axis_y = ori.toOrthoAxisY();
            Vec3C axis_z = ori.toOrthoAxisZ();

            // Local center (midpoint in local coords)
            Vec3C local_center = (nnn + ppp) * FloatC(0.5);
            Vec3C world_center = pos + ori.rotateVec3D(local_center);

            OBB box;
            box.center = world_center;
            box.axis[0] = axis_x;
            box.axis[1] = axis_y;
            box.axis[2] = axis_z;
            box.half_ext = (ppp - nnn) * FloatC(0.5);
            return box;
        }

        /** @brief Handle one contact partner */
        template <RIndex ContactIndex, typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateOneContact(
            Kernel *k,
            RHandle voxel,
            const OBB &obb_a,     // OBB for this voxel
            RHandle other_voxel,  // in/out handle for the contact partner
            FloatC dt,
            RS_VoxelStateUnion &state_union_in_out
        ) {
            auto &ctx = k->ctx;

            // Build OBB for the other voxel
            OBB obb_b = _buildOBB(k, other_voxel);

            // Perform SAT check, which orients the normal by the faces the pair can separate
            // along, so it already points from the other voxel toward this one.
            Vec3C contact_normal = Vec3C::zero();
            FloatC penetration_depth = FloatC(0);
            bool intersects = _SAT(
                obb_a,
                obb_b,
                state_union_in_out.voxel_state_packed,
                V_G(other_voxel, state),
                contact_normal,
                penetration_depth
            );

            if (!intersects) {
                // No contact => clear
                constexpr RHandle null_handle = RS_NULL_HANDLE;

                if constexpr (ContactIndex == 0) {
                    state_union_in_out.voxel_state.has_contact_voxel_0 = false;
                    V_S(contact_voxel_0, null_handle);
                    V_S(contact_position_offset_0, Vec3C::zero());
                    V_S(contact_normal_0, Vec3C::zero());
                    V_S(contact_depth_0, FloatC(0));
                    V_S(contact_lambda_n_0, FloatC(0));
                    V_S(contact_lambda_t_0, Vec3C::zero());
                } else {
                    state_union_in_out.voxel_state.has_contact_voxel_1 = false;
                    V_S(contact_voxel_1, null_handle);
                    V_S(contact_position_offset_1, Vec3C::zero());
                    V_S(contact_normal_1, Vec3C::zero());
                    V_S(contact_depth_1, FloatC(0));
                    V_S(contact_lambda_n_1, FloatC(0));
                    V_S(contact_lambda_t_1, Vec3C::zero());
                }
            } else {
                // Approx contact position (note the normal points from b to a)
                FloatC ra = _projectionRange(obb_a, contact_normal);
                Vec3C contact_pos =
                    obb_a.center - contact_normal * (ra - FloatC(0.5) * penetration_depth);

                Vec3C contact_force = Vec3C::zero();
                _updateContactForce<ContactIndex>(
                    k,
                    voxel,
                    other_voxel,
                    penetration_depth,
                    contact_normal,
                    contact_pos,
                    dt,
                    contact_force,
                    state_union_in_out
                );

                Vec3C contact_position_offset = contact_pos - Vec3C(V_G(position));
                if constexpr (ContactIndex == 0) {
                    state_union_in_out.voxel_state.has_contact_voxel_0 = true;
                    V_S(contact_position_offset_0, contact_position_offset);
                    V_S(contact_normal_0, contact_normal);
                    V_S(contact_force_0, contact_force);
                    V_S(contact_depth_0, penetration_depth);
                } else {
                    state_union_in_out.voxel_state.has_contact_voxel_1 = true;
                    V_S(contact_position_offset_1, contact_position_offset);
                    V_S(contact_normal_1, contact_normal);
                    V_S(contact_force_1, contact_force);
                    V_S(contact_depth_1, penetration_depth);
                }
            }
        }

        template <RIndex ContactIndex, typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateContactForce(
            Kernel *k,
            RHandle voxel,
            RHandle other_voxel,
            FloatC contact_depth,
            const Vec3C &contact_normal,
            const Vec3C &contact_position,
            FloatC dt,
            Vec3C &contact_force_out,
            RS_VoxelStateUnion &state_union_in_out
        ) {
            auto &ctx = k->ctx;

            RS_VoxelStateUnion contact_voxel_state_union{V_G(other_voxel, state)};
            bool is_self_rigid_or_static = state_union_in_out.voxel_state.is_rigid ||
                                           state_union_in_out.voxel_state.is_static;
            bool is_other_rigid_or_static = contact_voxel_state_union.voxel_state.is_rigid ||
                                            contact_voxel_state_union.voxel_state.is_static;
            if constexpr (ContactIndex == 0) {
                state_union_in_out.voxel_state.is_contact_voxel_0_rigid_or_static =
                    is_other_rigid_or_static;
            } else {
                state_union_in_out.voxel_state.is_contact_voxel_1_rigid_or_static =
                    is_other_rigid_or_static;
            }

            if (state_union_in_out.voxel_state.is_rigid && is_other_rigid_or_static) {
                // Resolved by the rigid body solver: rigid against another rigid body, or
                // against an immovable anchor (fixed or frozen) through its anchor path.
                contact_force_out = Vec3C::zero();
            } else {
                // Note: The following pipeline would not be energy conservative if the voxel
                // size differ dramatically and the large side voxels likely will lose contact
                // info if it is colliding with many smaller soft voxels. For efficiency, we
                // choose to accept low-fidelity and just let smaller soft-voxels being pushed
                // away. Rigid/Fixed-Rigid/Fixed relationship does not have this problem becase
                // We use the manifold reduction pipeline.
                RHandle voxel_material = V_G(voxel, voxel_material);
                RHandle other_voxel_material = V_G(other_voxel, voxel_material);

                // 1. Pick the stiffness and damping. A rigid or static side is immovable and
                // cannot deform, so a free side governs; with both sides free the softer one
                // governs. Both are identical from either voxel, so the contact force is
                // symmetric across the pair.
                FloatC self_stiffness = VM_G(voxel_material, penetration_stiffness);
                FloatC other_stiffness = VM_G(other_voxel_material, penetration_stiffness);
                FloatC self_damping = VM_G(voxel_material, contact_damping_translate_c);
                FloatC other_damping = VM_G(other_voxel_material, contact_damping_translate_c);
                FloatC stiffness, damping;
                if (!is_self_rigid_or_static && !is_other_rigid_or_static) {
                    stiffness = self_stiffness < other_stiffness ? self_stiffness : other_stiffness;
                    damping = self_damping < other_damping ? self_damping : other_damping;
                } else if (!is_self_rigid_or_static) {
                    stiffness = self_stiffness;
                    damping = self_damping;
                } else if (!is_other_rigid_or_static) {
                    stiffness = other_stiffness;
                    damping = other_damping;
                } else {
                    stiffness = FloatC(0);
                    damping = FloatC(0);
                }

                Vec3C velocity = _readVoxelVelocity(k, voxel, state_union_in_out, contact_position);
                Vec3C other_velocity =
                    _readVoxelVelocity(k, other_voxel, contact_voxel_state_union, contact_position);
                Vec3C rel_velocity = velocity - other_velocity;

                // 2. Solve the normal force. The normal points from the other voxel to this
                // one, so a positive speed along it means the pair is coming apart, and
                // damping along it is what lets a resting contact lose energy instead of
                // ringing. A contact pushes and never pulls, so the damping may cancel the
                // bounce but never reverse it into adhesion.
                FloatC separating_speed = rel_velocity.dot(contact_normal);
                FloatC spring_force = stiffness * contact_depth;
                contact_force_out =
                    RS_MAX(spring_force - damping * separating_speed, FloatC(0)) * contact_normal;

                // 3. Solve the friction force from the spring alone. The damper resists
                // closing without pressing the surfaces together, so it must not widen the
                // cone. A voxel pair carries dynamic friction only, since choosing between the
                // two coefficients needs a solve this force based contact cannot afford.
                //
                // The tangent lambda accumulates across steps, one sweep per step, warm
                // started from the last. A resting pair grows it until it cancels whatever
                // drives the slide and then holds there, which is what a per step force cannot
                // do: its rest point is a fixed creep speed, this one's rest point is zero.
                FloatC self_mass_inverse =
                    is_self_rigid_or_static ? FloatC(0) : VM_G(voxel_material, mass_inverse);
                FloatC other_mass_inverse =
                    is_other_rigid_or_static ? FloatC(0) : VM_G(other_voxel_material, mass_inverse);
                FloatC mass_inverse_sum = self_mass_inverse + other_mass_inverse;

                // The normal moves between steps, so the carried lambda is reprojected onto
                // the plane it now acts in.
                Vec3C lambda_t = ContactIndex == 0 ? Vec3C(V_G(voxel, contact_lambda_t_0))
                                                   : Vec3C(V_G(voxel, contact_lambda_t_1));
                lambda_t = lambda_t - lambda_t.project(contact_normal);

                if (mass_inverse_sum > FloatC(0) && dt > FloatC(0)) {
                    Vec3C tangent_vel = rel_velocity - rel_velocity.project(contact_normal);

                    // Only the increment needs a slide to work on. The lambda already earned
                    // is applied either way, since dropping it is what lets a stopped pair
                    // start drifting again.
                    lambda_t -= tangent_vel / mass_inverse_sum;

                    // Only dynamic friction, since static requires solving and in a force
                    // based scheme that is both unnecessary and inefficient.
                    RIndex mat_gid_a = VM_G(voxel_material, material_gid);
                    RIndex mat_gid_b = VM_G(other_voxel_material, material_gid);
                    FloatC u = FloatC(
                        ctx.material_friction_dynamic_table[{(RSize)mat_gid_a, (RSize)mat_gid_b}]
                    );
                    // Bounding the depth to protect from degenerate voxel penetrating too deep
                    // case.
                    FloatC cone_depth = RS_MIN(
                        contact_depth,
                        FloatC(k->softbody_friction_max_penetration_ratio) *
                            RS_MIN(
                                VM_G(voxel_material, voxel_size),
                                VM_G(other_voxel_material, voxel_size)
                            )
                    );
                    FloatC cone = u * stiffness * cone_depth * dt;
                    FloatC lambda_length = lambda_t.length();
                    if (lambda_length > cone) lambda_t = lambda_t * (cone / lambda_length);

                    contact_force_out += lambda_t / dt;
                }

                if constexpr (ContactIndex == 0)
                    V_S(voxel, contact_lambda_t_0, lambda_t);
                else
                    V_S(voxel, contact_lambda_t_1, lambda_t);
            }
        }

        /** @brief Read a voxel's velocity at a point in world coordinates */
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static Vec3C _readVoxelVelocity(
            Kernel *k,
            RHandle voxel,
            const RS_VoxelStateUnion &state,
            const Vec3C &point
        ) {
            auto &ctx = k->ctx;

            // A rigid voxel is driven by its body and never updates its own momentum, so its
            // velocity comes from the body and is carried out by the body spin.
            if (!state.voxel_state.is_rigid) {
                return linearVelocity(
                    Vec3C(V_G(voxel, linear_momentum)),
                    VM_G(V_G(voxel, voxel_material), mass_inverse)
                );
            }
            RHandle rigid_body = V_G(voxel, rigid_body);
            if (rigid_body == RS_NULL_HANDLE || state.voxel_state.is_static) return Vec3C::zero();
            Vec3C body_velocity =
                Vec3C(RB_G(rigid_body, linear_momentum)) * FloatC(RB_G(rigid_body, mass_inverse));
            Vec3C body_spin = Mat3C(RB_G(rigid_body, moment_inertia_inverse)) *
                              Vec3C(RB_G(rigid_body, angular_momentum));
            return body_velocity + body_spin.cross(point - Vec3C(RB_G(rigid_body, com)));
        }

        /**
         * @brief Full SAT for OBB vs OBB
         * @returns true if in contact, and also returns min-overlap axis + depth
         */
        HOST_DEVICE_INLINE_QUALIFIER static bool _SAT(
            const OBB &a,
            const OBB &b,
            RState state_a,
            RState state_b,
            Vec3C &contact_normal_out,
            FloatC &penetration_depth_out
        ) {
            FloatC min_overlap = RS_NumTraits<FloatC>::highest();
            Vec3C best_axis = Vec3C::zero();
            bool has_axis = false;

            Vec3C center_diff = b.center - a.center;

            // 1) Check the 3 axes of A
            for (int i = 0; i < 3; i++) {
                FloatC overlap = _overlapOnAxis(a, b, a.axis[i], center_diff);
                if (overlap < FloatC(0)) return false;
                Vec3C oriented;
                if (!_orientContactAxis(state_a, i, a.axis[i], center_diff, oriented)) continue;
                if (!has_axis || overlap < min_overlap) {
                    min_overlap = overlap;
                    best_axis = oriented;
                    has_axis = true;
                }
            }

            // 2) Check the 3 axes of B
            for (int i = 0; i < 3; i++) {
                FloatC overlap = _overlapOnAxis(a, b, b.axis[i], center_diff);
                if (overlap < FloatC(0)) return false;
                // The helper points from the partner toward the voxel owning the axis, which
                // for b is the way back, so it is flipped to point from b toward a.
                Vec3C oriented;
                if (!_orientContactAxis(state_b, i, b.axis[i], -center_diff, oriented)) continue;
                if (!has_axis || overlap < min_overlap) {
                    min_overlap = overlap;
                    best_axis = -oriented;
                    has_axis = true;
                }
            }

            // 3) Check cross products. An edge axis belongs to no face, so it keeps the
            // centre-difference orientation.
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    Vec3C cross_axis = a.axis[i].cross(b.axis[j]);
                    FloatC len = cross_axis.length();
                    if (len < 1e-12f) continue;  // skip degenerate

                    cross_axis = cross_axis / len;
                    FloatC overlap = _overlapOnAxis(a, b, cross_axis, center_diff);
                    if (overlap < FloatC(0)) return false;
                    if (!has_axis || overlap < min_overlap) {
                        min_overlap = overlap;
                        best_axis = center_diff.dot(cross_axis) < FloatC(0) ? cross_axis
                                                                            : -cross_axis;
                        has_axis = true;
                    }
                }
            }

            // Every face the pair could separate along is covered by a neighbor, so the two
            // cannot push each other apart and there is no contact to solve.
            if (!has_axis) return false;

            contact_normal_out = best_axis;
            penetration_depth_out = min_overlap;
            return true;
        }

        /**
         * @brief Point one face axis out through the face the voxel can separate along.
         * @details A face with a neighbor cannot carry a contact, since pushing along it would
         * drive the voxel into its own body. With one face of the axis open the direction is
         * forced, which is what keeps the normal steady once the two centres are close enough
         * that their difference no longer decides a sign. With both open the centre difference
         * decides, and with neither the axis is unusable.
         *
         * @param state      packed state of the voxel owning the axis.
         * @param axis_index axis of that voxel, 0 for x, 1 for y and 2 for z.
         * @param axis       that axis in world coordinates, unit.
         * @param to_partner centre of the partner less centre of this voxel.
         * @param normal_out oriented normal, pointing from the partner toward this voxel.
         * @returns whether the axis can carry a contact at all.
         */
        HOST_DEVICE_INLINE_QUALIFIER static bool _orientContactAxis(
            RState state,
            int axis_index,
            const Vec3C &axis,
            const Vec3C &to_partner,
            Vec3C &normal_out
        ) {
            bool is_positive_open = getFaceRole(state, axis_index * 2) == RSE_VOXEL_FACE_AIR;
            bool is_negative_open = getFaceRole(state, axis_index * 2 + 1) == RSE_VOXEL_FACE_AIR;

            // The open face is the one the partner touches, so the partner lies on that side
            // and the normal points back the other way.
            if (is_positive_open && is_negative_open)
                normal_out = to_partner.dot(axis) < FloatC(0) ? axis : -axis;
            else if (is_positive_open)
                normal_out = -axis;
            else if (is_negative_open)
                normal_out = axis;
            else
                return false;
            return true;
        }

        /** @brief Overlap test on one axis */
        HOST_DEVICE_INLINE_QUALIFIER static FloatC
        _overlapOnAxis(const OBB &a, const OBB &b, const Vec3C &axis, const Vec3C &center_diff) {
            FloatC dist = RS_ABS(center_diff.dot(axis));
            FloatC ra = _projectionRange(a, axis);
            FloatC rb = _projectionRange(b, axis);

            return ra + rb - dist;  // negative => no overlap
        }

        /** @brief Computes projection range (half extent) of an OBB on a normalized axis */
        HOST_DEVICE_INLINE_QUALIFIER static FloatC
        _projectionRange(const OBB &box, const Vec3C &axis) {
            FloatC r = box.half_ext.dot(
                Vec3C(box.axis[0].dot(axis), box.axis[1].dot(axis), box.axis[2].dot(axis)).abs()
            );
            return r;
        }
    };

    /*****************************************************************
     * Device side methods for acquiring info
     *****************************************************************/

    /** @brief Zero-stress nominal voxel size including expansion. */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC baseSize(FloatC vm_voxel_size, FloatC v_expansion) {
        return vm_voxel_size * (FloatC(1) + v_expansion);
    }

    /** @brief Transverse area of the voxel about an axis, accounting for Poisson strain. */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC transverseArea(
        FloatC vm_voxel_size,
        FloatC vm_nu,
        RSE_LinkAxis axis,
        const Vec3C &poissons_strain
    ) {
        if (vm_nu == 0) return vm_voxel_size * vm_voxel_size;

        switch (axis) {
            case RSE_X_AXIS:
                return vm_voxel_size * vm_voxel_size * (FloatC(1) + poissons_strain.y) *
                       (FloatC(1) + poissons_strain.z);
            case RSE_Y_AXIS:
                return vm_voxel_size * vm_voxel_size * (FloatC(1) + poissons_strain.x) *
                       (FloatC(1) + poissons_strain.z);
            case RSE_Z_AXIS:
                return vm_voxel_size * vm_voxel_size * (FloatC(1) + poissons_strain.x) *
                       (FloatC(1) + poissons_strain.y);
            default:
                return vm_voxel_size * vm_voxel_size;
        }
    }

    /** @brief Sum of strain in the two axes perpendicular to the given axis. */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC
    transverseStrainSum(FloatC vm_nu, RSE_LinkAxis axis, const Vec3C &poissons_strain) {
        if (vm_nu == 0) return 0;

        switch (axis) {
            case RSE_X_AXIS:
                return poissons_strain.y + poissons_strain.z;
            case RSE_Y_AXIS:
                return poissons_strain.x + poissons_strain.z;
            case RSE_Z_AXIS:
                return poissons_strain.x + poissons_strain.y;
            default:
                return FloatC(0.0);
        }
    }

    /** @brief Whether the world stands on a floor at all. */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool hasFloor(Kernel *k) {
        RSE_VisionFloorDataType floor_data_type = k->ctx.floor_vision_mem.data_type;
        return floor_data_type == RSE_VISION_FLOOR_DATA_FLAT ||
               floor_data_type == RSE_VISION_FLOOR_DATA_ARRAY;
    }

    /** @brief Whether the floor is one level plane, so its up-normal is always (0, 0, 1). */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool isFloorFlat(Kernel *k) {
        return k->ctx.floor_vision_mem.data_type == RSE_VISION_FLOOR_DATA_FLAT;
    }

    /**
     * @brief Interference in meters between the voxel box and the floor, negative if clear.
     *
     * @details Reaches along the floor normal with the turned box, so the answer is the depth
     * of whichever corner hangs lowest. A box lying square gives back half its size, the same
     * as measuring straight down, while a tilted one reaches further and is caught earlier.
     *
     * @param orientation voxel orientation.
     * @param nnn         box corner offset on the low side of each local axis.
     * @param ppp         box corner offset on the high side of each local axis.
     * @param position    voxel center in global coords.
     * @param floor_normal floor up-normal under the voxel, unit.
     * @param floor_z     floor height under the voxel.
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatC floorPenetration(
        const Quat3C &orientation,
        const Vec3C &nnn,
        const Vec3C &ppp,
        const Vec3C &position,
        const Vec3C &floor_normal,
        FloatC floor_z = 0
    ) {
        Vec3C axis_x = orientation.rotateVec3D(Vec3C(FloatC(1), FloatC(0), FloatC(0)));
        Vec3C axis_y = orientation.rotateVec3D(Vec3C(FloatC(0), FloatC(1), FloatC(0)));
        Vec3C axis_z = orientation.rotateVec3D(Vec3C(FloatC(0), FloatC(0), FloatC(1)));
        FloatC along_x = axis_x.dot(floor_normal);
        FloatC along_y = axis_y.dot(floor_normal);
        FloatC along_z = axis_z.dot(floor_normal);
        FloatC reach = RS_MAX(-along_x * nnn.x, -along_x * ppp.x) +
                       RS_MAX(-along_y * nnn.y, -along_y * ppp.y) +
                       RS_MAX(-along_z * nnn.z, -along_z * ppp.z);
        return reach - (position.z - floor_z) * floor_normal.z;
    }

    /** @brief Sample the floor height and up-normal under a position, false when no floor. */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool floorHeightAndNormal(
        Kernel *k,
        const Vec3C &position,
        FloatC &floor_height_out,
        Vec3C &normal_out
    ) {
        auto &ctx = k->ctx;
        RSE_VisionFloorDataType floor_data_type = k->ctx.floor_vision_mem.data_type;
        if (floor_data_type != RSE_VISION_FLOOR_DATA_FLAT &&
            floor_data_type != RSE_VISION_FLOOR_DATA_ARRAY)
            return false;

        floor_height_out = FloatC(ctx.floor.height);
        normal_out = {0, 0, 1};
        if (floor_data_type == RSE_VISION_FLOOR_DATA_ARRAY) {
            RSize x_values = ctx.floor.x_values;
            RSize y_values = ctx.floor.y_values;
            // Align world origin (0,0) to the floor center; sample the cell directly under the
            // position — heights and per-cell normals were precomputed at init.
            FloatC fx = position.x * ctx.floor.inv_cell_x + FloatC(0.5) * FloatC(x_values);
            FloatC fy = position.y * ctx.floor.inv_cell_y + FloatC(0.5) * FloatC(y_values);
            int ix = static_cast<int>(FI::rs_floor(fx));
            int iy = static_cast<int>(FI::rs_floor(fy));
            bool in_bounds = true;
            if (ctx.floor.is_x_repeat) {
                RSize u = static_cast<RSize>(ix + static_cast<int>(ctx.floor.x_wrap_offset));
                ix = static_cast<int>(ctx.floor.x_divmod.mod(u));
            } else if (ix < 0 || ix >= static_cast<int>(x_values)) {
                in_bounds = false;
            }
            if (ctx.floor.is_y_repeat) {
                RSize u = static_cast<RSize>(iy + static_cast<int>(ctx.floor.y_wrap_offset));
                iy = static_cast<int>(ctx.floor.y_divmod.mod(u));
            } else if (iy < 0 || iy >= static_cast<int>(y_values)) {
                in_bounds = false;
            }
            if (in_bounds) {
                device::rs_device_std::array<RSize, 2> coords{
                    static_cast<RSize>(iy), static_cast<RSize>(ix)
                };
                floor_height_out = ctx.floor.heights.get(coords);
                auto n = ctx.floor.normals.get(coords);
                normal_out = Vec3C(n.x, n.y, n.z);
            }
        }
        return true;
    }

    /** @brief Linear velocity of the voxel in m/s, global frame. */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    linearVelocity(const Vec3C &v_linear_momentum, FloatC vm_mass_inverse) {
        return v_linear_momentum * vm_mass_inverse;
    }

    /** @brief Angular velocity of the voxel in rad/s, global frame. */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    angularVelocity(const Vec3C &v_angular_momentum, FloatC vm_moment_inertia_inverse) {
        return v_angular_momentum * vm_moment_inertia_inverse;
    }

   private:
    /** @brief Corner `c` of a strained box in voxel space, bit n picking the positive side. */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _getVoxelBoxCorner(int c, const Vec3C &nnn, const Vec3C &ppp) {
        return Vec3C((c & 1) ? ppp.x : nnn.x, (c & 2) ? ppp.y : nnn.y, (c & 4) ? ppp.z : nnn.z);
    }

    /** @brief Neighbor slot facing corner `c` along `axis`, even slots being the positive side. */
    HOST_DEVICE_INLINE_QUALIFIER static int _getVoxelCornerNeighborDirection(int c, int axis) {
        return 2 * axis + (((c >> axis) & 1) ? 0 : 1);
    }

    /** @brief Neighbor across `direction`, or null when that bond is gone. */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static RHandle
    _getVoxelBondedNeighbor(Kernel *k, RHandle voxel, int direction, bool &is_rigid_or_static_out) {
        auto &ctx = k->ctx;
        RS_VoxelNeighborStateUnion neighbor_state_union{V_G(neighbor_state)};
        RState bit = RState(1) << direction;
        if (!(neighbor_state_union.neighbor_state.is_neighbor_linked & bit)) return RS_NULL_HANDLE;
        is_rigid_or_static_out =
            (neighbor_state_union.neighbor_state.is_neighbor_rigid_or_static & bit) != 0;
        auto neighbor_voxels = V_G(neighbor_voxels);
        return neighbor_voxels[direction];
    }

    /** @brief World point `voxel` reads for corner `c` of its own strained box. */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    _getVoxelWorldCorner(Kernel *k, RHandle voxel, int c) {
        auto &ctx = k->ctx;
        Vec3C corner = _getVoxelBoxCorner(c, Vec3C(V_G(nnn_offset)), Vec3C(V_G(ppp_offset)));
        Quat3C orientation = V_G(orientation);
        Vec3C position = V_G(position);
        return position + orientation.rotateVec3D(corner);
    }

    /**
     * @brief Offset from a soft voxel's center to corner `c`, averaged with the bonded neighbors
     *        that touch it.
     *
     * @details A corner is shared by up to eight voxels and each reads the same physical point off
     * its own strained box, so it only closes when every sharer averages the same set. Three face
     * neighbors are one hop away and three diagonals are two, reached through either face. A soft
     * diagonal counts only when both routes survive and agree, which lets a real tear open, while
     * one route is enough to reach a rigid or static one because every voxel of an undeformed body
     * reports the same point. The mean is unweighted so a group moving rigidly reproduces its own
     * corners, and a rigid or static sharer is authoritative, so the soft side grounds onto those
     * readings alone.
     *
     * @note A torn soft bond still welds to a rigid neighbor whenever the other route survives, so
     * a tear running flush against an anchor stays shut. That is benign because the anchor's
     * geometry is authoritative there anyway, but it does mean tears do not open against an anchor.
     *
     * @param k           simulation kernel.
     * @param voxel       the soft voxel owning the corner.
     * @param c           corner index, bit n set when the corner is on the positive side of axis n.
     * @param is_surface  whether the voxel is on the surface, gating the two-hop diagonals.
     * @param position    voxel center in world coordinates.
     * @param orientation rotation from voxel space to world.
     * @param nnn         negative-side strained offsets of this voxel.
     * @param ppp         positive-side strained offsets of this voxel.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C _computeVoxelSeamCornerOffset(
        Kernel *k,
        RHandle voxel,
        int c,
        bool is_surface,
        const Vec3C &position,
        const Quat3C &orientation,
        const Vec3C &nnn,
        const Vec3C &ppp
    ) {
        Vec3C soft_sum = position + orientation.rotateVec3D(_getVoxelBoxCorner(c, nnn, ppp));
        FloatC soft_num = FloatC(1);
        Vec3C rigid_sum = Vec3C::zero();
        FloatC rigid_num = FloatC(0);

        for (int axis = 0; axis < 3; axis++) {
            bool is_rigid_or_static = false;
            RHandle neighbor = _getVoxelBondedNeighbor(
                k, voxel, _getVoxelCornerNeighborDirection(c, axis), is_rigid_or_static
            );
            if (neighbor == RS_NULL_HANDLE) continue;

            // The neighbor faces us across this axis, so it reads the same point as the corner
            // with the axis bit flipped.
            Vec3C world = _getVoxelWorldCorner(k, neighbor, c ^ (1 << axis));
            if (is_rigid_or_static) {
                rigid_sum = rigid_sum + world;
                rigid_num = rigid_num + FloatC(1);
            } else {
                soft_sum = soft_sum + world;
                soft_num = soft_num + FloatC(1);
            }
        }

        // Interior corners are never traced, so the diagonals are only worth two hops on the
        // surface, where the readings that fail to meet are the ones a ray can see.
        if (is_surface) {
            for (int axis_a = 0; axis_a < 3; axis_a++) {
                int axis_b = (axis_a + 1) % 3;
                int direction_a = _getVoxelCornerNeighborDirection(c, axis_a);
                int direction_b = _getVoxelCornerNeighborDirection(c, axis_b);

                bool is_unused = false;
                bool is_by_a_rigid = false;
                bool is_by_b_rigid = false;
                RHandle diagonal_by_a = RS_NULL_HANDLE;
                RHandle diagonal_by_b = RS_NULL_HANDLE;
                if (RHandle by_a = _getVoxelBondedNeighbor(k, voxel, direction_a, is_unused);
                    by_a != RS_NULL_HANDLE)
                    diagonal_by_a = _getVoxelBondedNeighbor(k, by_a, direction_b, is_by_a_rigid);
                if (RHandle by_b = _getVoxelBondedNeighbor(k, voxel, direction_b, is_unused);
                    by_b != RS_NULL_HANDLE)
                    diagonal_by_b = _getVoxelBondedNeighbor(k, by_b, direction_a, is_by_b_rigid);

                // One route reaches a rigid or static diagonal, since an undeformed body reports
                // the same point from any of its voxels. A soft one needs both routes to survive
                // and agree, so a torn bond leaves the corner free to open.
                RHandle diagonal = RS_NULL_HANDLE;
                bool is_rigid_or_static = false;
                if (diagonal_by_a != RS_NULL_HANDLE && is_by_a_rigid) {
                    diagonal = diagonal_by_a;
                    is_rigid_or_static = true;
                } else if (diagonal_by_b != RS_NULL_HANDLE && is_by_b_rigid) {
                    diagonal = diagonal_by_b;
                    is_rigid_or_static = true;
                } else if (diagonal_by_a != RS_NULL_HANDLE && diagonal_by_a == diagonal_by_b) {
                    diagonal = diagonal_by_a;
                }
                if (diagonal == RS_NULL_HANDLE) continue;

                Vec3C world = _getVoxelWorldCorner(k, diagonal, c ^ (1 << axis_a) ^ (1 << axis_b));
                if (is_rigid_or_static) {
                    rigid_sum = rigid_sum + world;
                    rigid_num = rigid_num + FloatC(1);
                } else {
                    soft_sum = soft_sum + world;
                    soft_num = soft_num + FloatC(1);
                }
            }
        }

        Vec3C corner = rigid_num > FloatC(0) ? rigid_sum / rigid_num : soft_sum / soft_num;
        return corner - position;
    }

    /** @brief Compose contact-BVH leaf metadata from voxel state. */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    _composeVoxelContactBVHLeafMetadata(Kernel *k, RHandle voxel, RState &metadata_out) {
        auto &ctx = k->ctx;
        RS_VoxelStateUnion s_union{V_G(state)};
        RS_VoxelContactBVHLeafMetadataUnion m_union{0};
        m_union.meta.is_enabled = s_union.voxel_state.is_enabled;
        m_union.meta.is_surface = s_union.voxel_state.is_surface;
        m_union.meta.is_static = s_union.voxel_state.is_static;
        m_union.meta.is_rigid = s_union.voxel_state.is_rigid;
        m_union.meta.is_contact_active = isContactActive(s_union.voxel_state_packed) ? 1 : 0;
        m_union.meta.is_structure_constrained = s_union.voxel_state.is_structure_constrained;
        metadata_out = m_union.meta_packed;
    }

    /** @brief Compose ray-BVH leaf metadata from voxel state and vision material type. */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    _composeVoxelRayBVHLeafMetadata(Kernel *k, RHandle voxel, RState &metadata_out) {
        auto &ctx = k->ctx;
        RS_VoxelStateUnion s_union{V_G(state)};
        RHandle vision_material = V_G(vision_material);
        RSE_VisionMaterialType render_type = RSE_VISION_MATERIAL_OPAQUE;
        if (vision_material != RS_NULL_HANDLE) {
            RS_VisionMaterialStateUnion vm_union{VSM_G(vision_material, state)};
            render_type = vm_union.vision_material_state.render_type;
        }
        RS_VoxelRayBVHLeafMetadataUnion m_union{0};
        m_union.meta.is_enabled = s_union.voxel_state.is_enabled;
        m_union.meta.is_surface = s_union.voxel_state.is_surface;
        m_union.meta.is_static = s_union.voxel_state.is_static;
        m_union.meta.is_rigid = s_union.voxel_state.is_rigid;
        m_union.meta.is_emissive = (render_type == RSE_VISION_MATERIAL_EMISSIVE) ? 1 : 0;
        m_union.meta.is_transparent = (render_type == RSE_VISION_MATERIAL_TRANSPARENT) ? 1 : 0;
        m_union.meta.is_ray_active = isRayActive(s_union.voxel_state_packed) ? 1 : 0;
        m_union.meta.is_smooth_surface = 0;
        // Rigid and static voxels never deform, always a box.
        m_union.meta.is_box = (s_union.voxel_state.is_static || s_union.voxel_state.is_rigid) ? 1
                                                                                              : 0;
        m_union.meta.face_role = s_union.voxel_state.face_role;
        metadata_out = m_union.meta_packed;
    }

    /** @brief Compose the contact-BVH leaf identifier of metadata and overlap payload. */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _composeVoxelContactBVHLeafIdentifier(
        Kernel *k,
        RHandle voxel,
        RS_VoxelContactBVHObjectIdentifier &oid_out
    ) {
        auto &ctx = k->ctx;
        _composeVoxelContactBVHLeafMetadata(k, voxel, oid_out.object_metadata);
        oid_out.object_handle = voxel;
        oid_out.contact_filter_key = V_G(contact_filter_key);
        oid_out.position = RVec3f(Vec3C(V_G(position)));
        oid_out.radius = float(V_G(radius));
        oid_out.structure_gid = V_G(structure_gid);
    }

    /** @brief Compose the ray-BVH leaf identifier of metadata and the geometric shape. */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _composeVoxelRayBVHLeafIdentifier(
        Kernel *k,
        RHandle voxel,
        RS_VoxelRayBVHObjectIdentifier &oid_out
    ) {
        auto &ctx = k->ctx;
        Vec3C position = V_G(position);
        Quat3C orientation = V_G(orientation);

        _composeVoxelRayBVHLeafMetadata(k, voxel, oid_out.object_metadata);
        RState metadata = oid_out.object_metadata;
        oid_out.object_handle = voxel;
        oid_out.vision_material = V_G(vision_material);
        oid_out.structure_gid = V_G(structure_gid);
        oid_out.position = RVec3f(position);

        RS_VoxelRayBVHLeafMetadataUnion m_union{metadata};
        if (m_union.meta.is_box) {
            oid_out.shape.box.orientation = RQuat3f(orientation);
            oid_out.shape.box.size = float(V_G(size));
        } else {
            Vec3C nnn = V_G(nnn_offset);
            Vec3C ppp = V_G(ppp_offset);
            for (int c = 0; c < 8; c++) {
                oid_out.shape.hexahedron.corner_offset[c] = _computeVoxelSeamCornerOffset(
                    k, voxel, c, m_union.meta.is_surface, position, orientation, nnn, ppp
                );
            }
        }
    }

    /** @brief Loose world-AABB enclosing a voxel's rotated OBB. */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static RBound3f
    _computeVoxelContactLooseAABB(Kernel *k, RHandle voxel) {
        auto &ctx = k->ctx;
        Vec3C position = V_G(position);
        Quat3C orientation = V_G(orientation);
        Vec3C nnn = V_G(nnn_offset);
        Vec3C ppp = V_G(ppp_offset);
        Vec3C min_b, max_b;
        bool first = true;
        for (int i = 0; i < 8; i++) {
            Vec3C local = _getVoxelBoxCorner(i, nnn, ppp);
            Vec3C world = position + orientation.rotateVec3D(local);
            if (first) {
                min_b = max_b = world;
                first = false;
            } else {
                min_b = min_b.min(world);
                max_b = max_b.max(world);
            }
        }
        return RBound3f(RVec3f(min_b), RVec3f(max_b));
    }

    /** @brief Loose world-AABB enclosing a ray leaf's shape. */
    HOST_DEVICE_INLINE_QUALIFIER static RBound3f _computeVoxelRayLooseAABB(
        const RS_VoxelRayBVHObjectIdentifier &oid
    ) {
        Vec3C position = Vec3C(oid.position);
        RS_VoxelRayBVHLeafMetadataUnion m_union{oid.object_metadata};
        Vec3C min_b, max_b;
        if (m_union.meta.is_box) {
            Quat3C orientation = Quat3C(oid.shape.box.orientation);
            FloatC half = FloatC(0.5) * FloatC(oid.shape.box.size);
            for (int c = 0; c < 8; c++) {
                Vec3C local((c & 1) ? half : -half, (c & 2) ? half : -half, (c & 4) ? half : -half);
                Vec3C world = position + orientation.rotateVec3D(local);
                if (c == 0) {
                    min_b = max_b = world;
                } else {
                    min_b = min_b.min(world);
                    max_b = max_b.max(world);
                }
            }
        } else {
            for (int c = 0; c < 8; c++) {
                Vec3C world =
                    position + oid.shape.hexahedron.corner_offset[c].template toVec3D<FloatC>();
                if (c == 0) {
                    min_b = max_b = world;
                } else {
                    min_b = min_b.min(world);
                    max_b = max_b.max(world);
                }
            }
        }
        return RBound3f(RVec3f(min_b), RVec3f(max_b));
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_VOXEL_C_H