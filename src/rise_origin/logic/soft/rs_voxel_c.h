// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VOXEL_C_H
#define RS_VOXEL_C_H

#include "rise_origin/core/rs_kernel.h"
#include "rise_origin/device/rs_static_construct.h"

namespace rise {

namespace logic {

/** @brief Read the role of face `face` (0..5) from a packed voxel state. */
HOST_DEVICE_INLINE_QUALIFIER RSE_VoxelFaceRole getFaceRole(RState state, int face) {
    RS_VoxelStateUnion u{state};
    return static_cast<RSE_VoxelFaceRole>((u.voxel_state.face_role >> (2 * face)) & RState(0x3));
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

/** @brief Whether any face is air or a transparent boundary. */
HOST_DEVICE_INLINE_QUALIFIER bool isRenderActive(RState state) {
    for (int f = 0; f < 6; f++) {
        RSE_VoxelFaceRole role = getFaceRole(state, f);
        if (role == RSE_VOXEL_FACE_AIR || role == RSE_VOXEL_FACE_TRANSPARENT_BOUNDARY) return true;
    }
    return false;
}

/** @brief Whether any face is exposed to air; the voxel can contact another body. */
HOST_DEVICE_INLINE_QUALIFIER bool isContactActive(RState state) {
    for (int f = 0; f < 6; f++) {
        if (getFaceRole(state, f) == RSE_VOXEL_FACE_AIR) return true;
    }
    return false;
}

template <RSE_FloatType FloatComputeAccuracy>
class RS_LinkCompute;

template <RSE_FloatType FloatComputeAccuracy>
class RS_RigidBodyCompute;

class RSCF_DisableVoxel : public device::RS_ComputeFunction<RSCF_DisableVoxel> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel k, RSize voxel_segment_offset) {
        auto &ctx = k.ctx;
        RHandle voxel = index1D() + voxel_segment_offset;
        RS_VoxelStateUnion state_union{V_G(voxel, state)};
        state_union.voxel_state.is_enabled = false;
        V_S(state, state_union.voxel_state_packed);
    }
};

class RSCF_ResolveVoxelDependency : public device::RS_ComputeFunction<RSCF_ResolveVoxelDependency> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel k,
        RSize voxel_segment_offset,
        RSize link_segment_offset,
        RSize voxel_material_segment_offset,
        RSize rigid_body_segment_offset,
        RSize expansion_signal_segment_offset
    ) {
        RHandle voxel = index1D() + voxel_segment_offset;
        auto &ctx = k.ctx;
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

        RHandle expansion_signal = V_G(expansion_signal_sid) + expansion_signal_segment_offset;
        V_S(expansion_signal, expansion_signal);
    }
};

template <RSE_FloatType FloatComputeAccuracy>
class RS_VoxelCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using FI = RS_FloatIntrinsics<FloatC>;
    using LinkCompute = RS_LinkCompute<FloatComputeAccuracy>;
    using RigidBodyCompute = RS_RigidBodyCompute<FloatComputeAccuracy>;

    /*****************************************************************
     * Device side compute functions for updating
     *****************************************************************/
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

                if (state_union.voxel_state.is_fixed || !state_union.voxel_state.is_enabled) return;

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
                        k, voxel, voxel_material, state_union, ext_force, ext_moment
                    );
                } else {
                    _updateExpansionAndSize(k, voxel, voxel_material, dt, state_union, voxel_size);

                    _updateExternalForceAndMoment(
                        k, voxel, voxel_material, state_union, ext_force, ext_moment
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

#ifdef RS_DEBUG_VOXEL
                    CUDA_PRINTF_ASSERT(!isnan(total_force.x) && abs(total_force.x) < 1e6);
                    CUDA_PRINTF_ASSERT(!isnan(total_force.y) && abs(total_force.y) < 1e6);
                    CUDA_PRINTF_ASSERT(!isnan(total_force.z) && abs(total_force.z) < 1e6);
#endif

                    /// Integration
                    linear_momentum += total_force * dt;

                    // Note: previous_friction state is "current" after being updated by
                    // _computeTotalForceWithFloorFriction
                    bool is_static_friction = state_union.voxel_state.previous_friction_state ==
                                              RSE_VOXEL_STATIC_FRICTION;
                    // if we're in a state of static friction, zero out all linear momentum
                    // in plane
                    if (is_static_friction) {
                        linear_momentum = linear_momentum.project(normal_vec);
                    }

                    // movement of the voxel this timestep
                    Vec3C translate(linear_momentum * (dt * mass_inverse));
                    // if we're in a state of static friction, zero out all translation in
                    // plane
                    if (is_static_friction) {
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
                    V_S(state, state_union.voxel_state_packed);

#ifdef RS_DEBUG_VOXEL
                    CUDA_PRINTF_ASSERT(Vec3C(V_G(position)).isValid());
                    CUDA_PRINTF_ASSERT(Quat3C(V_G(orientation)).isValid());
#endif
                    _updatePoissonsStrainAndCornerOffset(k, voxel, voxel_material, voxel_size);
                }
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
                FloatC signal = ctx.structure_expansion_signals[V_G(expansion_signal)];
                FloatC target = state_union_in_out.voxel_state.expansion_control_mode ==
                                        RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE
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
            Vec3C &ext_force_out,
            Vec3C &ext_moment_out
        ) {
            auto &ctx = k->ctx;

            // forces from internal bonds
            ext_force_out = Vec3C::zero();
            ext_moment_out = Vec3C::zero();
            Vec3C ext_abs_force = Vec3C::zero();

            auto link_sid = V_G(link_sid);
            auto links = V_G(links);
            bool has_link = false;
            bool has_null_link = false;

            if (state_union_in_out.voxel_state.has_link) {
                device::loop<0, 2>([&](auto dir_const) {
                    device::loop<0, 6, 2>([&](auto axis_const) {
                        static constexpr RIndex i = dir_const.value + axis_const.value;
                        RHandle &link = sharedMemAt1D(SharedMemLayout::template get<i>());
                        if (link != RS_NULL_HANDLE) {
                            RS_LinkStateUnion link_state_union{L_G(link, state)};
                            if (link_state_union.link_state.is_failed) {
                                link_sid[i] = RS_NULL_INDEX;
                                links[i] = RS_NULL_HANDLE;
                                link = RS_NULL_HANDLE;
                                has_null_link = true;
                                // A torn bond exposes this face of a soft voxel to air.
                                if (!state_union_in_out.voxel_state.is_rigid) {
                                    setFaceRole(
                                        state_union_in_out.voxel_state_packed, i, RSE_VOXEL_FACE_AIR
                                    );
                                }
                            } else {
                                if (link) {
                                    has_link = true;
                                    if (!(state_union_in_out.voxel_state.link_is_tendon &
                                          (1u << i))) {
                                        // For link, it always points from voxel_neg to voxel_pos
                                        // and its direction is always in the positive direction
                                        if (isLinkDirectionNegative(
                                                static_cast<RSE_LinkDirection>(i)
                                            )) {
                                            ext_force_out += Vec3C(L_G(link, force_pos));
                                            ext_moment_out += Vec3C(L_G(link, moment_pos));
                                        } else {
                                            ext_force_out += Vec3C(L_G(link, force_neg));
                                            ext_moment_out += Vec3C(L_G(link, moment_neg));
                                        }
                                    } else {
                                        if (L_G(link, voxel_neg) == voxel)
                                            ext_abs_force += Vec3C(L_G(link, force_neg));
                                        else
                                            ext_abs_force += Vec3C(L_G(link, force_pos));
                                    }
                                }
                            }
                        } else {
                            has_null_link = true;
                        }
                    });
                });

                state_union_in_out.voxel_state.has_link = has_link;
                state_union_in_out.voxel_state.is_surface =
                    (!state_union_in_out.voxel_state.is_rigid && has_null_link) ||
                    state_union_in_out.voxel_state.is_surface;
            }

            // from local to global coordinates
            RQuat3rf orientation = V_G(orientation);
            ext_force_out = orientation.rotateVec3D(ext_force_out) + ext_abs_force;

            // Contact force
            if (state_union_in_out.voxel_state.has_contact_voxel_0 &&
                (!state_union_in_out.voxel_state.is_rigid ||
                 !state_union_in_out.voxel_state.is_contact_voxel_0_rigid)) {
                ext_force_out += Vec3C(V_G(contact_force_0));
            }
            if (state_union_in_out.voxel_state.has_contact_voxel_1 &&
                (!state_union_in_out.voxel_state.is_rigid ||
                 !state_union_in_out.voxel_state.is_contact_voxel_1_rigid)) {
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

#ifdef RS_DEBUG_VOXEL
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

            FloatC floor_height;
            if (floorHeightAndNormal(k, position, floor_height, normal_vec_out)) {
                FloatC floor_penetration = floorPenetration(voxel_size, position.z, floor_height);
                if (floor_penetration >= 0) {
                    Vec3C velocity = linearVelocity(linear_momentum_in_out, mass_inverse);
                    FloatC normal_force_norm = VM_G(penetration_stiffness) * floor_penetration;

                    // in the normal direction: k*x-C*v - spring and damping
                    total_force_in_out += normal_force_norm * normal_vec_out;
                    total_force_in_out -= VM_G(contact_damping_translate_c) * velocity;
                    Vec3C plane_force =
                        total_force_in_out - total_force_in_out.project(normal_vec_out);
                    Vec3C plane_velocity = velocity - velocity.project(normal_vec_out);
                    if (state_union_in_out.voxel_state.previous_friction_state !=
                        RSE_VOXEL_DYNAMIC_FRICTION) {
                        // If voxel is previously in static friction mode (no lateral
                        // motion) check whether the voxel will continue to stay in static
                        // friction state If voxel is previously in no contact mode, check
                        // which state the voxel will become

                        // use squares to avoid a square root
                        FloatC surface_force_sq = plane_force.length2();
                        FloatC friction_force_sq = VM_G(u_static) * normal_force_norm;
                        friction_force_sq = friction_force_sq * friction_force_sq;

                        current_friction_state = surface_force_sq < friction_force_sq
                                                     ? RSE_VOXEL_STATIC_FRICTION
                                                     : RSE_VOXEL_DYNAMIC_FRICTION;
                    } else {
                        // Check whether voxel will continue to stay in dynamic friction
                        // state Work of dynamic friction
                        FloatC friction_force = VM_G(u_dynamic) * normal_force_norm;
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
                            total_force_in_out +=
                                VM_G(u_dynamic) * normal_force_norm * friction_dir;
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
            state_union_in_out.voxel_state.previous_friction_state = current_friction_state;
        }

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updatePoissonsStrainAndCornerOffset(
            Kernel *k,
            RHandle voxel,
            RHandle voxel_material,
            FloatC voxel_size
        ) {
            // if no connections in the positive and negative directions of a particular
            // axis,
            // strain is zero.
            // if one connection in positive or negative direction of a particular
            // axis, strain is that strain - ?? and force or constraint?
            // if connections in both the positive and negative directions of a particular
            // axis, strain is the average.

            auto &ctx = k->ctx;

            int num_bond_axis[3] = {0, 0, 0};
            FloatC strains[6];

            device::loop<0, 2>([&](auto dir_const) {
                device::loop<0, 6, 2>([&](auto axis_const) {
                    static constexpr RIndex i = dir_const.value + axis_const.value;
                    RHandle link = sharedMemAt1D(SharedMemLayout::template get<i>());
                    num_bond_axis[i >> 1] += link != RS_NULL_HANDLE;
                    strains[i] = link == RS_NULL_HANDLE
                                     ? FloatC(0)
                                     : LinkCompute::axialStrain(
                                           L_G(link, strain), L_G(link, strain_ratio), false
                                       );
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
            tension[0] = num_bond_axis[0] == 2;
            axis_strain_sum.x *= tension[0] ? FloatC(0.5) : FloatC(1);
            tension[1] = num_bond_axis[1] == 2;
            axis_strain_sum.y *= tension[1] ? FloatC(0.5) : FloatC(1);
            tension[2] = num_bond_axis[2] == 2;
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
#ifdef RS_DEBUG_VOXEL
            CUDA_PRINTF_ASSERT(!isnan(axis_strain_sum.x) && axis_strain_sum.x < 1e6);
            CUDA_PRINTF_ASSERT(!isnan(axis_strain_sum.y) && axis_strain_sum.y < 1e6);
            CUDA_PRINTF_ASSERT(!isnan(axis_strain_sum.z) && axis_strain_sum.z < 1e6);
#endif
        }
    };

    /// Per voxel slot over the whole SoA: write 1 to flag[i] iff this voxel belongs in the static
    /// voxel BVH. Sweeps every slot, including fully-fixed structures absent from the active list.
    class RSCF_GetStaticVoxelFlag : public device::RS_ComputeFunction<RSCF_GetStaticVoxelFlag> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_GetStaticVoxelFlag>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(const Kernel *k, RS_MemoryPtr buffer, RSize num_voxels) {
            if (k == nullptr) return;
            auto &ctx = k->ctx;
            RIndex i = index1D();
            if (i >= num_voxels) return;
            RHandle voxel = RHandle(i);
            auto &flag = buffer.template get<RIndex>(i);
            RS_VoxelStateUnion s_union{V_G(state)};
            if (!s_union.voxel_state.is_enabled || !(isContactActive(s_union.voxel_state_packed) ||
                                                     isRenderActive(s_union.voxel_state_packed))) {
                flag = 0;
                return;
            }
            bool is_static = s_union.voxel_state.is_fixed;
            if (!is_static && s_union.voxel_state.is_rigid) {
                RHandle rigid_body = V_G(rigid_body);
                if (rigid_body != RS_NULL_HANDLE) {
                    RS_RigidBodyStateUnion rb_union{RB_G(rigid_body, state)};
                    is_static = rb_union.rigid_body_state.is_fixed;
                }
            }
            flag = is_static ? 1 : 0;
        }
    };

    /// Per voxel slot over the whole SoA: compact static voxels into a packed handle list.
    class RSCF_CompactStaticVoxels : public device::RS_ComputeFunction<RSCF_CompactStaticVoxels> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_CompactStaticVoxels>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            const Kernel *k,
            RS_MemoryPtr flags,
            RSize num_voxels,
            RS_MemoryPtr prefix_sums,
            RS_MemoryPtr active_handles
        ) {
            if (k == nullptr) return;
            RIndex i = index1D();
            if (i >= num_voxels) return;
            if (flags.template get<RIndex>(i)) {
                RIndex out = prefix_sums.template get<RIndex>(i) - 1;
                active_handles.template get<RHandle>(out) = RHandle(i);
            }
        }
    };

    /// Per voxel: write 1 to flag[i] iff this voxel belongs in the dynamic voxel BVH.
    class RSCF_GetDynamicVoxelFlag : public device::RS_ComputeFunction<RSCF_GetDynamicVoxelFlag> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(const Kernel *k, RS_MemoryPtr buffer, RHandle voxel, RIndex flag_index) {
            if (k == nullptr || voxel == RS_NULL_HANDLE) return;
            auto &ctx = k->ctx;
            auto &flag = buffer.template get<RIndex>(flag_index);
            RS_VoxelStateUnion s_union{V_G(state)};
            if (!s_union.voxel_state.is_enabled || !(isContactActive(s_union.voxel_state_packed) ||
                                                     isRenderActive(s_union.voxel_state_packed))) {
                flag = 0;
                return;
            }
            bool is_static = s_union.voxel_state.is_fixed;
            if (!is_static && s_union.voxel_state.is_rigid) {
                RHandle rigid_body = V_G(rigid_body);
                if (rigid_body != RS_NULL_HANDLE) {
                    RS_RigidBodyStateUnion rb_union{RB_G(rigid_body, state)};
                    is_static = rb_union.rigid_body_state.is_fixed;
                }
            }
            flag = is_static ? 0 : 1;
        }
    };

    /// Per voxel: compact dynamic voxels into a packed handle list using the prefix-sum.
    class RSCF_CompactDynamicVoxels : public device::RS_ComputeFunction<RSCF_CompactDynamicVoxels> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            const Kernel *k,
            RS_MemoryPtr flags,
            RHandle voxel,
            RIndex flag_index,
            RS_MemoryPtr prefix_sums,
            RS_MemoryPtr active_handles
        ) {
            if (k == nullptr || voxel == RS_NULL_HANDLE) return;
            if (flags.template get<RIndex>(flag_index)) {
                RIndex out = prefix_sums.template get<RIndex>(flag_index) - 1;
                active_handles.template get<RHandle>(out) = voxel;
            }
        }
    };

    /// Per (active) voxel: write {object_metadata, object_handle} into obj_ids
    /// and the loose world-AABB into leaf_bounds. Build-path kernel.
    class RSCF_FillVoxelBVHLeafData : public device::RS_ComputeFunction<RSCF_FillVoxelBVHLeafData> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_FillVoxelBVHLeafData>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            device::RS_Vector<RHandle> active_handles,
            device::RS_Vector<RS_VoxelBVHObjectIdentifier> obj_ids,
            device::RS_Vector<RBound3f> leaf_bounds
        ) {
            RIndex leaf_index = index1D();
            if (leaf_index >= active_handles.size()) return;
            RHandle voxel = active_handles[leaf_index];
            obj_ids[leaf_index] = _composeVoxelBVHLeafIdentifier(k, voxel);
            leaf_bounds[leaf_index] = _computeVoxelLooseAABB(k, voxel);
        }
    };

    /// Per existing BVH leaf: read the voxel handle out of obj_ids and refresh
    /// both metadata and bounds in place. Refit-path kernel; preserves Morton order.
    class RSCF_RefitVoxelBVHLeafData
        : public device::RS_ComputeFunction<RSCF_RefitVoxelBVHLeafData> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_RefitVoxelBVHLeafData>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            device::RS_Vector<RS_VoxelBVHObjectIdentifier> obj_ids,
            device::RS_Vector<RBound3f> leaf_bounds
        ) {
            RIndex leaf_index = index1D();
            RHandle voxel = obj_ids[leaf_index].object_handle;
            obj_ids[leaf_index] = _composeVoxelBVHLeafIdentifier(k, voxel);
            leaf_bounds[leaf_index] = _computeVoxelLooseAABB(k, voxel);
        }
    };

    /// Per leaf of the dynamic voxel BVH (Morton-sorted by HLBVH build): query
    /// both voxel BVHs for nearby candidates, apply same-body / connected-body
    /// filtering, and store the two largest sphere-overlap contacts into
    /// contact_voxel_0 / contact_voxel_1. Launched over
    /// `ctx.dynamic_voxel_bvh.object_identifiers` so adjacent warp threads work
    /// on spatially-adjacent voxels — BVH walks share node cache lines.
    class RSCF_FindContactVoxels : public device::RS_ComputeFunction<RSCF_FindContactVoxels> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_FindContactVoxels>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, device::RS_Vector<RS_VoxelBVHObjectIdentifier> sorted_object_identifiers) {
            if (k == nullptr) return;
            RIndex idx = index1D();
            if (idx >= sorted_object_identifiers.size()) return;

            // Metadata-only early-out: skip non-collision / disabled leaves without
            // touching the voxel SoA. is_fixed never enters here because the dynamic
            // BVH build excluded fixed and parent-fixed-rigid-body voxels.
            RS_VoxelBVHLeafMetadataUnion meta{sorted_object_identifiers[idx].object_metadata};
            if (!meta.meta.is_enabled || !meta.meta.is_contact_active) return;

            RHandle voxel = sorted_object_identifiers[idx].object_handle;
            if (voxel == RS_NULL_HANDLE) return;

            auto &ctx = k->ctx;
            RS_VoxelStateUnion state_union{V_G(state)};

            Vec3C position = V_G(position);
            FloatC radius = V_G(radius);

            RIndex self_structure_gid = V_G(structure_gid);
            RS_VoxelContactFilterKeyUnion self_filter_key{V_G(contact_filter_key)};
            RBigIndex self_connected_body_mask =
                self_filter_key.contact_filter_key.rigid_body_sid !=
                        RS_VOXEL_CONTACT_FILTER_KEY_NULL_RIGID_BODY_SID
                    ? RB_G(V_G(rigid_body), connected_body_mask)
                    : RBigIndex(0);

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
            if (ctx.dynamic_voxel_bvh.object_identifiers.size() > 0) {
                auto bvh_accessor = ctx.dynamic_voxel_bvh.root();
                _traverseBVHForContactCandidates(
                    k,
                    voxel,
                    position,
                    radius,
                    self_structure_gid,
                    self_filter_key,
                    self_connected_body_mask,
                    query_bound,
                    bvh_accessor,
                    contact_voxel_0,
                    overlap_0,
                    contact_voxel_1,
                    overlap_1
                );
            }
            if (ctx.static_voxel_bvh.object_identifiers.size() > 0) {
                auto bvh_accessor = ctx.static_voxel_bvh.root();
                _traverseBVHForContactCandidates(
                    k,
                    voxel,
                    position,
                    radius,
                    self_structure_gid,
                    self_filter_key,
                    self_connected_body_mask,
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
            RBigIndex self_connected_body_mask,
            const RBound3f &query_bound,
            BVHAccessor &bvh_accessor_in_out,
            RHandle &contact_voxel_0_in_out,
            FloatC &overlap_0_in_out,
            RHandle &contact_voxel_1_in_out,
            FloatC &overlap_1_in_out
        ) {
            auto &ctx = k->ctx;

            static constexpr RSize STACK_SIZE = 32;
            RIndex stack[STACK_SIZE];
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

                        // Same body + Chebyshev-1 in voxel grid → linked neighbor, skip.
                        if (self_filter_key.contact_filter_key.body_sid ==
                            other_filter_key.contact_filter_key.body_sid) {
                            int dx = int(self_filter_key.contact_filter_key.index_x) -
                                     int(other_filter_key.contact_filter_key.index_x);
                            int dy = int(self_filter_key.contact_filter_key.index_y) -
                                     int(other_filter_key.contact_filter_key.index_y);
                            int dz = int(self_filter_key.contact_filter_key.index_z) -
                                     int(other_filter_key.contact_filter_key.index_z);
                            if (RS_ABS(dx) <= 1 && RS_ABS(dy) <= 1 && RS_ABS(dz) <= 1) continue;
                        }

                        // Constraint-connected rigid-body pair, skip.
                        RBigIndex other_rb_sid = other_filter_key.contact_filter_key.rigid_body_sid;
                        if (other_rb_sid != RS_VOXEL_CONTACT_FILTER_KEY_NULL_RIGID_BODY_SID &&
                            self_filter_key.contact_filter_key.rigid_body_sid !=
                                RS_VOXEL_CONTACT_FILTER_KEY_NULL_RIGID_BODY_SID &&
                            ((self_connected_body_mask >> other_rb_sid) & RBigIndex(1))) {
                            continue;
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

                    if (internal_node.left != RS_NULL_INDEX && stack_top < STACK_SIZE &&
                        internal_node.left_bound.isIntersecting(query_bound)) {
                        stack[stack_top++] = internal_node.left;
                    }
                    if (internal_node.right != RS_NULL_INDEX && stack_top < STACK_SIZE &&
                        internal_node.right_bound.isIntersecting(query_bound)) {
                        stack[stack_top++] = internal_node.right;
                    }
                }
            }
        }
    };

    class RSCF_UpdateVoxelContact : public device::RS_ComputeFunction<RSCF_UpdateVoxelContact> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle voxel) {
            if (k != nullptr && voxel != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                RS_VoxelStateUnion state_union{V_G(state)};
                if (!state_union.voxel_state.is_enabled) return;

                // Detect floor contact for rigid air-exposed voxels; clear the stale floor
                // accumulator.
                bool floor_contact = false;
                if (state_union.voxel_state.is_rigid &&
                    isContactActive(state_union.voxel_state_packed) &&
                    !state_union.voxel_state.is_fixed) {
                    Vec3C voxel_pos = V_G(position);
                    FloatC voxel_size = VM_G(V_G(voxel_material), voxel_size);
                    FloatC floor_height;
                    Vec3C floor_normal;
                    if (floorHeightAndNormal(k, voxel_pos, floor_height, floor_normal) &&
                        floorPenetration(voxel_size, voxel_pos.z, floor_height) > FloatC(0))
                        floor_contact = true;
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

                // Process contact #0
                if (state_union.voxel_state.has_contact_voxel_0)
                    _updateOneContact<0>(k, voxel, obb_a, V_G(contact_voxel_0), state_union);

                // Process contact #1
                if (state_union.voxel_state.has_contact_voxel_1)
                    _updateOneContact<1>(k, voxel, obb_a, V_G(contact_voxel_1), state_union);

                V_S(state, state_union.voxel_state_packed);
            }
        }

       private:
        /// @brief OBB struct for SAT
        struct OBB {
            Vec3C center;    // world-space center
            Vec3C axis[3];   // orthonormal axes
            Vec3C half_ext;  // half extents along each axis
        };

        /// @brief Build an OBB from the voxel data
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

        /// @brief Handle one contact partner
        template <RIndex ContactIndex, typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _updateOneContact(
            Kernel *k,
            RHandle voxel,
            const OBB &obb_a,     // OBB for this voxel
            RHandle other_voxel,  // in/out handle for the contact partner
            RS_VoxelStateUnion &state_union_in_out
        ) {
            auto &ctx = k->ctx;

            // Build OBB for the other voxel
            OBB obb_b = _buildOBB(k, other_voxel);

            // Perform SAT check
            Vec3C contact_normal = Vec3C::zero();
            FloatC penetration_depth = FloatC(0);
            bool intersects = _SAT(obb_a, obb_b, contact_normal, penetration_depth);

            if (!intersects) {
                // No contact => clear
                constexpr RHandle null_handle = RS_NULL_HANDLE;

                if constexpr (ContactIndex == 0) {
                    state_union_in_out.voxel_state.has_contact_voxel_0 = false;
                    V_S(contact_voxel_0, null_handle);
                    V_S(contact_position_0, Vec3C::zero());
                    V_S(contact_normal_0, Vec3C::zero());
                    V_S(contact_depth_0, FloatC(0));
                    V_S(contact_lambda_n_0, FloatC(0));
                    V_S(contact_lambda_t_0, Vec3C::zero());
                } else {
                    state_union_in_out.voxel_state.has_contact_voxel_1 = false;
                    V_S(contact_voxel_1, null_handle);
                    V_S(contact_position_1, Vec3C::zero());
                    V_S(contact_normal_1, Vec3C::zero());
                    V_S(contact_depth_1, FloatC(0));
                    V_S(contact_lambda_n_1, FloatC(0));
                    V_S(contact_lambda_t_1, Vec3C::zero());
                }
            } else {
                // Flip normal if it doesn't point from B to A
                Vec3C center_diff = obb_a.center - obb_b.center;
                if (center_diff.dot(contact_normal) < FloatC(0)) {
                    contact_normal = -contact_normal;
                }

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
                    contact_force,
                    state_union_in_out
                );

                // Store back to current voxel memory
                if constexpr (ContactIndex == 0) {
                    state_union_in_out.voxel_state.has_contact_voxel_0 = true;
                    V_S(contact_position_0, contact_pos);
                    V_S(contact_normal_0, contact_normal);
                    V_S(contact_force_0, contact_force);
                    V_S(contact_depth_0, penetration_depth);
                } else {
                    state_union_in_out.voxel_state.has_contact_voxel_1 = true;
                    V_S(contact_position_1, contact_pos);
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
            Vec3C &contact_force_out,
            RS_VoxelStateUnion &state_union_in_out
        ) {
            auto &ctx = k->ctx;

            RS_VoxelStateUnion contact_voxel_state_union{V_G(other_voxel, state)};
            if constexpr (ContactIndex == 0) {
                state_union_in_out.voxel_state.is_contact_voxel_0_rigid =
                    contact_voxel_state_union.voxel_state.is_rigid;
            } else {
                state_union_in_out.voxel_state.is_contact_voxel_1_rigid =
                    contact_voxel_state_union.voxel_state.is_rigid;
            }

            if (state_union_in_out.voxel_state.is_rigid &&
                contact_voxel_state_union.voxel_state.is_rigid) {
                // Force is resolved by the rigid body solver
                contact_force_out = Vec3C::zero();
            } else {
                RHandle voxel_material = V_G(voxel, voxel_material);
                RHandle other_voxel_material = V_G(other_voxel, voxel_material);

                // 1. Solve the contact force
                if (state_union_in_out.voxel_state.is_rigid) {
                    // Use the stiffness of the contact soft voxel
                    contact_force_out = VM_G(other_voxel_material, penetration_stiffness) *
                                        contact_depth * contact_normal;
                } else {
                    // Use stiffness of the current soft voxel
                    contact_force_out = VM_G(voxel_material, penetration_stiffness) *
                                        contact_depth * contact_normal;
                }

                // 2. Solve the friction force
                // Note: for efficiency, for now we just have dynamic friction, and since
                // the friction coefficient is only between material and ground, for friction
                // coefficient between materials, we use the harmonic mean of the dynamic
                // friction coefficients of the two materials.

                // TODO:
                // 1. implement iteration based solving? because static frictions form
                // a cone and we need to solve it iteratively.
                // 2. have pair wise friction coefficient

                FloatC mass_inverse = VM_G(voxel_material, mass_inverse);
                FloatC other_mass_inverse = VM_G(other_voxel_material, mass_inverse);

                Vec3C velocity = linearVelocity(V_G(voxel, linear_momentum), mass_inverse);
                Vec3C other_velocity =
                    linearVelocity(V_G(other_voxel, linear_momentum), other_mass_inverse);

                // Calculate relative velocity in contact plane
                Vec3C rel_velocity = velocity - other_velocity;
                Vec3C tangent_vel = rel_velocity - rel_velocity.project(contact_normal);
                FloatC tangent_speed = tangent_vel.length();

                if (tangent_speed > 0) {
                    FloatC normal_force = contact_force_out.length();
                    FloatC u_dynamic = VM_G(voxel_material, u_dynamic);
                    FloatC u_dynamic_other = VM_G(other_voxel_material, u_dynamic);
                    FloatC u_dynamic_harmonic =
                        (FloatC(2.0) * u_dynamic * u_dynamic_other) / (u_dynamic + u_dynamic_other);

                    // Apply dynamic friction
                    Vec3C friction_force =
                        -(tangent_vel / tangent_speed) * (u_dynamic_harmonic * normal_force);
                    contact_force_out += friction_force;
                }
            }
        }

        /// @brief Full SAT for OBB vs OBB
        /// @returns true if in contact, and also returns min-overlap axis + depth
        HOST_DEVICE_INLINE_QUALIFIER static bool
        _SAT(const OBB &a, const OBB &b, Vec3C &contact_normal, FloatC &penetration_depth) {
            FloatC min_overlap = RS_NumTraits<FloatC>::highest();
            Vec3C best_axis = Vec3C::zero();

            Vec3C center_diff = b.center - a.center;

            // 1) Check the 3 axes of A
            for (int i = 0; i < 3; i++) {
                FloatC overlap = _overlapOnAxis(a, b, a.axis[i], center_diff);
                if (overlap < FloatC(0)) return false;
                if (overlap < min_overlap) {
                    min_overlap = overlap;
                    best_axis = a.axis[i];
                }
            }

            // 2) Check the 3 axes of B
            for (int i = 0; i < 3; i++) {
                FloatC overlap = _overlapOnAxis(a, b, b.axis[i], center_diff);
                if (overlap < FloatC(0)) return false;
                if (overlap < min_overlap) {
                    min_overlap = overlap;
                    best_axis = b.axis[i];
                }
            }

            // 3) Check cross products
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    Vec3C cross_axis = a.axis[i].cross(b.axis[j]);
                    FloatC len = cross_axis.length();
                    if (len < 1e-12f) continue;  // skip degenerate

                    cross_axis = cross_axis / len;
                    FloatC overlap = _overlapOnAxis(a, b, cross_axis, center_diff);
                    if (overlap < FloatC(0)) return false;
                    if (overlap < min_overlap) {
                        min_overlap = overlap;
                        best_axis = cross_axis;
                    }
                }
            }

            // If no separating axis, contact
            contact_normal = best_axis;
            penetration_depth = min_overlap;
            return true;
        }

        /// @brief Overlap test on one axis
        HOST_DEVICE_INLINE_QUALIFIER static FloatC
        _overlapOnAxis(const OBB &a, const OBB &b, const Vec3C &axis, const Vec3C &center_diff) {
            FloatC dist = RS_ABS(center_diff.dot(axis));
            FloatC ra = _projectionRange(a, axis);
            FloatC rb = _projectionRange(b, axis);

            return ra + rb - dist;  // negative => no overlap
        }

        /// @brief Computes projection range (half extent) of an OBB on a normalized axis
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

    /// Returns the nominal size of this voxel accounting for any
    /// specified temperature and external actuation. Specifically, returns
    /// the zero-stress size of the voxel if all forces/moments were removed.
    HOST_DEVICE_INLINE_QUALIFIER static FloatC baseSize(FloatC vm_voxel_size, FloatC v_expansion) {
        return vm_voxel_size * (FloatC(1) + v_expansion);
    }

    /// Returns the transverse area of this voxel with respect
    /// to the specified axis. This would normally be called
    /// only internally, but can be used to calculate the
    /// correct relationship between force and stress for this
    /// voxel if Poisson's ratio is non-zero.
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

    /// Returns the sum of the current strain of this voxel in
    /// the two mutually perpendicular axis to the specified
    /// axis. This would normally be called only internally, but
    /// can be used to correctly calculate stress for this voxel
    /// if Poisson's ratio is non-zero.
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

    /// Returns the interference (in meters) between the contact envelope
    /// of this voxel and the floor at Z=0. >=0 numbers correspond to
    /// interference. If the voxel is not touching the floor return a value < 0.
    HOST_DEVICE_INLINE_QUALIFIER static FloatC
    floorPenetration(FloatC v_voxel_size, FloatC v_position_z, FloatC floor_z = 0) {
        return v_voxel_size * FloatC(0.5) - (v_position_z - floor_z);
    }

    /// Samples the floor under a world position. Returns true when a floor exists (flat or
    /// array) and writes its height (Z) and up-normal; returns false for RSE_FLOOR_DATA_NONE.
    /// Flat floors are at Z=0 with a +Z normal; array floors read the precomputed cell under
    /// the position (out-of-bounds cells fall back to a +Z normal at Z=0).
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool floorHeightAndNormal(
        Kernel *k,
        const Vec3C &position,
        FloatC &floor_height_out,
        Vec3C &normal_out
    ) {
        auto &ctx = k->ctx;
        RSE_FloorDataType floor_data_type = k->floor_data_type;
        if (floor_data_type != RSE_FLOOR_DATA_FLAT && floor_data_type != RSE_FLOOR_DATA_ARRAY)
            return false;

        floor_height_out = FloatC(0);
        normal_out = {0, 0, 1};
        if (floor_data_type == RSE_FLOOR_DATA_ARRAY) {
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

    /// Returns the 3D linear linearVelocity of this voxel in m/s (GCS)
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    linearVelocity(const Vec3C &v_linear_momentum, FloatC vm_mass_inverse) {
        return v_linear_momentum * vm_mass_inverse;
    }

    /// Returns the 3D angular linearVelocity of this voxel in rad/s (GCS)
    HOST_DEVICE_INLINE_QUALIFIER static Vec3C
    angularVelocity(const Vec3C &v_angular_momentum, FloatC vm_moment_inertia_inverse) {
        return v_angular_momentum * vm_moment_inertia_inverse;
    }

   private:
    /// Compose BVH leaf metadata from a voxel's state and its vision material's
    /// render type; no prebake, so material render-type changes propagate at
    /// the next rebuild/refit.
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static RState
    _composeVoxelBVHLeafMetadata(Kernel *k, RHandle voxel) {
        auto &ctx = k->ctx;
        RS_VoxelStateUnion s_union{V_G(state)};
        RHandle vision_material = V_G(vision_material);
        RSE_VisionMaterialType render_type = RSE_VISION_MATERIAL_OPAQUE;
        if (vision_material != RS_NULL_HANDLE) {
            RS_VisionMaterialStateUnion vm_union{VSM_G(vision_material, state)};
            render_type = vm_union.vision_material_state.render_type;
        }
        RS_VoxelBVHLeafMetadataUnion m_union{0};
        m_union.meta.is_enabled = s_union.voxel_state.is_enabled;
        m_union.meta.is_surface = s_union.voxel_state.is_surface;
        m_union.meta.is_rigid = s_union.voxel_state.is_rigid;
        m_union.meta.is_fixed = s_union.voxel_state.is_fixed;
        m_union.meta.is_emissive = (render_type == RSE_VISION_MATERIAL_EMISSIVE) ? 1 : 0;
        m_union.meta.is_transparent = (render_type == RSE_VISION_MATERIAL_TRANSPARENT) ? 1 : 0;
        m_union.meta.is_render_active = isRenderActive(s_union.voxel_state_packed) ? 1 : 0;
        m_union.meta.is_contact_active = isContactActive(s_union.voxel_state_packed) ? 1 : 0;
        return m_union.meta_packed;
    }

    /// Compose the fat voxel-BVH leaf identifier: metadata plus the contact
    /// broad-phase payload.
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static RS_VoxelBVHObjectIdentifier
    _composeVoxelBVHLeafIdentifier(Kernel *k, RHandle voxel) {
        auto &ctx = k->ctx;
        RS_VoxelBVHObjectIdentifier oid;
        oid.object_metadata = _composeVoxelBVHLeafMetadata(k, voxel);
        oid.object_handle = voxel;
        oid.contact_filter_key = V_G(contact_filter_key);
        oid.position = RVec3f(Vec3C(V_G(position)));
        oid.radius = float(V_G(radius));
        oid.structure_gid = V_G(structure_gid);
        return oid;
    }

    /// Loose world-AABB enclosing a voxel's rotated OBB.
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static RBound3f _computeVoxelLooseAABB(Kernel *k, RHandle voxel) {
        auto &ctx = k->ctx;
        Vec3C position = V_G(position);
        Quat3C orientation = V_G(orientation);
        Vec3C nnn = V_G(nnn_offset);
        Vec3C ppp = V_G(ppp_offset);
        Vec3C min_b, max_b;
        bool first = true;
        for (int i = 0; i < 8; i++) {
            Vec3C local((i & 1) ? ppp.x : nnn.x, (i & 2) ? ppp.y : nnn.y, (i & 4) ? ppp.z : nnn.z);
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
};

}  // namespace logic

}  // namespace rise

#endif  // RS_VOXEL_C_H