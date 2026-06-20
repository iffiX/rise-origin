// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONTACT_C_H
#define RS_CONTACT_C_H

#include "rise_origin/device/rs_dev_util.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/utils/rs_conf.h"
#include "rs_contact_m.h"
#include "rs_rigid_body_m.h"

namespace rise {

namespace logic {

/// Canonical owner-pair key, smaller handle in the high word so a-b and b-a hash identically.
HOST_DEVICE_INLINE_QUALIFIER RS_RigidContactKey makeContactKey(RHandle owner_a, RHandle owner_b) {
    RHandle lo = owner_a < owner_b ? owner_a : owner_b;
    RHandle hi = owner_a < owner_b ? owner_b : owner_a;
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
}

/**
 * @brief Manifold-reduced flat rigid contact solver.
 *
 * @details Each surface voxel reports up to three raw contacts, two voxel-voxel slots and one floor
 * slot. The raw contacts are counted, prefix summed, and emitted compacted into a flat list, then
 * sorted by owner-pair key and reduced to a few spread points per rigid body pair. The reduction
 * also collapses the reciprocal contact emitted by the partner voxel, since both share the same
 * owner-pair key and very close geometry. The Jacobi velocity solve reads the reduced list and
 * applies an equal and opposite impulse to both owners, which keeps the solve symmetric and
 * resolves the large-versus-many-small voxel case without a per-voxel slot cap. Warm-start lambda
 * persists on the source voxel across steps. A split-impulse position correction pass runs after
 * the velocity solve and corrects center of mass and orientation without touching velocity:
 *
 *     CountRigidContacts -> prefix sum -> EmitRigidContacts          (compacted raw contacts)
 *     GetRigidContactKeys -> sort -> ReorderRigidContacts            (group same owner pair)
 *     BuildRigidContactManifold -> prefix sum -> CompactSelected     (K spread points per pair)
 *     ResetRigidBodyContactCount -> AccumulateRigidContactCount      (normalizer input)
 *     JacobiBuildRigidContact                                        (k_n, warm-start seed + apply)
 *     for it in 0 .. rigidbody_contact_iterations: JacobiUpdate      (symmetric atomicAdd)
 *     WritebackRigidContactLambda                                    (persist lambda on voxel)
 *     SnapshotPositionCorrectionCom
 *     for it in 0 .. position_correction_iterations: Solve           (split impulse, com + orient)
 *
 * Symbols, shared by the kernels below and matching the joint PGS solver in RS_ConstraintCompute:
 *   a, b          the two rigid bodies; b is RS_NULL_HANDLE for a floor contact
 *   v_a, v_b      linear velocity of a / b                       m/s
 *   w_a, w_b      angular velocity of a / b                      rad/s
 *   mass_inv_a/b  inverse mass of a / b                          kg^-1
 *   I_inv_a/b     inverse world-frame inertia of a / b           kg^-1 m^-2
 *   r_a, r_b      contact point - com of a / b                   m
 *   n             contact normal, unit, pointing from b toward a
 *   k_n           effective inverse mass K_a + K_b times the count normalizer
 *   lambda        accumulated constraint impulse along J^T       kg m/s
 */
template <RSE_FloatType FloatComputeAccuracy>
class RS_ContactCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using Mat3C = RS_Mat3D<FloatC>;
    using VoxelCompute = RS_VoxelCompute<FloatComputeAccuracy>;

    /**
     * @brief Per surface voxel: flag which of its three contact slots are live this step.
     *
     * @details Reads the packed voxel state and the per-slot penetration depth. The flag predicate
     * must match RSCF_EmitRigidContacts exactly, since the prefix sum of these flags reserves the
     * compacted output slot that the emit pass then fills.
     *
     * @param k                        simulation kernel.
     * @param flags                    per voxel-slot flag buffer, three entries per surface voxel.
     * @param rigid_body_surface_voxel handle of the surface voxel being flagged.
     * @param dense_index              dense index of this surface voxel in the launch.
     */
    class RSCF_CountRigidContacts : public device::RS_ComputeFunction<RSCF_CountRigidContacts> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_CountRigidContacts>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, RHandle rigid_body_surface_voxel, RIndex dense_index, RS_MemoryPtr flags) {
            if (k == nullptr) return;
            auto &ctx = k->ctx;
            RIndex base = dense_index * RIndex(3);
            flags.template get<RIndex>(base + RIndex(0)) = 0;
            flags.template get<RIndex>(base + RIndex(1)) = 0;
            flags.template get<RIndex>(base + RIndex(2)) = 0;

            if (rigid_body_surface_voxel == RS_NULL_HANDLE) return;
            RHandle voxel = RBSV_G(voxel);
            if (voxel == RS_NULL_HANDLE) return;

            RS_VoxelStateUnion su{V_G(voxel, state)};
            bool slot_0 = su.voxel_state.has_contact_voxel_0 &&
                          su.voxel_state.is_contact_voxel_0_rigid_or_static &&
                          FloatC(V_G(voxel, contact_depth_0)) > FloatC(0);
            bool slot_1 = su.voxel_state.has_contact_voxel_1 &&
                          su.voxel_state.is_contact_voxel_1_rigid_or_static &&
                          FloatC(V_G(voxel, contact_depth_1)) > FloatC(0);
            bool floor = su.voxel_state.has_contact_floor;
            flags.template get<RIndex>(base + RIndex(0)) = slot_0 ? 1 : 0;
            flags.template get<RIndex>(base + RIndex(1)) = slot_1 ? 1 : 0;
            flags.template get<RIndex>(base + RIndex(2)) = floor ? 1 : 0;
        }
    };

    /**
     * @brief Per surface voxel: emit each live contact straight into the compacted list.
     *
     * @details For every slot flagged live by RSCF_CountRigidContacts, writes one record at the
     * prefix sum offset. Both voxels of a pair emit; the reciprocal duplicate is dropped later by
     * the canonical voxel-pair dedup. The floor anchor is the penetration-weighted centroid of the
     * eight oriented bounding box corners below the floor plane, which is the bottom face center
     * for a flat resting voxel and shifts toward the deep side when tilted to give a restoring
     * lever.
     *
     * @param k                        simulation kernel.
     * @param flags                    per voxel-slot flag buffer from RSCF_CountRigidContacts.
     * @param rigid_body_surface_voxel handle of the surface voxel emitting contacts.
     * @param dense_index              dense index of this surface voxel in the launch.
     * @param contacts                 compacted rigid contact list, the emit destination.
     * @param prefix                   inclusive prefix sum of flags, the compacted slot offsets.
     */
    class RSCF_EmitRigidContacts : public device::RS_ComputeFunction<RSCF_EmitRigidContacts> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_EmitRigidContacts>;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            RHandle rigid_body_surface_voxel,
            RIndex dense_index,
            RS_MemoryPtr flags,
            Contacts contacts,
            RS_MemoryPtr prefix
        ) {
            if (k == nullptr) return;
            auto &ctx = k->ctx;
            using C = typename Contacts::ObjectType;
            RIndex base = dense_index * RIndex(3);
            if (rigid_body_surface_voxel == RS_NULL_HANDLE) return;
            RHandle voxel = RBSV_G(voxel);
            RHandle body = RBSV_G(rigid_body);
            if (voxel == RS_NULL_HANDLE || body == RS_NULL_HANDLE) return;

            RHandle voxel_material = V_G(voxel, voxel_material);
            RIndex mat_gid_a = VM_G(voxel_material, material_gid);

            if (flags.template get<RIndex>(base + RIndex(0))) {
                _emitRigidSlot(
                    k,
                    contacts,
                    prefix.template get<RIndex>(base + RIndex(0)) - 1,
                    voxel,
                    body,
                    mat_gid_a,
                    RState(0),
                    V_G(voxel, contact_voxel_0),
                    Vec3C(V_G(voxel, contact_normal_0)),
                    Vec3C(V_G(voxel, contact_position_0))
                );
            }
            if (flags.template get<RIndex>(base + RIndex(1))) {
                _emitRigidSlot(
                    k,
                    contacts,
                    prefix.template get<RIndex>(base + RIndex(1)) - 1,
                    voxel,
                    body,
                    mat_gid_a,
                    RState(1),
                    V_G(voxel, contact_voxel_1),
                    Vec3C(V_G(voxel, contact_normal_1)),
                    Vec3C(V_G(voxel, contact_position_1))
                );
            }
            if (flags.template get<RIndex>(base + RIndex(2))) {
                _emitFloorSlot(
                    k,
                    contacts,
                    prefix.template get<RIndex>(base + RIndex(2)) - 1,
                    voxel,
                    body,
                    mat_gid_a
                );
            }
        }

       private:
        /// Build one rigid-rigid contact record from a voxel slot into the compacted list.
        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void _emitRigidSlot(
            Kernel *k,
            Contacts contacts,
            RIndex out,
            RHandle voxel,
            RHandle body,
            RIndex mat_gid_a,
            RState slot,
            RHandle other_voxel,
            const Vec3C &normal,
            const Vec3C &anchor
        ) {
            auto &ctx = k->ctx;
            using C = typename Contacts::ObjectType;
            RHandle other_body = other_voxel == RS_NULL_HANDLE ? RS_NULL_HANDLE
                                                               : V_G(other_voxel, rigid_body);
            RIndex mat_gid_b = other_voxel == RS_NULL_HANDLE
                                   ? mat_gid_a
                                   : VM_G(V_G(other_voxel, voxel_material), material_gid);
            FloatC mu =
                FloatC(ctx.material_friction_dynamic_table[{(RSize)mat_gid_a, (RSize)mat_gid_b}]);
            FloatC depth = slot == 0 ? FloatC(V_G(voxel, contact_depth_0))
                                     : FloatC(V_G(voxel, contact_depth_1));

            C contact;
            contact.voxel_a = voxel;
            contact.voxel_b = other_voxel;
            contact.rigid_body_a = body;
            contact.rigid_body_b = other_body;
            RS_RigidContactStateUnion st{};
            st.contact_state.is_active = 1;
            st.contact_state.slot = slot;
            contact.state = st.contact_state_packed;
            contact.contact_point = anchor;
            contact.contact_normal = normal;
            contact.depth = depth;
            contact.mu = mu;
            contacts.set(out, contact);
        }

        /// Build the floor contact record at the penetration-weighted corner centroid.
        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void _emitFloorSlot(
            Kernel *k,
            Contacts contacts,
            RIndex out,
            RHandle voxel,
            RHandle body,
            RIndex mat_gid_a
        ) {
            auto &ctx = k->ctx;
            using C = typename Contacts::ObjectType;
            RIndex mat_gid_b = ctx.floor.material_gid;
            FloatC mu =
                FloatC(ctx.material_friction_dynamic_table[{(RSize)mat_gid_a, (RSize)mat_gid_b}]);
            Vec3C voxel_pos = V_G(voxel, position);
            FloatC floor_height;
            Vec3C floor_normal(FloatC(0), FloatC(0), FloatC(1));
            Vec3C contact_point = voxel_pos;
            FloatC depth = 0;
            if (VoxelCompute::floorHeightAndNormal(k, voxel_pos, floor_height, floor_normal)) {
                Quat3C ori = V_G(voxel, orientation);
                Vec3C nnn = V_G(voxel, nnn_offset);
                Vec3C ppp = V_G(voxel, ppp_offset);
                Vec3C plane_pt(voxel_pos.x, voxel_pos.y, floor_height);
                Vec3C weighted = Vec3C::zero();
                FloatC weight = 0;
                for (int c = 0; c < 8; ++c) {
                    Vec3C local(
                        (c & 1) ? ppp.x : nnn.x, (c & 2) ? ppp.y : nnn.y, (c & 4) ? ppp.z : nnn.z
                    );
                    Vec3C world = voxel_pos + ori.rotateVec3D(local);
                    FloatC pen = -(world - plane_pt).dot(floor_normal);
                    if (pen > FloatC(0)) {
                        weighted += world * pen;
                        weight += pen;
                        if (pen > depth) depth = pen;
                    }
                }
                if (weight > FloatC(0)) contact_point = weighted * (FloatC(1) / weight);
            }

            C contact;
            contact.voxel_a = voxel;
            contact.voxel_b = RS_NULL_HANDLE;
            contact.rigid_body_a = body;
            contact.rigid_body_b = RS_NULL_HANDLE;
            RS_RigidContactStateUnion st{};
            st.contact_state.is_active = 1;
            st.contact_state.slot = 2;
            contact.state = st.contact_state_packed;
            contact.contact_point = contact_point;
            contact.contact_normal = floor_normal;
            contact.depth = depth;
            contact.mu = mu;
            contacts.set(out, contact);
        }
    };

    /**
     * @brief Per dense contact: extract the voxel-pair sort key and the identity index sequence.
     *
     * @details The key is canonical over the two source voxels, so a contact and its reciprocal,
     * emitted from the partner voxel at the same location, hash to the same key and become adjacent
     * after the sort. The mark-and-compact then keeps one of each reciprocal pair.
     *
     * @param k        simulation kernel.
     * @param contacts compacted rigid contact list.
     * @param keys     output voxel-pair sort keys.
     * @param indices  output identity index sequence the sort permutes alongside the keys.
     */
    class RSCF_GetRigidContactVoxelPairKeys
        : public device::RS_ComputeFunction<RSCF_GetRigidContactVoxelPairKeys> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_GetRigidContactVoxelPairKeys>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, Contacts contacts, RS_MemoryPtr keys, RS_MemoryPtr indices) {
            if (k == nullptr) return;
            using C = typename Contacts::ObjectType;
            RIndex i = index1D();
            RHandle va = contacts.template get<Pr(C, voxel_a)>(i);
            RHandle vb = contacts.template get<Pr(C, voxel_b)>(i);
            keys.template get<RS_RigidContactKey>(i) = makeContactKey(va, vb);
            indices.template get<RIndex>(i) = i;
        }
    };

    /**
     * @brief Per dense contact: extract the owner-pair sort key and the identity index sequence.
     *
     * @param k        simulation kernel.
     * @param contacts compacted rigid contact list.
     * @param keys     output owner-pair sort keys.
     * @param indices  output identity index sequence the sort permutes alongside the keys.
     */
    class RSCF_GetRigidContactKeys : public device::RS_ComputeFunction<RSCF_GetRigidContactKeys> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_GetRigidContactKeys>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, Contacts contacts, RS_MemoryPtr keys, RS_MemoryPtr indices) {
            if (k == nullptr) return;
            using C = typename Contacts::ObjectType;
            RIndex i = index1D();
            RHandle oa = contacts.template get<Pr(C, rigid_body_a)>(i);
            RHandle ob = contacts.template get<Pr(C, rigid_body_b)>(i);
            keys.template get<RS_RigidContactKey>(i) = makeContactKey(oa, ob);
            indices.template get<RIndex>(i) = i;
        }
    };

    /**
     * @brief Per sorted slot: gather contacts[sorted_index] into the sorted output list.
     *
     * @param k              simulation kernel.
     * @param in_contacts    dense contact list to gather from.
     * @param out_contacts   sorted contact list to gather into.
     * @param sorted_indices the sort permutation produced alongside the keys.
     */
    class RSCF_ReorderRigidContacts : public device::RS_ComputeFunction<RSCF_ReorderRigidContacts> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ReorderRigidContacts>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, Contacts in_contacts, Contacts out_contacts, RS_MemoryPtr sorted_indices) {
            if (k == nullptr) return;
            RIndex i = index1D();
            out_contacts.set(i, in_contacts.get(sorted_indices.template get<RIndex>(i)));
        }
    };

    /**
     * @brief Per sorted contact: flag the first record of each sorted-key run.
     *
     * @details Used both to drop reciprocal duplicates after the voxel-pair sort and to mark
     * owner-pair group starts after the owner-pair sort. A flagged run start is the only record
     * kept by the following compaction, or the boundary that begins a new reduce group.
     *
     * @param k           simulation kernel.
     * @param sorted_keys keys in sorted order.
     * @param flags       output flag buffer, one per contact.
     * @param n           number of contacts.
     */
    class RSCF_FlagRigidContactKeyRunStart
        : public device::RS_ComputeFunction<RSCF_FlagRigidContactKeyRunStart> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_FlagRigidContactKeyRunStart>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, RS_MemoryPtr sorted_keys, RS_MemoryPtr flags, RIndex n) {
            if (k == nullptr) return;
            RIndex i = index1D();
            if (i >= n) return;
            RS_RigidContactKey key = sorted_keys.template get<RS_RigidContactKey>(i);
            bool first = (i == 0) || sorted_keys.template get<RS_RigidContactKey>(i - 1) != key;
            flags.template get<RIndex>(i) = first ? 1 : 0;
        }
    };

    /**
     * @brief Per sorted contact: compact group-start indices into a packed offset list.
     *
     * @param k            simulation kernel.
     * @param starts       run-start flags from RSCF_FlagRigidContactKeyRunStart.
     * @param group_prefix inclusive prefix sum of starts, the group ids plus one.
     * @param group_starts output packed start index of each group.
     */
    class RSCF_CompactRigidManifoldGroupStart
        : public device::RS_ComputeFunction<RSCF_CompactRigidManifoldGroupStart> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_CompactRigidManifoldGroupStart>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, RS_MemoryPtr starts, RS_MemoryPtr group_prefix, RS_MemoryPtr group_starts) {
            if (k == nullptr) return;
            RIndex i = index1D();
            if (starts.template get<RIndex>(i)) {
                RIndex g = group_prefix.template get<RIndex>(i) - 1;
                group_starts.template get<RIndex>(g) = i;
            }
        }
    };

    /**
     * @brief Per sorted contact: extract its depth and point into the deepest-reduce input.
     *
     * @param k        simulation kernel.
     * @param contacts sorted contact list.
     * @param deepest  output deepest-reduce input buffer.
     */
    class RSCF_GetRigidManifoldDepth
        : public device::RS_ComputeFunction<RSCF_GetRigidManifoldDepth> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_GetRigidManifoldDepth>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, Contacts contacts, RS_MemoryPtr deepest) {
            if (k == nullptr) return;
            using C = typename Contacts::ObjectType;
            RIndex i = index1D();
            RS_RigidManifoldDeepest e;
            e.depth = float(contacts.template get<Pr(C, depth)>(i));
            e.point = RVec3f(Vec3C(contacts.template get<Pr(C, contact_point)>(i)));
            e.index = i;
            deepest.template get<RS_RigidManifoldDeepest>(i) = e;
        }
    };

    /**
     * @brief Per group: flag the deepest contact as the manifold anchor.
     *
     * @param k            simulation kernel.
     * @param deepest_out  per-group deepest-reduce result.
     * @param flags        manifold output flag buffer.
     */
    class RSCF_SetRigidManifoldDeepest
        : public device::RS_ComputeFunction<RSCF_SetRigidManifoldDeepest> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_SetRigidManifoldDeepest>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, RS_MemoryPtr deepest_out, RS_MemoryPtr flags) {
            if (k == nullptr) return;
            RIndex g = index1D();
            RIndex index = deepest_out.template get<RS_RigidManifoldDeepest>(g).index;
            if (index != RS_NULL_INDEX) flags.template get<RIndex>(index) = 1;
        }
    };

    /**
     * @brief Per sorted contact: extract its squared distance to the group's deepest contact.
     *
     * @param k            simulation kernel.
     * @param contacts     sorted contact list.
     * @param group_prefix inclusive prefix sum of group starts, the group ids plus one.
     * @param deepest_out  per-group deepest-reduce result, supplying the anchor point.
     * @param farthest     output farthest-reduce input buffer, a singleton top-one per contact.
     */
    class RSCF_GetRigidManifoldDistance
        : public device::RS_ComputeFunction<RSCF_GetRigidManifoldDistance> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_GetRigidManifoldDistance>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            Contacts contacts,
            RS_MemoryPtr group_prefix,
            RS_MemoryPtr deepest_out,
            RS_MemoryPtr farthest
        ) {
            if (k == nullptr) return;
            using C = typename Contacts::ObjectType;
            RIndex i = index1D();
            RIndex g = group_prefix.template get<RIndex>(i) - 1;
            Vec3C anchor = Vec3C(deepest_out.template get<RS_RigidManifoldDeepest>(g).point);
            Vec3C point = Vec3C(contacts.template get<Pr(C, contact_point)>(i));
            RS_RigidManifoldFarthest e = RS_RigidManifoldFarthest::empty();
            e.dist[0] = float((point - anchor).length2());
            e.index[0] = i;
            farthest.template get<RS_RigidManifoldFarthest>(i) = e;
        }
    };

    /**
     * @brief Per group: flag the top-three farthest contacts, skipping the deepest if present.
     *
     * @param k            simulation kernel.
     * @param farthest_out per-group farthest-reduce result.
     * @param deepest_out  per-group deepest-reduce result, whose index is already flagged.
     * @param flags        manifold output flag buffer.
     */
    class RSCF_SetRigidManifoldFarthest
        : public device::RS_ComputeFunction<RSCF_SetRigidManifoldFarthest> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_SetRigidManifoldFarthest>;
        using BaseType::index1D;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel *k, RS_MemoryPtr farthest_out, RS_MemoryPtr deepest_out, RS_MemoryPtr flags) {
            if (k == nullptr) return;
            RIndex g = index1D();
            RIndex deepest_index = deepest_out.template get<RS_RigidManifoldDeepest>(g).index;
            RS_RigidManifoldFarthest e = farthest_out.template get<RS_RigidManifoldFarthest>(g);
            for (int s = 0; s < 3; ++s) {
                if (e.index[s] != RS_NULL_INDEX && e.index[s] != deepest_index)
                    flags.template get<RIndex>(e.index[s]) = 1;
            }
        }
    };

    /**
     * @brief Per sorted slot: compact the manifold-selected records into the dense list.
     *
     * @param k            simulation kernel.
     * @param in_contacts  sorted contact list to compact from.
     * @param out_contacts dense contact list to compact into.
     * @param flags        flag buffer marking the kept records.
     * @param prefix       inclusive prefix sum of flags, the compacted slot offsets.
     */
    class RSCF_CompactSelectedRigidContacts
        : public device::RS_ComputeFunction<RSCF_CompactSelectedRigidContacts> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_CompactSelectedRigidContacts>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void run(
            Kernel *k,
            Contacts in_contacts,
            Contacts out_contacts,
            RS_MemoryPtr flags,
            RS_MemoryPtr prefix
        ) {
            if (k == nullptr) return;
            RIndex i = index1D();
            if (flags.template get<RIndex>(i)) {
                RIndex out = prefix.template get<RIndex>(i) - 1;
                out_contacts.set(out, in_contacts.get(i));
            }
        }
    };

    /**
     * @brief Per rigid body: reset this step's contact count, the manifold normalizer input.
     *
     * @param k          simulation kernel.
     * @param rigid_body handle of the rigid body whose count is reset.
     */
    class RSCF_ResetRigidBodyContactCount
        : public device::RS_ComputeFunction<RSCF_ResetRigidBodyContactCount> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ResetRigidBodyContactCount>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body) {
            if (k == nullptr || rigid_body == RS_NULL_HANDLE) return;
            auto &ctx = k->ctx;
            RB_S(rigid_body, contact_count_now_step, RSize(0));
        }
    };

    /**
     * @brief Per selected contact: atomically add one to each owner's contact count.
     *
     * @param k        simulation kernel.
     * @param contacts reduced rigid contact list.
     */
    class RSCF_AccumulateRigidContactCount
        : public device::RS_ComputeFunction<RSCF_AccumulateRigidContactCount> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_AccumulateRigidContactCount>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, Contacts contacts) {
            if (k == nullptr) return;
            auto &ctx = k->ctx;
            using C = typename Contacts::ObjectType;
            RIndex i = index1D();
            RHandle oa = contacts.template get<Pr(C, rigid_body_a)>(i);
            RHandle ob = contacts.template get<Pr(C, rigid_body_b)>(i);
            if (oa != RS_NULL_HANDLE)
                device::deviceAtomicAdd(&RB_G(oa, contact_count_now_step), RSize(1));
            if (ob != RS_NULL_HANDLE)
                device::deviceAtomicAdd(&RB_G(ob, contact_count_now_step), RSize(1));
        }
    };

    /**
     * @brief Per selected contact: precompute the effective mass and count normalizer, seed the
     * warm-start lambda from the source voxel slot, and apply that seed to both owners.
     *
     * @details Folding the count normalizer into k_n under-relaxes a body touched by many contacts.
     * Applying the seeded impulse makes the bodies start the step already carrying last step's
     * resting load, so the iterations only refine via deltas. Without the apply each contact would
     * silently shed its resting force every step and a stack would collapse.
     *
     * @param k        simulation kernel.
     * @param contacts reduced rigid contact list.
     */
    class RSCF_JacobiBuildRigidContact
        : public device::RS_ComputeFunction<RSCF_JacobiBuildRigidContact> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_JacobiBuildRigidContact>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, Contacts contacts) {
            if (k == nullptr) return;
            auto &ctx = k->ctx;
            using C = typename Contacts::ObjectType;
            RIndex i = index1D();

            RHandle owner_a = contacts.template get<Pr(C, rigid_body_a)>(i);
            RHandle owner_b = contacts.template get<Pr(C, rigid_body_b)>(i);
            bool is_floor = (owner_b == RS_NULL_HANDLE);
            Vec3C n = Vec3C(contacts.template get<Pr(C, contact_normal)>(i));
            Vec3C cp = Vec3C(contacts.template get<Pr(C, contact_point)>(i));

            bool static_a = RS_RigidBodyStateUnion{RB_G(owner_a, state)}.rigid_body_state.is_static;
            Vec3C com_a = RB_G(owner_a, com);
            FloatC mass_inv_a = static_a ? FloatC(0) : FloatC(RB_G(owner_a, mass_inverse));
            Mat3C I_inv_a = static_a ? Mat3C::zero() : Mat3C(RB_G(owner_a, moment_inertia_inverse));
            Vec3C r_a = cp - com_a;
            FloatC K = contactEffectiveMassInverse(mass_inv_a, I_inv_a, r_a, n);

            bool static_b =
                is_floor ? true
                         : RS_RigidBodyStateUnion{RB_G(owner_b, state)}.rigid_body_state.is_static;
            Vec3C com_b = (is_floor || static_b) ? Vec3C::zero() : Vec3C(RB_G(owner_b, com));
            FloatC mass_inv_b = static_b ? FloatC(0) : FloatC(RB_G(owner_b, mass_inverse));
            Mat3C I_inv_b = static_b ? Mat3C::zero() : Mat3C(RB_G(owner_b, moment_inertia_inverse));
            Vec3C r_b = static_b ? Vec3C::zero() : (cp - com_b);
            if (!static_b) K += contactEffectiveMassInverse(mass_inv_b, I_inv_b, r_b, n);

            RSize count_a = RB_G(owner_a, contact_count_now_step);
            RSize count_b = static_b ? RSize(0) : RB_G(owner_b, contact_count_now_step);
            FloatC normalizer = FloatC(RS_MAX(RS_MAX(count_a, count_b), RSize(1)));
            contacts.template set<Pr(C, k_n)>(i, K * normalizer);
            contacts.template set<Pr(C, normalizer)>(i, normalizer);

            RHandle voxel = contacts.template get<Pr(C, voxel_a)>(i);
            RS_RigidContactStateUnion st{contacts.template get<Pr(C, state)>(i)};
            FloatC seed_scale = FloatC(1) - k->rigidbody_contact_warm_start_damping_z;
            FloatC lambda_n = 0;
            Vec3C lambda_t = Vec3C::zero();
            if (st.contact_state.slot == 0) {
                lambda_n = FloatC(V_G(voxel, contact_lambda_n_0));
                lambda_t = Vec3C(V_G(voxel, contact_lambda_t_0));
            } else if (st.contact_state.slot == 1) {
                lambda_n = FloatC(V_G(voxel, contact_lambda_n_1));
                lambda_t = Vec3C(V_G(voxel, contact_lambda_t_1));
            } else {
                lambda_n = FloatC(V_G(voxel, contact_lambda_n_floor));
                lambda_t = Vec3C(V_G(voxel, contact_lambda_t_floor));
            }
            lambda_n *= seed_scale;
            lambda_t = lambda_t * seed_scale;
            lambda_t = lambda_t - n * lambda_t.dot(n);
            contacts.template set<Pr(C, accumulated_lambda_n)>(i, lambda_n);
            contacts.template set<Pr(C, accumulated_lambda_t)>(i, lambda_t);

            if (lambda_n > FloatC(0) || lambda_t.length2() > FloatC(0)) {
                Vec3C imp = lambda_n * n + lambda_t;
                if (!static_a) {
                    device::deviceAtomicAdd(&RB_G(owner_a, linear_momentum), imp);
                    device::deviceAtomicAdd(&RB_G(owner_a, angular_momentum), r_a.cross(imp));
                }
                if (!static_b) {
                    device::deviceAtomicAdd(&RB_G(owner_b, linear_momentum), -imp);
                    device::deviceAtomicAdd(&RB_G(owner_b, angular_momentum), r_b.cross(-imp));
                }
            }
        }
    };

    /**
     * @brief Per selected contact: incremental PGS update of the contact's lambda accumulator.
     *
     * @details Reads the stored `(lambda_n, lambda_t)`, clamps the cumulative accumulators, writes
     * them back, and atomically emits `J^T * delta_lambda_eff` as equal and opposite to both
     * owners. `delta_lambda_eff` can be negative, since iterations can reduce the warm-start seed
     * left from the previous step. Position drift is corrected by the separate split-impulse pass,
     * so velocity has no Baumgarte term here. With `J = [n, r_a x n]` and `k_n = K * normalizer`:
     *
     *     K              =  K_a + K_b,   K_x = mass_inv_x + (r_x x n) . I_inv_x . (r_x x n)
     *     v_n            =  (v_a - v_b) . n
     *     v_target       =  restitution . max(-v_n, 0)
     *     delta_n_raw    =  (v_target - v_n) / k_n
     *     lambda_n_new   =  max[ 0 , lambda_n_old + delta_n_raw ]
     *     delta_n_eff    =  lambda_n_new - lambda_n_old
     *
     *     delta_t_raw    =  -v_t / K_t                       (current tangent direction)
     *     lambda_t_new   =  cone_clip( lambda_t_old + delta_t_raw , mu * lambda_n_new )
     *     delta_t_eff    =  lambda_t_new - lambda_t_old
     *
     * `lambda_t_old` is first projected onto the current tangent plane so a shifted normal does not
     * leak into friction. `K_t` is also folded with the count normalizer.
     *
     * @param k        simulation kernel.
     * @param contacts reduced rigid contact list.
     */
    class RSCF_JacobiUpdateRigidContact
        : public device::RS_ComputeFunction<RSCF_JacobiUpdateRigidContact> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_JacobiUpdateRigidContact>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, Contacts contacts) {
            if (k == nullptr) return;
            auto &ctx = k->ctx;
            using C = typename Contacts::ObjectType;
            RIndex i = index1D();

            RHandle owner_a = contacts.template get<Pr(C, rigid_body_a)>(i);
            RHandle owner_b = contacts.template get<Pr(C, rigid_body_b)>(i);
            bool is_floor = (owner_b == RS_NULL_HANDLE);

            RHandle voxel_a = contacts.template get<Pr(C, voxel_a)>(i);
            RHandle voxel_b = contacts.template get<Pr(C, voxel_b)>(i);
            RIndex mat_gid_a = VM_G(V_G(voxel_a, voxel_material), material_gid);
            // A null voxel_b is the floor; a fixed or frozen anchor still carries its voxel.
            RIndex mat_gid_b = voxel_b == RS_NULL_HANDLE
                                   ? ctx.floor.material_gid
                                   : VM_G(V_G(voxel_b, voxel_material), material_gid);
            FloatC restitution =
                FloatC(ctx.material_restitution_table[{(RSize)mat_gid_a, (RSize)mat_gid_b}]);

            Vec3C n = Vec3C(contacts.template get<Pr(C, contact_normal)>(i));
            Vec3C cp = Vec3C(contacts.template get<Pr(C, contact_point)>(i));
            FloatC k_n = contacts.template get<Pr(C, k_n)>(i);
            FloatC normalizer = contacts.template get<Pr(C, normalizer)>(i);
            FloatC mu = contacts.template get<Pr(C, mu)>(i);
            if (k_n <= FloatC(0)) return;

            bool static_a = RS_RigidBodyStateUnion{RB_G(owner_a, state)}.rigid_body_state.is_static;
            Vec3C com_a = RB_G(owner_a, com);
            FloatC mass_inv_a = static_a ? FloatC(0) : FloatC(RB_G(owner_a, mass_inverse));
            Mat3C I_inv_a = static_a ? Mat3C::zero() : Mat3C(RB_G(owner_a, moment_inertia_inverse));
            Vec3C v_a = Vec3C(RB_G(owner_a, linear_momentum)) * mass_inv_a;
            Vec3C w_a = I_inv_a * Vec3C(RB_G(owner_a, angular_momentum));
            Vec3C r_a = cp - com_a;

            bool static_b =
                is_floor ? true
                         : RS_RigidBodyStateUnion{RB_G(owner_b, state)}.rigid_body_state.is_static;
            Vec3C com_b = is_floor ? Vec3C::zero() : Vec3C(RB_G(owner_b, com));
            FloatC mass_inv_b = static_b ? FloatC(0) : FloatC(RB_G(owner_b, mass_inverse));
            Mat3C I_inv_b = static_b ? Mat3C::zero() : Mat3C(RB_G(owner_b, moment_inertia_inverse));
            Vec3C r_b = is_floor ? Vec3C::zero() : (cp - com_b);
            Vec3C v_b = static_b ? Vec3C::zero()
                                : Vec3C(RB_G(owner_b, linear_momentum)) * mass_inv_b;
            Vec3C w_b = static_b ? Vec3C::zero()
                                : (I_inv_b * Vec3C(RB_G(owner_b, angular_momentum)));

            FloatC lambda_n = contacts.template get<Pr(C, accumulated_lambda_n)>(i);
            Vec3C lambda_t = Vec3C(contacts.template get<Pr(C, accumulated_lambda_t)>(i));
            lambda_t = lambda_t - n * lambda_t.dot(n);

            Vec3C vel_a = v_a + w_a.cross(r_a);
            Vec3C vel_b = static_b ? Vec3C::zero() : (v_b + w_b.cross(r_b));
            Vec3C v_rel = vel_a - vel_b;
            FloatC v_n = v_rel.dot(n);

            FloatC v_target = restitution * RS_MAX(-v_n, FloatC(0));
            FloatC delta_n = (v_target - v_n) / k_n;
            FloatC lambda_n_new = lambda_n + delta_n;
            if (lambda_n_new < FloatC(0)) lambda_n_new = FloatC(0);
            FloatC delta_n_eff = lambda_n_new - lambda_n;
            lambda_n = lambda_n_new;

            Vec3C imp = delta_n_eff * n;

            Vec3C v_t = v_rel - n * v_n;
            FloatC v_t_len = v_t.length();
            if (v_t_len > FloatC(0)) {
                Vec3C tdir = v_t / v_t_len;
                FloatC K_t = contactEffectiveMassInverse(mass_inv_a, I_inv_a, r_a, tdir);
                if (!static_b) K_t += contactEffectiveMassInverse(mass_inv_b, I_inv_b, r_b, tdir);
                K_t *= normalizer;
                if (K_t > FloatC(0)) {
                    Vec3C lambda_t_new = lambda_t + (-v_t_len / K_t) * tdir;
                    FloatC cone = mu * lambda_n;
                    FloatC len = lambda_t_new.length();
                    if (len > cone)
                        lambda_t_new = len > FloatC(0) ? lambda_t_new * (cone / len)
                                                       : Vec3C::zero();
                    imp += (lambda_t_new - lambda_t);
                    lambda_t = lambda_t_new;
                }
            }

            contacts.template set<Pr(C, accumulated_lambda_n)>(i, lambda_n);
            contacts.template set<Pr(C, accumulated_lambda_t)>(i, lambda_t);

            if (!static_a) {
                device::deviceAtomicAdd(&RB_G(owner_a, linear_momentum), imp);
                device::deviceAtomicAdd(&RB_G(owner_a, angular_momentum), r_a.cross(imp));
            }
            if (!static_b) {
                device::deviceAtomicAdd(&RB_G(owner_b, linear_momentum), -imp);
                device::deviceAtomicAdd(&RB_G(owner_b, angular_momentum), r_b.cross(-imp));
            }
        }
    };

    /**
     * @brief Per selected contact: scatter the accumulated lambda back to the source voxel slot.
     *
     * @param k        simulation kernel.
     * @param contacts reduced rigid contact list.
     */
    class RSCF_WritebackRigidContactLambda
        : public device::RS_ComputeFunction<RSCF_WritebackRigidContactLambda> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_WritebackRigidContactLambda>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, Contacts contacts) {
            if (k == nullptr) return;
            auto &ctx = k->ctx;
            using C = typename Contacts::ObjectType;
            RIndex i = index1D();
            RHandle voxel = contacts.template get<Pr(C, voxel_a)>(i);
            if (voxel == RS_NULL_HANDLE) return;
            RS_RigidContactStateUnion st{contacts.template get<Pr(C, state)>(i)};
            FloatC lambda_n = contacts.template get<Pr(C, accumulated_lambda_n)>(i);
            Vec3C lambda_t = Vec3C(contacts.template get<Pr(C, accumulated_lambda_t)>(i));
            if (st.contact_state.slot == 0) {
                V_S(voxel, contact_lambda_n_0, lambda_n);
                V_S(voxel, contact_lambda_t_0, lambda_t);
            } else if (st.contact_state.slot == 1) {
                V_S(voxel, contact_lambda_n_1, lambda_n);
                V_S(voxel, contact_lambda_t_1, lambda_t);
            } else {
                V_S(voxel, contact_lambda_n_floor, lambda_n);
                V_S(voxel, contact_lambda_t_floor, lambda_t);
            }
        }
    };

    /**
     * @brief Per rigid body: snapshot center of mass at the start of the position correction loop.
     *
     * @param k          simulation kernel.
     * @param rigid_body handle of the rigid body being snapshotted.
     */
    class RSCF_SnapshotPositionCorrectionCom
        : public device::RS_ComputeFunction<RSCF_SnapshotPositionCorrectionCom> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_SnapshotPositionCorrectionCom>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body) {
            if (k == nullptr || rigid_body == RS_NULL_HANDLE) return;
            auto &ctx = k->ctx;
            RB_S(rigid_body, com_at_position_correction_start, Vec3C(RB_G(rigid_body, com)));
        }
    };

    /**
     * @brief Per selected contact: one split-impulse position correction iteration.
     *
     * @details Recomputes penetration from the center of mass displacement since the snapshot, then
     * applies a Baumgarte positional impulse divided by the same count normalizer the velocity
     * solve uses, which keeps the correction resolution independent. The linear part is added to
     * the center of mass atomically; the angular part is composed onto the orientation directly as
     * a small angle rotation. Velocity is not touched.
     *
     * @param k        simulation kernel.
     * @param contacts reduced rigid contact list.
     */
    class RSCF_SolveRigidContactPositionCorrection
        : public device::RS_ComputeFunction<RSCF_SolveRigidContactPositionCorrection> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_SolveRigidContactPositionCorrection>;
        using BaseType::index1D;

        template <typename Kernel, typename Contacts>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, Contacts contacts) {
            if (k == nullptr) return;
            auto &ctx = k->ctx;
            using C = typename Contacts::ObjectType;
            RIndex i = index1D();
            FloatC baumgarte = k->rigidbody_contact_position_correction_ratio;
            FloatC slop = k->rigidbody_contact_slop;

            RHandle owner_a = contacts.template get<Pr(C, rigid_body_a)>(i);
            RHandle owner_b = contacts.template get<Pr(C, rigid_body_b)>(i);
            bool is_floor = (owner_b == RS_NULL_HANDLE);
            Vec3C n = Vec3C(contacts.template get<Pr(C, contact_normal)>(i));
            Vec3C cp = Vec3C(contacts.template get<Pr(C, contact_point)>(i));
            FloatC depth_stored = contacts.template get<Pr(C, depth)>(i);

            bool static_a = RS_RigidBodyStateUnion{RB_G(owner_a, state)}.rigid_body_state.is_static;
            Vec3C com_a = RB_G(owner_a, com);
            Vec3C delta_a = com_a - Vec3C(RB_G(owner_a, com_at_position_correction_start));
            FloatC mass_inv_a = static_a ? FloatC(0) : FloatC(RB_G(owner_a, mass_inverse));
            Mat3C I_inv_a = static_a ? Mat3C::zero()
                                    : worldInertiaInverse(
                                          Mat3C(RB_G(owner_a, local_moment_inertia_inverse)),
                                          RB_G(owner_a, orientation)
                                      );

            bool static_b =
                is_floor ? true
                         : RS_RigidBodyStateUnion{RB_G(owner_b, state)}.rigid_body_state.is_static;
            Vec3C com_b = is_floor ? Vec3C::zero() : Vec3C(RB_G(owner_b, com));
            Vec3C delta_b = (is_floor || static_b)
                                ? Vec3C::zero()
                                : (com_b - Vec3C(RB_G(owner_b, com_at_position_correction_start)));
            FloatC mass_inv_b = static_b ? FloatC(0) : FloatC(RB_G(owner_b, mass_inverse));
            Mat3C I_inv_b = static_b ? Mat3C::zero()
                                    : worldInertiaInverse(
                                          Mat3C(RB_G(owner_b, local_moment_inertia_inverse)),
                                          RB_G(owner_b, orientation)
                                      );

            FloatC pen = depth_stored - (delta_a - delta_b).dot(n);
            if (pen <= slop) return;

            Vec3C r_a = (cp + delta_a) - com_a;
            Vec3C r_b = is_floor ? Vec3C::zero() : ((cp + delta_b) - com_b);
            FloatC K = contactEffectiveMassInverse(mass_inv_a, I_inv_a, r_a, n);
            if (!static_b) K += contactEffectiveMassInverse(mass_inv_b, I_inv_b, r_b, n);
            if (K <= FloatC(0)) return;

            FloatC normalizer = contacts.template get<Pr(C, normalizer)>(i);
            FloatC lambda_pos = baumgarte * (pen - slop) / (K * normalizer);
            Vec3C P = lambda_pos * n;

            if (!static_a) {
                device::deviceAtomicAdd(&RB_G(owner_a, com), mass_inv_a * P);
                _applyAngularCorrection(k, owner_a, I_inv_a * r_a.cross(P));
            }
            if (!static_b) {
                device::deviceAtomicAdd(&RB_G(owner_b, com), -(mass_inv_b * P));
                _applyAngularCorrection(k, owner_b, I_inv_b * r_b.cross(-P));
            }
        }

       private:
        /// Compose a small-angle axis-angle correction onto the body orientation.
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        _applyAngularCorrection(Kernel *k, RHandle rigid_body, const Vec3C &rotation_vec) {
            if (rotation_vec.length2() <= FloatC(0)) return;
            auto &ctx = k->ctx;
            Quat3C rotation;
            rotation.fromRotationVector(rotation_vec);
            Quat3C orientation = (rotation * Quat3C(RB_G(rigid_body, orientation))).normalized();
            RB_S(rigid_body, orientation, orientation);
        }
    };

    /// Returns the effective inverse mass along a direction, mass_inv + (I_inv (r x dir)) . (r x
    /// dir).
    HOST_DEVICE_INLINE_QUALIFIER static FloatC contactEffectiveMassInverse(
        FloatC mass_inv,
        const Mat3C &I_inv,
        const Vec3C &r,
        const Vec3C &dir
    ) {
        Vec3C rxd = r.cross(dir);
        return mass_inv + (I_inv * rxd).dot(rxd);
    }

    /// Returns the world-frame inverse inertia, R I_inv_local R^T.
    HOST_DEVICE_INLINE_QUALIFIER static Mat3C
    worldInertiaInverse(const Mat3C &local_moment_inertia_inverse, const Quat3C &orientation) {
        Mat3C R = orientation.toRotationMatrix();
        return R * local_moment_inertia_inverse * R.transpose();
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_CONTACT_C_H
