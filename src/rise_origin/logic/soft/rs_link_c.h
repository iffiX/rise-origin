// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_LINK_C_H
#define RS_LINK_C_H

#include "rise_origin/core/rs_kernel.h"
#include "rise_origin/device/rs_compute.h"
#include "rs_link_m.h"
#include "rs_material_c.h"

namespace rise {

namespace logic {
template <RSE_FloatType FloatComputeAccuracy>
class RS_VoxelCompute;

class RSCF_DisableLink : public device::RS_ComputeFunction<RSCF_DisableLink> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel k, RSize link_segment_offset) {
        auto &ctx = k.ctx;
        RHandle link = index1D() + link_segment_offset;

        RS_LinkStateUnion state_union{L_G(link, state)};
        state_union.link_state.is_enabled = false;
        L_S(state, state_union.link_state_packed);
    }
};

class RSCF_ResolveLinkDependency : public device::RS_ComputeFunction<RSCF_ResolveLinkDependency> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        Kernel k,
        RSize link_segment_offset,
        RSize voxel_segment_offset,
        RSize link_material_segment_offset
    ) {
        RHandle link = index1D() + link_segment_offset;
        auto &ctx = k.ctx;
        RHandle link_material = L_G(link_material_sid) + link_material_segment_offset;
        RHandle voxel_neg = L_G(voxel_neg_sid) + voxel_segment_offset;
        RHandle voxel_pos = L_G(voxel_pos_sid) + voxel_segment_offset;
        L_S(link_material, link_material);
        L_S(voxel_neg, voxel_neg);
        L_S(voxel_pos, voxel_pos);
    }
};

template <RSE_FloatType FloatComputeAccuracy>
class RS_LinkCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using FI = RS_FloatIntrinsics<FloatC>;
    using VoxelCompute = RS_VoxelCompute<FloatComputeAccuracy>;
    using MaterialCompute = RS_MaterialCompute<FloatComputeAccuracy>;
    /*****************************************************************
     * Device computation functions for updating
     *****************************************************************/
    class RSCF_UpdateBernoulliLink : public device::RS_ComputeFunction<RSCF_UpdateBernoulliLink> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle link) {
            using Vec3M = RS_Vec3DMem<Kernel::ContextType::FloatMemAccuracy>;

            if (k != nullptr && link != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                RS_LinkStateUnion state_union{L_G(state)};
                // Update deform state and force and moment
                if (state_union.link_state.is_enabled && !state_union.link_state.is_failed) {
                    RHandle voxel_neg = L_G(voxel_neg);
                    RHandle voxel_pos = L_G(voxel_pos);

                    if (!state_union.link_state.is_tendon) {
                        RHandle link_material = L_G(link_material);

                        FloatC rest_length, transverse_area, transverse_strain_sum;

                        auto axis = static_cast<RSE_LinkAxis>(state_union.link_state.axis);
                        bool neg_rigid = state_union.link_state.voxel_neg_rigid;
                        bool pos_rigid = state_union.link_state.voxel_pos_rigid;
                        FloatC neg_weight, pos_weight;
                        if (!neg_rigid && !pos_rigid) {
                            neg_weight = FloatC(0.5);
                            pos_weight = FloatC(0.5);
                        } else if (neg_rigid) {
                            neg_weight = FloatC(0);
                            pos_weight = FloatC(1);
                        } else {
                            neg_weight = FloatC(1);
                            pos_weight = FloatC(0);
                        }
                        FloatC size = LM_G(voxel_size);

                        /// Update rest length
                        FloatC neg_base_size = VoxelCompute::baseSize(
                            size, neg_rigid ? FloatC(0) : V_G(voxel_neg, expansion)
                        );
                        FloatC pos_base_size = VoxelCompute::baseSize(
                            size, pos_rigid ? FloatC(0) : V_G(voxel_pos, expansion)
                        );
                        rest_length = neg_base_size * neg_weight + pos_base_size * pos_weight;

                        /// Update transverse area
                        FloatC neg_poissons_ratio = LM_G(voxel_neg_nu);
                        Vec3C neg_poissons_strain = V_G(voxel_neg, poissons_strain);
                        FloatC pos_poissons_ratio = LM_G(voxel_pos_nu);
                        Vec3C pos_poissons_strain = V_G(voxel_pos, poissons_strain);

                        FloatC neg_area = VoxelCompute::transverseArea(
                            size, neg_poissons_ratio, axis, neg_poissons_strain
                        );
                        FloatC pos_area = VoxelCompute::transverseArea(
                            size, pos_poissons_ratio, axis, pos_poissons_strain
                        );
                        transverse_area = neg_area * neg_weight + pos_area * pos_weight;

                        FloatC neg_strain = VoxelCompute::transverseStrainSum(
                            neg_poissons_ratio, axis, neg_poissons_strain
                        );
                        FloatC pos_strain = VoxelCompute::transverseStrainSum(
                            pos_poissons_ratio, axis, pos_poissons_strain
                        );
                        transverse_strain_sum = neg_strain * neg_weight + pos_strain * pos_weight;

                        FloatC old_length = L_G(length);

                        // voxel neg (pos1) is always at origin (0, 0, 0)
                        // angle1-> voxel neg, pos2, angle2-> voxel pos
                        Vec3C pos2 = _toAxisX(
                            static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                            Vec3C(V_G(voxel_pos, position)) - Vec3C(V_G(voxel_neg, position))
                        );
                        FloatC length = pos2.length();
                        FloatC d_length = FloatC(0.5) * (length - old_length);

                        // FloatC strain = length / rest_length - FloatC(1);
                        // TMP-FIX: allow large
                        FloatC strain = RS_MAX(length / rest_length - FloatC(1), -0.8);

                        if (!MaterialCompute::isFailed(LM_G(fail_strain), strain)) {
                            Quat3C neg_orientation = _toAxisX(
                                static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                Quat3C(V_G(voxel_neg, orientation))
                            );
                            Quat3C pos_orientation = _toAxisX(
                                static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                Quat3C(V_G(voxel_pos, orientation))
                            );

                            L_S(length, length);
                            L_S(rest_length, rest_length);
                            L_S(strain, strain);
                            L_S(transverse_area, transverse_area);

                            FloatC stress = MaterialCompute::stress(
                                LM_G(E), LM_G(E_hat), LM_G(nu), strain, transverse_strain_sum
                            );
                            L_S(stress, stress);

                            /**
                             * We must cancel out the negative rotation because in some
                             extreme
                             * cases, the quaternion that produce similar orientation will
                             have
                             * very different values
                             * eg: [x=0.000, y=0.534, z=0.845, w=0.015] and
                             *     [x=0.000, y=0.859, z=-0.512, w=0.016]
                             * because the relative rotation between two voxels are usually
                             * small, by canceling out rotation on one end first, the angle1v
                             * and angle2v will be much more stable.
                             */
                            Vec3C old_angle1v = L_G(angle1v);
                            Vec3C old_angle2v = L_G(angle2v);
                            Quat3C canceled_rotation = neg_orientation.conjugate();
                            pos2 = canceled_rotation.rotateVec3D(pos2);
                            Quat3C relative_pos_orientation = canceled_rotation * pos_orientation;

                            Quat3C angle1;
                            angle1.fromAngleToPosX(pos2);
                            Vec3C angle1v = angle1.toRotationVector();
                            L_S(angle1v, angle1v);

                            Quat3C angle2 = angle1 * relative_pos_orientation;
                            Vec3C angle2v = angle2.toRotationVector();
                            L_S(angle2v, angle2v);

                            FloatC dt = k->getDeviceTimeSinceLastOperation(
                                RSE_KERNEL_UPDATE_OPERATION_SOFT_BODY
                            );
                            FloatC voxel_neg_damping = LM_G(voxel_neg_internal_damping_c) / dt;
                            FloatC voxel_pos_damping = LM_G(voxel_pos_internal_damping_c) / dt;

                            // Use current stress instead of -a1*length to account for
                            // non-linear deformation
                            FloatC b2 = LM_G(b2);

                            Vec3C force_neg = Vec3C(
                                stress * transverse_area,
                                -b2 * (angle1v.z + angle2v.z),
                                b2 * (angle1v.y + angle2v.y)
                            );
                            Vec3C force_pos = -force_neg;

                            Vec3C d_angle1v = FloatC(0.5) * (angle1v - old_angle1v);
                            Vec3C d_angle2v = FloatC(0.5) * (angle2v - old_angle2v);

                            FloatC sqA1 = LM_G(sqA1), sqB2xFMp = LM_G(sqB2xFMp);

                            Vec3C pos_calc(
                                sqA1 * d_length,
                                -sqB2xFMp * (d_angle1v.z + d_angle2v.z),
                                sqB2xFMp * (d_angle1v.y + d_angle2v.y)
                            );

                            force_neg += voxel_neg_damping * pos_calc;

                            force_neg = angle1.rotateVec3DInv(force_neg);
                            L_S(force_neg,
                                _toAxisOriginal(
                                    static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                    force_neg
                                ));
                            force_pos -= voxel_pos_damping * pos_calc;
                            force_pos = angle2.rotateVec3DInv(force_pos);

                            L_S(force_pos,
                                _toAxisOriginal(
                                    static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                    force_pos
                                ));
                            FloatC b3 = LM_G(b3), a2 = LM_G(a2);
                            Vec3C moment_neg = Vec3C(
                                a2 * (angle2v.x - angle1v.x),
                                -b3 * (FloatC(2) * angle1v.y + angle2v.y),
                                -b3 * (FloatC(2) * angle1v.z + angle2v.z)
                            );
                            Vec3C moment_pos = Vec3C(
                                a2 * (angle1v.x - angle2v.x),
                                -b3 * (angle1v.y + FloatC(2) * angle2v.y),
                                -b3 * (angle1v.z + FloatC(2) * angle2v.z)
                            );

                            FloatC sqA2xIp = LM_G(sqA2xIp), sqB3xIp = LM_G(sqB3xIp);

                            moment_neg -= FloatC(0.5) * voxel_neg_damping *
                                          Vec3C(
                                              -sqA2xIp * (d_angle2v.x - d_angle1v.x),
                                              sqB3xIp * (FloatC(2) * d_angle1v.y + d_angle2v.y),
                                              sqB3xIp * (FloatC(2) * d_angle1v.z + d_angle2v.z)
                                          );
                            moment_neg = angle1.rotateVec3DInv(moment_neg);
                            L_S(moment_neg,
                                _toAxisOriginal(
                                    static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                    moment_neg
                                ));

                            moment_pos -= FloatC(0.5) * voxel_pos_damping *
                                          Vec3C(
                                              -sqA2xIp * (d_angle1v.x - d_angle2v.x),
                                              sqB3xIp * (d_angle1v.y + FloatC(2) * d_angle2v.y),
                                              sqB3xIp * (d_angle1v.z + FloatC(2) * d_angle2v.z)
                                          );
                            moment_pos = angle2.rotateVec3DInv(moment_pos);
                            L_S(moment_pos,
                                _toAxisOriginal(
                                    static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                    moment_pos
                                ));
                        } else {
                            state_union.link_state.is_failed = true;
                            L_S(state, state_union.link_state_packed);
                            L_S(length, 0);
                            L_S(rest_length, 0);
                            L_S(strain, 0);
                            L_S(stress, 0);
                            L_S(transverse_area, 0);

                            L_S(force_neg, Vec3M::zero());
                            L_S(force_pos, Vec3M::zero());
                            L_S(moment_neg, Vec3M::zero());
                            L_S(moment_pos, Vec3M::zero());
                            L_S(angle1v, Vec3M::zero());
                            L_S(angle2v, Vec3M::zero());
                        }
                    } else {
                        RHandle material = L_G(link_material);

                        FloatC old_length = L_G(length);
                        FloatC rest_length = L_G(rest_length);

                        // voxel neg (pos1) is always at origin (0, 0, 0)
                        // angle1-> voxel neg, pos2, angle2-> voxel pos
                        Vec3C diff =
                            Vec3C(V_G(voxel_pos, position)) - Vec3C(V_G(voxel_neg, position));
                        Vec3C pos2 =
                            _toAxisX(static_cast<RSE_LinkAxis>(state_union.link_state.axis), diff);
                        FloatC length = pos2.length();
                        FloatC d_length = FloatC(0.5) * (length - old_length);

                        FloatC strain = length / rest_length - FloatC(1);
                        if (!MaterialCompute::isFailed(LM_G(material, fail_strain), strain)) {
                            L_S(length, length);
                            L_S(strain, strain);

                            FloatC E = LM_G(material, E);
                            FloatC stress = strain * E;
                            L_S(stress, stress);
                            FloatC transverse_strain = -LM_G(material, nu) * strain;
                            FloatC new_radius = LM_G(material, voxel_size) * FloatC(0.5) *
                                                (FloatC(1) + transverse_strain);
                            FloatC transverse_area = new_radius * new_radius * FloatC(RS_PI);
                            L_S(transverse_area, transverse_area);
                            FloatC spring_constant = transverse_area * E / rest_length;
                            FloatC force = spring_constant * strain * rest_length;
                            if (strain <= 0) {
                                // Normally tendon shouldn't provide force when compressed
                                // We choose to provide a much smaller force here to prevent
                                // over-compression
                                force *= FloatC(0.1);
                            }

                            FloatC diff_length = diff.length();
                            if (diff_length > 0) {
                                Vec3C force_neg = (diff / diff_length) * force;
                                L_S(force_neg, force_neg);
                                L_S(force_pos, -force_neg);
                            } else {
                                L_S(force_neg, Vec3M::zero());
                                L_S(force_pos, Vec3M::zero());
                            }

                        } else {
                            state_union.link_state.is_failed = true;
                            L_S(state, state_union.link_state_packed);
                            L_S(length, 0);
                            L_S(rest_length, 0);
                            L_S(strain, 0);
                            L_S(stress, 0);
                            L_S(transverse_area, 0);

                            L_S(force_neg, Vec3M::zero());
                            L_S(force_pos, Vec3M::zero());
                        }
                    }
#ifdef RS_DEBUG_LINK
                    CUDA_PRINTF_ASSERT(!isnan(L_G(strain)) && L_G(strain) < 1e6);
                    CUDA_PRINTF_ASSERT(!isnan(L_G(rest_length)) && L_G(rest_length) < 1e6);
                    CUDA_PRINTF_ASSERT(!isnan(L_G(transverse_area)) && L_G(transverse_area) < 1e6);
                    CUDA_PRINTF_ASSERT(!isnan(L_G(length)) && L_G(length) < 1e6);
                    CUDA_PRINTF_ASSERT(Vec3C(L_G(force_neg)).isValid());
                    CUDA_PRINTF_ASSERT(Vec3C(L_G(force_pos)).isValid());
                    CUDA_PRINTF_ASSERT(Vec3C(L_G(moment_neg)).isValid());
                    CUDA_PRINTF_ASSERT(Vec3C(L_G(moment_pos)).isValid());
#endif
                }
            }
        }
    };

    class RSCF_UpdateSpringLink : public device::RS_ComputeFunction<RSCF_UpdateSpringLink> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle link) {
            using Vec3M = RS_Vec3DMem<Kernel::ContextType::FloatMemAccuracy>;

            if (k != nullptr && link != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;
                RS_LinkStateUnion state_union{L_G(state)};

                // Update deform state and force and moment
                if (state_union.link_state.is_enabled && !state_union.link_state.is_failed) {
                    RHandle voxel_neg = L_G(voxel_neg);
                    RHandle voxel_pos = L_G(voxel_pos);

                    if (!state_union.link_state.is_tendon) {
                        RHandle link_material = L_G(link_material);
                        FloatC rest_length, transverse_area, transverse_strain_sum;

                        auto axis = static_cast<RSE_LinkAxis>(state_union.link_state.axis);
                        bool neg_rigid = state_union.link_state.voxel_neg_rigid;
                        bool pos_rigid = state_union.link_state.voxel_pos_rigid;
                        FloatC neg_weight, pos_weight;
                        if (!neg_rigid && !pos_rigid) {
                            neg_weight = FloatC(0.5);
                            pos_weight = FloatC(0.5);
                        } else if (neg_rigid) {
                            neg_weight = FloatC(0);
                            pos_weight = FloatC(1);
                        } else {
                            neg_weight = FloatC(1);
                            pos_weight = FloatC(0);
                        }
                        FloatC size = LM_G(voxel_size);

                        /// Update rest length
                        FloatC neg_base_size = VoxelCompute::baseSize(
                            size, neg_rigid ? FloatC(0) : V_G(voxel_neg, expansion)
                        );
                        FloatC pos_base_size = VoxelCompute::baseSize(
                            size, pos_rigid ? FloatC(0) : V_G(voxel_pos, expansion)
                        );
                        rest_length = neg_base_size * neg_weight + pos_base_size * pos_weight;

                        /// Update transverse area
                        FloatC neg_poissons_ratio = LM_G(voxel_neg_nu);
                        Vec3C neg_poissons_strain = V_G(voxel_neg, poissons_strain);
                        FloatC pos_poissons_ratio = LM_G(voxel_pos_nu);
                        Vec3C pos_poissons_strain = V_G(voxel_pos, poissons_strain);

                        FloatC neg_area = VoxelCompute::transverseArea(
                            size, neg_poissons_ratio, axis, neg_poissons_strain
                        );
                        FloatC pos_area = VoxelCompute::transverseArea(
                            size, pos_poissons_ratio, axis, pos_poissons_strain
                        );
                        transverse_area = neg_area * neg_weight + pos_area * pos_weight;

                        FloatC neg_strain = VoxelCompute::transverseStrainSum(
                            neg_poissons_ratio, axis, neg_poissons_strain
                        );
                        FloatC pos_strain = VoxelCompute::transverseStrainSum(
                            pos_poissons_ratio, axis, pos_poissons_strain
                        );
                        transverse_strain_sum = neg_strain * neg_weight + pos_strain * pos_weight;

                        FloatC old_length = L_G(length);

                        // voxel neg (pos1) is always at origin (0, 0, 0)
                        // angle1-> voxel neg, pos2, angle2-> voxel pos
                        Vec3C pos2 = _toAxisX(
                            static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                            Vec3C(V_G(voxel_pos, position)) - Vec3C(V_G(voxel_neg, position))
                        );
                        FloatC length = pos2.length();
                        FloatC d_length = FloatC(0.5) * (length - old_length);
                        FloatC strain = length / rest_length - FloatC(1);

                        if (!MaterialCompute::isFailed(LM_G(fail_strain), strain)) {
                            Quat3C neg_orientation = _toAxisX(
                                static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                Quat3C(V_G(voxel_neg, orientation))
                            );
                            Quat3C pos_orientation = _toAxisX(
                                static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                Quat3C(V_G(voxel_pos, orientation))
                            );
                            L_S(length, length);
                            L_S(rest_length, rest_length);
                            L_S(strain, strain);
                            L_S(transverse_area, transverse_area);

                            FloatC stress = MaterialCompute::stress(
                                LM_G(E), LM_G(E_hat), LM_G(nu), strain, transverse_strain_sum
                            );
                            L_S(stress, stress);
                            /**
                             * We must cancel out the negative rotation because in some extreme
                             * cases, the quaternion that produce similar orientation will have
                             * very different values
                             * eg: [x=0.000, y=0.534, z=0.845, w=0.015] and
                             *     [x=0.000, y=0.859, z=-0.512, w=0.016]
                             * because the relative rotation between two voxels are usually
                             * small, by canceling out rotation on one end first, the angle1v
                             * and angle2v will be much more stable.
                             */
                            Vec3C old_angle1v = L_G(angle1v);
                            Vec3C old_angle2v = L_G(angle2v);
                            Quat3C canceled_rotation = neg_orientation.conjugate();
                            pos2 = canceled_rotation.rotateVec3D(pos2);
                            Quat3C relative_pos_orientation = canceled_rotation * pos_orientation;

                            Quat3C angle1;
                            angle1.fromAngleToPosX(pos2);
                            Vec3C angle1v = angle1.toRotationVector();
                            L_S(angle1v, angle1v);

                            Quat3C angle2 = angle1 * relative_pos_orientation;
                            Vec3C angle2v = angle2.toRotationVector();
                            L_S(angle2v, angle2v);

                            FloatC dt = k->getDeviceTimeSinceLastOperation(
                                RSE_KERNEL_UPDATE_OPERATION_SOFT_BODY
                            );
                            FloatC voxel_neg_damping = LM_G(voxel_neg_internal_damping_c) / dt;
                            FloatC voxel_pos_damping = LM_G(voxel_pos_internal_damping_c) / dt;

                            // Use current stress instead of -a1*length to account for
                            // non-linear deformation
                            FloatC b2 = LM_G(b2);

                            Vec3C force_neg = Vec3C(
                                stress * transverse_area,
                                -b2 * (angle1v.z + angle2v.z),
                                b2 * (angle1v.y + angle2v.y)
                            );
                            Vec3C force_pos = -force_neg;

                            Vec3C d_angle1v = FloatC(0.5) * (angle1v - old_angle1v);
                            Vec3C d_angle2v = FloatC(0.5) * (angle2v - old_angle2v);

                            FloatC sqA1 = LM_G(sqA1), sqB2xFMp = LM_G(sqB2xFMp);

                            Vec3C pos_calc(
                                sqA1 * d_length,
                                -sqB2xFMp * (d_angle1v.z + d_angle2v.z),
                                sqB2xFMp * (d_angle1v.y + d_angle2v.y)
                            );

                            force_neg += voxel_neg_damping * pos_calc;
                            force_neg = angle1.rotateVec3DInv(force_neg);
                            L_S(force_neg,
                                _toAxisOriginal(
                                    static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                    force_neg
                                ));

                            force_pos -= voxel_pos_damping * pos_calc;
                            force_pos = angle2.rotateVec3DInv(force_pos);
                            L_S(force_pos,
                                _toAxisOriginal(
                                    static_cast<RSE_LinkAxis>(state_union.link_state.axis),
                                    force_pos
                                ));
                        } else {
                            state_union.link_state.is_failed = true;
                            L_S(state, state_union.link_state_packed);
                            L_S(length, 0);
                            L_S(rest_length, 0);
                            L_S(strain, 0);
                            L_S(stress, 0);
                            L_S(transverse_area, 0);
                            L_S(length, 0);

                            L_S(force_neg, Vec3M::zero());
                            L_S(force_pos, Vec3M::zero());
                        }
                    } else {
                        RHandle link_material = L_G(link_material);

                        FloatC old_length = L_G(length);
                        FloatC rest_length = L_G(rest_length);

                        // voxel neg (pos1) is always at origin (0, 0, 0)
                        // angle1-> voxel neg, pos2, angle2-> voxel pos
                        Vec3C diff =
                            Vec3C(V_G(voxel_pos, position)) - Vec3C(V_G(voxel_neg, position));
                        Vec3C pos2 =
                            _toAxisX(static_cast<RSE_LinkAxis>(state_union.link_state.axis), diff);
                        FloatC length = pos2.length();
                        FloatC d_length = FloatC(0.5) * (length - old_length);

                        FloatC strain = length / rest_length - FloatC(1);
                        if (!MaterialCompute::isFailed(LM_G(fail_strain), strain)) {
                            L_S(length, length);
                            L_S(strain, strain);

                            FloatC E = LM_G(E);
                            FloatC stress = strain * E;
                            L_S(stress, stress);
                            FloatC transverse_strain = -LM_G(nu) * strain;
                            FloatC new_radius =
                                LM_G(voxel_size) * FloatC(0.5) * (FloatC(1) + transverse_strain);
                            FloatC transverse_area = new_radius * new_radius * FloatC(RS_PI);
                            L_S(transverse_area, transverse_area);
                            FloatC spring_constant = transverse_area * E / rest_length;
                            FloatC force = spring_constant * strain * rest_length;
                            if (strain <= 0) {
                                // Normally tendon shouldn't provide force when compressed
                                // We choose to provide a much smaller force here to prevent
                                // over-compression
                                force *= FloatC(0.1);
                            }

                            FloatC diff_length = diff.length();
                            if (diff_length > 0) {
                                Vec3C force_neg = (diff / diff_length) * force;
                                L_S(force_neg, force_neg);
                                L_S(force_pos, -force_neg);
                            } else {
                                L_S(force_neg, Vec3M::zero());
                                L_S(force_pos, Vec3M::zero());
                            }

                        } else {
                            state_union.link_state.is_failed = true;
                            L_S(state, state_union.link_state_packed);
                            L_S(length, 0);
                            L_S(rest_length, 0);
                            L_S(strain, 0);
                            L_S(stress, 0);
                            L_S(transverse_area, 0);

                            L_S(force_neg, Vec3M::zero());
                            L_S(force_pos, Vec3M::zero());
                        }
                    }
                }
#ifdef RS_DEBUG_LINK
                CUDA_PRINTF_ASSERT(!isnan(L_G(strain)) && L_G(strain) < 1e6);
                CUDA_PRINTF_ASSERT(!isnan(L_G(rest_length)) && L_G(rest_length) < 1e6);
                CUDA_PRINTF_ASSERT(!isnan(L_G(transverse_area)) && L_G(transverse_area) < 1e6);
                CUDA_PRINTF_ASSERT(!isnan(L_G(length)) && L_G(length) < 1e6);
                CUDA_PRINTF_ASSERT(Vec3C(L_G(force_neg)).isValid());
                CUDA_PRINTF_ASSERT(Vec3C(L_G(force_pos)).isValid());
                CUDA_PRINTF_ASSERT(Vec3C(L_G(moment_neg)).isValid());
                CUDA_PRINTF_ASSERT(Vec3C(L_G(moment_pos)).isValid());
#endif
            }
        }
    };

    /*****************************************************************
     * Device side methods for acquiring info
     *****************************************************************/
    /// Returns true if the stress on this bond has ever
    /// exceeded its failure stress
    HOST_DEVICE_INLINE_QUALIFIER static bool isFailed(RState l_state) {
        return RS_LinkStateUnion{l_state}.link_state.is_failed;
    }

    HOST_DEVICE_INLINE_QUALIFIER static bool isTendon(RState l_state) {
        return RS_LinkStateUnion{l_state}.link_state.is_tendon;
    }

    /// Returns the current calculated axial strain of the half of
    /// the link contained in the specified voxel. @param[in]
    /// positive_end Specifies which voxel information is desired
    /// about.
    /// If voxel is on the positive end, it should be the center
    /// of a link in negative direction
    HOST_DEVICE_INLINE_QUALIFIER static FloatC
    axialStrain(FloatC l_strain, FloatC l_strain_ratio, bool positive_end) {
        return positive_end ? FloatC(2) * l_strain * l_strain_ratio / (FloatC(1) + l_strain_ratio)
                            : FloatC(2) * l_strain / (FloatC(1) + l_strain_ratio);
    }

    /// Calculates and return the strain energy of this
    /// link according to current forces and moments.
    /// (units: Joules, or Kg m^2 / s^2)
    HOST_DEVICE_INLINE_QUALIFIER static FloatC strainEnergy(
        const Vec3C &l_force_neg,
        const Vec3C &l_moment_neg,
        const Vec3C &l_moment_pos,
        FloatC lm_a1,
        FloatC lm_a2,
        FloatC lm_b3
    ) {
        return l_force_neg.x * l_force_neg.x / (FloatC(2) * lm_a1) +    // Tensile strain
               l_moment_neg.x * l_moment_neg.x / (FloatC(2) * lm_a2) +  // Torsion strain
               (l_moment_neg.z * l_moment_neg.z - l_moment_neg.z * l_moment_pos.z +
                l_moment_pos.z * l_moment_pos.z) /
                   (FloatC(3) * lm_b3) +  // Bending Z
               (l_moment_neg.y * l_moment_neg.y - l_moment_neg.y * l_moment_pos.y +
                l_moment_pos.y * l_moment_pos.y) /
                   (FloatC(3) * lm_b3);  // Bending Y
    }

    /// Calculates and returns the current linear axial
    /// stiffness of this link at it's current strain.
    HOST_DEVICE_INLINE_QUALIFIER static FloatC axialStiffness(
        FloatC lm_nu,
        FloatC lm_a1,
        FloatC lm_E_hat,
        FloatC l_current_transverse_area,
        FloatC l_current_rest_length,
        FloatC l_strain
    ) {
        if (MaterialCompute::isXyzIndependent(lm_nu))
            return lm_a1;
        else {
            return lm_E_hat * l_current_transverse_area /
                   ((l_strain + FloatC(1.0)) * l_current_rest_length);  // _a1;
        }
    }

   private:
    /// Transforms a RS_Vec3D in the original orientation of the bond to that
    /// as if the bond was in +X direction.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec3D<T>
    _toAxisX(RSE_LinkAxis axis, const RS_Vec3D<T> &v) {
        switch (axis) {
            case RSE_Y_AXIS:
                return RS_Vec3D<T>(v.y, -v.x, v.z);
            case RSE_Z_AXIS:
                return RS_Vec3D<T>(v.z, v.y, -v.x);
            default:
                return v;
        }
    }

    /// Transforms a RS_Quat3D in the original orientation of the bond to that
    /// as if the bond was in +X direction.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER static RS_Quat3D<T>
    _toAxisX(RSE_LinkAxis axis, const RS_Quat3D<T> &q) {
        switch (axis) {
            case RSE_Y_AXIS:
                return RS_Quat3D<T>(q.y, -q.x, q.z, q.w);
            case RSE_Z_AXIS:
                return RS_Quat3D<T>(q.z, q.y, -q.x, q.w);
            default:
                return q;
        }
    }

    /// Reverse the transformation on Vec3D/Quat3D.
    /// This is an in-place transformation.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec3D<T>
    _toAxisOriginal(RSE_LinkAxis axis, const RS_Vec3D<T> &v) {
        switch (axis) {
            case RSE_Y_AXIS:
                return RS_Vec3D<T>(-v.y, v.x, v.z);
            case RSE_Z_AXIS:
                return RS_Vec3D<T>(-v.z, v.y, v.x);
            default:
                return v;
        }
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_LINK_C_H