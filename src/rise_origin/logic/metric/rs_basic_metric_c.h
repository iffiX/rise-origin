// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BASIC_METRIC_C_H
#define RS_BASIC_METRIC_C_H

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/soft/rs_link_c.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"
#include "rise_origin/utils/rs_memory.h"
#include "rs_basic_metric_m.h"

namespace rise {

namespace logic {

class RSCF_GetMassDotPosMetric : public device::RS_ComputeFunction<RSCF_GetMassDotPosMetric> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_MemoryPtr result, RHandle voxel, RIndex result_index) {
        if (k != nullptr && voxel != RS_NULL_HANDLE) {
            auto &ctx = k->ctx;

            RIndex mat = V_G(voxel_material);
            RFloat mat_mass = VM_G(mat, mass);
            RVec3rf pos = V_G(position);
            RVec3rf dot = pos * mat_mass;
            auto &mdp = result.get<RS_MassDotPosMetricMem>(result_index);
            mdp.mass_dot_x = dot.x;
            mdp.mass_dot_y = dot.y;
            mdp.mass_dot_z = dot.z;
            mdp.mass = mat_mass;
        }
    }
};

class RSCF_GetVoxelBoundMetric : public device::RS_ComputeFunction<RSCF_GetVoxelBoundMetric> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_MemoryPtr result, RHandle voxel, RIndex result_index) {
        if (k != nullptr && voxel != RS_NULL_HANDLE) {
            auto &ctx = k->ctx;

            RVec3rf pos = V_G(position);
            RVec3rf nnn_offset = V_G(nnn_offset);
            RVec3rf ppp_offset = V_G(ppp_offset);
            RFloat radius = (ppp_offset - nnn_offset).length() * static_cast<RFloat>(0.5);

            RBound3f bound = RBound3f(
                pos - RVec3rf(radius, radius, radius), pos + RVec3rf(radius, radius, radius)
            );
            result.get<RBound3f>(result_index) = bound;
        }
    }
};

class RSCF_GetIsVoxelNanMetric : public device::RS_ComputeFunction<RSCF_GetIsVoxelNanMetric> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_MemoryPtr result, RHandle voxel, RIndex result_index) {
        if (k != nullptr && voxel != RS_NULL_HANDLE) {
            auto &ctx = k->ctx;

            auto &is_nan = result.get<bool>(result_index);
            auto state = RS_VoxelStateUnion{V_G(state)}.voxel_state;

            if (state.is_enabled && !state.is_fixed) {
                is_nan = !V_G(voxel, position).template toVec3D<RFloat>().isValid();
            } else {
                is_nan = false;
            }
        }
    }
};

class RSCF_GetLinkFrequencyMetric : public device::RS_ComputeFunction<RSCF_GetLinkFrequencyMetric> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_MemoryPtr result, RHandle link, RIndex result_index) {
        if (k != nullptr && link != RS_NULL_HANDLE) {
            auto &ctx = k->ctx;

            auto state = RS_LinkStateUnion{L_G(state)}.link_state;
            if (state.is_enabled && !state.is_failed && !state.is_tendon) {
                RIndex voxel_neg = L_G(voxel_neg);
                RIndex voxel_pos = L_G(voxel_pos);
                RIndex voxel_neg_mat = V_G(voxel_neg, voxel_material);
                RIndex voxel_pos_mat = V_G(voxel_pos, voxel_material);
                RFloat mass_neg = VM_G(voxel_neg_mat, mass);
                RFloat mass_pos = VM_G(voxel_pos_mat, mass);
                RIndex link_material = L_G(link_material);
                RFloat stiffness = RS_LinkCompute<RS_DEFAULT_FLOAT_TYPE>::axialStiffness(
                    LM_G(link_material, nu),
                    LM_G(link_material, a1),
                    LM_G(link_material, E_hat),
                    L_G(transverse_area),
                    L_G(rest_length),
                    L_G(strain)
                );

                // axial
                RFloat freq = stiffness / (mass_neg < mass_pos ? mass_neg : mass_pos);
                result.get<RFloat>(result_index) = freq;
            } else {
                result.get<RFloat>(result_index) = 0;
            }
        }
    }
};

class RSCF_GetVoxelFrequencyMetric
    : public device::RS_ComputeFunction<RSCF_GetVoxelFrequencyMetric> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_MemoryPtr result, RHandle voxel, RIndex result_index) {
        if (k != nullptr && voxel != RS_NULL_HANDLE) {
            auto &ctx = k->ctx;

            auto state = RS_VoxelStateUnion{V_G(state)}.voxel_state;
            if (state.is_enabled && !state.is_fixed && !state.is_rigid) {
                RIndex mat = V_G(voxel_material);
                RFloat youngs_modulus = VM_G(mat, E);
                RFloat nom_size = VM_G(mat, voxel_size);
                RFloat mass = VM_G(mat, mass);
                RFloat freq = youngs_modulus * nom_size / mass;
                result.get<RFloat>(result_index) = freq;
            } else {
                result.get<RFloat>(result_index) = 0;
            }
        }
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_BASIC_METRIC_C_H
