// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BASIC_METRIC_C_H
#define RS_BASIC_METRIC_C_H

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/soft/rs_link_c.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rs_basic_metric_m.h"

namespace rise {

namespace logic {

class RSCF_GetMassDotPosMetric : public device::RS_ComputeFunction<RSCF_GetMassDotPosMetric> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RHandle voxel, RIndex result_index, RS_MemoryPtr result) {
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
    run(const Kernel *k, RHandle voxel, RIndex result_index, RS_MemoryPtr result) {
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
    run(const Kernel *k, RHandle voxel, RIndex result_index, RS_MemoryPtr result) {
        if (k != nullptr && voxel != RS_NULL_HANDLE) {
            auto &ctx = k->ctx;

            auto &is_nan = result.get<bool>(result_index);
            auto state = RS_VoxelStateUnion{V_G(state)}.voxel_state;

            if (state.is_enabled && !state.is_static) {
                RVec3rf position = V_G(voxel, position).template toVec3D<RFloat>();
                RFloat size = V_G(voxel, size);
                RFloat radius = V_G(voxel, radius);
                RFloat ratio = radius / size;
                is_nan = !position.isValid() || ratio > 100 || ratio < 0.01;
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
    run(const Kernel *k, RHandle link, RIndex result_index, RS_MemoryPtr result) {
        if (k != nullptr && link != RS_NULL_HANDLE) {
            auto &ctx = k->ctx;

            auto state = RS_LinkStateUnion{L_G(state)}.link_state;

            // Note: when freezing a structure, its links are not in layout, so
            // no need to be checked, if both sides rigid/fixed link will not be 
            // generated so that also does not need to be checked.
            if (state.is_enabled && !state.is_failed) {
                RIndex voxel_neg = L_G(voxel_neg);
                RIndex voxel_pos = L_G(voxel_pos);

                RFloat mass_neg = 0, mass_pos = 0;
                if (!state.is_voxel_neg_rigid_or_fixed)
                    mass_neg = VM_G(V_G(voxel_neg, voxel_material), mass);
                if (!state.is_voxel_pos_rigid_or_fixed)
                    mass_pos = VM_G(V_G(voxel_pos, voxel_material), mass);
                RFloat mass = mass_neg > mass_pos ? mass_neg : mass_pos;

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
                RFloat freq = stiffness / mass;
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
    run(const Kernel *k, RHandle voxel, RIndex result_index, RS_MemoryPtr result) {
        if (k != nullptr && voxel != RS_NULL_HANDLE) {
            auto &ctx = k->ctx;

            auto state = RS_VoxelStateUnion{V_G(state)}.voxel_state;
            if (state.is_enabled && !state.is_static && !state.is_rigid && !state.is_fixed) {
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
