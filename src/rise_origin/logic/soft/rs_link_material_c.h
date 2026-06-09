// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_LINK_MATERIAL_C_H
#define RS_LINK_MATERIAL_C_H

#include "rs_voxel_material_c.h"

namespace rise {

namespace logic {

template <RSE_FloatType FloatComputeAccuracy>
class RS_LinkMaterialCompute : public RS_VoxelMaterialCompute<FloatComputeAccuracy> {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;

    /*****************************************************************
     * Host side computation functions
     *****************************************************************/

    /// @brief Initializer. This function is called by the kernel manager.
    template <typename LinkMaterialMem, typename VoxelMaterialMem>
    HOST_INLINE_QUALIFIER static void initMaterial(
        LinkMaterialMem &material,
        VoxelMaterialMem &voxel_neg_material,
        VoxelMaterialMem &voxel_pos_material,
        FloatC grav_acc
    ) {
        auto &m = material;
        m.voxel_size = 0.5 * (voxel_neg_material.voxel_size + voxel_pos_material.voxel_size);

        m.voxel_neg_nu = voxel_neg_material.nu;
        m.voxel_pos_nu = voxel_pos_material.nu;
        m.voxel_neg_internal_damping_c = voxel_neg_material.internal_damping_c;
        m.voxel_pos_internal_damping_c = voxel_pos_material.internal_damping_c;

        // linear, E, nu, and fail_stress are computed later
        m.strain_ratio = voxel_pos_material.E / voxel_neg_material.E;
        m.rho = 0.5 * (voxel_neg_material.rho + voxel_pos_material.rho);
        m.u_static = 0.5 * (voxel_neg_material.u_static + voxel_pos_material.u_static);
        m.u_dynamic = 0.5 * (voxel_neg_material.u_dynamic + voxel_pos_material.u_dynamic);
        m.zeta_internal =
            0.5 * (voxel_neg_material.zeta_internal + voxel_pos_material.zeta_internal);
        m.zeta_global = 0.5 * (voxel_neg_material.zeta_global + voxel_pos_material.zeta_global);
        m.zeta_contact = 0.5 * (voxel_neg_material.zeta_contact + voxel_pos_material.zeta_contact);

        // failure stress (f) is the minimum of the two failure stresses,
        // or if both are -1.0f
        // it should also be -1.0f to denote no failure specified
        FloatC f1 = voxel_neg_material.fail_stress, f2 = voxel_pos_material.fail_stress;
        if (f1 == -1.0)
            m.fail_stress = f2;  //-1.0f or vox2Mat fail
        else if (f2 == -1.0)
            m.fail_stress = f1;  // vox1_mat fail
        else
            m.fail_stress = f1 < f2 ? f1 : f2;  // the lesser stress denotes failure

        m.E = 2.0 * voxel_neg_material.E * voxel_pos_material.E /
              (voxel_neg_material.E + voxel_pos_material.E);

        // poissons ratio: choose such that Ehat ends up according to
        // spring in series of Ehat1 and EHat2
        if (voxel_neg_material.nu == 0 && voxel_pos_material.nu == 0)
            m.nu = 0;
        else {
            FloatC tmp_E_hat = 2 * voxel_neg_material.E_hat * voxel_pos_material.E_hat /
                               (voxel_neg_material.E_hat + voxel_pos_material.E_hat);
            // completing the square algorithm to solve for nu.
            // eHat = E/((1-2nu)(1+nu)) -> E/EHat = -2nu^2-nu+1 -> nu^2+0.5nu =
            // (EHat+E)/(2EHat)

            // nu^2+0.5nu+0.0625 = c2 -> (nu+0.25)^2 = c2
            FloatC c2 = (tmp_E_hat - m.E) / (2 * tmp_E_hat) + 0.0625;
            m.nu = sqrt(c2) - 0.25;  // from solving above
        }

        // Initialize base RS_Material class derived variables
        RS_VoxelMaterialCompute<FloatComputeAccuracy>::initMaterial(m, grav_acc);

        // Update derived attributes
        // stiffnesses terms for links
        FloatC L = (voxel_neg_material.voxel_size + voxel_pos_material.voxel_size) / 2.0;
        m.a1 = m.E * L;                                // EA/L : Units of N/m
        m.a2 = m.E * L * L * L / (12.0 * (1 + m.nu));  // GJ/L : Units of N-m
        m.b1 = m.E * L;                                // 12EI/L^3 : Units of N/m
        // 6EI/L^2 : Units of N (or N-m/m: torque related to linear distance)
        m.b2 = m.E * L * L / 2.0;
        m.b3 = m.E * L * L * L / 6.0;  // 2EI/L : Units of N-m

        // damping 2*zeta_internal*sqrt(mk) (or 2*zeta_internal*sqrt(Ik)) terms
        // 2*zeta_internal*sqrt(m) is factored out since voxel mass on two ends could be
        // different
        m.sqA1 = sqrt(m.a1);
        m.sqA2xIp = sqrt(m.a2 * L * L / 6.0);
        m.sqB1 = sqrt(m.b1);
        m.sqB2xFMp = sqrt(m.b2 * L / 2.0);
        m.sqB3xIp = sqrt(m.b3 * L * L / 6.0);
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_LINK_MATERIAL_C_H