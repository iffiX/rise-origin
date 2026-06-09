// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VOXEL_MATERIAL_C_H
#define RS_VOXEL_MATERIAL_C_H

#include "rs_material_c.h"
#include "rs_voxel_material_m.h"

namespace rise {

namespace logic {

template <RSE_FloatType FloatComputeAccuracy>
class RS_VoxelMaterialCompute : public RS_MaterialCompute<FloatComputeAccuracy> {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;

    /*****************************************************************
     * Host side computation functions
     *****************************************************************/

    /// @brief Initializer. This function is called by the kernel manager.
    template <typename VoxelMaterialMem>
    HOST_INLINE_QUALIFIER static void initMaterial(VoxelMaterialMem &material, FloatC grav_acc) {
        auto &m = material;
        RS_MaterialCompute<FloatComputeAccuracy>::initMaterial(m);

        FloatC voxel_size = m.voxel_size;
        FloatC volume = voxel_size * voxel_size * voxel_size;
        m.mass = volume * m.rho;
        m.moment_inertia = m.mass * voxel_size * voxel_size / 6.0;  // simple 1D approx
        m.first_moment = m.mass * voxel_size / 2.0;

        if (volume != 0 && m.mass != 0 && m.moment_inertia != 0) {
            m.mass_inverse = 1.0 / m.mass;
            m.sqrt_mass = sqrt(m.mass);
            m.moment_inertia_inverse = 1.0 / m.moment_inertia;
            FloatC _2xSqMxExS = 2.0 * sqrt(m.mass * m.E * voxel_size);
            FloatC _2xSqIxExSxSxS =
                2.0 * sqrt(m.moment_inertia * m.E * voxel_size * voxel_size * voxel_size);
            m.internal_damping_c = 2.0 * m.zeta_internal * sqrt(m.mass);
            m.global_damping_translate_c = m.zeta_global * _2xSqMxExS;
            m.global_damping_rotate_c = m.zeta_global * _2xSqIxExSxSxS;
            m.contact_damping_translate_c = m.zeta_contact * _2xSqMxExS;
            m.contact_damping_rotate_c = m.zeta_contact * _2xSqIxExSxSxS;
            m.penetration_stiffness = 2.0 * m.E * voxel_size;
            m.gravity = m.mass * grav_acc;
        }
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_VOXEL_MATERIAL_C_H