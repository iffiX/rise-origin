// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MATERIAL_LINK_M_H
#define RS_MATERIAL_LINK_M_H

#include "rise_origin/utils/rs_def.h"
#include "rs_voxel_material_m.h"

namespace rise {

namespace logic {

template <RSE_FloatType FloatMemAccuracy>
class RS_LinkMaterialMem : public RS_VoxelMaterialMem<FloatMemAccuracy> {
   public:
    using BaseType = RS_VoxelMaterialMem<FloatMemAccuracy>;
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;

    /**
     * Pre-set attributes
     */
    using BaseType::material_gid;
    using BaseType::sid;
    using BaseType::structure_gid;

    /**
     * Post-set attributes
     */
    using BaseType::E;
    using BaseType::E_hat;
    using BaseType::fail_strain;
    using BaseType::fail_stress;
    using BaseType::nu;
    using BaseType::rho;
    using BaseType::zeta_contact;
    using BaseType::zeta_global;
    using BaseType::zeta_internal;

    using BaseType::voxel_size;

    FloatM voxel_neg_nu = 0;
    FloatM voxel_pos_nu = 0;
    FloatM voxel_neg_internal_damping_c = 0;
    FloatM voxel_pos_internal_damping_c = 0;
    FloatM strain_ratio = 0;
    FloatM a1 = 0;        //!< Cached a1 beam constant.
    FloatM a2 = 0;        //!< Cached a2 beam constant.
    FloatM b1 = 0;        //!< Cached b1 beam constant.
    FloatM b2 = 0;        //!< Cached b2 beam constant.
    FloatM b3 = 0;        //!< Cached b3 beam constant.
    FloatM sqA1 = 0;      //!< Cached sqrt(a1) constant for damping calculations.
    FloatM sqA2xIp = 0;   //!< Cached sqrt(a2*L*L/6) constant for damping calculations.
    FloatM sqB1 = 0;      //!< Cached sqrt(b1) constant for damping calculations.
    FloatM sqB2xFMp = 0;  //!< Cached sqrt(b2*L/2) constant for damping calculations.
    FloatM sqB3xIp = 0;   //!< Cached sqrt(b3*L*L/6) constant for damping calculations.
};

}  // namespace logic

}  // namespace rise

#define REFLECT_LINK_MATERIAL_MEM(link_material_mem_type) \
    REFL_AUTO(                                            \
        type(link_material_mem_type),                     \
        field(E),                                         \
        field(nu),                                        \
        field(rho),                                       \
        field(fail_stress),                               \
        field(zeta_internal),                             \
        field(zeta_contact),                              \
        field(zeta_global),                               \
        field(E_hat),                                     \
        field(fail_strain),                               \
        field(structure_gid),                             \
        field(sid),                                       \
        field(material_gid),                              \
        field(voxel_size),                                \
        field(min_expansion),                             \
        field(max_expansion),                             \
        field(mass),                                      \
        field(mass_inverse),                              \
        field(sqrt_mass),                                 \
        field(first_moment),                              \
        field(moment_inertia),                            \
        field(moment_inertia_inverse),                    \
        field(internal_damping_c),                        \
        field(global_damping_translate_c),                \
        field(global_damping_rotate_c),                   \
        field(contact_damping_translate_c),               \
        field(contact_damping_rotate_c),                  \
        field(penetration_stiffness),                     \
        field(gravity),                                   \
        field(voxel_neg_nu),                              \
        field(voxel_pos_nu),                              \
        field(voxel_neg_internal_damping_c),              \
        field(voxel_pos_internal_damping_c),              \
        field(strain_ratio),                              \
        field(a1),                                        \
        field(a2),                                        \
        field(b1),                                        \
        field(b2),                                        \
        field(b3),                                        \
        field(sqA1),                                      \
        field(sqA2xIp),                                   \
        field(sqB1),                                      \
        field(sqB2xFMp),                                  \
        field(sqB3xIp)                                    \
    )

REFLECT_LINK_MATERIAL_MEM(rise::logic::RS_LinkMaterialMem<rise::RSE_FLOAT_32>)
REFLECT_LINK_MATERIAL_MEM(rise::logic::RS_LinkMaterialMem<rise::RSE_FLOAT_64>)

#endif  // RS_MATERIAL_LINK_M_H
