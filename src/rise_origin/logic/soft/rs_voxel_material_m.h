// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VOXEL_MATERIAL_M_H
#define RS_VOXEL_MATERIAL_M_H

#include "rise_origin/utils/rs_def.h"
#include "rs_material_m.h"

namespace rise {

namespace logic {
template <RSE_FloatType FloatMemAccuracy>
class RS_VoxelMaterialMem : public RS_MaterialMem<FloatMemAccuracy> {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;
    using BaseType = RS_MaterialMem<FloatMemAccuracy>;

    /**
     * Pre-set attributes
     */
    /// Structure gid of this voxel material
    RIndex structure_gid = RS_NULL_INDEX;

    /// Structure scope ID of this voxel material
    RIndex sid = RS_NULL_INDEX;

    /// Global ID of the source material
    RIndex material_gid = RS_NULL_INDEX;

    using BaseType::E;
    using BaseType::fail_stress;
    using BaseType::nu;
    using BaseType::rho;
    using BaseType::u_dynamic;
    using BaseType::u_static;
    using BaseType::zeta_contact;
    using BaseType::zeta_global;
    using BaseType::zeta_internal;

    FloatM voxel_size = 0;
    FloatM min_expansion = 0;
    FloatM max_expansion = 0;

    /**
     * Post-set attributes
     */
    using BaseType::E_hat;
    using BaseType::fail_strain;

    FloatM mass = 0;          //!< Cached mass of this voxel (kg)
    FloatM mass_inverse = 0;  //!< Cached 1/Mass (1/kg)
    FloatM sqrt_mass = 0;     //!< Cached sqrt(mass). (sqrt(Kg))
    FloatM first_moment = 0;  //!< Cached 1st moment "inertia" (needed for certain
    //!< calculations) (kg*m)
    FloatM moment_inertia = 0;  //!< Cached mass moment of inertia (i.e. rotational
    //!< "mass") (kg*m^2)
    FloatM moment_inertia_inverse = 0;  //!< Cached 1/Inertia (1/(kg*m^2))

    // Damping coefficients

    // Note:
    // internal_damping_c is different from other damping coefficients
    // refer to sq* variables in RS_LinkMaterial
    // internal_damping_c is the factored out 2*zeta_internal*sqrt(m)
    FloatM internal_damping_c = 0;

    FloatM global_damping_translate_c = 0;
    FloatM global_damping_rotate_c = 0;
    FloatM contact_damping_translate_c = 0;
    FloatM contact_damping_rotate_c = 0;

    FloatM penetration_stiffness = 0;
    FloatM gravity = 0;
};

}  // namespace logic

}  // namespace rise

#define REFLECT_VOXEL_MATERIAL_MEM(voxel_material_mem_type) \
    REFL_AUTO(                                              \
        type(voxel_material_mem_type),                      \
        field(E),                                           \
        field(nu),                                          \
        field(rho),                                         \
        field(fail_stress),                                 \
        field(u_static),                                    \
        field(u_dynamic),                                   \
        field(zeta_internal),                               \
        field(zeta_contact),                                \
        field(zeta_global),                                 \
        field(E_hat),                                       \
        field(fail_strain),                                 \
        field(structure_gid),                               \
        field(sid),                                         \
        field(material_gid),                                \
        field(voxel_size),                                  \
        field(min_expansion),                               \
        field(max_expansion),                               \
        field(mass),                                        \
        field(mass_inverse),                                \
        field(sqrt_mass),                                   \
        field(first_moment),                                \
        field(moment_inertia),                              \
        field(moment_inertia_inverse),                      \
        field(internal_damping_c),                          \
        field(global_damping_translate_c),                  \
        field(global_damping_rotate_c),                     \
        field(contact_damping_translate_c),                 \
        field(contact_damping_rotate_c),                    \
        field(penetration_stiffness),                       \
        field(gravity)                                      \
    )

REFLECT_VOXEL_MATERIAL_MEM(rise::logic::RS_VoxelMaterialMem<rise::RSE_FLOAT_32>)
REFLECT_VOXEL_MATERIAL_MEM(rise::logic::RS_VoxelMaterialMem<rise::RSE_FLOAT_64>)
#endif  // RS_VOXEL_MATERIAL_M_H
