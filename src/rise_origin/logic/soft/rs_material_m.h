// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MATERIAL_M_H
#define RS_MATERIAL_M_H

#include "refl.hpp"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_float.h"

namespace rise {

namespace logic {

template <RSE_FloatType FloatMemAccuracy>
class RS_MaterialMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;

    /**
     * Pre-set attributes
     */
    // All physical variables, name begins with their physical quantity
    // and end with meaning
    FloatM E = 0;                                          //!< Young's modulus (stiffness) in Pa.
    FloatM nu = 0;                                         //!< Poissons Ratio
    FloatM rho = 0;                                        //!< Density in Kg/m^3
    FloatM fail_stress = RS_NumTraits<FloatM>::highest();  //!< Failure stress in Pa.
    FloatM u_static = 0;                                   //!< Static coefficient of friction
    FloatM u_dynamic = 0;                                  //!< Dynamic coefficient of friction
    FloatM zeta_internal = 0;                              //!< Internal damping ratio
    FloatM zeta_contact = 0;                               //!< Contact damping ratio
    FloatM zeta_global = 0;                                //!< Global damping ratio

    /**
     * Post-set attributes
     */
    /// Effective elastic modulus for materials with non-zero Poisson's ratio.
    FloatM E_hat = 0;

    /// Failure strain
    FloatM fail_strain = 0;
};

}  // namespace logic

}  // namespace rise

#define REFLECT_MATERIAL_MEM(material_mem_type) \
    REFL_AUTO(                                  \
        type(material_mem_type),                \
        field(E),                               \
        field(nu),                              \
        field(rho),                             \
        field(fail_stress),                     \
        field(u_static),                        \
        field(u_dynamic),                       \
        field(zeta_internal),                   \
        field(zeta_contact),                    \
        field(zeta_global),                     \
        field(E_hat),                           \
        field(fail_strain)                      \
    )

REFLECT_MATERIAL_MEM(rise::logic::RS_MaterialMem<rise::RSE_FLOAT_32>)
REFLECT_MATERIAL_MEM(rise::logic::RS_MaterialMem<rise::RSE_FLOAT_64>)
#endif  // RS_MATERIAL_M_H
