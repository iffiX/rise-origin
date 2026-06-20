// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MATERIAL_C_H
#define RS_MATERIAL_C_H

#include <cfloat>

#include "rise_origin/utils/rs_exception.h"
#include "rs_material_m.h"

namespace rise {

namespace logic {

template <RSE_FloatType FloatComputeAccuracy>
class RS_MaterialCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;

    /*****************************************************************
     * Host side methods
     *****************************************************************/

    /// @brief Initializer. This function is called by the kernel manager.
    template <typename RS_MaterialMem>
    HOST_INLINE_QUALIFIER static void initMaterial(RS_MaterialMem &m) {
        // Bound pre-set values
        m.E = RS_MAX(m.E, 0);
        // exactly 0.5 will still cause problems, but it can get very close.
        m.nu = RS_BOUND(m.nu, 0, 0.5 - FLT_EPSILON * 2);
        m.rho = RS_MAX(m.rho, 0);
        m.zeta_internal = RS_MAX(m.zeta_internal, 0);
        m.zeta_contact = RS_MAX(m.zeta_contact, 0);
        m.zeta_global = RS_MAX(m.zeta_global, 0);

        if (m.fail_stress < 0 && m.fail_stress != -1)
            RS_THROW(RS_InvalidCharacterizationException("Failure stress is negative"));

        if (m.fail_stress > 0 && m.E > 0)
            m.fail_strain = m.fail_stress / m.E;
        else
            m.fail_strain = RS_NumTraits<typename RS_MaterialMem::FloatM>::highest();
        // update derived values
        m.E_hat = m.E / ((1 - 2 * m.nu) * (1 + m.nu));
    }

    /*****************************************************************
     * Device side methods for acquiring info
     *****************************************************************///

    /// @brief Returns the stress of the material model accounting for volumetric strain effects.
    /// @param [in] m_E The Young's modulus.
    /// @param [in] m_E_hat The Young's modulus divided by ((1 - 2 * nu) * (1 + nu)).
    /// @param [in] m_nu The Poisson's ratio.
    /// @param [in] strain The strain to query.
    /// @param [in] transverse_strain_sum The sum of the two principle normal strains in the
    /// transverse plane perpendicular to strain.
    HOST_DEVICE_INLINE_QUALIFIER static FloatC
    stress(FloatC m_E, FloatC m_E_hat, FloatC m_nu, FloatC strain, FloatC transverse_strain_sum) {
        // reference:
        // http://www.colorado.edu/engineering/CAS/courses.d/Structures.d/
        // IAST.Lect05.d/IAST.Lect05.pdf
        // page 10

        if (isXyzIndependent(m_nu)) return m_E * strain;
        return m_E_hat * ((1 - m_nu) * strain + m_nu * transverse_strain_sum);
    }

    /// @brief Returns true if the specified strain is past the failure point.
    /// @param [in] m_fail_strain The failure strain.
    /// @param [in] strain The strain to query.
    HOST_DEVICE_INLINE_QUALIFIER static bool isFailed(FloatC m_fail_strain, FloatC strain) {
        return strain > m_fail_strain;
    }

    /// @brief Returns true if poisson's ratio is zero - i.e. deformations in each dimension
    /// are independent of those in other dimensions.
    /// @param [in] m_nu The Poisson's ratio.
    HOST_DEVICE_INLINE_QUALIFIER static bool isXyzIndependent(FloatC m_nu) { return m_nu == 0; }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_MATERIAL_C_H