// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_COUPLING_M_H
#define RS_COUPLING_M_H

#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

namespace logic {

/**
 * Per-rigid-body soft-coupling correction, summed from its surface voxels by the reduce-by-group
 * machinery. Carries the linear and angular impulse the soft transmits to the body this soft step.
 */
struct RS_CouplingCorrection {
    RVec3f linear_impulse;   ///< Sigma coupling impulse on the body (kg m/s)
    RVec3f angular_impulse;  ///< Sigma coupling angular impulse about the com (kg m^2/s)

    RS_CouplingCorrection() = default;

    HOST_DEVICE_INLINE_QUALIFIER
    RS_CouplingCorrection(const RVec3f &linear_impulse, const RVec3f &angular_impulse)
        : linear_impulse(linear_impulse), angular_impulse(angular_impulse) {}

    HOST_DEVICE_INLINE_QUALIFIER static RS_CouplingCorrection zero() {
        return RS_CouplingCorrection(RVec3f::zero(), RVec3f::zero());
    }
};

class RS_ReduceCouplingCorrectionOp {
   public:
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return T(a.linear_impulse + b.linear_impulse, a.angular_impulse + b.angular_impulse);
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_COUPLING_M_H
