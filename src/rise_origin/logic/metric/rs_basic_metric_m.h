// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BASIC_METRIC_M_H
#define RS_BASIC_METRIC_M_H

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

namespace logic {

struct RS_MassDotPosMetricMem {
    RFloat mass_dot_x, mass_dot_y, mass_dot_z, mass;
    HOST_DEVICE_INLINE_QUALIFIER
    RS_MassDotPosMetricMem(RFloat mass_dot_x, RFloat mass_dot_y, RFloat mass_dot_z, RFloat mass)
        : mass_dot_x(mass_dot_x), mass_dot_y(mass_dot_y), mass_dot_z(mass_dot_z), mass(mass) {}
    HOST_DEVICE_INLINE_QUALIFIER RS_MassDotPosMetricMem
    operator+(const RS_MassDotPosMetricMem &mdp) const {
        return {
            mass_dot_x + mdp.mass_dot_x,
            mass_dot_y + mdp.mass_dot_y,
            mass_dot_z + mdp.mass_dot_z,
            mass + mdp.mass
        };
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_BASIC_METRIC_M_H
