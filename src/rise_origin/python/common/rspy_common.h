// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_COMMON_H
#define RSPY_COMMON_H

#include "common/rspy_numpy.h"
#include "common/rspy_optional.h"
#include "common/rspy_pybind.h"
#include "common/rspy_rsc_class.h"
#include "common/rspy_rsc_vec.h"
#include "common/rspy_serialize.h"
#include "common/rspy_traits.h"
#include "common/rspy_variant.h"
#include "common/rspy_vector_view.h"

/**
 * @brief Register the module-level common types, such as the shared vector view. Called once.
 * @param m The module to register into.
 */
void initRSPYCommon(py::module &m);

#endif  // RSPY_COMMON_H
