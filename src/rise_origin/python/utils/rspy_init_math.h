// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_INIT_MATH_H
#define RSPY_INIT_MATH_H

#include "common/rspy_common.h"

/// Bind the math value types (vectors, quaternion, matrix, colors, bounds) and math expressions.
void initRSPYMath(py::module &m);

#endif  // RSPY_INIT_MATH_H
