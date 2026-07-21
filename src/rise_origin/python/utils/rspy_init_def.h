// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_INIT_DEF_H
#define RSPY_INIT_DEF_H

#include "common/rspy_common.h"

/// Bind module-level constants, the float-type enum, and the opaque primitive vector types.
void initRSPYDef(py::module &m);

#endif  // RSPY_INIT_DEF_H
