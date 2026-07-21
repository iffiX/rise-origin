// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_INIT_WORLD_H
#define RSPY_INIT_WORLD_H

#include "common/rspy_common.h"

/// Bind the RSC variant wrappers, RSC_World, and the validator.
void initRSPYWorld(py::module &m);

#endif  // RSPY_INIT_WORLD_H
