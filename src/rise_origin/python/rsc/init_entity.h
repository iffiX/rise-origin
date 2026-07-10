// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_RSC_INIT_ENTITY_H
#define RSPY_RSC_INIT_ENTITY_H

#include "init_common.h"

/// Bind the RSC entity types (background, floor, structure, cameras, ray sensor).
void initRSPYEntity(py::module &m);

#endif  // RSPY_RSC_INIT_ENTITY_H
