// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_CORE_INIT_WORLD_SESSION_COMPONENT_H
#define RSPY_CORE_INIT_WORLD_SESSION_COMPONENT_H

#include "core/init_world_controller.h"
#include "core/init_world_handle.h"
#include "init_common.h"

/// Bind the session component base and its Generic/Camera/VR/MCP variants.
void initRSPYWorldSessionComponent(py::module &m);

#endif  // RSPY_CORE_INIT_WORLD_SESSION_COMPONENT_H
