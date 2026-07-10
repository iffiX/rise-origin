// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_CORE_INIT_WORLD_SESSION_H
#define RSPY_CORE_INIT_WORLD_SESSION_H

#include "core/init_world_callback.h"
#include "core/init_world_handle.h"
#include "init_common.h"

/// Bind the session scope enum and RS_WorldSession.
void initRSPYWorldSession(py::module &m);

#endif  // RSPY_CORE_INIT_WORLD_SESSION_H
