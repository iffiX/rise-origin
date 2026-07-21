// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_INIT_WORLD_SESSION_H
#define RSPY_INIT_WORLD_SESSION_H

#include "common/rspy_common.h"
#include "core/rspy_init_world_callback.h"
#include "core/rspy_init_world_handle.h"

/// Bind the session scope enum and RS_WorldSession.
void initRSPYWorldSession(py::module &m);

#endif  // RSPY_INIT_WORLD_SESSION_H
