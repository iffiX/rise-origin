// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_INIT_SIGNAL_H
#define RSPY_INIT_SIGNAL_H

#include "common/rspy_common.h"

/// Bind the RSC signal types (force, torque, transform).
void initRSPYSignal(py::module &m);

#endif  // RSPY_INIT_SIGNAL_H
