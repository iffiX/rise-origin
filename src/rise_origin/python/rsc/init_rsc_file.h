// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_RSC_INIT_RSC_FILE_H
#define RSPY_RSC_INIT_RSC_FILE_H

#include "init_common.h"

/// Bind RS_RSCFile: type-erased save/load of any RSC node plus version helpers.
void initRSPYRSCFile(py::module &m);

#endif  // RSPY_RSC_INIT_RSC_FILE_H
