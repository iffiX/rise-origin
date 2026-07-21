// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "common/rspy_common.h"

void initRSPYCommon(py::module &m) { rspyRegisterVectorView(m); }
