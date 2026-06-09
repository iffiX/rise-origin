// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)
//
// C++ shim for the vendored Hosek-Wilkie sky model. The upstream
// ArHosekSkyModel.h is C99 without linkage guards, so this wrapper supplies
// the extern "C" block once and offers a clean C++ include surface to the
// rest of rise-origin. No upstream sources are modified.

#ifndef HOSEK_WILKIE_HPP
#define HOSEK_WILKIE_HPP

extern "C" {
#include "ArHosekSkyModel.h"
}

#endif  // HOSEK_WILKIE_HPP
