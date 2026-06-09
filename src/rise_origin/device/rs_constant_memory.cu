// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/utils/rs_conf.h"
#include "rs_constant_memory.h"

#ifdef __CUDACC__

__constant__ char rs_cuda_constant_memory[RS_CONSTANT_MEMORY_MAX_SIZE];

#endif

char rs_host_constant_memory[RS_CONSTANT_MEMORY_MAX_SIZE];