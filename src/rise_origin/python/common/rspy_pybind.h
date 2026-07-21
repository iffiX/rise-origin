// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_PYBIND_H
#define RSPY_PYBIND_H

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rise_origin/core/rs_world_callback.h"
#include "rise_origin/core/rs_world_controller.h"
#include "rise_origin/core/rs_world_def.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_session.h"
#include "rise_origin/core/rs_world_session_component.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/rise_origin.h"
#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_file.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/utils/memory/rs_device.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/string/rs_string.h"

PYBIND11_MAKE_OPAQUE(std::vector<int16_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint16_t>);
PYBIND11_MAKE_OPAQUE(std::vector<int32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<int64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<char>);
PYBIND11_MAKE_OPAQUE(std::vector<unsigned char>);

PYBIND11_MAKE_OPAQUE(std::vector<bool>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);

PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_MathExprToken>);

namespace py = pybind11;
using namespace rise;
using namespace rise::rsc;

#endif  // RSPY_PYBIND_H
