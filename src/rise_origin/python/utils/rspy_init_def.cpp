// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "utils/rspy_init_def.h"

void initRSPYDef(py::module &m) {
    m.attr("RS_ENGINE_VERSION") = RS_ENGINE_VERSION;
    m.attr("RS_NULL_INDEX") = RS_NULL_INDEX;
    m.attr("RS_NULL_BIG_INDEX") = RS_NULL_BIG_INDEX;

    py::enum_<RSE_FloatType>(m, "RSE_FloatType")
        .value("RSE_FLOAT_32", RSE_FLOAT_32)
        .value("RSE_FLOAT_64", RSE_FLOAT_64)
        .export_values();

    py::bind_vector<std::vector<int16_t>>(m, "VecInt16");
    py::bind_vector<std::vector<uint16_t>>(m, "VecUInt16");
    py::bind_vector<std::vector<int32_t>>(m, "VecInt32");
    py::bind_vector<std::vector<uint32_t>>(m, "VecUInt32");
    py::bind_vector<std::vector<int64_t>>(m, "VecInt64");
    py::bind_vector<std::vector<uint64_t>>(m, "VecUInt64");
    py::bind_vector<std::vector<char>>(m, "VecChar");
    py::bind_vector<std::vector<unsigned char>>(m, "VecUChar");

    py::bind_vector<std::vector<bool>>(m, "VecBool");
    py::bind_vector<std::vector<float>>(m, "VecFloat");
    py::bind_vector<std::vector<double>>(m, "VecDouble");

    py::bind_vector<std::vector<std::string>>(m, "VecString");
    py::bind_vector<std::vector<RS_MathExprToken>>(m, "VecMathExprToken");
}
