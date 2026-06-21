// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef INIT_DEF_H
#define INIT_DEF_H

#include <pybind11/stl_bind.h>

#include "rise_origin/rise_origin.h"

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

inline void initRSPYDefinitions(py::module &m) {
    m.attr("RS_NULL_INDEX") = RS_NULL_INDEX;
    m.attr("RS_WORLD_MESSAGE_SYSTEM_ID") = RS_WORLD_MESSAGE_SYSTEM_ID;

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

    py::class_<RVec3rf>(m, "RVec3rf")
        .def(py::init<>())
        .def(py::init<RFloat, RFloat, RFloat>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_static("zero", &RVec3rf::zero)
        .def_static("one", &RVec3rf::one)
        .def_static("full", &RVec3rf::full, py::arg("value"))
        .def_readwrite("x", &RVec3rf::x)
        .def_readwrite("y", &RVec3rf::y)
        .def_readwrite("z", &RVec3rf::z);

    py::class_<RQuat3rf>(m, "RQuat3rf")
        .def(py::init<>())
        .def(
            py::init<RFloat, RFloat, RFloat, RFloat>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("w")
        )
        .def_static("unit", &RQuat3rf::unit)
        .def_readwrite("x", &RQuat3rf::x)
        .def_readwrite("y", &RQuat3rf::y)
        .def_readwrite("z", &RQuat3rf::z)
        .def_readwrite("w", &RQuat3rf::w);

    py::class_<RMat3rf>(m, "RMat3rf")
        .def(py::init<>())
        .def(
            py::init<RFloat, RFloat, RFloat, RFloat, RFloat, RFloat, RFloat, RFloat, RFloat>(),
            py::arg("m00"),
            py::arg("m01"),
            py::arg("m02"),
            py::arg("m10"),
            py::arg("m11"),
            py::arg("m12"),
            py::arg("m20"),
            py::arg("m21"),
            py::arg("m22")
        )
        .def_static("zero", &RMat3rf::zero)
        .def_static("eye", &RMat3rf::eye)
        .def_static("diagonal", &RMat3rf::diagonal, py::arg("m00"), py::arg("m11"), py::arg("m22"))
        .def(
            "__getitem__",
            [](const RMat3rf &self, std::pair<int, int> idx) {
                if (idx.first < 0 || idx.first >= 3 || idx.second < 0 || idx.second >= 3) {
                    throw py::index_error("RMat3rf index out of range");
                }
                return self[idx.first][idx.second];
            }
        )
        .def(
            "__setitem__",
            [](RMat3rf &self, std::pair<int, int> idx, RFloat v) {
                if (idx.first < 0 || idx.first >= 3 || idx.second < 0 || idx.second >= 3) {
                    throw py::index_error("RMat3rf index out of range");
                }
                self[idx.first][idx.second] = v;
            }
        )
        .def(
            "to_list",
            [](const RMat3rf &self) {
                std::vector<std::vector<RFloat>> rows(3, std::vector<RFloat>(3));
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) rows[r][c] = self[r][c];
                return rows;
            }
        )
        .def("from_list", [](RMat3rf &self, const std::vector<std::vector<RFloat>> &rows) {
            if (rows.size() != 3) throw py::value_error("RMat3rf.from_list expects 3 rows");
            for (int r = 0; r < 3; ++r) {
                if (rows[r].size() != 3)
                    throw py::value_error("RMat3rf.from_list expects 3 columns per row");
                for (int c = 0; c < 3; ++c) self[r][c] = rows[r][c];
            }
        });

    py::class_<RColor3b>(m, "RColor3b")
        .def(py::init<>())
        .def(
            py::init<unsigned char, unsigned char, unsigned char>(),
            py::arg("r"),
            py::arg("g"),
            py::arg("b")
        )
        .def_static("zero", &RColor3b::zero)
        .def_readwrite("r", &RColor3b::r)
        .def_readwrite("g", &RColor3b::g)
        .def_readwrite("b", &RColor3b::b);

    py::class_<RColor3f>(m, "RColor3f")
        .def(py::init<>())
        .def(py::init<float, float, float>(), py::arg("r"), py::arg("g"), py::arg("b"))
        .def_static("zero", &RColor3f::zero)
        .def_readwrite("r", &RColor3f::r)
        .def_readwrite("g", &RColor3f::g)
        .def_readwrite("b", &RColor3f::b);

    py::class_<RBound3rf>(m, "RBound3rf")
        .def(py::init<>())
        .def(
            py::init<RFloat, RFloat, RFloat, RFloat, RFloat, RFloat>(),
            py::arg("min_x"),
            py::arg("min_y"),
            py::arg("min_z"),
            py::arg("max_x"),
            py::arg("max_y"),
            py::arg("max_z")
        )
        .def_static("zero", &RBound3rf::zero)
        .def_static("lowest", &RBound3rf::lowest)
        .def_static("highest", &RBound3rf::highest)
        .def_readwrite("min_x", &RBound3rf::min_x)
        .def_readwrite("max_x", &RBound3rf::max_x)
        .def_readwrite("min_y", &RBound3rf::min_y)
        .def_readwrite("max_y", &RBound3rf::max_y)
        .def_readwrite("min_z", &RBound3rf::min_z)
        .def_readwrite("max_z", &RBound3rf::max_z);

    py::class_<RBound3rui>(m, "RBound3rui")
        .def(py::init<>())
        .def(
            py::init<RIndex, RIndex, RIndex, RIndex, RIndex, RIndex>(),
            py::arg("min_x"),
            py::arg("min_y"),
            py::arg("min_z"),
            py::arg("max_x"),
            py::arg("max_y"),
            py::arg("max_z")
        )
        .def_static("zero", &RBound3rui::zero)
        .def_static("lowest", &RBound3rui::lowest)
        .def_static("highest", &RBound3rui::highest)
        .def_readwrite("min_x", &RBound3rui::min_x)
        .def_readwrite("max_x", &RBound3rui::max_x)
        .def_readwrite("min_y", &RBound3rui::min_y)
        .def_readwrite("max_y", &RBound3rui::max_y)
        .def_readwrite("min_z", &RBound3rui::min_z)
        .def_readwrite("max_z", &RBound3rui::max_z);

    py::class_<RBound3ri>(m, "RBound3ri")
        .def(py::init<>())
        .def(
            py::init<
                RSignedIndex,
                RSignedIndex,
                RSignedIndex,
                RSignedIndex,
                RSignedIndex,
                RSignedIndex>(),
            py::arg("min_x"),
            py::arg("min_y"),
            py::arg("min_z"),
            py::arg("max_x"),
            py::arg("max_y"),
            py::arg("max_z")
        )
        .def_static("zero", &RBound3ri::zero)
        .def_static("lowest", &RBound3ri::lowest)
        .def_static("highest", &RBound3ri::highest)
        .def_readwrite("min_x", &RBound3ri::min_x)
        .def_readwrite("max_x", &RBound3ri::max_x)
        .def_readwrite("min_y", &RBound3ri::min_y)
        .def_readwrite("max_y", &RBound3ri::max_y)
        .def_readwrite("min_z", &RBound3ri::min_z)
        .def_readwrite("max_z", &RBound3ri::max_z);

    py::enum_<RSE_MathExprOperator>(m, "RSE_MathExprOperator")
        .value("RSE_ME_END", RSE_ME_END)
        .value("RSE_ME_CONST", RSE_ME_CONST)
        .value("RSE_ME_E", RSE_ME_E)
        .value("RSE_ME_PI", RSE_ME_PI)
        .value("RSE_ME_VAR", RSE_ME_VAR)
        .value("RSE_ME_ADD", RSE_ME_ADD)
        .value("RSE_ME_SUB", RSE_ME_SUB)
        .value("RSE_ME_MUL", RSE_ME_MUL)
        .value("RSE_ME_DIV", RSE_ME_DIV)
        .value("RSE_ME_MOD", RSE_ME_MOD)
        .value("RSE_ME_POW", RSE_ME_POW)
        .value("RSE_ME_SQRT", RSE_ME_SQRT)
        .value("RSE_ME_SIN", RSE_ME_SIN)
        .value("RSE_ME_COS", RSE_ME_COS)
        .value("RSE_ME_TAN", RSE_ME_TAN)
        .value("RSE_ME_ATAN", RSE_ME_ATAN)
        .value("RSE_ME_LOG", RSE_ME_LOG)
        .value("RSE_ME_INT", RSE_ME_INT)
        .value("RSE_ME_ABS", RSE_ME_ABS)
        .value("RSE_ME_GREATER_THAN", RSE_ME_GREATER_THAN)
        .value("RSE_ME_LESS_THAN", RSE_ME_LESS_THAN)
        .value("RSE_ME_GREATER_OR_EQUAL", RSE_ME_GREATER_OR_EQUAL)
        .value("RSE_ME_LESS_OR_EQUAL", RSE_ME_LESS_OR_EQUAL)
        .value("RSE_ME_NOT_EQUAL", RSE_ME_NOT_EQUAL)
        .value("RSE_ME_EQUAL", RSE_ME_EQUAL)
        .value("RSE_ME_AND", RSE_ME_AND)
        .value("RSE_ME_OR", RSE_ME_OR)
        .export_values();

    py::class_<RS_MathExprToken>(m, "RS_MathExprToken")
        .def(py::init<>())
        .def(py::init<RSE_MathExprOperator>())
        .def(py::init([](RSE_MathExprOperator op, py::object val) {
            RS_MathExprToken token(op);
            if (op == RSE_ME_CONST) {
                token.value = val.cast<RFloat>();
            } else if (op == RSE_ME_VAR) {
                token.value = val.cast<std::vector<std::string>>();
            }
            return token;
        }))
        .def_readwrite("op", &RS_MathExprToken::op)
        .def_property(
            "value",
            // Getter
            [](const RS_MathExprToken &token) -> py::object {
                if (token.op == RSE_ME_CONST) {
                    try {
                        return py::cast(token.value.as<RFloat>());
                    } catch (const RS_InvalidCastException &) {
                        return py::none();
                    }
                } else if (token.op == RSE_ME_VAR) {
                    try {
                        return py::cast(token.value.as<std::vector<std::string>>());
                    } catch (const RS_InvalidCastException &) {
                        return py::none();
                    }
                }
                return py::none();
            },
            // Setter
            [](RS_MathExprToken &token, py::object val) {
                if (token.op == RSE_ME_CONST) {
                    token.value = val.cast<RFloat>();
                } else if (token.op == RSE_ME_VAR) {
                    token.value = val.cast<std::vector<std::string>>();
                }
            }
        );

    py::class_<RS_MathExpr>(m, "RS_MathExpr")
        .def(py::init<>())
        .def("empty", &RS_MathExpr::empty)
        .def("__getitem__", [](const RS_MathExpr &expr, int idx) { return expr[idx]; })
        .def(
            "__setitem__",
            [](RS_MathExpr &expr, int idx, const RS_MathExprToken &token) { expr[idx] = token; }
        )
        .def_readwrite("expr", &RS_MathExpr::expr)
        .def(
            "__str__",
            [](const RS_MathExpr &expr) {
                std::stringstream ss;
                ss << "MathExpr(" << expr.expr.size() << " tokens)";
                return ss.str();
            }
        )
        .def("from_string", [](RS_MathExpr &expr, const std::string &expr_string) {
            expr = expr_string;
            return expr;
        });
}
#endif  // INIT_DEF_H
