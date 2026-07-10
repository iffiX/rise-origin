// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "utils/init_math.h"

#include <pybind11/operators.h>

#include <string>

namespace {

/// Bind the operators and member functions shared by every RS_VecND<RFloat> dimension.
template <typename V>
void rspyBindVecCommon(py::class_<V> &c) {
    c.def_static("zero", &V::zero)
        .def_static("one", &V::one)
        .def_static("full", &V::full, py::arg("value"))
        .def_static("highest", &V::highest)
        .def_static("lowest", &V::lowest);

    c.def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self * py::self)
        .def(py::self * RFloat())
        .def(RFloat() * py::self)
        .def(py::self / RFloat())
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= RFloat())
        .def(py::self /= RFloat())
        .def(py::self == py::self)
        .def(py::self != py::self);

    c.def("__getitem__", [](V &v, int i) {
         return v[i];
     }).def("__setitem__", [](V &v, int i, RFloat value) { v[i] = value; });

    c.def(
         "dot", [](const V &a, const V &b) { return a.dot(b); }, py::arg("other")
    )
        .def("length", [](const V &v) { return v.length(); })
        .def("length2", [](const V &v) { return v.length2(); })
        .def("normalized", [](const V &v) { return v.normalized(); })
        .def("abs", [](const V &v) { return v.abs(); })
        .def("floor", [](const V &v) { return v.floor(); })
        .def("ceil", [](const V &v) { return v.ceil(); })
        .def("round", [](const V &v) { return v.round(); })
        .def(
            "clip",
            [](V &v, RFloat low, RFloat high) { return v.clip(low, high); },
            py::arg("low"),
            py::arg("high")
        )
        .def("prod", [](const V &v) { return v.prod(); })
        .def("mean", [](const V &v) { return v.mean(); })
        .def("sum", [](const V &v) { return v.sum(); })
        .def("min", [](const V &v) { return v.min(); })
        .def(
            "min", [](const V &a, const V &b) { return a.min(b); }, py::arg("other")
        )
        .def("max", [](const V &v) { return v.max(); })
        .def(
            "max", [](const V &a, const V &b) { return a.max(b); }, py::arg("other")
        )
        .def("argmin", [](const V &v) { return v.argmin(); })
        .def("argmax", [](const V &v) { return v.argmax(); })
        .def(
            "elem_mul", [](const V &a, const V &b) { return a.elemMul(b); }, py::arg("other")
        )
        .def(
            "elem_div", [](const V &a, const V &b) { return a.elemDiv(b); }, py::arg("other")
        )
        .def(
            "elem_div_fast",
            [](const V &a, const V &b) { return a.elemDivFast(b); },
            py::arg("other")
        )
        .def("reciprocal", [](const V &v) { return v.reciprocal(); })
        .def("reciprocal_fast", [](const V &v) { return v.reciprocalFast(); })
        .def(
            "dist", [](const V &a, const V &b) { return a.dist(b); }, py::arg("other")
        )
        .def(
            "dist2", [](const V &a, const V &b) { return a.dist2(b); }, py::arg("other")
        )
        .def(
            "project", [](const V &a, const V &b) { return a.project(b); }, py::arg("other")
        )
        .def("is_valid", [](const V &v) { return v.isValid(); })
        .def(
            "is_in_bound",
            [](const V &v, RFloat low, RFloat high) { return v.isInBound(low, high); },
            py::arg("low"),
            py::arg("high")
        );
}

}  // namespace

void initRSPYMath(py::module &m) {
    py::class_<RVec2rf> vec2(m, "RVec2rf", "2D float vector");
    vec2.def(py::init<>())
        .def(py::init<RFloat, RFloat>(), py::arg("x"), py::arg("y"))
        .def(py::init([](const py::sequence &s) {
            if (py::len(s) != 2) {
                throw std::runtime_error("RVec2rf expects a sequence of 2 values");
            }
            return RVec2rf(s[0].cast<RFloat>(), s[1].cast<RFloat>());
        }))
        .def_readwrite("x", &RVec2rf::x)
        .def_readwrite("y", &RVec2rf::y)
        .def("__repr__", [](const RVec2rf &v) {
            return "RVec2rf(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
        });
    rspyBindVecCommon(vec2);
    py::implicitly_convertible<py::sequence, RVec2rf>();

    py::class_<RVec3rf> vec3(m, "RVec3rf", "3D float vector");
    vec3.def(py::init<>())
        .def(py::init<RFloat, RFloat, RFloat>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def(py::init([](const py::sequence &s) {
            if (py::len(s) != 3) {
                throw std::runtime_error("RVec3rf expects a sequence of 3 values");
            }
            return RVec3rf(s[0].cast<RFloat>(), s[1].cast<RFloat>(), s[2].cast<RFloat>());
        }))
        .def_readwrite("x", &RVec3rf::x)
        .def_readwrite("y", &RVec3rf::y)
        .def_readwrite("z", &RVec3rf::z)
        .def(
            "cross", [](const RVec3rf &a, const RVec3rf &b) { return a.cross(b); }, py::arg("other")
        )
        .def(
            "rotate",
            [](RVec3rf &v, const RVec3rf &axis, RFloat angle) { return v.rotate(axis, angle); },
            py::arg("axis"),
            py::arg("angle")
        )
        .def(
            "rotate_x", [](RVec3rf &v, RFloat angle) { return v.rotateX(angle); }, py::arg("angle")
        )
        .def(
            "rotate_y", [](RVec3rf &v, RFloat angle) { return v.rotateY(angle); }, py::arg("angle")
        )
        .def(
            "rotate_z", [](RVec3rf &v, RFloat angle) { return v.rotateZ(angle); }, py::arg("angle")
        )
        .def("arbitrary_normal", [](const RVec3rf &v) { return v.arbitraryNormal(); })
        .def("to_cross_matrix", [](const RVec3rf &v) { return v.toCrossMatrix(); })
        .def("__repr__", [](const RVec3rf &v) {
            return "RVec3rf(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " +
                   std::to_string(v.z) + ")";
        });
    rspyBindVecCommon(vec3);
    py::implicitly_convertible<py::sequence, RVec3rf>();

    py::class_<RVec4rf> vec4(m, "RVec4rf", "4D float vector");
    vec4.def(py::init<>())
        .def(
            py::init<RFloat, RFloat, RFloat, RFloat>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("w")
        )
        .def(py::init([](const py::sequence &s) {
            if (py::len(s) != 4) {
                throw std::runtime_error("RVec4rf expects a sequence of 4 values");
            }
            return RVec4rf(
                s[0].cast<RFloat>(), s[1].cast<RFloat>(), s[2].cast<RFloat>(), s[3].cast<RFloat>()
            );
        }))
        .def_readwrite("x", &RVec4rf::x)
        .def_readwrite("y", &RVec4rf::y)
        .def_readwrite("z", &RVec4rf::z)
        .def_readwrite("w", &RVec4rf::w)
        .def("__repr__", [](const RVec4rf &v) {
            return "RVec4rf(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " +
                   std::to_string(v.z) + ", " + std::to_string(v.w) + ")";
        });
    rspyBindVecCommon(vec4);
    py::implicitly_convertible<py::sequence, RVec4rf>();

    py::class_<RQuat3rf>(m, "RQuat3rf", "3D float quaternion")
        .def(py::init<>())
        .def(
            py::init<RFloat, RFloat, RFloat, RFloat>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("w")
        )
        .def(py::init<RFloat, const RVec3rf &>(), py::arg("angle"), py::arg("axis"))
        .def(
            py::init<const RVec3rf &, const RVec3rf &>(),
            py::arg("from_direction"),
            py::arg("to_direction")
        )
        .def(py::init([](const py::sequence &s) {
            if (py::len(s) != 4) {
                throw std::runtime_error("RQuat3rf expects a sequence of 4 values");
            }
            return RQuat3rf(
                s[0].cast<RFloat>(), s[1].cast<RFloat>(), s[2].cast<RFloat>(), s[3].cast<RFloat>()
            );
        }))
        .def_static("unit", &RQuat3rf::unit)
        .def_readwrite("x", &RQuat3rf::x)
        .def_readwrite("y", &RQuat3rf::y)
        .def_readwrite("z", &RQuat3rf::z)
        .def_readwrite("w", &RQuat3rf::w)
        .def(
            "__add__", [](RQuat3rf a, const RQuat3rf &b) { return a + b; }, py::is_operator()
        )
        .def(
            "__sub__", [](RQuat3rf a, const RQuat3rf &b) { return a - b; }, py::is_operator()
        )
        .def(py::self * py::self)
        .def(py::self * RFloat())
        .def(RFloat() * py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("to_vec", [](const RQuat3rf &q) { return q.toVec(); })
        .def("length", [](const RQuat3rf &q) { return q.length(); })
        .def("length2", [](const RQuat3rf &q) { return q.length2(); })
        .def("normalized", [](const RQuat3rf &q) { return q.normalized(); })
        .def("inverse", [](const RQuat3rf &q) { return q.inverse(); })
        .def("conjugate", [](const RQuat3rf &q) { return q.conjugate(); })
        .def(
            "slerp",
            [](const RQuat3rf &q, const RQuat3rf &to, RFloat t) { return q.slerp(to, t); },
            py::arg("to"),
            py::arg("t")
        )
        .def("angle", [](const RQuat3rf &q) { return q.angle(); })
        .def("angle_degrees", [](const RQuat3rf &q) { return q.angleDegrees(); })
        .def("is_negligible_angle", [](const RQuat3rf &q) { return q.isNegligibleAngle(); })
        .def("is_small_angle", [](const RQuat3rf &q) { return q.isSmallAngle(); })
        .def("axis", [](const RQuat3rf &q) { return q.axis(); })
        .def("axis_un_normalized", [](const RQuat3rf &q) { return q.axisUnNormalized(); })
        .def("to_rotation_matrix", [](const RQuat3rf &q) { return q.toRotationMatrix(); })
        .def("to_rotation_vector", [](const RQuat3rf &q) { return q.toRotationVector(); })
        .def(
            "rotate_vec3d",
            [](const RQuat3rf &q, const RVec3rf &v) { return q.rotateVec3D(v); },
            py::arg("vector")
        )
        .def(
            "rotate_vec3d_inv",
            [](const RQuat3rf &q, const RVec3rf &v) { return q.rotateVec3DInv(v); },
            py::arg("vector")
        )
        .def("to_ortho_axis_x", [](const RQuat3rf &q) { return q.toOrthoAxisX(); })
        .def("to_ortho_axis_y", [](const RQuat3rf &q) { return q.toOrthoAxisY(); })
        .def("to_ortho_axis_z", [](const RQuat3rf &q) { return q.toOrthoAxisZ(); })
        .def("is_valid", [](const RQuat3rf &q) { return q.isValid(); })
        .def(
            "from_rotation_matrix",
            [](RQuat3rf &q, const RMat3rf &mat) { return q.fromRotationMatrix(mat); },
            py::arg("matrix")
        )
        .def(
            "from_rotation_vector",
            [](RQuat3rf &q, const RVec3rf &vec) { return q.fromRotationVector(vec); },
            py::arg("vector")
        )
        .def(
            "from_angle_to_pos_x",
            [](RQuat3rf &q, const RVec3rf &vec) { return q.fromAngleToPosX(vec); },
            py::arg("rotate_from")
        )
        .def("__repr__", [](const RQuat3rf &q) {
            return "RQuat3rf(" + std::to_string(q.x) + ", " + std::to_string(q.y) + ", " +
                   std::to_string(q.z) + ", " + std::to_string(q.w) + ")";
        });
    py::implicitly_convertible<py::sequence, RQuat3rf>();

    py::class_<RMat3rf>(m, "RMat3rf", "3x3 float matrix (row-major)")
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
        .def(
            py::init([](const RVec3rf &r0, const RVec3rf &r1, const RVec3rf &r2) {
                return RMat3rf(r0, r1, r2);
            }),
            py::arg("row0"),
            py::arg("row1"),
            py::arg("row2")
        )
        .def_static("zero", &RMat3rf::zero)
        .def_static("eye", &RMat3rf::eye)
        .def_static("full", &RMat3rf::full, py::arg("value"))
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
            "__call__",
            [](const RMat3rf &self, int row, int col) {
                if (row < 0 || row >= 3 || col < 0 || col >= 3) {
                    throw py::index_error("RMat3rf index out of range");
                }
                return self(row, col);
            },
            py::arg("row"),
            py::arg("col")
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
        .def(
            "from_list",
            [](RMat3rf &self, const std::vector<std::vector<RFloat>> &rows) {
                if (rows.size() != 3) throw py::value_error("RMat3rf.from_list expects 3 rows");
                for (int r = 0; r < 3; ++r) {
                    if (rows[r].size() != 3)
                        throw py::value_error("RMat3rf.from_list expects 3 columns per row");
                    for (int c = 0; c < 3; ++c) self[r][c] = rows[r][c];
                }
            }
        )
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self * py::self)
        .def(py::self * RVec3rf())
        .def(RVec3rf() * py::self)
        .def(py::self * RFloat())
        .def(RFloat() * py::self)
        .def(py::self / RFloat())
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= RFloat())
        .def(py::self /= RFloat())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("transpose", [](const RMat3rf &self) { return self.transpose(); })
        .def("determinant", [](const RMat3rf &self) { return self.determinant(); })
        .def("adjoint", [](const RMat3rf &self) { return self.adjoint(); })
        .def("inverse", [](const RMat3rf &self) { return self.inverse(); })
        .def(
            "symmetric_eigen_values",
            [](const RMat3rf &self) { return self.symmetricEigenValues(); }
        )
        .def(
            "symmetric_eigen_vector",
            [](const RMat3rf &self, RFloat eigen_value) {
                return self.symmetricEigenVector(eigen_value);
            },
            py::arg("eigen_value")
        )
        .def("is_valid", [](const RMat3rf &self) { return self.isValid(); });

    py::class_<RColor3b>(m, "RColor3b", "24-bit RGB color")
        .def(py::init<>())
        .def(py::init<RByte, RByte, RByte>(), py::arg("r"), py::arg("g"), py::arg("b"))
        .def(py::init([](const py::sequence &s) {
            if (py::len(s) != 3) {
                throw std::runtime_error("RColor3b expects a sequence of 3 values");
            }
            return RColor3b(s[0].cast<RByte>(), s[1].cast<RByte>(), s[2].cast<RByte>());
        }))
        .def_static("zero", &RColor3b::zero)
        .def_readwrite("r", &RColor3b::r)
        .def_readwrite("g", &RColor3b::g)
        .def_readwrite("b", &RColor3b::b)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self * py::self)
        .def(py::self * RByte())
        .def(RByte() * py::self)
        .def(py::self / RByte())
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= RByte())
        .def(py::self /= RByte())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__getitem__", [](const RColor3b &c, int i) { return c[i]; })
        .def("__setitem__", [](RColor3b &c, int i, RByte value) { c[i] = value; })
        .def("clip", [](const RColor3b &c) { return c.clip(); })
        .def("luminance", [](const RColor3b &c) { return c.luminance(); })
        .def("__repr__", [](const RColor3b &c) {
            return "RColor3b(" + std::to_string(c.r) + ", " + std::to_string(c.g) + ", " +
                   std::to_string(c.b) + ")";
        });
    py::implicitly_convertible<py::sequence, RColor3b>();

    py::class_<RColor3f>(m, "RColor3f", "float RGB color")
        .def(py::init<>())
        .def(py::init<float, float, float>(), py::arg("r"), py::arg("g"), py::arg("b"))
        .def(py::init([](const py::sequence &s) {
            if (py::len(s) != 3) {
                throw std::runtime_error("RColor3f expects a sequence of 3 values");
            }
            return RColor3f(s[0].cast<float>(), s[1].cast<float>(), s[2].cast<float>());
        }))
        .def_static("zero", &RColor3f::zero)
        .def_readwrite("r", &RColor3f::r)
        .def_readwrite("g", &RColor3f::g)
        .def_readwrite("b", &RColor3f::b)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self * py::self)
        .def(py::self * float())
        .def(float() * py::self)
        .def(py::self / float())
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= float())
        .def(py::self /= float())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__getitem__", [](const RColor3f &c, int i) { return c[i]; })
        .def("__setitem__", [](RColor3f &c, int i, float value) { c[i] = value; })
        .def("clip", [](const RColor3f &c) { return c.clip(); })
        .def("luminance", [](const RColor3f &c) { return c.luminance(); })
        .def("srgb_to_linear", [](const RColor3f &c) { return c.srgbToLinear(); })
        .def("linear_to_srgb", [](const RColor3f &c) { return c.linearToSrgb(); })
        .def("to_vec3d", [](const RColor3f &c) { return c.toVec3D(); })
        .def("__repr__", [](const RColor3f &c) {
            return "RColor3f(" + std::to_string(c.r) + ", " + std::to_string(c.g) + ", " +
                   std::to_string(c.b) + ")";
        });
    py::implicitly_convertible<py::sequence, RColor3f>();

    auto bound_rf = py::class_<RBound3rf>(m, "RBound3rf", "3D float bounding box");
    bound_rf.def(py::init<>())
        .def(
            py::init<RFloat, RFloat, RFloat, RFloat, RFloat, RFloat>(),
            py::arg("min_x"),
            py::arg("min_y"),
            py::arg("min_z"),
            py::arg("max_x"),
            py::arg("max_y"),
            py::arg("max_z")
        )
        .def(py::init([](const py::sequence &s) {
            if (py::len(s) != 6) {
                throw std::runtime_error("RBound3rf expects a sequence of 6 values");
            }
            return RBound3rf(
                s[0].cast<RFloat>(),
                s[1].cast<RFloat>(),
                s[2].cast<RFloat>(),
                s[3].cast<RFloat>(),
                s[4].cast<RFloat>(),
                s[5].cast<RFloat>()
            );
        }))
        .def_static("zero", &RBound3rf::zero)
        .def_static("lowest", &RBound3rf::lowest)
        .def_static("highest", &RBound3rf::highest)
        .def_readwrite("min_x", &RBound3rf::min_x)
        .def_readwrite("max_x", &RBound3rf::max_x)
        .def_readwrite("min_y", &RBound3rf::min_y)
        .def_readwrite("max_y", &RBound3rf::max_y)
        .def_readwrite("min_z", &RBound3rf::min_z)
        .def_readwrite("max_z", &RBound3rf::max_z)
        .def("is_valid", [](const RBound3rf &b) { return b.isValid(); })
        .def("volume", [](const RBound3rf &b) { return b.volume(); })
        .def(
            "is_containing",
            [](const RBound3rf &b, const RBound3rf &other) { return b.isContaining(other); },
            py::arg("bound")
        )
        .def(
            "is_containing",
            [](const RBound3rf &b, const RVec3rf &point) { return b.isContaining(point); },
            py::arg("point")
        )
        .def(
            "is_intersecting",
            [](const RBound3rf &b, const RBound3rf &other) { return b.isIntersecting(other); },
            py::arg("bound")
        )
        .def(
            "union_with",
            [](const RBound3rf &b, const RBound3rf &other) { return b.unionWith(other); },
            py::arg("bound")
        )
        .def(
            "intersection_with",
            [](const RBound3rf &b, const RBound3rf &other) { return b.intersectionWith(other); },
            py::arg("bound")
        )
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(
            "expand",
            [](RBound3rf &b, RFloat amount) { return b.expand(amount); },
            py::arg("amount")
        )
        .def(
            "expand",
            [](RBound3rf &b, const RVec3rf &amount) { return b.expand(amount); },
            py::arg("amount")
        )
        .def(
            "shift",
            [](RBound3rf &b, const RVec3rf &vector) { return b.shift(vector); },
            py::arg("vector")
        )
        .def(
            "extend",
            [](RBound3rf &b, const RVec3rf &point) { return b.extend(point); },
            py::arg("point")
        )
        .def("min_bound", [](const RBound3rf &b) { return b.minBound(); })
        .def("max_bound", [](const RBound3rf &b) { return b.maxBound(); })
        .def("center", [](const RBound3rf &b) { return b.center(); })
        .def("size", [](const RBound3rf &b) { return b.size(); })
        .def(
            "closest_point",
            [](const RBound3rf &b, const RVec3rf &point) { return b.closestPoint(point); },
            py::arg("point")
        );
    py::implicitly_convertible<py::sequence, RBound3rf>();

    auto bound_rui = py::class_<RBound3rui>(m, "RBound3rui", "3D unsigned-integer bounding box");
    bound_rui.def(py::init<>())
        .def(
            py::init<RIndex, RIndex, RIndex, RIndex, RIndex, RIndex>(),
            py::arg("min_x"),
            py::arg("min_y"),
            py::arg("min_z"),
            py::arg("max_x"),
            py::arg("max_y"),
            py::arg("max_z")
        )
        .def(py::init([](const py::sequence &s) {
            if (py::len(s) != 6) {
                throw std::runtime_error("RBound3rui expects a sequence of 6 values");
            }
            return RBound3rui(
                s[0].cast<RIndex>(),
                s[1].cast<RIndex>(),
                s[2].cast<RIndex>(),
                s[3].cast<RIndex>(),
                s[4].cast<RIndex>(),
                s[5].cast<RIndex>()
            );
        }))
        .def_static("zero", &RBound3rui::zero)
        .def_static("lowest", &RBound3rui::lowest)
        .def_static("highest", &RBound3rui::highest)
        .def_readwrite("min_x", &RBound3rui::min_x)
        .def_readwrite("max_x", &RBound3rui::max_x)
        .def_readwrite("min_y", &RBound3rui::min_y)
        .def_readwrite("max_y", &RBound3rui::max_y)
        .def_readwrite("min_z", &RBound3rui::min_z)
        .def_readwrite("max_z", &RBound3rui::max_z)
        .def("is_valid", [](const RBound3rui &b) { return b.isValid(); })
        .def("volume", [](const RBound3rui &b) { return b.volume(); })
        .def(
            "is_containing",
            [](const RBound3rui &b, const RBound3rui &other) { return b.isContaining(other); },
            py::arg("bound")
        )
        .def(
            "is_intersecting",
            [](const RBound3rui &b, const RBound3rui &other) { return b.isIntersecting(other); },
            py::arg("bound")
        )
        .def(
            "union_with",
            [](const RBound3rui &b, const RBound3rui &other) { return b.unionWith(other); },
            py::arg("bound")
        )
        .def(
            "intersection_with",
            [](const RBound3rui &b, const RBound3rui &other) { return b.intersectionWith(other); },
            py::arg("bound")
        )
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(
            "expand",
            [](RBound3rui &b, RIndex amount) { return b.expand(amount); },
            py::arg("amount")
        );
    py::implicitly_convertible<py::sequence, RBound3rui>();

    auto bound_ri = py::class_<RBound3ri>(m, "RBound3ri", "3D signed-integer bounding box");
    bound_ri.def(py::init<>())
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
        .def(py::init([](const py::sequence &s) {
            if (py::len(s) != 6) {
                throw std::runtime_error("RBound3ri expects a sequence of 6 values");
            }
            return RBound3ri(
                s[0].cast<RSignedIndex>(),
                s[1].cast<RSignedIndex>(),
                s[2].cast<RSignedIndex>(),
                s[3].cast<RSignedIndex>(),
                s[4].cast<RSignedIndex>(),
                s[5].cast<RSignedIndex>()
            );
        }))
        .def_static("zero", &RBound3ri::zero)
        .def_static("lowest", &RBound3ri::lowest)
        .def_static("highest", &RBound3ri::highest)
        .def_readwrite("min_x", &RBound3ri::min_x)
        .def_readwrite("max_x", &RBound3ri::max_x)
        .def_readwrite("min_y", &RBound3ri::min_y)
        .def_readwrite("max_y", &RBound3ri::max_y)
        .def_readwrite("min_z", &RBound3ri::min_z)
        .def_readwrite("max_z", &RBound3ri::max_z)
        .def("is_valid", [](const RBound3ri &b) { return b.isValid(); })
        .def("volume", [](const RBound3ri &b) { return b.volume(); })
        .def(
            "is_containing",
            [](const RBound3ri &b, const RBound3ri &other) { return b.isContaining(other); },
            py::arg("bound")
        )
        .def(
            "is_intersecting",
            [](const RBound3ri &b, const RBound3ri &other) { return b.isIntersecting(other); },
            py::arg("bound")
        )
        .def(
            "union_with",
            [](const RBound3ri &b, const RBound3ri &other) { return b.unionWith(other); },
            py::arg("bound")
        )
        .def(
            "intersection_with",
            [](const RBound3ri &b, const RBound3ri &other) { return b.intersectionWith(other); },
            py::arg("bound")
        )
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(
            "expand",
            [](RBound3ri &b, RSignedIndex amount) { return b.expand(amount); },
            py::arg("amount")
        );
    py::implicitly_convertible<py::sequence, RBound3ri>();

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
