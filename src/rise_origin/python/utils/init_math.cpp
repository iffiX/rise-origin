// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "utils/init_math.h"

#include <pybind11/operators.h>

#include <string>

namespace {

/// Bind the operators and member functions shared by every RS_VecND<RFloat> dimension.
template <typename V>
void rspyBindVecCommon(py::class_<V> &c, int dim) {
    c.def_static("zero", &V::zero, "Return a vector of all zeros.")
        .def_static("one", &V::one, "Return a vector of all ones.")
        .def_static(
            "full", &V::full, py::arg("value"), "Return a vector filled with the given value."
        )
        .def_static("highest", &V::highest, "Return a vector of the highest possible values.")
        .def_static("lowest", &V::lowest, "Return a vector of the lowest possible values.");

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

    c.def(
         "__len__", [dim](const V &) { return dim; }, "Number of components."
    )
        .def(
            "__getitem__",
            [dim](V &v, int i) {
                if (i < 0) i += dim;
                if (i < 0 || i >= dim) throw py::index_error();
                return v[i];
            },
            "Return the component at the given index."
        )
        .def(
            "__setitem__",
            [dim](V &v, int i, RFloat value) {
                if (i < 0) i += dim;
                if (i < 0 || i >= dim) throw py::index_error();
                v[i] = value;
            },
            "Set the component at the given index."
        );

    c.def(
         "dot",
         [](const V &a, const V &b) { return a.dot(b); },
         py::arg("other"),
         "Dot product with another vector."
    )
        .def(
            "length", [](const V &v) { return v.length(); }, "Euclidean length (magnitude)."
        )
        .def(
            "length2",
            [](const V &v) { return v.length2(); },
            "Squared length, avoiding the square root."
        )
        .def(
            "normalized",
            [](const V &v) { return v.normalized(); },
            "Return this vector scaled to unit length; a zero vector stays zero."
        )
        .def(
            "abs", [](const V &v) { return v.abs(); }, "Component-wise absolute value."
        )
        .def(
            "floor", [](const V &v) { return v.floor(); }, "Component-wise floor."
        )
        .def(
            "ceil", [](const V &v) { return v.ceil(); }, "Component-wise ceiling."
        )
        .def(
            "round", [](const V &v) { return v.round(); }, "Component-wise rounding."
        )
        .def(
            "clip",
            [](V &v, RFloat low, RFloat high) { return v.clip(low, high); },
            py::arg("low"),
            py::arg("high"),
            "Return the vector with each component clamped to [low, high]."
        )
        .def(
            "prod", [](const V &v) { return v.prod(); }, "Product of the components."
        )
        .def(
            "mean", [](const V &v) { return v.mean(); }, "Mean of the components."
        )
        .def(
            "sum", [](const V &v) { return v.sum(); }, "Sum of the components."
        )
        .def(
            "min", [](const V &v) { return v.min(); }, "Smallest component."
        )
        .def(
            "min",
            [](const V &a, const V &b) { return a.min(b); },
            py::arg("other"),
            "Component-wise minimum with another vector."
        )
        .def(
            "max", [](const V &v) { return v.max(); }, "Largest component."
        )
        .def(
            "max",
            [](const V &a, const V &b) { return a.max(b); },
            py::arg("other"),
            "Component-wise maximum with another vector."
        )
        .def(
            "argmin", [](const V &v) { return v.argmin(); }, "Index of the smallest component."
        )
        .def(
            "argmax", [](const V &v) { return v.argmax(); }, "Index of the largest component."
        )
        .def(
            "elem_mul",
            [](const V &a, const V &b) { return a.elemMul(b); },
            py::arg("other"),
            "Component-wise multiplication with another vector."
        )
        .def(
            "elem_div",
            [](const V &a, const V &b) { return a.elemDiv(b); },
            py::arg("other"),
            "Component-wise division by another vector."
        )
        .def(
            "elem_div_fast",
            [](const V &a, const V &b) { return a.elemDivFast(b); },
            py::arg("other"),
            "Fast component-wise division by another vector."
        )
        .def(
            "reciprocal", [](const V &v) { return v.reciprocal(); }, "Component-wise reciprocal."
        )
        .def(
            "reciprocal_fast",
            [](const V &v) { return v.reciprocalFast(); },
            "Fast component-wise reciprocal."
        )
        .def(
            "dist",
            [](const V &a, const V &b) { return a.dist(b); },
            py::arg("other"),
            "Euclidean distance to another vector."
        )
        .def(
            "dist2",
            [](const V &a, const V &b) { return a.dist2(b); },
            py::arg("other"),
            "Squared euclidean distance to another vector."
        )
        .def(
            "project",
            [](const V &a, const V &b) { return a.project(b); },
            py::arg("other"),
            "Project this vector onto another vector."
        )
        .def(
            "is_valid",
            [](const V &v) { return v.isValid(); },
            "Whether all components are valid numbers."
        )
        .def(
            "is_in_bound",
            [](const V &v, RFloat low, RFloat high) { return v.isInBound(low, high); },
            py::arg("low"),
            py::arg("high"),
            "Whether every component lies within [low, high]."
        );
}

}  // namespace

void initRSPYMath(py::module &m) {
    py::class_<RVec2rf> vec2(m, "RVec2rf", "2D float vector");
    vec2.def(py::init<>(), "Create a zero vector.")
        .def(
            py::init<RFloat, RFloat>(),
            py::arg("x"),
            py::arg("y"),
            "Construct from individual components."
        )
        .def(
            py::init([](const py::sequence &s) {
                if (py::len(s) != 2) {
                    throw std::runtime_error("RVec2rf expects a sequence of 2 values");
                }
                return RVec2rf(s[0].cast<RFloat>(), s[1].cast<RFloat>());
            }),
            "Construct from a length-2 sequence."
        )
        .def_readwrite("x", &RVec2rf::x, "X component.")
        .def_readwrite("y", &RVec2rf::y, "Y component.")
        .def("__repr__", [](const RVec2rf &v) {
            return "RVec2rf(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
        });
    rspyBindVecCommon(vec2, 2);
    py::implicitly_convertible<py::sequence, RVec2rf>();

    py::class_<RVec3rf> vec3(m, "RVec3rf", "3D float vector");
    vec3.def(py::init<>(), "Create a zero vector.")
        .def(
            py::init<RFloat, RFloat, RFloat>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            "Construct from individual components."
        )
        .def(
            py::init([](const py::sequence &s) {
                if (py::len(s) != 3) {
                    throw std::runtime_error("RVec3rf expects a sequence of 3 values");
                }
                return RVec3rf(s[0].cast<RFloat>(), s[1].cast<RFloat>(), s[2].cast<RFloat>());
            }),
            "Construct from a length-3 sequence."
        )
        .def_readwrite("x", &RVec3rf::x, "X component.")
        .def_readwrite("y", &RVec3rf::y, "Y component.")
        .def_readwrite("z", &RVec3rf::z, "Z component.")
        .def(
            "cross",
            [](const RVec3rf &a, const RVec3rf &b) { return a.cross(b); },
            py::arg("other"),
            "Cross product with another vector."
        )
        .def(
            "rotate",
            [](RVec3rf &v, const RVec3rf &axis, RFloat angle) { return v.rotate(axis, angle); },
            py::arg("axis"),
            py::arg("angle"),
            "Rotate this vector about an axis by an angle in radians."
        )
        .def(
            "rotate_x",
            [](RVec3rf &v, RFloat angle) { return v.rotateX(angle); },
            py::arg("angle"),
            "Rotate this vector about the X axis by an angle in radians."
        )
        .def(
            "rotate_y",
            [](RVec3rf &v, RFloat angle) { return v.rotateY(angle); },
            py::arg("angle"),
            "Rotate this vector about the Y axis by an angle in radians."
        )
        .def(
            "rotate_z",
            [](RVec3rf &v, RFloat angle) { return v.rotateZ(angle); },
            py::arg("angle"),
            "Rotate this vector about the Z axis by an angle in radians."
        )
        .def(
            "arbitrary_normal",
            [](const RVec3rf &v) { return v.arbitraryNormal(); },
            "Return an arbitrary vector perpendicular to this one."
        )
        .def(
            "to_cross_matrix",
            [](const RVec3rf &v) { return v.toCrossMatrix(); },
            "Return the skew-symmetric cross-product matrix of this vector."
        )
        .def("__repr__", [](const RVec3rf &v) {
            return "RVec3rf(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " +
                   std::to_string(v.z) + ")";
        });
    rspyBindVecCommon(vec3, 3);
    py::implicitly_convertible<py::sequence, RVec3rf>();

    py::class_<RVec4rf> vec4(m, "RVec4rf", "4D float vector");
    vec4.def(py::init<>(), "Create a zero vector.")
        .def(
            py::init<RFloat, RFloat, RFloat, RFloat>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("w"),
            "Construct from individual components."
        )
        .def(
            py::init([](const py::sequence &s) {
                if (py::len(s) != 4) {
                    throw std::runtime_error("RVec4rf expects a sequence of 4 values");
                }
                return RVec4rf(
                    s[0].cast<RFloat>(),
                    s[1].cast<RFloat>(),
                    s[2].cast<RFloat>(),
                    s[3].cast<RFloat>()
                );
            }),
            "Construct from a length-4 sequence."
        )
        .def_readwrite("x", &RVec4rf::x, "X component.")
        .def_readwrite("y", &RVec4rf::y, "Y component.")
        .def_readwrite("z", &RVec4rf::z, "Z component.")
        .def_readwrite("w", &RVec4rf::w, "W component.")
        .def("__repr__", [](const RVec4rf &v) {
            return "RVec4rf(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " +
                   std::to_string(v.z) + ", " + std::to_string(v.w) + ")";
        });
    rspyBindVecCommon(vec4, 4);
    py::implicitly_convertible<py::sequence, RVec4rf>();

    py::class_<RQuat3rf>(m, "RQuat3rf", "3D float quaternion")
        .def(py::init<>(), "Create a zero quaternion.")
        .def(
            py::init<RFloat, RFloat, RFloat, RFloat>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("w"),
            "Construct from individual components."
        )
        .def(
            py::init<RFloat, const RVec3rf &>(),
            py::arg("angle"),
            py::arg("axis"),
            "Construct a rotation of angle radians about the given axis."
        )
        .def(
            py::init<const RVec3rf &, const RVec3rf &>(),
            py::arg("from_direction"),
            py::arg("to_direction"),
            "Construct the rotation that maps one direction onto another."
        )
        .def(
            py::init([](const py::sequence &s) {
                if (py::len(s) != 4) {
                    throw std::runtime_error("RQuat3rf expects a sequence of 4 values");
                }
                return RQuat3rf(
                    s[0].cast<RFloat>(),
                    s[1].cast<RFloat>(),
                    s[2].cast<RFloat>(),
                    s[3].cast<RFloat>()
                );
            }),
            "Construct from a length-4 sequence."
        )
        .def_static("unit", &RQuat3rf::unit, "Return the identity quaternion.")
        .def_readwrite("x", &RQuat3rf::x, "X component.")
        .def_readwrite("y", &RQuat3rf::y, "Y component.")
        .def_readwrite("z", &RQuat3rf::z, "Z component.")
        .def_readwrite("w", &RQuat3rf::w, "W (scalar) component.")
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
        .def(
            "to_vec",
            [](const RQuat3rf &q) { return q.toVec(); },
            "Drop w and return the (x, y, z) vector."
        )
        .def(
            "length", [](const RQuat3rf &q) { return q.length(); }, "Length (magnitude)."
        )
        .def(
            "length2", [](const RQuat3rf &q) { return q.length2(); }, "Squared length (magnitude)."
        )
        .def(
            "normalized",
            [](const RQuat3rf &q) { return q.normalized(); },
            "Return the normalized quaternion."
        )
        .def(
            "inverse",
            [](const RQuat3rf &q) { return q.inverse(); },
            "Return the inverse of this quaternion."
        )
        .def(
            "conjugate",
            [](const RQuat3rf &q) { return q.conjugate(); },
            "Return the conjugate of this quaternion."
        )
        .def(
            "slerp",
            [](const RQuat3rf &q, const RQuat3rf &to, RFloat t) { return q.slerp(to, t); },
            py::arg("to"),
            py::arg("t"),
            "Spherical linear interpolation toward another quaternion by t."
        )
        .def(
            "angle", [](const RQuat3rf &q) { return q.angle(); }, "Angular rotation in radians."
        )
        .def(
            "angle_degrees",
            [](const RQuat3rf &q) { return q.angleDegrees(); },
            "Angular rotation in degrees."
        )
        .def(
            "is_negligible_angle",
            [](const RQuat3rf &q) { return q.isNegligibleAngle(); },
            "Whether the rotation angle is likely negligible."
        )
        .def(
            "is_small_angle",
            [](const RQuat3rf &q) { return q.isSmallAngle(); },
            "Whether the rotation angle is small."
        )
        .def(
            "axis", [](const RQuat3rf &q) { return q.axis(); }, "The normalized rotation axis."
        )
        .def(
            "axis_un_normalized",
            [](const RQuat3rf &q) { return q.axisUnNormalized(); },
            "The un-normalized rotation axis."
        )
        .def(
            "to_rotation_matrix",
            [](const RQuat3rf &q) { return q.toRotationMatrix(); },
            "Convert to a 3x3 rotation matrix."
        )
        .def(
            "to_rotation_vector",
            [](const RQuat3rf &q) { return q.toRotationVector(); },
            "Convert to an axis-angle rotation vector."
        )
        .def(
            "rotate_vec3d",
            [](const RQuat3rf &q, const RVec3rf &v) { return q.rotateVec3D(v); },
            py::arg("vector"),
            "Rotate a vector by this quaternion."
        )
        .def(
            "rotate_vec3d_inv",
            [](const RQuat3rf &q, const RVec3rf &v) { return q.rotateVec3DInv(v); },
            py::arg("vector"),
            "Rotate a vector by the inverse of this quaternion."
        )
        .def(
            "to_ortho_axis_x",
            [](const RQuat3rf &q) { return q.toOrthoAxisX(); },
            "The local X axis after this rotation."
        )
        .def(
            "to_ortho_axis_y",
            [](const RQuat3rf &q) { return q.toOrthoAxisY(); },
            "The local Y axis after this rotation."
        )
        .def(
            "to_ortho_axis_z",
            [](const RQuat3rf &q) { return q.toOrthoAxisZ(); },
            "The local Z axis after this rotation."
        )
        .def(
            "is_valid",
            [](const RQuat3rf &q) { return q.isValid(); },
            "Whether all values are valid numbers."
        )
        .def(
            "from_rotation_matrix",
            [](RQuat3rf &q, const RMat3rf &mat) { return q.fromRotationMatrix(mat); },
            py::arg("matrix"),
            "Set this quaternion from a rotation matrix."
        )
        .def(
            "from_rotation_vector",
            [](RQuat3rf &q, const RVec3rf &vec) { return q.fromRotationVector(vec); },
            py::arg("vector"),
            "Set this quaternion from an axis-angle rotation vector."
        )
        .def(
            "from_angle_to_pos_x",
            [](RQuat3rf &q, const RVec3rf &vec) { return q.fromAngleToPosX(vec); },
            py::arg("rotate_from"),
            "Set this quaternion to the rotation from the given direction to +X."
        )
        .def("__repr__", [](const RQuat3rf &q) {
            return "RQuat3rf(" + std::to_string(q.x) + ", " + std::to_string(q.y) + ", " +
                   std::to_string(q.z) + ", " + std::to_string(q.w) + ")";
        });
    py::implicitly_convertible<py::sequence, RQuat3rf>();

    py::class_<RMat3rf>(m, "RMat3rf", "3x3 float matrix (row-major)")
        .def(py::init<>(), "Create a zero matrix.")
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
            py::arg("m22"),
            "Construct from individual elements in row-major order."
        )
        .def(
            py::init([](const RVec3rf &r0, const RVec3rf &r1, const RVec3rf &r2) {
                return RMat3rf(r0, r1, r2);
            }),
            py::arg("row0"),
            py::arg("row1"),
            py::arg("row2"),
            "Construct from three row vectors."
        )
        .def(
            py::init([](const py::sequence &s) {
                RFloat e[9];
                if (py::len(s) == 9) {
                    for (int i = 0; i < 9; ++i) e[i] = s[i].cast<RFloat>();
                } else if (py::len(s) == 3) {
                    for (int r = 0; r < 3; ++r) {
                        py::sequence row = py::reinterpret_borrow<py::sequence>(s[r]);
                        if (py::len(row) != 3) {
                            throw std::runtime_error("RMat3rf rows must each have 3 values");
                        }
                        for (int c = 0; c < 3; ++c) e[r * 3 + c] = row[c].cast<RFloat>();
                    }
                } else {
                    throw std::runtime_error("RMat3rf expects 9 values or 3 rows of 3 values");
                }
                return RMat3rf(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]);
            }),
            "Construct from a flat 9-element sequence or three rows of 3 values."
        )
        .def_static("zero", &RMat3rf::zero, "Create a zero matrix.")
        .def_static("eye", &RMat3rf::eye, "Create an identity matrix.")
        .def_static(
            "full", &RMat3rf::full, py::arg("value"), "Create a matrix filled with the given value."
        )
        .def_static(
            "diagonal",
            &RMat3rf::diagonal,
            py::arg("m00"),
            py::arg("m11"),
            py::arg("m22"),
            "Create a diagonal matrix from three elements."
        )
        .def(
            "__getitem__",
            [](const RMat3rf &self, std::pair<int, int> idx) {
                if (idx.first < 0 || idx.first >= 3 || idx.second < 0 || idx.second >= 3) {
                    throw py::index_error("RMat3rf index out of range");
                }
                return self[idx.first][idx.second];
            },
            "Return the element at the (row, col) index pair."
        )
        .def(
            "__getitem__",
            [](const RMat3rf &self, int row) {
                if (row < 0) row += 3;
                if (row < 0 || row >= 3) {
                    throw py::index_error("RMat3rf row index out of range");
                }
                py::list out;
                for (int c = 0; c < 3; ++c) out.append(self(row, c));
                return out;
            },
            "Return the given row as a list of three floats."
        )
        .def(
            "__setitem__",
            [](RMat3rf &self, std::pair<int, int> idx, RFloat v) {
                if (idx.first < 0 || idx.first >= 3 || idx.second < 0 || idx.second >= 3) {
                    throw py::index_error("RMat3rf index out of range");
                }
                self[idx.first][idx.second] = v;
            },
            "Set the element at the (row, col) index pair."
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
            py::arg("col"),
            "Return the element at (row, col)."
        )
        .def(
            "to_list",
            [](const RMat3rf &self) {
                py::list rows;
                for (int r = 0; r < 3; ++r) {
                    py::list row;
                    for (int c = 0; c < 3; ++c) row.append(self(r, c));
                    rows.append(row);
                }
                return rows;
            },
            "Return the matrix as a 3x3 nested list of floats."
        )
        .def(
            "from_list",
            [](RMat3rf &self, const py::sequence &rows) {
                if (py::len(rows) != 3) {
                    throw py::value_error("RMat3rf.from_list expects 3 rows");
                }
                for (int r = 0; r < 3; ++r) {
                    py::sequence row = py::reinterpret_borrow<py::sequence>(rows[r]);
                    if (py::len(row) != 3) {
                        throw py::value_error("RMat3rf.from_list expects 3 columns per row");
                    }
                    for (int c = 0; c < 3; ++c) self[r][c] = row[c].cast<RFloat>();
                }
            },
            "Set the matrix from three rows of three values (plain lists, tuples, or RVec3rf)."
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
        .def(
            "transpose",
            [](const RMat3rf &self) { return self.transpose(); },
            "Return the transpose."
        )
        .def(
            "determinant",
            [](const RMat3rf &self) { return self.determinant(); },
            "Return the determinant."
        )
        .def(
            "adjoint",
            [](const RMat3rf &self) { return self.adjoint(); },
            "Return the adjugate (adjoint) matrix."
        )
        .def(
            "inverse",
            [](const RMat3rf &self) { return self.inverse(); },
            "Return the inverse matrix."
        )
        .def(
            "symmetric_eigen_values",
            [](const RMat3rf &self) { return self.symmetricEigenValues(); },
            "Eigenvalues of this symmetric matrix."
        )
        .def(
            "symmetric_eigen_vector",
            [](const RMat3rf &self, RFloat eigen_value) {
                return self.symmetricEigenVector(eigen_value);
            },
            py::arg("eigen_value"),
            "Eigenvector for the given eigenvalue of this symmetric matrix."
        )
        .def(
            "is_valid",
            [](const RMat3rf &self) { return self.isValid(); },
            "Whether all values are valid numbers."
        )
        .def("__repr__", [](const RMat3rf &m) {
            return "RMat3rf([[" + std::to_string(m(0, 0)) + ", " + std::to_string(m(0, 1)) + ", " +
                   std::to_string(m(0, 2)) + "], [" + std::to_string(m(1, 0)) + ", " +
                   std::to_string(m(1, 1)) + ", " + std::to_string(m(1, 2)) + "], [" +
                   std::to_string(m(2, 0)) + ", " + std::to_string(m(2, 1)) + ", " +
                   std::to_string(m(2, 2)) + "]])";
        });
    py::implicitly_convertible<py::sequence, RMat3rf>();

    py::class_<RColor3b>(m, "RColor3b", "24-bit RGB color")
        .def(py::init<>(), "Create a zero color.")
        .def(
            py::init<RByte, RByte, RByte>(),
            py::arg("r"),
            py::arg("g"),
            py::arg("b"),
            "Construct from individual channels."
        )
        .def(
            py::init([](const py::sequence &s) {
                if (py::len(s) != 3) {
                    throw std::runtime_error("RColor3b expects a sequence of 3 values");
                }
                return RColor3b(s[0].cast<RByte>(), s[1].cast<RByte>(), s[2].cast<RByte>());
            }),
            "Construct from a length-3 sequence."
        )
        .def_static("zero", &RColor3b::zero, "Return a color of all zeros.")
        .def_readwrite("r", &RColor3b::r, "Red channel.")
        .def_readwrite("g", &RColor3b::g, "Green channel.")
        .def_readwrite("b", &RColor3b::b, "Blue channel.")
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
        .def(
            "__getitem__",
            [](const RColor3b &c, int i) { return c[i]; },
            "Return the channel at the given index."
        )
        .def(
            "__setitem__",
            [](RColor3b &c, int i, RByte value) { c[i] = value; },
            "Set the channel at the given index."
        )
        .def(
            "clip",
            [](const RColor3b &c) { return c.clip(); },
            "Return the color with channels clamped to the valid range."
        )
        .def(
            "luminance",
            [](const RColor3b &c) { return c.luminance(); },
            "CIE 1931 Y luminance of the RGB channels."
        )
        .def("__repr__", [](const RColor3b &c) {
            return "RColor3b(" + std::to_string(c.r) + ", " + std::to_string(c.g) + ", " +
                   std::to_string(c.b) + ")";
        });
    py::implicitly_convertible<py::sequence, RColor3b>();

    py::class_<RColor3f>(m, "RColor3f", "float RGB color")
        .def(py::init<>(), "Create a zero color.")
        .def(
            py::init<float, float, float>(),
            py::arg("r"),
            py::arg("g"),
            py::arg("b"),
            "Construct from individual channels."
        )
        .def(
            py::init([](const py::sequence &s) {
                if (py::len(s) != 3) {
                    throw std::runtime_error("RColor3f expects a sequence of 3 values");
                }
                return RColor3f(s[0].cast<float>(), s[1].cast<float>(), s[2].cast<float>());
            }),
            "Construct from a length-3 sequence."
        )
        .def_static("zero", &RColor3f::zero, "Return a color of all zeros.")
        .def_readwrite("r", &RColor3f::r, "Red channel.")
        .def_readwrite("g", &RColor3f::g, "Green channel.")
        .def_readwrite("b", &RColor3f::b, "Blue channel.")
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
        .def(
            "__getitem__",
            [](const RColor3f &c, int i) { return c[i]; },
            "Return the channel at the given index."
        )
        .def(
            "__setitem__",
            [](RColor3f &c, int i, float value) { c[i] = value; },
            "Set the channel at the given index."
        )
        .def(
            "clip",
            [](const RColor3f &c) { return c.clip(); },
            "Return the color with channels clamped to the valid range."
        )
        .def(
            "luminance",
            [](const RColor3f &c) { return c.luminance(); },
            "CIE 1931 Y luminance of the RGB channels."
        )
        .def(
            "srgb_to_linear",
            [](const RColor3f &c) { return c.srgbToLinear(); },
            "Convert from sRGB to linear color."
        )
        .def(
            "linear_to_srgb",
            [](const RColor3f &c) { return c.linearToSrgb(); },
            "Convert from linear to sRGB color."
        )
        .def(
            "to_vec3d",
            [](const RColor3f &c) { return c.toVec3D(); },
            "Return the channels as a 3D vector."
        )
        .def("__repr__", [](const RColor3f &c) {
            return "RColor3f(" + std::to_string(c.r) + ", " + std::to_string(c.g) + ", " +
                   std::to_string(c.b) + ")";
        });
    py::implicitly_convertible<py::sequence, RColor3f>();

    auto bound_rf = py::class_<RBound3rf>(m, "RBound3rf", "3D float bounding box");
    bound_rf.def(py::init<>(), "Default constructor.")
        .def(
            py::init<RFloat, RFloat, RFloat, RFloat, RFloat, RFloat>(),
            py::arg("min_x"),
            py::arg("min_y"),
            py::arg("min_z"),
            py::arg("max_x"),
            py::arg("max_y"),
            py::arg("max_z"),
            "Construct from individual min and max values."
        )
        .def(
            py::init([](const py::sequence &s) {
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
            }),
            "Construct from a length-6 sequence."
        )
        .def_static("zero", &RBound3rf::zero, "Return a bound with all dimensions set to zero.")
        .def_static("lowest", &RBound3rf::lowest, "Return a bound of the lowest possible values.")
        .def_static(
            "highest", &RBound3rf::highest, "Return a bound of the highest possible values."
        )
        .def_readwrite("min_x", &RBound3rf::min_x, "Minimum X.")
        .def_readwrite("max_x", &RBound3rf::max_x, "Maximum X.")
        .def_readwrite("min_y", &RBound3rf::min_y, "Minimum Y.")
        .def_readwrite("max_y", &RBound3rf::max_y, "Maximum Y.")
        .def_readwrite("min_z", &RBound3rf::min_z, "Minimum Z.")
        .def_readwrite("max_z", &RBound3rf::max_z, "Maximum Z.")
        .def(
            "is_valid",
            [](const RBound3rf &b) { return b.isValid(); },
            "Whether the bound is valid (min <= max on every axis)."
        )
        .def(
            "volume", [](const RBound3rf &b) { return b.volume(); }, "Volume of the bound."
        )
        .def(
            "is_containing",
            [](const RBound3rf &b, const RBound3rf &other) { return b.isContaining(other); },
            py::arg("bound"),
            "Whether this bound contains another bound."
        )
        .def(
            "is_containing",
            [](const RBound3rf &b, const RVec3rf &point) { return b.isContaining(point); },
            py::arg("point"),
            "Whether this bound contains a point."
        )
        .def(
            "is_intersecting",
            [](const RBound3rf &b, const RBound3rf &other) { return b.isIntersecting(other); },
            py::arg("bound"),
            "Whether this bound intersects another bound."
        )
        .def(
            "union_with",
            [](const RBound3rf &b, const RBound3rf &other) { return b.unionWith(other); },
            py::arg("bound"),
            "Return the union of this bound and another bound."
        )
        .def(
            "intersection_with",
            [](const RBound3rf &b, const RBound3rf &other) { return b.intersectionWith(other); },
            py::arg("bound"),
            "Return the intersection of this bound and another bound."
        )
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(
            "expand",
            [](RBound3rf &b, RFloat amount) { return b.expand(amount); },
            py::arg("amount"),
            "Expand the bound by a fixed amount in all directions."
        )
        .def(
            "expand",
            [](RBound3rf &b, const RVec3rf &amount) { return b.expand(amount); },
            py::arg("amount"),
            "Expand the bound by per-axis amounts."
        )
        .def(
            "shift",
            [](RBound3rf &b, const RVec3rf &vector) { return b.shift(vector); },
            py::arg("vector"),
            "Shift the bound by a vector."
        )
        .def(
            "extend",
            [](RBound3rf &b, const RVec3rf &point) { return b.extend(point); },
            py::arg("point"),
            "Extend the bound to include a point."
        )
        .def(
            "min_bound",
            [](const RBound3rf &b) { return b.minBound(); },
            "Return the minimum corner as a vector."
        )
        .def(
            "max_bound",
            [](const RBound3rf &b) { return b.maxBound(); },
            "Return the maximum corner as a vector."
        )
        .def(
            "center",
            [](const RBound3rf &b) { return b.center(); },
            "Return the center of the bound."
        )
        .def(
            "size",
            [](const RBound3rf &b) { return b.size(); },
            "Return the size of the bound (max - min)."
        )
        .def(
            "closest_point",
            [](const RBound3rf &b, const RVec3rf &point) { return b.closestPoint(point); },
            py::arg("point"),
            "Return the closest point on the bound to a point."
        )
        .def("__repr__", [](const RBound3rf &b) {
            return "RBound3rf(" + std::to_string(b.min_x) + ", " + std::to_string(b.min_y) + ", " +
                   std::to_string(b.min_z) + ", " + std::to_string(b.max_x) + ", " +
                   std::to_string(b.max_y) + ", " + std::to_string(b.max_z) + ")";
        });
    py::implicitly_convertible<py::sequence, RBound3rf>();

    auto bound_rui = py::class_<RBound3rui>(m, "RBound3rui", "3D unsigned-integer bounding box");
    bound_rui.def(py::init<>(), "Default constructor.")
        .def(
            py::init<RIndex, RIndex, RIndex, RIndex, RIndex, RIndex>(),
            py::arg("min_x"),
            py::arg("min_y"),
            py::arg("min_z"),
            py::arg("max_x"),
            py::arg("max_y"),
            py::arg("max_z"),
            "Construct from individual min and max values."
        )
        .def(
            py::init([](const py::sequence &s) {
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
            }),
            "Construct from a length-6 sequence."
        )
        .def_static("zero", &RBound3rui::zero, "Return a bound with all dimensions set to zero.")
        .def_static("lowest", &RBound3rui::lowest, "Return a bound of the lowest possible values.")
        .def_static(
            "highest", &RBound3rui::highest, "Return a bound of the highest possible values."
        )
        .def_readwrite("min_x", &RBound3rui::min_x, "Minimum X.")
        .def_readwrite("max_x", &RBound3rui::max_x, "Maximum X.")
        .def_readwrite("min_y", &RBound3rui::min_y, "Minimum Y.")
        .def_readwrite("max_y", &RBound3rui::max_y, "Maximum Y.")
        .def_readwrite("min_z", &RBound3rui::min_z, "Minimum Z.")
        .def_readwrite("max_z", &RBound3rui::max_z, "Maximum Z.")
        .def(
            "is_valid",
            [](const RBound3rui &b) { return b.isValid(); },
            "Whether the bound is valid (min <= max on every axis)."
        )
        .def(
            "volume", [](const RBound3rui &b) { return b.volume(); }, "Volume of the bound."
        )
        .def(
            "is_containing",
            [](const RBound3rui &b, const RBound3rui &other) { return b.isContaining(other); },
            py::arg("bound"),
            "Whether this bound contains another bound."
        )
        .def(
            "is_intersecting",
            [](const RBound3rui &b, const RBound3rui &other) { return b.isIntersecting(other); },
            py::arg("bound"),
            "Whether this bound intersects another bound."
        )
        .def(
            "union_with",
            [](const RBound3rui &b, const RBound3rui &other) { return b.unionWith(other); },
            py::arg("bound"),
            "Return the union of this bound and another bound."
        )
        .def(
            "intersection_with",
            [](const RBound3rui &b, const RBound3rui &other) { return b.intersectionWith(other); },
            py::arg("bound"),
            "Return the intersection of this bound and another bound."
        )
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(
            "expand",
            [](RBound3rui &b, RIndex amount) { return b.expand(amount); },
            py::arg("amount"),
            "Expand the bound by a fixed amount in all directions."
        )
        .def("__repr__", [](const RBound3rui &b) {
            return "RBound3rui(" + std::to_string(b.min_x) + ", " + std::to_string(b.min_y) + ", " +
                   std::to_string(b.min_z) + ", " + std::to_string(b.max_x) + ", " +
                   std::to_string(b.max_y) + ", " + std::to_string(b.max_z) + ")";
        });
    py::implicitly_convertible<py::sequence, RBound3rui>();

    auto bound_ri = py::class_<RBound3ri>(m, "RBound3ri", "3D signed-integer bounding box");
    bound_ri.def(py::init<>(), "Default constructor.")
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
            py::arg("max_z"),
            "Construct from individual min and max values."
        )
        .def(
            py::init([](const py::sequence &s) {
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
            }),
            "Construct from a length-6 sequence."
        )
        .def_static("zero", &RBound3ri::zero, "Return a bound with all dimensions set to zero.")
        .def_static("lowest", &RBound3ri::lowest, "Return a bound of the lowest possible values.")
        .def_static(
            "highest", &RBound3ri::highest, "Return a bound of the highest possible values."
        )
        .def_readwrite("min_x", &RBound3ri::min_x, "Minimum X.")
        .def_readwrite("max_x", &RBound3ri::max_x, "Maximum X.")
        .def_readwrite("min_y", &RBound3ri::min_y, "Minimum Y.")
        .def_readwrite("max_y", &RBound3ri::max_y, "Maximum Y.")
        .def_readwrite("min_z", &RBound3ri::min_z, "Minimum Z.")
        .def_readwrite("max_z", &RBound3ri::max_z, "Maximum Z.")
        .def(
            "is_valid",
            [](const RBound3ri &b) { return b.isValid(); },
            "Whether the bound is valid (min <= max on every axis)."
        )
        .def(
            "volume", [](const RBound3ri &b) { return b.volume(); }, "Volume of the bound."
        )
        .def(
            "is_containing",
            [](const RBound3ri &b, const RBound3ri &other) { return b.isContaining(other); },
            py::arg("bound"),
            "Whether this bound contains another bound."
        )
        .def(
            "is_intersecting",
            [](const RBound3ri &b, const RBound3ri &other) { return b.isIntersecting(other); },
            py::arg("bound"),
            "Whether this bound intersects another bound."
        )
        .def(
            "union_with",
            [](const RBound3ri &b, const RBound3ri &other) { return b.unionWith(other); },
            py::arg("bound"),
            "Return the union of this bound and another bound."
        )
        .def(
            "intersection_with",
            [](const RBound3ri &b, const RBound3ri &other) { return b.intersectionWith(other); },
            py::arg("bound"),
            "Return the intersection of this bound and another bound."
        )
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(
            "expand",
            [](RBound3ri &b, RSignedIndex amount) { return b.expand(amount); },
            py::arg("amount"),
            "Expand the bound by a fixed amount in all directions."
        )
        .def("__repr__", [](const RBound3ri &b) {
            return "RBound3ri(" + std::to_string(b.min_x) + ", " + std::to_string(b.min_y) + ", " +
                   std::to_string(b.min_z) + ", " + std::to_string(b.max_x) + ", " +
                   std::to_string(b.max_y) + ", " + std::to_string(b.max_z) + ")";
        });
    py::implicitly_convertible<py::sequence, RBound3ri>();

    py::enum_<RSE_MathExprOperator>(
        m, "RSE_MathExprOperator", "Operator or token kind in a math expression."
    )
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

    py::class_<RS_MathExprToken>(m, "RS_MathExprToken", "A single token in a math expression.")
        .def(py::init<>(), "Create an empty token.")
        .def(py::init<RSE_MathExprOperator>(), "Create a token for the given operator.")
        .def(
            py::init([](RSE_MathExprOperator op, py::object val) {
                RS_MathExprToken token(op);
                if (op == RSE_ME_CONST) {
                    token.value = val.cast<RFloat>();
                } else if (op == RSE_ME_VAR) {
                    token.value = val.cast<std::vector<std::string>>();
                }
                return token;
            }),
            "Create a token with a value (a constant for CONST, a name path for VAR)."
        )
        .def_readwrite("op", &RS_MathExprToken::op, "The operator or token kind.")
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
            },
            "The token payload: a float for CONST, a name path for VAR, else None."
        );

    py::class_<RS_MathExpr>(m, "RS_MathExpr", "A math expression as a sequence of tokens.")
        .def(py::init<>(), "Create an empty expression.")
        .def("empty", &RS_MathExpr::empty, "Whether the expression has no tokens.")
        .def(
            "__getitem__",
            [](const RS_MathExpr &expr, int idx) { return expr[idx]; },
            "Return the token at the given index."
        )
        .def(
            "__setitem__",
            [](RS_MathExpr &expr, int idx, const RS_MathExprToken &token) { expr[idx] = token; },
            "Set the token at the given index."
        )
        .def_readwrite("expr", &RS_MathExpr::expr, "The underlying token sequence.")
        .def(
            "__str__",
            [](const RS_MathExpr &expr) {
                std::stringstream ss;
                ss << "MathExpr(" << expr.expr.size() << " tokens)";
                return ss.str();
            }
        )
        .def(
            "from_string",
            [](RS_MathExpr &expr, const std::string &expr_string) {
                expr = expr_string;
                return expr;
            },
            "Parse an expression string into this object."
        );
}
