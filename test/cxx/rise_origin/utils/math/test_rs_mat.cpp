#include <gtest/gtest.h>

#include "rise_origin/utils/math/rs_mat.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_Mat3D is a plain old data 3x3 matrix templated on the
 * element type for either float or double resolution. It is row
 * major and carries matrix arithmetic plus the signature
 * operations transpose, determinant, adjoint, and inverse.
 * RS_Mat3DMem is a packed memory format that converts to and from
 * RS_Mat3D.
 *
 * - Construction and Factories
 *   parameter constructor, eye, zero, diagonal, full, row vectors
 *
 * - Element Access
 *   call operator, row, col accessors
 *
 * - Arithmetic Operators
 *   add, subtract, negate, scalar multiply and divide, compound
 *
 * - Matrix Operations
 *   matrix multiply, matrix times vector, transpose, determinant,
 *   inverse
 *
 * - Memory Format
 *   RS_Mat3DMem set, get, toMat3D, float32 and float64
 *****************************************************************/

// Returns true if a and b are within a relative tolerance of each other
template <typename T>
bool isRelativelyEqual(T a, T b, T tolerance) {
    return std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
}

const float RELATIVE_TOLERANCE_FLOAT32 = 1e-5f;
const double RELATIVE_TOLERANCE_FLOAT64 = 1e-9;

/**** Test cases for RS_Mat3D construction and factories ****/

TEST(RS_Mat3DTest, ParameterizedConstructor) {
    RS_Mat3D<float> m(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
    EXPECT_EQ(m(0, 0), 1.0f);
    EXPECT_EQ(m(0, 2), 3.0f);
    EXPECT_EQ(m(1, 1), 5.0f);
    EXPECT_EQ(m(2, 2), 9.0f);
}

TEST(RS_Mat3DTest, EyeAndZero) {
    RS_Mat3D<float> eye = RS_Mat3D<float>::eye();
    EXPECT_EQ(eye(0, 0), 1.0f);
    EXPECT_EQ(eye(1, 1), 1.0f);
    EXPECT_EQ(eye(2, 2), 1.0f);
    EXPECT_EQ(eye(0, 1), 0.0f);

    RS_Mat3D<float> zero = RS_Mat3D<float>::zero();
    EXPECT_EQ(zero(0, 0), 0.0f);
    EXPECT_EQ(zero(2, 2), 0.0f);
}

TEST(RS_Mat3DTest, DiagonalAndFull) {
    RS_Mat3D<float> d = RS_Mat3D<float>::diagonal(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(d(0, 0), 1.0f);
    EXPECT_EQ(d(1, 1), 2.0f);
    EXPECT_EQ(d(2, 2), 3.0f);
    EXPECT_EQ(d(0, 1), 0.0f);

    RS_Mat3D<float> f = RS_Mat3D<float>::full(7.0f);
    EXPECT_EQ(f(0, 0), 7.0f);
    EXPECT_EQ(f(2, 1), 7.0f);
}

TEST(RS_Mat3DTest, ConstructorFromRowVectors) {
    RS_Vec3D<float> r0(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> r1(4.0f, 5.0f, 6.0f);
    RS_Vec3D<float> r2(7.0f, 8.0f, 9.0f);
    RS_Mat3D<float> m(r0, r1, r2);
    EXPECT_EQ(m(0, 1), 2.0f);
    EXPECT_EQ(m(1, 2), 6.0f);
    EXPECT_EQ(m(2, 0), 7.0f);
}

/**** Test cases for RS_Mat3D element access ****/

TEST(RS_Mat3DTest, RowAndColAccessor) {
    RS_Mat3D<float> m(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
    RS_Vec3D<float> row1 = m.row(1).toVec();
    EXPECT_EQ(row1.x, 4.0f);
    EXPECT_EQ(row1.y, 5.0f);
    EXPECT_EQ(row1.z, 6.0f);

    RS_Vec3D<float> col2 = m.col(2).toVec();
    EXPECT_EQ(col2.x, 3.0f);
    EXPECT_EQ(col2.y, 6.0f);
    EXPECT_EQ(col2.z, 9.0f);
}

/**** Test cases for RS_Mat3D arithmetic operators ****/

TEST(RS_Mat3DTest, AdditionSubtraction) {
    RS_Mat3D<float> a = RS_Mat3D<float>::full(2.0f);
    RS_Mat3D<float> b = RS_Mat3D<float>::full(3.0f);
    RS_Mat3D<float> sum = a + b;
    EXPECT_EQ(sum(0, 0), 5.0f);
    EXPECT_EQ(sum(2, 2), 5.0f);

    RS_Mat3D<float> diff = b - a;
    EXPECT_EQ(diff(0, 0), 1.0f);
    EXPECT_EQ(diff(2, 2), 1.0f);
}

TEST(RS_Mat3DTest, ScalarMultiplyDivide) {
    RS_Mat3D<float> m = RS_Mat3D<float>::full(2.0f);
    RS_Mat3D<float> mul = m * 3.0f;
    EXPECT_EQ(mul(0, 0), 6.0f);

    RS_Mat3D<float> div = m / 2.0f;
    EXPECT_EQ(div(0, 0), 1.0f);
}

TEST(RS_Mat3DTest, Negation) {
    RS_Mat3D<float> m = RS_Mat3D<float>::full(2.0f);
    RS_Mat3D<float> n = -m;
    EXPECT_EQ(n(0, 0), -2.0f);
    EXPECT_EQ(n(2, 2), -2.0f);
}

TEST(RS_Mat3DTest, CompoundAssignment) {
    RS_Mat3D<float> m = RS_Mat3D<float>::full(2.0f);
    m += RS_Mat3D<float>::full(1.0f);
    EXPECT_EQ(m(0, 0), 3.0f);
    m -= RS_Mat3D<float>::full(1.0f);
    EXPECT_EQ(m(0, 0), 2.0f);
    m *= 2.0f;
    EXPECT_EQ(m(0, 0), 4.0f);
    m /= 2.0f;
    EXPECT_EQ(m(0, 0), 2.0f);
}

/**** Test cases for RS_Mat3D matrix operations ****/

TEST(RS_Mat3DTest, MatrixTimesIdentity) {
    RS_Mat3D<float> m(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
    RS_Mat3D<float> result = m * RS_Mat3D<float>::eye();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            EXPECT_EQ(result(row, col), m(row, col));
        }
    }
}

TEST(RS_Mat3DTest, MatrixMultiplication) {
    RS_Mat3D<double> a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    RS_Mat3D<double> b(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    RS_Mat3D<double> result = a * b;
    EXPECT_EQ(result(0, 0), 30.0);
    EXPECT_EQ(result(0, 1), 24.0);
    EXPECT_EQ(result(0, 2), 18.0);
    EXPECT_EQ(result(2, 0), 138.0);
}

TEST(RS_Mat3DTest, MatrixTimesVector) {
    RS_Mat3D<float> m = RS_Mat3D<float>::eye();
    RS_Vec3D<float> v(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> result = m * v;
    EXPECT_EQ(result.x, 1.0f);
    EXPECT_EQ(result.y, 2.0f);
    EXPECT_EQ(result.z, 3.0f);

    RS_Mat3D<float> scale = RS_Mat3D<float>::diagonal(2.0f, 3.0f, 4.0f);
    RS_Vec3D<float> scaled = scale * v;
    EXPECT_EQ(scaled.x, 2.0f);
    EXPECT_EQ(scaled.y, 6.0f);
    EXPECT_EQ(scaled.z, 12.0f);
}

TEST(RS_Mat3DTest, Transpose) {
    RS_Mat3D<float> m(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
    RS_Mat3D<float> t = m.transpose();
    EXPECT_EQ(t(0, 1), m(1, 0));
    EXPECT_EQ(t(0, 2), m(2, 0));
    EXPECT_EQ(t(1, 2), m(2, 1));
}

TEST(RS_Mat3DTest, Determinant) {
    RS_Mat3D<double> eye = RS_Mat3D<double>::eye();
    EXPECT_EQ(eye.determinant(), 1.0);

    RS_Mat3D<double> m(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
    EXPECT_EQ(m.determinant(), 24.0);
}

TEST(RS_Mat3DTest, Inverse) {
    RS_Mat3D<double> m(4.0, 7.0, 2.0, 3.0, 6.0, 1.0, 2.0, 5.0, 3.0);
    RS_Mat3D<double> inv = m.inverse();
    RS_Mat3D<double> product = m * inv;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double expected = (row == col) ? 1.0 : 0.0;
            EXPECT_NEAR(product(row, col), expected, 1e-9);
        }
    }
}

/**** Test cases for RS_Mat3DMem ****/

TEST(RS_Mat3DMemTest, SetGetFloat32) {
    RS_Mat3D<float> m(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
    RS_Mat3DMem<RSE_FLOAT_32> mem(m);
    RS_Mat3D<float> result = mem.toMat3D<float>();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            EXPECT_EQ(result(row, col), m(row, col));
        }
    }
}

TEST(RS_Mat3DMemTest, AssignmentAndConvertFloat64) {
    RS_Mat3D<double> m(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5);
    RS_Mat3DMem<RSE_FLOAT_64> mem;
    mem = m;
    RS_Mat3D<double> result = mem.toMat3D<double>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_TRUE(isRelativelyEqual(result.data[i], m.data[i], RELATIVE_TOLERANCE_FLOAT64));
    }
}

TEST(RS_Mat3DMemTest, Zero) {
    RS_Mat3DMem<RSE_FLOAT_32> mem = RS_Mat3DMem<RSE_FLOAT_32>::zero();
    RS_Mat3D<float> result = mem.toMat3D<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(result.data[i], 0.0f);
    }
}
