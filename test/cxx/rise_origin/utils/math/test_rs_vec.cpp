#include <gtest/gtest.h>

#include "rise_origin/utils/math/rs_vec.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_Vec2D, RS_Vec3D, and RS_Vec4D are plain old data vector
 * types templated on the element type for either float or double
 * resolution. RS_Vec3DMem is a packed memory format that converts
 * to and from RS_Vec3D.
 *
 * - Construction and Factories
 *   parameter constructor, zero, one, full
 *
 * - Arithmetic Operators
 *   add, subtract, negate, scalar multiply and divide, compound
 *   assign, equality
 *
 * - Vector Operations
 *   dot, cross, length, length2, normalized, dist, project
 *
 * - Reductions and Helpers
 *   sum, mean, min, max, abs, elemMul, elemDiv
 *
 * - Memory Format
 *   RS_Vec3DMem construction, set, get, toVec3D, float32 and float64
 *****************************************************************/

// Returns true if a and b are within a relative tolerance of each other
template <typename T>
bool isRelativelyEqual(T a, T b, T tolerance) {
    return std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
}

const float RELATIVE_TOLERANCE_FLOAT32 = 1e-6f;
const double RELATIVE_TOLERANCE_FLOAT64 = 1e-9;

// Small values for testing across all floating point types
std::array<double, 14> testValues = {
    4.57e-8,
    1.75e-7,
    3.45e-6,
    8.27e-5,
    1.42e-4,  //
    2.38e-3,
    5.71e-2,
    4.13e-1,
    1.81,
    12.9,  //
    103.5,
    8710,
    67300,
    552000
};

/**** Test cases for RS_Vec3D construction and factories ****/

TEST(RS_Vec3DTest, ParameterizedConstructor) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(vec.x, 1.0f);
    EXPECT_EQ(vec.y, 2.0f);
    EXPECT_EQ(vec.z, 3.0f);
}

TEST(RS_Vec3DTest, ZeroOneFull) {
    RS_Vec3D<float> z = RS_Vec3D<float>::zero();
    EXPECT_EQ(z.x, 0.0f);
    EXPECT_EQ(z.y, 0.0f);
    EXPECT_EQ(z.z, 0.0f);

    RS_Vec3D<float> o = RS_Vec3D<float>::one();
    EXPECT_EQ(o.x, 1.0f);
    EXPECT_EQ(o.y, 1.0f);
    EXPECT_EQ(o.z, 1.0f);

    RS_Vec3D<float> f = RS_Vec3D<float>::full(4.0f);
    EXPECT_EQ(f.x, 4.0f);
    EXPECT_EQ(f.y, 4.0f);
    EXPECT_EQ(f.z, 4.0f);
}

TEST(RS_Vec3DTest, IndexOperator) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(vec[0], 1.0f);
    EXPECT_EQ(vec[1], 2.0f);
    EXPECT_EQ(vec[2], 3.0f);
}

/**** Test cases for RS_Vec3D arithmetic operators ****/

TEST(RS_Vec3DTest, Addition) {
    RS_Vec3D<float> vec1(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec2(4.0f, 5.0f, 6.0f);
    RS_Vec3D<float> result = vec1 + vec2;
    EXPECT_EQ(result.x, 5.0f);
    EXPECT_EQ(result.y, 7.0f);
    EXPECT_EQ(result.z, 9.0f);
}

TEST(RS_Vec3DTest, Subtraction) {
    RS_Vec3D<float> vec1(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec2(4.0f, 5.0f, 6.0f);
    RS_Vec3D<float> result = vec1 - vec2;
    EXPECT_EQ(result.x, -3.0f);
    EXPECT_EQ(result.y, -3.0f);
    EXPECT_EQ(result.z, -3.0f);
}

TEST(RS_Vec3DTest, Negation) {
    RS_Vec3D<float> vec(1.0f, -2.0f, 3.0f);
    RS_Vec3D<float> result = -vec;
    EXPECT_EQ(result.x, -1.0f);
    EXPECT_EQ(result.y, 2.0f);
    EXPECT_EQ(result.z, -3.0f);
}

TEST(RS_Vec3DTest, ScalarMultiplication) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> result = vec * 2.0f;
    EXPECT_EQ(result.x, 2.0f);
    EXPECT_EQ(result.y, 4.0f);
    EXPECT_EQ(result.z, 6.0f);

    RS_Vec3D<float> left = 2.0f * vec;
    EXPECT_EQ(left.x, 2.0f);
    EXPECT_EQ(left.y, 4.0f);
    EXPECT_EQ(left.z, 6.0f);
}

TEST(RS_Vec3DTest, ScalarDivision) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> result = vec / 2.0f;
    EXPECT_EQ(result.x, 0.5f);
    EXPECT_EQ(result.y, 1.0f);
    EXPECT_EQ(result.z, 1.5f);
}

TEST(RS_Vec3DTest, CompoundAssignment) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    vec += RS_Vec3D<float>(1.0f, 1.0f, 1.0f);
    EXPECT_EQ(vec.x, 2.0f);
    EXPECT_EQ(vec.y, 3.0f);
    EXPECT_EQ(vec.z, 4.0f);

    vec -= RS_Vec3D<float>(1.0f, 1.0f, 1.0f);
    EXPECT_EQ(vec.x, 1.0f);
    EXPECT_EQ(vec.y, 2.0f);
    EXPECT_EQ(vec.z, 3.0f);

    vec *= 2.0f;
    EXPECT_EQ(vec.x, 2.0f);
    EXPECT_EQ(vec.y, 4.0f);
    EXPECT_EQ(vec.z, 6.0f);

    vec /= 2.0f;
    EXPECT_EQ(vec.x, 1.0f);
    EXPECT_EQ(vec.y, 2.0f);
    EXPECT_EQ(vec.z, 3.0f);
}

TEST(RS_Vec3DTest, Equality) {
    RS_Vec3D<float> vec1(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec2(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec3(1.0f, 2.0f, 4.0f);
    EXPECT_TRUE(vec1 == vec2);
    EXPECT_FALSE(vec1 == vec3);
    EXPECT_TRUE(vec1 != vec3);
    EXPECT_FALSE(vec1 != vec2);
}

/**** Test cases for RS_Vec3D vector operations ****/

TEST(RS_Vec3DTest, DotProduct) {
    RS_Vec3D<float> vec1(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec2(4.0f, 5.0f, 6.0f);
    EXPECT_EQ(vec1.dot(vec2), 32.0f);
}

TEST(RS_Vec3DTest, CrossProduct) {
    RS_Vec3D<float> vec1(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec2(4.0f, 5.0f, 6.0f);
    RS_Vec3D<float> result = vec1.cross(vec2);
    EXPECT_EQ(result.x, -3.0f);
    EXPECT_EQ(result.y, 6.0f);
    EXPECT_EQ(result.z, -3.0f);
}

TEST(RS_Vec3DTest, LengthAndLength2) {
    RS_Vec3D<float> vec(3.0f, 4.0f, 0.0f);
    EXPECT_EQ(vec.length2(), 25.0f);
    EXPECT_TRUE(isRelativelyEqual(vec.length(), 5.0f, RELATIVE_TOLERANCE_FLOAT32));
}

TEST(RS_Vec3DTest, Normalized) {
    RS_Vec3D<double> vec(0.0, 3.0, 4.0);
    RS_Vec3D<double> n = vec.normalized();
    EXPECT_TRUE(isRelativelyEqual(n.length(), 1.0, RELATIVE_TOLERANCE_FLOAT64));
    EXPECT_TRUE(isRelativelyEqual(n.y, 0.6, RELATIVE_TOLERANCE_FLOAT64));
    EXPECT_TRUE(isRelativelyEqual(n.z, 0.8, RELATIVE_TOLERANCE_FLOAT64));

    RS_Vec3D<double> zero = RS_Vec3D<double>::zero().normalized();
    EXPECT_EQ(zero.x, 0.0);
    EXPECT_EQ(zero.y, 0.0);
    EXPECT_EQ(zero.z, 0.0);
}

TEST(RS_Vec3DTest, Distance) {
    RS_Vec3D<float> a(0.0f, 0.0f, 0.0f);
    RS_Vec3D<float> b(3.0f, 4.0f, 0.0f);
    EXPECT_EQ(a.dist2(b), 25.0f);
    EXPECT_TRUE(isRelativelyEqual(a.dist(b), 5.0f, RELATIVE_TOLERANCE_FLOAT32));
}

TEST(RS_Vec3DTest, Project) {
    RS_Vec3D<double> v(2.0, 3.0, 0.0);
    RS_Vec3D<double> axis(1.0, 0.0, 0.0);
    RS_Vec3D<double> p = v.project(axis);
    EXPECT_TRUE(isRelativelyEqual(p.x, 2.0, RELATIVE_TOLERANCE_FLOAT64));
    EXPECT_EQ(p.y, 0.0);
    EXPECT_EQ(p.z, 0.0);
}

/**** Test cases for RS_Vec3D reductions and helpers ****/

TEST(RS_Vec3DTest, SumMeanProd) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(vec.sum(), 6.0f);
    EXPECT_TRUE(isRelativelyEqual(vec.mean(), 2.0f, RELATIVE_TOLERANCE_FLOAT32));
    EXPECT_EQ(vec.prod(), 6.0f);
}

TEST(RS_Vec3DTest, MinMaxArg) {
    RS_Vec3D<float> vec(3.0f, 1.0f, 2.0f);
    EXPECT_EQ(vec.min(), 1.0f);
    EXPECT_EQ(vec.max(), 3.0f);
    EXPECT_EQ(vec.argmin(), 1);
    EXPECT_EQ(vec.argmax(), 0);
}

TEST(RS_Vec3DTest, MinMaxWithVector) {
    RS_Vec3D<float> a(1.0f, 5.0f, 3.0f);
    RS_Vec3D<float> b(4.0f, 2.0f, 6.0f);
    RS_Vec3D<float> mn = a.min(b);
    RS_Vec3D<float> mx = a.max(b);
    EXPECT_EQ(mn.x, 1.0f);
    EXPECT_EQ(mn.y, 2.0f);
    EXPECT_EQ(mn.z, 3.0f);
    EXPECT_EQ(mx.x, 4.0f);
    EXPECT_EQ(mx.y, 5.0f);
    EXPECT_EQ(mx.z, 6.0f);
}

TEST(RS_Vec3DTest, AbsElemMulElemDiv) {
    RS_Vec3D<float> a(-1.0f, 2.0f, -3.0f);
    RS_Vec3D<float> abs = a.abs();
    EXPECT_EQ(abs.x, 1.0f);
    EXPECT_EQ(abs.y, 2.0f);
    EXPECT_EQ(abs.z, 3.0f);

    RS_Vec3D<float> b(2.0f, 3.0f, 4.0f);
    RS_Vec3D<float> c(2.0f, 2.0f, 2.0f);
    RS_Vec3D<float> mul = b.elemMul(c);
    EXPECT_EQ(mul.x, 4.0f);
    EXPECT_EQ(mul.y, 6.0f);
    EXPECT_EQ(mul.z, 8.0f);

    RS_Vec3D<float> div = b.elemDiv(c);
    EXPECT_EQ(div.x, 1.0f);
    EXPECT_EQ(div.y, 1.5f);
    EXPECT_EQ(div.z, 2.0f);
}

TEST(RS_Vec3DTest, IsInBound) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    EXPECT_TRUE(vec.isInBound(0.0f, 4.0f));
    EXPECT_FALSE(vec.isInBound(0.0f, 2.0f));
}

/**** Test cases for RS_Vec2D ****/

TEST(RS_Vec2DTest, ConstructorAndArithmetic) {
    RS_Vec2D<float> a(1.0f, 2.0f);
    RS_Vec2D<float> b(3.0f, 4.0f);
    EXPECT_EQ(a.x, 1.0f);
    EXPECT_EQ(a.y, 2.0f);

    RS_Vec2D<float> sum = a + b;
    EXPECT_EQ(sum.x, 4.0f);
    EXPECT_EQ(sum.y, 6.0f);

    EXPECT_EQ(a.dot(b), 11.0f);
}

TEST(RS_Vec2DTest, LengthAndNormalized) {
    RS_Vec2D<double> vec(3.0, 4.0);
    EXPECT_EQ(vec.length2(), 25.0);
    EXPECT_TRUE(isRelativelyEqual(vec.length(), 5.0, RELATIVE_TOLERANCE_FLOAT64));
    RS_Vec2D<double> n = vec.normalized();
    EXPECT_TRUE(isRelativelyEqual(n.length(), 1.0, RELATIVE_TOLERANCE_FLOAT64));
}

/**** Test cases for RS_Vec4D ****/

TEST(RS_Vec4DTest, ConstructorAndArithmetic) {
    RS_Vec4D<float> a(1.0f, 2.0f, 3.0f, 4.0f);
    RS_Vec4D<float> b(5.0f, 6.0f, 7.0f, 8.0f);
    EXPECT_EQ(a.w, 4.0f);

    RS_Vec4D<float> sum = a + b;
    EXPECT_EQ(sum.x, 6.0f);
    EXPECT_EQ(sum.w, 12.0f);

    EXPECT_EQ(a.dot(b), 70.0f);
}

/**** Test cases for RS_Vec3DMem ****/

TEST(RS_Vec3DMemTest, ConstructorFromXYZ) {
    RS_Vec3DMem<RSE_FLOAT_32> vec(0.0f, 1.0f, 3.0f);
    RS_Vec3D<float> vec3D = vec.toVec3D<float>();
    EXPECT_EQ(vec3D.x, 0.0f);
    EXPECT_EQ(vec3D.y, 1.0f);
    EXPECT_EQ(vec3D.z, 3.0f);
}

TEST(RS_Vec3DMemTest, ConstructorFromVec3D) {
    RS_Vec3D<float> vec3D(0.0f, 1.0f, 3.0f);
    RS_Vec3DMem<RSE_FLOAT_32> vec(vec3D);
    RS_Vec3D<float> result = vec.toVec3D<float>();
    EXPECT_EQ(result.x, 0.0f);
    EXPECT_EQ(result.y, 1.0f);
    EXPECT_EQ(result.z, 3.0f);
}

TEST(RS_Vec3DMemTest, AssignmentOperatorFromVec3D) {
    RS_Vec3D<float> vec3D(0.0f, 1.0f, 3.0f);
    RS_Vec3DMem<RSE_FLOAT_32> vec;
    vec = vec3D;
    RS_Vec3D<float> result = vec.toVec3D<float>();
    EXPECT_EQ(result.x, 0.0f);
    EXPECT_EQ(result.y, 1.0f);
    EXPECT_EQ(result.z, 3.0f);
}

TEST(RS_Vec3DMemTest, SetFromXYZ) {
    RS_Vec3DMem<RSE_FLOAT_32> vec;
    vec.set(0.0f, 1.0f, 3.0f);
    RS_Vec3D<float> vec3D = vec.toVec3D<float>();
    EXPECT_EQ(vec3D.x, 0.0f);
    EXPECT_EQ(vec3D.y, 1.0f);
    EXPECT_EQ(vec3D.z, 3.0f);
}

TEST(RS_Vec3DMemTest, ConstructorFromXYZ_Float64) {
    RS_Vec3DMem<RSE_FLOAT_64> vec(0.0, 1.0, 3.0);
    RS_Vec3D<double> vec3D = vec.toVec3D<double>();
    EXPECT_EQ(vec3D.x, 0.0);
    EXPECT_EQ(vec3D.y, 1.0);
    EXPECT_EQ(vec3D.z, 3.0);
}

TEST(RS_Vec3DMemTest, SmallValuesTest_Float32) {
    for (double v : testValues) {
        RS_Vec3DMem<RSE_FLOAT_32> vec(
            static_cast<float>(v), static_cast<float>(v), static_cast<float>(v)
        );
        RS_Vec3D<float> result = vec.toVec3D<float>();
        EXPECT_TRUE(isRelativelyEqual(result.x, static_cast<float>(v), RELATIVE_TOLERANCE_FLOAT32));
        EXPECT_TRUE(isRelativelyEqual(result.y, static_cast<float>(v), RELATIVE_TOLERANCE_FLOAT32));
        EXPECT_TRUE(isRelativelyEqual(result.z, static_cast<float>(v), RELATIVE_TOLERANCE_FLOAT32));
    }
}

TEST(RS_Vec3DMemTest, SmallValuesTest_Float64) {
    for (double v : testValues) {
        RS_Vec3DMem<RSE_FLOAT_64> vec(v, v, v);
        RS_Vec3D<double> result = vec.toVec3D<double>();
        EXPECT_TRUE(isRelativelyEqual(result.x, v, RELATIVE_TOLERANCE_FLOAT64));
        EXPECT_TRUE(isRelativelyEqual(result.y, v, RELATIVE_TOLERANCE_FLOAT64));
        EXPECT_TRUE(isRelativelyEqual(result.z, v, RELATIVE_TOLERANCE_FLOAT64));
    }
}
