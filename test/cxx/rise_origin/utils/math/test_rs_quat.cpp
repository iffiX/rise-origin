#include <gtest/gtest.h>

#include "rise_origin/utils/math/rs_quat.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_Quat3D is a plain old data quaternion templated on the
 * element type for either float or double resolution. It carries
 * rotation operations such as multiply, conjugate, inverse, and
 * vector rotation. RS_Quat3DMem is a packed memory format that is
 * always normalized on construction and converts to and from
 * RS_Quat3D.
 *
 * - Construction and Factories
 *   parameter constructor, unit, angle and axis constructor
 *
 * - Operators
 *   multiply, add, subtract, equality
 *
 * - Length and Inverse
 *   length, length2, normalized, conjugate, inverse
 *
 * - Rotation
 *   rotateVec3D, rotateVec3DInv, angle, axis, ortho axes
 *
 * - Memory Format
 *   RS_Quat3DMem construction, set, get, unit, float32 and float64
 *****************************************************************/

// Returns true if a and b are within a relative tolerance of each other
template <typename T>
bool isRelativelyEqual(T a, T b, T tolerance) {
    return std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
}

const float RELATIVE_TOLERANCE_FLOAT32 = 1e-5f;
const double RELATIVE_TOLERANCE_FLOAT64 = 1e-9;
const float RELATIVE_TOLERANCE_QUAT = 1e-4f;

/**** Test cases for RS_Quat3D construction and factories ****/

TEST(RS_Quat3DTest, ParameterizedConstructor) {
    RS_Quat3D<float> quat(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_EQ(quat.x, 1.0f);
    EXPECT_EQ(quat.y, 2.0f);
    EXPECT_EQ(quat.z, 3.0f);
    EXPECT_EQ(quat.w, 4.0f);
}

TEST(RS_Quat3DTest, Unit) {
    RS_Quat3D<float> quat = RS_Quat3D<float>::unit();
    EXPECT_EQ(quat.x, 0.0f);
    EXPECT_EQ(quat.y, 0.0f);
    EXPECT_EQ(quat.z, 0.0f);
    EXPECT_EQ(quat.w, 1.0f);
}

TEST(RS_Quat3DTest, AngleAxisConstructor) {
    RS_Vec3D<double> axis(0.0, 0.0, 1.0);
    double angle = RS_PI * 0.5;
    RS_Quat3D<double> quat(angle, axis);
    EXPECT_TRUE(isRelativelyEqual(quat.angle(), angle, 1e-6));
}

/**** Test cases for RS_Quat3D operators ****/

TEST(RS_Quat3DTest, Multiplication) {
    RS_Quat3D<double> a = RS_Quat3D<double>::unit();
    RS_Quat3D<double> b(0.1, 0.2, 0.3, 0.9);
    RS_Quat3D<double> result = a * b;
    EXPECT_TRUE(isRelativelyEqual(result.x, b.x, RELATIVE_TOLERANCE_FLOAT64));
    EXPECT_TRUE(isRelativelyEqual(result.y, b.y, RELATIVE_TOLERANCE_FLOAT64));
    EXPECT_TRUE(isRelativelyEqual(result.z, b.z, RELATIVE_TOLERANCE_FLOAT64));
    EXPECT_TRUE(isRelativelyEqual(result.w, b.w, RELATIVE_TOLERANCE_FLOAT64));
}

TEST(RS_Quat3DTest, ScalarMultiplication) {
    RS_Quat3D<float> quat(1.0f, 2.0f, 3.0f, 4.0f);
    RS_Quat3D<float> result = quat * 2.0f;
    EXPECT_EQ(result.x, 2.0f);
    EXPECT_EQ(result.y, 4.0f);
    EXPECT_EQ(result.z, 6.0f);
    EXPECT_EQ(result.w, 8.0f);
}

TEST(RS_Quat3DTest, AdditionSubtraction) {
    RS_Quat3D<float> a(1.0f, 2.0f, 3.0f, 4.0f);
    RS_Quat3D<float> b(1.0f, 1.0f, 1.0f, 1.0f);
    RS_Quat3D<float> sum = a + b;
    EXPECT_EQ(sum.x, 2.0f);
    EXPECT_EQ(sum.w, 5.0f);

    RS_Quat3D<float> diff = a - b;
    EXPECT_EQ(diff.x, 0.0f);
    EXPECT_EQ(diff.w, 3.0f);
}

TEST(RS_Quat3DTest, Equality) {
    RS_Quat3D<float> a(1.0f, 2.0f, 3.0f, 4.0f);
    RS_Quat3D<float> b(1.0f, 2.0f, 3.0f, 4.0f);
    RS_Quat3D<float> c(1.0f, 2.0f, 3.0f, 5.0f);
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a != c);
}

/**** Test cases for RS_Quat3D length and inverse ****/

TEST(RS_Quat3DTest, LengthAndLength2) {
    RS_Quat3D<float> quat(1.0f, 2.0f, 2.0f, 4.0f);
    EXPECT_EQ(quat.length2(), 25.0f);
    EXPECT_TRUE(isRelativelyEqual(quat.length(), 5.0f, RELATIVE_TOLERANCE_FLOAT32));
}

TEST(RS_Quat3DTest, Normalized) {
    RS_Quat3D<double> quat(1.0, 2.0, 2.0, 4.0);
    RS_Quat3D<double> n = quat.normalized();
    EXPECT_TRUE(isRelativelyEqual(n.length(), 1.0, RELATIVE_TOLERANCE_FLOAT64));
}

TEST(RS_Quat3DTest, Conjugate) {
    RS_Quat3D<float> quat(1.0f, 2.0f, 3.0f, 4.0f);
    RS_Quat3D<float> c = quat.conjugate();
    EXPECT_EQ(c.x, -1.0f);
    EXPECT_EQ(c.y, -2.0f);
    EXPECT_EQ(c.z, -3.0f);
    EXPECT_EQ(c.w, 4.0f);
}

TEST(RS_Quat3DTest, Inverse) {
    RS_Quat3D<double> quat(0.1, 0.2, 0.3, 0.9);
    RS_Quat3D<double> q = quat.normalized();
    RS_Quat3D<double> result = q * q.inverse();
    EXPECT_TRUE(isRelativelyEqual(result.w, 1.0, 1e-6));
    EXPECT_NEAR(result.x, 0.0, 1e-6);
    EXPECT_NEAR(result.y, 0.0, 1e-6);
    EXPECT_NEAR(result.z, 0.0, 1e-6);
}

/**** Test cases for RS_Quat3D rotation ****/

TEST(RS_Quat3DTest, RotateVec3DAboutZ) {
    RS_Vec3D<double> axis(0.0, 0.0, 1.0);
    RS_Quat3D<double> quat(RS_PI * 0.5, axis);
    RS_Vec3D<double> v(1.0, 0.0, 0.0);
    RS_Vec3D<double> rotated = quat.rotateVec3D(v);
    EXPECT_NEAR(rotated.x, 0.0, 1e-6);
    EXPECT_NEAR(rotated.y, 1.0, 1e-6);
    EXPECT_NEAR(rotated.z, 0.0, 1e-6);
}

TEST(RS_Quat3DTest, RotateVec3DInvUndoesRotation) {
    RS_Vec3D<double> axis(0.0, 1.0, 0.0);
    RS_Quat3D<double> quat(RS_PI * 0.3, axis);
    RS_Vec3D<double> v(1.0, 2.0, 3.0);
    RS_Vec3D<double> rotated = quat.rotateVec3D(v);
    RS_Vec3D<double> restored = quat.rotateVec3DInv(rotated);
    EXPECT_NEAR(restored.x, v.x, 1e-6);
    EXPECT_NEAR(restored.y, v.y, 1e-6);
    EXPECT_NEAR(restored.z, v.z, 1e-6);
}

TEST(RS_Quat3DTest, AngleAndAxis) {
    RS_Vec3D<double> axis(0.0, 0.0, 1.0);
    double angle = RS_PI * 0.5;
    RS_Quat3D<double> quat(angle, axis);
    EXPECT_TRUE(isRelativelyEqual(quat.angle(), angle, 1e-6));
    RS_Vec3D<double> recovered = quat.axis();
    EXPECT_NEAR(recovered.x, 0.0, 1e-6);
    EXPECT_NEAR(recovered.y, 0.0, 1e-6);
    EXPECT_NEAR(recovered.z, 1.0, 1e-6);
}

TEST(RS_Quat3DTest, OrthoAxesOfUnit) {
    RS_Quat3D<double> quat = RS_Quat3D<double>::unit();
    RS_Vec3D<double> ax = quat.toOrthoAxisX();
    RS_Vec3D<double> ay = quat.toOrthoAxisY();
    RS_Vec3D<double> az = quat.toOrthoAxisZ();
    EXPECT_NEAR(ax.x, 1.0, 1e-6);
    EXPECT_NEAR(ay.y, 1.0, 1e-6);
    EXPECT_NEAR(az.z, 1.0, 1e-6);
}

/**** Test cases for RS_Quat3DMem ****/

TEST(RS_Quat3DMemTest, ConstructorAndGetters) {
    RS_Quat3DMem<RSE_FLOAT_32> quat(0.574259f, 0.719324f, 0.364164f, 0.142171f);
    float x, y, z, w;
    quat.get(x, y, z, w);

    EXPECT_NEAR(x, 0.574259f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(y, 0.719324f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(z, 0.364164f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(w, 0.142171f, RELATIVE_TOLERANCE_QUAT);
}

TEST(RS_Quat3DMemTest, AssignmentOperator) {
    RS_Quat3DMem<RSE_FLOAT_32> quat1(0.574259f, 0.719324f, 0.364164f, 0.142171f);
    RS_Quat3DMem<RSE_FLOAT_32> quat2;
    quat2 = quat1;

    float x, y, z, w;
    quat2.get(x, y, z, w);

    EXPECT_NEAR(x, 0.574259f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(y, 0.719324f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(z, 0.364164f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(w, 0.142171f, RELATIVE_TOLERANCE_QUAT);
}

TEST(RS_Quat3DMemTest, NormalizedOnConstruction) {
    RS_Quat3DMem<RSE_FLOAT_64> quat(0.0, 0.0, 0.0, 5.0);
    double x, y, z, w;
    quat.get(x, y, z, w);
    EXPECT_NEAR(w, 1.0, RELATIVE_TOLERANCE_QUAT);
}

TEST(RS_Quat3DMemTest, Unit) {
    RS_Quat3DMem<RSE_FLOAT_32> quat = RS_Quat3DMem<RSE_FLOAT_32>::unit();
    float x, y, z, w;
    quat.get(x, y, z, w);
    EXPECT_EQ(x, 0.0f);
    EXPECT_EQ(y, 0.0f);
    EXPECT_EQ(z, 0.0f);
    EXPECT_EQ(w, 1.0f);
}

TEST(RS_Quat3DMemTest, ConvertToQuat3D) {
    RS_Quat3D<double> quat(0.1, 0.2, 0.3, 0.9);
    RS_Quat3D<double> normalized = quat.normalized();
    RS_Quat3DMem<RSE_FLOAT_64> mem(quat);
    RS_Quat3D<double> result = mem.toQuat3D<double>();
    EXPECT_NEAR(result.x, normalized.x, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(result.y, normalized.y, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(result.z, normalized.z, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(result.w, normalized.w, RELATIVE_TOLERANCE_QUAT);
}
