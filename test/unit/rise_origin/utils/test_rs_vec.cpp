#include <gtest/gtest.h>

#include "rise_origin/utils/rs_vec.h"

using namespace rise;

TEST(RSVec3DTest, ParameterizedConstructor) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(vec.x, 1.0f);
    EXPECT_EQ(vec.y, 2.0f);
    EXPECT_EQ(vec.z, 3.0f);
}

TEST(RSVec3DTest, Addition) {
    RS_Vec3D<float> vec1(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec2(4.0f, 5.0f, 6.0f);
    RS_Vec3D<float> result = vec1 + vec2;
    EXPECT_EQ(result.x, 5.0f);
    EXPECT_EQ(result.y, 7.0f);
    EXPECT_EQ(result.z, 9.0f);
}

TEST(RSVec3DTest, Subtraction) {
    RS_Vec3D<float> vec1(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec2(4.0f, 5.0f, 6.0f);
    RS_Vec3D<float> result = vec1 - vec2;
    EXPECT_EQ(result.x, -3.0f);
    EXPECT_EQ(result.y, -3.0f);
    EXPECT_EQ(result.z, -3.0f);
}

TEST(RSVec3DTest, Multiplication) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    float scalar = 2.0f;
    RS_Vec3D<float> result = vec * scalar;
    EXPECT_EQ(result.x, 2.0f);
    EXPECT_EQ(result.y, 4.0f);
    EXPECT_EQ(result.z, 6.0f);
}

TEST(RSVec3DTest, Division) {
    RS_Vec3D<float> vec(1.0f, 2.0f, 3.0f);
    float scalar = 2.0f;
    RS_Vec3D<float> result = vec / scalar;
    EXPECT_EQ(result.x, 0.5f);
    EXPECT_EQ(result.y, 1.0f);
    EXPECT_EQ(result.z, 1.5f);
}

TEST(RSVec3DTest, DotProduct) {
    RS_Vec3D<float> vec1(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec2(4.0f, 5.0f, 6.0f);
    float result = vec1.dot(vec2);
    EXPECT_EQ(result, 32.0f);
}

TEST(RSVec3DTest, CrossProduct) {
    RS_Vec3D<float> vec1(1.0f, 2.0f, 3.0f);
    RS_Vec3D<float> vec2(4.0f, 5.0f, 6.0f);
    RS_Vec3D<float> result = vec1.cross(vec2);
    EXPECT_EQ(result.x, -3.0f);
    EXPECT_EQ(result.y, 6.0f);
    EXPECT_EQ(result.z, -3.0f);
}

TEST(RSVec3DMemTest, ConstructorFromXYZ) {
    RS_Vec3DMem<RSE_FLOAT_32> vec(0.0f, 1.0f, 3.0f);
    RS_Vec3D<float> vec3D = vec.toVec3D<float>();
    EXPECT_EQ(vec3D.x, 0.0f);
    EXPECT_EQ(vec3D.y, 1.0f);
    EXPECT_EQ(vec3D.z, 3.0f);
}

TEST(RSVec3DMemTest, ConstructorFromVec3D) {
    RS_Vec3D<float> vec3D(0.0f, 1.0f, 3.0f);
    RS_Vec3DMem<RSE_FLOAT_32> vec(vec3D);
    RS_Vec3D<float> result = vec.toVec3D<float>();
    EXPECT_EQ(result.x, 0.0f);
    EXPECT_EQ(result.y, 1.0f);
    EXPECT_EQ(result.z, 3.0f);
}

TEST(RSVec3DMemTest, AssignmentOperatorFromVec3D) {
    RS_Vec3D<float> vec3D(0.0f, 1.0f, 3.0f);
    RS_Vec3DMem<RSE_FLOAT_32> vec;
    vec = vec3D;
    RS_Vec3D<float> result = vec.toVec3D<float>();
    EXPECT_EQ(result.x, 0.0f);
    EXPECT_EQ(result.y, 1.0f);
    EXPECT_EQ(result.z, 3.0f);
}

TEST(RSVec3DMemTest, ConvertToVec3D) {
    RS_Vec3DMem<RSE_FLOAT_32> vec(0.0f, 1.0f, 3.0f);
    RS_Vec3D<float> vec3D = vec.toVec3D<float>();
    EXPECT_EQ(vec3D.x, 0.0f);
    EXPECT_EQ(vec3D.y, 1.0f);
    EXPECT_EQ(vec3D.z, 3.0f);
}

TEST(RSVec3DMemTest, SetFromXYZ) {
    RS_Vec3DMem<RSE_FLOAT_32> vec;
    vec.set(0.0f, 1.0f, 3.0f);
    RS_Vec3D<float> vec3D = vec.toVec3D<float>();
    EXPECT_EQ(vec3D.x, 0.0f);
    EXPECT_EQ(vec3D.y, 1.0f);
    EXPECT_EQ(vec3D.z, 3.0f);
}

TEST(RSVec3DMemTest, ConstructorFromXYZ_Float64) {
    RS_Vec3DMem<RSE_FLOAT_64> vec(0.0, 1.0, 3.0);
    RS_Vec3D<double> vec3D = vec.toVec3D<double>();
    EXPECT_EQ(vec3D.x, 0.0);
    EXPECT_EQ(vec3D.y, 1.0);
    EXPECT_EQ(vec3D.z, 3.0);
}

TEST(RSVec3DMemTest, ConstructorFromVec3D_Float64) {
    RS_Vec3D<double> vec3D(0.0, 1.0, 3.0);
    RS_Vec3DMem<RSE_FLOAT_64> vec(vec3D);
    RS_Vec3D<double> result = vec.toVec3D<double>();
    EXPECT_EQ(result.x, 0.0);
    EXPECT_EQ(result.y, 1.0);
    EXPECT_EQ(result.z, 3.0);
}

TEST(RSVec3DMemTest, AssignmentOperatorFromVec3D_Float64) {
    RS_Vec3D<double> vec3D(0.0, 1.0, 3.0);
    RS_Vec3DMem<RSE_FLOAT_64> vec;
    vec = vec3D;
    RS_Vec3D<double> result = vec.toVec3D<double>();
    EXPECT_EQ(result.x, 0.0);
    EXPECT_EQ(result.y, 1.0);
    EXPECT_EQ(result.z, 3.0);
}

TEST(RSVec3DMemTest, ConvertToVec3D_Float64) {
    RS_Vec3DMem<RSE_FLOAT_64> vec(0.0, 1.0, 3.0);
    RS_Vec3D<double> vec3D = vec.toVec3D<double>();
    EXPECT_EQ(vec3D.x, 0.0);
    EXPECT_EQ(vec3D.y, 1.0);
    EXPECT_EQ(vec3D.z, 3.0);
}

TEST(RSVec3DMemTest, SetFromXYZ_Float64) {
    RS_Vec3DMem<RSE_FLOAT_64> vec;
    vec.set(0.0, 1.0, 3.0);
    RS_Vec3D<double> vec3D = vec.toVec3D<double>();
    EXPECT_EQ(vec3D.x, 0.0);
    EXPECT_EQ(vec3D.y, 1.0);
    EXPECT_EQ(vec3D.z, 3.0);
}

// Helper function to check relative error with tolerance
template <typename T>
bool isRelativelyEqual(T a, T b, T tolerance) {
    return std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
}

const float RELATIVE_TOLERANCE_FLOAT32 = 1e-6f;
const double RELATIVE_TOLERANCE_FLOAT64 = 1e-9;
// Small values for testing across all floating-point types
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

// Test small values for accuracy and relative tolerance for Float32
TEST(RSVec3DMemTest, SmallValuesTest_Float32) {
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

// Test small values for accuracy and relative tolerance for Float64
TEST(RSVec3DMemTest, SmallValuesTest_Float64) {
    for (double v : testValues) {
        RS_Vec3DMem<RSE_FLOAT_64> vec(v, v, v);
        RS_Vec3D<double> result = vec.toVec3D<double>();
        EXPECT_TRUE(isRelativelyEqual(result.x, v, RELATIVE_TOLERANCE_FLOAT64));
        EXPECT_TRUE(isRelativelyEqual(result.y, v, RELATIVE_TOLERANCE_FLOAT64));
        EXPECT_TRUE(isRelativelyEqual(result.z, v, RELATIVE_TOLERANCE_FLOAT64));
    }
}
