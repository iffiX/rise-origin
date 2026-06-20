#include <gtest/gtest.h>

#include "rise_origin/utils/math/rs_color.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_Color3D and RS_Color4D are plain old data color types whose
 * channel type is either RByte or float. They carry channel
 * arithmetic, luminance, gamma conversions, and conversions to
 * vectors and between the three and four channel forms.
 *
 * - Construction and Conversion
 *   parameter constructor, zero, byte and float cross conversion,
 *   vector constructor
 *
 * - Arithmetic Operators
 *   add, subtract, negate, scalar multiply and divide, dot,
 *   compound assign, equality
 *
 * - Color Operations
 *   luminance, clip, srgbToLinear, linearToSrgb
 *
 * - Channel Conversions
 *   toVec3D, toVec4D, toColor3D, toColor4D
 *****************************************************************/

// Returns true if a and b are within a relative tolerance of each other
template <typename T>
bool isRelativelyEqual(T a, T b, T tolerance) {
    return std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
}

const float RELATIVE_TOLERANCE_FLOAT32 = 1e-5f;

/**** Test cases for RS_Color3D construction and conversion ****/

TEST(RS_Color3DTest, ParameterizedConstructor) {
    RS_Color3D<float> c(0.1f, 0.2f, 0.3f);
    EXPECT_EQ(c.r, 0.1f);
    EXPECT_EQ(c.g, 0.2f);
    EXPECT_EQ(c.b, 0.3f);
}

TEST(RS_Color3DTest, ByteConstructor) {
    RS_Color3D<RByte> c(10, 20, 30);
    EXPECT_EQ(c.r, 10);
    EXPECT_EQ(c.g, 20);
    EXPECT_EQ(c.b, 30);
}

TEST(RS_Color3DTest, Zero) {
    RS_Color3D<float> c = RS_Color3D<float>::zero();
    EXPECT_EQ(c.r, 0.0f);
    EXPECT_EQ(c.g, 0.0f);
    EXPECT_EQ(c.b, 0.0f);
}

TEST(RS_Color3DTest, FloatToByteConversion) {
    RS_Color3D<float> f(1.0f, 0.5f, 0.0f);
    RS_Color3D<RByte> b(f);
    EXPECT_EQ(b.r, 255);
    EXPECT_EQ(b.g, static_cast<RByte>(0.5f * 255.0f));
    EXPECT_EQ(b.b, 0);
}

TEST(RS_Color3DTest, ByteToFloatConversion) {
    RS_Color3D<RByte> b(255, 0, 0);
    RS_Color3D<float> f(b);
    EXPECT_TRUE(isRelativelyEqual(f.r, 1.0f, RELATIVE_TOLERANCE_FLOAT32));
    EXPECT_EQ(f.g, 0.0f);
    EXPECT_EQ(f.b, 0.0f);
}

TEST(RS_Color3DTest, VectorConstructor) {
    RS_Vec3D<float> v(0.1f, 0.2f, 0.3f);
    RS_Color3D<float> c(v);
    EXPECT_EQ(c.r, 0.1f);
    EXPECT_EQ(c.g, 0.2f);
    EXPECT_EQ(c.b, 0.3f);
}

/**** Test cases for RS_Color3D arithmetic operators ****/

TEST(RS_Color3DTest, AdditionSubtraction) {
    RS_Color3D<float> a(0.1f, 0.2f, 0.3f);
    RS_Color3D<float> b(0.1f, 0.1f, 0.1f);
    RS_Color3D<float> sum = a + b;
    EXPECT_TRUE(isRelativelyEqual(sum.r, 0.2f, RELATIVE_TOLERANCE_FLOAT32));
    EXPECT_TRUE(isRelativelyEqual(sum.g, 0.3f, RELATIVE_TOLERANCE_FLOAT32));

    RS_Color3D<float> diff = a - b;
    EXPECT_NEAR(diff.r, 0.0f, 1e-5f);
    EXPECT_TRUE(isRelativelyEqual(diff.b, 0.2f, RELATIVE_TOLERANCE_FLOAT32));
}

TEST(RS_Color3DTest, ScalarMultiplyDivide) {
    RS_Color3D<float> c(0.1f, 0.2f, 0.4f);
    RS_Color3D<float> mul = c * 2.0f;
    EXPECT_TRUE(isRelativelyEqual(mul.r, 0.2f, RELATIVE_TOLERANCE_FLOAT32));
    EXPECT_TRUE(isRelativelyEqual(mul.b, 0.8f, RELATIVE_TOLERANCE_FLOAT32));

    RS_Color3D<float> div = c / 2.0f;
    EXPECT_TRUE(isRelativelyEqual(div.r, 0.05f, RELATIVE_TOLERANCE_FLOAT32));
    EXPECT_TRUE(isRelativelyEqual(div.b, 0.2f, RELATIVE_TOLERANCE_FLOAT32));
}

TEST(RS_Color3DTest, DotProduct) {
    RS_Color3D<float> a(1.0f, 2.0f, 3.0f);
    RS_Color3D<float> b(4.0f, 5.0f, 6.0f);
    EXPECT_EQ(a * b, 32.0f);
}

TEST(RS_Color3DTest, CompoundAssignment) {
    RS_Color3D<float> c(0.1f, 0.2f, 0.3f);
    c += RS_Color3D<float>(0.1f, 0.1f, 0.1f);
    EXPECT_TRUE(isRelativelyEqual(c.r, 0.2f, RELATIVE_TOLERANCE_FLOAT32));
    c *= 2.0f;
    EXPECT_TRUE(isRelativelyEqual(c.r, 0.4f, RELATIVE_TOLERANCE_FLOAT32));
}

TEST(RS_Color3DTest, Equality) {
    RS_Color3D<float> a(0.1f, 0.2f, 0.3f);
    RS_Color3D<float> b(0.1f, 0.2f, 0.3f);
    RS_Color3D<float> c(0.1f, 0.2f, 0.4f);
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a != c);
}

/**** Test cases for RS_Color3D color operations ****/

TEST(RS_Color3DTest, Luminance) {
    RS_Color3D<float> white(1.0f, 1.0f, 1.0f);
    EXPECT_TRUE(isRelativelyEqual(white.luminance(), 1.0f, 1e-4f));

    RS_Color3D<float> green(0.0f, 1.0f, 0.0f);
    EXPECT_TRUE(isRelativelyEqual(green.luminance(), 0.715160f, 1e-4f));
}

TEST(RS_Color3DTest, Clip) {
    RS_Color3D<float> c(1.5f, -0.5f, 0.5f);
    RS_Color3D<float> clipped = c.clip();
    EXPECT_EQ(clipped.r, 1.0f);
    EXPECT_EQ(clipped.g, 0.0f);
    EXPECT_EQ(clipped.b, 0.5f);
}

TEST(RS_Color3DTest, GammaRoundTrip) {
    RS_Color3D<float> c(0.5f, 0.25f, 0.75f);
    RS_Color3D<float> back = c.srgbToLinear().linearToSrgb();
    EXPECT_TRUE(isRelativelyEqual(back.r, 0.5f, 1e-4f));
    EXPECT_TRUE(isRelativelyEqual(back.g, 0.25f, 1e-4f));
    EXPECT_TRUE(isRelativelyEqual(back.b, 0.75f, 1e-4f));
}

/**** Test cases for RS_Color3D channel conversions ****/

TEST(RS_Color3DTest, ToVec3D) {
    RS_Color3D<float> c(0.1f, 0.2f, 0.3f);
    RS_Vec3D<float> v = c.toVec3D();
    EXPECT_EQ(v.x, 0.1f);
    EXPECT_EQ(v.y, 0.2f);
    EXPECT_EQ(v.z, 0.3f);
}

TEST(RS_Color3DTest, ToColor4D) {
    RS_Color3D<float> c(0.1f, 0.2f, 0.3f);
    RS_Color4D<float> c4 = c.toColor4D();
    EXPECT_EQ(c4.r, 0.1f);
    EXPECT_EQ(c4.g, 0.2f);
    EXPECT_EQ(c4.b, 0.3f);
    EXPECT_EQ(c4.a, 1.0f);
}

/**** Test cases for RS_Color4D ****/

TEST(RS_Color4DTest, ParameterizedConstructor) {
    RS_Color4D<float> c(0.1f, 0.2f, 0.3f, 0.4f);
    EXPECT_EQ(c.r, 0.1f);
    EXPECT_EQ(c.a, 0.4f);
}

TEST(RS_Color4DTest, AdditionAndDot) {
    RS_Color4D<float> a(1.0f, 2.0f, 3.0f, 4.0f);
    RS_Color4D<float> b(5.0f, 6.0f, 7.0f, 8.0f);
    RS_Color4D<float> sum = a + b;
    EXPECT_EQ(sum.r, 6.0f);
    EXPECT_EQ(sum.a, 12.0f);
    EXPECT_EQ(a * b, 70.0f);
}

TEST(RS_Color4DTest, ToColor3DAndToVec4D) {
    RS_Color4D<float> c(0.1f, 0.2f, 0.3f, 0.4f);
    RS_Color3D<float> c3 = c.toColor3D();
    EXPECT_EQ(c3.r, 0.1f);
    EXPECT_EQ(c3.b, 0.3f);

    RS_Vec4D<float> v = c.toVec4D();
    EXPECT_EQ(v.x, 0.1f);
    EXPECT_EQ(v.w, 0.4f);
}

TEST(RS_Color4DTest, ByteToFloatConversion) {
    RS_Color4D<RByte> b(255, 0, 255, 0);
    RS_Color4D<float> f(b);
    EXPECT_TRUE(isRelativelyEqual(f.r, 1.0f, RELATIVE_TOLERANCE_FLOAT32));
    EXPECT_EQ(f.g, 0.0f);
    EXPECT_TRUE(isRelativelyEqual(f.b, 1.0f, RELATIVE_TOLERANCE_FLOAT32));
    EXPECT_EQ(f.a, 0.0f);
}
