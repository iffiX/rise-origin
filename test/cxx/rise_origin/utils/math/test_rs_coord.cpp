#include <gtest/gtest.h>

#include "rise_origin/utils/math/rs_coord.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_Coord3D is a plain old data integer coordinate templated on
 * an integer element type. It carries integer arithmetic, bitwise
 * operators, lexicographic comparison, and grid helpers such as
 * align, ceilDiv, and floorDiv.
 *
 * - Construction and Factories
 *   parameter constructor, zero, one, full, upperBound, lowerBound
 *
 * - Element Wise Arithmetic
 *   add, subtract, multiply, divide, modulo with coord and scalar
 *
 * - Bitwise Operators
 *   and, or, xor, complement, shifts
 *
 * - Comparison
 *   equality, lexicographic less and greater
 *
 * - Reductions and Grid Helpers
 *   min, max, argmin, argmax, prod, align, ceilDiv, floorDiv,
 *   clip, toVec3D
 *****************************************************************/

using Coord = RS_Coord3D<int>;

/**** Test cases for RS_Coord3D construction and factories ****/

TEST(RS_Coord3DTest, ParameterizedConstructor) {
    Coord c(1, 2, 3);
    EXPECT_EQ(c.x, 1);
    EXPECT_EQ(c.y, 2);
    EXPECT_EQ(c.z, 3);
}

TEST(RS_Coord3DTest, ZeroOneFull) {
    Coord z = Coord::zero();
    EXPECT_EQ(z.x, 0);
    EXPECT_EQ(z.z, 0);

    Coord o = Coord::one();
    EXPECT_EQ(o.x, 1);
    EXPECT_EQ(o.z, 1);

    Coord f = Coord::full(5);
    EXPECT_EQ(f.x, 5);
    EXPECT_EQ(f.y, 5);
    EXPECT_EQ(f.z, 5);
}

TEST(RS_Coord3DTest, VectorConstructorRoundsDown) {
    RS_Vec3D<float> v(1.7f, 2.2f, 3.9f);
    Coord c(v);
    EXPECT_EQ(c.x, 1);
    EXPECT_EQ(c.y, 2);
    EXPECT_EQ(c.z, 3);
}

/**** Test cases for RS_Coord3D element wise arithmetic ****/

TEST(RS_Coord3DTest, CoordArithmetic) {
    Coord a(4, 6, 8);
    Coord b(2, 3, 4);
    EXPECT_TRUE((a + b) == Coord(6, 9, 12));
    EXPECT_TRUE((a - b) == Coord(2, 3, 4));
    EXPECT_TRUE((a * b) == Coord(8, 18, 32));
    EXPECT_TRUE((a / b) == Coord(2, 2, 2));
    EXPECT_TRUE((a % b) == Coord(0, 0, 0));
}

TEST(RS_Coord3DTest, ScalarArithmetic) {
    Coord a(4, 6, 8);
    EXPECT_TRUE((a + 2) == Coord(6, 8, 10));
    EXPECT_TRUE((a - 2) == Coord(2, 4, 6));
    EXPECT_TRUE((a * 2) == Coord(8, 12, 16));
    EXPECT_TRUE((a / 2) == Coord(2, 3, 4));
    EXPECT_TRUE((a % 3) == Coord(1, 0, 2));
}

TEST(RS_Coord3DTest, CompoundAssignment) {
    Coord a(4, 6, 8);
    a += Coord(1, 1, 1);
    EXPECT_TRUE(a == Coord(5, 7, 9));
    a -= Coord(1, 1, 1);
    EXPECT_TRUE(a == Coord(4, 6, 8));
    a *= 2;
    EXPECT_TRUE(a == Coord(8, 12, 16));
    a /= 2;
    EXPECT_TRUE(a == Coord(4, 6, 8));
}

TEST(RS_Coord3DTest, Negation) {
    Coord a(1, -2, 3);
    EXPECT_TRUE(-a == Coord(-1, 2, -3));
}

/**** Test cases for RS_Coord3D bitwise operators ****/

TEST(RS_Coord3DTest, BitwiseOperators) {
    Coord a(0b1100, 0b1010, 0b0110);
    Coord b(0b1010, 0b0110, 0b1100);
    EXPECT_TRUE((a & b) == Coord(0b1000, 0b0010, 0b0100));
    EXPECT_TRUE((a | b) == Coord(0b1110, 0b1110, 0b1110));
    EXPECT_TRUE((a ^ b) == Coord(0b0110, 0b1100, 0b1010));
}

TEST(RS_Coord3DTest, Shifts) {
    Coord a(1, 2, 4);
    EXPECT_TRUE((a << 1) == Coord(2, 4, 8));
    EXPECT_TRUE((a >> 1) == Coord(0, 1, 2));
}

/**** Test cases for RS_Coord3D comparison ****/

TEST(RS_Coord3DTest, Equality) {
    Coord a(1, 2, 3);
    Coord b(1, 2, 3);
    Coord c(1, 2, 4);
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a != c);
}

TEST(RS_Coord3DTest, LexicographicOrder) {
    Coord a(1, 2, 3);
    Coord b(1, 2, 4);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(a <= a);
    EXPECT_TRUE(a >= a);
}

/**** Test cases for RS_Coord3D reductions and grid helpers ****/

TEST(RS_Coord3DTest, MinMaxArg) {
    Coord c(3, 1, 2);
    EXPECT_EQ(c.min(), 1);
    EXPECT_EQ(c.max(), 3);
    EXPECT_EQ(c.argmin(), 1);
    EXPECT_EQ(c.argmax(), 0);
}

TEST(RS_Coord3DTest, MinMaxWithCoord) {
    Coord a(1, 5, 3);
    Coord b(4, 2, 6);
    EXPECT_TRUE(a.min(b) == Coord(1, 2, 3));
    EXPECT_TRUE(a.max(b) == Coord(4, 5, 6));
}

TEST(RS_Coord3DTest, Prod) {
    Coord c(2, 3, 4);
    EXPECT_EQ(c.prod(), 24);
}

TEST(RS_Coord3DTest, AlignAndDiv) {
    Coord c(5, 9, 16);
    EXPECT_TRUE(c.align(8) == Coord(8, 16, 16));
    EXPECT_TRUE(c.ceilDiv(8) == Coord(1, 2, 2));
    EXPECT_TRUE(c.floorDiv(8) == Coord(0, 1, 2));
}

TEST(RS_Coord3DTest, Clip) {
    Coord c(-2, 5, 12);
    c.clip(0, 10);
    EXPECT_EQ(c.x, 0);
    EXPECT_EQ(c.y, 5);
    EXPECT_EQ(c.z, 10);
}

TEST(RS_Coord3DTest, ToVec3D) {
    Coord c(1, 2, 3);
    RS_Vec3D<int> v = c.toVec3D();
    EXPECT_EQ(v.x, 1);
    EXPECT_EQ(v.y, 2);
    EXPECT_EQ(v.z, 3);
}

TEST(RS_Coord3DTest, IndexOperator) {
    Coord c(7, 8, 9);
    EXPECT_EQ(c[0], 7);
    EXPECT_EQ(c[1], 8);
    EXPECT_EQ(c[2], 9);
}
