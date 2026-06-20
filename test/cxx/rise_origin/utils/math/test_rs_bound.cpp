#include <gtest/gtest.h>

#include "rise_origin/utils/math/rs_bound.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_Bound3D is a plain old data axis aligned bounding box
 * templated on the element type. It stores the minimum and
 * maximum corner of the box and carries queries such as contains
 * and intersects plus combine operations like union and
 * intersection.
 *
 * - Construction and Factories
 *   parameter constructor, vector constructor, single point,
 *   zero, lowest, highest
 *
 * - Geometry Queries
 *   center, size, volume, minBound, maxBound, isValid
 *
 * - Containment and Intersection
 *   isContaining point, isContaining bound, isIntersecting,
 *   closestPoint
 *
 * - Combine and Transform
 *   unionWith, intersectionWith, extend, expand, shift
 *
 * - Operators
 *   equality
 *****************************************************************/

/**** Test cases for RS_Bound3D construction and factories ****/

TEST(RS_Bound3DTest, ParameterizedConstructor) {
    RS_Bound3D<float> b(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
    EXPECT_EQ(b.min_x, 0.0f);
    EXPECT_EQ(b.min_y, 1.0f);
    EXPECT_EQ(b.min_z, 2.0f);
    EXPECT_EQ(b.max_x, 3.0f);
    EXPECT_EQ(b.max_y, 4.0f);
    EXPECT_EQ(b.max_z, 5.0f);
}

TEST(RS_Bound3DTest, VectorConstructor) {
    RS_Vec3D<float> mn(0.0f, 0.0f, 0.0f);
    RS_Vec3D<float> mx(2.0f, 4.0f, 6.0f);
    RS_Bound3D<float> b(mn, mx);
    EXPECT_EQ(b.min_x, 0.0f);
    EXPECT_EQ(b.max_z, 6.0f);
}

TEST(RS_Bound3DTest, SinglePointConstructor) {
    RS_Vec3D<float> p(1.0f, 2.0f, 3.0f);
    RS_Bound3D<float> b(p);
    EXPECT_EQ(b.min_x, 1.0f);
    EXPECT_EQ(b.max_x, 1.0f);
    EXPECT_EQ(b.volume(), 0.0f);
}

TEST(RS_Bound3DTest, ZeroFactory) {
    RS_Bound3D<float> b = RS_Bound3D<float>::zero();
    EXPECT_EQ(b.min_x, 0.0f);
    EXPECT_EQ(b.max_z, 0.0f);
}

/**** Test cases for RS_Bound3D geometry queries ****/

TEST(RS_Bound3DTest, CenterSizeVolume) {
    RS_Bound3D<float> b(0.0f, 0.0f, 0.0f, 2.0f, 4.0f, 6.0f);
    RS_Vec3D<float> center = b.center();
    EXPECT_EQ(center.x, 1.0f);
    EXPECT_EQ(center.y, 2.0f);
    EXPECT_EQ(center.z, 3.0f);

    RS_Vec3D<float> size = b.size();
    EXPECT_EQ(size.x, 2.0f);
    EXPECT_EQ(size.y, 4.0f);
    EXPECT_EQ(size.z, 6.0f);

    EXPECT_EQ(b.volume(), 48.0f);
}

TEST(RS_Bound3DTest, MinMaxBound) {
    RS_Bound3D<float> b(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
    RS_Vec3D<float> mn = b.minBound();
    RS_Vec3D<float> mx = b.maxBound();
    EXPECT_EQ(mn.x, 0.0f);
    EXPECT_EQ(mn.z, 2.0f);
    EXPECT_EQ(mx.x, 3.0f);
    EXPECT_EQ(mx.z, 5.0f);
}

TEST(RS_Bound3DTest, IsValid) {
    RS_Bound3D<float> valid(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    EXPECT_TRUE(valid.isValid());

    RS_Bound3D<float> invalid(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f);
    EXPECT_FALSE(invalid.isValid());
}

/**** Test cases for RS_Bound3D containment and intersection ****/

TEST(RS_Bound3DTest, IsContainingPoint) {
    RS_Bound3D<float> b(0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f);
    EXPECT_TRUE(b.isContaining(RS_Vec3D<float>(1.0f, 1.0f, 1.0f)));
    EXPECT_TRUE(b.isContaining(RS_Vec3D<float>(0.0f, 0.0f, 0.0f)));
    EXPECT_FALSE(b.isContaining(RS_Vec3D<float>(3.0f, 1.0f, 1.0f)));
}

TEST(RS_Bound3DTest, IsContainingBound) {
    RS_Bound3D<float> outer(0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 4.0f);
    RS_Bound3D<float> inner(1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f);
    RS_Bound3D<float> overlap(2.0f, 2.0f, 2.0f, 5.0f, 5.0f, 5.0f);
    EXPECT_TRUE(outer.isContaining(inner));
    EXPECT_FALSE(outer.isContaining(overlap));
}

TEST(RS_Bound3DTest, IsIntersecting) {
    RS_Bound3D<float> a(0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f);
    RS_Bound3D<float> b(1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 3.0f);
    RS_Bound3D<float> c(5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f);
    EXPECT_TRUE(a.isIntersecting(b));
    EXPECT_FALSE(a.isIntersecting(c));
}

TEST(RS_Bound3DTest, ClosestPoint) {
    RS_Bound3D<float> b(0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f);
    RS_Vec3D<float> outside(5.0f, -1.0f, 1.0f);
    RS_Vec3D<float> closest = b.closestPoint(outside);
    EXPECT_EQ(closest.x, 2.0f);
    EXPECT_EQ(closest.y, 0.0f);
    EXPECT_EQ(closest.z, 1.0f);
}

/**** Test cases for RS_Bound3D combine and transform ****/

TEST(RS_Bound3DTest, UnionWith) {
    RS_Bound3D<float> a(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    RS_Bound3D<float> b(2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f);
    RS_Bound3D<float> u = a.unionWith(b);
    EXPECT_EQ(u.min_x, 0.0f);
    EXPECT_EQ(u.max_x, 3.0f);
    EXPECT_EQ(u.max_z, 3.0f);
}

TEST(RS_Bound3DTest, IntersectionWith) {
    RS_Bound3D<float> a(0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f);
    RS_Bound3D<float> b(1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 3.0f);
    RS_Bound3D<float> i = a.intersectionWith(b);
    EXPECT_EQ(i.min_x, 1.0f);
    EXPECT_EQ(i.max_x, 2.0f);
    EXPECT_EQ(i.max_z, 2.0f);
}

TEST(RS_Bound3DTest, Extend) {
    RS_Bound3D<float> b(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    RS_Bound3D<float> e = b.extend(RS_Vec3D<float>(3.0f, -1.0f, 2.0f));
    EXPECT_EQ(e.min_y, -1.0f);
    EXPECT_EQ(e.max_x, 3.0f);
    EXPECT_EQ(e.max_z, 2.0f);
}

TEST(RS_Bound3DTest, ExpandByAmount) {
    RS_Bound3D<float> b(1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f);
    RS_Bound3D<float> e = b.expand(1.0f);
    EXPECT_EQ(e.min_x, 0.0f);
    EXPECT_EQ(e.max_x, 3.0f);
}

TEST(RS_Bound3DTest, Shift) {
    RS_Bound3D<float> b(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    RS_Bound3D<float> s = b.shift(RS_Vec3D<float>(2.0f, 3.0f, 4.0f));
    EXPECT_EQ(s.min_x, 2.0f);
    EXPECT_EQ(s.max_y, 4.0f);
    EXPECT_EQ(s.max_z, 5.0f);
}

/**** Test cases for RS_Bound3D operators ****/

TEST(RS_Bound3DTest, Equality) {
    RS_Bound3D<float> a(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    RS_Bound3D<float> b(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    RS_Bound3D<float> c(0.0f, 0.0f, 0.0f, 2.0f, 1.0f, 1.0f);
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a != c);
}
