#include <gtest/gtest.h>

#include "rise_origin/utils/math/rs_triangle.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_Triangle is a stateless utility of static methods templated
 * on the float type. It provides triangle math such as face
 * normal, bounding box, barycentric coordinates, uniform
 * sampling, and a watertight ray triangle intersection.
 *
 * - Geometry
 *   normal, unnormalizedNormal, bound
 *
 * - Sampling and Coordinates
 *   samplePoint, barycentric, pointEdgeDistance
 *
 * - Ray Intersection
 *   rayIntersect hit, miss, behind ray
 *
 * - Triangle Triangle Tests
 *   testTriTriIntersect, testSimpleProximity
 *****************************************************************/

using Tri = RS_Triangle<double>;
using Vec3 = RS_Vec3D<double>;

// Returns true if a and b are within a relative tolerance of each other
template <typename T>
bool isRelativelyEqual(T a, T b, T tolerance) {
    return std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
}

/**** Test cases for RS_Triangle geometry ****/

TEST(RS_TriangleTest, NormalOfXYPlane) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(1.0, 0.0, 0.0);
    Vec3 v2(0.0, 1.0, 0.0);
    Vec3 n = Tri::normal(v0, v1, v2);
    EXPECT_NEAR(n.x, 0.0, 1e-6);
    EXPECT_NEAR(n.y, 0.0, 1e-6);
    EXPECT_NEAR(n.z, 1.0, 1e-6);
}

TEST(RS_TriangleTest, NormalIsUnitLength) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(2.0, 0.0, 0.0);
    Vec3 v2(0.0, 3.0, 1.0);
    Vec3 n = Tri::normal(v0, v1, v2);
    EXPECT_NEAR(n.length(), 1.0, 1e-6);
}

TEST(RS_TriangleTest, DegenerateNormalIsZero) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(1.0, 0.0, 0.0);
    Vec3 v2(2.0, 0.0, 0.0);
    Vec3 n = Tri::normal(v0, v1, v2);
    EXPECT_EQ(n.x, 0.0);
    EXPECT_EQ(n.y, 0.0);
    EXPECT_EQ(n.z, 0.0);
}

TEST(RS_TriangleTest, UnnormalizedNormal) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(2.0, 0.0, 0.0);
    Vec3 v2(0.0, 2.0, 0.0);
    Vec3 n = Tri::unnormalizedNormal(v0, v1, v2);
    EXPECT_NEAR(n.z, 4.0, 1e-9);
}

TEST(RS_TriangleTest, Bound) {
    Vec3 v0(0.0, 1.0, 2.0);
    Vec3 v1(3.0, -1.0, 0.0);
    Vec3 v2(1.0, 4.0, 5.0);
    RS_Bound3D<double> b = Tri::bound(v0, v1, v2);
    EXPECT_EQ(b.min_x, 0.0);
    EXPECT_EQ(b.min_y, -1.0);
    EXPECT_EQ(b.min_z, 0.0);
    EXPECT_EQ(b.max_x, 3.0);
    EXPECT_EQ(b.max_y, 4.0);
    EXPECT_EQ(b.max_z, 5.0);
}

/**** Test cases for RS_Triangle sampling and coordinates ****/

TEST(RS_TriangleTest, SamplePointInsideBound) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(1.0, 0.0, 0.0);
    Vec3 v2(0.0, 1.0, 0.0);
    Vec3 p = Tri::samplePoint(v0, v1, v2, 0.3, 0.4);
    EXPECT_GE(p.x, 0.0);
    EXPECT_GE(p.y, 0.0);
    EXPECT_LE(p.x + p.y, 1.0 + 1e-9);
    EXPECT_NEAR(p.z, 0.0, 1e-9);
}

TEST(RS_TriangleTest, BarycentricAtVertices) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(1.0, 0.0, 0.0);
    Vec3 v2(0.0, 1.0, 0.0);
    double u, v, w;

    Tri::barycentric(v0, v0, v1, v2, u, v, w);
    EXPECT_NEAR(u, 1.0, 1e-9);
    EXPECT_NEAR(v, 0.0, 1e-9);
    EXPECT_NEAR(w, 0.0, 1e-9);

    Tri::barycentric(v1, v0, v1, v2, u, v, w);
    EXPECT_NEAR(u, 0.0, 1e-9);
    EXPECT_NEAR(v, 1.0, 1e-9);
    EXPECT_NEAR(w, 0.0, 1e-9);
}

TEST(RS_TriangleTest, BarycentricAtCenter) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(1.0, 0.0, 0.0);
    Vec3 v2(0.0, 1.0, 0.0);
    Vec3 center = (v0 + v1 + v2) / 3.0;
    double u, v, w;
    Tri::barycentric(center, v0, v1, v2, u, v, w);
    EXPECT_NEAR(u, 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(v, 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(w, 1.0 / 3.0, 1e-9);
}

TEST(RS_TriangleTest, PointEdgeDistance) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(2.0, 0.0, 0.0);
    Vec3 p(1.0, 3.0, 0.0);
    EXPECT_NEAR(Tri::pointEdgeDistance(p, v0, v1), 3.0, 1e-9);

    Vec3 beyond(5.0, 0.0, 0.0);
    EXPECT_NEAR(Tri::pointEdgeDistance(beyond, v0, v1), 3.0, 1e-9);
}

/**** Test cases for RS_Triangle ray intersection ****/

TEST(RS_TriangleTest, RayIntersectHitsCenter) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(1.0, 0.0, 0.0);
    Vec3 v2(0.0, 1.0, 0.0);
    Vec3 origin(0.25, 0.25, 1.0);
    Vec3 dir(0.0, 0.0, -1.0);
    double t, u, v;
    bool hit = Tri::rayIntersect(origin, dir, v0, v1, v2, 0.0, 100.0, t, u, v);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(t, 1.0, 1e-6);
}

TEST(RS_TriangleTest, RayIntersectMissesOutside) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(1.0, 0.0, 0.0);
    Vec3 v2(0.0, 1.0, 0.0);
    Vec3 origin(2.0, 2.0, 1.0);
    Vec3 dir(0.0, 0.0, -1.0);
    double t, u, v;
    bool hit = Tri::rayIntersect(origin, dir, v0, v1, v2, 0.0, 100.0, t, u, v);
    EXPECT_FALSE(hit);
}

TEST(RS_TriangleTest, RayIntersectBehindOrigin) {
    Vec3 v0(0.0, 0.0, 0.0);
    Vec3 v1(1.0, 0.0, 0.0);
    Vec3 v2(0.0, 1.0, 0.0);
    Vec3 origin(0.25, 0.25, 1.0);
    Vec3 dir(0.0, 0.0, 1.0);
    double t, u, v;
    bool hit = Tri::rayIntersect(origin, dir, v0, v1, v2, 0.0, 100.0, t, u, v);
    EXPECT_FALSE(hit);
}

/**** Test cases for RS_Triangle triangle triangle tests ****/

TEST(RS_TriangleTest, TriTriIntersectCrossing) {
    Vec3 a0(0.0, 0.0, 0.0);
    Vec3 a1(2.0, 0.0, 0.0);
    Vec3 a2(0.0, 2.0, 0.0);
    Vec3 a_normal = Tri::normal(a0, a1, a2);

    // B straddles A's plane so the crossing is non degenerate, with no vertex
    // lying exactly on A's plane.
    Vec3 b0(0.5, 0.5, -1.0);
    Vec3 b1(0.5, 0.5, 1.0);
    Vec3 b2(1.5, 0.5, 0.5);
    Vec3 b_normal = Tri::normal(b0, b1, b2);

    Vec3 seg0, seg1;
    bool hit = Tri::testTriTriIntersect(a0, a1, a2, b0, b1, b2, a_normal, b_normal, seg0, seg1);
    EXPECT_TRUE(hit);
}

TEST(RS_TriangleTest, TriTriIntersectSeparated) {
    Vec3 a0(0.0, 0.0, 0.0);
    Vec3 a1(2.0, 0.0, 0.0);
    Vec3 a2(0.0, 2.0, 0.0);
    Vec3 a_normal = Tri::normal(a0, a1, a2);

    Vec3 b0(0.5, 0.5, 5.0);
    Vec3 b1(0.5, 0.5, 7.0);
    Vec3 b2(1.5, 0.5, 6.0);
    Vec3 b_normal = Tri::normal(b0, b1, b2);

    Vec3 seg0, seg1;
    bool hit = Tri::testTriTriIntersect(a0, a1, a2, b0, b1, b2, a_normal, b_normal, seg0, seg1);
    EXPECT_FALSE(hit);
}

TEST(RS_TriangleTest, SimpleProximityPenetration) {
    Vec3 a0(0.0, 0.0, 0.0);
    Vec3 a1(2.0, 0.0, 0.0);
    Vec3 a2(0.0, 2.0, 0.0);
    Vec3 a_normal(0.0, 0.0, 1.0);

    Vec3 b0(0.5, 0.5, -0.5);
    Vec3 b1(1.0, 0.5, 0.5);
    Vec3 b2(0.5, 1.0, 0.5);
    Vec3 b_normal(0.0, 0.0, 1.0);

    Vec3 contact_point, contact_normal;
    double depth;
    bool found = Tri::testSimpleProximity(
        a0, a1, a2, b0, b1, b2, a_normal, b_normal, contact_point, contact_normal, depth
    );
    EXPECT_TRUE(found);
    EXPECT_GT(depth, 0.0);
}
