// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_TRIANGLE_H
#define RS_TRIANGLE_H

#include "rise_origin/utils/math/rs_bound.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

/**
 * @brief Stateless triangle math utility with HOST_DEVICE_INLINE_QUALIFIER static methods.
 *
 * @tparam FloatType float or double
 */
template <typename FloatType = float>
struct RS_Triangle {
    using Vec3 = RS_Vec3D<FloatType>;
    using Bound3 = RS_Bound3D<FloatType>;
    using FI = RS_FloatIntrinsics<FloatType>;

    /**
     * @brief Compute normalized face normal via cross product: (v1 - v0) x (v2 - v0).
     * @note Intentionally using fast math for performance and not accuracy.
     * @return Normalized normal, or zero vector for degenerate triangles.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3
    normal(const Vec3 &v0, const Vec3 &v1, const Vec3 &v2) {
        Vec3 n = (v1 - v0).cross(v2 - v0);
        FloatType len2 = n * n;
        if (len2 > FloatType(RS_EPS) * FloatType(RS_EPS))
            return n * FI::rs_divide_fast(FloatType(1), FI::rs_sqrt_fast(len2));
        return Vec3::zero();
    }

    /**
     * @brief Compute unnormalized face normal via cross product: (v1 - v0) x (v2 - v0).
     * Useful when we want to area weight the normal accumulation.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3
    unnormalizedNormal(const Vec3 &v0, const Vec3 &v1, const Vec3 &v2) {
        return (v1 - v0).cross(v2 - v0);
    }

    /**
     * @brief Compute AABB of a triangle.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Bound3
    bound(const Vec3 &v0, const Vec3 &v1, const Vec3 &v2) {
        return Bound3(v0.min(v1).min(v2), v0.max(v1).max(v2));
    }

    /**
     * @brief Uniform barycentric sampling.
     * Given two uniform random numbers r1, r2 in [0,1], returns a point uniformly
     * distributed on the triangle.
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3
    samplePoint(const Vec3 &v0, const Vec3 &v1, const Vec3 &v2, FloatType r1, FloatType r2) {
        FloatType sqrt_r1 = FI::rs_sqrt_fast(r1);
        FloatType u = FloatType(1) - sqrt_r1;
        FloatType v = r2 * sqrt_r1;
        return v0 * u + v1 * v + v2 * (FloatType(1) - u - v);
    }

    /**
     * @brief Distance from point p to the line segment [v0, v1].
     */
    HOST_DEVICE_INLINE_QUALIFIER static FloatType
    pointEdgeDistance(const Vec3 &p, const Vec3 &v0, const Vec3 &v1) {
        Vec3 e = v1 - v0;
        Vec3 r = p - v0;
        FloatType e_len2 = e * e;
        FloatType t = (e_len2 > FloatType(1e-20)) ? (r * e) / e_len2 : FloatType(0);
        if (t < FloatType(0)) t = FloatType(0);
        if (t > FloatType(1)) t = FloatType(1);
        Vec3 diff = p - (v0 + e * t);
        return FI::rs_sqrt(diff * diff);
    }

    /**
     * @brief Barycentric coordinates of point p in the plane of (v0, v1, v2).
     *        u/v/w are the weights for v0/v1/v2 (sum to 1 in the plane).
     *        For a degenerate triangle returns (1, 0, 0).
     */
    HOST_DEVICE_INLINE_QUALIFIER static void barycentric(
        const Vec3 &p,
        const Vec3 &v0,
        const Vec3 &v1,
        const Vec3 &v2,
        FloatType &u,
        FloatType &v,
        FloatType &w
    ) {
        Vec3 e1 = v1 - v0, e2 = v2 - v0, ep = p - v0;
        FloatType d00 = e1 * e1, d01 = e1 * e2, d11 = e2 * e2;
        FloatType d20 = ep * e1, d21 = ep * e2;
        FloatType denom = d00 * d11 - d01 * d01;
        if (denom < FloatType(1e-20)) {
            u = FloatType(1);
            v = FloatType(0);
            w = FloatType(0);
            return;
        }
        FloatType inv = FloatType(1) / denom;
        v = (d11 * d20 - d01 * d21) * inv;
        w = (d00 * d21 - d01 * d20) * inv;
        u = FloatType(1) - v - w;
    }

    /**
     * @brief Watertight ray-triangle intersection (Woop et al. 2013).
     *        https://jcgt.org/published/0002/01/05/paper.pdf
     *
     * Bit-exact edge-function negation between adjacent triangles eliminates the
     * shared-edge speckle that Moller-Trumbore produces. Steps:
     *   1) translate vertices into ray-local space (origin becomes implicit 0)
     *   2) permute axes so |dir[kz]| is the largest component (well-conditioned shear)
     *   3) shear the ray to (0,0,1); triangle becomes a 2D inside test
     *   4) three 2D edge functions (edge_u, edge_v, edge_w); all same sign => inside
     *
     * The inside test accepts both all-positive and all-negative edge functions, so
     * the intersection itself is winding-agnostic. Callers that need front-face
     * culling should do it externally via a normal-vs-ray-direction dot.
     *
     * @tparam UseDoubleEdgeTest
     *     false: edge functions in FloatType. Fastest; watertight along edges
     *            but vertex-meeting corners may produce rare single-pixel speckle.
     *     true:  edge functions + det + t numerator in double. Kills vertex speckle
     *            and sliver-triangle fireflies at cost of ~20 extra double flops.
     *
     * @param origin    Ray origin
     * @param dir       Ray direction (not necessarily normalized)
     * @param v0,v1,v2  Triangle vertices (CCW convention => outward-facing normal)
     * @param t_min     Minimum parametric distance
     * @param t_max     Maximum parametric distance
     * @param t_out     Parametric distance of intersection
     * @param u_out     Barycentric weight for v1 (beta)
     * @param v_out     Barycentric weight for v2 (gamma); weight for v0 is 1-u_out-v_out
     * @return true if intersection found within [t_min, t_max]
     */
    template <bool UseDoubleEdgeTest = false>
    HOST_DEVICE_INLINE_QUALIFIER static bool rayIntersect(
        const Vec3 &origin,
        const Vec3 &dir,
        const Vec3 &v0,
        const Vec3 &v1,
        const Vec3 &v2,
        FloatType t_min,
        FloatType t_max,
        FloatType &t_out,
        FloatType &u_out,
        FloatType &v_out
    ) {
        // Step 1: translate vertices into ray-local space (origin is implicit 0)
        Vec3 a = v0 - origin;
        Vec3 b = v1 - origin;
        Vec3 c = v2 - origin;

        // Step 2: pick kz = argmax(|dir.*|). kx, ky are the other two axes in
        // fixed cyclic order — no kx/ky winding-flip needed because Step 4 accepts
        // all-same-sign edge functions (signed sz below already handles dir sign).
        FloatType adx = dir.x < FloatType(0) ? -dir.x : dir.x;
        FloatType ady = dir.y < FloatType(0) ? -dir.y : dir.y;
        FloatType adz = dir.z < FloatType(0) ? -dir.z : dir.z;

        FloatType dkx, dky, dkz;
        FloatType ax, ay, az, bx, by, bz, cx, cy, cz;
        if (adx >= ady && adx >= adz) {
            // kz = x, kx = y, ky = z
            dkx = dir.y;
            dky = dir.z;
            dkz = dir.x;
            ax = a.y;
            ay = a.z;
            az = a.x;
            bx = b.y;
            by = b.z;
            bz = b.x;
            cx = c.y;
            cy = c.z;
            cz = c.x;
        } else if (ady >= adz) {
            // kz = y, kx = z, ky = x
            dkx = dir.z;
            dky = dir.x;
            dkz = dir.y;
            ax = a.z;
            ay = a.x;
            az = a.y;
            bx = b.z;
            by = b.x;
            bz = b.y;
            cx = c.z;
            cy = c.x;
            cz = c.y;
        } else {
            // kz = z, kx = x, ky = y
            dkx = dir.x;
            dky = dir.y;
            dkz = dir.z;
            ax = a.x;
            ay = a.y;
            az = a.z;
            bx = b.x;
            by = b.y;
            bz = b.z;
            cx = c.x;
            cy = c.y;
            cz = c.z;
        }

        if (dkz == FloatType(0)) return false;

        // Step 3: shear — sends ray direction to (0, 0, 1) in sheared space.
        // Signed sz: az*sz gives positive t for hits in front of the ray regardless
        // of dkz sign (t flips sign with az's sign, which itself flips with dkz).
        // Note: use IEEE divide (not fast): shear error compounds into every sheared
        // coord and then into the edge functions, which is exactly where precision matters.
        FloatType sx = -dkx / dkz;
        FloatType sy = -dky / dkz;
        FloatType sz = FloatType(1) / dkz;

        ax = ax + sx * az;
        ay = ay + sy * az;
        bx = bx + sx * bz;
        by = by + sy * bz;
        cx = cx + sx * cz;
        cy = cy + sy * cz;

        // Step 4: 2D edge functions. Bit-exact negated between adjacent triangles
        // on a shared edge (e.g., bx*ay - by*ax flips sign when a and b swap).
        if constexpr (UseDoubleEdgeTest) {
            double ax_d = double(ax), ay_d = double(ay);
            double bx_d = double(bx), by_d = double(by);
            double cx_d = double(cx), cy_d = double(cy);

            double edge_u = cx_d * by_d - cy_d * bx_d;
            double edge_v = ax_d * cy_d - ay_d * cx_d;
            double edge_w = bx_d * ay_d - by_d * ax_d;

            // Inside test: edge_u/v/w must all share a sign (zeros are allowed).
            if ((edge_u < 0.0 || edge_v < 0.0 || edge_w < 0.0) &&
                (edge_u > 0.0 || edge_v > 0.0 || edge_w > 0.0))
                return false;

            double det = edge_u + edge_v + edge_w;
            if (det == 0.0) return false;

            double az_d = double(az) * double(sz);
            double bz_d = double(bz) * double(sz);
            double cz_d = double(cz) * double(sz);
            double t_num = edge_u * az_d + edge_v * bz_d + edge_w * cz_d;

            double inv_det = 1.0 / det;
            double t = t_num * inv_det;
            if (t < double(t_min) || t > double(t_max)) return false;

            t_out = FloatType(t);
            u_out = FloatType(edge_v * inv_det);
            v_out = FloatType(edge_w * inv_det);
            return true;
        } else {
            FloatType edge_u = cx * by - cy * bx;
            FloatType edge_v = ax * cy - ay * cx;
            FloatType edge_w = bx * ay - by * ax;

            if ((edge_u < FloatType(0) || edge_v < FloatType(0) || edge_w < FloatType(0)) &&
                (edge_u > FloatType(0) || edge_v > FloatType(0) || edge_w > FloatType(0)))
                return false;

            FloatType det = edge_u + edge_v + edge_w;
            if (det == FloatType(0)) return false;

            FloatType az_s = az * sz;
            FloatType bz_s = bz * sz;
            FloatType cz_s = cz * sz;
            FloatType t_num = edge_u * az_s + edge_v * bz_s + edge_w * cz_s;

            FloatType inv_det = FI::rs_divide_fast(FloatType(1), det);
            FloatType t = t_num * inv_det;
            if (t < t_min || t > t_max) return false;

            t_out = t;
            u_out = edge_v * inv_det;
            v_out = edge_w * inv_det;
            return true;
        }
    }

    /**
     * @brief Vertex-plane deepest-penetration test (cheap discovery variant).
     * @details Projects each triangle's 3 vertices onto the OTHER triangle's
     * plane and picks the most-negative signed distance (deepest "behind face").
     * No edge clipping → ~6 dot products + comparisons; cheap enough for the
     * BVH-walk discovery pass. Misses penetrations where the deepest point is
     * along an edge interior, but for AGREED (near-coplanar anti-parallel) face
     * pairs the deepest contact lies at a vertex in practice.
     *
     * @param a0,a1,a2/b0,b1,b2  Triangle vertices
     * @param a_normal,b_normal   Outward face normals (normalized)
     * @param contact_point       Out: deepest vertex
     * @param contact_normal      Out: -face_normal of the penetrated face (B→A convention)
     * @param depth               Out: positive penetration depth
     * @return true if any vertex penetrates the other plane
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool testSimpleProximity(
        const Vec3 &a0,
        const Vec3 &a1,
        const Vec3 &a2,
        const Vec3 &b0,
        const Vec3 &b1,
        const Vec3 &b2,
        const Vec3 &a_normal,
        const Vec3 &b_normal,
        Vec3 &contact_point,
        Vec3 &contact_normal,
        FloatType &depth
    ) {
        FloatType best_sd = FloatType(0);
        Vec3 best_cp = Vec3::zero();
        Vec3 best_cn = Vec3::zero();

        // A's vertices vs plane_B
        FloatType sd = b_normal * (a0 - b0);
        if (sd < best_sd) {
            best_sd = sd;
            best_cp = a0;
            best_cn = -b_normal;
        }
        sd = b_normal * (a1 - b0);
        if (sd < best_sd) {
            best_sd = sd;
            best_cp = a1;
            best_cn = -b_normal;
        }
        sd = b_normal * (a2 - b0);
        if (sd < best_sd) {
            best_sd = sd;
            best_cp = a2;
            best_cn = -b_normal;
        }

        // B's vertices vs plane_A
        sd = a_normal * (b0 - a0);
        if (sd < best_sd) {
            best_sd = sd;
            best_cp = b0;
            best_cn = -a_normal;
        }
        sd = a_normal * (b1 - a0);
        if (sd < best_sd) {
            best_sd = sd;
            best_cp = b1;
            best_cn = -a_normal;
        }
        sd = a_normal * (b2 - a0);
        if (sd < best_sd) {
            best_sd = sd;
            best_cp = b2;
            best_cn = -a_normal;
        }

        if (best_sd >= FloatType(0)) return false;
        depth = -best_sd;
        contact_point = best_cp;
        contact_normal = best_cn;
        return true;
    }

    /**
     * @brief Triangle-triangle edge-face deepest-penetration test.
     * @note Intentionally using fast math for performance and not accuracy.
     * @details For each edge of one triangle, clips it against the other triangle's
     * "back prism" (the infinite volume behind the face, bounded by the triangle's edges).
     * Tests 3 edges of A against face B + 3 edges of B against face A. Picks the MIN
     * signed distance across all tests (the deepest penetration), reports `depth = -min_sd`:
     *   positive depth = penetrating
     *   negative depth = approaching (shallowest separation distance)
     *
     * @param a0,a1,a2        Triangle A vertices
     * @param b0,b1,b2        Triangle B vertices
     * @param a_normal         Pre-computed outward face normal of A (normalized)
     * @param b_normal         Pre-computed outward face normal of B (normalized)
     * @param contact_point    Output: deepest contact point in world space
     * @param contact_normal   Output: face normal of the deeper side (points from B to A
     *                                  when edge-of-A-vs-face-B is deeper; -a_normal when
     *                                  edge-of-B-vs-face-A is deeper)
     * @param depth            Output: positive = penetrating, negative = approaching
     * @return true if any edge projects inside the opposite triangle's prism
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool testProximity(
        const Vec3 &a0,
        const Vec3 &a1,
        const Vec3 &a2,
        const Vec3 &b0,
        const Vec3 &b1,
        const Vec3 &b2,
        const Vec3 &a_normal,
        const Vec3 &b_normal,
        Vec3 &contact_point,
        Vec3 &contact_normal,
        FloatType &depth
    ) {
        FloatType best_depth =
            FloatType(RS_MAX_FLOAT);  // track MIN signed distance (most negative)
        Vec3 best_point = Vec3::zero();
        Vec3 best_normal = Vec3::zero();
        bool found = false;

        // 3 edges of A against face B
        {
            Vec3 cp;
            FloatType d;
            if (_edgeToFaceTest(a0, a1, b0, b1, b2, b_normal, cp, d) && d < best_depth) {
                best_depth = d;
                best_point = cp;
                best_normal = b_normal;
                found = true;
            }
            if (_edgeToFaceTest(a1, a2, b0, b1, b2, b_normal, cp, d) && d < best_depth) {
                best_depth = d;
                best_point = cp;
                best_normal = b_normal;
                found = true;
            }
            if (_edgeToFaceTest(a2, a0, b0, b1, b2, b_normal, cp, d) && d < best_depth) {
                best_depth = d;
                best_point = cp;
                best_normal = b_normal;
                found = true;
            }
        }

        // 3 edges of B against face A
        {
            Vec3 cp;
            FloatType d;
            if (_edgeToFaceTest(b0, b1, a0, a1, a2, a_normal, cp, d) && d < best_depth) {
                best_depth = d;
                best_point = cp;
                best_normal = -a_normal;
                found = true;
            }
            if (_edgeToFaceTest(b1, b2, a0, a1, a2, a_normal, cp, d) && d < best_depth) {
                best_depth = d;
                best_point = cp;
                best_normal = -a_normal;
                found = true;
            }
            if (_edgeToFaceTest(b2, b0, a0, a1, a2, a_normal, cp, d) && d < best_depth) {
                best_depth = d;
                best_point = cp;
                best_normal = -a_normal;
                found = true;
            }
        }

        if (found) {
            depth = -best_depth;  // flip: positive = penetrating (behind face)
            contact_point = best_point;
            contact_normal = best_normal;
        }
        return found;
    }

    /**
     * @brief Triangle-triangle intersection test (Moller 1997).
     *
     * Exact test for whether two triangles intersect in 3D. When they do, returns the
     * endpoints of the intersection line segment (the two triangles' planes meet along
     * a line; the clipped overlap within both triangles is a segment).
     *
     * Coplanar / near-coplanar pairs (|n_A × n_B| < EPS) are treated as non-intersecting
     * — a simple conservative choice. Those should be handled by testProximity.
     *
     * @param a0,a1,a2     Triangle A vertices
     * @param b0,b1,b2     Triangle B vertices
     * @param a_normal      Outward face normal of A (normalized)
     * @param b_normal      Outward face normal of B (normalized)
     * @param seg_p0       Output: first endpoint of intersection segment
     * @param seg_p1       Output: second endpoint of intersection segment
     * @return true if triangles intersect
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool testTriTriIntersect(
        const Vec3 &a0,
        const Vec3 &a1,
        const Vec3 &a2,
        const Vec3 &b0,
        const Vec3 &b1,
        const Vec3 &b2,
        const Vec3 &a_normal,
        const Vec3 &b_normal,
        Vec3 &seg_p0,
        Vec3 &seg_p1
    ) {
        const FloatType eps = FloatType(RS_EPS);
        const FloatType eps2 = FloatType(RS_EPS2);

        // Plane separation early-out: if all three of B's vertices lie on one
        // side of plane_A (or vice versa), no intersection.
        FloatType db0 = a_normal * (b0 - a0);
        FloatType db1 = a_normal * (b1 - a0);
        FloatType db2 = a_normal * (b2 - a0);
        if ((db0 > eps && db1 > eps && db2 > eps) || (db0 < -eps && db1 < -eps && db2 < -eps))
            return false;

        FloatType da0 = b_normal * (a0 - b0);
        FloatType da1 = b_normal * (a1 - b0);
        FloatType da2 = b_normal * (a2 - b0);
        if ((da0 > eps && da1 > eps && da2 > eps) || (da0 < -eps && da1 < -eps && da2 < -eps))
            return false;

        // Plane-intersection line direction; coplanar pairs are dropped.
        Vec3 L_dir = a_normal.cross(b_normal);
        FloatType L_len2 = L_dir * L_dir;
        if (L_len2 < eps2) return false;

        // 1D parameterization on the dominant axis of L_dir (no need to normalize).
        FloatType abs_x = L_dir.x < FloatType(0) ? -L_dir.x : L_dir.x;
        FloatType abs_y = L_dir.y < FloatType(0) ? -L_dir.y : L_dir.y;
        FloatType abs_z = L_dir.z < FloatType(0) ? -L_dir.z : L_dir.z;

        Vec3 ap0, ap1, bp0, bp1;
        FloatType at0, at1, bt0, bt1;
        if (!_clipTriOnOtherPlane(
                a0, a1, a2, da0, da1, da2, abs_x, abs_y, abs_z, ap0, ap1, at0, at1
            ))
            return false;
        if (!_clipTriOnOtherPlane(
                b0, b1, b2, db0, db1, db2, abs_x, abs_y, abs_z, bp0, bp1, bt0, bt1
            ))
            return false;

        // 1D interval overlap on the dominant axis.
        FloatType t_lo = at0 > bt0 ? at0 : bt0;
        FloatType t_hi = at1 < bt1 ? at1 : bt1;
        if (t_lo > t_hi) return false;

        // Recover 3D endpoints by interpolating along A's clipped segment
        // (B's segment lies on the same plane-intersection line, gives same point).
        seg_p0 = _interpAlongSegment(ap0, ap1, at0, at1, t_lo);
        seg_p1 = _interpAlongSegment(ap0, ap1, at0, at1, t_hi);
        return true;
    }

   private:
    /**
     * @brief Möller tri-tri helper: find where two of a triangle's edges cross
     *        the other triangle's plane, project onto the dominant axis of the
     *        plane-intersection line, and return as ordered 1D interval + 3D ends.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _clipTriOnOtherPlane(
        const Vec3 &v0,
        const Vec3 &v1,
        const Vec3 &v2,
        FloatType d0,
        FloatType d1,
        FloatType d2,
        FloatType abs_x,
        FloatType abs_y,
        FloatType abs_z,
        Vec3 &out_p0,
        Vec3 &out_p1,
        FloatType &out_t0,
        FloatType &out_t1
    ) {
        Vec3 i0, i1;
        int n = 0;
        if (d0 * d1 < FloatType(0)) {
            FloatType t = FI::rs_divide_fast(d0, d0 - d1);
            Vec3 p = v0 + (v1 - v0) * t;
            if (n == 0)
                i0 = p;
            else
                i1 = p;
            ++n;
        }
        if (n < 2 && d1 * d2 < FloatType(0)) {
            FloatType t = FI::rs_divide_fast(d1, d1 - d2);
            Vec3 p = v1 + (v2 - v1) * t;
            if (n == 0)
                i0 = p;
            else
                i1 = p;
            ++n;
        }
        if (n < 2 && d2 * d0 < FloatType(0)) {
            FloatType t = FI::rs_divide_fast(d2, d2 - d0);
            Vec3 p = v2 + (v0 - v2) * t;
            if (n == 0)
                i0 = p;
            else
                i1 = p;
            ++n;
        }
        if (n < 2) return false;  // grazing — treat as no overlap

        FloatType p0, p1;
        if (abs_x >= abs_y && abs_x >= abs_z) {
            p0 = i0.x;
            p1 = i1.x;
        } else if (abs_y >= abs_z) {
            p0 = i0.y;
            p1 = i1.y;
        } else {
            p0 = i0.z;
            p1 = i1.z;
        }

        if (p0 <= p1) {
            out_t0 = p0;
            out_t1 = p1;
            out_p0 = i0;
            out_p1 = i1;
        } else {
            out_t0 = p1;
            out_t1 = p0;
            out_p0 = i1;
            out_p1 = i0;
        }
        return true;
    }

    /**
     * @brief Linear interpolation along a 3D segment [p0, p1] parameterized by [t0, t1].
     */
    HOST_DEVICE_INLINE_QUALIFIER static Vec3
    _interpAlongSegment(const Vec3 &p0, const Vec3 &p1, FloatType t0, FloatType t1, FloatType t) {
        FloatType denom = t1 - t0;
        if (denom == FloatType(0)) return p0;
        FloatType s = FI::rs_divide_fast(t - t0, denom);
        return p0 + (p1 - p0) * s;
    }

    /**
     * @brief Test an edge (p0→p1) against a triangle face's back-prism.
     * @details Finds the point on the edge whose projection onto the face plane falls
     * inside the triangle and has the most negative signed distance (deepest behind face).
     *
     * Algorithm:
     *   1. Compute signed distances of edge endpoints to face plane
     *   2. Project edge onto face plane and clip projected segment against triangle (3 half-plane
     * clips)
     *   3. Evaluate signed distance at clip boundaries (linear, so extremes are at boundaries)
     *
     * @param p0, p1        Edge endpoints
     * @param f0, f1, f2    Face triangle vertices
     * @param f_normal       Face outward normal (normalized)
     * @param contact_point  Output: midpoint of deepest edge point and its face projection
     * @param signed_dist    Output: signed distance of deepest point (negative = behind face)
     * @return true if any part of the edge projects inside the triangle
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _edgeToFaceTest(
        const Vec3 &p0,
        const Vec3 &p1,
        const Vec3 &f0,
        const Vec3 &f1,
        const Vec3 &f2,
        const Vec3 &f_normal,
        Vec3 &contact_point,
        FloatType &signed_dist
    ) {
        /*
         * Clip the edge p(t) = p_0 + t*(p_1 - p_0), t in [0,1] against the
         * triangle's 3 half-plane edges, finding the parametric range
         * [t_lo, t_hi] where p(t)'s face-plane projection lies inside.
         *
         * 1. Original formulation:
         *   Let d_i = n · (p_i - f_0)  (signed distance to face plane),
         *       p'_i = p_i - n * d_i   (projection onto face plane),
         *       p'(t) = p'_0 + t * (p'_1 - p'_0).
         *
         *   For triangle edge f_k -> f_{k+1}, the inside half-plane test is:
         *     ((f_{k+1} - f_k) x (p'(t) - f_k)) · n >= 0
         *
         *   We want to find the range of t in [0,1] where p'(t) is inside
         *   all 3 triangle edges simultaneously. For each edge k, "inside" means
         *   p'(t) is on the inward side of edge k:
         *     (p'(t) - f_k) · h_k >= 0
         *
         *   where h_k = n x (f_{k+1} - f_k)   -- inward edge normal on the face plane
         *
         *   Expanding p'(t) = p'_0 + t * (p'_1 - p'_0), the condition becomes:
         *     s_k + t * r_k >= 0
         *   where s_k = (p'_0 - f_k) · h_k     -- signed distance of p'_0 from edge k
         *         r_k = (p'_1 - p'_0) · h_k    -- rate of change of that distance along p'(t)
         *
         *   Solving for t (Cyrus-Beck clipping):
         *     r_k > 0 (moving inward):  t >= -s_k/r_k  -- entering edge k, raises t_lo
         *     r_k < 0 (moving outward): t <= -s_k/r_k  -- leaving edge k, lowers t_hi
         *     r_k = 0 (parallel):       inside iff s_k >= 0, else no solution
         *   Intersecting all 3 edges gives valid range [t_lo, t_hi].
         *   If t_lo > t_hi the edge misses the triangle entirely.
         *
         * 2. Register optimization -- eliminate p'_0, p'_1, p'_1 - p'_0 (9 floats):
         *   Since n · (n x e) = 0 for any e (dot with own cross product),
         *   the projection terms vanish in the dot products:
         *
         *   s_k = (p'_0 - f_k) · h_k
         *       = (p_0 - n*d_0 - f_k) · h_k
         *       = (p_0 - f_k) · h_k  -  d_0 * (n · h_k)
         *                                     ^^^^^^^^^ = 0
         *       = (p_0 - f_k) · h_k
         *
         *   r_k = (p'_1 - p'_0) · h_k
         *       = ((p_1 - p_0) - n*(d_1 - d_0)) · h_k
         *       = (p_1 - p_0) · h_k  -  (d_1 - d_0) * (n · h_k)
         *                                             ^^^^^^^^^ = 0
         *       = (p_1 - p_0) · h_k
         *
         *   So s and r depend only on original vertices -- no projections needed.
         *   d_0, d_1 are deferred to after clipping for depth/contact computation.
         *   Loop is unrolled to avoid pointer arrays spilling to local memory.
         */
        Vec3 edge_dir = p1 - p0;

        FloatType t_lo = FloatType(0);
        FloatType t_hi = FloatType(1);

        // Edge 0: f0 → f1
        {
            Vec3 h = f_normal.cross(f1 - f0);
            FloatType s = (p0 - f0) * h;
            FloatType r = edge_dir * h;
            if (r > FloatType(RS_EPS)) {
                FloatType t_enter = FI::rs_divide_fast(-s, r);
                if (t_enter > t_lo) t_lo = t_enter;
            } else if (r < -FloatType(RS_EPS)) {
                FloatType t_leave = FI::rs_divide_fast(-s, r);
                if (t_leave < t_hi) t_hi = t_leave;
            } else {
                if (s < -FloatType(RS_EPS)) return false;
            }
            if (t_lo > t_hi + FloatType(RS_EPS)) return false;
        }
        // Edge 1: f1 → f2
        {
            Vec3 h = f_normal.cross(f2 - f1);
            FloatType s = (p0 - f1) * h;
            FloatType r = edge_dir * h;
            if (r > FloatType(RS_EPS)) {
                FloatType t_enter = FI::rs_divide_fast(-s, r);
                if (t_enter > t_lo) t_lo = t_enter;
            } else if (r < -FloatType(RS_EPS)) {
                FloatType t_leave = FI::rs_divide_fast(-s, r);
                if (t_leave < t_hi) t_hi = t_leave;
            } else {
                if (s < -FloatType(RS_EPS)) return false;
            }
            if (t_lo > t_hi + FloatType(RS_EPS)) return false;
        }
        // Edge 2: f2 → f0
        {
            Vec3 h = f_normal.cross(f0 - f2);
            FloatType s = (p0 - f2) * h;
            FloatType r = edge_dir * h;
            if (r > FloatType(RS_EPS)) {
                FloatType t_enter = FI::rs_divide_fast(-s, r);
                if (t_enter > t_lo) t_lo = t_enter;
            } else if (r < -FloatType(RS_EPS)) {
                FloatType t_leave = FI::rs_divide_fast(-s, r);
                if (t_leave < t_hi) t_hi = t_leave;
            } else {
                if (s < -FloatType(RS_EPS)) return false;
            }
            if (t_lo > t_hi + FloatType(RS_EPS)) return false;
        }

        // Clipped range [t_lo, t_hi] is non-empty — compute depth and contact.
        // d0, d1 deferred to here so they don't consume registers during clipping.
        FloatType d0 = f_normal * (p0 - f0);
        FloatType d1 = f_normal * (p1 - f0);
        FloatType d_lo = d0 + t_lo * (d1 - d0);
        FloatType d_hi = d0 + t_hi * (d1 - d0);
        FloatType t_best = (d_lo <= d_hi) ? t_lo : t_hi;
        FloatType d_best = (d_lo <= d_hi) ? d_lo : d_hi;

        // Contact point: midpoint of edge point and its face projection
        Vec3 edge_point = p0 + edge_dir * t_best;
        Vec3 face_proj = edge_point - f_normal * d_best;
        contact_point = (edge_point + face_proj) * FloatType(0.5);
        signed_dist = d_best;
        return true;
    }
};

}  // namespace rise

#endif  // RS_TRIANGLE_H
