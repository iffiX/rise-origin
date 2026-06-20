// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONTACT_M_H
#define RS_CONTACT_M_H

#include <cstdint>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

namespace logic {

union RS_RigidContactStateUnion {
    RState contact_state_packed;
    struct RS_RigidContactState {
        RState is_active : 1;
        RState slot : 2;
    } contact_state;
};

/// Packed owner pair sort key, high word holds the smaller rigid body handle.
using RS_RigidContactKey = uint64_t;

/**
 * @brief One reduced rigid contact, a transient relationship record between two rigid bodies.
 *
 * @details Holds a back reference to the source voxel and slot so the warm start lambda can be
 * seeded and written back, plus an in place copy of the contact geometry so the iterative solve
 * reads coalesced. The owners are rigid bodies; rigid_body_b is RS_NULL_HANDLE for a body versus
 * floor contact, disambiguated by slot 2. The lambda accumulators live here only for one step's
 * solve, their persistent home is the source voxel.
 */
template <RSE_FloatType FloatMemAccuracy>
class RS_RigidContactMem {
   public:
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;
    using Vec3M = RS_Vec3DMem<FloatMemAccuracy>;

    /// Source voxel for warm start seed and write back
    RHandle voxel_a = RS_NULL_HANDLE;

    /// Partner voxel, RS_NULL_HANDLE for floor
    RHandle voxel_b = RS_NULL_HANDLE;

    /// Rigid body owner a
    RHandle rigid_body_a = RS_NULL_HANDLE;

    /// Rigid body owner b, RS_NULL_HANDLE for floor
    RHandle rigid_body_b = RS_NULL_HANDLE;

    /// Packed is_active and slot
    RState state = 0;

    /// Contact point in global coords
    Vec3M contact_point = Vec3M::zero();

    /// Contact normal in global coords, unit, pointing from b toward a
    Vec3M contact_normal = Vec3M::zero();

    /// Penetration depth, positive when overlapping
    FloatM depth = 0;

    /// Normal generalized inverse mass times the count normalizer
    FloatM k_n = 0;

    /// Count normalizer, the larger per body contact count
    FloatM normalizer = 1;

    /// Pair friction coefficient
    FloatM mu = 0;

    /// Accumulated normal lambda for this step
    FloatM accumulated_lambda_n = 0;

    /// Accumulated tangent friction lambda for this step
    Vec3M accumulated_lambda_t = Vec3M::zero();
};

/**
 * @brief Reduce element holding the deepest contact of an owner-pair group, the manifold anchor.
 * @details Carries the contact point so the farthest-point pass reads it without a scattered fetch.
 */
struct RS_RigidManifoldDeepest {
    float depth;
    RVec3f point;
    RIndex index;

    HOST_DEVICE_INLINE_QUALIFIER static RS_RigidManifoldDeepest empty() {
        return {float(-1e30), RVec3f::zero(), RS_NULL_INDEX};
    }
};

/// @brief Reduce element holding the three contacts farthest from the group's deepest, descending.
struct RS_RigidManifoldFarthest {
    float dist[3];
    RIndex index[3];

    HOST_DEVICE_INLINE_QUALIFIER static RS_RigidManifoldFarthest empty() {
        return {{-1.f, -1.f, -1.f}, {RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX}};
    }
};

/// @brief Reduce operator: keep the deeper of two contacts per group.
struct RS_ReduceRigidManifoldDeepestOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) const {
        return a.depth >= b.depth ? a : b;
    }
};

/// @brief Reduce operator: merge two descending top-three lists into the top three by distance.
struct RS_ReduceRigidManifoldFarthestOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) const {
        T r = T::empty();
        int ia = 0, ib = 0;
        for (int s = 0; s < 3; ++s) {
            float da = ia < 3 ? a.dist[ia] : -1.f;
            float db = ib < 3 ? b.dist[ib] : -1.f;
            if (da < 0.f && db < 0.f) break;
            if (da >= db) {
                r.dist[s] = da;
                r.index[s] = a.index[ia];
                ++ia;
            } else {
                r.dist[s] = db;
                r.index[s] = b.index[ib];
                ++ib;
            }
        }
        return r;
    }
};

}  // namespace logic

}  // namespace rise

#define REFLECT_RIGID_CONTACT(contact_mem_type) \
    REFL_AUTO(                                  \
        type(contact_mem_type),                 \
        field(voxel_a),                         \
        field(voxel_b),                         \
        field(rigid_body_a),                    \
        field(rigid_body_b),                    \
        field(state),                           \
        protected_field(contact_point),         \
        protected_field(contact_normal),        \
        field(depth),                           \
        field(k_n),                             \
        field(normalizer),                      \
        field(mu),                              \
        field(accumulated_lambda_n),            \
        protected_field(accumulated_lambda_t)   \
    )

REFLECT_RIGID_CONTACT(rise::logic::RS_RigidContactMem<rise::RSE_FLOAT_32>)
REFLECT_RIGID_CONTACT(rise::logic::RS_RigidContactMem<rise::RSE_FLOAT_64>)

#endif  // RS_CONTACT_M_H
