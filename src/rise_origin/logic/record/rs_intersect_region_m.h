// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_INTERSECT_REGION_M_H
#define RS_INTERSECT_REGION_M_H

#include <functional>
#include <refl.hpp>

#include "rise_origin/device/data/rs_bvh.h"
#include "rise_origin/utils/math/rs_bound.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

RS_ENUM(
    RSE_IntersectRegionType,
    unsigned int,
    (RSE_INTERSECT_REGION_CUBOID, 0),
    (RSE_INTERSECT_REGION_SPHERE, 1),
    (RSE_INTERSECT_REGION_CUBOID_RIGID_BODY_LINKED, 2),
    (RSE_INTERSECT_REGION_SPHERE_RIGID_BODY_LINKED, 3)
)

/**
 * @brief Source-of-truth region mem record data.
 */
struct RS_IntersectRegionMem {
    /**
     * Pre-set attributes
     */
    /// Global id of the owning record.
    RIndex gid = RS_NULL_INDEX;

    /// Region shape discriminator.
    RSE_IntersectRegionType type = RSE_INTERSECT_REGION_CUBOID;

    /// Linked rigid body structure scope id (linked type only).
    RIndex rigid_body_sid = RS_NULL_INDEX;

    /// Half extents (cuboid types).
    RVec3f half_size = RVec3f::zero();

    /// Radius (sphere type).
    float radius = 0.0f;

    /// Linked anchor position in rigid body space.
    RVec3f rigid_body_anchor_rb_position = RVec3f::zero();

    /// Linked anchor orientation in rigid body space.
    RQuat3f rigid_body_anchor_rb_orientation = RQuat3f::unit();

    /**
     * Post-set attributes
     */
    /// Linked rigid body handle; RS_NULL_HANDLE for world-fixed regions.
    RHandle rigid_body = RS_NULL_HANDLE;

    /// World-space center (cuboid / sphere); initial center for the linked type.
    RVec3f position = RVec3f::zero();

    /// World-space orientation (cuboid types).
    RQuat3f orientation = RQuat3f::unit();
};

/**
 * @brief Minimal region-BVH leaf payload, materialized from RS_IntersectRegionMem each record step.
 */
struct RS_IntersectRegionObjectIdentifier {
    /// Global id of the owning record (stable across BVH rebuilds).
    RIndex record_gid = RS_NULL_INDEX;

    /// Region shape discriminator.
    RSE_IntersectRegionType type = RSE_INTERSECT_REGION_CUBOID;

    /// Current world-space center.
    RVec3f position = RVec3f::zero();

    /// Current world-space orientation (cuboid types).
    RQuat3f orientation = RQuat3f::unit();

    /// Half extents (cuboid types).
    RVec3f half_size = RVec3f::zero();

    /// Radius (sphere type).
    float radius = 0.0f;

    /// True iff the world point lies inside the region.
    HOST_DEVICE_INLINE_QUALIFIER bool contains(const RVec3f& point) const {
        if (type == RSE_INTERSECT_REGION_SPHERE ||
            type == RSE_INTERSECT_REGION_SPHERE_RIGID_BODY_LINKED) {
            return (point - position).length2() <= radius * radius;
        }
        RVec3f local = orientation.rotateVec3DInv(point - position).abs();
        return local.x <= half_size.x && local.y <= half_size.y && local.z <= half_size.z;
    }

    /// World-space axis-aligned bound enclosing the region.
    HOST_DEVICE_INLINE_QUALIFIER RBound3f worldBound() const {
        if (type == RSE_INTERSECT_REGION_SPHERE ||
            type == RSE_INTERSECT_REGION_SPHERE_RIGID_BODY_LINKED) {
            RVec3f r(radius, radius, radius);
            return RBound3f(position - r, position + r);
        }
        RVec3f he = orientation.rotateVec3D(RVec3f(half_size.x, 0, 0)).abs() +
                    orientation.rotateVec3D(RVec3f(0, half_size.y, 0)).abs() +
                    orientation.rotateVec3D(RVec3f(0, 0, half_size.z)).abs();
        return RBound3f(position - he, position + he);
    }

    bool operator==(const RS_IntersectRegionObjectIdentifier& other) const {
        return record_gid == other.record_gid;
    }
};

struct RS_IntersectRegionObjectIdentifierHostHasher {
    std::size_t operator()(const RS_IntersectRegionObjectIdentifier& identifier) const {
        return std::hash<RIndex>()(identifier.record_gid);
    }
};

using RS_IntersectRegionBVH = device::RS_BVHBase<
    RBigIndex,
    RS_IntersectRegionObjectIdentifier,
    RS_IntersectRegionObjectIdentifierHostHasher>;

/**
 * @brief One emitted membership fact: structure `structure_gid` has a voxel inside record
 * `record_gid`. Packed record-gid-major so a sorted run of pairs is contiguous per region.
 */
struct RS_IntersectRegionPair {
    RIndex record_gid = RS_NULL_INDEX;
    RIndex structure_gid = RS_NULL_INDEX;

    /// The voxel center minus the region center, in world coords.
    RVec3f closest_relative_offset = RVec3f::zero();

    HOST_DEVICE_INLINE_QUALIFIER RBigIndex key() const {
        return (static_cast<RBigIndex>(record_gid) << 32) | static_cast<RBigIndex>(structure_gid);
    }
};

/**
 * @brief Composite sort key: groups pairs by region and structure, then orders each group by
 * distance so the closest voxel to the region center sorts first.
 */
struct RS_IntersectRegionSortKey {
    RBigIndex group_key = 0;
    float distance_squared = 0.0f;
};

struct RS_IntersectRegionSortKeyLess {
    HOST_DEVICE_INLINE_QUALIFIER bool
    operator()(const RS_IntersectRegionSortKey& a, const RS_IntersectRegionSortKey& b) const {
        if (a.group_key != b.group_key) return a.group_key < b.group_key;
        return a.distance_squared < b.distance_squared;
    }
};

}  // namespace logic

}  // namespace rise

REFL_AUTO(
    type(rise::logic::RS_IntersectRegionMem),
    field(gid),
    field(type),
    field(rigid_body_sid),
    field(half_size),
    field(radius),
    field(rigid_body_anchor_rb_position),
    field(rigid_body_anchor_rb_orientation),
    field(rigid_body),
    field(position),
    field(orientation)
)

#endif  // RS_INTERSECT_REGION_M_H
