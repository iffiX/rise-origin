// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/record/rsc_intersect_region.h"

namespace rise {

namespace rsc {

bool RSC_IntersectRegionCuboid::operator==(const RSC_IntersectRegionCuboid& other) const {
    return RSC_RecordBase::operator==(other) && position == other.position &&
           orientation == other.orientation && size == other.size;
}

void RSC_IntersectRegionCuboid::_validate() const {
    RSC_RecordBase::_validate();
    RSC_EXPECT_GT(RSC_IntersectRegionCuboid, size.x, 0);
    RSC_EXPECT_GT(RSC_IntersectRegionCuboid, size.y, 0);
    RSC_EXPECT_GT(RSC_IntersectRegionCuboid, size.z, 0);
}

void RSC_IntersectRegionCuboid::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_RecordBase::_read(rsc_tree);
}

bool RSC_IntersectRegionCuboidRigidBodyLinked::operator==(
    const RSC_IntersectRegionCuboidRigidBodyLinked& other
) const {
    return RSC_RecordBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid &&
           relative_anchor == other.relative_anchor &&
           relative_orientation == other.relative_orientation && size == other.size;
}

void RSC_IntersectRegionCuboidRigidBodyLinked::_validate() const {
    RSC_RecordBase::_validate();
    RSC_EXPECT_NOT_NULL(RSC_IntersectRegionCuboidRigidBodyLinked, body_sid, RS_NULL_INDEX);
    RSC_EXPECT_GT(RSC_IntersectRegionCuboidRigidBodyLinked, size.x, 0);
    RSC_EXPECT_GT(RSC_IntersectRegionCuboidRigidBodyLinked, size.y, 0);
    RSC_EXPECT_GT(RSC_IntersectRegionCuboidRigidBodyLinked, size.z, 0);
}

void RSC_IntersectRegionCuboidRigidBodyLinked::_read(
    const RS_CharacterizationPropertyTree& rsc_tree
) {
    RSC_RecordBase::_read(rsc_tree);
}

bool RSC_IntersectRegionSphere::operator==(const RSC_IntersectRegionSphere& other) const {
    return RSC_RecordBase::operator==(other) && position == other.position &&
           radius == other.radius;
}

void RSC_IntersectRegionSphere::_validate() const {
    RSC_RecordBase::_validate();
    RSC_EXPECT_GT(RSC_IntersectRegionSphere, radius, 0);
}

void RSC_IntersectRegionSphere::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_RecordBase::_read(rsc_tree);
}

}  // namespace rsc

}  // namespace rise
