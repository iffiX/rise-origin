// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/record/rsc_intersect_region.h"

#include "rise_origin/rsc/rs_rsc_validator.h"

namespace rise {

namespace rsc {

bool RSC_IntersectRegionCuboid::operator==(const RSC_IntersectRegionCuboid& other) const {
    return RSC_RecordBase::operator==(other) && position == other.position &&
           orientation == other.orientation && size == other.size;
}

void RSC_IntersectRegionCuboid::_validate(RS_RSCValidator& validator) const {
    RSC_RecordBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_IntersectRegionCuboid::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_IntersectRegionCuboid, size.x, 0);
        RSC_EXPECT_GT(RSC_IntersectRegionCuboid, size.y, 0);
        RSC_EXPECT_GT(RSC_IntersectRegionCuboid, size.z, 0);
    });
}

void RSC_IntersectRegionCuboid::_read(const RS_RSCPropertyTreeReader& tree) {
    RSC_RecordBase::_read(tree);
}

void RSC_IntersectRegionCuboid::_write(RS_RSCPropertyTreeWriter& tree) const {
    RSC_RecordBase::_write(tree);
}

bool RSC_IntersectRegionCuboidRigidBodyLinked::operator==(
    const RSC_IntersectRegionCuboidRigidBodyLinked& other
) const {
    return RSC_RecordBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid &&
           relative_anchor == other.relative_anchor &&
           relative_orientation == other.relative_orientation && size == other.size;
}

void RSC_IntersectRegionCuboidRigidBodyLinked::_validate(RS_RSCValidator& validator) const {
    RSC_RecordBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_IntersectRegionCuboidRigidBodyLinked::_validate", this, [this] {
            RSC_EXPECT_NOT_NULL(RSC_IntersectRegionCuboidRigidBodyLinked, body_sid, RS_NULL_INDEX);
            RSC_EXPECT_GT(RSC_IntersectRegionCuboidRigidBodyLinked, size.x, 0);
            RSC_EXPECT_GT(RSC_IntersectRegionCuboidRigidBodyLinked, size.y, 0);
            RSC_EXPECT_GT(RSC_IntersectRegionCuboidRigidBodyLinked, size.z, 0);
        }
    );
}

void RSC_IntersectRegionCuboidRigidBodyLinked::_read(const RS_RSCPropertyTreeReader& tree) {
    RSC_RecordBase::_read(tree);
}

void RSC_IntersectRegionCuboidRigidBodyLinked::_write(RS_RSCPropertyTreeWriter& tree) const {
    RSC_RecordBase::_write(tree);
}

bool RSC_IntersectRegionSphere::operator==(const RSC_IntersectRegionSphere& other) const {
    return RSC_RecordBase::operator==(other) && position == other.position &&
           radius == other.radius;
}

void RSC_IntersectRegionSphere::_validate(RS_RSCValidator& validator) const {
    RSC_RecordBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_IntersectRegionSphere::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_IntersectRegionSphere, radius, 0);
    });
}

void RSC_IntersectRegionSphere::_read(const RS_RSCPropertyTreeReader& tree) {
    RSC_RecordBase::_read(tree);
}

void RSC_IntersectRegionSphere::_write(RS_RSCPropertyTreeWriter& tree) const {
    RSC_RecordBase::_write(tree);
}

bool RSC_IntersectRegionSphereRigidBodyLinked::operator==(
    const RSC_IntersectRegionSphereRigidBodyLinked& other
) const {
    return RSC_RecordBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid &&
           relative_anchor == other.relative_anchor && radius == other.radius;
}

void RSC_IntersectRegionSphereRigidBodyLinked::_validate(RS_RSCValidator& validator) const {
    RSC_RecordBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_IntersectRegionSphereRigidBodyLinked::_validate", this, [this] {
            RSC_EXPECT_NOT_NULL(RSC_IntersectRegionSphereRigidBodyLinked, body_sid, RS_NULL_INDEX);
            RSC_EXPECT_GT(RSC_IntersectRegionSphereRigidBodyLinked, radius, 0);
        }
    );
}

void RSC_IntersectRegionSphereRigidBodyLinked::_read(const RS_RSCPropertyTreeReader& tree) {
    RSC_RecordBase::_read(tree);
}

void RSC_IntersectRegionSphereRigidBodyLinked::_write(RS_RSCPropertyTreeWriter& tree) const {
    RSC_RecordBase::_write(tree);
}

}  // namespace rsc

}  // namespace rise
