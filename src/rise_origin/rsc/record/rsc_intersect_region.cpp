// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/record/rsc_intersect_region.h"

#include "rise_origin/rsc/rs_rsc_archive.h"
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

void RSC_IntersectRegionCuboid::_read(const RS_RSCFileTreeReader& tree) {
    RSC_RecordBase::_read(tree);
}

void RSC_IntersectRegionCuboid::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_RecordBase::_write(tree);
}

void RSC_IntersectRegionCuboid::_deserialize(RS_RSCArchive& archive) {
    RSC_RecordBase::_deserialize(archive);
    archive.deserialize("position", position);
    archive.deserialize("orientation", orientation);
    archive.deserialize("size", size);
}

void RSC_IntersectRegionCuboid::_serialize(RS_RSCArchive& archive) const {
    RSC_RecordBase::_serialize(archive);
    archive.serialize("position", position);
    archive.serialize("orientation", orientation);
    archive.serialize("size", size);
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

void RSC_IntersectRegionCuboidRigidBodyLinked::_read(const RS_RSCFileTreeReader& tree) {
    RSC_RecordBase::_read(tree);
}

void RSC_IntersectRegionCuboidRigidBodyLinked::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_RecordBase::_write(tree);
}

void RSC_IntersectRegionCuboidRigidBodyLinked::_deserialize(RS_RSCArchive& archive) {
    RSC_RecordBase::_deserialize(archive);
    archive.deserialize("structure_name", structure_name);
    archive.deserialize("body_sid", body_sid);
    archive.deserialize("segment_bid", segment_bid);
    archive.deserialize("relative_anchor", relative_anchor);
    archive.deserialize("relative_orientation", relative_orientation);
    archive.deserialize("size", size);
}

void RSC_IntersectRegionCuboidRigidBodyLinked::_serialize(RS_RSCArchive& archive) const {
    RSC_RecordBase::_serialize(archive);
    archive.serialize("structure_name", structure_name);
    archive.serialize("body_sid", body_sid);
    archive.serialize("segment_bid", segment_bid);
    archive.serialize("relative_anchor", relative_anchor);
    archive.serialize("relative_orientation", relative_orientation);
    archive.serialize("size", size);
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

void RSC_IntersectRegionSphere::_read(const RS_RSCFileTreeReader& tree) {
    RSC_RecordBase::_read(tree);
}

void RSC_IntersectRegionSphere::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_RecordBase::_write(tree);
}

void RSC_IntersectRegionSphere::_deserialize(RS_RSCArchive& archive) {
    RSC_RecordBase::_deserialize(archive);
    archive.deserialize("position", position);
    archive.deserialize("radius", radius);
}

void RSC_IntersectRegionSphere::_serialize(RS_RSCArchive& archive) const {
    RSC_RecordBase::_serialize(archive);
    archive.serialize("position", position);
    archive.serialize("radius", radius);
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

void RSC_IntersectRegionSphereRigidBodyLinked::_read(const RS_RSCFileTreeReader& tree) {
    RSC_RecordBase::_read(tree);
}

void RSC_IntersectRegionSphereRigidBodyLinked::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_RecordBase::_write(tree);
}

void RSC_IntersectRegionSphereRigidBodyLinked::_deserialize(RS_RSCArchive& archive) {
    RSC_RecordBase::_deserialize(archive);
    archive.deserialize("structure_name", structure_name);
    archive.deserialize("body_sid", body_sid);
    archive.deserialize("segment_bid", segment_bid);
    archive.deserialize("relative_anchor", relative_anchor);
    archive.deserialize("radius", radius);
}

void RSC_IntersectRegionSphereRigidBodyLinked::_serialize(RS_RSCArchive& archive) const {
    RSC_RecordBase::_serialize(archive);
    archive.serialize("structure_name", structure_name);
    archive.serialize("body_sid", body_sid);
    archive.serialize("segment_bid", segment_bid);
    archive.serialize("relative_anchor", relative_anchor);
    archive.serialize("radius", radius);
}

}  // namespace rsc

}  // namespace rise
