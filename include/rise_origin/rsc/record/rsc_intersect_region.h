// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_INTERSECT_REGION_H
#define RSC_INTERSECT_REGION_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_record_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Record of which structures currently have voxels inside a world-fixed oriented box.
 */
class RSC_IntersectRegionCuboid : public RSC_RecordBase {
   public:
    /// Center of the box in world space.
    RVec3rf position = RVec3rf::zero();

    /// Orientation of the box in world space.
    RQuat3rf orientation = RQuat3rf::unit();

    /// Full extents of the box.
    RVec3rf size = RVec3rf::zero();

    bool operator==(const RSC_IntersectRegionCuboid& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Record of which structures intersect an oriented box rigidly linked to a rigid body.
 */
class RSC_IntersectRegionCuboidRigidBodyLinked : public RSC_RecordBase {
   public:
    /// The name of the structure that the region is linked to.
    std::string structure_name;

    /// The index of the body that the region is linked to.
    RIndex body_sid = RS_NULL_INDEX;

    /// The index of the segment that the region is linked to.
    RIndex segment_bid = RS_NULL_INDEX;

    /// The relative anchor point of the box in structure coordinate system.
    RVec3rf relative_anchor = RVec3rf::zero();

    /// The relative orientation of the box in structure coordinate system.
    RQuat3rf relative_orientation = RQuat3rf::unit();

    /// Full extents of the box.
    RVec3rf size = RVec3rf::zero();

    bool operator==(const RSC_IntersectRegionCuboidRigidBodyLinked& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Record of which structures currently have voxels inside a world-fixed sphere.
 */
class RSC_IntersectRegionSphere : public RSC_RecordBase {
   public:
    /// Center of the sphere in world space.
    RVec3rf position = RVec3rf::zero();

    /// Radius of the sphere.
    RFloat radius = 0;

    bool operator==(const RSC_IntersectRegionSphere& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Record of which structures intersect a sphere rigidly linked to a rigid body.
 */
class RSC_IntersectRegionSphereRigidBodyLinked : public RSC_RecordBase {
   public:
    /// The name of the structure that the region is linked to.
    std::string structure_name;

    /// The index of the body that the region is linked to.
    RIndex body_sid = RS_NULL_INDEX;

    /// The index of the segment that the region is linked to.
    RIndex segment_bid = RS_NULL_INDEX;

    /// The relative anchor point of the sphere center in structure coordinate system.
    RVec3rf relative_anchor = RVec3rf::zero();

    /// Radius of the sphere.
    RFloat radius = 0;

    bool operator==(const RSC_IntersectRegionSphereRigidBodyLinked& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_INTERSECT_REGION_H
