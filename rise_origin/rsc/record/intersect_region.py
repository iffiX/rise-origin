# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Intersect-region records: per-step structure-membership queries.

Each region reports, every record step, which structures currently have at
least one voxel inside it. Membership is instantaneous and stateless: the
current ``structure_gid`` set is emitted every step, leaving enter/exit diffing
to the consumer. A voxel counts as inside when its center lies within the
region.
"""

from typing import Optional

from rise_origin import (
    RSC_IntersectRegionCuboid,
    RSC_IntersectRegionCuboidRigidBodyLinked,
    RSC_IntersectRegionSphere,
    RSC_IntersectRegionSphereRigidBodyLinked,
)

from rise_origin import RQuat3rf, RVec3rf

__all__ = [
    "IntersectRegionCuboid",
    "IntersectRegionCuboidRigidBodyLinked",
    "IntersectRegionSphere",
    "IntersectRegionSphereRigidBodyLinked",
]


class IntersectRegionCuboid(RSC_IntersectRegionCuboid):
    """A world-fixed oriented box that reports the structures inside it.

    The box is centered at ``position``, rotated by ``orientation``, and spans
    ``size`` (full edge lengths). The region is static for the whole run.

    Args:
        name: Unique record name.
        position: Box center ``(x, y, z)`` in world meters.
        orientation: Box orientation quaternion ``(x, y, z, w)``.
        size: Full box extents ``(x, y, z)`` in meters (not half-extents).
        target_controller_reference: Name of the controller that receives this
            record through the world callback. Empty disables controller
            delivery.
        target_recorder_reference: Name of the recorder that persists this
            record. Empty disables recorder delivery.
    """

    def __init__(
        self,
        name: str,
        *,
        position: RVec3rf = RVec3rf.zero(),
        orientation: RQuat3rf = RQuat3rf.unit(),
        size: RVec3rf = RVec3rf.one(),
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.position = position
        self.orientation = orientation
        self.size = size
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference


class IntersectRegionCuboidRigidBodyLinked(RSC_IntersectRegionCuboidRigidBodyLinked):
    """An oriented box that rides a structure's rigid body.

    Identical in semantics to :class:`IntersectRegionCuboid`, except the box
    pose is anchored to a rigid body of ``structure_name`` and recomputed each
    record step from that body's live pose. The anchor is given in the body's
    local frame.

    Args:
        name: Unique record name.
        structure_name: Name of the structure whose rigid body the box follows.
        body_sid: Body sub-id within the structure the box is attached to.
        segment_bid: Segment id of that body the box is attached to.
        relative_anchor: Box center in the body's local frame, ``(x, y, z)`` in
            meters.
        relative_orientation: Box orientation in the body's local frame,
            ``(x, y, z, w)``.
        size: Full box extents ``(x, y, z)`` in meters (not half-extents).
        target_controller_reference: Name of the controller that receives this
            record through the world callback. Empty disables controller
            delivery.
        target_recorder_reference: Name of the recorder that persists this
            record. Empty disables recorder delivery.
    """

    def __init__(
        self,
        name: str,
        *,
        structure_name: str = "",
        body_sid: int = 0,
        segment_bid: int = 0,
        relative_anchor: RVec3rf = RVec3rf.zero(),
        relative_orientation: RQuat3rf = RQuat3rf.unit(),
        size: RVec3rf = RVec3rf.one(),
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.structure_name = structure_name
        self.body_sid = body_sid
        self.segment_bid = segment_bid
        self.relative_anchor = relative_anchor
        self.relative_orientation = relative_orientation
        self.size = size
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference


class IntersectRegionSphere(RSC_IntersectRegionSphere):
    """A world-fixed sphere that reports the structures inside it.

    A voxel is counted as inside when its center is within ``radius`` of
    ``position``. The region is static for the whole run.

    Args:
        name: Unique record name.
        position: Sphere center ``(x, y, z)`` in world meters.
        radius: Sphere radius in meters.
        target_controller_reference: Name of the controller that receives this
            record through the world callback. Empty disables controller
            delivery.
        target_recorder_reference: Name of the recorder that persists this
            record. Empty disables recorder delivery.
    """

    def __init__(
        self,
        name: str,
        *,
        position: RVec3rf = RVec3rf.zero(),
        radius: float = 1.0,
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.position = position
        self.radius = radius
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference


class IntersectRegionSphereRigidBodyLinked(RSC_IntersectRegionSphereRigidBodyLinked):
    """A sphere that rides a structure's rigid body.

    Identical in semantics to :class:`IntersectRegionSphere`, except the sphere
    center is anchored to a rigid body of ``structure_name`` and recomputed each
    record step from that body's live pose. A sphere is rotation invariant, so
    only the anchor position rides the body.

    Args:
        name: Unique record name.
        structure_name: Name of the structure whose rigid body the sphere
            follows.
        body_sid: Body sub-id within the structure the sphere is attached to.
        segment_bid: Segment id of that body the sphere is attached to.
        relative_anchor: Sphere center in the body's local frame, ``(x, y, z)``
            in meters.
        radius: Sphere radius in meters.
        target_controller_reference: Name of the controller that receives this
            record through the world callback. Empty disables controller
            delivery.
        target_recorder_reference: Name of the recorder that persists this
            record. Empty disables recorder delivery.
    """

    def __init__(
        self,
        name: str,
        *,
        structure_name: str = "",
        body_sid: int = 0,
        segment_bid: int = 0,
        relative_anchor: RVec3rf = RVec3rf.zero(),
        radius: float = 1.0,
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.structure_name = structure_name
        self.body_sid = body_sid
        self.segment_bid = segment_bid
        self.relative_anchor = relative_anchor
        self.radius = radius
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference
