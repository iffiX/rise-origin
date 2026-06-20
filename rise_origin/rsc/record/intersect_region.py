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
)

from rise_origin.rsc._utils import Quat3rf, Vec3rf, _quat3rf, _vec3rf
from rise_origin.rsc.entity.structure_voxel import BodyRef, StructureVoxelBody

__all__ = [
    "IntersectRegionCuboid",
    "IntersectRegionCuboidRigidBodyLinked",
    "IntersectRegionSphere",
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
        position: Vec3rf = (0.0, 0.0, 0.0),
        orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
        size: Vec3rf = (1.0, 1.0, 1.0),
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.position = _vec3rf(position)
        self.orientation = _quat3rf(orientation)
        self.size = _vec3rf(size)
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
        attached_to: Body object (or ``(body, segment_bid)`` tuple) the box
            follows; resolves the ids below. The body must already be added to
            its structure. ``None`` uses the explicit ids.
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
        attached_to: Optional[BodyRef] = None,
        structure_name: str = "",
        body_sid: int = 0,
        segment_bid: int = 0,
        relative_anchor: Vec3rf = (0.0, 0.0, 0.0),
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
        size: Vec3rf = (1.0, 1.0, 1.0),
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        if attached_to is not None:
            structure_name, body_sid, segment_bid = (
                StructureVoxelBody._resolve_body_link(attached_to)
            )
        self.name = name
        self.structure_name = structure_name
        self.body_sid = body_sid
        self.segment_bid = segment_bid
        self.relative_anchor = _vec3rf(relative_anchor)
        self.relative_orientation = _quat3rf(relative_orientation)
        self.size = _vec3rf(size)
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
        position: Vec3rf = (0.0, 0.0, 0.0),
        radius: float = 1.0,
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.position = _vec3rf(position)
        self.radius = radius
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference
