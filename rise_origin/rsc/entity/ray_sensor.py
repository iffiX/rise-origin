# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Ray-sensor entities and their embedded record/signal.

A ray sensor casts a fixed set of rays from its origin and reports the closest
hit per ray. It carries an embedded record (which per-ray fields to capture and
how to route them) and an embedded signal (how many controller-driven pose
signals it reads). Two placements exist: a free sensor at an absolute pose, and
one rigidly linked to a structure's rigid body. Ray directions are given
relative to the sensor origin and are normalized when committed.
"""

from typing import List, Optional

from rise_origin import (
    RSC_RaySensorFree,
    RSC_RaySensorRecord,
    RSC_RaySensorRigidBodyLinked,
    RSC_RaySensorSignal,
)

from rise_origin.rsc._utils import Quat3rf, Vec3rf

__all__ = [
    "RaySensorRecord",
    "RaySensorSignal",
    "RaySensorFree",
    "RaySensorRigidBodyLinked",
]


def _apply_ray_sensor(
    sensor,
    ray_directions: "List[Vec3rf]",
    max_range: float,
    record: "Optional[RaySensorRecord]",
    signal: "Optional[RaySensorSignal]",
) -> None:
    """Set the ray directions, range, and optional embedded record/signal."""
    sensor.ray_directions = list(ray_directions)
    sensor.max_range = max_range
    if record is not None:
        sensor.record = record
    if signal is not None:
        sensor.signal = signal


class RaySensorRecord(RSC_RaySensorRecord):
    """Selects which per-ray fields to capture and how to route them.

    Args:
        name: Record name (unique within the sensor).
        has_hit: Whether to capture the hit flag, distance, and hit position.
        has_entity_info: Whether to capture the hit entity global id and type.
        has_entity_detail: Whether to capture the hit body, segment, and voxel id.
        has_material: Whether to capture the hit material id and emission.
        has_normal: Whether to capture the world-space hit normal.
        has_velocity: Whether to capture the hit velocity, relative velocity, and
            radial velocity.
        target_controller_reference: Controller that receives this record via the
            world callback. Empty disables controller delivery.
        target_recorder_reference: Recorder that persists this record. Empty
            disables recorder delivery.
    """

    def __init__(
        self,
        name: str = "",
        *,
        has_hit: bool = True,
        has_entity_info: bool = False,
        has_entity_detail: bool = False,
        has_material: bool = False,
        has_normal: bool = False,
        has_velocity: bool = False,
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.has_hit = has_hit
        self.has_entity_info = has_entity_info
        self.has_entity_detail = has_entity_detail
        self.has_material = has_material
        self.has_normal = has_normal
        self.has_velocity = has_velocity
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference


class RaySensorSignal(RSC_RaySensorSignal):
    """Declares how many controller-driven pose signals the sensor reads.

    Args:
        name: Signal name (unique within the sensor).
        source_controller_reference: Controller that drives these signals each
            callback step.
        position_signal_num: Number of position signals.
        orientation_signal_num: Number of orientation signals.
    """

    def __init__(
        self,
        name: str = "",
        *,
        source_controller_reference: str = "",
        position_signal_num: int = 1,
        orientation_signal_num: int = 1,
    ) -> None:
        super().__init__()
        self.name = name
        self.source_controller_reference = source_controller_reference
        self.position_signal_num = position_signal_num
        self.orientation_signal_num = orientation_signal_num


class RaySensorFree(RSC_RaySensorFree):
    """A ray sensor placed at an absolute world pose.

    Args:
        name: Unique entity name.
        ray_directions: Ray directions ``(x, y, z)`` relative to the sensor
            origin. Normalized when committed.
        origin_position: World position ``(x, y, z)`` in meters.
        orientation: Orientation quaternion ``(x, y, z, w)``.
        max_range: Maximum ray distance in meters.
        record: Embedded output record. ``None`` keeps the default record.
        signal: Embedded signal declaration. ``None`` keeps the default signal.
    """

    def __init__(
        self,
        name: str = "ray_sensor",
        *,
        ray_directions: List[Vec3rf] = ((1.0, 0.0, 0.0),),
        origin_position: Vec3rf = (0.0, 0.0, 0.0),
        orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
        max_range: float = 100.0,
        record: Optional[RaySensorRecord] = None,
        signal: Optional[RaySensorSignal] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.origin_position = origin_position
        self.orientation = orientation
        _apply_ray_sensor(self, ray_directions, max_range, record, signal)


class RaySensorRigidBodyLinked(RSC_RaySensorRigidBodyLinked):
    """A ray sensor rigidly attached to a structure's rigid body.

    The sensor rides a structure's rigid body segment, offset by a body-relative
    anchor and orientation, so it moves with that body. Identify the target with
    ``structure_name`` / ``body_sid`` / ``segment_bid``.

    Args:
        name: Unique entity name.
        ray_directions: Ray directions ``(x, y, z)`` relative to the sensor
            origin. Normalized when committed.
        structure_name: Name of the structure to attach to.
        body_sid: Body sub-id within the structure the sensor rides.
        segment_bid: Segment id of that body the sensor rides.
        relative_anchor: Body-relative mount position ``(x, y, z)`` in meters.
        relative_orientation: Body-relative orientation quaternion
            ``(x, y, z, w)``.
        max_range: Maximum ray distance in meters.
        record: Embedded output record. ``None`` keeps the default record.
        signal: Embedded signal declaration. ``None`` keeps the default signal.
    """

    def __init__(
        self,
        name: str = "ray_sensor",
        *,
        ray_directions: List[Vec3rf] = ((1.0, 0.0, 0.0),),
        structure_name: str = "",
        body_sid: int = 0,
        segment_bid: int = 0,
        relative_anchor: Vec3rf = (0.0, 0.0, 0.0),
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
        max_range: float = 100.0,
        record: Optional[RaySensorRecord] = None,
        signal: Optional[RaySensorSignal] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.structure_name = structure_name
        self.body_sid = body_sid
        self.segment_bid = segment_bid
        self.relative_anchor = relative_anchor
        self.relative_orientation = relative_orientation
        _apply_ray_sensor(self, ray_directions, max_range, record, signal)
