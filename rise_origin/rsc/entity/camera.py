# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Perspective-camera entities and their embedded record/signal.

A camera renders the world by path tracing. It carries an embedded record (which
output channels to capture and how to route them) and an embedded signal (how
many controller-driven pose/optics signals it reads). Two placements exist: a
free camera at an absolute pose, and one rigidly linked to a structure's rigid
body. The camera's local forward axis is +X and local up is +Z.

Ray-tracing cost scales with ``image_width * image_height * samples_per_pixel``
and with ``max_depth`` (ray bounces); keep them modest for on-board sensors.
"""

from typing import Optional

from rise_origin import (
    RSC_CameraPerspectiveFree,
    RSC_CameraPerspectiveRecord,
    RSC_CameraPerspectiveRigidBodyLinked,
    RSC_CameraPerspectiveSignal,
)

from rise_origin.rsc._utils import Quat3rf, Vec3rf, _quat3rf, _vec3rf
from rise_origin.rsc.entity.structure_voxel import BodyRef, StructureVoxelBody

__all__ = [
    "CameraPerspectiveRecord",
    "CameraPerspectiveSignal",
    "CameraPerspectiveFree",
    "CameraPerspectiveRigidBodyLinked",
]


def _apply_perspective(
    camera,
    image_width: int,
    image_height: int,
    samples_per_pixel: int,
    max_depth: int,
    field_of_view: float,
    defocus_angle: float,
    focus_distance: float,
    record: "Optional[CameraPerspectiveRecord]",
    signal: "Optional[CameraPerspectiveSignal]",
) -> None:
    """Set the perspective intrinsics and optional embedded record/signal."""
    camera.image_width = image_width
    camera.image_height = image_height
    camera.samples_per_pixel = samples_per_pixel
    camera.max_depth = max_depth
    camera.field_of_view = field_of_view
    camera.defocus_angle = defocus_angle
    camera.focus_distance = focus_distance
    if record is not None:
        camera.record = record
    if signal is not None:
        camera.signal = signal


class CameraPerspectiveRecord(RSC_CameraPerspectiveRecord):
    """Selects which camera output channels to capture and how to route them.

    Args:
        name: Record name (unique within the camera).
        has_rgb: Whether to capture the RGB image.
        has_depth: Whether to capture the depth buffer.
        has_entity_info: Whether to capture per-pixel entity ids.
        target_controller_reference: Controller that receives this record via
            the world callback. Empty disables controller delivery.
        target_recorder_reference: Recorder that persists this record. Empty
            disables recorder delivery.
    """

    def __init__(
        self,
        name: str = "",
        *,
        has_rgb: bool = True,
        has_depth: bool = False,
        has_entity_info: bool = False,
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.has_rgb = has_rgb
        self.has_depth = has_depth
        self.has_entity_info = has_entity_info
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference


class CameraPerspectiveSignal(RSC_CameraPerspectiveSignal):
    """Declares how many controller-driven pose/optics signals the camera reads.

    Args:
        name: Signal name (unique within the camera).
        source_controller_name: Controller that drives these signals each
            callback step.
        position_signal_num: Number of position signals.
        orientation_signal_num: Number of orientation signals.
        optics_signal_num: Number of optics (focus / field-of-view) signals.
    """

    def __init__(
        self,
        name: str = "",
        *,
        source_controller_name: str = "",
        position_signal_num: int = 1,
        orientation_signal_num: int = 1,
        optics_signal_num: int = 1,
    ) -> None:
        super().__init__()
        self.name = name
        self.source_controller_name = source_controller_name
        self.position_signal_num = position_signal_num
        self.orientation_signal_num = orientation_signal_num
        self.optics_signal_num = optics_signal_num


class CameraPerspectiveFree(RSC_CameraPerspectiveFree):
    """A perspective camera placed at an absolute world pose.

    Args:
        name: Unique entity name.
        origin_position: World position ``(x, y, z)`` in meters.
        orientation: Orientation quaternion ``(x, y, z, w)``.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        samples_per_pixel: Ray-tracing samples per pixel; higher is less noisy
            but slower.
        max_depth: Maximum ray bounces; higher captures more indirect light but
            is slower.
        field_of_view: Vertical field of view in degrees.
        defocus_angle: Depth-of-field defocus angle in degrees; 0 disables depth
            of field.
        focus_distance: Focus distance in meters; matters only with a non-zero
            ``defocus_angle``.
        record: Embedded output record. ``None`` keeps the default record.
        signal: Embedded signal declaration. ``None`` keeps the default signal.
    """

    def __init__(
        self,
        name: str = "camera",
        *,
        origin_position: Vec3rf = (0.0, 0.0, 0.0),
        orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
        image_width: int = 1280,
        image_height: int = 720,
        samples_per_pixel: int = 8,
        max_depth: int = 5,
        field_of_view: float = 60.0,
        defocus_angle: float = 0.0,
        focus_distance: float = 1.0,
        record: Optional[CameraPerspectiveRecord] = None,
        signal: Optional[CameraPerspectiveSignal] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.origin_position = _vec3rf(origin_position)
        self.orientation = _quat3rf(orientation)
        _apply_perspective(
            self,
            image_width,
            image_height,
            samples_per_pixel,
            max_depth,
            field_of_view,
            defocus_angle,
            focus_distance,
            record,
            signal,
        )


class CameraPerspectiveRigidBodyLinked(RSC_CameraPerspectiveRigidBodyLinked):
    """A perspective camera rigidly attached to a structure's rigid body.

    The camera rides a structure's rigid body segment, offset by a body-relative
    anchor and orientation, so it moves with that body. Either pass
    ``attached_to`` as the :class:`StructureVoxelBody` it rides (or a
    ``(body, segment_bid)`` tuple) and the structure/body/segment ids are
    resolved automatically, or set ``structure_name`` / ``body_sid`` /
    ``segment_bid`` directly.

    Args:
        name: Unique entity name.
        attached_to: Body object (or ``(body, segment_bid)`` tuple) the camera
            rides; resolves the ids below. The body must already be added to its
            structure. ``None`` uses the explicit ids.
        structure_name: Name of the structure to attach to.
        body_sid: Body sub-id within the structure the camera rides.
        segment_bid: Segment id of that body the camera rides.
        relative_anchor: Body-relative mount position ``(x, y, z)`` in meters.
        relative_orientation: Body-relative orientation quaternion
            ``(x, y, z, w)``.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        samples_per_pixel: Ray-tracing samples per pixel; higher is less noisy
            but slower.
        max_depth: Maximum ray bounces; higher captures more indirect light but
            is slower.
        field_of_view: Vertical field of view in degrees.
        defocus_angle: Depth-of-field defocus angle in degrees; 0 disables depth
            of field.
        focus_distance: Focus distance in meters; matters only with a non-zero
            ``defocus_angle``.
        record: Embedded output record. ``None`` keeps the default record.
        signal: Embedded signal declaration. ``None`` keeps the default signal.
    """

    def __init__(
        self,
        name: str = "camera",
        *,
        attached_to: Optional[BodyRef] = None,
        structure_name: str = "",
        body_sid: int = 0,
        segment_bid: int = 0,
        relative_anchor: Vec3rf = (0.0, 0.0, 0.0),
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
        image_width: int = 256,
        image_height: int = 256,
        samples_per_pixel: int = 8,
        max_depth: int = 5,
        field_of_view: float = 60.0,
        defocus_angle: float = 0.0,
        focus_distance: float = 0.5,
        record: Optional[CameraPerspectiveRecord] = None,
        signal: Optional[CameraPerspectiveSignal] = None,
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
        _apply_perspective(
            self,
            image_width,
            image_height,
            samples_per_pixel,
            max_depth,
            field_of_view,
            defocus_angle,
            focus_distance,
            record,
            signal,
        )
