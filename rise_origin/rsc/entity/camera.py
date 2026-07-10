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
    RSC_CameraStereoFree,
    RSC_CameraStereoRecord,
    RSC_CameraStereoRigidBodyLinked,
    RSC_CameraStereoSignal,
)

from rise_origin import RQuat3rf, RVec3rf

__all__ = [
    "CameraPerspectiveRecord",
    "CameraPerspectiveSignal",
    "CameraPerspectiveFree",
    "CameraPerspectiveRigidBodyLinked",
    "CameraStereoRecord",
    "CameraStereoSignal",
    "CameraStereoFree",
    "CameraStereoRigidBodyLinked",
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


def _apply_stereo(
    camera,
    image_width: int,
    image_height: int,
    samples_per_pixel: int,
    max_depth: int,
    interpupillary_distance: float,
    convergence_distance: float,
    angle_left: float,
    angle_right: float,
    angle_up: float,
    angle_down: float,
    record: "Optional[CameraStereoRecord]",
    signal: "Optional[CameraStereoSignal]",
) -> None:
    """Set the stereo intrinsics and optional embedded record/signal."""
    camera.image_width = image_width
    camera.image_height = image_height
    camera.samples_per_pixel = samples_per_pixel
    camera.max_depth = max_depth
    camera.interpupillary_distance = interpupillary_distance
    camera.convergence_distance = convergence_distance
    camera.angle_left = angle_left
    camera.angle_right = angle_right
    camera.angle_up = angle_up
    camera.angle_down = angle_down
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
        has_normal: Whether to capture the world-space normal buffer.
        has_motion: Whether to capture the screen-space motion buffer.
        has_entity_info: Whether to capture per-pixel entity ids.
        is_depth_hyperbolic: Whether to encode depth as hyperbolic device depth
            instead of metric meters.
        hyperbolic_near_depth: Near plane distance for hyperbolic depth encoding.
        hyperbolic_far_depth: Far plane distance for hyperbolic depth encoding.
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
        has_normal: bool = False,
        has_motion: bool = False,
        has_entity_info: bool = False,
        is_depth_hyperbolic: bool = False,
        hyperbolic_near_depth: float = 0.1,
        hyperbolic_far_depth: float = 1000.0,
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.has_rgb = has_rgb
        self.has_depth = has_depth
        self.has_normal = has_normal
        self.has_motion = has_motion
        self.has_entity_info = has_entity_info
        self.is_depth_hyperbolic = is_depth_hyperbolic
        self.hyperbolic_near_depth = hyperbolic_near_depth
        self.hyperbolic_far_depth = hyperbolic_far_depth
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference


class CameraPerspectiveSignal(RSC_CameraPerspectiveSignal):
    """Declares how many controller-driven pose/optics signals the camera reads.

    Args:
        name: Signal name (unique within the camera).
        source_controller_reference: Controller that drives these signals each
            callback step.
        position_signal_num: Number of position signals.
        orientation_signal_num: Number of orientation signals.
        optics_signal_num: Number of optics (focus / field-of-view) signals.
    """

    def __init__(
        self,
        name: str = "",
        *,
        source_controller_reference: str = "",
        position_signal_num: int = 1,
        orientation_signal_num: int = 1,
        optics_signal_num: int = 1,
    ) -> None:
        super().__init__()
        self.name = name
        self.source_controller_reference = source_controller_reference
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
        origin_position: RVec3rf = RVec3rf.zero(),
        orientation: RQuat3rf = RQuat3rf.unit(),
        image_width: int = 1280,
        image_height: int = 720,
        samples_per_pixel: int = 4,
        max_depth: int = 3,
        field_of_view: float = 60.0,
        defocus_angle: float = 0.0,
        focus_distance: float = 1.0,
        record: Optional[CameraPerspectiveRecord] = None,
        signal: Optional[CameraPerspectiveSignal] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.origin_position = origin_position
        self.orientation = orientation
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
    anchor and orientation, so it moves with that body. Identify the target with
    ``structure_name`` / ``body_sid`` / ``segment_bid``.

    Args:
        name: Unique entity name.
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
        structure_name: str = "",
        body_sid: int = 0,
        segment_bid: int = 0,
        relative_anchor: RVec3rf = RVec3rf.zero(),
        relative_orientation: RQuat3rf = RQuat3rf.unit(),
        image_width: int = 1280,
        image_height: int = 720,
        samples_per_pixel: int = 4,
        max_depth: int = 3,
        field_of_view: float = 60.0,
        defocus_angle: float = 0.0,
        focus_distance: float = 1.0,
        record: Optional[CameraPerspectiveRecord] = None,
        signal: Optional[CameraPerspectiveSignal] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.structure_name = structure_name
        self.body_sid = body_sid
        self.segment_bid = segment_bid
        self.relative_anchor = relative_anchor
        self.relative_orientation = relative_orientation
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


class CameraStereoRecord(RSC_CameraStereoRecord):
    """Selects which stereo camera output channels to capture and how to route them.

    Args:
        name: Record name (unique within the camera).
        has_rgb: Whether to capture the RGB image of each eye.
        has_depth: Whether to capture the depth buffer of each eye.
        has_normal: Whether to capture the world-space normal buffer of each eye.
        has_motion: Whether to capture the screen-space motion buffer of each eye.
        has_entity_info: Whether to capture per-pixel entity ids of each eye.
        is_depth_hyperbolic: Whether to encode depth as hyperbolic device depth
            instead of metric meters.
        hyperbolic_near_depth: Near plane distance for hyperbolic depth encoding.
        hyperbolic_far_depth: Far plane distance for hyperbolic depth encoding.
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
        has_depth: bool = True,
        has_normal: bool = False,
        has_motion: bool = False,
        has_entity_info: bool = False,
        is_depth_hyperbolic: bool = False,
        hyperbolic_near_depth: float = 0.1,
        hyperbolic_far_depth: float = 1000.0,
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.has_rgb = has_rgb
        self.has_depth = has_depth
        self.has_normal = has_normal
        self.has_motion = has_motion
        self.has_entity_info = has_entity_info
        self.is_depth_hyperbolic = is_depth_hyperbolic
        self.hyperbolic_near_depth = hyperbolic_near_depth
        self.hyperbolic_far_depth = hyperbolic_far_depth
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference


class CameraStereoSignal(RSC_CameraStereoSignal):
    """Declares the controller that drives the stereo camera's per-eye head pose.

    Args:
        name: Signal name (unique within the camera).
        source_controller_reference: Controller that drives the per-eye pose each
            callback step.
    """

    def __init__(
        self,
        name: str = "",
        *,
        source_controller_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.source_controller_reference = source_controller_reference


class CameraStereoFree(RSC_CameraStereoFree):
    """A free stereo camera placed at an absolute world head pose.

    Args:
        name: Unique entity name.
        origin_position: World head position ``(x, y, z)`` in meters.
        orientation: Head orientation quaternion ``(x, y, z, w)``.
        image_width: Per-eye image width in pixels.
        image_height: Per-eye image height in pixels.
        samples_per_pixel: Ray-tracing samples per pixel; higher is less noisy
            but slower.
        max_depth: Maximum ray bounces; higher captures more indirect light but
            is slower.
        interpupillary_distance: Distance between the two eyes in meters.
        convergence_distance: Toe-in convergence distance in meters; 0 keeps the
            eyes parallel.
        angle_left: Left edge angle of the eye frustum in radians.
        angle_right: Right edge angle of the eye frustum in radians.
        angle_up: Top edge angle of the eye frustum in radians.
        angle_down: Bottom edge angle of the eye frustum in radians.
        record: Embedded output record. ``None`` keeps the default record.
        signal: Embedded signal declaration. ``None`` keeps the default signal.
    """

    def __init__(
        self,
        name: str = "camera",
        *,
        origin_position: RVec3rf = RVec3rf.zero(),
        orientation: RQuat3rf = RQuat3rf.unit(),
        image_width: int = 1280,
        image_height: int = 720,
        samples_per_pixel: int = 4,
        max_depth: int = 3,
        interpupillary_distance: float = 0.063,
        convergence_distance: float = 0.0,
        angle_left: float = -0.7853981633974483,
        angle_right: float = 0.7853981633974483,
        angle_up: float = 0.7853981633974483,
        angle_down: float = -0.7853981633974483,
        record: Optional[CameraStereoRecord] = None,
        signal: Optional[CameraStereoSignal] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.origin_position = origin_position
        self.orientation = orientation
        _apply_stereo(
            self,
            image_width,
            image_height,
            samples_per_pixel,
            max_depth,
            interpupillary_distance,
            convergence_distance,
            angle_left,
            angle_right,
            angle_up,
            angle_down,
            record,
            signal,
        )


class CameraStereoRigidBodyLinked(RSC_CameraStereoRigidBodyLinked):
    """A stereo camera rigidly attached to a structure's rigid body.

    The stereo rig rides a structure's rigid body segment, offset by a body-relative
    anchor and orientation, so it moves with that body, for robot teleoperation.
    Identify the target with ``structure_name`` / ``body_sid`` / ``segment_bid``.

    Args:
        name: Unique entity name.
        structure_name: Name of the structure to attach to.
        body_sid: Body sub-id within the structure the camera rides.
        segment_bid: Segment id of that body the camera rides.
        relative_anchor: Body-relative mount position ``(x, y, z)`` in meters.
        relative_orientation: Body-relative orientation quaternion
            ``(x, y, z, w)``.
        image_width: Per-eye image width in pixels.
        image_height: Per-eye image height in pixels.
        samples_per_pixel: Ray-tracing samples per pixel; higher is less noisy
            but slower.
        max_depth: Maximum ray bounces; higher captures more indirect light but
            is slower.
        interpupillary_distance: Distance between the two eyes in meters.
        convergence_distance: Toe-in convergence distance in meters; 0 keeps the
            eyes parallel.
        angle_left: Left edge angle of the eye frustum in radians.
        angle_right: Right edge angle of the eye frustum in radians.
        angle_up: Top edge angle of the eye frustum in radians.
        angle_down: Bottom edge angle of the eye frustum in radians.
        record: Embedded output record. ``None`` keeps the default record.
        signal: Embedded signal declaration. ``None`` keeps the default signal.
    """

    def __init__(
        self,
        name: str = "camera",
        *,
        structure_name: str = "",
        body_sid: int = 0,
        segment_bid: int = 0,
        relative_anchor: RVec3rf = RVec3rf.zero(),
        relative_orientation: RQuat3rf = RQuat3rf.unit(),
        image_width: int = 1280,
        image_height: int = 720,
        samples_per_pixel: int = 4,
        max_depth: int = 3,
        interpupillary_distance: float = 0.063,
        convergence_distance: float = 0.0,
        angle_left: float = -0.7853981633974483,
        angle_right: float = 0.7853981633974483,
        angle_up: float = 0.7853981633974483,
        angle_down: float = -0.7853981633974483,
        record: Optional[CameraStereoRecord] = None,
        signal: Optional[CameraStereoSignal] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.structure_name = structure_name
        self.body_sid = body_sid
        self.segment_bid = segment_bid
        self.relative_anchor = relative_anchor
        self.relative_orientation = relative_orientation
        _apply_stereo(
            self,
            image_width,
            image_height,
            samples_per_pixel,
            max_depth,
            interpupillary_distance,
            convergence_distance,
            angle_left,
            angle_right,
            angle_up,
            angle_down,
            record,
            signal,
        )
