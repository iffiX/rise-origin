# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""One-line builders for :class:`RS_CameraConfig`.

Two camera kinds are supported:

* **Free perspective** — placed at an absolute position/orientation in the
  world. Useful for a fixed observer or a viewer-driven orbit camera.
* **Rigid-body-linked perspective** — rigidly attached to a rigid body of a
  structure (e.g. a robot's torso), so it moves with that body. Useful for an
  on-board "what the robot sees" sensor.

Ray-tracing cost scales with ``image_width * image_height * samples_per_pixel``
and with ``max_depth`` (ray bounces); keep them modest for on-board sensors.

Example::

    from rise_origin.create import CameraCreator
    cam = CameraCreator.free("observer", origin_position=(-0.5, 0.0, 0.3))
    eye = CameraCreator.rigid_body_linked(
        "torso_cam", structure_name="quadruped",
        attached_to=torso.segment(0),
        relative_anchor=(0.2, 0.0, 0.27),
    )
"""

from rise_origin import (
    RQuat3rf,
    RVec3rf,
    RS_CameraConfig,
    RS_FreePerspectiveCameraConfig,
    RS_RigidBodyLinkedPerspectiveCameraConfig,
    RSE_CameraType,
)
from rise_origin.create.structure import SegmentRef
from rise_origin.create.utils import Quat, Vec3

__all__ = ["CameraCreator"]


class CameraCreator:
    """Factory of :class:`RS_CameraConfig` presets (free and body-linked)."""

    @staticmethod
    def free(
        name: str = "camera",
        *,
        origin_position: Vec3 = (0.0, 0.0, 0.0),
        orientation: Quat = (0.0, 0.0, 0.0, 1.0),
        image_width: int = 1280,
        image_height: int = 720,
        samples_per_pixel: int = 8,
        max_depth: int = 5,
        field_of_view: float = 60.0,
        focus_distance: float = 1.0,
        defocus_angle: float = 0.0,
    ) -> RS_CameraConfig:
        """Build a free perspective camera at an absolute pose.

        The camera's local forward axis is +X and local up is +Z.

        Args:
            name: Unique camera name.
            origin_position: World position ``(x, y, z)`` in meters.
            orientation: Orientation quaternion ``(x, y, z, w)``.
            image_width: Image width in pixels. Must be > 0.
            image_height: Image height in pixels. Must be > 0.
            samples_per_pixel: Ray-tracing samples per pixel (> 0). Higher is
                less noisy but slower.
            max_depth: Maximum ray bounces (> 0). Higher captures more indirect
                light but is slower.
            field_of_view: Vertical field of view in degrees (> 0).
            focus_distance: Focus distance in meters (> 0); matters only with a
                non-zero ``defocus_angle``.
            defocus_angle: Depth-of-field defocus angle in degrees (>= 0; 0
                disables depth of field).

        Returns:
            A populated :class:`RS_CameraConfig`.
        """
        camera = RS_CameraConfig()
        camera.name = name
        camera.type = RSE_CameraType.RSE_CAMERA_FREE_PERSPECTIVE

        cfg = RS_FreePerspectiveCameraConfig()
        cfg.image_width = image_width
        cfg.image_height = image_height
        cfg.samples_per_pixel = samples_per_pixel
        cfg.max_depth = max_depth
        cfg.field_of_view = field_of_view
        cfg.focus_distance = focus_distance
        cfg.defocus_angle = defocus_angle
        cfg.origin_position = RVec3rf(*origin_position)
        cfg.orientation = RQuat3rf(*orientation)
        camera.config = cfg
        return camera

    @staticmethod
    def rigid_body_linked(
        name: str,
        *,
        structure_name: str,
        attached_to: SegmentRef,
        relative_anchor: Vec3 = (0.0, 0.0, 0.0),
        relative_orientation: Quat = (0.0, 0.0, 0.0, 1.0),
        image_width: int = 256,
        image_height: int = 256,
        samples_per_pixel: int = 8,
        max_depth: int = 5,
        field_of_view: float = 60.0,
        focus_distance: float = 0.5,
        defocus_angle: float = 0.0,
    ) -> RS_CameraConfig:
        """Build a camera rigidly attached to a structure's rigid body segment.

        The camera rides the named structure's rigid body segment (passed as a
        :class:`SegmentRef`), offset by a body-relative anchor and orientation.
        Its local forward axis is +X and local up is +Z.

        Args:
            name: Unique camera name.
            structure_name: Name of the structure to attach to.
            attached_to: Segment of the structure that the camera rides
                (``body.segment(bid)`` from the structure builder).
            relative_anchor: Body-relative mount position ``(x, y, z)`` in meters.
            relative_orientation: Body-relative orientation quaternion
                ``(x, y, z, w)``.
            image_width: Image width in pixels. Must be > 0.
            image_height: Image height in pixels. Must be > 0.
            samples_per_pixel: Ray-tracing samples per pixel (> 0).
            max_depth: Maximum ray bounces (> 0).
            field_of_view: Vertical field of view in degrees (> 0).
            focus_distance: Focus distance in meters (> 0).
            defocus_angle: Depth-of-field defocus angle in degrees (>= 0).

        Returns:
            A populated :class:`RS_CameraConfig`.
        """
        camera = RS_CameraConfig()
        camera.name = name
        camera.type = RSE_CameraType.RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE

        cfg = RS_RigidBodyLinkedPerspectiveCameraConfig()
        cfg.structure_name = structure_name
        cfg.body_sid = attached_to.body_sid
        cfg.segment_bid = attached_to.segment_bid
        cfg.relative_anchor = RVec3rf(*relative_anchor)
        cfg.relative_orientation = RQuat3rf(*relative_orientation)
        cfg.image_width = image_width
        cfg.image_height = image_height
        cfg.samples_per_pixel = samples_per_pixel
        cfg.max_depth = max_depth
        cfg.field_of_view = field_of_view
        cfg.focus_distance = focus_distance
        cfg.defocus_angle = defocus_angle
        camera.config = cfg
        return camera
