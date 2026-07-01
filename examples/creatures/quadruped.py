# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Quadruped walker — hinge joints and a trotting gait.

What you'll learn:

* Reconstruct a full articulated robot: a torso plus four legs, each leg an upper
  and lower segment, every part a rigid core wrapped in a soft shell.
* Place tilted limb segments with body orientations and structure-relative hinge
  anchors (hips and knees).
* Drive a diagonal trot from a control callback with phase-offset sine waves.
* Mount an on-board camera that rides the torso, drawn picture-in-picture in a
  ``hud`` overlay.

The body, upper limbs, and lower limbs each get their own colour.

Run it::

    python examples/creatures/quadruped.py
"""

import math

import numpy as np
from PySide6 import QtCore, QtGui

from rise_origin import (
    RS_WorldCameraPerspectiveRecord,
    RS_WorldStructureVoxelSignal,
    RSE_StructureVoxelConstraintType,
    rsc,
)

import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E401,E702 — find sibling _common
import _common

CAMERA_NAME = "torso_cam"  # on-board camera mounted on the torso
PIP_SIZE = 240  # on-screen size of the picture-in-picture, in pixels
CONTROLLER = "quadruped#brain"

# Latest on-board camera frame, captured in control() and drawn in hud().
_robot_frame = None

VOXEL = 0.01

# Voxel grids (rigid cores) for the torso and each leg segment.
TORSO = (30, 11, 7)
LEG = (2, 2, 12)

# Soft-shell thickness (voxels) around each rigid core.
TORSO_SHELL = 1
UPPER_SHELL = 1
LOWER_SHELL = 2

# Limb tilt angles about +Y.
UPPER_ANGLE = math.radians(45.0)
LOWER_ANGLE = math.radians(-45.0)

# Lateral offsets (voxels) that splay the legs out to avoid self-collision.
UPPER_LATERAL = 3
LOWER_LATERAL = 7

# Hinge limits and motor torque.
HIP_MIN, HIP_MAX = -0.7, 0.7
KNEE_MIN, KNEE_MAX = -1, 1
HINGE_TORQUE = 15.0

# Part colours (body / upper limb / lower limb).
BODY_COLOR = _common.PALETTE["deep_blue"]
UPPER_COLOR = _common.PALETTE["blue_green"]
LOWER_COLOR = _common.PALETTE["amber"]

GAIT_HZ = 2
HIP_AMPLITUDE = 0.5
KNEE_AMPLITUDE = 0.4

# Material reference order; per-voxel ids index into it.
MATERIAL_REFERENCES = [
    "body_core",
    "body_skin",
    "upper_core",
    "upper_skin",
    "lower_core",
    "lower_skin",
]
BODY_CORE, BODY_SKIN, UPPER_CORE, UPPER_SKIN, LOWER_CORE, LOWER_SKIN = range(6)


def _quat_axis_angle(axis, angle):
    """Quaternion ``(x, y, z, w)`` for a rotation of ``angle`` about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    s = math.sin(half)
    return (
        float(axis[0] * s),
        float(axis[1] * s),
        float(axis[2] * s),
        float(math.cos(half)),
    )


def _rotate(vec, quat):
    """Rotate 3-vector ``vec`` by quaternion ``quat`` given as ``(x, y, z, w)``."""
    vec = np.asarray(vec, dtype=np.float64)
    qv = np.asarray(quat[:3], dtype=np.float64)
    w = quat[3]
    return vec + 2.0 * np.cross(qv, np.cross(qv, vec) + w * vec)


def _full_corner(core_corner, orientation, shell):
    """Min-corner of the full (shelled) body so its rigid core stays put.

    The shell adds ``shell`` voxels on every side; shifting the body origin by
    the rotated shell offset keeps the rigid core at ``core_corner``.
    """
    offset = _rotate((shell * VOXEL, shell * VOXEL, shell * VOXEL), orientation)
    return tuple(np.asarray(core_corner, dtype=np.float64) - offset)


def _shelled_box(core_size, shell, core_id, skin_id, core_corner, orientation):
    """A rigid core box (segment 0) wrapped in a soft skin shell (segment 1)."""
    cx, cy, cz = core_size
    fx, fy, fz = cx + 2 * shell, cy + 2 * shell, cz + 2 * shell
    material = np.full((fx, fy, fz), skin_id, dtype=np.int64)
    segment = np.ones((fx, fy, fz), dtype=np.int64)
    material[shell : shell + cx, shell : shell + cy, shell : shell + cz] = core_id
    segment[shell : shell + cx, shell : shell + cy, shell : shell + cz] = 0
    return rsc.entity.StructureVoxelBody.from_grid(
        np.ones((fx, fy, fz), dtype=bool),
        material_grid=material,
        segment_grid=segment,
        voxel_size=VOXEL,
        relative_origin_position=_full_corner(core_corner, orientation, shell),
        relative_orientation=orientation,
    )


def _hinge(a_body_sid, b_body_sid, *, axis, anchor, signal_sid, angle_min, angle_max):
    """A motorized hinge driven by the structure signal ``signal_sid``."""
    return rsc.entity.StructureVoxelConstraint(
        RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT,
        a_body_sid=a_body_sid,
        a_segment_bid=0,
        b_body_sid=b_body_sid,
        b_segment_bid=0,
        relative_anchor=anchor,
        hinge_relative_axis=axis,
        hinge_min=angle_min,
        hinge_max=angle_max,
        hinge_max_torque=HINGE_TORQUE,
        hinge_signal_sid=signal_sid,
    )


def build():
    """Build the quadruped world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec("body_core", color=BODY_COLOR, kind="rigid"),
        _common.MaterialSpec("body_skin", color=BODY_COLOR, kind="soft"),
        _common.MaterialSpec("upper_core", color=UPPER_COLOR, kind="rigid"),
        _common.MaterialSpec("upper_skin", color=UPPER_COLOR, kind="soft"),
        _common.MaterialSpec("lower_core", color=LOWER_COLOR, kind="rigid"),
        _common.MaterialSpec("lower_skin", color=LOWER_COLOR, kind="soft"),
    ]

    torso_size = np.array(TORSO) * VOXEL
    leg_size = np.array(LEG) * VOXEL

    # Torso centred in XY and lifted so the angled legs reach the floor.
    torso_height = torso_size[2] / 2.0 + leg_size[2] * math.cos(UPPER_ANGLE) * 2.0
    torso_corner = (
        -torso_size[0] / 2.0,
        -torso_size[1] / 2.0,
        torso_height - torso_size[2] / 2.0,
    )
    torso = _shelled_box(
        TORSO, TORSO_SHELL, BODY_CORE, BODY_SKIN, torso_corner, (0.0, 0.0, 0.0, 1.0)
    )
    bodies = [torso]
    constraints = []

    # Four legs at the torso's bottom corners. Front legs splay -Y, back legs +Y.
    leg_corners = [
        (leg_size[0] / 2.0, 0.0, True),
        (torso_size[0] - leg_size[0] / 2.0, 0.0, True),
        (leg_size[0] / 2.0, torso_size[1], False),
        (torso_size[0] - leg_size[0] / 2.0, torso_size[1], False),
    ]
    hip_z = torso_size[2] / 2.0

    for leg_idx, (corner_x, corner_y, is_front) in enumerate(leg_corners):
        lateral = -1.0 if is_front else 1.0
        axis = (0.0, 1.0 if is_front else -1.0, 0.0)

        upper_quat = _quat_axis_angle((0.0, 1.0, 0.0), UPPER_ANGLE)
        hip_anchor = (
            torso_corner[0] + corner_x,
            torso_corner[1] + corner_y + lateral * UPPER_LATERAL * VOXEL,
            torso_corner[2] + hip_z,
        )
        # Place the upper leg so its top-centre lands on the hip anchor.
        top_local = np.array([leg_size[0] / 2.0, leg_size[1] / 2.0, leg_size[2]])
        upper_corner = np.array(hip_anchor) - _rotate(top_local, upper_quat)
        upper = _shelled_box(
            LEG, UPPER_SHELL, UPPER_CORE, UPPER_SKIN, tuple(upper_corner), upper_quat
        )

        # Knee at the bottom-centre of the upper leg.
        knee_local = np.array([leg_size[0] / 2.0, leg_size[1] / 2.0, 0.0])
        knee_anchor = upper_corner + _rotate(knee_local, upper_quat)
        lower_quat = _quat_axis_angle((0.0, 1.0, 0.0), LOWER_ANGLE)
        lower_corner = np.array(knee_anchor) - _rotate(top_local, lower_quat)
        lower_corner[1] += lateral * (LOWER_LATERAL - UPPER_LATERAL) * VOXEL
        lower = _shelled_box(
            LEG, LOWER_SHELL, LOWER_CORE, LOWER_SKIN, tuple(lower_corner), lower_quat
        )
        upper_sid = len(bodies)  # torso is body 0; legs append in (upper, lower) pairs
        lower_sid = upper_sid + 1
        bodies.extend([upper, lower])

        constraints.append(
            _hinge(
                0,  # torso
                upper_sid,
                axis=axis,
                anchor=hip_anchor,
                signal_sid=leg_idx,
                angle_min=HIP_MIN,
                angle_max=HIP_MAX,
            )
        )
        constraints.append(
            _hinge(
                upper_sid,
                lower_sid,
                axis=axis,
                anchor=tuple(knee_anchor),
                signal_sid=4 + leg_idx,
                angle_min=KNEE_MIN,
                angle_max=KNEE_MAX,
            )
        )

    quadruped = rsc.entity.StructureVoxel(
        "quadruped",
        material_references=MATERIAL_REFERENCES,
        bodies=bodies,
        constraints=constraints,
        signal=rsc.entity.StructureVoxelSignal(
            "quadruped#signal", source_controller_reference=CONTROLLER, hinge_signal_num=8
        ),
    )

    # On-board camera riding the torso's rigid core, delivering frames to the
    # control callback through the same controller that drives the gait.
    camera = rsc.entity.CameraPerspectiveRigidBodyLinked(
        CAMERA_NAME,
        structure_name="quadruped",
        body_sid=0,  # torso body
        relative_anchor=(0.20, 0.0, 0.27),
        relative_orientation=(0.0, 0.2164, 0.0, 0.9763),
        image_width=PIP_SIZE,
        image_height=PIP_SIZE,
        samples_per_pixel=9,
        field_of_view=60.0,
        record=rsc.entity.CameraPerspectiveRecord(
            "torso_cam#rec", has_rgb=True, target_controller_reference=CONTROLLER
        ),
    )

    return _common.make_world(
        "quadruped",
        materials=materials,
        structures=[quadruped, camera],
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
        # Tight rigid-soft coupling needs a finer constraint step and gentler
        # joint position correction to keep the bonded skin shells stable.
        time=rsc.simulation.Time(dt_rigidbody_contact_and_constraint_update=0.0005),
        rigid_body=rsc.simulation.RigidBody(constraint_position_correction_ratio=0.1),
    )


def control(time, controllers):
    """Drive a diagonal trot and capture the on-board camera frame.

    Front-left and back-right share a phase; the other diagonal is offset by pi.
    """
    global _robot_frame
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return

    wave = 2.0 * math.pi * GAIT_HZ * time
    for signal in controller.signals():
        if not isinstance(signal, RS_WorldStructureVoxelSignal):
            continue
        hinge = signal.hinge_signals().hinge()
        if len(hinge) < 8:
            continue
        for leg_idx in range(4):
            phase = 0.0 if leg_idx in (0, 3) else math.pi
            hinge[leg_idx] = HIP_AMPLITUDE * math.sin(wave + phase)
            hinge[4 + leg_idx] = KNEE_AMPLITUDE * math.sin(wave + phase + math.pi / 2.0)

    for record in controller.records():
        if isinstance(record, RS_WorldCameraPerspectiveRecord):
            if record.is_valid() and record.width() > 0:
                _robot_frame = np.array(record.rgb.pixel(), copy=True)


def hud(painter, viewer):
    """Draw the on-board camera as a white-bordered picture-in-picture."""
    if _robot_frame is None:
        return
    height, width, _ = _robot_frame.shape
    image = QtGui.QImage(
        np.ascontiguousarray(_robot_frame).data,
        width,
        height,
        3 * width,
        QtGui.QImage.Format.Format_RGB888,
    ).copy()

    x = painter.device().width() - PIP_SIZE - 16
    y = 16
    painter.fillRect(
        x - 3, y - 3, PIP_SIZE + 6, PIP_SIZE + 6, QtGui.QColor(255, 255, 255)
    )
    painter.drawImage(QtCore.QRect(x, y, PIP_SIZE, PIP_SIZE), image)


def main():
    """Build the world and launch it in the selected run mode."""
    _common.launch(build(), control, hud=hud, camera_focus=(0.0, 0.0, 0.12))


if __name__ == "__main__":
    main()
