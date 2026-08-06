# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Minecraft wolf — a voxel dog displaying a leg-phase walking gait.

A blocky wolf in vanilla-Minecraft proportions: a bigger front body (the
mane/shoulders) carrying the head, a smaller rear body carrying the tail, joined
by a spine hinge, on four thin legs. The legs replay a fixed phase-offset gait;
there is no balance or heading control. A ball you push with the arrow keys is
left in the scene as a lure, and two eye cameras render the dog's view as a
duo picture-in-picture.

Run it and push the ball with the arrow keys::

    python examples/creatures/dog.py
"""

import math

import numpy as np
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt

from rise_origin import (
    RS_NULL_INDEX,
    RS_WorldCameraPerspectiveRecord,
    RS_WorldForceSignal,
    RS_WorldStructureVoxelSignal,
    RSE_StructureVoxelConstraintType,
    RSE_StructureVoxelHingeControlMode,
    rsc,
)

import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E401,E702 — find sibling _common
import _common

VOXEL = 0.03

FUR = (211, 209, 204)
DARK = (90, 88, 84)
SNOUT = (224, 214, 198)
BLACK = (35, 35, 35)
WHITE = (238, 238, 235)
BROWN = (168, 140, 95)
COLLAR = (196, 58, 52)
BALL_COLOR = (231, 111, 81)

LEG = (2, 2, 5)
LEG_TOP = LEG[2] * VOXEL

# Each leg angle = HIP_OFFSET + LEG_AMP[leg] * sin(2*pi*LEG_HZ[leg]*t + LEG_PHASE[leg]).
# Legs are indexed [front-right, front-left, back-right, back-left] (right = -Y).
HIP_OFFSET = 0.0  # neutral angle the legs swing around (rad)
HIP_LIMIT = 1.2  # hinge travel limit (rad); keep >= |HIP_OFFSET| + max(LEG_AMP)
LEG_PHASE = (0.0, math.pi, 0.0, math.pi)  # per-leg phase
LEG_AMP = (0.4, 0.4, 0.4, 0.4)
LEG_HZ = (1.0, 1.0, 1.0, 1.0)
PAW_FRICTION = 1.0  # foot grip in [0, 1]; raise to slip less
HIP_TORQUE, HIP_KP, HIP_KD = 20.0, 80.0, 1.0  # hip hinge motor (max torque, kp, kd)
SPINE_SIG = 4  # spine hinge signal index (held straight)
TAIL_SIG = 5  # tail hinge signal index
TAIL_HZ, TAIL_WAG = 2.5, 0.0  # tail wag frequency (Hz) and amplitude (rad)

# Ball lure (a separate structure you push with the arrow keys).
BALL_VOXEL = 0.02
BALL_R = 4
BALL_SHELL = 1
BALL_PUSH = 9.0
BALL_START = (0.7, 0.0, 0.06)

# Eye cameras + duo-view picture-in-picture. Mounted just ahead of each eye
# (voxel x=13, one past the nose) at eye height (z=4). The dog faces +X, up +Z,
# so its left eye is on the +Y side and its right eye on the -Y side.
PIP = 200
CAM_VOX_L = (13, 4, 4)  # dog's left eye (+Y side)
CAM_VOX_R = (13, 1, 4)  # dog's right eye (-Y side)
EYE_LABELS = ["Dog's left eye", "Dog's right eye"]
HINT = "Push the ball with the UP / DOWN / LEFT / RIGHT arrow keys"

CONTROLLER = "dog#brain"

# Material reference order for the dog; the per-voxel grids index into it.
DOG_MATERIALS = ["fur", "dark", "snout", "black", "white", "brown", "collar", "paw"]
_ID = {name: i for i, name in enumerate(DOG_MATERIALS)}

# Latest eye frames and the current key-driven push, shared control <-> hud.
_eye_frames = [None, None]
_push = np.zeros(2, dtype=float)


def _ids(grid):
    """Convert a string/None voxel grid into a material-id grid (RS_NULL for empty)."""
    out = np.full(grid.shape, RS_NULL_INDEX, dtype=np.int64)
    for name, index in _ID.items():
        out[grid == name] = index
    return out


def _front_body():
    """Front body grid (shoulders + head + ears + snout + collar)."""
    g = np.full((13, 6, 9), None, dtype=object)
    g[0:6, :, 0:6] = "fur"
    g[5:6, :, 0:6] = "collar"
    g[6:11, :, 1:7] = "fur"
    g[8:9, 0:2, 7:9] = "dark"  # left ear
    g[8:9, 4:6, 7:9] = "dark"  # right ear
    g[10:11, 0:1, 4:5] = "white"
    g[10:11, 1:2, 4:5] = "black"
    g[10:11, 2:4, 4:5] = "brown"
    g[10:11, 4:5, 4:5] = "black"
    g[10:11, 5:6, 4:5] = "white"
    g[10:11, 1:2, 5:6] = "white"
    g[10:11, 4:5, 5:6] = "white"
    g[11:13, 1:4, 1:4] = "snout"
    g[11:13, 1:4, 1:2] = "black"  # mouth base
    g[12:13, 2:3, 3:4] = "black"  # nose dot
    return g


def _back_body():
    """Rear body grid (torso)."""
    g = np.full((8, 6, 5), None, dtype=object)
    g[:, :, :] = "fur"
    return g


def _tail():
    """Tail grid — a separate stub that hinges off the rear."""
    g = np.full((5, 2, 2), None, dtype=object)
    g[:, :, :] = "fur"
    g[4:5, :, :] = "dark"
    return g


def _leg():
    """Leg grid (fur with a darker high-friction paw)."""
    g = np.full(LEG, None, dtype=object)
    g[:, :, :] = "fur"
    g[:, :, 0:1] = "paw"
    return g


def _eye_anchor(vox):
    """Camera mount point for an eye, in the structure frame.

    The anchor lives in the same frame as the front body's min corner, so the
    corner offset is added back to the voxel-center position.
    """
    corner = np.array((0.0, -3 * VOXEL, LEG_TOP), dtype=float)
    return tuple(corner + (np.array(vox, dtype=float) + 0.5) * VOXEL)


def _body(grid, corner):
    """A single rigid voxel body (segment 0) from a string/None grid."""
    ids = _ids(grid)
    return rsc.entity.StructureVoxelBody.from_grid(
        ids,
        voxel_size=VOXEL,
        relative_origin_position=corner,
    )


def _hinge(a_body_sid, b_body_sid, *, axis, anchor, signal_sid, angle_min, angle_max, torque, kp, kd):
    """A PD-position hinge driven by the structure signal ``signal_sid``."""
    return rsc.entity.StructureVoxelConstraint(
        RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT,
        a_body_sid=a_body_sid,
        a_segment_bid=0,
        b_body_sid=b_body_sid,
        b_segment_bid=0,
        relative_anchor=anchor,
        hinge_relative_axis=axis,
        hinge_control_mode=(
            RSE_StructureVoxelHingeControlMode.RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION
        ),
        hinge_min=angle_min,
        hinge_max=angle_max,
        hinge_max_torque=torque,
        hinge_position_control_kp=kp,
        hinge_position_control_kd=kd,
        hinge_signal_sid=signal_sid,
    )


def _ball_body():
    """A rigid ball core (segment 0) in a soft skin shell (segment 1)."""
    n = 2 * BALL_R
    coords = np.arange(n) + 0.5 - BALL_R
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    dist2 = gx * gx + gy * gy + gz * gz
    occupancy = dist2 <= float(BALL_R) ** 2
    core = dist2 <= float(BALL_R - BALL_SHELL) ** 2

    material = np.full((n, n, n), RS_NULL_INDEX, dtype=np.int64)
    segment = np.full((n, n, n), RS_NULL_INDEX, dtype=np.int64)
    material[occupancy] = 1  # ball_skin shell
    segment[occupancy] = 1
    material[core] = 0  # ball_core
    segment[core] = 0

    return rsc.entity.StructureVoxelBody.from_grid(
        material,
        segment_grid=segment,
        voxel_size=BALL_VOXEL,
        relative_origin_position=(
            BALL_START[0] - BALL_R * BALL_VOXEL,
            BALL_START[1] - BALL_R * BALL_VOXEL,
            BALL_START[2],
        ),
    )


def build():
    """Build the wolf + ball world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec("fur", color=FUR, kind="rigid"),
        _common.MaterialSpec("dark", color=DARK, kind="rigid"),
        _common.MaterialSpec("snout", color=SNOUT, kind="rigid"),
        _common.MaterialSpec("black", color=BLACK, kind="rigid"),
        _common.MaterialSpec("white", color=WHITE, kind="rigid"),
        _common.MaterialSpec("brown", color=BROWN, kind="rigid"),
        _common.MaterialSpec("collar", color=COLLAR, kind="rigid"),
        _common.MaterialSpec(
            "paw", color=DARK, kind="rigid", friction=(PAW_FRICTION, PAW_FRICTION)
        ),
        _common.MaterialSpec("ball_core", color=BALL_COLOR, kind="rigid"),
        _common.MaterialSpec("ball_skin", color=BALL_COLOR, kind="soft"),
    ]

    v = VOXEL
    front = _body(_front_body(), (0.0, -3 * v, LEG_TOP))
    back = _body(_back_body(), (-8 * v, -3 * v, LEG_TOP))
    tail = _body(_tail(), (-13 * v, -1 * v, LEG_TOP + 3 * v))
    bodies = [front, back, tail]

    # Body ids follow the bodies list: front=0, back=1, tail=2, then legs from 3.
    constraints = [
        _hinge(
            1,  # back
            2,  # tail
            axis=(0, 0, 1),
            anchor=(-8 * v, 0.0, LEG_TOP + 4 * v),
            signal_sid=TAIL_SIG,
            angle_min=-0.7,
            angle_max=0.7,
            torque=4.0,
            kp=3.0,
            kd=0.3,
        ),
        _hinge(
            0,  # front
            1,  # back
            axis=(0, 0, 1),
            anchor=(0.0, 0.0, LEG_TOP + 3 * v),
            signal_sid=SPINE_SIG,
            angle_min=-0.5,
            angle_max=0.5,
            torque=20.0,
            kp=10.0,
            kd=1.0,
        ),
    ]

    legs = [
        (0, (8 * v, -2 * v, LEG_TOP), (7 * v, -3 * v, 0.0), 0),  # front hip
        (0, (8 * v, 2 * v, LEG_TOP), (7 * v, 1 * v, 0.0), 1),  # front hip
        (1, (-6 * v, -2 * v, LEG_TOP), (-7 * v, -3 * v, 0.0), 2),  # back hip
        (1, (-6 * v, 2 * v, LEG_TOP), (-7 * v, 1 * v, 0.0), 3),  # back hip
    ]
    for hip_sid, anchor, corner, sig in legs:
        leg = _body(_leg(), corner)
        leg_sid = len(bodies)
        bodies.append(leg)
        constraints.append(
            _hinge(
                hip_sid,
                leg_sid,
                axis=(0, 1, 0),
                anchor=anchor,
                signal_sid=sig,
                angle_min=-HIP_LIMIT,
                angle_max=HIP_LIMIT,
                torque=HIP_TORQUE,
                kp=HIP_KP,
                kd=HIP_KD,
            )
        )

    dog = rsc.entity.StructureVoxel(
        "dog",
        material_references=DOG_MATERIALS,
        bodies=bodies,
        constraints=constraints,
        signal=rsc.entity.StructureVoxelSignal(
            "dog#signal", source_controller_reference=CONTROLLER, hinge_signal_num=6
        ),
    )

    # Ball lure: rigid core (the push acts here) + soft shell (touches the floor).
    ball_body = _ball_body()
    ball = rsc.entity.StructureVoxel(
        "ball",
        material_references=["ball_core", "ball_skin"],
        bodies=[ball_body],
    )
    push = rsc.signal.Force(
        "ball_push",
        structure_name="ball",
        source_controller_reference=CONTROLLER,
        body_sid=0,
        segment_bid=0,
    )

    # Two eye cameras, mounted just ahead of each eye, looking forward (+X).
    cameras = []
    for cam_name, vox in [("eye_left", CAM_VOX_L), ("eye_right", CAM_VOX_R)]:
        cameras.append(
            rsc.entity.CameraPerspectiveRigidBodyLinked(
                cam_name,
                structure_name="dog",
                body_sid=0,  # front body
                relative_anchor=_eye_anchor(vox),
                image_width=PIP,
                image_height=PIP,
                samples_per_pixel=4,
                field_of_view=80.0,
                record=rsc.entity.CameraPerspectiveRecord(
                    f"{cam_name}#rec",
                    has_rgb=True,
                    target_controller_reference=CONTROLLER,
                ),
            )
        )

    return _common.make_world(
        "dog",
        materials=materials,
        structures=[dog, ball, *cameras],
        signals=[push],
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
        rigid_body=rsc.simulation.RigidBody(constraint_position_correction_ratio=0.3),
    )


def control(time, controllers):
    """Replay the leg gait, push the ball, and capture the eye-camera frames."""
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return

    for signal in controller.signals():
        if isinstance(signal, RS_WorldStructureVoxelSignal):
            hinge = signal.hinge_signals().hinge()
            if len(hinge) >= 6:
                for leg in range(4):
                    wave = 2.0 * math.pi * LEG_HZ[leg] * time
                    hinge[leg] = HIP_OFFSET + LEG_AMP[leg] * math.sin(
                        wave + LEG_PHASE[leg]
                    )
                hinge[SPINE_SIG] = 0.0
                hinge[TAIL_SIG] = TAIL_WAG * math.sin(2.0 * math.pi * TAIL_HZ * time)
        elif isinstance(signal, RS_WorldForceSignal):
            force = (
                signal.force_signals().force()
            )  # shape (1, 3): one rigid-body force vector
            if force.shape[0] >= 1:
                force[0] = (float(_push[0]), float(_push[1]), 0.0)

    eye = 0
    for record in controller.records():
        if isinstance(record, RS_WorldCameraPerspectiveRecord):
            if eye < 2 and record.is_valid() and record.width() > 0:
                _eye_frames[eye] = np.array(record.rgb.pixel(), copy=True)
            eye += 1


def _draw_pip(painter, frame, x, y, label, font):
    """Draw one eye frame as a black-bordered picture-in-picture with a label."""
    height, width, _ = frame.shape
    image = QtGui.QImage(
        np.ascontiguousarray(frame).data,
        width,
        height,
        3 * width,
        QtGui.QImage.Format.Format_RGB888,
    ).copy()
    painter.fillRect(x - 3, y - 3, PIP + 6, PIP + 6, QtGui.QColor(0, 0, 0))
    painter.drawImage(QtCore.QRect(x, y, PIP, PIP), image)
    painter.setFont(font)
    painter.setPen(QtGui.QColor(40, 40, 40))
    metrics = QtGui.QFontMetrics(font)
    painter.drawText(
        x + (PIP - metrics.horizontalAdvance(label)) // 2, y + PIP + 20, label
    )


def hud(painter, viewer):
    """Read the arrow keys into the push vector and draw the duo eye view."""
    _push[0] = (BALL_PUSH if viewer.is_key_down(Qt.Key.Key_Right) else 0.0) - (
        BALL_PUSH if viewer.is_key_down(Qt.Key.Key_Left) else 0.0
    )
    _push[1] = (BALL_PUSH if viewer.is_key_down(Qt.Key.Key_Up) else 0.0) - (
        BALL_PUSH if viewer.is_key_down(Qt.Key.Key_Down) else 0.0
    )

    hint_font = QtGui.QFont()
    hint_font.setPointSize(11)
    painter.setFont(hint_font)
    painter.setPen(QtGui.QColor(40, 40, 40))
    painter.drawText(16, 24, HINT)

    if any(frame is None for frame in _eye_frames):
        return
    label_font = QtGui.QFont()
    label_font.setPointSize(10)
    total_w = 2 * PIP + 12
    x0 = (painter.device().width() - total_w) // 2
    for i, frame in enumerate(_eye_frames):
        _draw_pip(painter, frame, x0 + i * (PIP + 12), 44, EYE_LABELS[i], label_font)


def main():
    """Build the world and launch it in the selected run mode."""
    _common.launch(build(), control, hud=hud, camera_focus=(0.0, 0.0, 0.2))


if __name__ == "__main__":
    main()
