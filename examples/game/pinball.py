# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Space pinball — a levitated, steeply-tilted voxel pinball table.

A whole pinball table built as one structure, tilted to face you and floating
in space. Gravity rolls the ball down the playfield into two PD-hinge flippers;
a prismatic-joint piston launches the ball up the lane to start.

What you'll learn:

* Build a whole machine as one structure mixing **fixed** scenery (the pad, the
  perimeter wall with a smooth domed top, and a clear glass cover) with **rigid**
  moving parts (flippers, piston) joined by hinge and prismatic constraints, then
  tilt and levitate it by setting the structure's orientation once.
* Shape non-boxy scenery from a per-voxel material grid (the arch is a voxel
  ring) and emit it with :meth:`rsc.entity.StructureVoxelBody.from_grid`.
* Launch with a **prismatic piston** (normalized [0, 1] signal) and bat with
  **PD-controlled hinge flippers** (raw-radian target angles).
* Watch the drain with an **intersect-region record**: an oriented box under the
  bottom gap reports when the ball is lost, and the referee respawns it.

Controls: SPACE launches, LEFT / RIGHT work the flippers.

Run::

    python examples/game/pinball.py
"""

import math
import os, sys

import numpy as np
from pyquaternion import Quaternion
from PySide6 import QtGui
from PySide6.QtCore import Qt

from rise_origin import (
    RS_NULL_INDEX,
    RS_WorldIntersectRegionRecord,
    RS_WorldStructureVoxelSignal,
    RSC_EntityVariant,
    RSE_StructureVoxelConstraintType,
    RSE_StructureVoxelHingeControlMode,
    RSE_StructureVoxelPrismaticControlMode,
    rsc,
)

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E401,E702 — find sibling _common
import _common

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
V = 0.01  # voxel size (m)

# Playfield (designed flat in local voxels; the structure orientation stands it
# up). +Y is up-table (toward the domed top); -Y is the drain/flipper end.
W, L = 100, 150
PAD_T = 4
WALL_T = 3
WALL_H = 18  # walls, flippers, piston and bumpers all share this height
CX = W // 2
Y_ARCH = 98  # straight side walls below this; domed arch above
DRAIN_HALF = 4  # half-width of the bottom drain gap (voxels)
LANE_WALL_X = 86  # inner wall of the right launch lane
LANE_TOP = 88  # lane wall stops here so the ball spills into play

# Left re-entry tunnels (Space-Cadet style): two vertical dividers make a bank
# of parallel lanes in the upper left that the ball drops down through.
TUNNEL_DIVIDERS = (14, 28)  # x of each 3-voxel-thick divider (wide ~11-voxel lanes)
TUNNEL_Y0, TUNNEL_Y1 = 60, 96

# Ball: a bare rigid sphere parked in the launch lane.
BALL_R = 3  # ball radius (voxels)
BALL_START = (93 - 3, 20, PAD_T)  # min-corner local voxel of the launch-lane ball

# Stand the table up this many degrees from the floor and levitate it. The
# orientation also yaws the playfield to face the viewer's default camera.
TILT_FROM_FLOOR = 60.0
TABLE_POS = (0.30, 0.25, 0.10)

# Launch piston (prismatic) — fires +Y up the right lane.
PLUNGER_SIG = 0
PLUNGER_AXIS = (0, -1, 0)  # piston travels opposite to axis -> +Y toward ball
PLUNGER_FORCE = 100.0
PLUNGER_SPEED = 100.0
PLUNGER_STROKE = 0.40

# Flippers (PD hinges) — raw-radian target angles.
LEFT_SIG, RIGHT_SIG = 0, 1
FLIP_KP, FLIP_KD = 8.0, 0.5
FLIP_TORQUE, FLIP_SPEED = 12.0, 25.0
REST_L, FLIP_L = -0.35, 0.5
REST_R, FLIP_R = 0.35, -0.5
FLIP_H = 12  # shorter than WALL_H so the flippers clear the glass cover

BUMPER_R = 5
BUMPER_H = WALL_H  # same height as everything else
# Spread so the ball (6-voxel diameter) fits between them, and clear of the
# left tunnels (which end near x=31).
BUMPERS = [(44, 66), (74, 66), (59, 79), (44, 92), (74, 92)]

PAD_C = _common.PALETTE["deep_blue"]
RAIL_C = _common.PALETTE["blue_green"]
BUMPER_C = _common.PALETTE["amber"]
FLIP_C = _common.PALETTE["tiger_orange"]
BALL_C = _common.PALETTE["sky_blue"]
GLASS_C = (255, 255, 255)

# Table material reference order; per-voxel ids index into it. The ball is a
# separate structure with its own single-material reference.
TABLE_MATERIALS = ["anchor", "board", "rail", "bumper", "flipper", "glass"]
_T = {name: i for i, name in enumerate(TABLE_MATERIALS)}

CONTROLLER = "table#brain"
DRAIN_RECORD = "table#drain"

HINT = "SPACE = launch    LEFT / RIGHT = flippers"

_HINGE = RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT
_PRISMATIC = RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT
_HINGE_PD = (
    RSE_StructureVoxelHingeControlMode.RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION
)
_PRISMATIC_POSITION = (
    RSE_StructureVoxelPrismaticControlMode.RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION
)

# ---------------------------------------------------------------------------
# Shared run state
# ---------------------------------------------------------------------------
# Key state shared hud -> control, plus the drain/respawn handshake. ``drained``
# is raised by the referee callback and cleared by the hud once it has dropped a
# fresh ball back into the launch lane.
_state = {
    "fire": False,
    "left": False,
    "right": False,
    "populated": False,
    "drained": False,
    "in_drain": False,
}


# ---------------------------------------------------------------------------
# Scene content
# ---------------------------------------------------------------------------
def _table_quat():
    """Quaternion (x, y, z, w) standing the table up and facing the camera.

    Maps the flat local frame so the playfield normal points toward the
    viewer's default camera (which looks along +X), the up-table axis points
    up (leaning back by ``90 - TILT_FROM_FLOOR``), and the width runs across.
    """
    lean = math.radians(90.0 - TILT_FROM_FLOOR)
    sa, ca = math.sin(lean), math.cos(lean)
    rotation = np.array(
        [
            [
                0.0,
                sa,
                -ca,
            ],  # local X -> -Y, local Y -> (sa,0,ca), local Z -> (-ca,0,sa)
            [-1.0, 0.0, 0.0],
            [0.0, ca, sa],
        ]
    )
    q = Quaternion(matrix=rotation)
    return (q.x, q.y, q.z, q.w)


def _to_world(local_voxels):
    """Map a local voxel position on the table to a world position (m)."""
    x, y, z, w = _table_quat()
    rotated = Quaternion(w, x, y, z).rotate(np.array(local_voxels, dtype=float) * V)
    return tuple(np.array(TABLE_POS) + rotated)


def _wall_mask():
    """Boolean (W, L, WALL_H) occupancy of the perimeter wall.

    Posts and segments tile the sides, a voxel ring domes the top, the bottom
    wall keeps a central drain gap, the right launch lane has an inner wall, and
    two dividers in the upper left make a bank of re-entry tunnels.
    """
    xs = np.arange(W)[:, None] + 0.5
    ys = np.arange(L)[None, :] + 0.5

    # Straight side walls up to the arch.
    side = (ys >= WALL_T) & (ys < Y_ARCH) & ((xs < WALL_T) | (xs >= W - WALL_T))
    # Bottom wall with a central drain gap.
    bottom = (ys < WALL_T) & (np.abs(xs - CX) >= DRAIN_HALF)
    # Domed top: a voxel ring centred on (CX, Y_ARCH).
    r = np.sqrt((xs - CX) ** 2 + (ys - Y_ARCH) ** 2)
    arch = (ys >= Y_ARCH) & (r <= CX) & (r > CX - WALL_T)
    # Inner wall of the right launch lane.
    lane = (
        (xs >= LANE_WALL_X)
        & (xs < LANE_WALL_X + WALL_T)
        & (ys >= WALL_T)
        & (ys < LANE_TOP)
    )
    # Left re-entry tunnels: vertical dividers making a bank of parallel lanes.
    tunnels = np.zeros_like(side)
    for dx in TUNNEL_DIVIDERS:
        tunnels |= (
            (xs >= dx) & (xs < dx + WALL_T) & (ys >= TUNNEL_Y0) & (ys < TUNNEL_Y1)
        )

    mask = side | bottom | arch | lane | tunnels
    return np.repeat(mask[:, :, None], WALL_H, axis=2)


def _hinge(a, b, *, anchor, signal_sid, angle_min, angle_max):
    """A PD-position flipper hinge about the playfield normal (local Z)."""
    return rsc.entity.StructureVoxelConstraint(
        _HINGE,
        a=a,
        b=b,
        relative_anchor=anchor,
        hinge_relative_axis=(0, 0, 1),
        hinge_control_mode=_HINGE_PD,
        hinge_min=angle_min,
        hinge_max=angle_max,
        hinge_max_torque=FLIP_TORQUE,
        hinge_max_speed=FLIP_SPEED,
        hinge_position_control_kp=FLIP_KP,
        hinge_position_control_kd=FLIP_KD,
        hinge_signal_sid=signal_sid,
    )


def _make_table():
    """The levitated, tilted table: pad, walls, glass cover, bumpers, flippers, piston."""
    z0 = PAD_T * V
    body = rsc.entity.StructureVoxelBody

    # Fixed pad base that anchors the flipper and piston joints.
    pad = body.box(
        W,
        L,
        PAD_T,
        material_sid=_T["board"],
        voxel_size=V,
        relative_origin_position=(0, 0, 0),
    )
    # Perimeter wall as one voxel grid (sides, domed top, lane, tunnels).
    wall_mask = _wall_mask()
    wall = body.from_grid(
        wall_mask,
        material_grid=np.where(wall_mask, _T["rail"], RS_NULL_INDEX),
        voxel_size=V,
        relative_origin_position=(0, 0, z0),
    )
    # Clear glass cover over the whole table so the tilted ball never flies out.
    glass = body.box(
        W,
        L,
        1,
        material_sid=_T["glass"],
        voxel_size=V,
        relative_origin_position=(0, 0, (PAD_T + WALL_H) * V),
    )

    # Launch piston: fixed barrel + rigid piston sliding +Y up the lane.
    barrel = body.box(
        8,
        6,
        WALL_H,
        material_sid=_T["anchor"],
        voxel_size=V,
        relative_origin_position=(89 * V, 3 * V, z0),
    )
    piston = body.box(
        8,
        6,
        WALL_H,
        material_sid=_T["flipper"],
        voxel_size=V,
        relative_origin_position=(89 * V, 9 * V, z0),
    )

    # Flipper holder/guide walls (inlanes) between the side walls and flippers.
    left_inlane = body.box(
        4,
        22,
        WALL_H,
        material_sid=_T["rail"],
        voxel_size=V,
        relative_origin_position=(22 * V, 18 * V, z0),
    )
    right_inlane = body.box(
        4,
        22,
        WALL_H,
        material_sid=_T["rail"],
        voxel_size=V,
        relative_origin_position=(78 * V, 18 * V, z0),
    )

    # Flippers: rigid bars on PD hinges, kept shorter than the walls so their
    # tops never rub the glass cover.
    left = body.box(
        16,
        4,
        FLIP_H,
        material_sid=_T["flipper"],
        voxel_size=V,
        relative_origin_position=(30 * V, 18 * V, z0),
    )
    right = body.box(
        16,
        4,
        FLIP_H,
        material_sid=_T["flipper"],
        voxel_size=V,
        relative_origin_position=(54 * V, 18 * V, z0),
    )

    bodies = [pad, wall, glass, barrel, piston, left_inlane, right_inlane, left, right]

    # Pop bumpers (fixed cylinders); the ball bounces off them physically.
    for cx, cy in BUMPERS:
        bodies.append(
            body.cylinder(
                BUMPER_R,
                BUMPER_H,
                axis="z",
                material_sid=_T["bumper"],
                voxel_size=V,
                relative_origin_position=((cx - BUMPER_R) * V, (cy - BUMPER_R) * V, z0),
            )
        )

    constraints = [
        rsc.entity.StructureVoxelConstraint(
            _PRISMATIC,
            a=piston,
            b=barrel,
            relative_anchor=(93 * V, 9 * V, (PAD_T + WALL_H // 2) * V),
            prismatic_relative_axis=PLUNGER_AXIS,
            prismatic_control_mode=_PRISMATIC_POSITION,
            prismatic_min=0.0,
            prismatic_max=PLUNGER_STROKE,
            prismatic_max_force=PLUNGER_FORCE,
            prismatic_max_speed=PLUNGER_SPEED,
            prismatic_signal_sid=PLUNGER_SIG,
        ),
        _hinge(
            left,
            pad,
            anchor=(30 * V, 20 * V, (PAD_T + WALL_H // 2) * V),
            signal_sid=LEFT_SIG,
            angle_min=-0.6,
            angle_max=0.7,
        ),
        _hinge(
            right,
            pad,
            anchor=(70 * V, 20 * V, (PAD_T + WALL_H // 2) * V),
            signal_sid=RIGHT_SIG,
            angle_min=-0.7,
            angle_max=0.6,
        ),
    ]

    return rsc.entity.StructureVoxel(
        "table",
        material_references=TABLE_MATERIALS,
        origin_position=TABLE_POS,
        orientation=_table_quat(),
        bodies=bodies,
        constraints=constraints,
        signal=rsc.entity.StructureVoxelSignal(
            "table#signal",
            source_controller_name=CONTROLLER,
            hinge_signal_num=2,
            prismatic_signal_num=1,
        ),
    )


def _make_ball():
    """A bare rigid sphere parked in the launch lane.

    Placed in the table's own tilted frame (same ``origin_position`` /
    ``orientation`` as the table) so it rests on the playfield instead of
    jutting out behind the board.
    """
    return rsc.entity.StructureVoxel(
        "ball",
        material_references=["ball"],
        origin_position=TABLE_POS,
        orientation=_table_quat(),
        bodies=[
            rsc.entity.StructureVoxelBody.sphere(
                BALL_R,
                material_sid=0,
                voxel_size=V,
                relative_origin_position=(
                    BALL_START[0] * V,
                    BALL_START[1] * V,
                    BALL_START[2] * V,
                ),
            )
        ],
    )


def _make_drain():
    """An oriented box just below the bottom drain gap, outside the table.

    Only a lost ball that has fallen through the gap reaches it, so its
    membership rises from zero to one the moment the ball drains.
    """
    return rsc.record.IntersectRegionCuboid(
        DRAIN_RECORD,
        position=_to_world((CX, -8, PAD_T + WALL_H / 2)),
        orientation=_table_quat(),
        size=((2 * DRAIN_HALF + 8) * V, 16 * V, WALL_H * V),
        target_controller_reference=CONTROLLER,
    )


# ---------------------------------------------------------------------------
# World build and ball lifecycle
# ---------------------------------------------------------------------------
def build():
    """Build the levitated, tilted pinball table, the drain region, and the referee.

    The ball is dropped in after the simulation starts by :func:`_populate`, so
    the same path that respawns a drained ball also seeds the first one.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec("anchor", color=PAD_C, kind="rigid_fixed"),
        _common.MaterialSpec(
            "board",
            color=PAD_C,
            kind="rigid_fixed",
            emission_color=(255, 250, 235),
            emission_intensity=0.5,
        ),
        _common.MaterialSpec("rail", color=RAIL_C, kind="fixed"),
        _common.MaterialSpec("bumper", color=BUMPER_C, kind="fixed"),
        _common.MaterialSpec("flipper", color=FLIP_C, kind="rigid"),
        _common.MaterialSpec(
            "glass",
            color=GLASS_C,
            kind="fixed",
            roughness=0.0,
            is_transparent=True,
            ior=1.5,
        ),
        _common.MaterialSpec("ball", color=BALL_C, kind="rigid"),
    ]

    return _common.make_world(
        "pinball",
        materials=materials,
        structures=[_make_table()],
        records=[_make_drain()],
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
    )


def _populate(handle):
    """Drop a fresh ball into the launch lane."""
    handle.begin_modification_transaction()
    handle.add_entity(RSC_EntityVariant(_make_ball()))
    handle.end_modification_transaction()


def _respawn(viewer):
    """Remove the drained ball and drop a fresh one back into the launch lane."""
    handle = viewer.handle
    if handle is None:
        return
    handle.begin_modification_transaction()
    handle.remove_entity("ball")
    handle.add_entity(RSC_EntityVariant(_make_ball()))
    handle.end_modification_transaction()
    _state["drained"] = False
    # The fresh ball starts outside the drain, so the next loss needs a fresh entry.
    _state["in_drain"] = False


# ---------------------------------------------------------------------------
# Runtime callbacks
# ---------------------------------------------------------------------------
def control(world_name, time, controllers):
    """Drive the piston and flippers, and watch the drain region for a lost ball."""
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return

    for signal in controller.signals():
        if isinstance(signal, RS_WorldStructureVoxelSignal):
            hinge = signal.hinge_signals().hinge()
            if hinge.shape[0] > RIGHT_SIG:
                hinge[LEFT_SIG] = FLIP_L if _state["left"] else REST_L
                hinge[RIGHT_SIG] = FLIP_R if _state["right"] else REST_R
            prismatic = signal.prismatic_signals().prismatic()
            if prismatic.shape[0] > PLUNGER_SIG:
                prismatic[PLUNGER_SIG] = 1.0 if _state["fire"] else 0.0

    # The drain box is empty until a ball falls through the gap. Fire on the
    # rising edge only so a ball sitting in the box does not retrigger.
    inside = False
    for record in controller.records():
        if isinstance(record, RS_WorldIntersectRegionRecord):
            if record.memberships.size() >= 1:
                inside = True
    if inside and not _state["in_drain"]:
        _state["drained"] = True
    _state["in_drain"] = inside


def hud(painter, viewer):
    """Seed the ball, respawn it on a drain, read the keys, and draw the hint."""
    if viewer.handle is not None and not _state["populated"]:
        _populate(viewer.handle)
        _state["populated"] = True

    if _state["drained"]:
        _respawn(viewer)

    _state["fire"] = viewer.is_key_down(Qt.Key.Key_Space)
    _state["left"] = viewer.is_key_down(Qt.Key.Key_Left)
    _state["right"] = viewer.is_key_down(Qt.Key.Key_Right)

    hint_font = QtGui.QFont()
    hint_font.setPointSize(11)
    painter.setFont(hint_font)
    painter.setPen(QtGui.QColor(40, 40, 40))
    painter.drawText(16, 24, HINT)


def main():
    """Build the scene and launch it in the selected run mode."""
    _common.launch(
        build(),
        control,
        hud=hud,
        camera_focus=_to_world((W, L / 2, 0)),
    )


if __name__ == "__main__":
    main()
