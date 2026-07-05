# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Sphere intersect region linked to a rigid body: steer a ball into a fixed cube's zone.

What you'll learn:

* Anchor a sphere intersect region to a structure's rigid body with
  :class:`rsc.record.IntersectRegionSphereRigidBodyLinked`. The region rides the
  body's live pose, so a region on a moving body follows it every record step.
* Read the region record on the controller side: its membership lists the gid of
  every structure with a voxel inside the sphere, and, per member, the world-axes
  offset of that structure's closest voxel to the region center.
* Steer a rigid ball with a **force signal** (:class:`rsc.signal.Force`) written
  each control step, wired to the arrow keys through ``viewer.is_key_down``, the
  same pattern as ``vr2.py``.

The cube is a ``rigid_fixed`` material: a rigid body (so the region can link to
it) that is also anchored, so it hangs fixed in the air and never falls or gets
knocked away. Gravity is off, so the ball floats in the cube's plane and flies
with momentum, like an air-hockey puck.

Run it and fly the ball into the cube with the arrow keys::

    python examples/basic/intersection.py
"""

import os
import sys

import numpy as np
from PySide6 import QtGui
from PySide6.QtCore import Qt

from rise_origin import (
    RS_NULL_INDEX,
    RS_WorldForceSignal,
    RS_WorldIntersectRegionRecord,
    RS_WorldTaskDataCreated,
    RSC_EntityVariant,
    RSC_RecordVariant,
    RSC_SignalVariant,
    rsc,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

VOXEL = 0.05
CUBE_ORIGIN = (0.0, 0.0, 1.0)  # cube min corner; a fixed cube one meter up in the air
BALL_R = 3  # ball radius in voxels -> 0.15 m
BALL_START = (2.6, 0.0, 1.025)  # start outside the region, level with the cube center
REGION_RADIUS = 2.0  # sphere radius around the cube center (m)
PUSH = 4.0  # steering force (N)

CONTROLLER = "intersection#watch"
REGION = "cube_zone"
CUBE_COLOR = (255, 45, 35)
BALL_COLOR = (60, 140, 255)

HINT = "Fly the ball into the cube with the UP / DOWN / LEFT / RIGHT arrow keys"

# Current push direction (XY), set by the hud from held keys and read by control.
_push = [0.0, 0.0]

# Shared readout: written by control from the region record, drawn by the hud.
_state = {"populated": False, "ball_gid": RS_NULL_INDEX, "inside": False, "distance": None}


def build():
    """Build the world: a fixed cube in the air with a sphere region and a watcher."""
    materials = [
        _common.MaterialSpec("cube", color=CUBE_COLOR, kind="rigid_fixed"),
        _common.MaterialSpec("ball", color=BALL_COLOR, kind="rigid"),
    ]
    return _common.make_world(
        "intersection",
        materials=materials,
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
        gravity=0.0,
    )


def control(time, controllers):
    """Apply the arrow-key push to the ball and read the region membership.

    The region lists the gid of every structure inside it; the ball is inside the
    moment its gid appears. For that member the record also carries the closest
    voxel's offset from the region center, whose length is the ball's distance to
    the cube center.
    """
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return

    for signal in controller.signals():
        if isinstance(signal, RS_WorldForceSignal):
            force = signal.force_signals().force()  # shape (1, 3): one rigid-body force
            if force.shape[0] >= 1:
                force[0] = (float(_push[0]), float(_push[1]), 0.0)

    ball_gid = _state["ball_gid"]
    inside = False
    distance = None
    for record in controller.records():
        if not isinstance(record, RS_WorldIntersectRegionRecord):
            continue
        gids = record.memberships.structure_gid()
        rows = np.where(gids == ball_gid)[0]
        if ball_gid != RS_NULL_INDEX and rows.size > 0:
            inside = True
            offset = record.memberships.closest_relative_position()[rows[0]]
            distance = float(np.linalg.norm(offset))
    _state["inside"] = inside
    _state["distance"] = distance


def hud(painter, viewer):
    """Seed the scene on the first frame, read the keys, and draw the readout."""
    if viewer.handle is not None and not _state["populated"]:
        _populate(viewer.handle)
        _state["populated"] = True

    _push[0] = (PUSH if viewer.is_key_down(Qt.Key.Key_Right) else 0.0) - (
        PUSH if viewer.is_key_down(Qt.Key.Key_Left) else 0.0
    )
    _push[1] = (PUSH if viewer.is_key_down(Qt.Key.Key_Up) else 0.0) - (
        PUSH if viewer.is_key_down(Qt.Key.Key_Down) else 0.0
    )

    font = QtGui.QFont()
    font.setPointSize(11)
    painter.setFont(font)
    painter.setPen(QtGui.QColor(40, 40, 40))
    painter.drawText(16, 24, HINT)

    status_font = QtGui.QFont()
    status_font.setPointSize(16)
    status_font.setBold(True)
    painter.setFont(status_font)
    if _state["inside"]:
        painter.setPen(QtGui.QColor(40, 180, 70))
        painter.drawText(16, 56, f"INSIDE  -  distance to cube center: {_state['distance']:.3f} m")
    else:
        painter.setPen(QtGui.QColor(180, 60, 40))
        painter.drawText(16, 56, "OUTSIDE the region")


def _populate(handle):
    """Add the fixed cube, its sphere region, the ball, and the ball's push signal.

    The region resolves against the cube's rigid body, so the cube is added first.
    The cube is one voxel with its grid origin at the structure origin, so its
    voxel center sits half a voxel along each axis; the region anchor matches that
    offset to place the sphere at the cube center rather than the grid corner.
    """
    half_voxel = 0.5 * VOXEL

    cube = rsc.entity.StructureVoxel(
        "cube",
        origin_position=CUBE_ORIGIN,
        material_references=["cube"],
        bodies=[rsc.entity.StructureVoxelBody.box(1, 1, 1, voxel_size=VOXEL)],
    )
    region = rsc.record.IntersectRegionSphereRigidBodyLinked(
        REGION,
        structure_name="cube",
        body_sid=0,
        segment_bid=0,
        relative_anchor=(half_voxel, half_voxel, half_voxel),
        radius=REGION_RADIUS,
        target_controller_reference=CONTROLLER,
    )
    ball = rsc.entity.StructureVoxel(
        "ball",
        origin_position=BALL_START,
        material_references=["ball"],
        bodies=[
            rsc.entity.StructureVoxelBody.sphere(
                BALL_R,
                voxel_size=VOXEL,
                relative_origin_position=(-BALL_R * VOXEL, -BALL_R * VOXEL, -BALL_R * VOXEL),
            )
        ],
    )
    push = rsc.signal.Force(
        "push",
        structure_name="ball",
        source_controller_reference=CONTROLLER,
        body_sid=0,
        segment_bid=0,
    )

    handle.begin_modification_transaction()
    handle.add_entity(RSC_EntityVariant(cube))
    handle.add_record(RSC_RecordVariant(region))
    ball_task = handle.add_entity(RSC_EntityVariant(ball))
    handle.add_signal(RSC_SignalVariant(push))
    handle.end_modification_transaction()

    created = ball_task.get().value
    _state["ball_gid"] = (
        created.gid if isinstance(created, RS_WorldTaskDataCreated) else RS_NULL_INDEX
    )


def main():
    """Build the world and launch it, camera aimed at the floating cube."""
    _common.launch(
        build(),
        control,
        hud=hud,
        camera_position=(1.4, -1.4, 1.5),
        camera_focus=(0.0, 0.0, 1.0),
    )


if __name__ == "__main__":
    main()
