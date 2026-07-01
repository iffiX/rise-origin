# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Transform signal: kick four cubes of different physics kinds up with a key.

What you'll learn:

* Drive a structure's pose directly with a transform signal
  (:class:`rsc.signal.Transform`) written each step. A transform carries a
  translation, a rotation, and the world pivot the rotation turns about. The
  engine applies it once to the whole structure, then resets it to identity.
* A transform sets pose, not momentum, so it is a one-shot kick rather than a
  sustained push: each key press hops the cube up a fixed distance and spins it
  about its own z axis once, then gravity takes over again.
* The same signal works on every physics kind. Red is a soft cube, blue a rigid
  cube, green a fixed cube, and yellow a rigid-fixed (anchored) cube; the
  transform re-poses all four the same way.

Run it and press 1 / 2 / 3 / 4 to kick each cube::

    python examples/basic/transform.py
"""

import math
import os
import sys

from PySide6 import QtGui
from PySide6.QtCore import Qt

from rise_origin import RS_WorldTransformSignal, rsc

# Put the examples root on the path so the sibling _common module is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

VOXEL = 0.03
CUBE = 8  # cube edge in voxels
GROUND_GAP = 0.05  # height the cubes start above the floor (m)
SPACING = 0.45  # gap between cube centers along x (m)

JUMP = 1  # one-shot upward hop per key press (m)
TURN = math.radians(30.0)  # one-shot turn about the cube's own z axis per key press (rad)

HINT = "Press 1 / 2 / 3 / 4 to kick the cubes  -  red soft, blue rigid, green fixed, yellow rigid-fixed"

# One cube per physics kind, laid out left to right.
CUBES = [
    {"name": "red", "color": (235, 64, 52), "kind": "soft", "key": Qt.Key.Key_1},
    {"name": "blue", "color": (52, 96, 235), "kind": "rigid", "key": Qt.Key.Key_2},
    {"name": "green", "color": (66, 196, 80), "kind": "fixed", "key": Qt.Key.Key_3},
    {"name": "yellow", "color": (240, 214, 56), "kind": "rigid_fixed", "key": Qt.Key.Key_4},
]

# Whether each cube's key is currently held, written by the hud and read by control.
_held = [False] * len(CUBES)

# Whether each cube's key was held last callback, so a hold kicks only once.
_was_held = [False] * len(CUBES)


def build():
    """Build the four-cube transform world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    half = CUBE * VOXEL / 2.0
    materials = []
    structures = []
    signals = []
    controllers = []
    for i, cube in enumerate(CUBES):
        materials.append(
            _common.MaterialSpec(cube["name"], color=cube["color"], kind=cube["kind"])
        )
        structures.append(
            rsc.entity.StructureVoxel(
                cube["name"],
                origin_position=(_cube_x(i), 0.0, GROUND_GAP),
                material_references=[cube["name"]],
                bodies=[
                    rsc.entity.StructureVoxelBody.box(
                        CUBE,
                        CUBE,
                        CUBE,
                        voxel_size=VOXEL,
                        relative_origin_position=(-half, -half, 0.0),
                    )
                ],
            )
        )
        controllers.append(rsc.controller.ControllerCallback(_controller_name(i)))
        signals.append(
            rsc.signal.Transform(
                _signal_name(i),
                structure_name=cube["name"],
                source_controller_reference=_controller_name(i),
            )
        )

    return _common.make_world(
        "transform",
        materials=materials,
        structures=structures,
        signals=signals,
        controllers=controllers,
    )


def control(time, controllers):
    """Kick each cube once on its key's rising edge, idle otherwise."""
    for i, cube in enumerate(CUBES):
        is_kick = _held[i] and not _was_held[i]
        _was_held[i] = _held[i]
        controller = _common.controller_by_name(controllers, _controller_name(i))
        if controller is None:
            continue
        for signal in controller.signals():
            if isinstance(signal, RS_WorldTransformSignal):
                _drive(signal, is_kick, _cube_x(i))


def hud(painter, viewer):
    """Read the number keys into the held state and draw the hint banner."""
    for i, cube in enumerate(CUBES):
        _held[i] = viewer.is_key_down(cube["key"])

    font = QtGui.QFont()
    font.setPointSize(11)
    painter.setFont(font)
    painter.setPen(QtGui.QColor(40, 40, 40))
    painter.drawText(16, 24, HINT)


def main():
    """Build the world and launch it in the selected run mode."""
    _common.launch(
        build(),
        control,
        hud=hud,
        camera_position=(0.0, -2.0, 1.2),
        camera_focus=(0.0, 0.0, 0.3),
    )


def _drive(signal, is_kick, center_x):
    """Set one transform signal to a single hop and turn, or to identity."""
    translation = signal.translation_signals().translation()  # shape (1, 3)
    rotation = signal.rotation_signals().rotation()  # shape (1, 4), quaternion xyzw
    pivot = signal.rotation_signals().pivot()  # shape (1, 3)
    if translation.shape[0] < 1:
        return

    if is_kick:
        half_angle = TURN / 2.0
        translation[0] = (0.0, 0.0, JUMP)
        rotation[0] = (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))
        # The cube only ever moves along z, so its own z axis stays at the start x, y.
        pivot[0] = (center_x, 0.0, 0.0)
    else:
        translation[0] = (0.0, 0.0, 0.0)
        rotation[0] = (0.0, 0.0, 0.0, 1.0)


def _cube_x(index):
    """World x of the cube at the given index, centered around the origin."""
    return (index - (len(CUBES) - 1) / 2.0) * SPACING


def _controller_name(index):
    """Name of the controller driving the cube at the given index."""
    return f"control_{CUBES[index]['name']}"


def _signal_name(index):
    """Name of the transform signal for the cube at the given index."""
    return f"transform_{CUBES[index]['name']}"


if __name__ == "__main__":
    main()
