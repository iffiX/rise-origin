# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Pause world: a ring of boxes drops while a hud button freezes the whole world.

What you'll learn:

* Pause and resume the live world through the world handle's :meth:`pause` /
  :meth:`resume` API: a paused world holds every body frozen in time.
* A paused world keeps sensing, so the free camera still orbits and renders fresh
  frames and the control callback keeps firing while physics time stands still.
* Drive a hud button from the keyboard, toggling pause on the space bar's rising
  edge, and read the current simulation time back onto the hud.

Run it::

    python examples/basic/pause.py
    RISE_MODE=record RISE_OUT=pause.mp4 python examples/basic/pause.py
"""

import math
import os
import sys

from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt

from rise_origin import rsc

# Put the examples root on the path so the sibling _common module is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

VOXEL = 0.03

BOX = 6  # box edge in voxels
RING_COUNT = 16  # boxes evenly spaced around the ring
RING_RADIUS = 0.7  # ring radius (m)
DROP_Z = 2.5  # height the boxes start at (m)

# Latest simulation time from the control callback and the space bar's last state,
# so the hud can show the time and toggle pause only on the key's rising edge.
_state = {"time": 0.0, "was_toggle_down": False}


def build():
    """Build the pause-demo world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    colors = _common.PALETTE_CYCLE
    materials = [
        _common.MaterialSpec(f"box{i}", color=color, kind="rigid")
        for i, color in enumerate(colors)
    ]

    half = BOX * VOXEL / 2.0
    structures = []
    for i in range(RING_COUNT):
        angle = 2.0 * math.pi * i / RING_COUNT
        structures.append(
            rsc.entity.StructureVoxel(
                _box_name(i),
                origin_position=(
                    RING_RADIUS * math.cos(angle),
                    RING_RADIUS * math.sin(angle),
                    DROP_Z,
                ),
                material_references=[f"box{i % len(colors)}"],
                bodies=[
                    rsc.entity.StructureVoxelBody.box(
                        BOX,
                        BOX,
                        BOX,
                        voxel_size=VOXEL,
                        relative_origin_position=(-half, -half, 0.0),
                    )
                ],
            )
        )

    return _common.make_world("pause", gravity=-0.1,materials=materials, structures=structures)


def control(time, controllers):
    """Record the simulation time so the hud can display it."""
    _state["time"] = time


def hud(painter, viewer):
    """Toggle pause on the space bar and draw the pause button and the clock."""
    is_toggle_down = viewer.is_key_down(Qt.Key.Key_Space)
    if is_toggle_down and not _state["was_toggle_down"] and viewer.handle is not None:
        if viewer.handle.is_paused():
            viewer.handle.resume()
        else:
            viewer.handle.pause()
    _state["was_toggle_down"] = is_toggle_down

    is_paused = viewer.handle is not None and viewer.handle.is_paused()
    _draw_button(painter, is_paused)
    _draw_clock(painter, _state["time"], is_paused)


def main():
    """Build the world and launch it in the selected run mode."""
    _common.launch(
        build(),
        control,
        hud=hud,
        camera_position=(-2.6, -2.0, 2.4),
        camera_focus=(0.0, 0.0, 1.1),
    )


def _box_name(index):
    """Ring-order name of the box at the given index."""
    return f"box_{index:02d}"


def _draw_button(painter, is_paused):
    """Draw the pause button, green while running and red while paused."""
    rect = QtCore.QRect(16, 16, 150, 40)
    fill = QtGui.QColor(196, 64, 64) if is_paused else QtGui.QColor(66, 160, 84)
    painter.setPen(QtGui.QColor(20, 20, 20))
    painter.setBrush(fill)
    painter.drawRoundedRect(rect, 6, 6)

    font = QtGui.QFont()
    font.setPointSize(12)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QtGui.QColor(250, 250, 250))
    label = "Resume [Space]" if is_paused else "Pause [Space]"
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)


def _draw_clock(painter, time, is_paused):
    """Draw the current simulation time, noting when it is frozen."""
    font = QtGui.QFont()
    font.setPointSize(12)
    painter.setFont(font)
    painter.setPen(QtGui.QColor(40, 40, 40))
    suffix = "  (frozen)" if is_paused else ""
    painter.drawText(182, 42, f"t = {time:6.2f} s{suffix}")


if __name__ == "__main__":
    main()
