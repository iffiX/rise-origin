# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Freeze ring: 20 boxes drop while a freeze wave sweeps around the ring.

What you'll learn:

* Drive the live world from a control callback through the world handle's
  :meth:`freeze_entity` / :meth:`unfreeze_entity` API: a frozen entity holds its
  pose mid-air, an unfrozen one falls again.
* Address entities by name at runtime, here a ring of 20 boxes dropped from mid
  air.
* Sweep a frozen arc around the ring in order, so each box halts mid-fall as the
  arc reaches it and resumes once the arc passes, a wave travelling the ring.

Run it::

    python examples/basic/freeze.py
    RISE_MODE=record RISE_OUT=freeze.mp4 python examples/basic/freeze.py
"""

import math
import os
import sys

from rise_origin import rsc
from rise_origin.runtime.viewer import Viewer

# Put the examples root on the path so the sibling _common module is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

VOXEL = 0.03

BOX = 6  # box edge in voxels
RING_COUNT = 20  # boxes evenly spaced around the ring
RING_RADIUS = 0.7  # ring radius (m)
DROP_Z = 2.5  # height the boxes start at (m)

# A frozen arc covering this fraction of the ring sweeps around once per period.
SWEEP_PERIOD = 1.0  # seconds for the frozen arc to travel the ring once
FROZEN_FRACTION = 0.5  # fraction of the ring frozen at any moment

# Holds the running viewer so the control callback can reach its world handle.
_viewer = None

# Tracks each box's last applied frozen state to issue commands only on change.
_is_frozen = [False] * RING_COUNT


def build():
    """Build the freeze-ring world.

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

    return _common.make_world("freeze", materials=materials, structures=structures)


def control(world_name, time, controllers):
    """Sweep the frozen arc around the ring, toggling each box as it crosses."""
    if _viewer is None or _viewer.handle is None:
        return
    head = (time / SWEEP_PERIOD) % 1.0
    for i in range(RING_COUNT):
        phase = ((i / RING_COUNT) - head) % 1.0
        target = phase < FROZEN_FRACTION
        if target == _is_frozen[i]:
            continue
        if target:
            _viewer.handle.freeze_entity(_box_name(i))
        else:
            _viewer.handle.unfreeze_entity(_box_name(i))
        _is_frozen[i] = target


def main():
    """Build the world and launch it in the selected run mode."""
    global _viewer
    device = os.environ.get("RISE_DEVICE", "cuda:0") or None
    mode = os.environ.get("RISE_MODE", "interactive")
    duration = float(os.environ.get("RISE_DURATION", "8"))

    _viewer = Viewer(
        instance=(device, build()),
        callback=control,
        camera_position=(-2.6, -2.0, 2.4),
        camera_focus=(0.0, 0.0, 1.1),
    )
    if mode == "record":
        _viewer.run_record(os.environ.get("RISE_OUT", "out.mp4"), duration=duration)
    elif mode == "headless":
        _viewer.run_headless(duration=duration)
    else:
        _viewer.run_interactive()
    _viewer.wait()


def _box_name(index):
    """Ring-order name of the box at the given index."""
    return f"box_{index:02d}"


if __name__ == "__main__":
    main()
