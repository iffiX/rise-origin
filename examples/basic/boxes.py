# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Many falling boxes: a shower of rigid and soft blocks settling into a pile.

What you'll learn:

* Spawn many independent structures in one world, each a single-body
  :class:`rsc.entity.StructureVoxel` with no joints.
* Mix rigid and soft blocks (half and half) and watch them collide and settle.

Run it::

    python examples/basic/boxes.py
    RISE_MODE=record RISE_OUT=boxes.mp4 python examples/basic/boxes.py
"""

import os
import sys

import numpy as np

from rise_origin import rsc

# Put the examples root on the path so the sibling _common module is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

VOXEL = 0.01
N = 16  # number of boxes


def build():
    """Build the falling-boxes world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    rng = np.random.default_rng(0)
    structures = []
    for i in range(N):
        size = rng.integers(3, 8, size=3)  # random box, 3-7 voxels per side
        x, y = rng.uniform(-0.12, 0.12, size=2)  # random spot above the floor
        z = 0.15 + 0.06 * i  # staggered heights so they drop in turn
        material = "rigid" if i % 2 == 0 else "soft"  # half rigid, half soft

        structures.append(
            rsc.entity.StructureVoxel(
                f"box{i}",
                origin_position=(float(x), float(y), float(z)),
                material_references=[material],
                bodies=[
                    rsc.entity.StructureVoxelBody.box(
                        int(size[0]),
                        int(size[1]),
                        int(size[2]),
                        voxel_size=VOXEL,
                    )
                ],
            )
        )

    materials = [
        _common.MaterialSpec(
            "rigid", color=_common.PALETTE["blue_green"], kind="rigid"
        ),
        _common.MaterialSpec("soft", color=_common.PALETTE["amber"], kind="soft"),
    ]
    return _common.make_world("boxes", materials=materials, structures=structures)


def main():
    """Build the world and launch it in the selected run mode (no control)."""
    _common.launch(build(), None, camera_focus=(0.0, 0.0, 0.05))


if __name__ == "__main__":
    main()
