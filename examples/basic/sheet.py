# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Soft sheet on balls: a 50x50 cloth draping over a bed of spheres.

What you'll learn:

* Build a large thin soft body (``50 x 50 x 1`` voxels) in one call and let it
  deform under gravity.
* Pin obstacles in place with the ``fixed`` material so the sheet drapes over
  them instead of pushing them away.
* Appreciate the soft-body solver: a single sheet conforms to the spheres,
  sagging in the gaps between them.

Run it::

    python examples/basic/sheet.py
    RISE_MODE=record RISE_OUT=sheet.mp4 python examples/basic/sheet.py
"""

import os
import sys

from rise_origin import rsc

# Put the examples root on the path so the sibling _common module is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

VOXEL = 0.01

SHEET = (50, 50, 1)  # cloth in voxels
SHEET_Z = 0.20  # drop height of the sheet (m)

BALL_RADIUS = 5  # voxels
BALL_GRID = 3  # balls per side
BALL_SPACING = 0.16  # spacing between balls (m)

SHEET_COLOR = _common.PALETTE["blue_green"]
BALL_COLOR = _common.PALETTE["amber"]


def build():
    """Build the sheet-on-balls world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec(
            "cloth", color=SHEET_COLOR, kind="soft", elastic_modulus=3e4
        ),
        _common.MaterialSpec("ball", color=BALL_COLOR, kind="fixed"),
    ]

    structures = []

    # A bed of pinned spheres for the sheet to drape over.
    span = (BALL_GRID - 1) * BALL_SPACING
    radius = BALL_RADIUS * VOXEL
    for ix in range(BALL_GRID):
        for iy in range(BALL_GRID):
            x = ix * BALL_SPACING - span / 2.0
            y = iy * BALL_SPACING - span / 2.0
            structures.append(
                rsc.entity.StructureVoxel(
                    f"ball_{ix}_{iy}",
                    is_fixed=True,
                    material_references=["ball"],
                    bodies=[
                        rsc.entity.StructureVoxelBody.sphere(
                            BALL_RADIUS,
                            voxel_size=VOXEL,
                            relative_origin_position=(x - radius, y - radius, 0.04),
                        )
                    ],
                )
            )

    # One big soft sheet centred above the balls.
    structures.append(
        rsc.entity.StructureVoxel(
            "sheet",
            material_references=["cloth"],
            bodies=[
                rsc.entity.StructureVoxelBody.box(
                    SHEET[0],
                    SHEET[1],
                    SHEET[2],
                    voxel_size=VOXEL,
                    relative_origin_position=(
                        -SHEET[0] * VOXEL / 2.0,
                        -SHEET[1] * VOXEL / 2.0,
                        SHEET_Z,
                    ),
                )
            ],
        )
    )

    return _common.make_world("sheet", materials=materials, structures=structures)


def main():
    """Build the world and launch it in the selected run mode (no control)."""
    _common.launch(build(), None, camera_focus=(0.0, 0.0, 0.08))


if __name__ == "__main__":
    main()
