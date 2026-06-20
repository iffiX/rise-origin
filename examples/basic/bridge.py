# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Broken bridge: a heavy rigid block drops onto a long soft span and snaps it.

What you'll learn:

* Build one body from many materials with
  :meth:`rsc.entity.StructureVoxelBody.from_grid`: two fixed piers that float in
  the air and a soft deck slung between them, quartered into four colors so the
  bending shows.
* Give a soft material a finite ``fail_stress`` so its links break under load:
  the deck holds its own weight but tears apart where the block lands.
* Drop a separate rigid block from above and watch the span fail and fall.

Layout along the bridge's long axis: pier, soft deck (four colors), pier.

Run it::

    python examples/basic/bridge.py
    RISE_MODE=record RISE_OUT=bridge.mp4 python examples/basic/bridge.py
"""

import os
import sys

import numpy as np

from rise_origin import RS_NULL_INDEX, rsc

# Put the examples root on the path so the sibling _common module is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

VOXEL = 0.02

PIER_X = 5  # length of each fixed pier in voxels
SPAN_X = 72  # length of the soft deck in voxels (divisible by 4)
WIDTH = 12  # bridge width in voxels
THICK = 4  # deck thickness in voxels

DECK_Z = 0.70  # height the deck floats at (m)

BLOCK = 16  # rigid block edge in voxels
DROP = 0.25  # gap between the block and the deck at release (m)

# A finite failure stress lets the deck hold itself but tear under the block.
DECK_MODULUS = 2.0e5
DECK_FAIL_STRESS = 1.0e5

QUAD_COLORS = [
    _common.PALETTE["sky_blue"],
    _common.PALETTE["blue_green"],
    _common.PALETTE["amber"],
    _common.PALETTE["tiger_orange"],
]
PIER_COLOR = _common.PALETTE["deep_blue"]
BLOCK_COLOR = (150, 150, 150)

# Material reference order for the bridge structure; per-voxel ids index into it.
MATERIAL_REFERENCES = ["pier", "soft0", "soft1", "soft2", "soft3"]


def build():
    """Build the broken-bridge world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec("pier", color=PIER_COLOR, kind="fixed"),
        *(
            _common.MaterialSpec(
                f"soft{i}",
                color=color,
                kind="soft",
                elastic_modulus=DECK_MODULUS,
                fail_stress=DECK_FAIL_STRESS,
            )
            for i, color in enumerate(QUAD_COLORS)
        ),
        _common.MaterialSpec("block", color=BLOCK_COLOR, kind="rigid"),
    ]

    total_x = PIER_X + SPAN_X + PIER_X
    mat = np.full((total_x, WIDTH, THICK), RS_NULL_INDEX, dtype=np.int64)

    # Fixed piers at each end float in the air and anchor the deck.
    mat[:PIER_X, :, :] = 0
    mat[total_x - PIER_X :, :, :] = 0

    # Soft deck between the piers, quartered along its length into four colors.
    quarter = SPAN_X // 4
    for i in range(4):
        lo = PIER_X + i * quarter
        hi = PIER_X + (i + 1) * quarter
        mat[lo:hi, :, :] = i + 1

    # Each material region is its own segment, so segment id tracks material id.
    seg = mat.copy()

    deck_body = rsc.entity.StructureVoxelBody.from_grid(
        mat != RS_NULL_INDEX,
        material_grid=mat,
        segment_grid=seg,
        voxel_size=VOXEL,
        relative_origin_position=(
            -total_x * VOXEL / 2.0,
            -WIDTH * VOXEL / 2.0,
            0.0,
        ),
    )

    bridge = rsc.entity.StructureVoxel(
        "bridge",
        origin_position=(0.0, 0.0, DECK_Z),
        material_references=MATERIAL_REFERENCES,
        bodies=[deck_body],
    )

    block = rsc.entity.StructureVoxel(
        "block",
        origin_position=(0.0, 0.0, DECK_Z + THICK * VOXEL + DROP),
        material_references=["block"],
        bodies=[
            rsc.entity.StructureVoxelBody.box(
                BLOCK,
                BLOCK,
                BLOCK,
                voxel_size=VOXEL,
                relative_origin_position=(
                    -BLOCK * VOXEL / 2.0,
                    -BLOCK * VOXEL / 2.0,
                    0.0,
                ),
            )
        ],
    )

    return _common.make_world("bridge", materials=materials, structures=[bridge, block])


def main():
    """Build the world and launch it in the selected run mode (no control)."""
    _common.launch(
        build(),
        None,
        camera_position=(-1.5, -1.1, 1.2),
        camera_focus=(0.0, 0.0, DECK_Z),
    )


if __name__ == "__main__":
    main()
