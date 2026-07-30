# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview the slate block: a broad ragged block beside a small boulder.

The preview injects a default Hosek-Wilkie sky and a grid floor, so the
scene needs nothing but the blocks. Run it directly for the interactive
viewer (paused on start, close the window to leave); pass ``--headless``
to play a short headless run instead, which is what the automated smoke
uses.

    python examples/create/nature/rock/slate_block_preview.py [--headless]
"""

import shutil
import sys
import tempfile

import numpy as np

from rise_origin import create as cr


def make_step_mask() -> "np.ndarray":
    """Return a stepped occupancy grid, a cliff shoulder cut from the box."""
    mask = np.ones((8, 4, 6), dtype=bool)
    mask[5:, :, 3:] = False
    mask[6:, :, 2:] = False
    mask[:2, :, 5:] = False
    return mask


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="slate_block_preview_")
    editor = cr.Editor(
        "slate_block_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        wall = editor.create(
            "rise_origin.nature.rock.slate_block",
            {
                "width": 3.0,
                "depth": 1.5,
                "height": 2.5,
                "tone": 0,
                "break_dir": (0.3, 1.0, 0.45),
                "layer_dir": (1.0, 0.3, 0.35),
                "raggedness": 0.55,
                "stain": 0.4,
                "seed": 5,
            },
        )
        boulder = editor.create(
            "rise_origin.nature.rock.slate_block",
            {
                "width": 1.0,
                "depth": 0.7,
                "height": 0.8,
                "break_dir": (0.6, 1.0, 0.3),
                "raggedness": 0.7,
                "stain": 0.25,
                "is_fixed": False,
                "seed": 6,
            },
        )
        carved = editor.create(
            "rise_origin.nature.rock.slate_block",
            {
                "width": 3.5,
                "depth": 1.2,
                "height": 2.2,
                "occupancy": make_step_mask(),
                "tone": 0,
                "break_dir": (0.3, 1.0, 0.45),
                "raggedness": 0.45,
                "stain": 0.35,
                "seed": 9,
            },
        )
        editor.insert(wall, cr.Pose((0.0, 0.0, 1.3)))
        editor.insert(carved, cr.Pose((-4.5, 0.0, 1.15)))
        editor.insert(boulder, cr.Pose((3.0, 0.0, 0.5)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
