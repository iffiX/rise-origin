# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview the pebble bed: a heap, a bank, and a flat bed of stones.

The three beds run the tone dial from pale green-gray stone to a
balanced gray, and each is shaped by an occupancy grid rather than left
as an even slab: a heap mounded in the middle, a bank ridged along one
side, and a thin flat bed. The preview injects a default Hosek-Wilkie
sky and a grid floor, so the scene needs nothing but the beds. Run it
directly for the interactive viewer (paused on start, close the window
to leave); pass ``--headless`` to play a short headless run instead,
which is what the automated smoke uses.

    python examples/create/nature/rock/pebble_bed_preview.py [--headless]
"""

import shutil
import sys
import tempfile

import numpy as np

from rise_origin import create as cr


def make_heap_mask() -> "np.ndarray":
    """Return a heaped occupancy, tall in the middle and thin at the rim."""
    counts = (16, 16, 10)
    lines = [np.linspace(-1.0, 1.0, counts[side]) for side in range(2)]
    across, along = np.meshgrid(*lines, indexing="ij")
    height = np.clip(1.0 - (across**2 + along**2), 0.0, 1.0) ** 0.55
    level = (np.arange(counts[2]) + 0.5) / counts[2]
    return height[:, :, None] > level[None, None, :]


def make_bank_mask() -> "np.ndarray":
    """Return a bank ridged along one side and thinning to the water."""
    counts = (18, 10, 8)
    across = np.linspace(0.0, 1.0, counts[1])
    ridge = np.clip(1.0 - ((across - 0.3) / 0.7) ** 2, 0.0, 1.0)
    sway = 0.3 * np.sin(np.linspace(0.0, 3.4, counts[0]))
    height = np.clip(ridge[None, :] + sway[:, None], 0.0, 1.0)
    level = (np.arange(counts[2]) + 0.5) / counts[2]
    return height[:, :, None] > level[None, None, :]


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="pebble_bed_preview_")
    editor = cr.Editor(
        "pebble_bed_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        heap = editor.create(
            "rise_origin.nature.rock.pebble_bed",
            {
                "width": 1.0,
                "depth": 1.0,
                "thickness": 0.32,
                "stone_size": 0.1,
                "size_mix": 0.5,
                "flatness": 0.25,
                "weathering": 0.6,
                "tone": 0.0,
                "moving_depth_share": 0.3,
                "occupancy": make_heap_mask(),
                "seed": 5,
            },
        )
        bank = editor.create(
            "rise_origin.nature.rock.pebble_bed",
            {
                "width": 1.6,
                "depth": 0.9,
                "thickness": 0.22,
                "stone_size": 0.13,
                "size_mix": 0.25,
                "flatness": 0.15,
                "weathering": 0.7,
                "tone": 0.4,
                "moving_depth_share": 0.35,
                "occupancy": make_bank_mask(),
                "seed": 11,
            },
        )
        flat = editor.create(
            "rise_origin.nature.rock.pebble_bed",
            {
                "width": 1.2,
                "depth": 1.2,
                "thickness": 0.14,
                "stone_size": 0.11,
                "size_mix": 0.7,
                "flatness": 0.5,
                "weathering": 0.4,
                "tone": 0.75,
                "moving_depth_share": 0.2,
                "seed": 17,
            },
        )
        editor.insert(heap, cr.Pose((0.0, 0.0, 0.24)))
        editor.insert(bank, cr.Pose((2.4, 0.0, 0.20)))
        editor.insert(flat, cr.Pose((-2.2, 0.0, 0.15)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
