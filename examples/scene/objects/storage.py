# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Storage objects: locker, crate stack, floor grate.

Fine 0.02 m grid.
"""

from __future__ import annotations

import numpy as np

from .. import materials as M
from ..stamp import Stamp, stamp_cached


# ============================================================================
# Industrial vent-louvered locker
# ============================================================================


@stamp_cached(8)
def build_locker(rng: np.random.Generator) -> Stamp:
    """A 1.8 m vent-louvered locker with kick plate recess, slanted louvers,
    hinge knuckles, recessed handle, and a small number plate.
    """
    s = Stamp((15, 15, 90))
    # Body shell (1-voxel walls).
    s.fill((slice(None), slice(None), slice(None)), M.METAL_DARK)
    s.clear((slice(1, 14), slice(1, 14), slice(1, 89)))

    # Recessed kick plate at z=0..6: front face stepped back 1 voxel.
    for z in range(0, 6):
        s.clear((slice(None), 0, z))
        s.fill((slice(None), 1, z), M.METAL_DARK)

    # Door panel inset 1 voxel from front (at y=1) covering z=8..82.
    door_z0, door_z1 = 8, 82
    is_open = rng.random() < 0.30
    if not is_open:
        s.fill((slice(1, 14), 1, slice(door_z0, door_z1)), M.METAL_HOUSING)

        # Slanted louvered vents at z=70..78: 3 rows of slanted slits.
        for row in range(3):
            z_row = 70 + row * 3
            for x in range(2, 13):
                # Slant: shift by row.
                if (x + row) % 2 == 0:
                    s.clear((x, 1, z_row))

        # Hinge knuckles on left edge (3 protrusions).
        for hz in (15, 45, 75):
            s.fill((0, slice(0, 2), slice(hz, hz + 3)), M.METAL_HOUSING)

        # Recessed handle at z=42.
        s.clear((slice(7, 10), 1, 42))

        # Number plate at z=82.
        s.fill((slice(7, 10), 1, 82), M.METAL_HOUSING)
        # Emissive number digit.
        plate_em = (M.EM_CYAN, M.EM_AMBER, M.EM_RED)[int(rng.integers(0, 3))]
        s.set(8, 0, 82, plate_em)
    else:
        # Open locker: 1 hinge knuckle visible "open", a few contents.
        s.fill((0, slice(0, 2), slice(15, 18)), M.METAL_HOUSING)
        for _ in range(5):
            cx = int(rng.integers(3, 12))
            cy = int(rng.integers(2, 12))
            cz = int(rng.integers(10, 75))
            s.set(cx, cy, cz, M.RUBBER_BLACK)
        # Sometimes an emissive indicator inside.
        if rng.random() < 0.5:
            s.set(7, 7, 40, M.EM_AMBER)

    # 10% dented (knock a 1-voxel region inward).
    if rng.random() < 0.10:
        dent_z = int(rng.integers(20, 60))
        s.clear((slice(5, 8), 1, slice(dent_z, dent_z + 2)))

    return s


# ============================================================================
# Crate (single)
# ============================================================================


@stamp_cached(8)
def build_crate(rng: np.random.Generator) -> Stamp:
    """A single crate with edge banding, a colour label, and possible damage."""
    s = Stamp((25, 25, 20))
    # Outer shell.
    s.fill((slice(None), slice(None), slice(None)), M.METAL_DARK)
    s.clear((slice(1, 24), slice(1, 24), slice(1, 19)))
    # Edge banding (top, bottom, 4 vertical corners — darker material).
    # Top and bottom rim
    s.fill((slice(None), slice(None), 0), M.PLASTIC_DARK)
    s.fill((slice(None), slice(None), 19), M.PLASTIC_DARK)
    # Vertical corner edges
    for cx, cy in ((0, 0), (24, 0), (0, 24), (24, 24)):
        s.fill((cx, cy, slice(None)), M.PLASTIC_DARK)
    # Label colour square on front face.
    label_mat = (M.PAINTED_YELLOW, M.PAINTED_RED, M.PAINTED_CYAN)[int(rng.integers(0, 3))]
    s.fill((slice(10, 14), 0, slice(12, 15)), label_mat)
    # Rope-handle notch (1-voxel slot on each side).
    s.clear((slice(11, 14), 0, 17))
    s.clear((slice(11, 14), 24, 17))
    # 10% damage: knock a corner.
    if rng.random() < 0.10:
        s.clear((slice(22, 25), slice(22, 25), slice(15, 20)))
    return s


@stamp_cached(8)
def build_crate_stack(rng: np.random.Generator, count: int = 3) -> Stamp:
    """A stack of 1..3 crates with random horizontal offsets."""
    count = max(1, min(3, count))
    # 25x25 footprint, height = count * 20
    s = Stamp((30, 30, count * 22))
    for i in range(count):
        ox = int(rng.integers(-2, 3))
        oy = int(rng.integers(-2, 3))
        crate = build_crate(rng)
        crate.paste_into(
            s.materials,
            np.zeros(s.shape, dtype=np.int64),  # discarded segments
            3 + ox, 3 + oy, i * 22,
            M.SEG_OF,
            mode="overlay",
        )
    return s


# ============================================================================
# Floor grate
# ============================================================================


@stamp_cached(8)
def build_floor_grate(rng: np.random.Generator) -> Stamp:
    """A floor grate: 1-voxel-thick metal frame + checkerboard hollow grid."""
    s = Stamp((20, 20, 2))
    s.fill((slice(None), slice(None), 0), M.METAL_DARK)
    # Checkerboard hollow pattern: alternate 2-voxel cells.
    for cx in range(2, 18, 4):
        for cy in range(2, 18, 4):
            s.clear((slice(cx, cx + 2), slice(cy, cy + 2), 0))
    # Outer rim is solid (the frame).
    s.fill((slice(0, 1), slice(None), 0), M.METAL_DARK)
    s.fill((slice(19, 20), slice(None), 0), M.METAL_DARK)
    s.fill((slice(None), slice(0, 1), 0), M.METAL_DARK)
    s.fill((slice(None), slice(19, 20), 0), M.METAL_DARK)
    return s
