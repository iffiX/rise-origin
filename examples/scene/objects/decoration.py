# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Decoration / damage: plant in pot, debris pile.

Knocked-out ceiling sections are NOT a stamp — they're applied directly to the
walls_structure World via :meth:`World.knock_out` from the damaged-biome
furnisher in :mod:`scene.biomes`.

Fine 0.02 m grid.
"""

from __future__ import annotations

import numpy as np

from .. import materials as M
from ..stamp import Stamp


# ============================================================================
# Potted plant
# ============================================================================


def build_plant_in_pot(rng: np.random.Generator) -> Stamp:
    """A potted plant with 2-3 distinct stems rising from the pot, each topped
    with a small leaf cluster.
    """
    s = Stamp((10, 10, 15))
    # Pot: 10x10x6 with 1-voxel-darker rim.
    s.fill((slice(None), slice(None), slice(0, 6)), M.PLASTIC_DARK)
    # Hollow inside.
    s.clear((slice(1, 9), slice(1, 9), slice(1, 6)))
    # Pot rim.
    s.fill((slice(None), slice(None), 5), M.METAL_DARK)
    # Soil layer.
    s.fill((slice(1, 9), slice(1, 9), 1), M.RUBBER_BLACK)

    # 2-3 stems from random points on the soil.
    n_stems = int(rng.integers(2, 4))
    for _ in range(n_stems):
        bx = int(rng.integers(3, 8))
        by = int(rng.integers(3, 8))
        # Stem: 1-voxel column rising 5-8 voxels.
        stem_h = int(rng.integers(5, 9))
        for z in range(2, 2 + stem_h):
            s.set(bx, by, z, M.PAINTED_CYAN)
        # Leaf cluster at top via small random walk.
        leaf_n = int(rng.integers(6, 12))
        lx, ly, lz = bx, by, 2 + stem_h
        for _ in range(leaf_n):
            lx += int(rng.integers(-1, 2))
            ly += int(rng.integers(-1, 2))
            lz += int(rng.integers(0, 2))
            if 1 <= lx <= 8 and 1 <= ly <= 8 and lz < 15:
                s.set(lx, ly, lz, M.PAINTED_CYAN)
    return s


# ============================================================================
# Debris pile (damaged biome)
# ============================================================================


def build_debris_pile(rng: np.random.Generator) -> Stamp:
    """A pile of mixed-shape debris: random small voxels, plus a few recognisable
    broken-subobject chunks (a bent pipe segment, a torn door panel piece),
    plus 1-2 emissive sparking voxels.
    """
    s = Stamp((50, 50, 8))

    # Random small voxels (Poisson-disc-ish).
    n_small = int(rng.integers(80, 180))
    for _ in range(n_small):
        x = int(rng.integers(0, 50))
        y = int(rng.integers(0, 50))
        z = int(rng.integers(0, 3))
        s.set(x, y, z, M.METAL_DARK)

    # 3-6 larger chunks.
    n_chunks = int(rng.integers(3, 7))
    for _ in range(n_chunks):
        cx = int(rng.integers(2, 47))
        cy = int(rng.integers(2, 47))
        cz = int(rng.integers(0, 2))
        cdx = int(rng.integers(2, 4))
        cdy = int(rng.integers(2, 4))
        cdz = int(rng.integers(1, 3))
        chunk_mat = (M.METAL_HOUSING, M.PLASTIC_DARK, M.METAL_DARK)[int(rng.integers(0, 3))]
        s.fill((slice(cx, cx + cdx),
                slice(cy, cy + cdy),
                slice(cz, cz + cdz)), chunk_mat)

    # 1 bent pipe segment (a few connected voxels in a bent path).
    if rng.random() < 0.7:
        bx, by = int(rng.integers(10, 40)), int(rng.integers(10, 40))
        for d in range(8):
            s.set(bx + d, by, 1, M.METAL_DARK)
            s.set(bx + d, by, 2, M.METAL_DARK)
        # Bend.
        for d in range(5):
            s.set(bx + 7, by + d, 1, M.METAL_DARK)

    # 1-2 sparking emissive voxels at random higher z.
    for _ in range(int(rng.integers(1, 3))):
        sx = int(rng.integers(5, 45))
        sy = int(rng.integers(5, 45))
        s.set(sx, sy, 6, M.EM_WHITE)

    return s
