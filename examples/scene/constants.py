# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Grid sizes and world extent for the two cooperating voxel resolutions.

The scene uses **two voxel grids**:

* a coarse 0.10 m grid for walls/floor/ceiling (the architectural shell), and
* a fine 0.02 m grid for fixtures and furniture (5x finer detail).

The 5:1 ratio means every coarse-grid anchor is also a fine-grid anchor, so the
two structures share the same world origin without sub-voxel error.
"""

#: Coarse voxel size in meters (walls/floor/ceiling). Chunky like Minecraft stone.
VOXEL_W = 0.10

#: Fine voxel size in meters (fixtures and furniture). 5x finer than walls.
VOXEL_O = 0.02

#: Integer ratio between the two grids; must be exact for alignment.
WO_RATIO = int(round(VOXEL_W / VOXEL_O))
assert WO_RATIO == 5, "VOXEL_W must be exactly 5 * VOXEL_O for grid alignment"

# --- Cell footprint (in COARSE voxels) ------------------------------------

CELL_W = 40   # 4.0 m per cell side (40 coarse / 200 fine voxels)
CELL_H = 30   # 3.0 m ceiling height (30 coarse / 150 fine voxels)
WALL_T = 1    # 0.1 m wall thickness (1 coarse / 5 fine voxels)

# --- Cell grid ------------------------------------------------------------

CELLS_X = 11    # 44 m wide (per the design diagram)
CELLS_Y = 13    # 52 m deep (per the design diagram)

# --- Walls-structure total dimensions (coarse voxels) ---------------------

NX_W = CELLS_X * CELL_W + WALL_T * (CELLS_X + 1)   # 206 coarse voxels
NY_W = CELLS_Y * CELL_W + WALL_T * (CELLS_Y + 1)   # 165 coarse voxels
NZ_W = CELL_H + 2 * WALL_T                          # 32 coarse voxels

# --- World extent in meters (resolution-independent) ----------------------

EXTENT_X = NX_W * VOXEL_W
EXTENT_Y = NY_W * VOXEL_W
EXTENT_Z = NZ_W * VOXEL_W

# --- Doorway opening size (in COARSE voxels) ------------------------------

DOOR_W = 12   # 1.2 m wide opening (12 coarse / 60 fine)
DOOR_H = 20   # 2.0 m tall opening (20 coarse / 100 fine)
# Future Corp corporate glass door — the grand single entrance from the
# lobby into the central spine. Much wider than the industrial blast door.
CORPORATE_DOOR_W = 48   # 4.8 m wide opening (48 coarse / 240 fine) — spans
                        # parts of cx=4 and cx=6 walls in addition to all of
                        # cx=5; the strip-wall is one continuous run so the
                        # knock-out cuts cleanly across cell boundaries.
CORPORATE_DOOR_H = 24   # 2.4 m tall opening (24 coarse / 120 fine)

# --- Random seed for reproducible procedural builds ----------------------

SEED = 11


# ============================================================================
# Inspection toggles
# ----------------------------------------------------------------------------
# Set to ``True`` to make the corresponding part of the shell less obstructive
# so you can fly over / inspect the procedural content. These are render-only
# tweaks; they don't change object placement.
# ============================================================================

#: If True, the ceiling slab is omitted (open-roof inspection mode).
INSPECT_NO_ROOF = True

#: If True, internal cell-divider walls are dropped (perimeter walls stay).
#: Doorways become irrelevant because there are no walls to gate them.
INSPECT_NO_INTERNAL_WALLS = False


def cell_origin_w(cx: int, cy: int) -> tuple:
    """Return the coarse-voxel ``(x, y, z)`` origin of cell ``(cx, cy)``'s floor.

    Cells are 0-indexed on the grid. ``(0, 0, 0)`` is the bottom-left wall
    corner of the whole structure; cell ``(cx, cy)`` starts ``WALL_T`` voxels
    in from the wall on each axis, so cell ``(0, 0)``'s floor interior starts
    at coarse coords ``(WALL_T, WALL_T, WALL_T)``.
    """
    x = WALL_T + cx * (CELL_W + WALL_T)
    y = WALL_T + cy * (CELL_W + WALL_T)
    z = WALL_T
    return x, y, z


def cell_origin_o(cx: int, cy: int) -> tuple:
    """Coarse cell origin reprojected into FINE-voxel coordinates."""
    x, y, z = cell_origin_w(cx, cy)
    return x * WO_RATIO, y * WO_RATIO, z * WO_RATIO
