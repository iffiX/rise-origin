# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Architecture detail: wall vent, pipe run, cable bundle, conduit.

Fine 0.02 m grid.
"""

from __future__ import annotations

import numpy as np

from .. import materials as M
from ..stamp import Stamp, stamp_cached, octagon_offsets, ring_offsets


# ============================================================================
# Wall vent grille (angled-louver style, matching the locker vents)
# ============================================================================


@stamp_cached(8)
def build_wall_vent(rng: np.random.Generator) -> Stamp:
    """A wall vent grille with angled louvers and a darker rim."""
    s = Stamp((12, 2, 17))
    # Outer frame.
    s.fill((slice(None), slice(0, 2), slice(None)), M.METAL_DARK)
    # Hollow centre.
    s.clear((slice(1, 11), slice(1, 2), slice(1, 16)))
    # Angled louvers (4 slats with a 1-voxel slant offset).
    for slat_i in range(4):
        z = 3 + slat_i * 3
        for x in range(1, 11):
            if (x + slat_i) % 2 == 0:
                s.set(x, 1, z, M.METAL_HOUSING)
    # 4 corner bolt-dots.
    for cx, cz in ((1, 1), (10, 1), (1, 15), (10, 15)):
        s.set(cx, 0, cz, M.METAL_HOUSING)
    return s


# ============================================================================
# Pipe run with flanges, insulation, gauge, and wheel valve
# ============================================================================


@stamp_cached(8)
def build_pipe_run(
    rng: np.random.Generator,
    *,
    length: int,
    axis: str = "x",
    colour: str = M.METAL_HOUSING,
    add_valve: bool = False,
    add_gauge: bool = False,
) -> Stamp:
    """A pipe with octagonal cross-section running ``length`` voxels along ``axis``.

    The pipe gets:
      * a flanged collar every 60 voxels,
      * one optional insulation segment of 30 voxels (off-white wrap),
      * one optional pressure gauge protruding from the side (``add_gauge``),
      * one optional wheel valve at a random position (``add_valve``).
    """
    axis = axis.lower()
    if axis == "x":
        s = Stamp((length, 7, 7))
        long_dim = length
    elif axis == "y":
        s = Stamp((7, length, 7))
        long_dim = length
    else:
        s = Stamp((7, 7, length))
        long_dim = length

    base_off = octagon_offsets(3)            # 7x7 octagon
    flange_off = octagon_offsets(5)          # 11x11 collar
    ins_off = octagon_offsets(4)             # 9x9 insulation

    # --- Insulation segment (random 30-voxel stretch in mid-run) ---
    ins_start = -1
    ins_end = -1
    if long_dim > 50:
        ins_start = int(rng.integers(10, long_dim - 40))
        ins_end = ins_start + 30

    # --- Wheel valve position ---
    valve_pos = -1
    if add_valve and long_dim > 30:
        valve_pos = int(rng.integers(15, long_dim - 15))

    # --- Gauge position ---
    gauge_pos = -1
    if add_gauge and long_dim > 30:
        gauge_pos = int(rng.integers(10, long_dim - 10))

    cx = 3  # cross-section centre (in the 7-axis)

    for u in range(long_dim):
        # Default base cross-section
        offs = base_off
        mat = colour
        in_insulation = ins_start <= u < ins_end
        # Flange every 60 voxels (4-voxel-long collar)
        in_flange = ((u % 60) < 4) and u > 0
        if in_flange:
            offs = flange_off
            mat = M.METAL_DARK
        elif in_insulation:
            offs = ins_off
            mat = M.METAL_INSULATION
            # Thin metal banding at the start and end of insulation.
            if u == ins_start or u == ins_end - 1:
                offs = flange_off
                mat = M.METAL_DARK

        for dx, dy in offs:
            if axis == "x":
                xx, yy, zz = u, cx + dx, cx + dy
            elif axis == "y":
                xx, yy, zz = cx + dx, u, cx + dy
            else:
                xx, yy, zz = cx + dx, cx + dy, u
            s.set(xx, yy, zz, mat)

    # --- Wheel valve at valve_pos ---
    if valve_pos >= 0:
        # Pipe expands to a 9x9 bulge over 6 voxels centred on valve_pos.
        for u_off in range(-3, 4):
            u = valve_pos + u_off
            if u < 0 or u >= long_dim:
                continue
            for dx, dy in octagon_offsets(4):
                if axis == "x":
                    xx, yy, zz = u, cx + dx, cx + dy
                elif axis == "y":
                    xx, yy, zz = cx + dx, u, cx + dy
                else:
                    xx, yy, zz = cx + dx, cx + dy, u
                s.set(xx, yy, zz, M.METAL_DARK)
        # Wheel stem and ring on perpendicular axis (pick z+ side).
        for r_off in ring_offsets(3):
            # The wheel handle sticks out in the +Z direction on the pipe.
            if axis == "x":
                xx, yy, zz = valve_pos + r_off[0], cx + r_off[1], cx + 5
            elif axis == "y":
                xx, yy, zz = cx + r_off[0], valve_pos + r_off[1], cx + 5
            else:
                xx, yy, zz = cx + r_off[0], cx + 5, valve_pos + r_off[1]
            if 0 <= xx < s.nx and 0 <= yy < s.ny and 0 <= zz < s.nz:
                s.set(xx, yy, zz, M.METAL_HOUSING)
        # Spokes.
        for so in range(-2, 3):
            if so == 0:
                continue
            if axis == "x":
                s.set(valve_pos, cx + so, cx + 5, M.METAL_HOUSING)
                s.set(valve_pos + so, cx, cx + 5, M.METAL_HOUSING)
            elif axis == "y":
                s.set(cx + so, valve_pos, cx + 5, M.METAL_HOUSING)
                s.set(cx, valve_pos + so, cx + 5, M.METAL_HOUSING)
            else:
                s.set(cx + so, cx + 5, valve_pos, M.METAL_HOUSING)
                s.set(cx, cx + 5, valve_pos + so, M.METAL_HOUSING)

    # --- Gauge ---
    if gauge_pos >= 0:
        # 5x1x5 disc at the side of the pipe (+Y side).
        for dx, dz in ring_offsets(2):
            if axis == "x":
                xx, yy, zz = gauge_pos + dx, cx + 4, cx + dz
            elif axis == "y":
                xx, yy, zz = cx + dx, gauge_pos + 4, cx + dz
            else:
                xx, yy, zz = cx + dx, cx + 4, gauge_pos + dz
            if 0 <= xx < s.nx and 0 <= yy < s.ny and 0 <= zz < s.nz:
                s.set(xx, yy, zz, M.PLASTIC_DARK)
        # Centre hub + needle (rotated by RNG).
        ang = int(rng.integers(0, 4))
        nd = ((1, 0), (0, 1), (-1, 0), (0, -1))[ang]
        if axis == "x":
            s.set(gauge_pos, cx + 4, cx, M.METAL_HOUSING)
            s.set(gauge_pos + nd[0], cx + 4, cx + nd[1], M.METAL_HOUSING)
        elif axis == "y":
            s.set(cx, gauge_pos + 4, cx, M.METAL_HOUSING)
            s.set(cx + nd[0], gauge_pos + 4, cx + nd[1], M.METAL_HOUSING)
        else:
            s.set(cx, cx + 4, gauge_pos, M.METAL_HOUSING)
            s.set(cx + nd[0], cx + 4, gauge_pos + nd[1], M.METAL_HOUSING)

    return s


# ============================================================================
# Cable bundle with junction-box anchors
# ============================================================================


@stamp_cached(8)
def build_cable_bundle(rng: np.random.Generator, *, length: int) -> Stamp:
    """A 5x5 cable bundle with periodic junction boxes and a 1-voxel sag."""
    s = Stamp((length, 5, 5))
    for u in range(length):
        # 1-voxel sag at midpoints between anchors (anchors every 30 voxels).
        sag = 1 if 12 < (u % 30) < 18 else 0
        # 5x5 bundle.
        for dy in range(5):
            for dz in range(5):
                if dy * dy + dz * dz <= 8 + sag:
                    if 0 <= dz + sag < 5:
                        s.set(u, dy, dz + sag, M.RUBBER_BLACK)
    # Junction-box anchors at u=0, 30, 60, ...
    for u in range(0, length, 30):
        s.fill((slice(u, min(length, u + 3)),
                slice(1, 4),
                slice(0, 3)), M.METAL_DARK)
        s.set(u + 1, 2, 0, M.EM_AMBER)
    return s


# ============================================================================
# Conduit run (1-voxel-thick electrical conduit)
# ============================================================================


@stamp_cached(8)
def build_conduit(rng: np.random.Generator, *, length: int) -> Stamp:
    """A thin 1x1 electrical conduit with periodic junction boxes."""
    s = Stamp((length, 3, 3))
    for u in range(length):
        s.set(u, 1, 1, M.METAL_DARK)
    # Junction boxes every 60 voxels.
    for u in range(0, length, 60):
        s.fill((slice(u, min(length, u + 3)),
                slice(0, 2),
                slice(0, 3)), M.METAL_DARK)
        s.set(u + 1, 0, 1, M.EM_CYAN)
    return s
