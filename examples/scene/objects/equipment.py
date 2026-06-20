# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Equipment builders — monitor, server rack, specimen tank, reactor core.

Sizes are in the **fine 0.02 m grid**. Each builder follows the plan's
five-point framework (inspiration, silhouette, asymmetries, signature,
variation) — see the plan file for the design rationale.
"""

from __future__ import annotations

import numpy as np

from .. import materials as M
from ..stamp import Stamp, stamp_cached, disc_offsets, ring_offsets, octagon_offsets


# ============================================================================
# Industrial CRT monitor
# ============================================================================


@stamp_cached(8)
def build_monitor(rng: np.random.Generator) -> Stamp:
    """A chunky control-room CRT — flat front, tapered back, knob row.

    Y is the depth axis (Y=0 is the screen face). The body steps inward over
    ~13 voxels along Y so the side profile is a rounded brick on its side
    (shorter than the v1 design — it fits on the desk now).
    """
    s = Stamp((15, 16, 18))

    # --- Front bezel + screen + rounded corners ---
    # Front bezel at y=0..2
    s.fill((slice(0, 15), slice(0, 2), slice(2, 15)), M.PLASTIC_DARK)
    # Knock 4 corners on the bezel face (rounded)
    for cx, cz in ((0, 2), (0, 14), (14, 2), (14, 14)):
        s.clear((cx, slice(0, 2), cz))
    # Screen (1-voxel-thick emissive at y=0).
    screen_state = rng.integers(0, 4)
    if screen_state == 3:
        screen_mat = M.PLASTIC_DARK  # dead
    else:
        screen_mat = (M.EM_CYAN, M.EM_AMBER, M.EM_GREEN)[int(screen_state)]
    s.fill((slice(1, 14), 0, slice(3, 13)), screen_mat)
    # Sparse static pattern for cyan screens (every 7th voxel darker)
    if screen_state == 0:
        for x in range(1, 14):
            for z in range(3, 13):
                if (x * 11 + z * 7) % 13 == 0:
                    s.set(x, 0, z, M.PLASTIC_DARK)

    # --- Tapered body (stepped inward as y grows) ---
    body_layers = (
        ((2, 6), (1, 14), (3, 13)),  # full size near the front
        ((6, 10), (2, 13), (4, 12)),
        ((10, 13), (3, 12), (5, 11)),
        ((13, 15), (3, 12), (6, 10)),  # back panel + cooling slot
    )
    for (y0, y1), (xl, xr), (zb, zt) in body_layers:
        s.fill((slice(xl - 1, xr + 1),
                slice(y0, y1),
                slice(zb - 1, zt + 1)), M.PLASTIC_DARK)
        # Hollow interior to save voxels (only outer shell at this y range).
        if y1 - y0 >= 2:
            s.clear((slice(xl, xr),
                     slice(y0 + 1, y1),
                     slice(zb, zt)))

    # Cooling vent slot on back panel.
    s.fill((slice(6, 10), 15, slice(6, 8)), M.METAL_DARK)

    # --- Knob row at z=1 below the screen ---
    for k, kx in enumerate((2, 5, 8, 11)):
        s.set(kx, 0, 1, M.METAL_DARK)
        # rotation indicator (offset by 1 voxel based on RNG)
        s.set(kx, 0, 1 + (rng.integers(0, 2)), M.METAL_DARK)
    # Bigger power knob at far right.
    s.fill((slice(13, 15), 0, slice(0, 2)), M.METAL_DARK)

    # --- Brand plate above screen ---
    s.fill((slice(3, 11), 0, 14), M.METAL_DARK)

    return s


# ============================================================================
# 4-post server rack
# ============================================================================


@stamp_cached(8)
def build_server_rack(rng: np.random.Generator) -> Stamp:
    """A 19" 4-post equipment rack with 5 1U units at varying depths,
    one of them pulled forward.
    """
    s = Stamp((30, 20, 120))

    # --- 4 corner rails ---
    for rx, ry in ((0, 0), (29, 0), (0, 19), (29, 19)):
        s.fill((rx, ry, slice(0, 120)), M.METAL_DARK)
        # Mounting-hole dots every 4 voxels.
        for hz in range(2, 120, 4):
            # Just slightly differentiate — at this resolution it reads as texture.
            if (hz // 4) % 2 == 0:
                s.set(rx, ry, hz, M.METAL_HOUSING)

    # --- Top and bottom frames ---
    s.fill((slice(0, 30), slice(0, 20), slice(0, 3)), M.METAL_DARK)  # base
    s.fill((slice(0, 30), slice(0, 20), slice(118, 120)), M.METAL_DARK)  # top
    # Top fan-grille pattern (knock out a 5x3 checkerboard of holes).
    for gx in range(3, 27, 5):
        for gy in range(2, 18, 5):
            s.clear((slice(gx, gx + 2), slice(gy, gy + 2), 119))

    # --- 5 unit slots ---
    pulled_idx = int(rng.integers(0, 5))
    for slot_i, z0 in enumerate((20, 40, 60, 80, 100)):
        pulled = (slot_i == pulled_idx)
        y_front = -4 if pulled else 0
        # Unit body
        for x in range(1, 29):
            s.fill((x, slice(y_front + 1, 19), slice(z0, z0 + 16)), M.PLASTIC_DARK)
        # If pulled, also fill the side at the new front face for the side panel.
        if pulled:
            # Visible side face
            for z in range(z0, z0 + 16):
                s.set(0, y_front, z, M.METAL_DARK)
                s.set(29, y_front, z, M.METAL_DARK)

        # Per-unit front face — different layouts.
        front_y = max(0, y_front)
        if slot_i == 1:
            # Network switch: 12 small LEDs in two rows.
            for row in range(2):
                for k in range(6):
                    lx = 4 + k * 4
                    lz = z0 + 5 + row * 4
                    led_mat = M.EM_GREEN
                    if rng.random() < 0.20:
                        led_mat = M.EM_AMBER
                    if rng.random() < 0.05:
                        led_mat = M.EM_RED
                    s.set(lx, front_y, lz, led_mat)
        elif slot_i == 2:
            # UPS / power: LCD display (amber digit pattern) + 2 buttons.
            s.fill((slice(8, 12), front_y, slice(z0 + 5, z0 + 9)), M.EM_AMBER)
            # Buttons
            s.fill((slice(18, 20), front_y, slice(z0 + 5, z0 + 7)), M.PLASTIC_DARK)
            s.fill((slice(22, 24), front_y, slice(z0 + 5, z0 + 7)), M.PLASTIC_DARK)
        elif slot_i == 3:
            # Patch panel: 8 dark inset sockets in two rows, no LEDs.
            for row in range(2):
                for k in range(4):
                    sx = 4 + k * 6
                    sz = z0 + 5 + row * 4
                    s.set(sx, front_y, sz, M.METAL_DARK)
        elif slot_i == 0:
            # Storage server: 6 LEDs in a row + small LCD slot.
            for k in range(6):
                lx = 4 + k * 3
                lz = z0 + 7
                led_mat = M.EM_CYAN if rng.random() < 0.6 else M.EM_AMBER
                s.set(lx, front_y, lz, led_mat)
            s.fill((slice(22, 25), front_y, slice(z0 + 5, z0 + 8)), M.METAL_DARK)
        else:
            # Pulled-out unit: empty front (just a 4-voxel-wide cable tail dangling).
            for cz in range(z0 - 4, z0):
                if cz < 0:
                    continue
                s.set(15, 18, cz, M.RUBBER_BLACK)
                s.set(16, 18, cz, M.RUBBER_BLACK)

    # --- Cable bundle running down the back-right rail ---
    for z in range(2, 118):
        # 5x5 bundle, slightly sagging — 1-voxel sag between anchor points
        sag = 1 if (z % 30 == 15) else 0
        s.fill((slice(24, 29),
                slice(15 + sag, 20),
                slice(z, z + 1)), M.RUBBER_BLACK)

    return s


# ============================================================================
# Specimen tank
# ============================================================================


@stamp_cached(8)
def build_specimen_tank(rng: np.random.Generator) -> Stamp:
    """A lab containment tank with bioluminescent specimen on a perforated
    platform, bubble stream, top control panel, and base gauges.
    """
    s = Stamp((27, 27, 47))

    # --- Base pedestal (27x27x8 dark metal) ---
    s.fill((slice(None), slice(None), slice(0, 8)), M.METAL_DARK)
    # Gauges (two round dark discs on front face).
    for gauge_x in (8, 18):
        for dx, dz in (
                (-2, 0), (2, 0), (0, -2), (0, 2),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
                (-2, -1), (-2, 1), (2, -1), (2, 1),
                (-1, -2), (1, -2), (-1, 2), (1, 2),
        ):
            xx, zz = gauge_x + dx, 4 + dz
            if 0 <= xx < 27 and 0 <= zz < 8:
                s.set(xx, 0, zz, M.PLASTIC_DARK)
        # Center hub.
        s.set(gauge_x, 0, 4, M.METAL_HOUSING)
        # Needle: random angle (one of 4)
        ang = int(rng.integers(0, 4))
        offsets = [(2, 0), (0, 2), (-2, 0), (0, -2)][ang]
        s.set(gauge_x + offsets[0], 0, 4 + offsets[1], M.METAL_HOUSING)

    # Side valve wheel.
    vc_y = 13
    for dx, dz in ring_offsets(3):
        xx, zz = vc_y + dx, 4 + dz
        if 0 <= xx < 27 and 0 <= zz < 8:
            s.set(0, xx, zz, M.METAL_HOUSING)
    for off in range(-2, 3):
        if off == 0:
            continue
        s.set(0, vc_y + off, 4, M.METAL_HOUSING)
        s.set(0, vc_y, 4 + off, M.METAL_HOUSING)
    s.set(0, vc_y, 4, M.METAL_DARK)

    # --- 4 corner posts ---
    for px, py in ((1, 1), (25, 1), (1, 25), (25, 25)):
        s.fill((px, py, slice(8, 43)), M.METAL_HOUSING)
        s.set(px, py, 42, M.METAL_DARK)  # cap

    # --- Glass walls (1-voxel-thick transparent shell, inside the posts) ---
    glass_z0, glass_z1 = 9, 41
    # 4 side walls
    for z in range(glass_z0, glass_z1):
        # Front + back wall (at y=1 and y=25)
        s.fill((slice(2, 25), 1, z), M.GLASS_TANK)
        s.fill((slice(2, 25), 25, z), M.GLASS_TANK)
        # Left + right wall
        s.fill((1, slice(2, 25), z), M.GLASS_TANK)
        s.fill((25, slice(2, 25), z), M.GLASS_TANK)

    # --- Top control panel (25x25x3 lid above glass) ---
    s.fill((slice(1, 26), slice(1, 26), slice(43, 46)), M.METAL_DARK)
    # 4 knobs in a 2x2 grid.
    for kx in (8, 18):
        for ky in (8, 18):
            s.set(kx, ky, 46, M.METAL_HOUSING)
            # Rotation indicator
            s.set(kx + int(rng.integers(-1, 2)), ky, 46, M.PLASTIC_DARK)
    # LCD strip (amber).
    s.fill((slice(11, 15), slice(11, 15), 46), M.EM_AMBER)
    # Active LED.
    s.set(13, 22, 46, M.EM_CYAN)

    # --- Perforated platform inside the glass at z = 14 ---
    plat_z = 14
    s.fill((slice(4, 23), slice(4, 23), plat_z), M.METAL_DARK)
    # Random 1-voxel holes.
    holes = rng.choice(np.arange(4 * 19 * 19).reshape(-1), size=6, replace=False) if rng else []
    for h in (6, 10, 18, 22, 14, 8):
        s.clear((4 + (h % 19), 4 + ((h * 5) % 19), plat_z))

    # --- Bioluminescent specimen ---
    species = (M.EM_BIO_TEAL, M.EM_BIO_GREEN, M.EM_BIO_VIOLET)[int(rng.integers(0, 3))]
    cluster_count = int(rng.integers(40, 121))
    pos = np.array([13, 13, plat_z + 1])
    placed = 0
    while placed < cluster_count:
        # 1-voxel stalk for the first 6 voxels going up.
        if placed < 6:
            x, y, z = pos[0], pos[1], plat_z + 1 + placed
            s.set(int(x), int(y), int(z), species)
            placed += 1
            continue
        # Random-walk cap above z = plat_z + 6
        step = rng.integers(-1, 2, size=3)
        cand = pos + step
        cand[2] = max(plat_z + 6, cand[2])
        if 2 <= cand[0] <= 24 and 2 <= cand[1] <= 24 and plat_z + 1 <= cand[2] <= 40:
            pos = cand
            s.set(int(pos[0]), int(pos[1]), int(pos[2]), species)
            placed += 1

    # --- Bubble stream: 4 transparent voxels rising in a vertical line ---
    bub_x, bub_y = 6, 6
    for i, bz in enumerate((plat_z + 4, plat_z + 12, plat_z + 20, plat_z + 28)):
        if bz < 41:
            s.set(bub_x, bub_y, bz, M.GLASS_TANK)

    # --- Feeder line going up from lid to ceiling ---
    for fz in range(46, 47):
        s.set(13, 23, fz, M.EM_BIO_TEAL)

    return s


# ============================================================================
# Thinking-machine rack (CM-5 style: tall slim glossy black with red LED grid)
# ============================================================================


@stamp_cached(8)
def build_thinking_machine_rack(rng: np.random.Generator) -> Stamp:
    """A classic Thinking-Machines CM-5-style rack — standard 19" server cabinet
    dimensions (0.6 m wide x 1.0 m deep x 2.0 m tall) in glossy black, with a
    large grid of red LEDs filling the front face.

    Inspiration: photograph of a row of CM-5 cabinets — black hexahedral
    cabinets with their iconic *grid* of small red status LEDs (no other
    fixtures on the front face, no knobs, no buttons — just the LED matrix).

    Variation:
      * Each LED is independently on/off with ~85 % "on" probability — gives
        every cabinet a different pattern at first glance.
      * 5 % chance the WHOLE cabinet is dead (no LEDs lit).
    """
    s = Stamp((30, 50, 100))

    # --- Glossy-black cabinet body (1-voxel-thick shell) ---
    s.fill((slice(None), slice(None), slice(None)), M.RUBBER_BLACK)
    s.clear((slice(1, 29), slice(1, 49), slice(1, 99)))

    # --- Edge bevels (slightly different black on the front-face corners) ---
    for cx, cy in ((0, 0), (29, 0)):
        s.fill((cx, cy, slice(None)), M.PLASTIC_DARK)

    # --- Top + base ridge (slightly raised lip) ---
    s.fill((slice(None), slice(None), 99), M.PLASTIC_DARK)
    s.fill((slice(None), slice(None), 0), M.PLASTIC_DARK)
    s.fill((slice(None), slice(None), 1), M.PLASTIC_DARK)

    # --- Red LED grid filling most of the front face ---
    led_x0, led_x1 = 3, 27  # 24 cols wide
    led_z0, led_z1 = 8, 92  # 84 rows tall
    is_dead = rng.random() < 0.05
    on_prob = 0.0 if is_dead else 0.85
    for z in range(led_z0, led_z1, 3):  # row pitch 3 fine voxels
        for x in range(led_x0, led_x1, 2):  # col pitch 2 fine voxels
            if rng.random() < on_prob:
                s.set(x, 0, z, M.EM_RED)

    # --- Small power LED bottom-front: green or amber ---
    s.set(15, 0, 4, M.EM_GREEN if rng.random() < 0.8 else M.EM_AMBER)

    # --- Cooling vent slits on the back panel (top half) ---
    for z in range(50, 90, 4):
        for x in range(4, 26, 4):
            s.clear((slice(x, x + 2), 49, slice(z, z + 2)))

    return s


# ============================================================================
# Laser beam (Resident Evil corridor)
# ============================================================================


@stamp_cached(8)
def build_laser_beam(rng: np.random.Generator, *, length: int,
                     axis: str = "y", colour: str = M.EM_CYAN) -> Stamp:
    """A single laser beam ``length`` voxels along ``axis`` ('x', 'y', or 'z').

    1-voxel-cross-section emissive line — bright but only ``length`` lights.
    """
    if axis == "x":
        s = Stamp((length, 1, 1))
        s.fill((slice(None), 0, 0), colour)
    elif axis == "y":
        s = Stamp((1, length, 1))
        s.fill((0, slice(None), 0), colour)
    elif axis == "z":
        s = Stamp((1, 1, length))
        s.fill((0, 0, slice(None)), colour)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    return s


@stamp_cached(8)
def build_laser_emitter(rng: np.random.Generator, *,
                        colour: str = M.EM_CYAN) -> Stamp:
    """A small wall-mounted laser emitter housing — a 4x3x3 dark plastic box
    with a single bright LED on the emitting face."""
    s = Stamp((4, 3, 3))
    s.fill((slice(None), slice(None), slice(None)), M.PLASTIC_DARK)
    s.set(2, 2, 1, colour)
    return s


@stamp_cached(8)
def build_glass_panel(rng: np.random.Generator, *,
                      length: int, height: int, axis: str = "y") -> Stamp:
    """A vertical glass wall panel (1-voxel-thick) of size ``length`` x ``height``.

    ``axis`` is the world direction the panel runs along ('x' or 'y'). The
    panel's normal is perpendicular to ``axis``. A 1-voxel metal frame runs
    along the top + bottom + sides for structural read.
    """
    if axis == "x":
        s = Stamp((length, 1, height))
        s.fill((slice(None), 0, slice(None)), M.GLASS_TANK)
        s.fill((slice(None), 0, 0), M.METAL_DARK)  # bottom rail
        s.fill((slice(None), 0, height - 1), M.METAL_DARK)  # top rail
        s.fill((0, 0, slice(None)), M.METAL_DARK)  # left frame
        s.fill((length - 1, 0, slice(None)), M.METAL_DARK)  # right frame
    else:
        s = Stamp((1, length, height))
        s.fill((0, slice(None), slice(None)), M.GLASS_TANK)
        s.fill((0, slice(None), 0), M.METAL_DARK)
        s.fill((0, slice(None), height - 1), M.METAL_DARK)
        s.fill((0, 0, slice(None)), M.METAL_DARK)
        s.fill((0, length - 1, slice(None)), M.METAL_DARK)
    return s


@stamp_cached(2)
def build_laser_corridor_glass_wall(rng: np.random.Generator,
                                    *, length: int, height: int) -> Stamp:
    """Aluminium-framed full-height glass wall for the laser-corridor.

    Stamp axes (designed for placement against an X-axis-normal wall —
    runs along Y, thin along X, tall along Z):
      * NX = 4 fine voxels deep (the frame depth)
      * NY = ``length`` fine voxels along Y (the wall length)
      * NZ = ``height`` fine voxels tall

    Geometry:
      * Top + bottom 3-voxel METAL_HOUSING horizontal rails.
      * Vertical METAL_HOUSING mullions every 50 fine voxels (1 m), 3
        voxels wide, full inner height.
      * GLASS_TANK fills the middle 2 voxels of the stamp depth (y planes
        x=1..2) between mullions, so the wall reads as a transparent
        panel system framed by silver aluminium.
    """
    NX = 4
    s = Stamp((NX, length, height))
    # Top + bottom horizontal rails.
    s.fill((slice(None), slice(None), slice(0, 3)), M.METAL_HOUSING)
    s.fill((slice(None), slice(None), slice(height - 3, height)),
           M.METAL_HOUSING)
    # Fill the middle-depth band with glass between the rails.
    s.fill((slice(1, 3), slice(None), slice(3, height - 3)), M.GLASS_TANK)
    # Then overwrite at mullion positions with aluminium frame.
    mullion_pitch = 50
    mullion_w = 3
    for y0 in range(0, length - mullion_w + 1, mullion_pitch):
        s.fill((slice(None), slice(y0, y0 + mullion_w),
                slice(3, height - 3)), M.METAL_HOUSING)
    # Final mullion at the far end if the run doesn't land on a stride.
    if (length - mullion_w) % mullion_pitch != 0:
        s.fill((slice(None), slice(length - mullion_w, length),
                slice(3, height - 3)), M.METAL_HOUSING)
    return s


# ============================================================================
# Reactor core (centrepiece)
# ============================================================================


@stamp_cached(8)
def build_reactor_core(rng: np.random.Generator) -> Stamp:
    """Fluted column with crenellated top, cooling fins, vertical slit-lights,
    containment posts, and a glowing floor halo.

    Footprint ~80x80 in the fine grid (centre is at (40, 40)).
    """
    s = Stamp((80, 80, 140))
    cx, cy = 40, 40
    base_z = 0

    # --- Glowing emissive floor halo (annulus radius 30..50) at z=0 ---
    for dx in range(-50, 51):
        for dy in range(-50, 51):
            r2 = dx * dx + dy * dy
            if 30 * 30 <= r2 <= 50 * 50:
                xx, yy = cx + dx, cy + dy
                if 0 <= xx < 80 and 0 <= yy < 80:
                    s.set(xx, yy, base_z, M.EM_FLOOR_RING)

    # --- Outer fluted octagonal column ---
    # Cross-section: octagonal box of "radius" 26 (half-side 26, chamfered corner 13).
    col_offsets = octagon_offsets(26)
    # The shell is 2-voxel-thick. We do this by filling at the rim (outer minus inner).
    inner_offsets_set = set(octagon_offsets(24))
    rim_offsets = [(dx, dy) for (dx, dy) in col_offsets if (dx, dy) not in inner_offsets_set]
    column_top = 100
    for z in range(base_z, column_top):
        for dx, dy in rim_offsets:
            xx, yy = cx + dx, cy + dy
            if 0 <= xx < 80 and 0 <= yy < 80:
                s.set(xx, yy, z, M.METAL_HOUSING)

    # --- Vertical flutes (1-voxel ribs protruding 1 voxel outward) ---
    flute_angles = [(26, 0), (-26, 0), (0, 26), (0, -26),
                    (18, 18), (-18, 18), (18, -18), (-18, -18)]
    for fdx, fdy in flute_angles:
        # Push the flute one voxel further out (clip to octagonal envelope)
        for z in range(base_z, column_top):
            xx, yy = cx + fdx, cy + fdy
            if 0 <= xx < 80 and 0 <= yy < 80:
                s.set(xx, yy, z, M.METAL_HOUSING)

    # --- Vertical slits in the column wall (between flutes) ---
    # Cut a thin 2x1 vertical gap at 8 angular positions between flutes.
    slit_angles = [(22, 13), (-22, 13), (22, -13), (-22, -13),
                   (13, 22), (-13, 22), (13, -22), (-13, -22)]
    for sdx, sdy in slit_angles:
        for z in range(base_z + 10, base_z + 70):
            for dz in (0,):  # 1-voxel-wide tall slit
                xx, yy = cx + sdx, cy + sdy
                if 0 <= xx < 80 and 0 <= yy < 80:
                    s.clear((xx, yy, z + dz))

    # --- Inner emissive shell (rectangular 20x20x100, 1-voxel walls + top cap) ---
    iz0, iz1 = base_z + 4, base_z + 104
    # 4 walls
    for z in range(iz0, iz1):
        for dx in range(-10, 11):
            s.set(cx + dx, cy - 10, z, M.EM_REACTOR)
            s.set(cx + dx, cy + 10, z, M.EM_REACTOR)
        for dy in range(-10, 11):
            s.set(cx - 10, cy + dy, z, M.EM_REACTOR)
            s.set(cx + 10, cy + dy, z, M.EM_REACTOR)
    # Top cap.
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            s.set(cx + dx, cy + dy, iz1, M.EM_REACTOR)

    # --- Horizontal cooling fins at z = 60, 66, 72, 78 ---
    for fz in (60, 66, 72, 78):
        for dx, dy in octagon_offsets(34):
            xx, yy = cx + dx, cy + dy
            if 0 <= xx < 80 and 0 <= yy < 80:
                s.set(xx, yy, fz, M.METAL_HOUSING)

    # --- Crenellated top: alternating 4x4 metal blocks and gaps at z=100..130 ---
    cren_z0 = column_top
    for ang_i, (adx, ady) in enumerate(flute_angles):
        if ang_i % 2 == 0:
            for z in range(cren_z0, cren_z0 + 8):
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        s.set(cx + adx + dx, cy + ady + dy, z, M.METAL_HOUSING)

    # --- Octagonal dome cap ---
    for dx, dy in octagon_offsets(18):
        for dz in range(0, 8):
            s.set(cx + dx, cy + dy, cren_z0 + 8 + dz, M.METAL_HOUSING)

    # --- Base indicator ring (8 LEDs around the column at z=2) ---
    base_z_ind = base_z + 2
    ind_colours = (M.EM_RED, M.EM_AMBER, M.EM_GREEN, M.EM_AMBER)
    for i, (idx, idy) in enumerate(((30, 0), (-30, 0), (0, 30), (0, -30),
                                    (21, 21), (-21, 21), (21, -21), (-21, -21))):
        s.set(cx + idx, cy + idy, base_z_ind, ind_colours[i % 4])
        s.set(cx + idx, cy + idy, base_z_ind + 1, ind_colours[i % 4])

    # --- 6 radial pipes leaving the base, L-bending into the floor ---
    pipe_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    for pdx, pdy in pipe_dirs:
        # Pipe goes out 24 voxels horizontally from radius 26.
        for d in range(24):
            x = cx + pdx * (28 + d) - (1 if abs(pdy) else 0)
            y = cy + pdy * (28 + d) - (1 if abs(pdx) else 0)
            if 0 <= x < 80 and 0 <= y < 80:
                s.set(int(x), int(y), base_z + 2, M.METAL_DARK)
                s.set(int(x), int(y), base_z + 3, M.METAL_DARK)

    # --- 4 containment posts at radius 35 ---
    for pdx, pdy in ((35, 0), (-35, 0), (0, 35), (0, -35)):
        px, py = cx + pdx, cy + pdy
        if not (1 < px < 78 and 1 < py < 78):
            continue
        for z in range(base_z, base_z + 110):
            s.fill((slice(px - 1, px + 2),
                    slice(py - 1, py + 2),
                    z), M.METAL_DARK)
        # Capacitor block on top.
        s.fill((slice(px - 2, px + 3),
                slice(py - 2, py + 3),
                slice(base_z + 110, base_z + 114)), M.METAL_HOUSING)
        # Indicator LED.
        s.set(px, py, base_z + 114, M.EM_RED)

    # --- Cable arcs between containment posts ---
    arc_z = base_z + 110
    post_positions = [(cx + 35, cy), (cx, cy + 35), (cx - 35, cy), (cx, cy - 35)]
    for i in range(4):
        p1 = post_positions[i]
        p2 = post_positions[(i + 1) % 4]
        steps = 12
        for t in range(steps + 1):
            tt = t / steps
            x = int(p1[0] * (1 - tt) + p2[0] * tt)
            y = int(p1[1] * (1 - tt) + p2[1] * tt)
            sag = -int(2 * np.sin(tt * np.pi))
            zz = arc_z + sag
            if 0 <= x < 80 and 0 <= y < 80 and 0 <= zz < 140:
                s.set(x, y, zz, M.RUBBER_BLACK)

    return s


# ============================================================================
# Bio-lab equipment catalog (round 3)
# ============================================================================


@stamp_cached(8)
def build_centrifuge(rng: np.random.Generator) -> Stamp:
    """Eppendorf 5424 benchtop centrifuge — 22 × 28 × 35 (0.44 × 0.56 × 0.70 m).

    Octagonal-footprint body with a domed lid, control LCD + buttons, side
    louvers, and an EM_AMBER 'running' indicator around the lid porthole.
    """
    s = Stamp((22, 28, 35))
    # Octagonal body footprint (knock the 4 corner quads of a 22x28 rect).
    body_top = 22
    for z in range(2, body_top):
        for x in range(22):
            for y in range(28):
                if (x < 3 and y < 3) or (x < 3 and y > 24) or \
                        (x > 18 and y < 3) or (x > 18 and y > 24):
                    continue
                s.set(x, y, z, M.PLASTIC_WHITE)
    # Hollow interior (save voxels).
    s.clear((slice(2, 20), slice(2, 26), slice(3, body_top - 1)))
    # Base plate.
    s.fill((slice(None), slice(None), slice(0, 2)), M.PLASTIC_DARK)
    # Domed lid: 2-step stepped chamfer.
    for z in range(body_top, body_top + 4):
        inset = z - body_top
        s.fill((slice(2 + inset, 20 - inset),
                slice(2 + inset, 26 - inset),
                z), M.METAL_DARK)
    # Top of lid: dark glass porthole + emissive amber ring.
    lid_top = body_top + 4
    cxp, cyp = 11, 14
    # 6-voxel-diameter porthole (discrete disc).
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx * dx + dy * dy <= 9:
                s.set(cxp + dx, cyp + dy, lid_top, M.PLASTIC_DARK)
    # Amber ring at radius 4.
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            r2 = dx * dx + dy * dy
            if 12 < r2 <= 16:
                s.set(cxp + dx, cyp + dy, lid_top, M.EM_AMBER)
    # Continue lid up one more voxel for the dome cap.
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx * dx + dy * dy <= 9:
                s.set(cxp + dx, cyp + dy, lid_top, M.METAL_DARK)
    # Inset front control panel.
    s.fill((slice(6, 16), 0, slice(4, 10)), M.PLASTIC_DARK)
    # LCD strip on the panel.
    s.fill((slice(7, 13), 0, slice(7, 9)), M.EM_AMBER)
    # 3 button dots in a row (cyan/amber/red).
    for i, mat in enumerate((M.EM_CYAN, M.EM_AMBER, M.EM_RED)):
        s.set(8 + i * 2, 0, 5, mat)
    # Side louvers (3 slanted slits each side).
    for y_side in (0, 27):
        for ly in (8, 13, 18):
            for x in range(4, 18):
                if (x + (ly // 5)) % 3 == 0:
                    s.clear((x, y_side, 12 + (ly - 8) // 2))
    return s


@stamp_cached(8)
def build_incubator(rng: np.random.Generator) -> Stamp:
    """Thermo Forma cell-culture incubator — 32 × 32 × 110 (0.64 × 0.64 × 2.2 m).

    Tall stainless cabinet with a full-height glass door, internal back-wall
    EM_WHITE backlight, 4 visible shelves carrying petri-dish stacks, top
    control panel with two amber LCDs and status LEDs.
    """
    s = Stamp((32, 32, 110))
    # Body shell.
    s.fill((slice(None), slice(None), slice(None)), M.METAL_HOUSING)
    s.clear((slice(1, 31), slice(1, 31), slice(1, 109)))
    # Bottom kick-plate (recessed 1 voxel).
    for z in range(0, 4):
        s.clear((slice(None), 0, z))
        s.fill((slice(None), 1, z), M.METAL_HOUSING)
    # Glass door inset 1 voxel at y=1 (covers most of the front).
    door_z0, door_z1 = 6, 92
    s.fill((slice(2, 30), 1, slice(door_z0, door_z1)), M.GLASS_TANK)
    # Door frame (1-voxel METAL_DARK).
    s.fill((1, 1, slice(door_z0 - 1, door_z1 + 1)), M.METAL_DARK)
    s.fill((30, 1, slice(door_z0 - 1, door_z1 + 1)), M.METAL_DARK)
    s.fill((slice(1, 31), 1, door_z0 - 1), M.METAL_DARK)
    s.fill((slice(1, 31), 1, door_z1), M.METAL_DARK)
    # Vertical handle on the right edge of the door.
    s.fill((28, 0, slice(40, 65)), M.METAL_DARK)
    # Internal back-wall EM_WHITE backlight strip.
    s.fill((slice(3, 29), 30, slice(door_z0, door_z1)), M.EM_WHITE)
    # 4 internal shelves with dish stacks.
    for shelf_z in (20, 40, 60, 80):
        s.fill((slice(2, 30), slice(2, 30), shelf_z), M.METAL_HOUSING)
        for px in (8, 16, 24):
            mat = M.EM_BIO_GREEN if (shelf_z + px) % 2 == 0 else M.EM_BIO_TEAL
            s.fill((slice(px - 1, px + 2),
                    slice(12, 15),
                    slice(shelf_z + 1, shelf_z + 5)), mat)
    # Top control panel above the door (z=94..104).
    panel_z0 = 94
    s.fill((slice(2, 30), 0, slice(panel_z0, panel_z0 + 10)), M.PLASTIC_DARK)
    # Two amber LCDs.
    s.fill((slice(5, 12), 0, slice(panel_z0 + 3, panel_z0 + 6)), M.EM_AMBER)
    s.fill((slice(15, 22), 0, slice(panel_z0 + 3, panel_z0 + 6)), M.EM_AMBER)
    # 3 status LEDs in a row.
    for i, mat in enumerate((M.EM_CYAN, M.EM_GREEN, M.EM_AMBER)):
        s.set(24 + i * 2, 0, panel_z0 + 7, mat)
    return s


@stamp_cached(8)
def build_fume_hood(rng: np.random.Generator) -> Stamp:
    """Labconco walk-in chemical fume hood — 90 × 60 × 140 (1.8 × 1.2 × 2.8 m).

    Tall wide cabinet with a raised sash glass window between two counterweight
    tracks, interior workspace + faucet, vent hood at top with louvers, side
    walls in PLASTIC_WHITE, frame in METAL_HOUSING.
    """
    s = Stamp((90, 60, 140))
    # Outer side walls (PLASTIC_WHITE).
    for y in range(60):
        for z in range(8, 120):
            s.set(0, y, z, M.PLASTIC_WHITE)
            s.set(89, y, z, M.PLASTIC_WHITE)
    # Back wall.
    for x in range(90):
        for z in range(8, 120):
            s.set(x, 59, z, M.PLASTIC_WHITE)
    # Bottom (stainless workspace floor).
    s.fill((slice(None), slice(None), slice(0, 10)), M.METAL_HOUSING)
    # Front face — frame columns + raised sash window.
    # Two vertical counterweight tracks at x=1 and x=88 (METAL_DARK).
    s.fill((1, 0, slice(8, 120)), M.METAL_DARK)
    s.fill((88, 0, slice(8, 120)), M.METAL_DARK)
    # Workspace opening: x=2..87, y=0, z=10..70 (open below the sash).
    # (no fill — that's the open air gap).
    # Raised sash window glass at z=70..110.
    s.fill((slice(2, 88), 0, slice(70, 110)), M.GLASS_TANK)
    # Sash handle (horizontal bar at the bottom of the glass).
    s.fill((slice(30, 60), 0, slice(68, 70)), M.METAL_DARK)
    # Vent hood at top (z=120..140).
    s.fill((slice(0, 90), slice(0, 60), slice(120, 140)), M.METAL_HOUSING)
    s.clear((slice(2, 88), slice(2, 58), slice(122, 138)))
    # 6 horizontal louvers cut into the front face of the hood.
    for ly in range(122, 138, 3):
        for x in range(10, 80):
            s.clear((x, 0, ly))
    # Airflow amber LCD on the hood front.
    s.fill((slice(38, 50), 0, slice(125, 128)), M.EM_AMBER)
    # 2 front control LEDs (light + fan switches).
    s.set(15, 0, 124, M.EM_CYAN)
    s.set(20, 0, 124, M.EM_GREEN)
    # Interior faucet on rear wall (1×1×4 column + 1×1×1 spout).
    s.fill((45, 58, slice(12, 16)), M.METAL_HOUSING)
    s.set(45, 57, 16, M.METAL_HOUSING)
    return s


@stamp_cached(8)
def build_autoclave(rng: np.random.Generator) -> Stamp:
    """Tuttnauer 3870 bench autoclave — 44 × 40 × 95 (0.88 × 0.80 × 1.90 m).

    Vertical octagonal chamber on a stand frame, circular front door with a
    wheel handle, gasket bands, heat-warning red stripe, pressure gauges,
    control LCDs, top exhaust pipe.
    """
    s = Stamp((44, 40, 95))
    cx, cy = 22, 20
    chamber_z0, chamber_z1 = 15, 80
    # Chamber: octagonal cross-section (half-side 12).
    chamber_offsets = octagon_offsets(12)
    for z in range(chamber_z0, chamber_z1):
        for dx, dy in chamber_offsets:
            xx, yy = cx + dx, cy + dy
            if 0 <= xx < 44 and 0 <= yy < 40:
                s.set(xx, yy, z, M.METAL_HOUSING)
    # Hollow interior.
    inner = set(octagon_offsets(10))
    for z in range(chamber_z0 + 1, chamber_z1 - 1):
        for dx, dy in inner:
            xx, yy = cx + dx, cy + dy
            if 0 <= xx < 44 and 0 <= yy < 40:
                s.clear((xx, yy, z))
    # Circular front door (METAL_DARK disc at y=0).
    door_cy_z = (chamber_z0 + chamber_z1) // 2
    for dx in range(-10, 11):
        for dz in range(-10, 11):
            if dx * dx + dz * dz <= 100:
                s.set(cx + dx, 0, door_cy_z + dz, M.METAL_DARK)
    # Wheel handle on door: ring + 4 spokes.
    for dx in range(-7, 8):
        for dz in range(-7, 8):
            r2 = dx * dx + dz * dz
            if 36 <= r2 <= 49:
                s.set(cx + dx, 0, door_cy_z + dz, M.METAL_HOUSING)
    for off in range(-6, 7):
        if off == 0:
            continue
        s.set(cx + off, 0, door_cy_z, M.METAL_HOUSING)
        s.set(cx, 0, door_cy_z + off, M.METAL_HOUSING)
    s.set(cx, 0, door_cy_z, M.PLASTIC_DARK)
    # Heat-warning red band at z=60.
    for dx, dy in chamber_offsets:
        xx, yy = cx + dx, cy + dy
        if 0 <= xx < 44 and 0 <= yy < 40:
            s.set(xx, yy, 60, M.PAINTED_RED)
    # 2 side pressure gauges (round discs with needles).
    for gx, gz in ((4, 30), (39, 30)):
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                r2 = dx * dx + dz * dz
                if r2 <= 4:
                    s.set(gx, 5 + (1 if gx > cx else -1) * 0,
                          gz + dz, M.PLASTIC_DARK)
        # Approximate side mounting — use a fixed Y inside the body.
        # Needle (random direction).
        ang = int(rng.integers(0, 4))
        nd = ((1, 0), (0, 1), (-1, 0), (0, -1))[ang]
        s.set(gx, 5, gz + nd[1], M.METAL_HOUSING)
    # Front control panel below door (z=10..14, y=0).
    s.fill((slice(8, 36), 0, slice(10, 14)), M.PLASTIC_DARK)
    s.fill((slice(11, 18), 0, slice(11, 13)), M.EM_AMBER)
    s.fill((slice(26, 33), 0, slice(11, 13)), M.EM_AMBER)
    # Stand frame.
    for (px, py) in ((4, 4), (39, 4), (4, 35), (39, 35)):
        s.fill((px, py, slice(0, 15)), M.METAL_DARK)
    # Top exhaust pipe.
    for dz in range(0, 6):
        s.fill((slice(cx - 1, cx + 2),
                slice(cy - 1, cy + 2),
                chamber_z1 + dz), M.METAL_DARK)
    return s


@stamp_cached(8)
def build_lab_bench(rng: np.random.Generator) -> Stamp:
    """Stainless lab bench with central sink + faucet — 120 × 45 × 50
    (2.4 × 0.9 × 1.0 m envelope; bench top is 0.84 m, faucet extends ~0.96 m).
    """
    s = Stamp((120, 45, 50))
    # Top.
    s.fill((slice(None), slice(None), slice(39, 42)), M.METAL_HOUSING)
    # Recessed sink (offset slightly left of centre).
    sink_x0, sink_x1 = 50, 75
    sink_y0, sink_y1 = 10, 35
    s.clear((slice(sink_x0, sink_x1),
             slice(sink_y0, sink_y1),
             slice(36, 42)))
    s.fill((slice(sink_x0, sink_x1),
            slice(sink_y0, sink_y1),
            slice(35, 36)), M.METAL_DARK)
    # Drain centre dot.
    s.set((sink_x0 + sink_x1) // 2, (sink_y0 + sink_y1) // 2, 35, M.PLASTIC_DARK)
    # L-shaped faucet (back-left of sink).
    fx, fy = sink_x0 + 2, sink_y1 + 2
    s.fill((fx, fy, slice(41, 49)), M.METAL_HOUSING)
    s.fill((slice(fx, fx + 5), fy, 48), M.METAL_HOUSING)
    s.set(fx + 4, fy, 47, M.METAL_HOUSING)
    # Hot/cold knob dots beside faucet base.
    s.set(fx - 1, fy, 41, M.PLASTIC_DARK)
    s.set(fx + 1, fy, 41, M.PLASTIC_DARK)
    # Backsplash.
    s.fill((slice(0, 120), slice(40, 44), slice(42, 50)), M.METAL_HOUSING)
    # 5 evenly-spaced power outlets along the backsplash with green LEDs.
    for i in range(5):
        ox = 8 + i * 22
        s.fill((slice(ox, ox + 2), 40, slice(45, 47)), M.PLASTIC_DARK)
        s.set(ox, 40, 46, M.EM_GREEN)
    # Under-cabinet (hollow shell to save voxels).
    s.fill((slice(0, 120), slice(0, 40), slice(0, 39)), M.PLASTIC_WHITE)
    s.clear((slice(1, 119), slice(1, 39), slice(1, 38)))
    # 3 vertical drawer-front grooves on the FRONT face (y=0).
    for gx in (30, 60, 90):
        s.fill((slice(gx, gx + 2), 0, slice(2, 37)), M.METAL_DARK)
    # Cup-pull handles on drawers.
    for gx in (30, 60, 90):
        s.set(gx + 1, 0, 25, M.METAL_HOUSING)
    # Recessed kick-plate.
    for z in range(0, 4):
        s.clear((slice(None), 0, z))
        s.fill((slice(None), 1, z), M.PLASTIC_WHITE)
    return s


@stamp_cached(8)
def build_pcr_machine(rng: np.random.Generator) -> Stamp:
    """BioRad C1000 Touch PCR thermocycler — 22 × 18 × 30 (0.44 × 0.36 × 0.60 m).

    Compact countertop with a sloped front carrying a big amber touchscreen,
    hinged top lid with finger pull, side ventilation slits.
    """
    s = Stamp((22, 18, 30))
    # Body shell.
    s.fill((slice(None), slice(None), slice(None)), M.PLASTIC_WHITE)
    s.clear((slice(1, 21), slice(1, 17), slice(1, 29)))
    # Sloped front face — push the top 4 z-layers back 2 voxels.
    for dz in range(0, 4):
        for x in range(22):
            for y_offset in range(2):
                # Clear the front face at increasing y
                s.clear((x, y_offset, 22 + dz))
            # Re-fill at offset back.
            s.set(x, 2, 22 + dz, M.PLASTIC_WHITE)
    # Touchscreen on the sloped face (z=10..22, y=0).
    s.fill((slice(5, 17), 0, slice(10, 22)), M.EM_AMBER)
    # Sparse darker icon voxels.
    for ix in range(6, 16, 2):
        for iz in range(11, 21, 2):
            if (ix + iz) % 3 == 0:
                s.set(ix, 0, iz, M.PLASTIC_DARK)
    # Hinged top lid (visible step at z=22..27).
    s.fill((slice(None), slice(None), slice(22, 27)), M.METAL_HOUSING)
    s.clear((slice(1, 21), slice(1, 17), slice(23, 27)))
    # Finger pull at front of lid.
    s.fill((slice(8, 14), 0, slice(23, 25)), M.PLASTIC_DARK)
    # Side ventilation slits.
    for y_side in (0, 17):
        for sz in (4, 8, 12):
            for x in range(4, 18):
                if (x + sz) % 2 == 0:
                    s.clear((x, y_side, sz))
    # Power LED at bottom-front.
    s.set(1, 0, 2, M.EM_GREEN)
    return s


@stamp_cached(8)
def build_microscope(rng: np.random.Generator) -> Stamp:
    """Olympus BX-style compound microscope — 10 × 14 × 25 (0.20 × 0.28 × 0.50 m).

    Wide base + C-shaped arm with overhang carrying eyepieces + objective
    turret + stage with sample slide + focus knobs + sub-stage illumination.
    """
    s = Stamp((10, 14, 25))
    # Base.
    s.fill((slice(0, 10), slice(0, 14), slice(0, 3)), M.PLASTIC_WHITE)
    # C-arm: 4x4 column rising from back-left.
    s.fill((slice(0, 4), slice(10, 14), slice(3, 21)), M.PLASTIC_WHITE)
    # Top overhang curving forward.
    s.fill((slice(0, 4), slice(4, 10), slice(18, 21)), M.PLASTIC_WHITE)
    # Binocular eyepiece blocks (angled forward at the tip).
    s.fill((slice(0, 2), slice(2, 6), slice(20, 23)), M.METAL_DARK)
    s.fill((slice(2, 4), slice(2, 6), slice(20, 23)), M.METAL_DARK)
    # Objective turret (disc) under the arm at z=12.
    cxm = 2
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            r2 = dx * dx + dy * dy
            if r2 <= 4:
                xx, yy = cxm + dx, 7 + dy
                if 0 <= xx < 10 and 0 <= yy < 14:
                    s.set(xx, yy, 12, M.METAL_DARK)
    # 4 objective lens pegs with cyan tips.
    for i, (lx, ly) in enumerate(((cxm + 2, 7), (cxm - 2, 7), (cxm, 9), (cxm, 5))):
        if 0 <= lx < 10 and 0 <= ly < 14:
            s.set(lx, ly, 11, M.METAL_DARK)
            s.set(lx, ly, 10, M.EM_CYAN)
    # Stage at z=8 (under the objective).
    s.fill((slice(0, 6), slice(4, 10), 8), M.PLASTIC_WHITE)
    # Sample slide.
    s.fill((slice(1, 5), slice(5, 9), 9), M.PLASTIC_DARK)
    # Stage clips.
    s.set(0, 4, 8, M.METAL_DARK)
    s.set(5, 9, 8, M.METAL_DARK)
    # Focus knobs on right side of arm.
    for kz in (10, 13):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx * dx + dy * dy <= 1:
                    s.set(4 + dx, 11 + dy, kz, M.METAL_DARK)
    # Sub-stage illumination.
    s.set(2, 7, 3, M.EM_WHITE)
    return s


@stamp_cached(8)
def build_lab_freezer(rng: np.random.Generator) -> Stamp:
    """Thermo Revco -80 °C upright freezer — 38 × 38 × 115 (0.76 × 0.76 × 2.30 m).

    Off-white industrial cabinet with a heavy latched door, top control
    panel with amber LCD + 4 status LEDs, frost-ring door gasket, kick-plate,
    3 visible hinge pegs on the left.
    """
    s = Stamp((38, 38, 115))
    # Body shell.
    s.fill((slice(None), slice(None), slice(None)), M.PLASTIC_WHITE)
    s.clear((slice(1, 37), slice(1, 37), slice(1, 114)))
    # Bottom kick-plate.
    for z in range(0, 4):
        s.clear((slice(None), 0, z))
        s.fill((slice(None), 1, z), M.PLASTIC_WHITE)
    # Door panel (METAL_DARK frame + PLASTIC_WHITE door).
    door_z0, door_z1 = 6, 92
    # Frame (1-voxel METAL_DARK ring around the door).
    s.fill((slice(0, 38), 0, slice(door_z0 - 1, door_z0)), M.PLASTIC_DARK)
    s.fill((slice(0, 38), 0, slice(door_z1, door_z1 + 1)), M.PLASTIC_DARK)
    s.fill((0, 0, slice(door_z0, door_z1)), M.PLASTIC_DARK)
    s.fill((37, 0, slice(door_z0, door_z1)), M.PLASTIC_DARK)
    # Door panel inset 1 voxel.
    s.fill((slice(1, 37), 1, slice(door_z0, door_z1)), M.PLASTIC_WHITE)
    # Latched handle (vertical bar with two bracket dots).
    s.fill((34, 0, slice(40, 65)), M.METAL_DARK)
    s.set(34, 0, 40, M.METAL_HOUSING)  # top bracket
    s.set(34, 0, 64, M.METAL_HOUSING)  # bottom bracket
    # 3 hinge pegs on left edge.
    for hz in (15, 45, 75):
        s.set(0, 1, hz, M.METAL_DARK)
    # Top control panel (z=95..103).
    s.fill((slice(2, 36), 0, slice(95, 103)), M.PLASTIC_DARK)
    # Amber LCD strip with -80.0°C readout.
    s.fill((slice(6, 18), 0, slice(97, 100)), M.EM_AMBER)
    # 4 status LEDs in a row.
    for i, mat in enumerate((M.EM_CYAN, M.EM_GREEN, M.EM_AMBER, M.EM_RED)):
        s.set(22 + i * 3, 0, 99, mat)
    # Brand plate strip.
    s.fill((slice(10, 26), 0, slice(105, 108)), M.PLASTIC_DARK)
    return s


@stamp_cached(8)
def build_glass_display_cabinet(rng: np.random.Generator) -> Stamp:
    """Blade Runner backlit glass display cabinet — 28 × 22 × 130
    (0.56 × 0.44 × 2.6 m).

    All-glass tower on a heavy PLASTIC_DARK plinth with 4 thin METAL_DARK
    corner posts, internal EM_SEPIA backlight strip along the back wall,
    specimen on a stone pedestal inside (random walk in EM_BIO_GREEN),
    bottom label plate.
    """
    s = Stamp((28, 22, 130))
    # Plinth (heavy stone-look base).
    s.fill((slice(None), slice(None), slice(0, 12)), M.PLASTIC_DARK)
    # 1-voxel-darker top reveal of plinth.
    s.fill((slice(None), slice(None), 12), M.METAL_DARK)
    # 4 corner posts.
    for (px, py) in ((0, 0), (27, 0), (0, 21), (27, 21)):
        s.fill((px, py, slice(12, 128)), M.METAL_DARK)
    # Glass walls.
    glass_z0, glass_z1 = 13, 127
    for z in range(glass_z0, glass_z1):
        s.fill((slice(1, 27), 0, z), M.GLASS_TANK)
        s.fill((slice(1, 27), 21, z), M.GLASS_TANK)
        s.fill((0, slice(1, 21), z), M.GLASS_TANK)
        s.fill((27, slice(1, 21), z), M.GLASS_TANK)
    # Top cap.
    s.fill((slice(None), slice(None), slice(128, 130)), M.PLASTIC_DARK)
    # Internal sepia backlight on back wall (1-voxel-thick strip full height).
    s.fill((slice(2, 26), 20, slice(glass_z0, glass_z1)), M.EM_SEPIA)
    # Pedestal inside.
    s.fill((slice(10, 18), slice(8, 14), slice(15, 19)), M.PLASTIC_DARK)
    # Specimen via random walk (8-15 voxels in EM_BIO_GREEN).
    n_spec = int(rng.integers(8, 16))
    sx, sy, sz = 14, 11, 19
    for _ in range(n_spec):
        s.set(int(sx), int(sy), int(sz), M.EM_BIO_GREEN)
        sx += int(rng.integers(-1, 2))
        sy += int(rng.integers(-1, 2))
        sz += int(rng.integers(0, 2))
        if not (10 <= sx <= 17 and 8 <= sy <= 13 and 19 <= sz <= 50):
            sx, sy, sz = 14, 11, 19
    # Bottom label plate.
    s.fill((slice(8, 16), 0, slice(13, 15)), M.EM_AMBER)
    return s


@stamp_cached(8)
def build_lab_shelf_rack(rng: np.random.Generator) -> Stamp:
    """Open industrial lab shelving — 32 × 25 × 130 (0.64 × 0.50 × 2.60 m).

    4 corner posts + 5 horizontal shelves, each shelf carrying RNG-driven
    lab consumables (bottles with coloured caps, boxes with labels, beakers
    in glass, petri-dish stacks, pipette tip racks).
    """
    s = Stamp((32, 25, 130))
    # 4 corner posts.
    for (px, py) in ((0, 0), (31, 0), (0, 24), (31, 24)):
        s.fill((px, py, slice(0, 130)), M.METAL_DARK)
    # 5 horizontal shelves.
    shelf_levels = (20, 45, 70, 95, 120)
    for sz in shelf_levels:
        s.fill((slice(None), slice(None), sz), M.METAL_HOUSING)
        # Front-edge darker band.
        s.fill((slice(None), 0, sz), M.METAL_DARK)
    # Per-shelf items.
    item_choices = (
        ("bottle", M.METAL_HOUSING, (M.PAINTED_RED, M.PAINTED_YELLOW, M.PAINTED_CYAN)),
        ("box", M.PLASTIC_WHITE, (M.PAINTED_RED, M.PAINTED_YELLOW, M.PAINTED_CYAN)),
        ("beaker", M.GLASS_TANK, None),
        ("petri", M.PLASTIC_WHITE, None),
        ("pipette", M.PLASTIC_DARK, None),
    )
    for sz in shelf_levels:
        n_items = int(rng.integers(3, 6))
        x = 3
        for _ in range(n_items):
            if x > 28:
                break
            kind, body_mat, accent_choices = item_choices[
                int(rng.integers(0, len(item_choices)))
            ]
            if kind == "bottle":
                s.fill((slice(x, x + 1), slice(8, 9), slice(sz + 1, sz + 5)), body_mat)
                cap = accent_choices[int(rng.integers(0, 3))]
                s.set(x, 8, sz + 5, cap)
                x += 3
            elif kind == "box":
                s.fill((slice(x, x + 4), slice(6, 9), slice(sz + 1, sz + 3)), body_mat)
                lab = accent_choices[int(rng.integers(0, 3))]
                s.set(x + 1, 6, sz + 2, lab)
                x += 6
            elif kind == "beaker":
                s.fill((slice(x, x + 2), slice(8, 10), slice(sz + 1, sz + 4)), body_mat)
                x += 4
            elif kind == "petri":
                s.fill((slice(x, x + 3), slice(7, 10), slice(sz + 1, sz + 4)), body_mat)
                x += 5
            elif kind == "pipette":
                s.fill((slice(x, x + 4), slice(6, 8), slice(sz + 1, sz + 5)), body_mat)
                # Darker grid pattern on top face.
                for gx in range(x, x + 4, 1):
                    if gx % 2 == 0:
                        s.set(gx, 6, sz + 5, M.METAL_DARK)
                x += 6
    return s


# ============================================================================
# Lobby decorative pendant lamp + Future Corp corporate glass door
# ============================================================================


@stamp_cached(4)
def build_decorative_pendant(rng: np.random.Generator) -> Stamp:
    """Tom-Dixon-Beat-inspired warm pendant for the lobby ceiling.

    30 × 30 × 50 envelope: a thin brushed-brass stem + a flared tulip-shaped
    bell that's wider at the bottom than the top. The bell is a 1-voxel-thick
    shell so the interior backlight (EM_WARM_PENDANT disc at the bottom) shows.
    """
    s = Stamp((30, 30, 50))
    # Stem (40 voxels tall) at the centre.
    cxp, cyp = 15, 15
    s.fill((cxp, cyp, slice(12, 50)), M.METAL_HOUSING)
    # Ceiling-rose mount plate at z=49.
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            s.set(cxp + dx, cyp + dy, 49, M.METAL_DARK)
    # Bell: stepped chamfer wider at the bottom.
    # z=0..4: 30x30 wide (octagonal-ish).
    # z=4..8: narrower.
    # z=8..12: narrower still.
    shells = (
        (0, 4, 13),
        (4, 8, 10),
        (8, 12, 7),
    )
    for z0, z1, half in shells:
        # 1-voxel-thick shell of the octagonal ring at this z range.
        for z in range(z0, z1):
            for dx in range(-half, half + 1):
                for dy in range(-half, half + 1):
                    # Octagonal mask: knock corners.
                    if abs(dx) > half - half // 3 and abs(dy) > half - half // 3:
                        continue
                    # Only set if on the rim (boundary of the octagon).
                    on_rim = (
                            abs(dx) == half or abs(dy) == half
                            or (abs(dx) == half - 1 and abs(dy) == half - 1)
                    )
                    if on_rim:
                        s.set(cxp + dx, cyp + dy, z, M.METAL_HOUSING)
    # Interior emissive disc at z=0 (the open bottom of the bell).
    for dx in range(-12, 13):
        for dy in range(-12, 13):
            if abs(dx) <= 11 and abs(dy) <= 11:
                # Skip the very corners to give the disc a soft octagonal shape.
                if abs(dx) > 8 and abs(dy) > 8:
                    continue
                s.set(cxp + dx, cyp + dy, 0, M.EM_WARM_PENDANT)
    return s


def _draw_f_logo(stamp: Stamp, ox: int, oy: int, oz: int,
                 mat: str = M.EM_AMBER) -> None:
    """Paint the Future Corp "F" pictograph (8 wide × 1 deep × 8 tall) into
    ``stamp`` starting at fine-voxel ``(ox, oy, oz)``.
    """
    # F pattern: 8x8 bitmap stored as rows top-down.
    F_BITMAP = (
        "11111100",  # z = top (7)
        "10000000",  # (6)
        "10000000",  # (5)
        "11111000",  # (4)  middle arm
        "10000000",  # (3)
        "10000000",  # (2)
        "11000000",  # (1)  bottom kicker
        "10000000",  # (0)
    )
    for i, row in enumerate(F_BITMAP):
        z = oz + (7 - i)
        for dx, c in enumerate(row):
            if c == "1":
                stamp.set(ox + dx, oy, z, mat)


@stamp_cached(4)
def build_corporate_glass_door(rng: np.random.Generator) -> Stamp:
    """Future Corp grand entrance — 240 × 6 × 120 (4.8 × 0.12 × 2.4 m).

    A wider double-door entrance with 4 glass panels divided by 3 vertical
    metal mullions (centre divider thicker than the two side dividers).
    The two centre panels act as the operating door slabs (with pull
    handles); the two outer panels are fixed sidelights. EM_AMBER lintel
    strip across the full width, cyan trim along each panel's inside
    edges, side-mounted card reader and camera dome.

    The Future Corp logotype lives on the wall behind/above the door
    (placed separately by :func:`emit_lobby`), NOT on the door itself.
    """
    NX = 240
    NZ = 120
    s = Stamp((NX, 6, NZ))

    # Outer frame (4-voxel-thick black metal).
    s.fill((slice(0, 4), slice(0, 6), slice(0, NZ)), M.RUBBER_BLACK)
    s.fill((slice(NX - 4, NX), slice(0, 6), slice(0, NZ)), M.RUBBER_BLACK)
    s.fill((slice(0, NX), slice(0, 6), slice(NZ - 8, NZ)), M.RUBBER_BLACK)
    s.fill((slice(0, NX), slice(0, 6), slice(0, 4)), M.RUBBER_BLACK)

    # 4-panel layout: side frame + 4 glass panels + 3 mullions
    # Panel width: (240 - 8 - 6)/4 = 56.5, round to 56
    # Layout: 0..4 frame | 4..60 P1 | 60..62 div | 62..118 P2 | 118..122 mid
    #         | 122..178 P3 | 178..180 div | 180..236 P4 | 236..240 frame
    slab_z0, slab_z1 = 8, NZ - 10
    panel_ranges = ((4, 60), (62, 118), (122, 178), (180, 236))
    for x0, x1 in panel_ranges:
        s.fill((slice(x0, x1), 2, slice(slab_z0, slab_z1)), M.GLASS_TANK)
        # Cyan trim along top and bottom edges of each panel.
        s.fill((slice(x0, x1), 2, slab_z0), M.EM_CYAN)
        s.fill((slice(x0, x1), 2, slab_z1 - 1), M.EM_CYAN)
        # Cyan trim along the inside vertical edges too.
        s.fill((x0, 2, slice(slab_z0, slab_z1)), M.EM_CYAN)
        s.fill((x1 - 1, 2, slice(slab_z0, slab_z1)), M.EM_CYAN)

    # Side mullions (2 voxels thick).
    for div_x0, div_x1 in ((60, 62), (178, 180)):
        s.fill((slice(div_x0, div_x1), slice(0, 6), slice(4, NZ - 6)),
               M.METAL_DARK)
    # Centre mullion: thicker (4 voxels) so the centre divider reads as
    # the door-pair junction.
    s.fill((slice(118, 122), slice(0, 6), slice(4, NZ - 6)), M.METAL_DARK)

    # Pull handles on the TWO CENTRE panels (the operating door slabs).
    # Left slab (P2 at 62..118): handle on the right side near the centre
    # divider. Right slab (P3 at 122..178): handle on the left side.
    handle_z0, handle_z1 = 48, 64
    s.fill((slice(106, 110), 1, slice(handle_z0, handle_z1)), M.METAL_HOUSING)
    s.fill((slice(130, 134), 1, slice(handle_z0, handle_z1)), M.METAL_HOUSING)

    # Lintel header amber strip across the full width.
    s.fill((slice(8, NX - 8), 2, NZ - 10), M.EM_AMBER)

    # Card reader on the +X jamb.
    s.fill((slice(NX - 4, NX - 2), 2, slice(50, 56)), M.EM_CYAN)

    # Small camera dome above the lintel near the +X corner.
    for dx in range(-1, 2):
        for dy in range(0, 2):
            for dz in range(0, 2):
                s.set(NX - 12 + dx, dy, NZ - 4 + dz, M.PLASTIC_DARK)
    return s


@stamp_cached(2)
def build_future_corp_wall_logo(rng: np.random.Generator) -> Stamp:
    """Wall-mounted "FUTURE / CORP" logotype — 4 deep × 200 wide × 140 tall
    fine voxels (0.08 × 4.0 × 2.8 m).

    Designed for a SIDE wall (mounted on the east perimeter wall of the
    lobby). Three layers along the stamp's X axis (= wall-normal depth):

    * **x = 0..1** — opaque ``METAL_DARK`` letter glyphs facing the viewer.
    * **x = 2** — air gap.
    * **x = 3** — large ``EM_WHITE`` backplate (slightly bigger than the
      letter cluster). Its light leaks AROUND the letters to give the
      diffuse-letter-with-halo look from the user's reference image; the
      letters themselves emit nothing.

    Stamp X = wall-normal depth, Y = horizontal along the wall, Z = vertical.

    Layout is **stacked** — "FUTURE" on the top line, "CORP" on the
    bottom line. Glyphs are 7-col × 10-row futuristic bold bitmaps
    rendered at 4 × 4 fine voxels per pixel, so each letter is 28 × 40
    fine voxels — roughly 2× larger than the previous single-line version.
    """
    # BOLD futuristic 7-col × 10-row bitmap glyphs. Strokes are 2-pixel
    # thick within the bitmap so each letter reads as a solid bold mark
    # rather than a thin outline.
    GLYPHS = {
        "F": (
            "1111111",  # top arm (full width)
            "1111111",
            "1100000",
            "1100000",
            "1111110",  # middle arm
            "1111110",
            "1100000",
            "1100000",
            "1100000",
            "1100000",
        ),
        "U": (
            "1100011",
            "1100011",
            "1100011",
            "1100011",
            "1100011",
            "1100011",
            "1100011",
            "1100011",
            "1111111",
            "0111110",
        ),
        "T": (
            "1111111",
            "1111111",
            "0011100",
            "0011100",
            "0011100",
            "0011100",
            "0011100",
            "0011100",
            "0011100",
            "0011100",
        ),
        "R": (
            "1111110",
            "1111111",
            "1100011",
            "1100011",
            "1111110",
            "1111100",
            "1101100",
            "1100110",
            "1100011",
            "1100011",
        ),
        "E": (
            "1111111",
            "1111111",
            "1100000",
            "1100000",
            "1111110",
            "1111110",
            "1100000",
            "1100000",
            "1111111",
            "1111111",
        ),
        "C": (
            "0111110",
            "1111111",
            "1100011",
            "1100000",
            "1100000",
            "1100000",
            "1100000",
            "1100011",
            "1111111",
            "0111110",
        ),
        "O": (
            "0111110",
            "1111111",
            "1100011",
            "1100011",
            "1100011",
            "1100011",
            "1100011",
            "1100011",
            "1111111",
            "0111110",
        ),
        "P": (
            "1111110",
            "1111111",
            "1100011",
            "1100011",
            "1111110",
            "1111110",
            "1100000",
            "1100000",
            "1100000",
            "1100000",
        ),
        " ": tuple(["0000000"] * 10),
    }

    NX = 4  # depth (0 = lobby-facing, 3 = wall-facing)
    NY = 200  # width along the wall (4.0 m)
    NZ = 140  # height (2.8 m)
    s = Stamp((NX, NY, NZ))

    PX = 4  # fine voxels per glyph pixel
    GW = 7 * PX  # 28 fine voxels per glyph width
    GH = 10 * PX  # 40 fine voxels per glyph height
    GAP = PX  # 4 fine voxels between glyphs
    LINE_GAP = 2 * PX  # 8 fine voxels vertical gap between the two text lines

    lines = ("FUTURE", "CORP")
    line_widths = tuple(len(t) * GW + (len(t) - 1) * GAP for t in lines)
    block_h = len(lines) * GH + (len(lines) - 1) * LINE_GAP
    margin_z = (NZ - block_h) // 2

    # --- Backplate (EM_WHITE) at x = NX - 1, very large halo around the
    # letter block so the wall reads as a big glowing panel with dark
    # letter silhouettes on it (rather than letters with a faint outline).
    # Backplate spans almost the full stamp area.
    HALO_Y = 30  # 0.6 m halo on each side along Y
    HALO_Z = 16  # 0.32 m halo on top + bottom along Z
    bp_w = max(line_widths) + 2 * HALO_Y
    bp_y0 = max(0, (NY - bp_w) // 2)
    bp_y1 = min(NY, bp_y0 + bp_w)
    bp_z0 = max(0, margin_z - HALO_Z)
    bp_z1 = min(NZ, margin_z + block_h + HALO_Z)
    s.fill((NX - 1, slice(bp_y0, bp_y1), slice(bp_z0, bp_z1)), M.EM_WHITE)

    # --- Diffuse METAL_DARK letters at x = 0..2 (3 voxels thick) -------
    # Logo mounts on the east perimeter wall facing -X. Viewer in the lobby
    # looks +X with +Z up, so viewer's RIGHT = forward × up = (+X)×(+Z) = -Y.
    # Two mirrorings are needed so the text reads correctly:
    #   * GLYPH ORDER per line: "F" at HIGH Y, last char at LOW Y.
    #   * WITHIN-GLYPH COLUMNS: col_i=0 sits at HIGH-Y end of its slot, so
    #     each letter shape is mirrored along Y as well — otherwise the
    #     letters render correctly-ordered but back-to-front.
    for line_i, line_text in enumerate(lines):
        total_w = line_widths[line_i]
        line_margin_y = (NY - total_w) // 2
        text_end_y = NY - line_margin_y
        # Top line ("FUTURE") sits at high Z; bottom line ("CORP") at low Z.
        line_z_bottom = margin_z + (len(lines) - 1 - line_i) * (GH + LINE_GAP)
        for ci, ch in enumerate(line_text):
            glyph = GLYPHS.get(ch, GLYPHS[" "])
            y0_glyph = text_end_y - (ci + 1) * GW - ci * GAP
            for row_i, row in enumerate(glyph):
                for col_i, c in enumerate(row):
                    if c != "1":
                        continue
                    gy0 = y0_glyph + (7 - 1 - col_i) * PX
                    gz0 = line_z_bottom + (9 - row_i) * PX
                    s.fill((slice(0, 3),
                            slice(gy0, gy0 + PX),
                            slice(gz0, gz0 + PX)), M.METAL_DARK)
    return s


@stamp_cached(0)
def build_telephone(rng: np.random.Generator) -> Stamp:
    """Reception desk telephone — 8 × 6 × 4 fine voxels.

    Cradle base + curved handset + 3x3 number-pad button grid + EM_CYAN
    status LED + zig-zag coiled cord in EM_AMBER for flair.
    """
    s = Stamp((8, 6, 5))
    # Cradle base.
    s.fill((slice(0, 8), slice(0, 6), slice(0, 2)), M.PLASTIC_DARK)
    # Round the 4 top corners.
    s.clear((0, 0, 1))
    s.clear((7, 0, 1))
    s.clear((0, 5, 1))
    s.clear((7, 5, 1))
    # Handset on top.
    s.fill((slice(1, 6), slice(0, 2), 2), M.PLASTIC_DARK)
    # Mouth/earpiece bulges.
    s.set(1, 0, 3, M.PLASTIC_DARK)
    s.set(5, 0, 3, M.PLASTIC_DARK)
    # Number pad 3x3.
    for ix in range(3):
        for iz in range(3):
            s.set(2 + ix, 3, ix + iz, M.PLASTIC_WHITE)
    # Status LED.
    s.set(6, 3, 1, M.EM_CYAN)
    return s


@stamp_cached(0)
def build_keyboard(rng: np.random.Generator) -> Stamp:
    """A 14 x 4 x 2 keyboard with a visible 6x3 key grid + space bar + caps LED.

    The body is a slight wedge (z=0..1 baseline, z=1 raised at the back) so
    the side silhouette reads NOT as a flat brick.
    """
    s = Stamp((14, 4, 2))
    # Body wedge.
    s.fill((slice(None), slice(None), 0), M.PLASTIC_DARK)
    s.fill((slice(None), slice(2, 4), 1), M.PLASTIC_DARK)
    # 6x3 key grid inset 1 voxel into the body.
    for kx in range(6):
        for ky in range(3):
            s.set(1 + kx * 2, ky, 1, M.PLASTIC_WHITE)
    # Wider space bar replacing the leftmost 4 keys in the bottom row.
    s.fill((slice(1, 9), 0, 1), M.PLASTIC_WHITE)
    # Caps-lock LED at top-right.
    s.set(12, 3, 1, M.EM_GREEN)
    return s


# ============================================================================
# Bigger reactor core (round 3)
# ============================================================================


@stamp_cached(2)
def build_reactor_core_large(rng: np.random.Generator) -> Stamp:
    """Scaled-up reactor — 130 × 130 × 180 (2.6 m × 2.6 m × 3.6 m envelope).

    Same five-feature framework as ``build_reactor_core``: fluted octagonal
    column with slit-lights, inner emissive shell, horizontal cooling fins,
    crenellated top, base indicator ring, radial pipes, 4 containment posts
    with cable arcs, glowing floor halo.

    Used by the reactor biome in the 3×3-cell reactor chamber — it visually
    fills more of the room than the v1 stamp.
    """
    NX = 130
    s = Stamp((NX, NX, 180))
    cx, cy = NX // 2, NX // 2
    base_z = 0

    # Glowing emissive floor halo (annulus radius ~50..82).
    for dx in range(-82, 83):
        for dy in range(-82, 83):
            r2 = dx * dx + dy * dy
            if 50 * 50 <= r2 <= 82 * 82:
                xx, yy = cx + dx, cy + dy
                if 0 <= xx < NX and 0 <= yy < NX:
                    s.set(xx, yy, base_z, M.EM_REACTOR_FLOOR_RING)

    # Outer fluted octagonal column (radius 42, 2-voxel-thick shell).
    col_offsets = octagon_offsets(42)
    inner_offsets_set = set(octagon_offsets(40))
    rim_offsets = [(dx, dy) for (dx, dy) in col_offsets if (dx, dy) not in inner_offsets_set]
    column_top = 130
    for z in range(base_z, column_top):
        for dx, dy in rim_offsets:
            xx, yy = cx + dx, cy + dy
            if 0 <= xx < NX and 0 <= yy < NX:
                s.set(xx, yy, z, M.METAL_HOUSING)

    # Vertical flutes — 8 angular positions, 1-voxel rib outside the column.
    flute_angles = [(42, 0), (-42, 0), (0, 42), (0, -42),
                    (29, 29), (-29, 29), (29, -29), (-29, -29)]
    for fdx, fdy in flute_angles:
        for z in range(base_z, column_top):
            xx, yy = cx + fdx, cy + fdy
            if 0 <= xx < NX and 0 <= yy < NX:
                s.set(xx, yy, z, M.METAL_HOUSING)

    # Vertical slits between flutes (8 angular positions, narrow gaps).
    slit_angles = [(35, 21), (-35, 21), (35, -21), (-35, -21),
                   (21, 35), (-21, 35), (21, -35), (-21, -35)]
    for sdx, sdy in slit_angles:
        for z in range(base_z + 12, base_z + 100):
            xx, yy = cx + sdx, cy + sdy
            if 0 <= xx < NX and 0 <= yy < NX:
                s.clear((xx, yy, z))

    # Inner emissive shell on a 34x34x130 column (1-voxel walls + top cap).
    iz0, iz1 = base_z + 5, base_z + 135
    for z in range(iz0, iz1):
        for dx in range(-17, 18):
            s.set(cx + dx, cy - 17, z, M.EM_REACTOR)
            s.set(cx + dx, cy + 17, z, M.EM_REACTOR)
        for dy in range(-17, 18):
            s.set(cx - 17, cy + dy, z, M.EM_REACTOR)
            s.set(cx + 17, cy + dy, z, M.EM_REACTOR)
    # Top cap.
    for dx in range(-17, 18):
        for dy in range(-17, 18):
            s.set(cx + dx, cy + dy, iz1, M.EM_REACTOR)

    # Horizontal cooling fins at z=78, 86, 94, 102.
    for fz in (78, 86, 94, 102):
        for dx, dy in octagon_offsets(56):
            xx, yy = cx + dx, cy + dy
            if 0 <= xx < NX and 0 <= yy < NX:
                s.set(xx, yy, fz, M.METAL_HOUSING)

    # Crenellated top at z=130..142 (alternating 6x6 metal blocks).
    cren_z0 = column_top
    for ang_i, (adx, ady) in enumerate(flute_angles):
        if ang_i % 2 == 0:
            for z in range(cren_z0, cren_z0 + 12):
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        s.set(cx + adx + dx, cy + ady + dy, z, M.METAL_HOUSING)

    # Octagonal dome cap.
    for dx, dy in octagon_offsets(29):
        for dz in range(0, 12):
            s.set(cx + dx, cy + dy, cren_z0 + 12 + dz, M.METAL_HOUSING)

    # Base indicator ring (8 LEDs around the column at z=3..4).
    ind_colours = (M.EM_RED, M.EM_AMBER, M.EM_GREEN, M.EM_AMBER)
    for i, (idx, idy) in enumerate(((48, 0), (-48, 0), (0, 48), (0, -48),
                                    (34, 34), (-34, 34), (34, -34), (-34, -34))):
        for dz in (3, 4):
            s.set(cx + idx, cy + idy, dz, ind_colours[i % 4])

    # 6 radial pipes leaving the base, L-bending into the floor.
    pipe_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    for pdx, pdy in pipe_dirs:
        for d in range(38):
            x = cx + pdx * (45 + d) - (1 if abs(pdy) else 0)
            y = cy + pdy * (45 + d) - (1 if abs(pdx) else 0)
            if 0 <= x < NX and 0 <= y < NX:
                s.set(int(x), int(y), base_z + 3, M.METAL_DARK)
                s.set(int(x), int(y), base_z + 4, M.METAL_DARK)

    # 4 containment posts at radius 56.
    for pdx, pdy in ((56, 0), (-56, 0), (0, 56), (0, -56)):
        px, py = cx + pdx, cy + pdy
        if not (2 < px < NX - 2 and 2 < py < NX - 2):
            continue
        for z in range(base_z, base_z + 140):
            s.fill((slice(px - 1, px + 2),
                    slice(py - 1, py + 2),
                    z), M.METAL_DARK)
        # Capacitor block on top.
        s.fill((slice(px - 2, px + 3),
                slice(py - 2, py + 3),
                slice(base_z + 140, base_z + 146)), M.METAL_HOUSING)
        # Indicator LED.
        s.set(px, py, base_z + 146, M.EM_RED)

    # Cable arcs between containment posts.
    arc_z = base_z + 140
    post_positions = [(cx + 56, cy), (cx, cy + 56), (cx - 56, cy), (cx, cy - 56)]
    for i in range(4):
        p1 = post_positions[i]
        p2 = post_positions[(i + 1) % 4]
        steps = 18
        for t in range(steps + 1):
            tt = t / steps
            x = int(p1[0] * (1 - tt) + p2[0] * tt)
            y = int(p1[1] * (1 - tt) + p2[1] * tt)
            sag = -int(3 * np.sin(tt * np.pi))
            zz = arc_z + sag
            if 0 <= x < NX and 0 <= y < NX and 0 <= zz < 180:
                s.set(x, y, zz, M.RUBBER_BLACK)

    return s
