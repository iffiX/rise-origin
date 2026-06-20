# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Furniture: ergonomic chair, stool, desk, workbench, bunk bed, cafeteria table.

Fine 0.02 m grid. Each builder emphasises the silhouette over straight rectangles:
chairs have lumbar bumps, desks have inset drawer fronts, bunks have lumpy
mattresses and a draped blanket.
"""

from __future__ import annotations

import numpy as np

from .. import materials as M
from ..stamp import Stamp, stamp_cached


# ============================================================================
# Ergonomic office chair
# ============================================================================


@stamp_cached(8)
def build_chair(rng: np.random.Generator) -> Stamp:
    """A 5-arm wheeled office chair with a curved back and lumbar bump.

    Total height ~1.15 m (58 voxels). Footprint ~28x28 voxels.
    """
    s = Stamp((28, 28, 58))
    cx, cy = 14, 14

    # --- 5-arm wheeled base at z=0..2 ---
    # Approximate 5-armed star using 5 angles spaced 72°.
    arm_dirs = []
    for k in range(5):
        ang = 2 * np.pi * k / 5
        arm_dirs.append((np.cos(ang), np.sin(ang)))
    for ax_unit, ay_unit in arm_dirs:
        for d in range(0, 13):
            x = int(round(cx + ax_unit * d))
            y = int(round(cy + ay_unit * d))
            if 0 <= x < 28 and 0 <= y < 28:
                s.set(x, y, 0, M.METAL_DARK)
                if d > 0:  # 2-voxel-wide arm
                    s.set(x, y, 1, M.METAL_DARK)
        # Wheel ball at tip.
        wx = int(round(cx + ax_unit * 12))
        wy = int(round(cy + ay_unit * 12))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(0, 2):
                    if abs(dx) + abs(dy) <= 2:
                        xx, yy = wx + dx, wy + dy
                        if 0 <= xx < 28 and 0 <= yy < 28:
                            s.set(xx, yy, dz, M.RUBBER_BLACK)

    # --- Pedestal hydraulic cylinder ---
    # Wide collar at z=2..5.
    s.fill((slice(cx - 3, cx + 4), slice(cy - 3, cy + 4), slice(2, 5)),
           M.PLASTIC_DARK)
    # Cylinder z=5..17.
    s.fill((slice(cx - 2, cx + 3), slice(cy - 2, cy + 3), slice(5, 17)),
           M.PLASTIC_DARK)
    # Seam line at z=8.
    s.fill((slice(cx - 2, cx + 3), slice(cy - 2, cy + 3), 8), M.METAL_DARK)
    # Swivel plate at z=17.
    s.fill((slice(cx - 4, cx + 5), slice(cy - 4, cy + 5), 17), M.METAL_DARK)

    # --- Seat cushion (22x22x3 with corner knock-outs + chamfered front edge) ---
    seat_z0 = 18
    for x in range(cx - 11, cx + 11):
        for y in range(cy - 11, cy + 11):
            # Knock 4 corners (3-voxel chamfer).
            d_corner = min(
                (x - (cx - 11)) + (y - (cy - 11)),
                ((cx + 10) - x) + (y - (cy - 11)),
                (x - (cx - 11)) + ((cy + 10) - y),
                ((cx + 10) - x) + ((cy + 10) - y),
            )
            if d_corner < 2:
                continue
            s.fill((x, y, slice(seat_z0, seat_z0 + 3)), M.RUBBER_BLACK)
    # Chamfered front edge: front 2 voxels lower by 1.
    front_y0 = cy + 8
    for x in range(cx - 11, cx + 11):
        for y in range(front_y0, cy + 11):
            s.clear((x, y, seat_z0 + 2))

    # --- Curved back (z=22..46) ---
    back_y = cy - 10
    # Base of back (wide).
    for z in range(22, 26):
        s.fill((slice(cx - 9, cx + 9), slice(back_y, back_y + 3), z), M.PLASTIC_DARK)
    # Lumbar (narrower).
    for z in range(26, 32):
        s.fill((slice(cx - 8, cx + 8), slice(back_y, back_y + 3), z), M.PLASTIC_DARK)
    # Lumbar bump: extra layer protruding 1 voxel forward.
    for z in range(28, 32):
        s.fill((slice(cx - 5, cx + 5), back_y + 3, z), M.PLASTIC_DARK)
    # Upper back (widens back).
    for z in range(32, 42):
        s.fill((slice(cx - 10, cx + 10), slice(back_y, back_y + 3), z), M.PLASTIC_DARK)
    # Rounded top (knock 4 top corners).
    for z in range(42, 46):
        edge = 46 - z + 1
        for x in range(cx - 10 + edge, cx + 10 - edge):
            s.fill((x, slice(back_y, back_y + 3), z), M.PLASTIC_DARK)

    # --- Headrest ---
    head_z0 = 48
    for x in range(cx - 7, cx + 7):
        for z in range(head_z0, head_z0 + 8):
            if z < head_z0 + 7 or (cx - 6 <= x < cx + 6):
                s.fill((x, slice(back_y, back_y + 3), z), M.PLASTIC_DARK)

    # --- T-shaped armrests both sides ---
    for arm_x in (cx - 12, cx + 11):
        # Vertical post
        for z in range(seat_z0, seat_z0 + 6):
            s.set(arm_x, cy - 3, z, M.PLASTIC_DARK)
        # Padded armrest cap
        for x in range(arm_x - 1, arm_x + 2):
            for y in range(cy - 5, cy - 1):
                for z in range(seat_z0 + 6, seat_z0 + 8):
                    if 0 <= x < 28 and 0 <= y < 28:
                        s.set(x, y, z, M.PLASTIC_DARK)

    return s


# ============================================================================
# Lab stool
# ============================================================================


@stamp_cached(8)
def build_stool(rng: np.random.Generator) -> Stamp:
    """A stool with 4 legs, a footrest ring, and a round-cornered seat."""
    s = Stamp((18, 18, 23))
    # 4 legs at corners
    for lx, ly in ((1, 1), (16, 1), (1, 16), (16, 16)):
        s.fill((lx, ly, slice(0, 18)), M.METAL_DARK)
    # Foot ring at z=8 connecting all 4 legs.
    s.fill((slice(1, 17), 1, 8), M.METAL_DARK)
    s.fill((slice(1, 17), 16, 8), M.METAL_DARK)
    s.fill((1, slice(1, 17), 8), M.METAL_DARK)
    s.fill((16, slice(1, 17), 8), M.METAL_DARK)
    # Circular seat with knocked-out corners.
    seat_z = 19
    for x in range(2, 16):
        for y in range(2, 16):
            dx, dy = x - 9, y - 9
            if dx * dx + dy * dy <= 49:  # radius 7
                s.fill((x, y, slice(seat_z, seat_z + 3)), M.RUBBER_BLACK)
    return s


# ============================================================================
# Desk with drawer pedestal (+ optional monitor placed by caller)
# ============================================================================


@stamp_cached(8)
def build_desk(rng: np.random.Generator) -> Stamp:
    """A desk with inset drawer-front grooves, back kick plate, and a corner
    cable cutout in the top.
    """
    s = Stamp((50, 25, 38))
    # Top
    s.fill((slice(None), slice(None), slice(35, 38)), M.PLASTIC_WHITE)
    # Cable cutout in rear corner.
    s.clear((slice(45, 49), slice(20, 24), slice(35, 38)))
    # Drawer pedestal (left side) — shell only (hollow interior saves voxels).
    s.fill((slice(0, 20), slice(None), slice(0, 35)), M.PLASTIC_DARK)
    s.clear((slice(1, 19), slice(1, 24), slice(1, 34)))   # hollow interior
    # 3 horizontal drawer-front grooves.
    for gz in (10, 20, 30):
        s.fill((slice(2, 18), 0, gz), M.METAL_DARK)
        # Drawer pull (cup handle: 1-voxel-deep slot + 1x1 lip)
        s.set(9, 0, gz + 2, M.METAL_HOUSING)
        s.set(10, 0, gz + 2, M.METAL_HOUSING)
    # Back kick plate (recessed 1 voxel from front).
    s.fill((slice(20, 50), slice(None), slice(0, 4)), M.PLASTIC_DARK)
    # Far-side legs.
    for lx in (44, 48):
        s.fill((lx, slice(2, 5), slice(4, 35)), M.METAL_DARK)
        s.fill((lx, slice(20, 23), slice(4, 35)), M.METAL_DARK)
    return s


# ============================================================================
# Workbench with pegboard + visible tool silhouettes
# ============================================================================


@stamp_cached(8)
def build_workbench(rng: np.random.Generator) -> Stamp:
    """A workbench with a pegboard back, hung tool silhouettes, and under-storage."""
    s = Stamp((50, 25, 43))
    # Surface.
    s.fill((slice(None), slice(None), slice(40, 43)), M.PLASTIC_WHITE)
    # Tool rack (pegboard) at the back.
    s.fill((slice(None), slice(0, 3), slice(13, 38)), M.METAL_DARK)
    # Tool silhouettes (random placement).
    tools = [
        # wrench: 1x1x5 with 2x2 head
        (4, 0, 18, 1, 1, 5, M.METAL_HOUSING),
        (3, 0, 18, 2, 1, 2, M.METAL_HOUSING),
        # screwdriver
        (10, 0, 20, 1, 1, 4, M.METAL_HOUSING),
        (10, 0, 24, 2, 1, 1, M.PLASTIC_DARK),
        # hammer
        (16, 0, 22, 1, 1, 4, M.PLASTIC_DARK),
        (16, 0, 26, 3, 1, 1, M.METAL_HOUSING),
        # gauge
        (22, 0, 24, 4, 1, 4, M.METAL_HOUSING),
        # pliers
        (30, 0, 22, 1, 1, 5, M.METAL_HOUSING),
        (30, 0, 27, 2, 1, 1, M.METAL_HOUSING),
    ]
    for x, y, z, dx, dy, dz, mat in tools:
        s.fill((slice(x, x + dx), slice(y, y + dy), slice(z, z + dz)), mat)
    # Under-storage with horizontal drawer-line grooves — shell only.
    s.fill((slice(None), slice(None), slice(0, 40)), M.PLASTIC_DARK)
    s.clear((slice(1, 49), slice(1, 24), slice(1, 39)))   # hollow interior
    for gz in (12, 24, 36):
        s.fill((slice(2, 48), 0, gz), M.METAL_DARK)
    # Scattered tools on top.
    for _ in range(5):
        tx = int(rng.integers(4, 46))
        ty = int(rng.integers(4, 21))
        s.set(tx, ty, 43, M.METAL_HOUSING)
    # Bench vise at corner.
    s.fill((slice(2, 7), slice(8, 14), slice(43, 47)), M.METAL_HOUSING)
    s.fill((slice(3, 6), slice(9, 13), slice(45, 50)), M.METAL_DARK)
    return s


# ============================================================================
# Bunk bed (with frame headboard, lumpy mattress, draped blanket, safety rail)
# ============================================================================


@stamp_cached(8)
def build_bunk_bed(rng: np.random.Generator) -> Stamp:
    """A metal-frame bunk bed with a side ladder, safety rail with gap, lumpy
    mattresses, draped blanket, and a single pillow per bunk.
    """
    s = Stamp((50, 25, 90))

    # 4 corner posts
    for px, py in ((0, 0), (47, 0), (0, 22), (47, 22)):
        s.fill((slice(px, px + 3), slice(py, py + 3), slice(0, 90)), M.METAL_DARK)
        # Bolt-cap on top.
        s.set(px + 1, py + 1, 89, M.METAL_HOUSING)

    # Cross-bar headboards (head + foot ends) at z=18 and z=58.
    for hz in (18, 58):
        s.fill((slice(0, 50), slice(0, 3), slice(hz, hz + 6)), M.METAL_DARK)
        s.fill((slice(0, 50), slice(22, 25), slice(hz, hz + 6)), M.METAL_DARK)

    # Mattress frames (lower at z=20, upper at z=60).
    for fz in (20, 60):
        s.fill((slice(0, 50), slice(0, 25), slice(fz, fz + 2)), M.METAL_DARK)
        # Mattress on top (inset by 2).
        for x in range(2, 48):
            for y in range(2, 23):
                # Lumps
                lump = 1 if ((x * 17 + y * 31) % 23 == 0) else 0
                s.fill((x, y, slice(fz + 2, fz + 6 + lump)), M.PLASTIC_WHITE)
        # Pillow at one end (random side).
        pillow_x0 = 2 if rng.random() < 0.5 else 36
        s.fill((slice(pillow_x0, pillow_x0 + 12),
                slice(5, 20),
                slice(fz + 6, fz + 9)), M.PLASTIC_WHITE)
        # Tangled blanket overlapping the foot.
        blanket_x0 = pillow_x0 + 14 if pillow_x0 == 2 else 4
        for x in range(blanket_x0, blanket_x0 + 28):
            for y in range(3, 22):
                # Random folds (1-voxel bulges)
                bulge = 1 if ((x * 7 + y * 11) % 13 == 0) else 0
                if 0 <= x < 50:
                    s.fill((x, y, slice(fz + 6, fz + 7 + bulge)), M.FABRIC_BLANKET)

    # Safety rail on upper bunk (z=62..68, with a gap at one end).
    rail_x0 = 8 if rng.random() < 0.5 else 0
    s.fill((slice(rail_x0, rail_x0 + 32), 24, slice(62, 68)), M.METAL_DARK)
    # Side ladder.
    for rz in range(10, 62, 10):
        s.set(48, 0, rz, M.METAL_HOUSING)
        s.set(48, 1, rz, M.METAL_HOUSING)
        s.set(48, 0, rz + 1, M.METAL_HOUSING)
    # Ladder verticals
    s.fill((48, 0, slice(10, 62)), M.METAL_HOUSING)

    return s


# ============================================================================
# Cafeteria table with benches
# ============================================================================


@stamp_cached(8)
def build_sofa(rng: np.random.Generator) -> Stamp:
    """A 3-seater lobby sofa with wool-yellow upholstery.

    1.6 m x 0.7 m x 0.7 m footprint -> 80 x 35 x 35 fine voxels.
    Structural base + leg stubs stay PLASTIC_DARK; cushion + back + arm-rests
    are FABRIC_LOUNGE_YELLOW (soft wool) with 1-voxel-thick PLASTIC_DARK welt
    seams for stitched-upholstery read.
    """
    s = Stamp((80, 35, 35))
    # Structural base / under-cushion frame.
    s.fill((slice(None), slice(None), slice(0, 8)), M.PLASTIC_DARK)
    # Knock the bottom corners (legs poking through).
    for cx, cy in ((0, 0), (79, 0), (0, 34), (79, 34)):
        s.clear((cx, cy, slice(2, 8)))
    # 3 wool-yellow seat cushions on top (separated by 1-voxel grooves).
    cushion_xs = ((2, 26), (28, 52), (54, 78))
    for x0, x1 in cushion_xs:
        s.fill((slice(x0, x1), slice(2, 33), slice(8, 13)), M.FABRIC_LOUNGE_YELLOW)
        # Welt seam (1-voxel darker pinstripe along the cushion's front edge).
        s.fill((slice(x0, x1), 2, 8), M.PLASTIC_DARK)
        # Rounded top-back corners on each cushion.
        s.clear((x0, slice(2, 33), 12))
        s.clear((x1 - 1, slice(2, 33), 12))
    # Back rest (slightly leaning back -> stepped) in wool yellow.
    for dz in range(13):
        back_y_start = max(0, 28 - dz // 2)
        s.fill((slice(2, 78),
                slice(back_y_start, 33),
                13 + dz), M.FABRIC_LOUNGE_YELLOW)
    # Welt seam down the back-front edge.
    for dz in range(13):
        back_y_start = max(0, 28 - dz // 2)
        s.set(2, back_y_start, 13 + dz, M.PLASTIC_DARK)
        s.set(77, back_y_start, 13 + dz, M.PLASTIC_DARK)
    # Arm-rests in wool yellow.
    for ax in (0, 78):
        s.fill((slice(ax, ax + 2), slice(0, 33), slice(8, 18)),
               M.FABRIC_LOUNGE_YELLOW)
        # Welt seam on the arm-rest top.
        s.fill((slice(ax, ax + 2), slice(0, 33), 17), M.PLASTIC_DARK)
    return s


@stamp_cached(8)
def build_armchair(rng: np.random.Generator) -> Stamp:
    """A single-seat lobby armchair with wool-yellow upholstery."""
    s = Stamp((35, 35, 35))
    s.fill((slice(None), slice(None), slice(0, 8)), M.PLASTIC_DARK)
    # Seat cushion in wool yellow.
    s.fill((slice(2, 33), slice(2, 33), slice(8, 13)), M.FABRIC_LOUNGE_YELLOW)
    # Welt seam along the front cushion edge.
    s.fill((slice(2, 33), 2, 8), M.PLASTIC_DARK)
    # Back rest (stepped) in wool yellow.
    for dz in range(13):
        back_y_start = max(0, 28 - dz // 2)
        s.fill((slice(2, 33),
                slice(back_y_start, 33),
                13 + dz), M.FABRIC_LOUNGE_YELLOW)
    # Welt seam along the back-front edges.
    for dz in range(13):
        back_y_start = max(0, 28 - dz // 2)
        s.set(2, back_y_start, 13 + dz, M.PLASTIC_DARK)
        s.set(32, back_y_start, 13 + dz, M.PLASTIC_DARK)
    # Arm rests in wool yellow.
    for ax in (0, 33):
        s.fill((slice(ax, ax + 2), slice(0, 33), slice(8, 18)),
               M.FABRIC_LOUNGE_YELLOW)
        s.fill((slice(ax, ax + 2), slice(0, 33), 17), M.PLASTIC_DARK)
    return s


@stamp_cached(8)
def build_water_cooler(rng: np.random.Generator) -> Stamp:
    """A floor-standing water cooler: 0.3 m x 0.3 m x 1.1 m -> 15 x 15 x 55."""
    s = Stamp((15, 15, 55))
    # Main body.
    s.fill((slice(None), slice(None), slice(0, 48)), M.PLASTIC_WHITE)
    s.clear((slice(1, 14), slice(1, 14), slice(1, 47)))
    # Top water bottle (large transparent-ish dome -> use white plastic dome shape).
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            if dx * dx + dy * dy <= 36:
                for z in range(48, 55):
                    if dx * dx + dy * dy <= 36 - max(0, (z - 48) * 4):
                        s.set(7 + dx, 7 + dy, z, M.GLASS_TANK)
    # Hot + cold spigot (red + blue voxels on the front).
    s.set(6, 0, 25, M.EM_RED)    # hot
    s.set(8, 0, 25, M.EM_CYAN)   # cold
    # Drip tray below spigots.
    s.fill((slice(4, 11), 0, slice(20, 23)), M.METAL_DARK)
    return s


@stamp_cached(4)
def build_reception_desk(rng: np.random.Generator) -> Stamp:
    """A 2.8 m wooden curved reception desk — body + stone-look counter top
    only. Monitors, keyboards, telephone and mug are placed SEPARATELY by
    :func:`scene.biomes.emit_lobby` so they sit on top of the counter (not
    buried inside the desk's z envelope) and remain visible to the camera.

    Footprint 140 × 70 × 47 fine voxels (2.8 m × 1.4 m × 0.94 m). The front
    face curves toward the customer (concave from the customer side) via a
    3-step stairstep chamfer at each front corner.
    """
    s = Stamp((140, 70, 47))

    # --- Wood front face (south wall of the desk) -----------------------
    # Curved front: front corners stairstepped over 3 voxels.
    front_widths = ((0, 4, 5),     # y range, inset from each side
                    (4, 11, 3),
                    (11, 17, 1),
                    (17, 70, 0))
    for y0, y1, inset in front_widths:
        s.fill((slice(inset, 140 - inset),
                slice(y0, y1),
                slice(0, 47)), M.WOOD_DESK)
    # 3 evenly-spaced 1-voxel-darker vertical wood-grain seams on the front.
    for sx in (35, 70, 105):
        s.fill((sx, 0, slice(0, 47)), M.PLASTIC_DARK)

    # --- Stone-look counter top (polished PLASTIC_WHITE) ---------------
    s.fill((slice(2, 138), slice(2, 68), slice(44, 47)), M.PLASTIC_WHITE)
    # 1-voxel-darker edge band around the top perimeter.
    s.fill((slice(2, 138), 2, 44), M.PLASTIC_DARK)
    s.fill((slice(2, 138), 67, 44), M.PLASTIC_DARK)
    s.fill((2, slice(2, 68), 44), M.PLASTIC_DARK)
    s.fill((137, slice(2, 68), 44), M.PLASTIC_DARK)

    # --- Hollow knee well (staff side) ---------------------------------
    s.clear((slice(4, 136), slice(8, 65), slice(0, 44)))

    # --- Stone-look base molding strip along the very bottom of the front
    s.fill((slice(0, 140), slice(0, 4), slice(0, 3)), M.PLASTIC_DARK)

    return s


@stamp_cached(8)
def build_cafeteria_table(rng: np.random.Generator) -> Stamp:
    """A long cafeteria table with benches alongside."""
    s = Stamp((60, 30, 38))
    # Top
    s.fill((slice(None), slice(None), slice(35, 38)), M.PLASTIC_WHITE)
    # 4 legs
    for lx in (3, 56):
        for ly in (3, 26):
            s.fill((lx, ly, slice(0, 35)), M.METAL_DARK)
    # Benches at z=18..21
    for by0 in (-3, 30):  # Negative means we crop on paste; just place legitimately
        if by0 < 0:
            continue
        s.fill((slice(5, 55), slice(by0 - 6, by0 - 3), slice(20, 23)), M.PLASTIC_WHITE)
        # Bench legs
        for blx in (8, 50):
            s.fill((blx, slice(by0 - 5, by0 - 4), slice(0, 20)), M.METAL_DARK)
    return s
