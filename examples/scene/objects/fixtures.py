# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Lit fixture builders: ceiling strip, doorway frame, signs, hazard chevron.

Every builder returns a :class:`scene.stamp.Stamp` and follows the same shape:

* a docstring naming the real-world inspiration and the silhouette,
* an RNG-driven variation budget so identical instances don't duplicate,
* deliberate use of corner knock-outs, recesses, and 1-voxel emissive shells
  (never solid emissive volumes — those waste lights on interior voxels that
  can't contribute, per the budget-section rule in the plan).

Sizes are in the **fine 0.02 m grid**.
"""

from __future__ import annotations

import numpy as np

from .. import materials as M
from ..stamp import Stamp, stamp_cached


# ============================================================================
# Industrial fluorescent pendant
# ============================================================================


@stamp_cached(8)
def build_ceiling_strip(rng: np.random.Generator) -> Stamp:
    """A 1.8 m linear-fluorescent pendant with a trapezoidal cross-section.

    Inspiration: hanging industrial fluorescent fixture — narrow housing top,
    sides flaring outward via a stepped chamfer, two parallel emissive tubes
    visible behind a 1-voxel-thick "diffuser" layer, end caps, and a small
    stem to a ceiling mount plate.

    Variation:
      * 10 % one tube dim — every other voxel of one tube off (flicker effect),
      * 5 % whole fixture dead — no emissive voxels (creates a dark corner).
    """
    L = 90  # length in fine voxels
    s = Stamp((L, 20, 14))

    # Ceiling mount plate at the top (z = 13), 8 voxels square, with a small LED.
    s.fill((slice(L // 2 - 4, L // 2 + 4),
            slice(6, 14),
            13), M.METAL_DARK)
    s.set(L // 2, 10, 13, M.EM_CYAN)  # tiny mount indicator
    # Drop stem from plate to housing top.
    s.fill((slice(L // 2 - 1, L // 2 + 1),
            slice(9, 11),
            slice(5, 13)), M.METAL_DARK)

    # Housing top: narrow 6-wide strip at z = 4.
    s.fill((slice(0, L), slice(7, 13), 4), M.METAL_HOUSING)
    # Stepped chamfer: each row down widens by 2.
    for step, width_extra in enumerate((0, 2, 4, 6)):
        z = 3 - step
        if z < 0:
            break
        y0 = 7 - width_extra // 2 if step > 0 else 7
        y1 = 13 + (width_extra - width_extra // 2) if step > 0 else 13
        # left + right side voxels only (1-voxel-thick metal wall)
        s.fill((slice(0, L), slice(y0, y0 + 1), z), M.METAL_HOUSING)
        s.fill((slice(0, L), slice(y1 - 1, y1), z), M.METAL_HOUSING)

    # Two parallel emissive tubes at z = 0 (the bottom face).
    is_dead = rng.random() < 0.05
    dim_one_tube = (not is_dead) and rng.random() < 0.10
    for tube_idx, y_start in enumerate((2, 12)):
        y_end = y_start + 6
        if is_dead:
            continue
        for x in range(3, L - 3):
            if dim_one_tube and tube_idx == 1 and (x % 2):
                continue  # flicker pattern on second tube
            s.fill((x, slice(y_start, y_end), 0), M.EM_WHITE)

    # End caps: 1-voxel-wide dark metal at each end of the bottom panel.
    s.fill((slice(0, 1), slice(2, 18), slice(0, 4)), M.METAL_DARK)
    s.fill((slice(L - 1, L), slice(2, 18), slice(0, 4)), M.METAL_DARK)

    # Rivet dots on housing top (1-darker voxels every 8 voxels).
    for x in range(8, L - 1, 8):
        s.set(x, 9, 4, M.METAL_DARK)
        s.set(x, 11, 4, M.METAL_DARK)

    return s


# ============================================================================
# Sliding blast doorway frame + door slab
# ============================================================================


@stamp_cached(8)
def build_doorway_frame(rng: np.random.Generator, *,
                         force_open: bool = False) -> Stamp:
    """A 1.2 m wide x 2.0 m tall sliding blast door, partly open, with
    porthole, wheel handle, hinge knuckles, and a hazard zebra lintel.

    The stamp is oriented so:
      * X is the doorway "width" axis (the door slides along X)
      * Y is the depth into the wall (Y = 0 is the corridor-facing side)
      * Z is height

    Args:
        force_open: If True, guarantees the slab is in the half-open state.
            Used by the scene's sightline doors so the reactor remains visible.

    Variation (when ``force_open=False``):
      * door open state: 60 % half-open (slid 35 voxels), 30 % closed,
        10 % jammed (slid 12 voxels with a 1-voxel z-tilt).
      * porthole glass occasionally lit (cyan emissive) for "active" rooms.
    """
    W = 60  # opening width in fine voxels (1.2 m)
    H = 100  # opening height in fine voxels (2.0 m)
    pad = 10  # extra X room for the slid-open slab to peek past the jamb
    s = Stamp((W + 2 * pad, 6, H + 20))  # +20 in z for lintel + top rail

    cx_l, cx_r = pad, pad + W
    base_z = 0
    lintel_z = H
    top_rail_z = lintel_z + 16

    # Jambs: 5x5x100 metal columns at the opening edges.
    for jamb_x in (cx_l - 5, cx_r):
        s.fill((slice(jamb_x, jamb_x + 5),
                slice(0, 5),
                slice(base_z, lintel_z)), M.METAL_HOUSING)
        # Rivet dots down the front face of each jamb.
        for z in range(base_z + 10, lintel_z - 10, 16):
            s.set(jamb_x + 2, 0, z, M.METAL_DARK)
            s.set(jamb_x + 2, 0, z + 2, M.METAL_DARK)
        # Emissive cyan trim line on inside-bottom of jamb.
        inner_x = jamb_x + (4 if jamb_x == cx_l - 5 else 0)
        for z in range(base_z, lintel_z):
            s.set(inner_x, 0, z, M.EM_CYAN)

    # Lintel: heavy metal beam above opening.
    s.fill((slice(cx_l - 5, cx_r + 5),
            slice(0, 5),
            slice(lintel_z, lintel_z + 15)), M.METAL_HOUSING)

    # Hazard zebra strip on lintel bottom-inner edge.
    for x in range(cx_l, cx_r):
        if ((x - cx_l) // 4) % 2 == 0:
            s.set(x, 0, lintel_z, M.EM_AMBER)
        else:
            s.set(x, 0, lintel_z, M.METAL_DARK)

    # Top sliding rail above the lintel.
    s.fill((slice(cx_l - 10, cx_r + 10),
            slice(1, 4),
            slice(top_rail_z, top_rail_z + 3)), M.METAL_HOUSING)

    # --- Door slab state ---------------------------------------------------
    if force_open:
        offset = W // 2 + rng.integers(-3, 4)
        tilt = 0
    else:
        state_roll = rng.random()
        if state_roll < 0.30:
            offset = W  # closed (slab fully covers opening)
        elif state_roll < 0.90:
            offset = W // 2 + rng.integers(-3, 4)  # half-open variants
        else:
            offset = 12  # jammed
        tilt = 1 if state_roll >= 0.90 else 0

    slab_w = W - 10
    slab_x0 = cx_l + (W - offset)  # slab leading edge x
    slab_x1 = slab_x0 + slab_w
    slab_z0 = base_z
    slab_z1 = base_z + H - 10

    if slab_x1 > 0 and slab_x0 < s.nx:
        x0 = max(0, slab_x0)
        x1 = min(s.nx, slab_x1)
        for z in range(slab_z0, slab_z1):
            # tilt: bump z up by 1 for the right end of the slab
            zz = z + (tilt if (x1 - 1) - cx_l > slab_w // 2 else 0)
            if zz >= s.nz:
                continue
            s.fill((slice(x0, x1), slice(1, 4), zz), M.METAL_DARK)

        # Porthole window: dark "glass" 8-voxel-diameter circle at z=60.
        port_cx = slab_x0 + slab_w // 2
        port_cz = base_z + 60
        for dx in range(-4, 5):
            for dz in range(-4, 5):
                if dx * dx + dz * dz <= 16:
                    px, pz = port_cx + dx, port_cz + dz
                    if 0 <= px < s.nx and 0 <= pz < s.nz:
                        s.set(px, 2, pz, M.PLASTIC_DARK)
        # Maybe lit porthole (30 %).
        if rng.random() < 0.30:
            s.set(port_cx, 1, port_cz, M.EM_CYAN)

        # Wheel handle at z=40 — protruding ring + 4 spokes.
        wh_cx = slab_x0 + slab_w // 2
        wh_cz = base_z + 40
        for dx, dz in (
            (-3, 0), (3, 0), (0, -3), (0, 3),
            (-2, -2), (-2, 2), (2, -2), (2, 2),
            (-3, -1), (-3, 1), (3, -1), (3, 1),
            (-1, -3), (1, -3), (-1, 3), (1, 3),
        ):
            xx, zz = wh_cx + dx, wh_cz + dz
            if 0 <= xx < s.nx and 0 <= zz < s.nz:
                s.set(xx, 0, zz, M.METAL_HOUSING)  # ring
        # Spokes
        for off in range(-2, 3):
            if off == 0:
                continue
            s.set(wh_cx + off, 0, wh_cz, M.METAL_HOUSING)
            s.set(wh_cx, 0, wh_cz + off, M.METAL_HOUSING)
        s.set(wh_cx, 0, wh_cz, M.METAL_DARK)  # hub

        # Hinge knuckles on trailing edge.
        trailing_x = slab_x1 - 1
        for hz in (base_z + 15, base_z + 45, base_z + 75):
            if 0 <= trailing_x < s.nx and 0 <= hz < s.nz:
                s.fill((trailing_x, slice(4, 5), slice(hz, hz + 3)), M.METAL_DARK)

    return s


# ============================================================================
# Door control panel (wall-mounted next to a doorway)
# ============================================================================


@stamp_cached(8)
def build_door_control_panel(rng: np.random.Generator) -> Stamp:
    """A small 0.20 m x 0.05 m x 0.20 m wall-mounted control panel.

    Inspiration: door control unit — dark plastic body with a small screen,
    one status LED, and a 2x2 array of darker button dots.
    """
    s = Stamp((10, 2, 10))
    s.fill((slice(None), slice(0, 2), slice(None)), M.PLASTIC_DARK)
    # Screen (cyan/amber, RNG).
    screen_mat = M.EM_CYAN if rng.random() < 0.7 else M.EM_AMBER
    s.fill((slice(2, 8), 0, slice(5, 9)), screen_mat)
    # Status LED.
    s.set(5, 0, 1, M.EM_GREEN if rng.random() < 0.85 else M.EM_RED)
    # Button dots: 2x2 grid of darker plastic.
    for dx in (3, 6):
        for dz in (2, 4):
            s.set(dx, 0, dz, M.METAL_DARK)
    return s


# ============================================================================
# Exit sign (ceiling-hung and wall)
# ============================================================================


@stamp_cached(8)
def build_exit_sign(rng: np.random.Generator, *, large: bool = False) -> Stamp:
    """An EXIT sign — dark frame with an emissive green arrow pictogram.

    Two sizes via ``large``:
      * small wall sign: 20 x 2 x 15
      * large hanging ceiling sign: 40 x 4 x 15
    """
    if large:
        s = Stamp((40, 4, 15))
        bezel_y = (0, 4)
        emis_y = 1
        arrow_x0 = 5
        arrow_x_len = 30
    else:
        s = Stamp((20, 2, 15))
        bezel_y = (0, 2)
        emis_y = 0
        arrow_x0 = 2
        arrow_x_len = 16

    # Dark backing frame.
    s.fill((slice(None), slice(*bezel_y), slice(None)), M.PLASTIC_DARK)
    # Knock out the 4 corner voxels (rounded).
    for cx in (0, s.nx - 1):
        for cz in (0, s.nz - 1):
            s.clear((cx, slice(*bezel_y), cz))

    # Emissive green arrow shaft: a horizontal line at z = 5..9.
    shaft_z = s.nz // 2 - 1
    s.fill((slice(arrow_x0, arrow_x0 + arrow_x_len),
            emis_y, slice(shaft_z, shaft_z + 2)), M.EM_GREEN)
    # Arrowhead: a triangular cluster at the right end.
    head_x = arrow_x0 + arrow_x_len - 1
    for k in range(4):
        # widening as we approach the tip
        for dz in range(-k, k + 1):
            zz = shaft_z + 1 + dz
            xx = head_x - k
            if 0 <= xx < s.nx and 0 <= zz < s.nz:
                s.set(xx, emis_y, zz, M.EM_GREEN)
    # Tip itself
    s.set(head_x, emis_y, shaft_z + 1, M.EM_GREEN)

    return s


# ============================================================================
# Hazard chevron (floor decal in front of doorways)
# ============================================================================


@stamp_cached(4)
def build_hazard_chevron(rng: np.random.Generator,
                         *, length: int = 30, height: int = 20) -> Stamp:
    """An angled hazard-yellow chevron painted into the top of the floor.

    Implemented as a 1-voxel-thick painted decal (uses PAINTED_YELLOW, not
    emissive). Caller should paste this onto top of a floor voxel layer in
    "replace" mode so the painted material overrides the underlying concrete.
    """
    s = Stamp((length, height, 1))
    # Two diagonal stripes forming a ">" chevron.
    for stripe_offset in (0, 5):
        for k in range(min(length, height // 2)):
            for thick in range(3):  # 3-voxel-thick stripe
                x = k + stripe_offset + thick
                if x >= length:
                    continue
                # Top half of chevron
                z_top = (height // 2) - k - 1
                if 0 <= z_top < height:
                    s.set(x, z_top, 0, M.PAINTED_YELLOW)
                # Bottom half
                z_bot = (height // 2) + k
                if 0 <= z_bot < height:
                    s.set(x, z_bot, 0, M.PAINTED_YELLOW)
    return s
