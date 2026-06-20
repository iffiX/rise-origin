# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Room-level emitters built on the :class:`WorldEmitter` API.

Each ``emit_<biome>(layout, emitter, rng)`` function consumes the layout to
find its room's world-meter bounds and uses :class:`WorldEmitter` to place
stamps at world coordinates. There is NO per-cell partitioning here —
multi-cell rooms get genuinely continuous content with no inter-cell seams.

:func:`emit_all_rooms` is the entry point called from :func:`main.build`.
"""

from __future__ import annotations

import numpy as np

from . import materials as M
from .emitter import WorldEmitter, room_world_bounds, all_rooms_with_biome, \
    room_world_bounds_by_id
from .layout import Biome, Layout
from .objects import architecture, decoration, equipment, fixtures, furniture, \
    storage


# ============================================================================
# Helpers
# ============================================================================


def _ceiling_z() -> float:
    """The world Z where the ceiling sits (used to mount ceiling fixtures)."""
    from .constants import CELL_H, WALL_T, VOXEL_W
    return (CELL_H + WALL_T) * VOXEL_W


def _eye_level_z() -> float:
    """A useful "wall mid-height" anchor."""
    return _ceiling_z() / 2


def _floor_z() -> float:
    from .constants import WALL_T, VOXEL_W
    return WALL_T * VOXEL_W


# ============================================================================
# Lobby (south entrance)
# ============================================================================


def emit_lobby(layout: Layout, emitter: WorldEmitter,
               rng: np.random.Generator) -> None:
    """The south Future Corp lobby — reception desk + sofas + armchairs +
    water cooler + plant + warm pendant overhead lighting."""
    rx0, ry0, rx1, ry1 = room_world_bounds(layout, Biome.ENTRANCE)
    z0 = _floor_z()
    ceiling = _ceiling_z()
    cx = (rx0 + rx1) / 2

    # --- Reception desk at the NORTH end of the lobby (against the wall
    #     that faces the main hall) -------------------------------------
    # Staff stand behind the desk facing south (toward customers entering
    # from the entrance arch). The desk's brand backboard ends up against
    # the dark-glass wall to the main hall, so the F-logo backboard reads
    # to anyone approaching from the lobby.
    desk = furniture.build_reception_desk(rng)
    desk_w_m = desk.nx * emitter.voxel_size
    desk_d_m = desk.ny * emitter.voxel_size
    desk_x = cx - desk_w_m / 2
    desk_y = ry1 - desk_d_m - 0.3  # close to the north wall (main-hall side)
    emitter.place(desk, desk_x, desk_y, z0)

    # --- Reception-desk props sitting ON TOP of the counter ----------
    # Counter top sits at z = desk_z + 47 fine voxels (0.94 m above desk
    # base). Place 2 monitor+keyboard stations facing south (toward
    # customers), a telephone in the centre, and a coffee mug at one end.
    # All anchored at world coords so they're visible to the camera and
    # not buried/clipped inside the desk's z envelope.
    desk_top_z = z0 + 47 * emitter.voxel_size
    # Two monitors on the customer-edge of the counter top, screens facing
    # +Y (NORTH, toward the staff who stand behind the desk). The default
    # build_monitor stamp has the screen at y=0..2 facing -Y; rotate=2
    # flips it 180° so the screen faces +Y instead.
    # After rotate=2, the stamp's "back" of the body is now at low-y, the
    # screen at high-y. To place the screen facing the staff at the north
    # edge of the desk, anchor the stamp on the south side of its envelope
    # at the customer side of the counter.
    monitor_y = desk_y + 0.10
    emitter.place(equipment.build_monitor(rng),
                  desk_x + 0.40, monitor_y, desk_top_z, rotate=2)
    emitter.place(equipment.build_monitor(rng),
                  desk_x + 1.75, monitor_y, desk_top_z, rotate=2)
    # Keyboards north of each monitor (on the staff side), so staff
    # standing behind the desk can type while looking at the screens.
    keyboard_y = desk_y + 0.42
    emitter.place(equipment.build_keyboard(rng),
                  desk_x + 0.50, keyboard_y, desk_top_z)
    emitter.place(equipment.build_keyboard(rng),
                  desk_x + 1.85, keyboard_y, desk_top_z)
    # Telephone in the centre-staff position.
    emitter.place(equipment.build_telephone(rng),
                  desk_x + desk_w_m / 2 - 0.08, desk_y + 0.55, desk_top_z)
    # Coffee mug at one far end (a tiny dark cylinder — visual flavour).
    # Reuse the laser_emitter pyramid pictogram for a small prop; it's the
    # right scale and reads as a desktop item.
    emitter.place(equipment.build_laser_emitter(rng, colour=M.METAL_DARK),
                  desk_x + desk_w_m - 0.30, desk_y + 0.40, desk_top_z)

    # --- Sofas + armchairs SOUTH of the desk, facing it (rotate 180°) -
    sofa = furniture.build_sofa(rng)
    sofa_w_m = sofa.nx * emitter.voxel_size
    ac = furniture.build_armchair(rng)
    ac_w_m = ac.nx * emitter.voxel_size
    seating_y = desk_y - 3.0  # ~3 m south of the desk
    # rotate=2 flips the sofa 180° around Z so its back faces south (the
    # entrance) and its cushions face north toward the desk.
    emitter.place(sofa, cx - sofa_w_m - 1.8, seating_y, z0, rotate=2)
    emitter.place(sofa, cx + 1.8, seating_y, z0, rotate=2)
    emitter.place(ac, cx - sofa_w_m - ac_w_m - 2.4, seating_y, z0, rotate=2)
    emitter.place(ac, cx + sofa_w_m + 2.4, seating_y, z0, rotate=2)

    # --- Water cooler at the far west south corner --------------------
    wc = furniture.build_water_cooler(rng)
    emitter.place(wc, rx0 + 0.4, ry0 + 0.6, z0)

    # --- Plant at the far east south corner ---------------------------
    plant = decoration.build_plant_in_pot(rng)
    emitter.place(plant, rx1 - 0.6, ry0 + 0.6, z0)

    # --- Warm decorative pendants overhead (every ~4 m) ---------------
    pend = equipment.build_decorative_pendant(rng)
    pend_z = ceiling - 1.0
    nx_pend = max(1, int((rx1 - rx0 - 1) / 4))
    pend_pitch = (rx1 - rx0 - 2) / max(1, nx_pend)
    emitter.repeat_along(
        equipment.build_decorative_pendant,
        start=rx0 + 1.0, end=rx1 - 1.0, axis="x", pitch=max(1.0, pend_pitch),
        offset_xy=(0, (ry0 + ry1) / 2), z=pend_z, rng=rng,
    )

    # --- FUTURE CORP wall logo on the lobby's EAST perimeter wall ----
    # Mounted on the SIDE wall (not the front door wall, per round-4
    # feedback). Diffuse-dark METAL_DARK letters + an EM_WHITE backplate
    # sitting behind them: light leaks around the letter silhouettes to
    # produce a halo, no emissive-yellow voxels in the letters themselves.
    # The stamp is designed as (depth=4, width=NY, height=NZ) so the
    # back side (low X in the stamp) sits AGAINST the east wall and the
    # letter face is visible to the lobby (west-facing).
    logo = equipment.build_future_corp_wall_logo(rng)
    logo_w_m = logo.ny * emitter.voxel_size
    logo_d_m = logo.nx * emitter.voxel_size
    logo_h_m = logo.nz * emitter.voxel_size
    # East wall interior surface is at world x = rx1; mount so the stamp's
    # back (x=NX-1) sits at the wall surface and the letter face protrudes
    # ~0.06 m into the lobby.
    logo_x = rx1 - logo_d_m
    logo_y = (ry0 + ry1) / 2 - logo_w_m / 2
    # Centre the logo vertically on the 3 m wall so the top stays below
    # the ceiling (previous z=z0+0.9 m put the 2.8 m-tall logo's top at
    # 3.8 m — past the 3.1 m ceiling).
    wall_h_m = ceiling - z0
    logo_z = z0 + (wall_h_m - logo_h_m) / 2
    emitter.place(logo, logo_x, logo_y, logo_z)


# ============================================================================
# Bio lab (4 wide × 5 deep, west side)
# ============================================================================


def emit_bio_lab(layout: Layout, emitter: WorldEmitter,
                 rng: np.random.Generator) -> None:
    """The 4 × 5 cell Future Corp bio-lab — dense equipment clusters in 3
    bench rows down the middle, specimen tanks providing the warm bio glow,
    freezers / incubators flanking the east + west outer walls.

    (Round-3 fix: the perimeter glass-display-cabinet rows were too bright
    and dominated the room as bright yellow boxes — removed. The warm
    accent now comes only from the specimen tanks on the bench tops.)
    """
    # The big "main" bio lab cluster is the (cx=0..3, cy=2..6) room.
    rx0, ry0, rx1, ry1 = room_world_bounds_by_id(
        layout, int(layout.room_id[0, 2])
    )
    z0 = _floor_z()
    ceiling = _ceiling_z()

    # --- West + east outer walls: alternating freezer + incubator ----
    for wy in (ry0 + 4.0, ry0 + 8.0, ry0 + 12.0, ry0 + 16.0):
        emitter.place(equipment.build_lab_freezer(rng),  rx0 + 0.5, wy, z0)
        emitter.place(equipment.build_incubator(rng),    rx1 - 1.0, wy, z0)

    # --- Three central E–W bench clusters with biome creature boxes ---
    # The bench is 2.4 m wide (120 fine vox); we lay two of them end-to-end
    # so the row of specimen tanks always has continuous bench underneath
    # — without the second bench the east-most tanks were floating.
    bench_ys = (ry0 + 6.0, ry0 + 11.0, ry0 + 16.0)
    for by in bench_ys:
        # Bench 1 (under centrifuge + PCR + microscope).
        emitter.place(equipment.build_lab_bench(rng), rx0 + 2.5, by, z0)
        # Bench 2 (under the specimen-tank row), butting east of bench 1.
        emitter.place(equipment.build_lab_bench(rng), rx0 + 4.9, by, z0)
        # Equipment on top of the bench (bench top at z = 0.84 m).
        bench_top = z0 + 0.84
        emitter.place(equipment.build_centrifuge(rng),     rx0 + 2.7, by + 0.4, bench_top)
        emitter.place(equipment.build_pcr_machine(rng),    rx0 + 3.4, by + 0.4, bench_top)
        emitter.place(equipment.build_microscope(rng),     rx0 + 3.9, by + 0.4, bench_top)
        # **Specimen tanks** ("biome creature boxes") sit on bench 2.
        # 4 tanks at 0.55 m pitch, spanning x = rx0+5.1..rx0+6.75 — fully
        # within bench 2 (which covers rx0+4.9..rx0+7.3).
        emitter.repeat_along(
            equipment.build_specimen_tank,
            start=rx0 + 5.1, end=rx0 + 6.75, axis="x", pitch=0.55,
            offset_xy=(0, by + 0.3), z=bench_top, rng=rng,
        )
        emitter.place(equipment.build_autoclave(rng),      rx0 + 7.5, by + 0.1, z0)
        emitter.place(equipment.build_fume_hood(rng),      rx0 + 8.7, by + 0.1, z0)

    # --- Lab shelf racks along the west wall, between the freezer slots
    emitter.repeat_along(
        equipment.build_lab_shelf_rack,
        start=ry0 + 3.0, end=ry1 - 3.0, axis="y", pitch=2.0,
        offset_xy=(rx0 + 1.7, 0), z=z0, rng=rng,
    )

    # --- Industrial ceiling strips on a sparse grid -------------------
    strip_z = ceiling - 0.3
    emitter.array(
        fixtures.build_ceiling_strip,
        x0=rx0 + 1.5, y0=ry0 + 2.0, z=strip_z,
        x_count=4, y_count=5, x_pitch=3.5, y_pitch=3.5,
        rng=rng,
    )


# ============================================================================
# Server farm (4 wide × 5 deep, east side)
# ============================================================================


def emit_server_farm(layout: Layout, emitter: WorldEmitter,
                     rng: np.random.Generator) -> None:
    """The 4 × 5 cell Thinking-Machines server room — rows of CM-5 cabinets
    facing along +Y, narrow access aisles between them, cable bundles
    overhead, ceiling strips on a sparse grid.
    """
    rid_arr = np.asarray(layout.room_id)
    biome_arr = np.asarray(layout.biome)
    # The big server farm room is the contiguous one at (cx=7, cy=2).
    rid = int(rid_arr[7, 2])
    rx0, ry0, rx1, ry1 = room_world_bounds_by_id(layout, rid)
    z0 = _floor_z()
    ceiling = _ceiling_z()

    # --- Racks: 3 N-S rows × ~6 racks each along X ------------------
    # Standard 19" rack envelope is 0.6 m W × 1.0 m D × 2.0 m H.
    for row_y in (ry0 + 2.0, ry0 + 8.0, ry0 + 14.0):
        emitter.repeat_along(
            equipment.build_thinking_machine_rack,
            start=rx0 + 1.0, end=rx1 - 1.0, axis="x", pitch=0.7,
            offset_xy=(0, row_y), z=z0, rng=rng,
        )

    # --- Multiple cable bundles overhead running along Y (corridor axis)
    # build_cable_bundle's length axis is X by default, so we rotate=1 to
    # swap X<->Y. Place 4 parallel bundles across the room's width.
    cb_length_v = int((ry1 - ry0 - 1) / emitter.voxel_size)
    n_bundles = 4
    bundle_x_anchors = np.linspace(rx0 + 1.2, rx1 - 1.2, n_bundles)
    for bx in bundle_x_anchors:
        cb = architecture.build_cable_bundle(rng, length=cb_length_v)
        emitter.place(cb, bx, ry0 + 0.5, ceiling - 0.2, rotate=1)

    # --- Industrial ceiling strips on a sparse grid -----------------
    emitter.array(
        fixtures.build_ceiling_strip,
        x0=rx0 + 1.5, y0=ry0 + 1.5, z=ceiling - 0.3,
        x_count=4, y_count=5, x_pitch=3.5, y_pitch=3.5,
        rng=rng,
    )


# ============================================================================
# Main hall (3 wide × 5 deep, central security transition hall)
# ============================================================================


def emit_main_hall(layout: Layout, emitter: WorldEmitter,
                   rng: np.random.Generator) -> None:
    """A wide open security transition hall — ceiling strips, a hazard
    chevron painted on the floor pointing toward the laser hallway.
    """
    rx0, ry0, rx1, ry1 = room_world_bounds(layout, Biome.MAIN_HALL)
    ceiling = _ceiling_z()
    z0 = _floor_z()

    emitter.array(
        fixtures.build_ceiling_strip,
        x0=rx0 + 1.5, y0=ry0 + 1.5, z=ceiling - 0.3,
        x_count=3, y_count=5, x_pitch=3.5, y_pitch=3.5,
        rng=rng,
    )
    # Hazard chevron on the floor pointing north (toward the laser hall).
    chev = fixtures.build_hazard_chevron(rng, length=60, height=30)
    emitter.place(chev, (rx0 + rx1) / 2 - 0.6, (ry0 + ry1) / 2 - 0.3, z0)


# ============================================================================
# Laser corridor (3 wide × 3 deep)
# ============================================================================


def emit_laser_corridor(layout: Layout, emitter: WorldEmitter,
                        rng: np.random.Generator) -> None:
    """A grand laser hallway redesigned to the user's reference image.

    Geometry (north-south long axis = Y; corridor centreline = X):

    * **Floor** — 5 alternating BLACK/WHITE/BLACK/WHITE/BLACK strips
      running the full corridor length along Y, each 0.5 m wide.
    * **Side walls** — two FULL-HEIGHT aluminium-framed glass walls
      flanking the central 2.5 m walkway (one west, one east). Vertical
      aluminium mullions every 1 m; the rest is transparent glass.
    * **Lasers behind the glass** — many horizontal EM_CYAN strips
      running along Y, stacked at 22 cm vertical pitch, set ~0.45 m
      behind each glass wall so the lit lines read as parallel
      horizontal emitters glowing through the glass.
    * **Ceiling** — a 3 × 4 grid of rectangular EM_WHITE light panels
      embedded flush with the ceiling above the walkway.
    """
    from .stamp import Stamp

    rx0, ry0, rx1, ry1 = room_world_bounds(layout, Biome.LASER_CORRIDOR)
    ceiling = _ceiling_z()
    z0 = _floor_z()
    rcx = (rx0 + rx1) / 2     # corridor centreline (X)
    corridor_y0 = ry0 + 0.1
    corridor_len_m = ry1 - ry0 - 0.2
    strip_len_v = int(corridor_len_m / emitter.voxel_size)

    # --- Floor: 5 alternating BLACK/WHITE strips ---------------------
    walkway_w_m = 2.5
    walkway_half = walkway_w_m / 2
    strip_w_m = walkway_w_m / 5    # 0.5 m each
    strip_w_v = int(strip_w_m / emitter.voxel_size)  # 25 fine voxels
    floor_stamp = Stamp((5 * strip_w_v, strip_len_v, 1))
    strip_mats = (M.PLASTIC_DARK, M.PLASTIC_WHITE,
                  M.PLASTIC_DARK, M.PLASTIC_WHITE, M.PLASTIC_DARK)
    for i, mat in enumerate(strip_mats):
        floor_stamp.fill(
            (slice(i * strip_w_v, (i + 1) * strip_w_v), slice(None), 0),
            mat,
        )
    emitter.place(floor_stamp, rcx - walkway_half, corridor_y0, z0)

    # --- Two full-height aluminium-framed glass walls ----------------
    glass_h_m = 2.9
    glass_h_v = int(glass_h_m / emitter.voxel_size)
    glass_wall = equipment.build_laser_corridor_glass_wall(
        rng, length=strip_len_v, height=glass_h_v,
    )
    glass_d_m = glass_wall.nx * emitter.voxel_size
    # West wall sits just outside the west edge of the walkway,
    # east wall just outside the east edge.
    emitter.place(glass_wall,
                  rcx - walkway_half - glass_d_m, corridor_y0, z0)
    emitter.place(glass_wall,
                  rcx + walkway_half, corridor_y0, z0)

    # --- Horizontal cyan laser strips BEHIND each glass wall ---------
    # Backside of each wall is ~0.5 m further out from the walkway.
    LASER_PITCH = 0.22
    LASER_Z_MIN = z0 + 0.18
    LASER_Z_MAX = z0 + glass_h_m - 0.10
    laser_zs = []
    z = LASER_Z_MIN
    while z <= LASER_Z_MAX:
        laser_zs.append(z)
        z += LASER_PITCH

    # Each emitter pair is positioned just behind its glass wall so the
    # cyan lines glow through the panels along the corridor's length.
    for behind_x in (rcx - walkway_half - 0.55,
                     rcx + walkway_half + 0.55):
        for lz in laser_zs:
            beam = equipment.build_laser_beam(
                rng, length=strip_len_v, axis="y", colour=M.EM_CYAN,
            )
            emitter.place(beam, behind_x, corridor_y0, lz)

    # --- Ceiling: 3 × 4 grid of EM_WHITE rectangular light panels ----
    panel_w_v = 28      # ~0.56 m wide (X)
    panel_l_v = 36      # ~0.72 m long (Y)
    panel_stamp = Stamp((panel_w_v, panel_l_v, 1))
    panel_stamp.fill((slice(None), slice(None), 0), M.EM_WHITE)

    n_cols = 3
    n_rows = 4
    panel_col_xs = [
        rcx - walkway_half + (k + 0.5) * (walkway_w_m / n_cols) - panel_w_v * emitter.voxel_size / 2
        for k in range(n_cols)
    ]
    panel_row_ys = np.linspace(corridor_y0 + 1.0, ry1 - 1.0 - panel_l_v * emitter.voxel_size,
                               n_rows)
    panel_z = ceiling - 0.04
    for py in panel_row_ys:
        for px in panel_col_xs:
            emitter.place(panel_stamp, px, py, panel_z)


# ============================================================================
# Reactor chamber (3 × 3 cells)
# ============================================================================


def emit_reactor(layout: Layout, emitter: WorldEmitter,
                 rng: np.random.Generator) -> None:
    """The reactor chamber — a single big reactor core in the centre cell of
    the room, hazard chevrons painted around it.
    """
    rx0, ry0, rx1, ry1 = room_world_bounds(layout, Biome.REACTOR)
    z0 = _floor_z()
    core = equipment.build_reactor_core_large(rng)
    core_w_m = core.nx * emitter.voxel_size
    core_d_m = core.ny * emitter.voxel_size
    emitter.place(
        core,
        (rx0 + rx1) / 2 - core_w_m / 2,
        (ry0 + ry1) / 2 - core_d_m / 2,
        z0,
    )

    # Hazard chevrons in 4 cardinal directions pointing toward the reactor.
    chev = fixtures.build_hazard_chevron(rng, length=60, height=30)
    cx = (rx0 + rx1) / 2
    cy = (ry0 + ry1) / 2
    emitter.place(chev, cx - 0.6, ry0 + 0.5, z0)
    emitter.place(chev, cx - 0.6, ry1 - 0.8, z0, rotate=2)
    emitter.place(chev, rx0 + 0.5, cy - 0.3, z0, rotate=1)
    emitter.place(chev, rx1 - 0.8, cy - 0.3, z0, rotate=3)


# ============================================================================
# Singleton side rooms (control / engineering / storage / observation)
# ============================================================================


def emit_singleton_rooms(layout: Layout, emitter: WorldEmitter,
                         rng: np.random.Generator) -> None:
    """Iterate the small singleton rooms (anything not in the big multi-cell
    rooms above) and place modest content per room.

    Biome → content:
    * CONTROL_ROOM → 1 desk + monitor + chair + 1 server rack at the NW corner.
    * ENGINEERING → 1 pipe-run along the south wall + a workbench at one corner.
    * STORAGE → 4 lockers along the south wall + 1 crate stack at the centre.
    * BIO_LAB satellite rooms (the four corner observation labs) → 4 display cabinets.
    """
    rid_arr = np.asarray(layout.room_id)
    biome_arr = np.asarray(layout.biome)
    seen_rids = set()
    z0 = _floor_z()
    ceiling = _ceiling_z()
    # Skip the multi-cell rooms we've already emitted.
    main_rids = {
        int(rid_arr[5, 5]),                  # main hall (cx=5, cy=5)
        int(rid_arr[5, 8]),                  # laser corridor
        int(rid_arr[5, 11]),                 # reactor
        int(rid_arr[0, 2]),                  # main bio lab
        int(rid_arr[7, 2]),                  # main server farm
        int(rid_arr[5, 1]),                  # lobby
    }
    for cx in range(layout.biome.shape[0]):
        for cy in range(layout.biome.shape[1]):
            rid = int(rid_arr[cx, cy])
            if rid < 0 or rid in seen_rids or rid in main_rids:
                continue
            seen_rids.add(rid)
            biome = biome_arr[cx, cy]
            rx0, ry0, rx1, ry1 = room_world_bounds_by_id(layout, rid)
            mid_x = (rx0 + rx1) / 2
            mid_y = (ry0 + ry1) / 2

            # Ceiling strip(s).
            emitter.place(
                fixtures.build_ceiling_strip(rng),
                mid_x - 0.9, mid_y - 0.2, ceiling - 0.3,
            )

            if biome is Biome.CONTROL_ROOM:
                # Count this room's cells: a big merged office vs a single
                # 3x1 control panel room.
                rid_mask = np.asarray(rid_arr == rid)
                cells_in_room = int(rid_mask.sum())
                room_w_m = rx1 - rx0
                room_d_m = ry1 - ry0
                if cells_in_room >= 4 and min(room_w_m, room_d_m) >= 6.0:
                    # ----- Big open-plan office ---------------------
                    # 2 rows of workstations, each row has 5 desk + chair
                    # + monitor + keyboard sets facing each other across
                    # the room's centre aisle. Plus 2 server racks on the
                    # opposite ends.
                    n_per_row = 5
                    desk_xs = np.linspace(
                        rx0 + 1.4, rx1 - 1.4, n_per_row,
                    )
                    aisle_y = mid_y
                    # Desk envelopes: 1.0 m (X) × 0.5 m (Y) × 0.76 m top.
                    # South row staff sit south of the desks; north row
                    # staff sit north. South-row monitor screens face -Y
                    # so the south-row staff see them. North-row use
                    # rotate=2 so monitors and chairs flip 180° to face -Y
                    # (so screen face +Y, toward the north-row staff).
                    south_desk_y = aisle_y - 1.6
                    north_desk_y = aisle_y + 0.8
                    DESK_TOP_Z = z0 + 0.76   # desk top surface (= top voxel boundary)
                    for dx in desk_xs:
                        # ----- SOUTH row workstation (no rotation) ------
                        emitter.place(furniture.build_desk(rng),
                                      dx - 0.5, south_desk_y, z0)
                        # Chair south of desk; centre its pedestal on desk X.
                        emitter.place(furniture.build_chair(rng),
                                      dx - 0.28, south_desk_y - 0.6, z0)
                        # Monitor on BACK (north) half of desk; screen at
                        # stamp y=0 faces -Y toward the chair.
                        emitter.place(equipment.build_monitor(rng),
                                      dx - 0.15, south_desk_y + 0.18, DESK_TOP_Z)
                        # Keyboard on FRONT (south) half of desk, between
                        # the chair and the monitor.
                        emitter.place(equipment.build_keyboard(rng),
                                      dx - 0.14, south_desk_y + 0.06, DESK_TOP_Z)

                        # ----- NORTH row workstation (rotate=2) ---------
                        emitter.place(furniture.build_desk(rng),
                                      dx - 0.5, north_desk_y, z0, rotate=2)
                        emitter.place(furniture.build_chair(rng),
                                      dx - 0.28, north_desk_y + 0.85, z0,
                                      rotate=2)
                        # Monitor on SOUTH half of desk (anchor y =
                        # north_desk_y). After rotate=2 the screen ends
                        # up on the north side of the monitor envelope,
                        # facing +Y toward the staff at north_desk_y+0.85.
                        emitter.place(equipment.build_monitor(rng),
                                      dx - 0.15, north_desk_y, DESK_TOP_Z,
                                      rotate=2)
                        # Keyboard on NORTH half of desk (between monitor
                        # and the staff member at north_desk_y + 0.85).
                        emitter.place(equipment.build_keyboard(rng),
                                      dx - 0.14, north_desk_y + 0.42,
                                      DESK_TOP_Z)
                    # Two server racks against the east + west walls (corners).
                    emitter.place(equipment.build_server_rack(rng),
                                  rx0 + 0.2, ry0 + 0.3, z0)
                    emitter.place(equipment.build_server_rack(rng),
                                  rx1 - 0.8, ry1 - 1.4, z0)
                    # Two extra ceiling strips for the larger room.
                    emitter.place(
                        fixtures.build_ceiling_strip(rng),
                        rx0 + 2.5, mid_y - 0.2, ceiling - 0.3,
                    )
                    emitter.place(
                        fixtures.build_ceiling_strip(rng),
                        rx1 - 4.0, mid_y - 0.2, ceiling - 0.3,
                    )
                else:
                    # Small control panel room — original single-desk layout.
                    desk = furniture.build_desk(rng)
                    emitter.place(desk, rx1 - 1.2, mid_y - 0.25, z0)
                    emitter.place(furniture.build_chair(rng),
                                  rx1 - 1.8, mid_y - 0.3, z0)
                    emitter.place(equipment.build_monitor(rng),
                                  rx1 - 1.0, mid_y - 0.18, z0 + 0.78)
                    emitter.place(equipment.build_server_rack(rng),
                                  rx0 + 0.3, ry0 + 0.3, z0)
            elif biome is Biome.ENGINEERING:
                # Pipe run along the south wall.
                length_v = int((rx1 - rx0 - 0.4) / emitter.voxel_size)
                pipe = architecture.build_pipe_run(
                    rng, length=length_v, axis="x",
                    colour=M.PAINTED_CYAN, add_valve=True,
                )
                emitter.place(pipe, rx0 + 0.2, ry0 + 0.2, z0 + 1.8)
                # Workbench at NW corner.
                emitter.place(furniture.build_workbench(rng),
                              rx0 + 0.3, ry0 + 1.0, z0)
            elif biome is Biome.STORAGE:
                # 4 lockers along the south wall.
                for i in range(4):
                    emitter.place(storage.build_locker(rng),
                                  rx0 + 0.3 + i * 0.35, ry0 + 0.3, z0)
                # Crate stack at centre.
                emitter.place(storage.build_crate_stack(rng, count=2),
                              mid_x - 0.15, mid_y - 0.15, z0)
            elif biome is Biome.BIO_LAB:
                # Observation lab — a couple of lab shelf racks against the
                # north wall (closed storage, no big emissive surfaces).
                emitter.repeat_along(
                    equipment.build_lab_shelf_rack,
                    start=rx0 + 0.5, end=rx1 - 0.5, axis="x", pitch=0.8,
                    offset_xy=(0, ry1 - 0.7), z=z0, rng=rng,
                )


# ============================================================================
# Top-level dispatcher
# ============================================================================


def emit_all_rooms(layout: Layout, emitter: WorldEmitter,
                   rng: np.random.Generator) -> None:
    """Drive every room emitter in turn. Called once from ``main.build``."""
    emit_lobby(layout, emitter, rng)
    emit_main_hall(layout, emitter, rng)
    emit_bio_lab(layout, emitter, rng)
    emit_server_farm(layout, emitter, rng)
    emit_laser_corridor(layout, emitter, rng)
    emit_reactor(layout, emitter, rng)
    emit_singleton_rooms(layout, emitter, rng)
