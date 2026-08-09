# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Entry point: build a decommissioned underground research outpost.

What you'll learn:

* See a voxel scene that reads as **a place** — corridors, rooms, doorways,
  fixtures. Two voxel resolutions cooperate: 0.10 m blocky walls/ceiling/floor
  + 0.02 m fine fixtures and furniture.
* Stress ReSTIR DI in the regime the paper's screenshots take — many small
  embedded lights behind heavy architectural occlusion, so each pixel sees
  only a few visible lights through doorways. One NEE sample is mostly a
  shadow ray; ReSTIR's reservoir resampling + temporal reuse pays off visibly
  even at one sample per pixel.

Run it::

    python examples/scene/lab.py
    RISE_MODE=record RISE_OUT=lab.mp4 python examples/scene/lab.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import numpy as np

from rise_origin import rsc

# --- sys.path bootstrap so `_common` and the scene package both import ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.dirname(_HERE)
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)
import _common  # noqa: E402

from scene import biomes, layout, materials  # noqa: E402  (scene = examples/scene)
from scene.constants import (  # noqa: E402
    CELL_W,
    CELL_H,
    WALL_T,
    CELLS_X,
    CELLS_Y,
    NX_W,
    NY_W,
    NZ_W,
    EXTENT_X,
    EXTENT_Y,
    DOOR_W,
    DOOR_H,
    CORPORATE_DOOR_W,
    CORPORATE_DOOR_H,
    SEED,
    VOXEL_W,
    VOXEL_O,
    WO_RATIO,
    INSPECT_NO_ROOF,
    INSPECT_NO_INTERNAL_WALLS,
    cell_origin_w,
)
from scene.layout import Biome  # noqa: E402
from scene.objects import fixtures, equipment  # noqa: E402
from scene.emitter import WorldEmitter  # noqa: E402
from scene.world import World  # noqa: E402

# Cell interior dimensions in fine voxels.
CW = CELL_W * WO_RATIO  # 200
CH = CELL_H * WO_RATIO  # 150


# ============================================================================
# Walls structure assembly (coarse 0.10 m)
# ============================================================================


def _build_walls(layout_: layout.Layout, rng: np.random.Generator) -> World:
    """Fill the coarse walls World: floors, ceilings, walls, doorways carved.

    Damage knock-outs of ceiling sections (damaged biomes) are also applied here.

    Walls between two cells whose :attr:`Layout.room_id` matches are SKIPPED
    (they're the same room). Walls between cells whose biome pair is in
    :attr:`Layout.glass_pairs` are emitted as :data:`materials.GLASS_TANK`
    instead of concrete. Honors the two ``INSPECT_*`` toggles.
    """
    w = World((NX_W, NY_W, NZ_W), VOXEL_W, materials.SEG_OF)
    concrete = materials.CONCRETE_WALL
    glass = materials.GLASS_TANK

    # --- Floor + optional ceiling slabs over the whole footprint ---
    w.fill((slice(None), slice(None), 0), materials.CONCRETE_FLOOR)
    if not INSPECT_NO_ROOF:
        w.fill((slice(None), slice(None), NZ_W - 1), materials.CONCRETE_CEILING)

    # --- Perimeter walls (always present so the structure has bounds) ---
    w.fill((0, slice(None), slice(0, NZ_W)), concrete)
    w.fill((NX_W - 1, slice(None), slice(0, NZ_W)), concrete)
    w.fill((slice(None), 0, slice(0, NZ_W)), concrete)
    w.fill((slice(None), NY_W - 1, slice(0, NZ_W)), concrete)

    # --- Internal cell-divider walls (edge-traced; per-run material + strip) ---
    if not INSPECT_NO_INTERNAL_WALLS:
        rid = layout_.room_id
        glass_pairs = layout_.glass_pairs or set()
        strip_pairs = layout_.light_strip_pairs or set()
        dark_glass_pairs = layout_.dark_glass_pairs or set()
        dark_glass = materials.DARK_GLASS
        # Sentinel: material is "concrete with embedded strip lights".
        CONCRETE_STRIP = "_concrete_strip"

        def _pair_material(a_cx, a_cy, b_cx, b_cy):
            """Material to use between two adjacent cells, or None for no wall.

            Returns one of: ``None`` (same room — skip wall), ``concrete``,
            ``glass``, ``dark_glass``, or the sentinel ``CONCRETE_STRIP``
            (concrete fill + a strip-light pass painted after).
            """
            if rid is not None:
                ra = int(rid[a_cx, a_cy])
                rb = int(rid[b_cx, b_cy])
                if ra >= 0 and ra == rb:
                    return None
            biome_a = layout_.biome[a_cx, a_cy]
            biome_b = layout_.biome[b_cx, b_cy]
            pair = frozenset((biome_a, biome_b))
            if pair in glass_pairs:
                return glass
            if pair in dark_glass_pairs:
                return dark_glass
            if pair in strip_pairs:
                return CONCRETE_STRIP
            return concrete

        # Strip-wall LED design: 4 horizontal flowing EM_CYAN bars at
        # varied z heights with offset tangent ranges so the wall reads
        # as "WAC-lighting-style" embedded LED lines rather than uniform
        # vertical strips. Z values are in coarse voxels (1 vx = 0.1 m).
        # Each entry is (z, t_frac0, t_frac1) — the strip spans the run
        # from tang_start + t_frac0*length to tang_start + t_frac1*length.
        STRIP_LINES = (
            (7, 0.05, 0.48),  # low bar, left half
            (12, 0.20, 0.85),  # mid bar, spanning centre
            (8, 0.55, 0.97),  # low bar, right half (creates stepped down look)
            (18, 0.10, 0.42),  # high bar, left
            (22, 0.55, 0.93),  # high bar, right
        )

        def _flush_run(axis: str, fixed_w: int, start_c: int, end_c: int, run_mat: str):
            """Emit one wall run; ``start_c`` and ``end_c`` are cell indices
            along the wall's tangent direction (inclusive..exclusive)."""
            tang_start = WALL_T + start_c * (CELL_W + WALL_T)
            tang_end = WALL_T + end_c * (CELL_W + WALL_T) - WALL_T
            length = tang_end - tang_start
            fill_mat = concrete if run_mat == CONCRETE_STRIP else run_mat
            if axis == "x":  # wall normal is X axis; tangent is Y
                w.fill((fixed_w, slice(tang_start, tang_end), slice(0, NZ_W)), fill_mat)
                if run_mat == CONCRETE_STRIP:
                    for z, f0, f1 in STRIP_LINES:
                        t0 = tang_start + int(length * f0)
                        t1 = tang_start + int(length * f1)
                        if t1 > t0:
                            w.fill((fixed_w, slice(t0, t1), z), materials.EM_CYAN)
            else:  # axis == "y"; wall normal is Y axis; tangent is X
                w.fill((slice(tang_start, tang_end), fixed_w, slice(0, NZ_W)), fill_mat)
                if run_mat == CONCRETE_STRIP:
                    for z, f0, f1 in STRIP_LINES:
                        t0 = tang_start + int(length * f0)
                        t1 = tang_start + int(length * f1)
                        if t1 > t0:
                            w.fill((slice(t0, t1), fixed_w, z), materials.EM_CYAN)

        # X-direction internal walls (wall normal = X axis).
        for cx_div in range(CELLS_X - 1):
            wx = WALL_T + (cx_div + 1) * (CELL_W + WALL_T) - 1
            cur_mat, cur_start = None, None
            for cy in range(CELLS_Y + 1):
                if cy < CELLS_Y:
                    mat = _pair_material(cx_div, cy, cx_div + 1, cy)
                else:
                    mat = None  # flush sentinel
                if mat != cur_mat:
                    if cur_mat is not None:
                        _flush_run("x", wx, cur_start, cy, cur_mat)
                    cur_mat, cur_start = mat, cy
        # Y-direction internal walls.
        for cy_div in range(CELLS_Y - 1):
            wy = WALL_T + (cy_div + 1) * (CELL_W + WALL_T) - 1
            cur_mat, cur_start = None, None
            for cx in range(CELLS_X + 1):
                if cx < CELLS_X:
                    mat = _pair_material(cx, cy_div, cx, cy_div + 1)
                else:
                    mat = None
                if mat != cur_mat:
                    if cur_mat is not None:
                        _flush_run("y", wy, cur_start, cx, cur_mat)
                    cur_mat, cur_start = mat, cx

    # --- Carve doorways (corporate doors use the wider hole) ---
    corp = layout_.corporate_door_locations or set()
    for cx in range(CELLS_X - 1):
        for cy in range(CELLS_Y):
            if not layout_.door_x[cx, cy]:
                continue
            is_corp = (cx, cy, "x") in corp
            dw = CORPORATE_DOOR_W if is_corp else DOOR_W
            dh = CORPORATE_DOOR_H if is_corp else DOOR_H
            wx = WALL_T + (cx + 1) * (CELL_W + WALL_T) - 1
            cell_y0 = WALL_T + cy * (CELL_W + WALL_T)
            y0 = cell_y0 + (CELL_W - dw) // 2
            y1 = y0 + dw
            w.knock_out((wx, slice(y0, y1), slice(WALL_T, WALL_T + dh)))

    for cx in range(CELLS_X):
        for cy in range(CELLS_Y - 1):
            if not layout_.door_y[cx, cy]:
                continue
            is_corp = (cx, cy, "y") in corp
            dw = CORPORATE_DOOR_W if is_corp else DOOR_W
            dh = CORPORATE_DOOR_H if is_corp else DOOR_H
            wy = WALL_T + (cy + 1) * (CELL_W + WALL_T) - 1
            cell_x0 = WALL_T + cx * (CELL_W + WALL_T)
            x0 = cell_x0 + (CELL_W - dw) // 2
            x1 = x0 + dw
            w.knock_out((slice(x0, x1), wy, slice(WALL_T, WALL_T + dh)))

    # --- Damaged cells: knock out a chunk of ceiling ---
    for cx, cy in layout_.cells:
        if layout_.biome[cx, cy] is not Biome.DAMAGED:
            continue
        cell_x0 = WALL_T + cx * (CELL_W + WALL_T)
        cell_y0 = WALL_T + cy * (CELL_W + WALL_T)
        # 6x6-coarse-voxel patch of ceiling
        hx = cell_x0 + CELL_W // 2 + int(rng.integers(-5, 6))
        hy = cell_y0 + CELL_W // 2 + int(rng.integers(-5, 6))
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if rng.random() < 0.5:
                    px, py = hx + dx, hy + dy
                    if 0 <= px < NX_W and 0 <= py < NY_W:
                        w.knock_out((px, py, NZ_W - 1))

    return w


# ============================================================================
# Cross-cell doorway frames
# ============================================================================


def _emit_doorway_frames(
    layout_: layout.Layout,
    rng: np.random.Generator,
    bodies: list,
) -> tuple:
    """Append a small body per doorway holding the door frame + slab stamp."""
    occupied = 0
    lights = 0
    rid = layout_.room_id
    corp = layout_.corporate_door_locations or set()
    # Force-open the door(s) along the central north spine so the reactor reads
    # from the camera. The spine runs at world X = entrance_cx (= reactor_cx).
    spine_cx = layout_.entrance[0]

    def _door_stamp(rng, *, is_corporate: bool, force_open: bool):
        if is_corporate:
            return equipment.build_corporate_glass_door(rng)
        return fixtures.build_doorway_frame(rng, force_open=force_open)

    # ----- X-walls (wall normal = world X) -----
    # The wall's tangent is world Y, so the door slides along world Y. The stamp
    # is built with its own X axis as the door-width direction, so it must be
    # rotated 90° about Z so that stamp.X aligns with world Y.
    for cx in range(CELLS_X - 1):
        for cy in range(CELLS_Y):
            if not layout_.door_x[cx, cy]:
                continue
            if (
                rid is not None
                and int(rid[cx, cy]) >= 0
                and int(rid[cx, cy]) == int(rid[cx + 1, cy])
            ):
                continue  # same multi-cell room -> no doorway here
            wx = WALL_T + (cx + 1) * (CELL_W + WALL_T) - 1
            cell_y0 = WALL_T + cy * (CELL_W + WALL_T)
            is_corporate = (cx, cy, "x") in corp
            dw_door = CORPORATE_DOOR_W if is_corporate else DOOR_W
            y0 = cell_y0 + (CELL_W - dw_door) // 2
            force_open = False
            frame = _door_stamp(
                rng, is_corporate=is_corporate, force_open=force_open
            ).rotate_z(1)
            # After rotate_z(1): frame.nx = wall depth (small), frame.ny = door width (large).
            world = World((frame.nx, frame.ny, frame.nz), VOXEL_O, materials.SEG_OF)
            world.place(frame, 0, 0, 0)
            # Centre the door stamp along Y on the carved-hole centre so any
            # stamp width (industrial 80, corporate 120) lines up with the hole.
            # After rotate_z(1): frame.nx = original-y span (small, depth),
            #                    frame.ny = original-x span (large, door width).
            hole_centre_fine_y = (y0 + dw_door // 2) * WO_RATIO
            fine_x = (
                wx * WO_RATIO - frame.nx // 2
            )  # straddle the wall along X (small dim)
            fine_y = hole_centre_fine_y - frame.ny // 2  # centre door width on hole
            fine_z = WALL_T * WO_RATIO
            body = world.emit_body(
                origin_offset=(
                    fine_x * VOXEL_O - EXTENT_X / 2,
                    fine_y * VOXEL_O - EXTENT_Y / 2,
                    fine_z * VOXEL_O,
                )
            )
            if body is not None:
                bodies.append(body)
            occupied += world.occupied_count()
            lights += world.emissive_count(materials.EMISSIVE_NAMES)
    # ----- Y-walls (wall normal = world Y) -----
    # The wall's tangent is world X, which matches the stamp's natural X axis,
    # so NO rotation is needed here.
    for cx in range(CELLS_X):
        for cy in range(CELLS_Y - 1):
            if not layout_.door_y[cx, cy]:
                continue
            if (
                rid is not None
                and int(rid[cx, cy]) >= 0
                and int(rid[cx, cy]) == int(rid[cx, cy + 1])
            ):
                continue  # same multi-cell room -> no doorway here
            wy = WALL_T + (cy + 1) * (CELL_W + WALL_T) - 1
            cell_x0 = WALL_T + cx * (CELL_W + WALL_T)
            is_corporate = (cx, cy, "y") in corp
            dw_door = CORPORATE_DOOR_W if is_corporate else DOOR_W
            x0 = cell_x0 + (CELL_W - dw_door) // 2
            # Force the doors on the central N-S spine open so the reactor reads.
            force_open = cx == spine_cx
            frame = _door_stamp(rng, is_corporate=is_corporate, force_open=force_open)
            # No rotation: frame.nx = door width (large), frame.ny = wall depth (small).
            world = World((frame.nx, frame.ny, frame.nz), VOXEL_O, materials.SEG_OF)
            world.place(frame, 0, 0, 0)
            # Centre the door stamp along X on the carved-hole centre so any
            # stamp width (industrial 80, corporate 120) lines up with the hole.
            hole_centre_fine_x = (x0 + dw_door // 2) * WO_RATIO
            fine_x = hole_centre_fine_x - frame.nx // 2  # centre door width on hole
            fine_y = wy * WO_RATIO - frame.ny // 2  # straddle the wall along Y
            fine_z = WALL_T * WO_RATIO
            body = world.emit_body(
                origin_offset=(
                    fine_x * VOXEL_O - EXTENT_X / 2,
                    fine_y * VOXEL_O - EXTENT_Y / 2,
                    fine_z * VOXEL_O,
                )
            )
            if body is not None:
                bodies.append(body)
            occupied += world.occupied_count()
            lights += world.emissive_count(materials.EMISSIVE_NAMES)
    return occupied, lights


# ============================================================================
# Sky and floor (separate from the walls voxel structure)
# ============================================================================


SKY = (4, 5, 8)  # near-black
# SKY = (200, 200, 200)  # white debug

# Dark concrete plane the structure sits on.
GROUND_COLOR = (28, 30, 36)


def _sky_and_floor(resources: list, essences: list, entities: list) -> None:
    """Add the near-black sky background and the dark concrete ground plane."""
    resources.append(rsc.resource.TextureConstant("sky#color", value=_common._rgb(SKY)))
    entities.append(
        rsc.entity.CosmicBackgroundUniform("sky", texture_reference="sky#color")
    )

    resources.append(
        rsc.resource.TextureConstant("ground#albedo", value=_common._rgb(GROUND_COLOR))
    )
    resources.append(rsc.resource.TextureConstant("ground#rm", value=(0.6, 0.0, 0.0)))
    essences.append(rsc.essence.Material("ground"))
    essences.append(
        rsc.essence.MaterialPhysicalPropertyFixed(
            "ground#phys", material_reference="ground"
        )
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "ground#optic",
            material_reference="ground",
            albedo_texture_reference="ground#albedo",
            roughness_metalness_texture_reference="ground#rm",
        )
    )
    entities.append(rsc.entity.FloorFlat("floor", material_reference="ground"))


# ============================================================================
# Build
# ============================================================================


def build() -> rsc.World:
    """Build the full :class:`rsc.World` for the facility scene."""
    rng = np.random.default_rng(SEED)
    material_references = list(materials.MATERIAL_NAMES)

    resources: list = []
    essences: list = []
    entities: list = []
    _sky_and_floor(resources, essences, entities)
    materials.build_materials(resources, essences)

    # --- Layout ---------------------------------------------------------
    t = time.time()
    layout_ = layout.generate_diagram_layout(rng)
    t_layout = time.time() - t

    # --- Walls structure -----------------------------------------------
    t = time.time()
    walls_world = _build_walls(layout_, rng)
    # Centre the walls structure on the world origin in meters.
    walls_body = walls_world.emit_body(
        origin_offset=(-EXTENT_X / 2, -EXTENT_Y / 2, 0.0)
    )
    entities.append(
        rsc.entity.StructureVoxel(
            "walls",
            material_references=material_references,
            bodies=[walls_body] if walls_body is not None else [],
        )
    )
    t_walls = time.time() - t

    # --- Objects structure (room-level, via WorldEmitter) --------------
    t = time.time()
    emitter = WorldEmitter(VOXEL_O, materials.SEG_OF, materials.EMISSIVE_NAMES)
    biomes.emit_all_rooms(layout_, emitter, rng)
    obj_occ = emitter.voxels_emitted
    obj_lights = emitter.lights_emitted
    object_bodies = list(emitter.bodies)
    door_occ, door_lights = _emit_doorway_frames(layout_, rng, object_bodies)
    entities.append(
        rsc.entity.StructureVoxel(
            "objects",
            material_references=material_references,
            bodies=object_bodies,
        )
    )
    t_objects = time.time() - t

    # --- Report ---------------------------------------------------------
    walls_occ = walls_world.occupied_count()
    total_occ = walls_occ + obj_occ + door_occ
    total_lights = obj_lights + door_lights

    print(
        f"[lab] layout {t_layout:.2f}s  walls {t_walls:.2f}s  objects {t_objects:.2f}s"
    )
    print(
        f"[lab] walls={walls_occ:>6d}  objects={obj_occ:>6d}  doorways={door_occ:>5d}  "
        f"TOTAL={total_occ:>7d} voxels"
    )
    print(
        f"[lab] lights={total_lights:>5d}  extent={EXTENT_X:.1f}x{EXTENT_Y:.1f} m  "
        f"cells={CELLS_X}x{CELLS_Y}"
    )

    # Stash stats on the module so tests / sanity checks can read them.
    build.last_stats = {
        "walls_occupied": walls_occ,
        "objects_occupied": obj_occ,
        "doorways_occupied": door_occ,
        "total_occupied": total_occ,
        "total_lights": total_lights,
    }

    # ReSTIR DI is the point of this scene; SVGF denoises the 1-spp output.
    simulation = rsc.simulation.Simulation(
        environment=rsc.simulation.Environment(gravity_acceleration=-9.81),
        vision=rsc.simulation.Vision(denoise=rsc.simulation.VisionDenoiseSVGF(temporal_alpha=0.1, temporal_alpha_moments=0.1)),
        runtime=rsc.simulation.Runtime(log_level="info"),
    )
    return rsc.World(
        "lab",
        simulation=simulation,
        resources=resources,
        essences=essences,
        entities=entities,
    )


# ============================================================================
# Main / Viewer launch
# ============================================================================


def _cell_center_world(cx: int, cy: int) -> Tuple[float, float, float]:
    """World-meters coordinates of the centre of cell ``(cx, cy)``."""
    ox, oy, oz = cell_origin_w(cx, cy)
    # Walls structure is emitted with origin offset (-EXTENT_X/2, -EXTENT_Y/2, 0).
    wx = -EXTENT_X / 2 + (ox + CELL_W / 2) * VOXEL_W
    wy = -EXTENT_Y / 2 + (oy + CELL_W / 2) * VOXEL_W
    wz = oz * VOXEL_W
    return wx, wy, wz


def main():
    """Build and launch the facility scene with an eye-level camera looking
    down the central corridor toward the reactor."""
    entrance_x, entrance_y, _ = _cell_center_world(0, CELLS_Y // 2)
    reactor_x, reactor_y, _ = _cell_center_world(CELLS_X - 1, CELLS_Y // 2)
    # Stand inside the entrance cell, eye-level, looking down the corridor.
    camera_position = (entrance_x + 0.5, entrance_y, 1.6)
    camera_focus = (reactor_x, reactor_y, 1.4)
    _common.launch(
        build(),
        None,
        camera_position=camera_position,
        camera_focus=camera_focus,
        samples_per_pixel=16,
        max_depth=4,
    )


if __name__ == "__main__":
    main()
