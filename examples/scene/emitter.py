# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""WorldEmitter — decoupled per-stamp placement into a body list.

Replaces the previous per-cell objects-emission scheme. Each stamp placed
becomes its own small fixed :class:`rsc.entity.StructureVoxelBody`, anchored at
world meter coordinates (NOT cell-local coordinates), so multi-cell rooms get
seamless content across cell boundaries. The collected bodies are read from
:attr:`WorldEmitter.bodies` and handed to one :class:`rsc.entity.StructureVoxel`.

The emitter takes care of:

* converting world-meter anchors into the per-body :class:`World`'s origin
  offset (each body carries its own grid origin),
* applying optional rotation about Z to the stamp,
* tracking total body count, voxel count, and emissive ("light") count
  so build-time stats can be reported by ``main.build()``.

Two convenience constructs are exposed for repeating-content rooms:

* :meth:`WorldEmitter.array` — a regular X×Y grid of stamps (server farm,
  shelf row, ceiling strip grid).
* :meth:`WorldEmitter.repeat_along` — a 1-D run of stamps along an axis
  at a fixed pitch (display-cabinet row, pipe-run wall, beam row).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .stamp import Stamp
from .world import World


class WorldEmitter:
    """Place stamps as individual bodies at world meters, collected in a list.

    Args:
        voxel_size: physical voxel edge length (meters) — must match the
            structure's ``voxel_size``.
        seg_of: material name -> segment id (typically
            :data:`scene.materials.SEG_OF`).
        emissive_names: frozenset of material names that count as lights
            (typically :data:`scene.materials.EMISSIVE_NAMES`).
    """

    def __init__(self, voxel_size: float, seg_of: dict, emissive_names: frozenset):
        self.voxel_size = float(voxel_size)
        self.seg_of = seg_of
        self.emissive_names = emissive_names
        self.bodies: list = []
        self.bodies_emitted = 0
        self.voxels_emitted = 0
        self.lights_emitted = 0

    # ------------------------------------------------------------------ place

    def place(
        self,
        stamp: Stamp,
        world_x_m: float,
        world_y_m: float,
        world_z_m: float,
        rotate: int = 0,
    ) -> None:
        """Paste a single ``stamp`` at world meters ``(world_x_m, world_y_m,
        world_z_m)`` and emit it as its own body. ``rotate`` is in 90 deg
        units about the Z axis (0..3).
        """
        if rotate:
            stamp = stamp.rotate_z(rotate)
        # Each placement is its own tiny World sized exactly to the stamp.
        tiny = World(stamp.shape, self.voxel_size, self.seg_of)
        n = stamp.paste_into(
            tiny.materials,
            tiny.segments,
            0,
            0,
            0,
            self.seg_of,
            mode="replace",
        )
        if n == 0:
            return
        body = tiny.emit_body(origin_offset=(world_x_m, world_y_m, world_z_m))
        if body is None:
            return
        self.bodies.append(body)
        self.bodies_emitted += 1
        self.voxels_emitted += n
        self.lights_emitted += tiny.emissive_count(self.emissive_names)

    # ---------------------------------------------------------------- array

    def array(
        self,
        stamp_builder: Callable,
        *,
        x0: float,
        y0: float,
        z: float,
        x_count: int,
        y_count: int,
        x_pitch: float,
        y_pitch: float,
        rng: np.random.Generator,
        rotate: int = 0,
        builder_kwargs: Optional[dict] = None,
    ) -> None:
        """Place an ``x_count x y_count`` grid of stamps with the given pitches.

        Each slot calls ``stamp_builder(rng, **builder_kwargs)`` to get its
        stamp; with cached builders this is fast and gives RNG-shuffled
        variants. Useful for a server farm grid, a shelf row, ceiling strips.
        """
        kw = builder_kwargs or {}
        for ix in range(x_count):
            for iy in range(y_count):
                stamp = stamp_builder(rng, **kw)
                self.place(
                    stamp, x0 + ix * x_pitch, y0 + iy * y_pitch, z, rotate=rotate
                )

    # -------------------------------------------------------- repeat_along

    def repeat_along(
        self,
        stamp_builder: Callable,
        *,
        start: float,
        end: float,
        axis: str,
        pitch: float,
        offset_xy: Tuple[float, float],
        z: float,
        rng: np.random.Generator,
        rotate: int = 0,
        builder_kwargs: Optional[dict] = None,
    ) -> None:
        """Place stamps along a line from ``start`` to ``end`` meters at fixed
        ``pitch``. ``axis`` is 'x' or 'y'; ``offset_xy`` is the perpendicular
        anchor for the OTHER world coordinate.

        ``start`` and ``end`` are world meters. The line steps by ``pitch``
        meters; ``stamp_builder`` is called once per slot.
        """
        kw = builder_kwargs or {}
        if pitch <= 0:
            raise ValueError("pitch must be > 0")
        n = int((end - start) / pitch) + 1
        for i in range(n):
            u = start + i * pitch
            if axis == "x":
                wx, wy = u, offset_xy[1]
            elif axis == "y":
                wx, wy = offset_xy[0], u
            else:
                raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
            stamp = stamp_builder(rng, **kw)
            self.place(stamp, wx, wy, z, rotate=rotate)


# ============================================================================
# Layout helpers
# ============================================================================


def room_world_bounds(layout, biome) -> Tuple[float, float, float, float]:
    """Return the ``(x0_m, y0_m, x1_m, y1_m)`` world-meter rectangle covering
    the room whose ``layout.biome`` is ``biome``.

    For multi-cell rooms this returns the bounding rectangle of all the cells
    in the room. For biomes that appear in multiple separate rooms (e.g.
    BIO_LAB also used for the observation labs), this returns the bounding
    box of ALL cells with that biome — call sites that need a specific room
    should use :func:`room_world_bounds_by_id` instead.
    """
    from .constants import (
        WALL_T,
        CELL_W,
        EXTENT_X,
        EXTENT_Y,
        VOXEL_W,
    )

    mask = np.asarray(layout.biome == biome)
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0
    cxs, cys = np.where(mask)
    cx0, cx1 = int(cxs.min()), int(cxs.max())
    cy0, cy1 = int(cys.min()), int(cys.max())
    # Cell (cx, cy) interior starts at coarse-voxel
    # (WALL_T + cx*(CELL_W+WALL_T), WALL_T + cy*(CELL_W+WALL_T)).
    x0_c = WALL_T + cx0 * (CELL_W + WALL_T)
    y0_c = WALL_T + cy0 * (CELL_W + WALL_T)
    # Cell (cx1, cy1) interior ends at coarse-voxel
    # x0_c1 + CELL_W = WALL_T + cx1*(CELL_W+WALL_T) + CELL_W.
    x1_c = WALL_T + cx1 * (CELL_W + WALL_T) + CELL_W
    y1_c = WALL_T + cy1 * (CELL_W + WALL_T) + CELL_W
    # Convert to world meters (walls structure origin at (-EXTENT_X/2, -EXTENT_Y/2)).
    return (
        x0_c * VOXEL_W - EXTENT_X / 2,
        y0_c * VOXEL_W - EXTENT_Y / 2,
        x1_c * VOXEL_W - EXTENT_X / 2,
        y1_c * VOXEL_W - EXTENT_Y / 2,
    )


def room_world_bounds_by_id(layout, room_id: int) -> Tuple[float, float, float, float]:
    """World-meter bounds of the cells with the given ``room_id``."""
    from .constants import (
        WALL_T,
        CELL_W,
        EXTENT_X,
        EXTENT_Y,
        VOXEL_W,
    )

    rid_arr = layout.room_id
    mask = np.asarray(rid_arr == room_id)
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0
    cxs, cys = np.where(mask)
    cx0, cx1 = int(cxs.min()), int(cxs.max())
    cy0, cy1 = int(cys.min()), int(cys.max())
    x0_c = WALL_T + cx0 * (CELL_W + WALL_T)
    y0_c = WALL_T + cy0 * (CELL_W + WALL_T)
    x1_c = WALL_T + cx1 * (CELL_W + WALL_T) + CELL_W
    y1_c = WALL_T + cy1 * (CELL_W + WALL_T) + CELL_W
    return (
        x0_c * VOXEL_W - EXTENT_X / 2,
        y0_c * VOXEL_W - EXTENT_Y / 2,
        x1_c * VOXEL_W - EXTENT_X / 2,
        y1_c * VOXEL_W - EXTENT_Y / 2,
    )


def all_rooms_with_biome(layout, biome) -> list:
    """Return the list of ``room_id`` values whose biome matches ``biome``."""
    rid_arr = layout.room_id
    mask = np.asarray(layout.biome == biome)
    if rid_arr is None or not mask.any():
        return []
    rids = np.unique(rid_arr[mask])
    return [int(r) for r in rids if int(r) >= 0]
