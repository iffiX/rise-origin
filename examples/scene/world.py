# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The :class:`World` wrapper around a single voxel body.

A :class:`World` owns one (materials, segments) pair of arrays at one voxel
resolution. Stamps are pasted into it via :meth:`World.place`. When all
content has been placed, :meth:`World.emit_body` walks the world's voxel
arrays and returns one :class:`rsc.entity.StructureVoxelBody`.

The facility scene uses *two* worlds, one per resolution: a coarse 0.10 m
world for the architectural shell and a fine 0.02 m world for fixtures and
furniture. Both share the same :data:`scene.materials.SEG_OF` map so a single
material list is registered globally.

A future ``fab.world.World`` would generalise this with multi-body chunking
(emit one body per cell instead of one body per world) to bound peak memory
on very large fine grids — see ``layout.py`` for where that would slot in.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from rise_origin import RS_NULL_INDEX, rsc

from .stamp import Stamp


class World:
    """A voxel world: one (materials, segments) grid + a name->segment map.

    Args:
        shape: ``(NX, NY, NZ)`` world extent in voxels.
        voxel_size: physical voxel edge length in meters (e.g. ``0.10`` for the
            coarse walls grid, ``0.02`` for the fine objects grid).
        seg_of: material-name -> segment-id map (typically
            :data:`scene.materials.SEG_OF`).
    """

    def __init__(self, shape: Tuple[int, int, int], voxel_size: float, seg_of: dict):
        self.shape = shape
        self.voxel_size = float(voxel_size)
        self.seg_of = seg_of
        self.materials = np.full(shape, None, dtype=object)
        self.segments = np.zeros(shape, dtype=np.int64)
        self.total_pasted = 0

    # ----------------------------------------------------------------- placing

    def place(
        self,
        stamp: Stamp,
        ax: int,
        ay: int,
        az: int,
        rotate: int = 0,
        mode: str = "overlay",
    ) -> int:
        """Paste ``stamp`` at the given anchor (lower corner), optionally rotated.

        Returns the number of voxels written.
        """
        if rotate:
            stamp = stamp.rotate_z(rotate)
        n = stamp.paste_into(
            self.materials, self.segments, ax, ay, az, self.seg_of, mode=mode
        )
        self.total_pasted += n
        return n

    def fill(self, slc: Tuple, material: str) -> None:
        """Write ``material`` into a world region (no overlay check, always replaces).

        Convenience for the shell pass — slabs of floor / ceiling / walls.
        """
        self.materials[slc] = material
        self.segments[slc] = self.seg_of[material]

    def knock_out(self, slc: Tuple) -> int:
        """Clear voxels in a world region back to ``None``. Returns count removed."""
        view = self.materials[slc]
        # `view` may be a scalar (all-int indexing) or an array; normalize.
        present = view != None  # noqa: E711
        if np.isscalar(present):
            n = 1 if present else 0
        else:
            n = int(np.asarray(present).sum())
        self.materials[slc] = None
        self.segments[slc] = 0
        return n

    # ---------------------------------------------------------------- stats

    def occupied_count(self) -> int:
        return int(np.count_nonzero(self.materials != None))  # noqa: E711

    def emissive_count(self, emissive_names: frozenset) -> int:
        """Count voxels whose material is one of ``emissive_names`` (one light each)."""
        if self.occupied_count() == 0:
            return 0
        mats = self.materials
        flat = mats.ravel()
        present_mask = flat != None  # noqa: E711
        if not present_mask.any():
            return 0
        # Vectorise the lookup via numpy.
        present_names = flat[present_mask]
        is_em = np.array([n in emissive_names for n in present_names])
        return int(is_em.sum())

    # ----------------------------------------------------------------- emit

    def emit_body(
        self,
        origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Optional["rsc.entity.StructureVoxelBody"]:
        """Return this world as a single fixed :class:`rsc.entity.StructureVoxelBody`.

        ``origin_offset`` is the body's ``relative_origin_position`` (the world's
        grid origin in meters). Use this to centre the structure on the world
        origin in the parent :class:`rsc.World` frame. Returns ``None`` when the
        world is empty.

        Each material owns its own segment, so the segment-id grid doubles as the
        material-reference grid; empty voxels are masked to ``RS_NULL_INDEX``.
        """
        if self.occupied_count() == 0:
            return None
        occupancy = self.materials != None  # noqa: E711
        ids = np.where(occupancy, self.segments, RS_NULL_INDEX)
        return rsc.entity.StructureVoxelBody.from_grid(
            occupancy,
            material_grid=ids,
            segment_grid=ids,
            voxel_size=self.voxel_size,
            relative_origin_position=tuple(float(v) for v in origin_offset),
        )
