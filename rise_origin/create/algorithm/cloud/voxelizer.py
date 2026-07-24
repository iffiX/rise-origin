# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Sweep a cloud density snapshot into shell-layered voxels.

The density snapshot, normalized to [0, 1] and placed in the field frame,
is sampled at every output cell and thresholded at an iso value into a
``material_grid`` whose present cells carry a material sid and whose empty
cells hold ``RS_NULL_INDEX``. The sid is the depth shell: a cell's
distance to the nearest air, as a fraction of the body's deepest reach,
bands into the skin at sid zero out to the core at the highest sid, so a
body of ``layer_count`` shells stacks that many materials from skin to
core and the shells scale with the cloud. Each shell is its own material,
so its boundary is a rendered face and the inner shells show through the
outer ones. Bodies stream one at a time into the entity.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from rise_origin import RQuat3rf, RS_NULL_INDEX, RVec3rf
from rise_origin.rsc import entity

from ...protocol.logger import Logger
from ...protocol.worker_pool import WorkerPool
from ..grid_span import GridSpan

__all__ = ["CloudBody", "CloudSnapshot", "CloudVoxelizer"]

# The depth fraction one shell peels; each shell is this share of the
# body's deepest reach, so the shells scale with the cloud. The last shell
# takes the whole remaining core.
_SHELL_PEEL_FRACTION = 0.3


@dataclass
class CloudSnapshot:
    """One cloud density snapshot placed in the field frame.

    Attributes:
        density: The normalized cloud density in [0, 1], float32,
            indexed x, y, z.
        grid: The frame of the density samples (unit: m).
    """

    density: np.ndarray
    grid: GridSpan


@dataclass
class CloudBody:
    """One swept tile of a cloud field, ready to become an engine body.

    Attributes:
        material_grid: The per-cell material sid, uint32; ``RS_NULL_INDEX``
            marks an empty cell. The sid is the depth shell, zero at the
            skin up to the core.
        origin: The body min corner relative to the structure origin
            (unit: m).
        voxel_size: The cell edge length (unit: m).
    """

    material_grid: np.ndarray
    origin: Tuple[float, float, float]
    voxel_size: float


class CloudVoxelizer:
    """Sweeps a cloud snapshot into streaming engine bodies.

    Args:
        pool: The session worker pool, or None for a serial build that the
            pooled path matches byte for byte on one device.
        logger: The session logger, or None for a silent build.
    """

    def __init__(
        self, pool: Optional[WorkerPool] = None, logger: Optional[Logger] = None
    ) -> None:
        self._pool = pool
        self._logger = logger

    def make_bodies(
        self,
        snapshot: CloudSnapshot,
        grid: GridSpan,
        *,
        iso: float = 0.1,
        layer_count: int = 1,
        structure_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_body_cells: int = 8_000_000,
    ) -> List[CloudBody]:
        """Return the swept bodies of one cloud, one per grid tile.

        Args:
            snapshot: The cloud snapshot in the field frame.
            grid: The whole output voxel grid in the field frame.
            iso: The density at or above which a cell becomes a voxel.
            layer_count: How many shells the body is banded into, one
                material sid each from the skin to the core.
            structure_origin: The structure origin (unit: m); body
                positions are answered relative to it.
            max_body_cells: The most cells one tile may hold before the
                grid splits.
        """
        tiles = grid.make_tiles(max_body_cells, overlap_cell_num=0)
        tasks = [
            (snapshot, spec, iso, layer_count, structure_origin) for _, spec in tiles
        ]
        if self._pool is None:
            bodies = [CloudVoxelizer._make_tile_body(*task) for task in tasks]
        else:
            bodies = self._pool.starmap(CloudVoxelizer._make_tile_body, tasks)
        result = [body for body in bodies if body is not None]
        if self._logger is not None:
            self._logger.info("swept: %d bodies from %d tiles", len(result), len(tiles))
        return result

    @staticmethod
    def iterate_structure_bodies(bodies: List[CloudBody]):
        """Yield each body as an engine body, clearing its grids as it goes.

        Args:
            bodies: The swept cloud bodies to convert.
        """
        for body in bodies:
            yield entity.StructureVoxelBody.from_grid(
                body.material_grid,
                voxel_size=body.voxel_size,
                relative_origin_position=RVec3rf(tuple(body.origin)),
                relative_orientation=RQuat3rf.unit(),
            )
            body.material_grid = None

    @staticmethod
    def _make_tile_body(
        snapshot: CloudSnapshot,
        tile: GridSpan,
        iso: float,
        layer_count: int,
        structure_origin: Tuple[float, float, float],
    ) -> Optional[CloudBody]:
        """Return one tile's swept body, or None when the tile is empty.

        Args:
            snapshot: The cloud snapshot in the field frame.
            tile: The tile's evaluation grid.
            iso: The density threshold for a present voxel.
            layer_count: How many shells the body is banded into, one
                material sid each from the skin to the core.
            structure_origin: The structure origin (unit: m).
        """
        density = CloudVoxelizer._sample_density(snapshot, tile)
        present = CloudVoxelizer._clean_speckle(density >= iso)
        if not present.any():
            return None
        shell = CloudVoxelizer._make_shell_layers(present, layer_count)
        material_grid = np.where(present, shell, np.uint32(RS_NULL_INDEX))
        return CloudVoxelizer._trim_body(material_grid, tile, structure_origin)

    @staticmethod
    def _sample_density(snapshot: CloudSnapshot, tile: GridSpan) -> np.ndarray:
        """Return the snapshot density at every cell center of a tile.

        Trilinear sampling as a pure function of world position, so tile
        seams agree; positions outside the snapshot clamp to its edge
        cells.

        Args:
            snapshot: The cloud snapshot in the field frame.
            tile: The tile's evaluation grid.
        """
        values = snapshot.density
        lines = []
        for side in range(3):
            centers = (
                tile.origin[side]
                + (np.arange(tile.cell_num[side]) + 0.5) * tile.cell_size
            )
            coords = (
                centers - snapshot.grid.origin[side]
            ) / snapshot.grid.cell_size - 0.5
            lines.append(
                np.clip(coords, 0.0, snapshot.grid.cell_num[side] - 1.0).astype(
                    np.float32
                )
            )
        parts = []
        for side, line in enumerate(lines):
            low = np.floor(line)
            shape = [1, 1, 1]
            shape[side] = -1
            parts.append(
                (
                    low.astype(np.int64).reshape(shape),
                    (line - low).astype(np.float32).reshape(shape),
                )
            )
        (ix, tx), (iy, ty), (iz, tz) = parts
        jx = np.minimum(ix + 1, values.shape[0] - 1)
        jy = np.minimum(iy + 1, values.shape[1] - 1)
        jz = np.minimum(iz + 1, values.shape[2] - 1)
        return (
            values[ix, iy, iz] * (1 - tx) * (1 - ty) * (1 - tz)
            + values[jx, iy, iz] * tx * (1 - ty) * (1 - tz)
            + values[ix, jy, iz] * (1 - tx) * ty * (1 - tz)
            + values[jx, jy, iz] * tx * ty * (1 - tz)
            + values[ix, iy, jz] * (1 - tx) * (1 - ty) * tz
            + values[jx, iy, jz] * tx * (1 - ty) * tz
            + values[ix, jy, jz] * (1 - tx) * ty * tz
            + values[jx, jy, jz] * tx * ty * tz
        )

    @staticmethod
    def _clean_speckle(present: np.ndarray) -> np.ndarray:
        """Return the occupancy with isolated voxels and pinholes removed.

        A one-pass face-neighbor clean: a present cell with no present
        neighbor is dropped, and an empty cell walled in by five or six
        present neighbors is filled. This clears the salt-and-pepper a
        hard threshold of a soft field leaves, while keeping the billowy
        silhouette.

        Args:
            present: The raw occupancy from the density threshold.
        """
        padded = np.pad(present, 1).astype(np.uint8)
        count = (
            padded[:-2, 1:-1, 1:-1]
            + padded[2:, 1:-1, 1:-1]
            + padded[1:-1, :-2, 1:-1]
            + padded[1:-1, 2:, 1:-1]
            + padded[1:-1, 1:-1, :-2]
            + padded[1:-1, 1:-1, 2:]
        )
        cleaned = present.copy()
        cleaned[present & (count == 0)] = False
        cleaned[~present & (count >= 5)] = True
        return cleaned

    @staticmethod
    def _make_shell_layers(present: np.ndarray, layer_count: int) -> np.ndarray:
        """Return the per-cell material sid, the depth shell index.

        Each cell's distance to the nearest air is taken as a fraction of
        the body's deepest reach, so the shells scale with the cloud. The
        outer skin is sid zero out to one peel fraction deep, each further
        peel a higher sid, and the last shell keeps the whole remaining
        core. A thin wisp stays all skin; a thick body grows a full core.

        Args:
            present: The tile's present-cell mask.
            layer_count: How many shells to band the body into.
        """
        sids = np.zeros(present.shape, dtype=np.uint32)
        if layer_count <= 1:
            return sids
        distance = ndimage.distance_transform_edt(present)
        deepest = float(distance.max())
        if deepest <= 0.0:
            return sids
        fraction = distance / deepest
        shell = np.minimum(
            (fraction / _SHELL_PEEL_FRACTION).astype(np.int64), layer_count - 1
        )
        sids[present] = shell[present].astype(np.uint32)
        return sids

    @staticmethod
    def _trim_body(
        material_grid: np.ndarray,
        tile: GridSpan,
        structure_origin: Tuple[float, float, float],
    ) -> CloudBody:
        """Return the body trimmed to its filled bounding box.

        Args:
            material_grid: The per-cell material sid, ``RS_NULL_INDEX`` for
                empty cells.
            tile: The tile's evaluation grid.
            structure_origin: The structure origin (unit: m).
        """
        filled = np.argwhere(material_grid != RS_NULL_INDEX)
        lo = filled.min(axis=0)
        hi = filled.max(axis=0) + 1
        cut = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        origin = tuple(
            float(tile.origin[a] + lo[a] * tile.cell_size - structure_origin[a])
            for a in range(3)
        )
        return CloudBody(
            material_grid=np.ascontiguousarray(material_grid[cut]),
            origin=origin,
            voxel_size=tile.cell_size,
        )
