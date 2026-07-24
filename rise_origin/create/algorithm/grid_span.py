# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The grid contract: one description every grid tensor is indexed by.

A ``GridSpan`` names an evaluation grid in world space and holds no
tensor. Because its origin is a world coordinate and every field is a
pure function of world position, two grids that overlap the same region
sample identical values on their shared cells, at any cell size.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["GridSpan"]


@dataclass(frozen=True)
class GridSpan:
    """One evaluation grid in world space.

    Cell centers sit at ``origin + (index + 0.5) * cell_size`` per side,
    so the span fully names every coordinate without storing any.

    Attributes:
        origin: The world position of the grid's min corner (unit: m).
        cell_size: The cell edge length (unit: m).
        cell_num: The cell counts along x, y, and z.
    """

    origin: Tuple[float, float, float]
    cell_size: float
    cell_num: Tuple[int, int, int]

    def get_extent(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the world min and max corners of the grid."""
        lo = np.asarray(self.origin, dtype=np.float64)
        hi = lo + self.cell_size * np.asarray(self.cell_num, dtype=np.float64)
        return lo, hi

    def get_cell_num(self) -> int:
        """Return the total cell count of the grid."""
        return int(self.cell_num[0]) * int(self.cell_num[1]) * int(self.cell_num[2])

    def get_memory_bytes(self, bytes_per_cell: int) -> int:
        """Return the memory forecast of one grid-shaped array.

        Args:
            bytes_per_cell: The bytes each cell holds across the working
                arrays.
        """
        return self.get_cell_num() * int(bytes_per_cell)

    def make_axes(self, device: Optional[str] = None):
        """Return the three cell-center coordinate lines of the grid.

        Args:
            device: None returns numpy lines on the host; a torch device
                name returns torch lines living there.

        Returns:
            Three 1-D float64 arrays of cell centers along x, y, and z.
        """
        lines = [
            self.origin[side] + (np.arange(self.cell_num[side]) + 0.5) * self.cell_size
            for side in range(3)
        ]
        if device is None:
            return tuple(lines)
        import torch

        return tuple(torch.as_tensor(line, device=device) for line in lines)

    def make_finer_grid_span(self, subdivision: int) -> "GridSpan":
        """Return the same extent divided into finer cells.

        Args:
            subdivision: How many finer cells each cell splits into per
                side; the finer centers nest exactly inside the coarse
                cells.
        """
        return GridSpan(
            self.origin,
            self.cell_size / int(subdivision),
            tuple(count * int(subdivision) for count in self.cell_num),
        )

    def make_tiles(
        self, max_cell_num: int, overlap_cell_num: int
    ) -> List[Tuple[Tuple[int, int, int], "GridSpan"]]:
        """Return overlapping sub-grids that jointly cover the grid.

        The longest side splits repeatedly until every tile holds at
        most ``max_cell_num`` cells including its overlap band, so a job
        too large for one device allocation runs tile by tile.

        Args:
            max_cell_num: The most cells one tile may hold.
            overlap_cell_num: The cells each tile extends past its core
                on every split side.

        Returns:
            Pairs of the tile's core start index in this grid and the
            tile's span including overlap.
        """
        splits = [1, 1, 1]
        while True:
            cores = [-(-self.cell_num[side] // splits[side]) for side in range(3)]
            tile_counts = [
                min(cores[side] + 2 * overlap_cell_num, self.cell_num[side])
                for side in range(3)
            ]
            if tile_counts[0] * tile_counts[1] * tile_counts[2] <= max_cell_num:
                break
            longest = max(range(3), key=lambda side: cores[side])
            if cores[longest] <= 1:
                break
            splits[longest] += 1
        tiles = []
        for ix in range(splits[0]):
            for iy in range(splits[1]):
                for iz in range(splits[2]):
                    start = [0, 0, 0]
                    stop = [0, 0, 0]
                    for side, index in zip(range(3), (ix, iy, iz)):
                        core = -(-self.cell_num[side] // splits[side])
                        lo = min(index * core, self.cell_num[side])
                        hi = min(lo + core, self.cell_num[side])
                        start[side] = max(lo - overlap_cell_num, 0)
                        stop[side] = min(hi + overlap_cell_num, self.cell_num[side])
                    span = GridSpan(
                        tuple(
                            self.origin[side] + start[side] * self.cell_size
                            for side in range(3)
                        ),
                        self.cell_size,
                        tuple(stop[side] - start[side] for side in range(3)),
                    )
                    tiles.append(((start[0], start[1], start[2]), span))
        return tiles
