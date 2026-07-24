# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the grid contract."""

import numpy as np

from rise_origin.create.algorithm import GridSpan


class TestGridSpec:
    def test_axes_and_extent_round_trip(self):
        spec = GridSpan((1.0, -2.0, 0.5), 0.25, (4, 2, 8))
        ax, ay, az = spec.make_axes()
        assert np.allclose(ax, [1.125, 1.375, 1.625, 1.875])
        assert np.allclose(ay, [-1.875, -1.625])
        lo, hi = spec.get_extent()
        assert np.allclose(lo, [1.0, -2.0, 0.5])
        assert np.allclose(hi, [2.0, -1.5, 2.5])
        assert spec.get_cell_num() == 64
        assert spec.get_memory_bytes(4) == 256

    def test_finer_centers_nest_inside_the_coarse_cells(self):
        spec = GridSpan((0.0, 0.0, 0.0), 0.4, (3, 3, 3))
        fine = spec.make_finer_grid_span(4)
        assert fine.cell_size == 0.1
        assert fine.cell_num == (12, 12, 12)
        coarse_lo, coarse_hi = spec.get_extent()
        fine_lo, fine_hi = fine.get_extent()
        assert np.allclose(coarse_lo, fine_lo) and np.allclose(coarse_hi, fine_hi)

    def test_tiles_cover_the_grid_with_overlap(self):
        spec = GridSpan((0.0, 0.0, 0.0), 0.1, (16, 4, 4))
        tiles = spec.make_tiles(max_cell_num=200, overlap_cell_num=1)
        assert len(tiles) > 1
        covered = np.zeros(spec.cell_num, dtype=bool)
        for start, tile in tiles:
            assert tile.get_cell_num() <= 200
            stop = tuple(start[side] + tile.cell_num[side] for side in range(3))
            covered[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = True
        assert covered.all()

    def test_overlapping_specs_agree_on_shared_centers(self):
        first = GridSpan((0.0, 0.0, 0.0), 0.2, (10, 10, 10))
        second = GridSpan((0.6, 0.2, 0.4), 0.2, (10, 10, 10))
        ax_first = first.make_axes()[0]
        ax_second = second.make_axes()[0]
        assert np.allclose(ax_first[3:], ax_second[: len(ax_first) - 3])
