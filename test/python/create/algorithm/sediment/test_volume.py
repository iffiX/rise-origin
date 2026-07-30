# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the sediment volume."""

import numpy as np
import pytest
import torch

from rise_origin.create.algorithm import GridSpan, Volume


def make_grid(cell_num=(8, 8, 8), cell_size=0.1):
    return GridSpan((0.0, 0.0, 0.0), cell_size, cell_num)


class TestVolumeBirths:
    def test_density_birth_fills_positive_cells(self):
        grid = make_grid()
        density = torch.full(grid.cell_num, -1.0)
        density[2:6, 2:6, 2:6] = 1.0
        volume = Volume.make_from_density(grid, density, 2, "cpu")
        assert int(volume.get_solid_cells().sum()) == 64
        assert int(volume.material_id[3, 3, 3]) == 2
        assert int(volume.material_id[0, 0, 0]) == 0

    def test_height_birth_fills_below_the_surface(self):
        grid = make_grid()
        heights = np.full((8, 8), 0.35)
        volume = Volume.make_from_height(grid, heights, 1, "cpu")
        assert bool(volume.get_solid_cells()[4, 4, 2])
        assert not bool(volume.get_solid_cells()[4, 4, 4])

    def test_gather_reads_rows_through_materials(self):
        grid = make_grid()
        density = torch.ones(grid.cell_num)
        volume = Volume.make_from_density(grid, density, 1, "cpu")
        volume.material_id[4:, :, :] = 2
        rows = volume.gather((0.0, 0.5, 2.0))
        assert float(rows[0, 0, 0]) == pytest.approx(0.5)
        assert float(rows[7, 0, 0]) == pytest.approx(2.0)


class TestVolumeCoarsening:
    def test_ratio_must_divide_the_counts(self):
        grid = make_grid((6, 6, 6))
        volume = Volume(grid, "cpu")
        with pytest.raises(ValueError):
            volume.make_coarser(4)

    def test_solid_majority_decides_occupancy(self):
        grid = make_grid((4, 4, 4))
        volume = Volume(grid, "cpu")
        volume.material_id[:2, :2, :2] = 1
        volume.material_id[0, 0, 2] = 1
        coarse = volume.make_coarser(2)
        assert coarse.grid.cell_size == pytest.approx(0.2)
        assert bool(coarse.get_solid_cells()[0, 0, 0])
        assert not bool(coarse.get_solid_cells()[0, 0, 1])

    def test_material_majority_wins_the_block(self):
        grid = make_grid((2, 2, 2))
        volume = Volume(grid, "cpu")
        volume.material_id[:, :, :] = 1
        volume.material_id[0, 0, 0] = 2
        coarse = volume.make_coarser(2)
        assert int(coarse.material_id[0, 0, 0]) == 1

    def test_segment_majority_survives_an_air_center(self):
        grid = make_grid((2, 2, 2))
        volume = Volume(grid, "cpu")
        volume.material_id[:, :, :] = 1
        volume.material_id[1, 1, 1] = 0
        volume.segment_id = torch.full(grid.cell_num, 7, dtype=torch.int32)
        volume.segment_id[1, 1, 1] = -1
        volume.segment_id[0, 0, 0] = 3
        coarse = volume.make_coarser(2)
        assert int(coarse.segment_id[0, 0, 0]) == 7

    def test_optical_mid_majority_follows_the_winning_material(self):
        grid = make_grid((2, 2, 2))
        volume = Volume(grid, "cpu")
        volume.material_id[:, :, :] = 1
        volume.material_id[0, :, :] = 2
        volume.optical_mid = torch.zeros(grid.cell_num, dtype=torch.uint8)
        volume.optical_mid[1, :, :] = 5
        coarse = volume.make_coarser(2)
        assert int(coarse.material_id[0, 0, 0]) == 1
        assert int(coarse.optical_mid[0, 0, 0]) == 5
