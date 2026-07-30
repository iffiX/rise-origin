# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the erosion phase, named after the sources they pin."""

import math

import cc3d
import numpy as np
import torch

from rise_origin.create.algorithm import (
    BallErosion,
    DetailErosion,
    GridSpan,
    Volume,
)
from rise_origin.create.protocol.rng import Rng


def make_cube_volume(cell_num=(24, 24, 24), cell_size=0.05, half_num=8):
    origin = tuple(-count * cell_size / 2.0 for count in cell_num)
    grid = GridSpan(origin, cell_size, cell_num)
    volume = Volume(grid, "cpu")
    center = tuple(count // 2 for count in cell_num)
    window = tuple(
        slice(center[side] - half_num, center[side] + half_num) for side in range(3)
    )
    volume.material_id[window] = 1
    return volume, window


class TestBallErosion:
    def test_jones_cube_corners_round_before_faces(self):
        volume, window = make_cube_volume()
        erosion = BallErosion(
            Rng(3),
            step_num=6,
            ball_radius=0.2,
            wear_pace=0.4,
            flat_weight=0.0,
            round_weight=2.0,
            pit_weight=0.0,
            hardness_rows=(1.0, 1.0),
            durability_sway_ratio=0.0,
        )
        erosion.erode(volume)
        solid = volume.get_solid_cells()
        corner = (window[0].start, window[1].start, window[2].start)
        assert not bool(solid[corner])
        assert bool(solid[12, 12, window[2].start])

    def test_bloore_more_steps_wear_more(self):
        counts = []
        for step_num in (2, 6):
            volume, _window = make_cube_volume()
            BallErosion(
                Rng(3),
                step_num=step_num,
                ball_radius=0.2,
                wear_pace=0.3,
                flat_weight=1.0,
                round_weight=1.2,
                pit_weight=0.0,
                hardness_rows=(1.0, 1.0),
                durability_sway_ratio=0.0,
            ).erode(volume)
            counts.append(int(volume.get_solid_cells().sum()))
        assert counts[1] < counts[0]

    def test_seams_open_deeper_than_faces_wear(self):
        volume, window = make_cube_volume(cell_num=(40, 40, 40), half_num=14)
        volume.segment_id = torch.where(
            volume.get_solid_cells(),
            torch.zeros(volume.grid.cell_num, dtype=torch.int32),
            torch.full(volume.grid.cell_num, -1, dtype=torch.int32),
        )
        volume.structure_mark = torch.zeros(volume.grid.cell_num, dtype=torch.uint8)
        volume.structure_mark[20, window[1], window[2]] = Volume.SEAM_MARK
        BallErosion(
            Rng(5),
            step_num=20,
            ball_radius=0.15,
            wear_pace=0.4,
            flat_weight=0.3,
            round_weight=1.0,
            pit_weight=0.0,
            hardness_rows=(1.0, 0.5),
            durability_sway_ratio=0.0,
            seam_rate_scale=8.0,
        ).erode(volume)
        solid = volume.get_solid_cells()
        seam_top = int(solid[20, 20].nonzero().max())
        face_top = int(solid[12, 20].nonzero().max())
        assert seam_top < face_top

    def test_contacts_hold_against_wear(self):
        volume, window = make_cube_volume()
        volume.structure_mark = torch.zeros(volume.grid.cell_num, dtype=torch.uint8)
        top = window[2].stop - 1
        volume.structure_mark[10:14, 10:14, top] = Volume.CONTACT_MARK
        BallErosion(
            Rng(5),
            step_num=6,
            ball_radius=0.1,
            wear_pace=0.5,
            flat_weight=1.0,
            round_weight=0.0,
            pit_weight=0.0,
            hardness_rows=(1.0, 1.0),
            durability_sway_ratio=0.0,
        ).erode(volume)
        solid = volume.get_solid_cells()
        assert bool(solid[10:14, 10:14, top].all())
        assert not bool(solid[6, 6, top])

    def test_durability_rows_slow_the_hard_material(self):
        volume, window = make_cube_volume()
        volume.material_id[window[0], window[1], 12:] = torch.where(
            volume.material_id[window[0], window[1], 12:] > 0,
            torch.tensor(2, dtype=torch.uint8),
            torch.tensor(0, dtype=torch.uint8),
        )
        soft_before = int((volume.material_id == 1).sum())
        hard_before = int((volume.material_id == 2).sum())
        BallErosion(
            Rng(7),
            step_num=4,
            ball_radius=0.15,
            wear_pace=0.3,
            flat_weight=1.0,
            round_weight=1.0,
            pit_weight=0.0,
            hardness_rows=(1.0, 0.4, 2.0),
            durability_sway_ratio=0.0,
        ).erode(volume)
        soft_lost = soft_before - int((volume.material_id == 1).sum())
        hard_lost = hard_before - int((volume.material_id == 2).sum())
        assert soft_lost > hard_lost


class TestAirAccessGate:
    def test_a_seamed_cube_never_splits(self):
        volume, window = make_cube_volume(cell_num=(40, 40, 40), half_num=14)
        volume.segment_id = torch.where(
            volume.get_solid_cells(),
            torch.zeros(volume.grid.cell_num, dtype=torch.int32),
            torch.full(volume.grid.cell_num, -1, dtype=torch.int32),
        )
        volume.structure_mark = torch.zeros(volume.grid.cell_num, dtype=torch.uint8)
        volume.structure_mark[20, window[1], window[2]] = Volume.SEAM_MARK
        BallErosion(
            Rng(11),
            step_num=40,
            ball_radius=0.15,
            wear_pace=0.4,
            flat_weight=0.1,
            round_weight=1.2,
            pit_weight=0.5,
            hardness_rows=(1.0, 0.5),
            durability_sway_ratio=0.0,
            seam_rate_scale=6.0,
        ).erode(volume)
        solid = volume.get_solid_cells().numpy()
        sizes = np.bincount(cc3d.connected_components(solid, connectivity=6).ravel())
        sizes[0] = 0
        assert int(sizes.max()) > 0.99 * int(solid.sum())

    def test_a_body_filling_its_grid_keeps_air_access(self):
        volume, _window = make_cube_volume(cell_num=(40, 40, 40), half_num=18)
        before = int(volume.get_solid_cells().sum())
        BallErosion(
            Rng(17),
            step_num=8,
            ball_radius=0.7,
            wear_pace=0.5,
            flat_weight=1.0,
            round_weight=1.2,
            pit_weight=0.0,
            hardness_rows=(1.0, 0.5),
            durability_sway_ratio=0.0,
        ).erode(volume)
        assert int(volume.get_solid_cells().sum()) < before

    def test_a_sealed_pocket_never_erodes(self):
        volume, _window = make_cube_volume(cell_num=(40, 40, 40), half_num=16)
        pocket = (slice(18, 22), slice(18, 22), slice(18, 22))
        volume.material_id[pocket] = 0
        before = int((volume.material_id[pocket] == 0).sum())
        BallErosion(
            Rng(13),
            step_num=20,
            ball_radius=0.15,
            wear_pace=0.5,
            flat_weight=1.0,
            round_weight=1.2,
            pit_weight=2.0,
            hardness_rows=(1.0, 0.5),
            durability_sway_ratio=0.0,
        ).erode(volume)
        assert int((volume.material_id[pocket] == 0).sum()) == before


class TestPooledMeasurement:
    def test_pooled_air_share_tracks_the_exact_one(self):
        volume, _window = make_cube_volume()
        air = (~volume.get_solid_cells()).to(torch.float32)
        erosion = BallErosion(
            Rng(3),
            step_num=1,
            ball_radius=0.4,
            wear_pace=0.1,
            flat_weight=0.0,
            round_weight=2.0,
            pit_weight=0.0,
            hardness_rows=(1.0, 1.0),
        )
        exact_ball = erosion._make_ball(volume, 8, 1)
        pooled_ball = erosion._make_ball(volume, 8, 2)
        exact = BallErosion._make_air_share(air, exact_ball, 1)
        pooled = BallErosion._make_air_share(air, pooled_ball, 2)
        assert float((exact - pooled).abs().mean()) < 0.03


class TestDetailErosion:
    def test_sub_cell_depth_skips_cleanly(self):
        volume, _window = make_cube_volume()
        before = int(volume.get_solid_cells().sum())
        DetailErosion(Rng(3), depth=0.01, dent_spacing=0.2, dent_width=0.12).erode(
            volume
        )
        assert int(volume.get_solid_cells().sum()) == before

    def test_dents_pit_the_surface(self):
        volume, _window = make_cube_volume()
        before = int(volume.get_solid_cells().sum())
        DetailErosion(
            Rng(3), depth=0.05, dent_spacing=0.2, dent_width=0.12, dent_share=1.0
        ).erode(volume)
        after = int(volume.get_solid_cells().sum())
        assert after < before

    def test_dents_stay_within_depth(self):
        volume, _window = make_cube_volume()
        depth = 0.1
        depth_num = int(math.ceil(depth / volume.grid.cell_size))
        DetailErosion(
            Rng(3), depth=depth, dent_spacing=0.2, dent_width=0.12, dent_share=1.0
        ).erode(volume)
        center = tuple(count // 2 for count in volume.grid.cell_num)
        margin = 8 - depth_num - 1
        core = tuple(
            slice(center[side] - margin, center[side] + margin) for side in range(3)
        )
        assert bool(volume.get_solid_cells()[core].all())

    def test_dents_never_hole_a_thin_wall(self):
        volume, _window = make_cube_volume()
        volume.material_id[:] = 0
        wall = (slice(2, 22), slice(11, 13), slice(2, 22))
        volume.material_id[wall] = 1
        DetailErosion(
            Rng(3), depth=0.05, dent_spacing=0.2, dent_width=0.12, dent_share=1.0
        ).erode(volume)
        assert bool(volume.get_solid_cells()[wall].all())
