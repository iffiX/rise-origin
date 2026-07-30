# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the fracture phase."""

import numpy as np
import torch

from rise_origin.create.algorithm import (
    ChunkFracture,
    GridSpan,
    PlaneFracture,
    SeamFracture,
    UniformGenerator,
    Volume,
)
from rise_origin.create.protocol.rng import Rng


def make_filled_volume(cell_num=(16, 16, 16), cell_size=0.05):
    origin = tuple(-count * cell_size / 2.0 for count in cell_num)
    grid = GridSpan(origin, cell_size, cell_num)
    density = torch.ones(cell_num)
    return Volume.make_from_density(grid, density, 1, "cpu")


class TestFractureFragment:
    def test_center_is_inside_and_far_is_outside(self):
        fragment = PlaneFracture.make_fragment(Rng(3), 0.5, 0.0, 0.0)
        center = fragment.get_distance(0.0, 0.0, 0.0, np)
        far = fragment.get_distance(10.0, 0.0, 0.0, np)
        assert float(center) < 0.0
        assert float(far) > 0.0

    def test_prism_fragment_is_a_bounded_column(self):
        fragment = PlaneFracture.make_prism_fragment(Rng(3), radius=0.3, height=0.8)
        assert float(fragment.get_distance(0.0, 0.0, 0.0, np)) < 0.0
        assert float(fragment.get_distance(0.0, 0.0, 0.2, np)) < 0.0
        assert float(fragment.get_distance(0.5, 0.0, 0.0, np)) > 0.0
        assert float(fragment.get_distance(0.0, 0.0, 0.6, np)) > 0.0

    def test_elongation_stretches_the_x_reach(self):
        round_fragment = PlaneFracture.make_fragment(Rng(3), 0.5, 0.0, 0.0)
        long_fragment = PlaneFracture.make_fragment(Rng(3), 0.5, 0.0, 1.0)
        probe = 0.31
        assert float(round_fragment.get_distance(probe, 0.0, 0.0, np)) > 0.0
        assert float(long_fragment.get_distance(probe, 0.0, 0.0, np)) < float(
            round_fragment.get_distance(probe, 0.0, 0.0, np)
        )


class TestPlaneFracture:
    def test_fracture_clears_outside_and_segments_inside(self):
        volume = make_filled_volume()
        fragment = PlaneFracture.make_fragment(Rng(5), 0.4, 0.0, 0.0)
        PlaneFracture((fragment,), np.zeros((1, 3))).fracture(volume)
        solid = volume.get_solid_cells()
        assert bool(solid[8, 8, 8])
        assert not bool(solid[0, 0, 0])
        assert int(volume.segment_id[8, 8, 8]) == 0
        assert int(volume.segment_id[0, 0, 0]) == -1

    def test_corner_radius_fillets_the_junctions_inward(self):
        sharp = make_filled_volume(cell_num=(32, 32, 32))
        rounded = make_filled_volume(cell_num=(32, 32, 32))
        sharp_fragment = PlaneFracture.make_fragment(
            Rng(5), 1.0, 0.0, 0.0, corner_radius_share=0.0
        )
        rounded_fragment = PlaneFracture.make_fragment(
            Rng(5), 1.0, 0.0, 0.0, corner_radius_share=0.12
        )
        PlaneFracture((sharp_fragment,), np.zeros((1, 3))).fracture(sharp)
        PlaneFracture((rounded_fragment,), np.zeros((1, 3))).fracture(rounded)
        sharp_solid = sharp.get_solid_cells()
        rounded_solid = rounded.get_solid_cells()
        assert int(rounded_solid.sum()) < int(sharp_solid.sum())
        assert bool((~rounded_solid | sharp_solid).all())
        assert bool(rounded_solid[16, 16, 16])

    def test_surface_roughness_wanders_the_boundary(self):
        flat_volume = make_filled_volume(cell_num=(32, 32, 32))
        rough_volume = make_filled_volume(cell_num=(32, 32, 32))
        fragment = PlaneFracture.make_fragment(Rng(5), 1.0, 0.0, 0.0)
        PlaneFracture((fragment,), np.zeros((1, 3))).fracture(flat_volume)
        PlaneFracture(
            (fragment,),
            np.zeros((1, 3)),
            rng=Rng(9),
            surface_roughness=0.08,
            roughness_feature_size=0.3,
        ).fracture(rough_volume)
        flat = flat_volume.get_solid_cells()
        rough = rough_volume.get_solid_cells()
        carved = int((flat & ~rough).sum())
        bulged = int((~flat & rough).sum())
        assert carved > 0
        assert bulged > 0


class TestSeamFracture:
    def test_seaming_keeps_every_solid_cell(self):
        volume = make_filled_volume()
        before = int(volume.get_solid_cells().sum())
        SeamFracture(Rng(1), (0.6, 0.6, 0.6), 0.3).fracture(volume)
        assert int(volume.get_solid_cells().sum()) == before
        assert int(volume.segment_id.max()) > 0

    def test_touching_pieces_carry_seam_marks(self):
        volume = make_filled_volume()
        SeamFracture(Rng(1), (0.6, 0.6, 0.6), 0.3, seam_width=0.1).fracture(volume)
        seam = (volume.structure_mark & Volume.SEAM_MARK) != 0
        assert bool(seam.any())

    def test_a_seam_finer_than_a_cell_marks_nothing(self):
        volume = make_filled_volume()
        SeamFracture(Rng(1), (0.6, 0.6, 0.6), 0.3, seam_width=0.02).fracture(volume)
        assert volume.structure_mark is None


class TestChunkFracture:
    def _make_block(self):
        volume = make_filled_volume(cell_num=(32, 32, 32))
        fragment = PlaneFracture.make_fragment(Rng(5), 1.2, 0.0, 0.0)
        PlaneFracture((fragment,), np.zeros((1, 3))).fracture(volume)
        return volume

    def test_chunks_knock_material_off_and_keep_the_core(self):
        volume = self._make_block()
        before = int(volume.get_solid_cells().sum())
        ChunkFracture(Rng(11), chunk_num=4, chunk_radius=0.3).fracture(volume)
        assert int(volume.get_solid_cells().sum()) < before
        assert bool(volume.get_solid_cells()[16, 16, 16])

    def test_chunks_break_convexity(self):
        volume = self._make_block()
        ChunkFracture(Rng(11), chunk_num=4, chunk_radius=0.3).fracture(volume)
        solid = volume.get_solid_cells()
        concave = False
        for axis in range(3):
            moved = torch.movedim(solid, axis, 0)
            column_solid = moved.any(dim=0)
            first = moved.to(torch.int32).argmax(dim=0)
            last = (
                moved.shape[0]
                - 1
                - torch.flip(moved, dims=[0]).to(torch.int32).argmax(dim=0)
            )
            span = (last - first + 1).clamp(min=0)
            filled = moved.to(torch.int32).sum(dim=0)
            if bool((column_solid & (filled < span)).any()):
                concave = True
        assert concave
