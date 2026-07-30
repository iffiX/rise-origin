# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the generator phase."""

import math

import pytest
import torch

from rise_origin.create.algorithm import (
    GrainGenerator,
    GridSpan,
    StrataGenerator,
    StrataGeneratorLayer,
    UniformGenerator,
    VeinGenerator,
)
from rise_origin.create.protocol.rng import Rng


def make_grid(cell_num=(16, 16, 16), cell_size=0.05):
    origin = tuple(-count * cell_size / 2.0 for count in cell_num)
    return GridSpan(origin, cell_size, cell_num)


class TestUniformGenerator:
    def test_box_shape_fills_the_centered_box(self):
        grid = make_grid()
        generator = UniformGenerator(
            grid,
            "cpu",
            Rng(7),
            material_id=1,
            shape="box",
            box_size=(0.4, 0.4, 0.4),
        )
        volume = generator.generate()
        assert int(volume.get_solid_cells().sum()) == 8**3
        assert bool(volume.get_solid_cells()[8, 8, 8])
        assert not bool(volume.get_solid_cells()[0, 0, 0])

    def test_unknown_shape_is_refused(self):
        with pytest.raises(ValueError):
            UniformGenerator(make_grid(), "cpu", Rng(7), 1, "blob")

    def test_ridge_shape_stays_inside_its_height_band(self):
        grid = GridSpan((0.0, 0.0, 0.0), 0.5, (24, 24, 16))
        generator = UniformGenerator(
            grid,
            "cpu",
            Rng(11),
            material_id=1,
            shape="ridge",
            base_height=1.0,
            ridge_height=5.0,
            ridge_feature_size=6.0,
            warp_size=1.0,
            warp_feature_size=4.0,
        )
        heights = generator._make_ridge_heights()
        assert float(heights.min()) >= 1.0
        assert float(heights.max()) <= 6.0
        assert bool(generator.generate().get_solid_cells()[:, :, 0].all())


class TestStrataGenerator:
    def test_flat_layers_stack_bottom_up_in_meters(self):
        grid = make_grid((8, 8, 16), 0.05)
        layers = (
            StrataGeneratorLayer(material_id=1, thickness=0.3),
            StrataGeneratorLayer(material_id=2, thickness=0.2),
            StrataGeneratorLayer(material_id=1, thickness=0.3),
        )
        volume = StrataGenerator(
            grid, "cpu", Rng(3), layers, shape="box", box_size=(0.4, 0.4, 0.8)
        ).generate()
        column = volume.material_id[4, 4, :]
        heights = grid.make_axes()[2]
        for index in range(16):
            if not bool(volume.get_solid_cells()[4, 4, index]):
                continue
            height = heights[index] - grid.origin[2]
            wanted = 1 if height < 0.3 or height >= 0.5 else 2
            assert int(column[index]) == wanted

    def test_fold_bends_the_band_boundary(self):
        grid = make_grid((16, 8, 16), 0.05)
        layers = (
            StrataGeneratorLayer(material_id=1, thickness=0.4),
            StrataGeneratorLayer(material_id=2, thickness=0.4),
        )
        flat = StrataGenerator(
            grid, "cpu", Rng(5), layers, shape="box", box_size=(0.8, 0.4, 0.8)
        ).generate()
        folded = StrataGenerator(
            grid,
            "cpu",
            Rng(5),
            layers,
            shape="box",
            box_size=(0.8, 0.4, 0.8),
            fold_size=0.2,
            fold_feature_size=0.5,
        ).generate()
        assert not torch.equal(flat.material_id, folded.material_id)

    def test_tone_bias_places_the_layer_in_its_run(self):
        grid = make_grid((8, 8, 12), 0.05)
        light = StrataGeneratorLayer(
            material_id=1,
            thickness=1.0,
            optical_mids=tuple(range(8)),
            tone_bias=0.0,
            tone_spread=0.0,
        )
        dark = StrataGeneratorLayer(
            material_id=1,
            thickness=1.0,
            optical_mids=tuple(range(8)),
            tone_bias=1.0,
            tone_spread=0.0,
        )
        light_vol = StrataGenerator(
            grid, "cpu", Rng(3), (light,), shape="box", box_size=(0.4, 0.4, 0.6)
        ).generate()
        dark_vol = StrataGenerator(
            grid, "cpu", Rng(3), (dark,), shape="box", box_size=(0.4, 0.4, 0.6)
        ).generate()
        solid = light_vol.get_solid_cells()
        assert int(light_vol.optical_mid[solid].float().mean()) == 0
        assert int(dark_vol.optical_mid[solid].max()) == 7

    def test_lamination_angle_tilts_the_laminae(self):
        grid = make_grid((20, 8, 20), 0.04)
        run = tuple(range(6))
        flat = StrataGeneratorLayer(
            material_id=1,
            thickness=2.0,
            optical_mids=run,
            tone_spread=0.0,
            lamination_feature_size=0.16,
            lamination_strength=0.5,
        )
        tilted = StrataGeneratorLayer(
            material_id=1,
            thickness=2.0,
            optical_mids=run,
            tone_spread=0.0,
            lamination_feature_size=0.16,
            lamination_strength=0.5,
            lamination_angle=0.5,
        )
        flat_vol = StrataGenerator(
            grid, "cpu", Rng(4), (flat,), shape="box", box_size=(0.8, 0.3, 0.8)
        ).generate()
        tilted_vol = StrataGenerator(
            grid, "cpu", Rng(4), (tilted,), shape="box", box_size=(0.8, 0.3, 0.8)
        ).generate()
        assert not torch.equal(flat_vol.optical_mid, tilted_vol.optical_mid)

    def test_layer_thicknesses_sum_and_stay_above_the_floor(self):
        thicknesses = StrataGenerator.make_layer_thicknesses(
            Rng(9), total_thickness=1.0, mean_thickness=0.1, min_thickness=0.02
        )
        assert abs(sum(thicknesses) - 1.0) < 1.0e-6
        assert min(thicknesses) >= 0.02 - 1.0e-9
        assert len(thicknesses) >= 3


class TestGrainGenerator:
    def test_crystals_take_mids_from_the_groups(self):
        grid = make_grid((16, 16, 16), 0.05)
        volume = GrainGenerator(
            grid,
            "cpu",
            Rng(5),
            material_id=1,
            grain_mids=((0,), (0,), (2,), (3, 4)),
            grain_feature_size=0.1,
            cluster_feature_size=0.3,
            cluster_strength=0.5,
            dither_feature_size=0.05,
            shape="box",
            box_size=(0.6, 0.6, 0.6),
        ).generate()
        solid = volume.get_solid_cells()
        picks = set(volume.optical_mid[solid].tolist())
        assert picks <= {0, 2, 3, 4}
        assert len(picks) > 1

    def test_dither_varies_shades_within_a_group(self):
        grid = make_grid((16, 16, 16), 0.05)
        volume = GrainGenerator(
            grid,
            "cpu",
            Rng(5),
            material_id=1,
            grain_mids=((0, 1, 2),),
            grain_feature_size=0.1,
            cluster_feature_size=0.3,
            cluster_strength=0.0,
            dither_feature_size=0.06,
            shape="box",
            box_size=(0.6, 0.6, 0.6),
        ).generate()
        picks = set(volume.optical_mid[volume.get_solid_cells()].tolist())
        assert len(picks & {0, 1, 2}) > 1

    def test_clustering_groups_same_mid_neighbors(self):
        grid = make_grid((24, 24, 8), 0.05)
        kwargs = dict(
            material_id=1,
            grain_mids=tuple((mid,) for mid in range(8)),
            grain_feature_size=0.08,
            dither_feature_size=0.05,
            shape="box",
            box_size=(1.0, 1.0, 0.3),
        )
        scattered = GrainGenerator(
            grid,
            "cpu",
            Rng(6),
            cluster_feature_size=0.3,
            cluster_strength=0.0,
            **kwargs,
        ).generate()
        clustered = GrainGenerator(
            grid,
            "cpu",
            Rng(6),
            cluster_feature_size=0.4,
            cluster_strength=0.9,
            **kwargs,
        ).generate()
        # Clustering raises the share of face neighbors sharing a mid.
        assert _same_neighbor_share(clustered) > _same_neighbor_share(scattered)


class TestVeinGenerator:
    def test_body_and_veins_both_appear(self):
        grid = make_grid((16, 16, 16), 0.05)
        volume = VeinGenerator(
            grid,
            "cpu",
            Rng(7),
            material_id=1,
            body_mid=1,
            vein_mids=(4, 5),
            vein_feature_size=0.15,
            vein_share=0.3,
            shape="box",
            box_size=(0.6, 0.6, 0.6),
        ).generate()
        solid = volume.get_solid_cells()
        picks = set(volume.optical_mid[solid].tolist())
        assert picks & {1}
        assert picks & {4, 5}

    def test_one_stone_carries_one_body_tone(self):
        grid = make_grid((16, 16, 16), 0.05)
        volume = VeinGenerator(
            grid,
            "cpu",
            Rng(7),
            material_id=1,
            body_mid=2,
            vein_mids=(4, 5),
            vein_feature_size=0.15,
            vein_share=0.0,
            shape="box",
            box_size=(0.6, 0.6, 0.6),
        ).generate()
        solid = volume.get_solid_cells()
        assert set(volume.optical_mid[solid].tolist()) == {2}


def _same_neighbor_share(volume):
    mid = volume.optical_mid
    solid = volume.get_solid_cells()
    same = 0
    total = 0
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        pair = solid[tuple(lo)] & solid[tuple(hi)]
        total += int(pair.sum())
        same += int((pair & (mid[tuple(lo)] == mid[tuple(hi)])).sum())
    return same / max(total, 1)
