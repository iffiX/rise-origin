# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the field library."""

import pickle

import numpy as np

from rise_origin.create.algorithm import (
    CellNoise,
    FBMNoiseField,
    GradientNoiseField,
    GridSpan,
    RidgedNoiseField,
    TurbulenceNoiseField,
    ValueNoiseField,
    dtype,
)
from rise_origin.create.protocol.rng import Rng


def make_points(seed=0, num=2048, reach=40.0):
    generator = np.random.default_rng(seed)
    return generator.uniform(-reach, reach, size=(num, 3))


class TestValueNoiseField:
    def test_values_pin_the_reference_math(self):
        field = ValueNoiseField(Rng(7).split("moss"), 0.5)
        assert field.salt == 1268042725
        points = np.array([[0.0, 0.0, 0.0], [0.13, -0.4, 2.7], [-3.3, 1.9, -0.05]])
        expected = np.array([-0.27326215, 0.2796553, 0.09959189])
        assert np.allclose(field(points), expected, atol=1e-8)

    def test_values_stay_inside_the_range(self):
        values = ValueNoiseField(Rng(3), 0.37)(make_points())
        assert values.min() >= -1.0 and values.max() <= 1.0

    def test_same_stream_same_field(self):
        points = make_points()
        first = ValueNoiseField(Rng(11).split("peel"), 0.21)(points)
        second = ValueNoiseField(Rng(11).split("peel"), 0.21)(points)
        assert np.array_equal(first, second)

    def test_a_shipped_salt_rebuilds_the_exact_field(self):
        points = make_points()
        field = ValueNoiseField(Rng(11).split("peel"), 0.21)
        rebuilt = ValueNoiseField(field.salt, 0.21)
        assert np.array_equal(field(points), rebuilt(points))

    def test_a_field_travels_by_pickle(self):
        points = make_points()
        field = ValueNoiseField(Rng(5), 0.8)
        copied = pickle.loads(pickle.dumps(field))
        assert np.array_equal(field(points), copied(points))

    def test_the_dtype_dial_sets_the_answer_dtype(self):
        points = make_points()
        assert ValueNoiseField(Rng(5), 0.8)(points).dtype == dtype.float64
        values = ValueNoiseField(Rng(5), 0.8, dtype=dtype.float32)(points)
        assert values.dtype == dtype.float32
        assert values.min() >= -1.0 and values.max() <= 1.0

    def test_a_grid_samples_like_its_own_cell_centers(self):
        spec = GridSpan((-1.0, 0.5, 2.0), 0.3, (5, 4, 3))
        field = ValueNoiseField(Rng(9), 0.7)
        ax, ay, az = spec.make_axes()
        gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
        points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        assert np.array_equal(field(spec), field(points).reshape(spec.cell_num))


class TestGradientNoiseField:
    def test_values_stay_inside_the_range(self):
        values = GradientNoiseField(Rng(3), 0.4)(make_points())
        assert values.min() >= -1.0 and values.max() <= 1.0

    def test_same_stream_same_field(self):
        points = make_points()
        first = GradientNoiseField(Rng(11).split("a"), 0.21)(points)
        second = GradientNoiseField(Rng(11).split("a"), 0.21)(points)
        assert np.array_equal(first, second)

    def test_a_field_travels_by_pickle(self):
        points = make_points()
        field = GradientNoiseField(Rng(5), 0.8)
        copied = pickle.loads(pickle.dumps(field))
        assert np.array_equal(field(points), copied(points))

    def test_second_differences_stay_smooth_at_lattice_lines(self):
        field = GradientNoiseField(Rng(7), 1.0, dtype=dtype.float64)
        step = 1.0e-3
        offsets = np.arange(-4, 5) * step
        points = np.zeros((9, 3))
        points[:, 0] = 1.0 + offsets
        points[:, 1] = 0.3
        points[:, 2] = 0.7
        values = field(points)
        second = values[2:] - 2.0 * values[1:-1] + values[:-2]
        assert np.abs(np.diff(second)).max() < 1.0e-4

    def test_a_grid_samples_like_its_own_cell_centers(self):
        spec = GridSpan((0.2, -0.4, 1.0), 0.15, (4, 5, 6))
        field = GradientNoiseField(Rng(2), 0.5)
        ax, ay, az = spec.make_axes()
        gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
        points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        assert np.array_equal(field(spec), field(points).reshape(spec.cell_num))


class TestLayeredNoiseFields:
    def test_fbm_values_stay_inside_the_range(self):
        values = FBMNoiseField(Rng(3), 0.9)(make_points())
        assert values.min() >= -1.0 and values.max() <= 1.0

    def test_ridged_and_turbulence_stay_inside_zero_one(self):
        points = make_points()
        ridged = RidgedNoiseField(Rng(4), 0.9)(points)
        folded = TurbulenceNoiseField(Rng(4), 0.9)(points)
        assert ridged.min() >= 0.0 and ridged.max() <= 1.0
        assert folded.min() >= 0.0 and folded.max() <= 1.0

    def test_same_stream_same_field(self):
        points = make_points()
        first = TurbulenceNoiseField(Rng(6).split("layer"), 0.6)(points)
        second = TurbulenceNoiseField(Rng(6).split("layer"), 0.6)(points)
        assert np.array_equal(first, second)

    def test_a_field_travels_by_pickle(self):
        points = make_points()
        field = RidgedNoiseField(Rng(5), 0.8)
        copied = pickle.loads(pickle.dumps(field))
        assert np.array_equal(field(points), copied(points))


class TestCellNoise:
    def test_distances_order_and_owner_stability(self):
        points = make_points()
        sample = CellNoise(Rng(3), 0.5)(points)
        assert (sample.nearest_distance <= sample.second_distance).all()
        assert sample.nearest_distance.min() >= 0.0
        again = CellNoise(Rng(3), 0.5)(points)
        assert np.array_equal(sample.cell_id, again.cell_id)

    def test_neighbors_share_their_owner_cell(self):
        base = np.array([[0.31, 0.42, 0.53]])
        near = base + 1.0e-4
        first = CellNoise(Rng(8), 1.0)(base)
        second = CellNoise(Rng(8), 1.0)(near)
        assert first.cell_id[0] == second.cell_id[0]

    def test_a_grid_samples_like_its_own_cell_centers(self):
        spec = GridSpan((0.0, 0.0, 0.0), 0.2, (4, 4, 4))
        noise = CellNoise(Rng(2), 0.5)
        ax, ay, az = spec.make_axes()
        gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
        points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        on_grid = noise(spec)
        on_points = noise(points)
        assert np.array_equal(
            on_grid.nearest_distance, on_points.nearest_distance.reshape(spec.cell_num)
        )
        assert np.array_equal(on_grid.cell_id, on_points.cell_id.reshape(spec.cell_num))
