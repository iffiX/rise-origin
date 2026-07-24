# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the snapshot voxelizer: depth-shell layering and a stable build."""

import numpy as np

from rise_origin import RS_NULL_INDEX
from rise_origin.create.algorithm import (
    CloudSnapshot,
    CloudVoxelizer,
    GridSpan,
)
from rise_origin.create.protocol.worker_pool import WorkerPool

ISO = 0.1


def make_snapshot():
    counts = (24, 24, 24)
    step = 8.0
    lines = [
        (np.arange(counts[side]) + 0.5) * step - 0.5 * counts[side] * step
        for side in range(3)
    ]
    x = lines[0][:, None, None]
    y = lines[1][None, :, None]
    z = lines[2][None, None, :]
    bump = np.exp(-(x**2 + y**2 + z**2) / (2.0 * 40.0**2)).astype(np.float32)
    grid = GridSpan(
        origin=(
            -0.5 * counts[0] * step,
            -0.5 * counts[1] * step,
            -0.5 * counts[2] * step,
        ),
        cell_size=step,
        cell_num=counts,
    )
    return CloudSnapshot(density=bump, grid=grid)


def make_out_grid():
    return GridSpan(origin=(-96.0, -96.0, -96.0), cell_size=4.0, cell_num=(48, 48, 48))


def bodies_equal(one, two):
    return len(one) == len(two) and all(
        np.array_equal(a.material_grid, b.material_grid) and a.origin == b.origin
        for a, b in zip(one, two)
    )


class TestCloudVoxelizer:
    def test_the_shells_band_from_skin_to_core(self):
        bodies = CloudVoxelizer().make_bodies(
            make_snapshot(), make_out_grid(), iso=ISO, layer_count=3
        )
        assert bodies
        material = bodies[0].material_grid
        present = material != RS_NULL_INDEX
        assert present.any()
        sids = set(np.unique(material[present]).tolist())
        assert sids == {0, 1, 2}

    def test_a_single_layer_is_all_skin(self):
        bodies = CloudVoxelizer().make_bodies(
            make_snapshot(), make_out_grid(), iso=ISO, layer_count=1
        )
        material = bodies[0].material_grid
        present = material != RS_NULL_INDEX
        assert set(np.unique(material[present]).tolist()) == {0}

    def test_same_input_same_bodies(self):
        one = CloudVoxelizer().make_bodies(
            make_snapshot(), make_out_grid(), iso=ISO, layer_count=3
        )
        two = CloudVoxelizer().make_bodies(
            make_snapshot(), make_out_grid(), iso=ISO, layer_count=3
        )
        assert bodies_equal(one, two)

    def test_pooled_build_matches_serial(self):
        snapshot = make_snapshot()
        grid = make_out_grid()
        serial = CloudVoxelizer().make_bodies(
            snapshot, grid, iso=ISO, layer_count=3, max_body_cells=8_000
        )
        pool = WorkerPool(2)
        try:
            pooled = CloudVoxelizer(pool=pool).make_bodies(
                snapshot, grid, iso=ISO, layer_count=3, max_body_cells=8_000
            )
        finally:
            pool.close()
        assert bodies_equal(serial, pooled)
