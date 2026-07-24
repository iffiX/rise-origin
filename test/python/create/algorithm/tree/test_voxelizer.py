# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the tree voxelizer: bodies, tiles, holders, and budgets."""

import cc3d
import numpy as np

from rise_origin.create.algorithm import (
    TreeBark,
    TreeBarkBands,
    TreeBarkFurrows,
    TreeBarkPatches,
    TreeBarkPlates,
    TreeBodyBudget,
    TreeLeaf,
    TreeVoxelizer,
    WeberPennParams,
    WeberPennTree,
)
from rise_origin.create.protocol import Rng

PRESET = WeberPennParams(
    shape=1,
    scale=8.0,
    levels=3,
    base_size=0.35,
    ratio=0.02,
    curve_v=(15.0, 40.0, 60.0),
    branches=(1, 18, 10),
    curve_res=(6, 4, 3),
    leaves=12,
)

BARK = TreeBark(moss_coverage=0.3)
LEAF = TreeLeaf(outline="maple_star", length=0.2, width_ratio=0.85)
MIDS = (4, 5, 6)


def make_bodies(budget=None, shed_fraction=0.0):
    graph = WeberPennTree.grow(Rng(7), PRESET)
    return TreeVoxelizer().make_bodies(
        Rng(7),
        graph,
        trunk_voxel_size=0.02,
        canopy_voxel_size=0.02,
        bark=BARK,
        leaf=LEAF,
        leaf_optical_mids=MIDS,
        shed_fraction=shed_fraction,
        budget=budget or TreeBodyBudget(max_leaf_area=60.0),
    )


def gather_tile_keys(body):
    cells = np.argwhere(body.occupancy)
    lattice = np.round(body.origin / body.voxel_size).astype(np.int64)
    return {tuple(cell) for cell in (cells + lattice)}


def count_loose_parts(bodies):
    loose = 0
    for body in bodies:
        if not (body.material_grid == 1)[body.occupancy].any():
            continue
        labels = cc3d.connected_components(body.occupancy, connectivity=6)
        wood = body.occupancy & (body.material_grid == 0)
        anchored = set(np.unique(labels[wood]).tolist())
        present = set(np.unique(labels[body.occupancy]).tolist())
        loose += len(present - anchored)
    return loose


class TestBodies:
    def test_a_tree_lands_in_few_bodies(self):
        bodies = make_bodies()
        assert 0 < len(bodies) < 100
        for body in bodies:
            assert body.occupancy.any()
            assert body.optical_mid_grid.shape == body.occupancy.shape
            assert body.material_grid.shape == body.occupancy.shape
            assert abs(np.linalg.norm(body.orientation) - 1.0) < 1e-6

    def test_thick_limbs_sweep_oriented_and_twigs_tile_flat(self):
        bodies = make_bodies()
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        oriented = [
            body
            for body in bodies
            if not np.allclose(body.orientation, identity, atol=1e-6)
        ]
        tiles = [body for body in bodies if np.allclose(body.orientation, identity)]
        assert oriented
        assert tiles
        for body in oriented:
            assert (body.material_grid[body.occupancy] == 0).all()

    def test_tiles_are_disjoint_in_space(self):
        bodies = make_bodies()
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        seen = set()
        for body in bodies:
            if not np.allclose(body.orientation, identity):
                continue
            keys = gather_tile_keys(body)
            assert not (keys & seen)
            seen |= keys

    def test_looks_stay_inside_their_material_lists(self):
        bodies = make_bodies()
        wood_mids = {
            BARK.optical_mid_base,
            BARK.optical_mid_shade,
            BARK.optical_mid_moss,
            BARK.optical_mid_crevice,
            BARK.optical_mid_light,
        }
        for body in bodies:
            wood = body.occupancy & (body.material_grid == 0)
            leaf = body.occupancy & (body.material_grid == 1)
            if wood.any():
                assert set(np.unique(body.optical_mid_grid[wood])) <= wood_mids
            if leaf.any():
                assert set(np.unique(body.optical_mid_grid[leaf])) <= set(MIDS)

    def test_leaf_tiles_carry_fixed_wood_holders(self):
        bodies = make_bodies()
        with_leaves = [
            body for body in bodies if (body.material_grid == 1)[body.occupancy].any()
        ]
        assert with_leaves
        holders = sum(
            int((body.occupancy & (body.material_grid == 0)).sum())
            for body in with_leaves
        )
        assert holders > 0

    def test_the_leaf_budget_thins_evenly(self):
        small = make_bodies(TreeBodyBudget(max_leaf_area=16.0))
        large = make_bodies(TreeBodyBudget(max_leaf_area=60.0))

        def count_leaves(bodies):
            return sum(
                int((body.occupancy & (body.material_grid == 1)).sum())
                for body in bodies
            )

        assert count_leaves(small) < 55_000
        assert count_leaves(small) < count_leaves(large)

    def test_no_leaf_mids_grows_bare_wood(self):
        graph = WeberPennTree.grow(Rng(7), PRESET)
        bodies = TreeVoxelizer().make_bodies(
            Rng(7),
            graph,
            trunk_voxel_size=0.02,
            canopy_voxel_size=0.02,
            bark=BARK,
            leaf=LEAF,
            leaf_optical_mids=(),
        )
        for body in bodies:
            assert (body.material_grid[body.occupancy] == 0).all()

    def test_every_leaf_part_is_anchored_at_shed_zero(self):
        assert count_loose_parts(make_bodies()) == 0

    def test_the_shed_dial_keeps_loose_parts(self):
        pool = count_loose_parts(make_bodies(shed_fraction=1.0))
        half = count_loose_parts(make_bodies(shed_fraction=0.5))
        assert pool > 0
        assert 0 < half < pool
        assert abs(half - pool / 2) <= pool * 0.2

    def test_shedding_is_deterministic(self):
        first = make_bodies(shed_fraction=0.5)
        second = make_bodies(shed_fraction=0.5)
        for one, two in zip(first, second):
            assert (one.occupancy == two.occupancy).all()

    def test_bodies_are_deterministic(self):
        first = make_bodies()
        second = make_bodies()
        assert len(first) == len(second)
        for one, two in zip(first, second):
            assert (one.occupancy == two.occupancy).all()
            assert (one.optical_mid_grid == two.optical_mid_grid).all()
            assert (one.material_grid == two.material_grid).all()
            assert np.allclose(one.origin, two.origin)
            assert np.allclose(one.orientation, two.orientation)

    def test_every_bark_detail_family_builds(self):
        details = (
            TreeBarkFurrows(),
            TreeBarkBands(coverage=0.4),
            TreeBarkPlates(),
            TreeBarkPatches(optical_mid_levels=(3, 0, 4)),
        )
        for detail in details:
            graph = WeberPennTree.grow(Rng(7), PRESET)
            bodies = TreeVoxelizer().make_bodies(
                Rng(7),
                graph,
                trunk_voxel_size=0.02,
                canopy_voxel_size=0.02,
                bark=TreeBark(detail=detail),
                leaf=LEAF,
                leaf_optical_mids=MIDS,
                budget=TreeBodyBudget(max_leaf_area=24.0),
            )
            assert bodies
            allowed = {0, 1, 2, 3, 4}
            for body in bodies:
                wood = body.occupancy & (body.material_grid == 0)
                if wood.any():
                    assert set(np.unique(body.optical_mid_grid[wood])) <= allowed

    def test_structure_bodies_convert(self):
        bodies = make_bodies()
        converted = list(TreeVoxelizer.iterate_structure_bodies(bodies[:3]))
        assert len(converted) == 3
        for engine_body in converted:
            assert engine_body.x_voxels > 0
            assert abs(engine_body.voxel_size - 0.02) < 1e-6


class TestLeafTemplates:
    def test_every_outline_covers_its_attach_stalk(self):
        for outline in ("ellipse", "hex", "tee", "maple_star", "fan", "strip"):
            template, stalk_count = TreeVoxelizer._make_leaf_template(
                TreeLeaf(outline=outline, length=0.2, width_ratio=0.85), 0.02
            )
            assert template.shape[0] > 10
            assert 0 < stalk_count < template.shape[0]
            assert np.abs(template[:stalk_count, 0]).min() < 0.03
            assert template[:, 0].max() <= 0.2 + 0.02


class TestBodyCellBudget:
    def test_every_body_stays_inside_the_cell_budget(self):
        graph = WeberPennTree.grow(Rng(7), PRESET)
        budget = TreeBodyBudget(max_leaf_area=12.0, max_body_cells=300_000)
        bodies = TreeVoxelizer().make_bodies(
            Rng(7),
            graph,
            trunk_voxel_size=0.01,
            canopy_voxel_size=0.01,
            bark=BARK,
            leaf=LEAF,
            leaf_optical_mids=MIDS,
            budget=budget,
        )
        assert bodies
        assert max(body.occupancy.size for body in bodies) <= 300_000

    def test_default_sizes_keep_their_body_count(self):
        loose = TreeBodyBudget(max_leaf_area=60.0, max_body_cells=10**12)
        assert len(make_bodies()) == len(make_bodies(budget=loose))

    def test_split_chains_keep_the_wood_together(self):
        graph = WeberPennTree.grow(Rng(7), PRESET)
        kept = TreeVoxelizer().make_bodies(
            Rng(7),
            graph,
            trunk_voxel_size=0.02,
            canopy_voxel_size=0.02,
            bark=BARK,
            leaf=None,
            leaf_optical_mids=(),
            budget=TreeBodyBudget(),
        )
        split = TreeVoxelizer().make_bodies(
            Rng(7),
            graph,
            trunk_voxel_size=0.02,
            canopy_voxel_size=0.02,
            bark=BARK,
            leaf=None,
            leaf_optical_mids=(),
            budget=TreeBodyBudget(max_body_cells=100_000),
        )
        kept_total = sum(int(body.occupancy.sum()) for body in kept)
        split_total = sum(int(body.occupancy.sum()) for body in split)
        assert len(split) > len(kept)
        # Split spans share their joint spheres, the same overlap limb
        # joints already carry, so the split total may run a bit over.
        assert split_total >= kept_total
        assert (split_total - kept_total) / kept_total < 0.05


class TestCoarseCleanliness:
    def test_line_cells_are_face_connected(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            a = rng.uniform(-1.0, 1.0, 3)
            b = rng.uniform(-1.0, 1.0, 3)
            chain = TreeVoxelizer._rasterize_line_cells(a, b, 0.02)
            assert chain.shape[0] >= 1
            if chain.shape[0] > 1:
                steps = np.abs(np.diff(chain.astype(int), axis=0)).sum(axis=1)
                assert (steps == 1).all()

    def test_detail_shallower_than_a_voxel_keeps_the_surface(self):
        graph = WeberPennTree.grow(Rng(7), PRESET)

        def build(bark):
            return TreeVoxelizer().make_bodies(
                Rng(7),
                graph,
                trunk_voxel_size=0.1,
                canopy_voxel_size=0.1,
                bark=bark,
                leaf=None,
                leaf_optical_mids=(),
                budget=TreeBodyBudget(),
            )

        carved = build(BARK)
        plain = build(TreeBark(detail=TreeBarkFurrows(depth=0.0), moss_coverage=0.3))
        assert len(carved) == len(plain)
        for one, other in zip(carved, plain):
            assert (one.occupancy == other.occupancy).all()


class TestPooledSweep:
    def test_pooled_build_matches_serial_bytes(self):
        from rise_origin.create.protocol.worker_pool import WorkerPool

        graph = WeberPennTree.grow(Rng(7), PRESET)
        budget = TreeBodyBudget(max_leaf_area=60.0)

        def build(voxelizer):
            return voxelizer.make_bodies(
                Rng(7),
                graph,
                trunk_voxel_size=0.02,
                canopy_voxel_size=0.02,
                bark=BARK,
                leaf=LEAF,
                leaf_optical_mids=MIDS,
                shed_fraction=0.3,
                budget=budget,
            )

        serial = build(TreeVoxelizer())
        pool = WorkerPool(2)
        try:
            pooled = build(TreeVoxelizer(pool=pool))
        finally:
            pool.close()
        assert len(serial) == len(pooled)
        for one, other in zip(serial, pooled):
            assert one.occupancy.shape == other.occupancy.shape
            assert (one.occupancy == other.occupancy).all()
            assert (one.material_grid == other.material_grid).all()
            assert (one.optical_mid_grid == other.optical_mid_grid).all()
            assert np.allclose(one.origin, other.origin)
            assert np.allclose(one.orientation, other.orientation)
