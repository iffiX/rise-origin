# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the Weber-Penn growth model: stems, leaves, and pruning."""

import numpy as np

from rise_origin.create.algorithm import WeberPennParams, WeberPennTree
from rise_origin.create.protocol import Rng

UPRIGHT = WeberPennParams(
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

GNARLED = WeberPennParams(
    shape=2,
    scale=4.0,
    levels=2,
    base_size=0.1,
    ratio=0.035,
    base_splits=3,
    seg_splits=(0.6, 0.4),
    split_angle=(35.0, 30.0),
    split_angle_v=(10.0, 10.0),
    curve_v=(90.0, 120.0),
    curve=(0.0, -30.0),
    branches=(1, 12),
    curve_res=(8, 5),
    length=(1.0, 0.5),
    down_angle=(0.0, 70.0),
    leaves=10,
    attraction_up=-0.4,
)


class TestGrow:
    def test_the_same_seed_grows_the_same_tree(self):
        first = WeberPennTree.grow(Rng(7), UPRIGHT)
        second = WeberPennTree.grow(Rng(7), UPRIGHT)
        assert len(first.stems) == len(second.stems)
        assert [stem.node_ids for stem in first.stems] == [
            stem.node_ids for stem in second.stems
        ]
        assert np.allclose(np.vstack(first.positions), np.vstack(second.positions))
        assert np.allclose(first.leaf_attaches, second.leaf_attaches)

    def test_different_seeds_grow_different_trees(self):
        first = WeberPennTree.grow(Rng(7), UPRIGHT)
        second = WeberPennTree.grow(Rng(8), UPRIGHT)
        shared = min(len(first.positions), len(second.positions))
        assert not np.allclose(
            np.vstack(first.positions)[:shared],
            np.vstack(second.positions)[:shared],
        )

    def test_the_trunk_rises_and_stems_carry_levels(self):
        graph = WeberPennTree.grow(Rng(7), UPRIGHT)
        lo, hi = graph.get_bounds()
        assert hi[2] > 5.0
        assert graph.stems[0].level == 0
        assert graph.stems[0].parent_stem_id == -1
        assert {stem.level for stem in graph.stems} == {0, 1, 2}
        for stem in graph.stems:
            if stem.parent_stem_id >= 0:
                assert stem.parent_stem_id < len(graph.stems)

    def test_the_gnarled_preset_is_wider_than_tall(self):
        upright = WeberPennTree.grow(Rng(7), UPRIGHT)
        gnarled = WeberPennTree.grow(Rng(7), GNARLED)

        def width_over_height(graph):
            lo, hi = graph.get_bounds()
            return max(hi[0] - lo[0], hi[1] - lo[1]) / max(hi[2], 1e-6)

        assert width_over_height(gnarled) > width_over_height(upright)

    def test_base_splits_make_several_trunks(self):
        graph = WeberPennTree.grow(Rng(7), GNARLED)
        trunk_chains = [stem for stem in graph.stems if stem.level == 0]
        assert len(trunk_chains) >= 4

    def test_no_stem_descends_into_the_ground_band(self):
        graph = WeberPennTree.grow(Rng(7), GNARLED)
        clearance = GNARLED.ground_clearance
        for stem in graph.stems:
            spine = np.vstack([graph.positions[i] for i in stem.node_ids])
            for prev_z, next_z in zip(spine[:-1, 2], spine[1:, 2]):
                assert not (next_z < clearance - 1e-9 and next_z < prev_z)
        if graph.leaf_attaches.shape[0]:
            assert graph.leaf_attaches[:, 2].min() >= clearance - 1e-9

    def test_max_stems_caps_the_tree(self):
        capped = WeberPennParams(
            **{**GNARLED.__dict__, "max_stems": 40},
        )
        graph = WeberPennTree.grow(Rng(7), capped)
        assert len(graph.stems) <= 40

    def test_whorl_branches_ring_the_trunk(self):
        whorled = WeberPennParams(
            shape=0,
            scale=10.0,
            levels=2,
            base_size=0.3,
            branches=(1, 8),
            whorl_branches=(1, 4),
            curve_res=(10, 4),
        )
        graph = WeberPennTree.grow(Rng(7), whorled)
        attach_counts = {}
        for stem in graph.stems:
            if stem.level == 1:
                base = stem.node_ids[0]
                attach_counts[base] = attach_counts.get(base, 0) + 1
        assert attach_counts
        assert max(attach_counts.values()) >= 4
        assert sum(attach_counts.values()) >= 24

    def test_negative_rotate_spreads_children_around_the_parent(self):
        alternating = WeberPennParams(
            shape=3,
            scale=10.0,
            levels=2,
            base_size=0.1,
            branches=(1, 20),
            curve_res=(10, 4),
            rotate=(0.0, -120.0),
        )
        graph = WeberPennTree.grow(Rng(7), alternating)
        headings = []
        for stem in graph.stems:
            if stem.level != 1 or len(stem.node_ids) < 2:
                continue
            span = graph.positions[stem.node_ids[1]] - graph.positions[stem.node_ids[0]]
            headings.append(np.arctan2(span[1], span[0]))
        assert len(headings) >= 10
        spread = np.ptp(np.array(headings))
        assert spread > np.pi

    def test_break_fraction_snaps_stems_short(self):
        broken = WeberPennParams(
            shape=0,
            scale=8.0,
            levels=2,
            branches=(1, 8),
            curve_res=(8, 5),
            break_fraction=(0.5, 0.4),
        )
        graph = WeberPennTree.grow(Rng(7), broken)
        trunk = graph.stems[0]
        assert trunk.is_broken
        assert len(trunk.node_ids) < 9
        assert any(stem.is_broken for stem in graph.stems if stem.level == 1)


class TestLeaves:
    def test_leaves_hang_on_the_deepest_stems_with_unit_frames(self):
        graph = WeberPennTree.grow(Rng(7), UPRIGHT)
        count = graph.leaf_attaches.shape[0]
        assert count > 0
        assert graph.leaf_length_axes.shape == (count, 3)
        assert graph.leaf_width_axes.shape == (count, 3)
        assert graph.leaf_stem_ids.shape == (count,)
        deepest = max(stem.level for stem in graph.stems)
        for sid in np.unique(graph.leaf_stem_ids):
            assert graph.stems[int(sid)].level == deepest
        assert np.allclose(np.linalg.norm(graph.leaf_length_axes, axis=1), 1.0)
        assert np.allclose(np.linalg.norm(graph.leaf_width_axes, axis=1), 1.0)
        dots = np.sum(graph.leaf_length_axes * graph.leaf_width_axes, axis=1)
        assert np.abs(dots).max() < 1e-6

    def test_a_bare_tree_grows_no_leaves(self):
        bare = WeberPennParams(shape=1, scale=8.0, levels=3, branches=(1, 18, 10))
        graph = WeberPennTree.grow(Rng(7), bare)
        assert graph.leaf_attaches.shape[0] == 0


class TestRotate:
    def test_rotate_turns_positions_and_leaf_frames_together(self):
        graph = WeberPennTree.grow(Rng(7), UPRIGHT)
        top_before = max(position[2] for position in graph.positions)
        attaches_before = graph.leaf_attaches.copy()
        graph.rotate(np.array([1.0, 0.0, 0.0]), np.pi / 2.0)
        top_after = max(position[2] for position in graph.positions)
        assert top_after < top_before * 0.5
        assert np.allclose(
            np.linalg.norm(graph.leaf_attaches, axis=1),
            np.linalg.norm(attaches_before, axis=1),
        )
        assert np.allclose(np.linalg.norm(graph.leaf_length_axes, axis=1), 1.0)
