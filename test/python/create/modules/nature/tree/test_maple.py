# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the maple: seasons, softness, budgets, and purity."""

import hashlib
import os
import tempfile

import numpy as np

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import plant
from rise_origin.create.modules.nature.tree import maple

RS_NULL_SID = np.iinfo(np.uint32).max

QUICK = {
    "height": 4.0,
    "trunk_voxel_size": "5cm",
    "canopy_voxel_size": "10cm",
}


def make_editor(tmp_path, name="world"):
    return cr.Editor(
        name,
        session_dir=str(tmp_path / "session"),
        cache=cr.RealizationCache(str(tmp_path / "cache")),
    )


def make_rsc_digests(draft):
    digests = []
    for node in draft.iterate_subtree_nodes():
        for rsc in node.rscs:
            handle, path = tempfile.mkstemp(suffix=".rscb")
            os.close(handle)
            try:
                rsc.save(path)
                with open(path, "rb") as saved:
                    digests.append(hashlib.sha256(saved.read()).hexdigest())
            finally:
                os.remove(path)
    return digests


def count_occupied_by_sid(tree, sid):
    total = 0
    for body in tree.bodies:
        sids = np.asarray(body.material_reference_sid)
        total += int((sids == sid).sum())
    return total


def gather_mids_by_sid(tree, sid):
    mids = set()
    for body in tree.bodies:
        sids = np.asarray(body.material_reference_sid)
        used = sids == sid
        if used.any():
            values = np.asarray(body.material_optical_reference_mid)
            mids |= set(values[used].tolist())
    return mids


class TestMaple:
    def test_a_summer_maple_carries_wood_and_leaf_voxels(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(maple.MODULE, QUICK)
            tree = draft.rscs[0]
            assert list(tree.material_references) == [plant.WOOD, plant.LEAF]
            assert [len(inner) for inner in tree.material_optical_references] == [
                len(plant.WOOD_ALBEDOS),
                len(plant.LEAF_ALBEDOS),
            ]
            assert len(tree.bodies) > 2
            assert tree.is_fixed
            assert count_occupied_by_sid(tree, 0) > 0
            assert count_occupied_by_sid(tree, 1) > 0
            assert len(draft.requires) == 2

    def test_every_season_builds_and_winter_stands_bare(self, tmp_path):
        with make_editor(tmp_path) as editor:
            for season in ("spring", "summer", "autumn"):
                draft = editor.create(maple.MODULE, {**QUICK, "season": season})
                assert count_occupied_by_sid(draft.rscs[0], 1) > 0
            bare = editor.create(maple.MODULE, {**QUICK, "season": "winter"})
            assert count_occupied_by_sid(bare.rscs[0], 1) == 0
            assert bare.rscs[0].is_fixed

    def test_soft_leaves_make_the_tree_dynamic(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(maple.MODULE, {**QUICK, "is_leaf_soft": True})
            tree = draft.rscs[0]
            assert not tree.is_fixed
            assert list(tree.material_references) == [plant.WOOD, plant.LEAF_SOFT]
            assert draft.requires[1].material_reference == "leaf_soft"

    def test_leaf_mids_stay_inside_the_season_subset(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(maple.MODULE, {**QUICK, "season": "autumn"})
            mids = gather_mids_by_sid(draft.rscs[0], 1)
            assert mids
            assert mids <= set(plant.SEASON_LEAF_MIDS["autumn"])

    def test_the_default_maple_stays_inside_its_budget(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(maple.MODULE)
            tree = draft.rscs[0]
            assert 0 < count_occupied_by_sid(tree, 0) < maple.LIMB_VOXEL_CAP
            assert 0 < count_occupied_by_sid(tree, 1) < maple.CANOPY_VOXEL_CAP
            assert len(tree.bodies) < maple.BODY_CAP

    def test_a_gnarled_thick_maple_stays_under_the_body_cap(self, tmp_path):
        values = {**QUICK, "gnarl": 0.9, "trunk_count": 3, "spread": 0.8}
        with make_editor(tmp_path) as editor:
            draft = editor.create(maple.MODULE, values)
            assert len(draft.rscs[0].bodies) < maple.BODY_CAP

    def test_a_double_build_is_byte_identical(self, tmp_path):
        values = {**QUICK, "seed": 11, "gnarl": 0.8, "trunk_count": 3}
        with make_editor(tmp_path / "a") as first:
            one = make_rsc_digests(first.create(maple.MODULE, values))
        with make_editor(tmp_path / "b") as second:
            two = make_rsc_digests(second.create(maple.MODULE, values))
        assert one == two
