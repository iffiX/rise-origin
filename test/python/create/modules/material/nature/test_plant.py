# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the plant material package: optical order and purity."""

import hashlib
import os
import tempfile

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import plant


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


class TestPlantPackage:
    def test_the_package_releases_three_materials_with_optics(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(plant.MODULE)
            names = [rsc.name for rsc in draft.rscs]
            assert plant.WOOD in names
            assert plant.LEAF in names
            assert plant.LEAF_SOFT in names
            for reference in (
                plant.WOOD_OPTICAL_REFERENCES
                + plant.LEAF_OPTICAL_REFERENCES
                + plant.LEAF_SOFT_OPTICAL_REFERENCES
            ):
                assert reference in names

    def test_optical_orders_and_season_subsets_agree(self):
        assert len(plant.WOOD_OPTICAL_REFERENCES) == len(plant.WOOD_ALBEDOS)
        assert len(plant.LEAF_OPTICAL_REFERENCES) == len(plant.LEAF_ALBEDOS)
        count = len(plant.LEAF_ALBEDOS)
        for season, mids in plant.SEASON_LEAF_MIDS.items():
            assert all(0 <= mid < count for mid in mids)
        assert plant.SEASON_LEAF_MIDS["winter"] == ()

    def test_a_double_build_is_byte_identical(self, tmp_path):
        with make_editor(tmp_path / "a") as first:
            one = make_rsc_digests(first.create(plant.MODULE))
        with make_editor(tmp_path / "b") as second:
            two = make_rsc_digests(second.create(plant.MODULE))
        assert one == two
