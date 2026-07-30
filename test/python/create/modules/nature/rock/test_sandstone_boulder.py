# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the sandstone boulder: shape, strata, budget, and errors."""

import numpy as np
import pytest

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import mineral
from rise_origin.create.modules.nature.rock import sandstone_boulder

RS_NULL_SID = np.iinfo(np.uint32).max

QUICK = {
    "size": 0.6,
    "voxel_size": "5cm",
    "wear_subdivision": 3,
}


def make_editor(tmp_path, name="world"):
    return cr.Editor(
        name,
        session_dir=str(tmp_path / "session"),
        cache=cr.RealizationCache(str(tmp_path / "cache")),
    )


def get_stone(draft):
    for node in draft.iterate_subtree_nodes():
        for rsc in node.rscs:
            if hasattr(rsc, "bodies"):
                return rsc
    raise AssertionError("no structure in the draft")


class TestSandstoneBoulder:
    def test_default_build_is_one_dynamic_layered_stone(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(sandstone_boulder.MODULE, {"seed": 3, **QUICK})
            stone = get_stone(draft)
            assert stone.is_fixed is False
            assert stone.material_references == [
                mineral.SANDSTONE_SOFT,
                mineral.SANDSTONE_HARD,
            ]
            assert len(stone.bodies) == 1
            sids = np.asarray(stone.bodies[0].material_reference_sid)
            occupied = sids != RS_NULL_SID
            occupied_num = int(occupied.sum())
            assert occupied_num >= 4
            assert occupied_num <= sandstone_boulder.STONE_VOXEL_CAP
            used = set(np.unique(sids[occupied]).tolist())
            assert used == {0, 1}

    def test_weathering_wears_the_stone_smaller(self, tmp_path):
        with make_editor(tmp_path) as editor:
            fresh = editor.create(
                sandstone_boulder.MODULE, {"seed": 5, "weathering": 0.0, **QUICK}
            )
            worn = editor.create(
                sandstone_boulder.MODULE, {"seed": 5, "weathering": 0.9, **QUICK}
            )
            fresh_sids = np.asarray(get_stone(fresh).bodies[0].material_reference_sid)
            worn_sids = np.asarray(get_stone(worn).bodies[0].material_reference_sid)
            assert int((worn_sids != RS_NULL_SID).sum()) < int(
                (fresh_sids != RS_NULL_SID).sum()
            )

    def test_too_coarse_dials_fail_at_draft_time(self, tmp_path):
        with make_editor(tmp_path) as editor:
            with pytest.raises(Exception):
                editor.create(
                    sandstone_boulder.MODULE,
                    {
                        "seed": 3,
                        "size": 0.2,
                        "voxel_size": "10cm",
                        "wear_subdivision": 1,
                    },
                )
