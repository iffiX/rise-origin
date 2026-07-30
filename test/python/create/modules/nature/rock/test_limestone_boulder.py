# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the limestone boulder: shape, tone, karst, budget, errors."""

import numpy as np
import pytest

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import mineral
from rise_origin.create.modules.nature.rock import limestone_boulder

RS_NULL_SID = np.iinfo(np.uint32).max

QUICK = {"size": 0.6, "voxel_size": "5cm", "wear_subdivision": 3}


def make_editor(tmp_path):
    return cr.Editor(
        "world",
        session_dir=str(tmp_path / "session"),
        cache=cr.RealizationCache(str(tmp_path / "cache")),
    )


def get_stone(draft):
    for node in draft.iterate_subtree_nodes():
        for rsc in node.rscs:
            if hasattr(rsc, "bodies"):
                return rsc
    raise AssertionError("no structure in the draft")


class TestLimestoneBoulder:
    def test_default_build_is_one_dynamic_limestone_stone(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(limestone_boulder.MODULE, {"seed": 5, **QUICK})
            stone = get_stone(draft)
            assert stone.is_fixed is False
            assert stone.material_references == [mineral.LIMESTONE]
            assert len(stone.bodies) == 1
            sids = np.asarray(stone.bodies[0].material_reference_sid)
            occupied_num = int((sids != RS_NULL_SID).sum())
            assert 4 <= occupied_num <= limestone_boulder.STONE_VOXEL_CAP

    def test_karst_pits_the_stone_smaller(self, tmp_path):
        with make_editor(tmp_path) as editor:
            smooth = editor.create(
                limestone_boulder.MODULE, {"seed": 5, "karst": 0.0, **QUICK}
            )
            pocked = editor.create(
                limestone_boulder.MODULE, {"seed": 5, "karst": 1.0, **QUICK}
            )
            smooth_num = int(
                (
                    np.asarray(get_stone(smooth).bodies[0].material_reference_sid)
                    != RS_NULL_SID
                ).sum()
            )
            pocked_num = int(
                (
                    np.asarray(get_stone(pocked).bodies[0].material_reference_sid)
                    != RS_NULL_SID
                ).sum()
            )
            assert pocked_num < smooth_num

    def test_too_coarse_dials_fail_at_draft_time(self, tmp_path):
        with make_editor(tmp_path) as editor:
            with pytest.raises(Exception):
                editor.create(
                    limestone_boulder.MODULE,
                    {
                        "seed": 5,
                        "size": 0.2,
                        "voxel_size": "10cm",
                        "wear_subdivision": 1,
                    },
                )
