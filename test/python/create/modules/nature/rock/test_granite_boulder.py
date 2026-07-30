# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the granite boulder: shape, grain mix, budget, and errors."""

import numpy as np
import pytest

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import mineral
from rise_origin.create.modules.nature.rock import granite_boulder

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


def gather_mids(stone):
    mids = set()
    for body in stone.bodies:
        used = np.asarray(body.material_reference_sid) != RS_NULL_SID
        optical = np.asarray(body.material_optical_reference_mid)
        mids.update(np.unique(optical[used]).tolist())
    return mids


class TestGraniteBoulder:
    def test_default_build_is_one_dynamic_granite_stone(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(granite_boulder.MODULE, {"seed": 7, **QUICK})
            stone = get_stone(draft)
            assert stone.is_fixed is False
            assert stone.material_references == [mineral.GRANITE]
            assert len(stone.bodies) == 1
            sids = np.asarray(stone.bodies[0].material_reference_sid)
            occupied_num = int((sids != RS_NULL_SID).sum())
            assert 4 <= occupied_num <= granite_boulder.STONE_VOXEL_CAP

    def test_gray_tone_grows_no_pink_feldspar(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(
                granite_boulder.MODULE, {"seed": 7, "tone": 0.0, **QUICK}
            )
            mids = gather_mids(get_stone(draft))
            assert not (mids & set(mineral.GRANITE_ORTHOCLASE_MIDS))
            assert mids & set(mineral.GRANITE_PLAGIOCLASE_MIDS)

    def test_pink_tone_grows_orthoclase(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(
                granite_boulder.MODULE, {"seed": 7, "tone": 1.0, **QUICK}
            )
            mids = gather_mids(get_stone(draft))
            assert mids & set(mineral.GRANITE_ORTHOCLASE_MIDS)

    def test_crystals_dither_within_a_mineral(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(
                granite_boulder.MODULE, {"seed": 7, "tone": 0.0, **QUICK}
            )
            mids = gather_mids(get_stone(draft))
            # More than one plagioclase shade appears, so patches vary.
            assert len(mids & set(mineral.GRANITE_PLAGIOCLASE_MIDS)) > 1

    def test_too_coarse_dials_fail_at_draft_time(self, tmp_path):
        with make_editor(tmp_path) as editor:
            with pytest.raises(Exception):
                editor.create(
                    granite_boulder.MODULE,
                    {
                        "seed": 7,
                        "size": 0.2,
                        "voxel_size": "10cm",
                        "wear_subdivision": 1,
                    },
                )
