# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the marble boulder: shape, veins, budget, and errors."""

import numpy as np
import pytest

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import mineral
from rise_origin.create.modules.nature.rock import marble_boulder

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


def gather_mids(stone):
    mids = set()
    for body in stone.bodies:
        used = np.asarray(body.material_reference_sid) != RS_NULL_SID
        optical = np.asarray(body.material_optical_reference_mid)
        mids.update(np.unique(optical[used]).tolist())
    return mids


class TestMarbleBoulder:
    def test_default_build_has_body_and_veins(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(marble_boulder.MODULE, {"seed": 3, **QUICK})
            stone = get_stone(draft)
            assert stone.is_fixed is False
            assert stone.material_references == [mineral.MARBLE]
            assert len(stone.bodies) == 1
            mids = gather_mids(stone)
            body_mid, vein_mids = marble_boulder._MarbleShape.make_tone_mids(0.0)
            assert body_mid in mids
            assert mids & set(vein_mids)

    def test_one_stone_carries_one_body_tone(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(
                marble_boulder.MODULE, {"seed": 3, "vein_thickness": 0.0, **QUICK}
            )
            mids = gather_mids(get_stone(draft))
            body_mid, vein_mids = marble_boulder._MarbleShape.make_tone_mids(0.0)
            assert mids - set(vein_mids) == {body_mid}

    def test_tone_moves_the_body_down_the_run(self, tmp_path):
        with make_editor(tmp_path) as editor:
            pale = editor.create(
                marble_boulder.MODULE, {"seed": 3, "tone": 0.0, **QUICK}
            )
            dark = editor.create(
                marble_boulder.MODULE, {"seed": 3, "tone": 1.0, **QUICK}
            )
            pale_body, _ = marble_boulder._MarbleShape.make_tone_mids(0.0)
            dark_body, _ = marble_boulder._MarbleShape.make_tone_mids(1.0)
            assert dark_body > pale_body
            assert pale_body in gather_mids(get_stone(pale))
            assert dark_body in gather_mids(get_stone(dark))

    def test_thicker_veins_cover_more(self, tmp_path):
        with make_editor(tmp_path) as editor:
            thin = editor.create(
                marble_boulder.MODULE, {"seed": 3, "vein_thickness": 0.0, **QUICK}
            )
            thick = editor.create(
                marble_boulder.MODULE, {"seed": 3, "vein_thickness": 1.0, **QUICK}
            )
            assert _vein_share(get_stone(thick)) > _vein_share(get_stone(thin))

    def test_too_coarse_dials_fail_at_draft_time(self, tmp_path):
        with make_editor(tmp_path) as editor:
            with pytest.raises(Exception):
                editor.create(
                    marble_boulder.MODULE,
                    {
                        "seed": 3,
                        "size": 0.2,
                        "voxel_size": "10cm",
                        "wear_subdivision": 1,
                    },
                )


def _vein_share(stone):
    vein = 0
    total = 0
    _body_mid, vein_mids = marble_boulder._MarbleShape.make_tone_mids(0.0)
    for body in stone.bodies:
        used = np.asarray(body.material_reference_sid) != RS_NULL_SID
        optical = np.asarray(body.material_optical_reference_mid)[used]
        total += optical.size
        vein += int(np.isin(optical, vein_mids).sum())
    return vein / max(total, 1)
