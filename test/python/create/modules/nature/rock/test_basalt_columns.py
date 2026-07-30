# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the basalt columns: formation, tops, material, budget, errors."""

import numpy as np
import pytest

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import mineral
from rise_origin.create.modules.nature.rock import basalt_columns

RS_NULL_SID = np.iinfo(np.uint32).max

QUICK = {
    "width": 1.4,
    "depth": 1.0,
    "height": 1.6,
    "column_width": 0.3,
    "voxel_size": "2cm",
    "wear_subdivision": 1,
}


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


def occupied_num(draft):
    sids = np.asarray(get_stone(draft).bodies[0].material_reference_sid)
    return int((sids != RS_NULL_SID).sum())


class TestBasaltColumns:
    def test_default_build_is_one_fixed_basalt_formation(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(basalt_columns.MODULE, {"seed": 5, **QUICK})
            stone = get_stone(draft)
            assert stone.is_fixed is True
            assert stone.material_references == [mineral.BASALT]
            assert len(stone.bodies) == 1
            assert 4 <= occupied_num(draft) <= basalt_columns.FORMATION_VOXEL_CAP

    def test_uneven_tops_keep_fewer_voxels(self, tmp_path):
        with make_editor(tmp_path) as editor:
            flat = editor.create(
                basalt_columns.MODULE, {"seed": 5, "top_unevenness": 0.0, **QUICK}
            )
            jagged = editor.create(
                basalt_columns.MODULE, {"seed": 5, "top_unevenness": 1.0, **QUICK}
            )
            assert occupied_num(jagged) < occupied_num(flat)

    def test_too_narrow_columns_fail_at_draft_time(self, tmp_path):
        with make_editor(tmp_path) as editor:
            with pytest.raises(Exception):
                editor.create(
                    basalt_columns.MODULE,
                    {
                        "seed": 5,
                        "width": 1.0,
                        "depth": 1.0,
                        "height": 1.0,
                        "column_width": 0.1,
                        "voxel_size": "20cm",
                        "wear_subdivision": 1,
                    },
                )
