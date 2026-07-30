# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the slate boulder: build, shedding, one part, modes."""

import cc3d
import numpy as np

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import mineral
from rise_origin.create.modules.nature.rock import slate_boulder

RS_NULL_SID = np.iinfo(np.uint32).max

QUICK = {
    "size": 0.6,
    "plate_thickness": 0.05,
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


def get_occupancy(draft):
    body = get_stone(draft).bodies[0]
    sids = np.asarray(body.material_reference_sid)
    shape = (body.x_voxels, body.y_voxels, body.z_voxels)
    return (sids != RS_NULL_SID).reshape(shape, order="F")


class TestSlateBoulder:
    def test_default_build_is_one_dynamic_slate_part(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(slate_boulder.MODULE, {"seed": 5, **QUICK})
            stone = get_stone(draft)
            assert stone.is_fixed is False
            assert stone.material_references == [mineral.SLATE]
            occupancy = get_occupancy(draft)
            occupied_num = int(occupancy.sum())
            assert 4 <= occupied_num <= slate_boulder.STONE_VOXEL_CAP
            labels = cc3d.connected_components(occupancy, connectivity=6)
            assert int(labels.max()) == 1

    def test_raggedness_sheds_more(self, tmp_path):
        with make_editor(tmp_path) as editor:
            smooth = editor.create(
                slate_boulder.MODULE, {"seed": 5, "raggedness": 0.0, **QUICK}
            )
            ragged = editor.create(
                slate_boulder.MODULE, {"seed": 5, "raggedness": 1.0, **QUICK}
            )
            assert int(get_occupancy(ragged).sum()) < int(get_occupancy(smooth).sum())

    def test_fixed_chunk_is_static(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(
                slate_boulder.MODULE, {"seed": 5, "is_fixed": True, **QUICK}
            )
            assert get_stone(draft).is_fixed is True
