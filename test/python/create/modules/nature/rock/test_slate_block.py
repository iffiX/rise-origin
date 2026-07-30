# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the slate block: build, shedding, one part, modes, errors."""

import cc3d
import numpy as np
import pytest

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import mineral
from rise_origin.create.modules.nature.rock import slate_block

RS_NULL_SID = np.iinfo(np.uint32).max

QUICK = {
    "width": 1.2,
    "depth": 0.8,
    "height": 1.0,
    "plate_thickness": 0.08,
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


class TestSlateBlock:
    def test_default_build_is_one_fixed_slate_part(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(slate_block.MODULE, {"seed": 5, **QUICK})
            stone = get_stone(draft)
            assert stone.is_fixed is True
            assert stone.material_references == [mineral.SLATE]
            occupancy = get_occupancy(draft)
            occupied_num = int(occupancy.sum())
            assert 4 <= occupied_num <= slate_block.FORMATION_VOXEL_CAP
            labels = cc3d.connected_components(occupancy, connectivity=6)
            assert int(labels.max()) == 1

    def test_raggedness_sheds_more(self, tmp_path):
        with make_editor(tmp_path) as editor:
            smooth = editor.create(
                slate_block.MODULE, {"seed": 5, "raggedness": 0.0, **QUICK}
            )
            ragged = editor.create(
                slate_block.MODULE, {"seed": 5, "raggedness": 1.0, **QUICK}
            )
            assert int(get_occupancy(ragged).sum()) < int(get_occupancy(smooth).sum())

    def test_occupancy_mask_carves_the_block(self, tmp_path):
        mask = np.ones((6, 4, 5), dtype=bool)
        mask[3:, :, 2:] = False
        with make_editor(tmp_path) as editor:
            full = editor.create(slate_block.MODULE, {"seed": 5, **QUICK})
            carved = editor.create(
                slate_block.MODULE, {"seed": 5, "occupancy": mask, **QUICK}
            )
            occupancy = get_occupancy(carved)
            assert int(occupancy.sum()) < int(get_occupancy(full).sum())
            labels = cc3d.connected_components(occupancy, connectivity=6)
            assert int(labels.max()) == 1
            half_x = occupancy.shape[0] // 2
            half_z = occupancy.shape[2] // 2
            kept = occupancy[: half_x - 2, :, half_z + 2 :]
            gone = occupancy[half_x + 2 :, :, half_z + 2 :]
            assert kept.any()
            assert int(gone.sum()) < int(kept.sum()) // 4

    def test_movable_block_is_dynamic(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(
                slate_block.MODULE, {"seed": 5, "is_fixed": False, **QUICK}
            )
            assert get_stone(draft).is_fixed is False

    def test_too_coarse_plates_fail_at_draft_time(self, tmp_path):
        with make_editor(tmp_path) as editor:
            with pytest.raises(Exception):
                editor.create(
                    slate_block.MODULE,
                    {
                        "seed": 5,
                        "width": 1.0,
                        "depth": 1.0,
                        "height": 1.0,
                        "plate_thickness": 0.04,
                        "voxel_size": "10cm",
                        "wear_subdivision": 1,
                    },
                )
