# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the sandstone block: build, occupancy carve, one part, modes."""

import cc3d
import numpy as np

from rise_origin import create as cr
from rise_origin.create.modules.nature.rock import sandstone_block

RS_NULL_SID = np.iinfo(np.uint32).max

QUICK = {
    "width": 1.4,
    "depth": 0.9,
    "height": 1.1,
    "seam_spacing": 0.5,
    "layer_thickness": 0.12,
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


class TestSandstoneBlock:
    def test_default_build_is_one_fixed_part(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(sandstone_block.MODULE, {"seed": 4, **QUICK})
            stone = get_stone(draft)
            assert stone.is_fixed is True
            occupancy = get_occupancy(draft)
            occupied_num = int(occupancy.sum())
            assert 4 <= occupied_num <= sandstone_block.FORMATION_VOXEL_CAP
            labels = cc3d.connected_components(occupancy, connectivity=6)
            assert int(labels.max()) == 1

    def test_occupancy_mask_carves_the_block(self, tmp_path):
        mask = np.ones((6, 4, 5), dtype=bool)
        mask[3:, :, 2:] = False
        with make_editor(tmp_path) as editor:
            full = editor.create(sandstone_block.MODULE, {"seed": 4, **QUICK})
            carved = editor.create(
                sandstone_block.MODULE, {"seed": 4, "occupancy": mask, **QUICK}
            )
            occupancy = get_occupancy(carved)
            assert int(occupancy.sum()) < int(get_occupancy(full).sum())
            labels = cc3d.connected_components(occupancy, connectivity=6)
            assert int(labels.max()) == 1

    def test_movable_block_is_dynamic(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(
                sandstone_block.MODULE, {"seed": 4, "is_fixed": False, **QUICK}
            )
            assert get_stone(draft).is_fixed is False
