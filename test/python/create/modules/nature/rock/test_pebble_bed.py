# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the pebble bed: stones, base, budget, and errors."""

import numpy as np
import pytest

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import mineral
from rise_origin.create.modules.nature.rock import pebble_bed
from rise_origin.create.modules.nature.rock.build import (
    BED_DYNAMIC_CAP,
    MIN_STONE_VOXEL_NUM,
)

RS_NULL_SID = np.iinfo(np.uint32).max

QUICK = {
    "width": 0.5,
    "depth": 0.5,
    "thickness": 0.14,
    "stone_size": 0.1,
    "voxel_size": "1cm",
    "wear_subdivision": 2,
}


def make_editor(tmp_path):
    return cr.Editor(
        "world",
        session_dir=str(tmp_path / "session"),
        cache=cr.RealizationCache(str(tmp_path / "cache")),
    )


def get_structures(draft):
    found = {}
    for node in draft.iterate_subtree_nodes():
        for rsc in node.rscs:
            if hasattr(rsc, "bodies"):
                found[rsc.name] = rsc
    if not found:
        raise AssertionError("no structure in the draft")
    return found


def get_stone_sizes(structure):
    """Return the voxel count of every rigid segment of a structure."""
    body = structure.bodies[0]
    used = np.asarray(body.material_reference_sid) != RS_NULL_SID
    segments = np.asarray(body.segment_bid)[used]
    return np.bincount(segments.astype(np.int64))


class TestPebbleBed:
    def test_default_build_stands_stones_on_a_still_base(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(pebble_bed.MODULE, {"seed": 3, **QUICK})
            found = get_structures(draft)
            assert set(found) == {"pebble_bed", "pebble_bed_base"}
            stones = found["pebble_bed"]
            base = found["pebble_bed_base"]
            assert stones.is_fixed is False
            assert base.is_fixed is True
            assert stones.material_references == [mineral.RIVER_STONE]
            # Every stone is its own rigid segment, and the base is one.
            sizes = get_stone_sizes(stones)
            assert len(sizes) > 1
            assert len(sizes) <= BED_DYNAMIC_CAP
            assert int(sizes.min()) >= MIN_STONE_VOXEL_NUM
            assert len(get_stone_sizes(base)) == 1

    def test_the_bed_holds_its_voxel_budget(self, tmp_path):
        # Built at the real defaults, since that is what the cap states.
        with make_editor(tmp_path) as editor:
            draft = editor.create(pebble_bed.MODULE, {"seed": 3})
            total = 0
            for structure in get_structures(draft).values():
                used = (
                    np.asarray(structure.bodies[0].material_reference_sid)
                    != RS_NULL_SID
                )
                total += int(used.sum())
            assert total > MIN_STONE_VOXEL_NUM
            assert total <= pebble_bed.BED_VOXEL_CAP

    def test_a_still_bed_wakes_no_stone(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(
                pebble_bed.MODULE, {"seed": 3, "moving_depth_share": 0.0, **QUICK}
            )
            found = get_structures(draft)
            assert set(found) == {"pebble_bed_base"}
            assert found["pebble_bed_base"].is_fixed is True

    def test_a_deeper_share_wakes_more_stones(self, tmp_path):
        with make_editor(tmp_path) as editor:
            shallow = editor.create(
                pebble_bed.MODULE, {"seed": 3, "moving_depth_share": 0.15, **QUICK}
            )
            deep = editor.create(
                pebble_bed.MODULE, {"seed": 3, "moving_depth_share": 0.9, **QUICK}
            )
            assert len(get_stone_sizes(get_structures(deep)["pebble_bed"])) > len(
                get_stone_sizes(get_structures(shallow)["pebble_bed"])
            )

    def test_the_bed_is_one_stone_over_a_run_of_shades(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(pebble_bed.MODULE, {"seed": 3, **QUICK})
            used = set()
            shades = set()
            for structure in get_structures(draft).values():
                body = structure.bodies[0]
                sids = np.asarray(body.material_reference_sid)
                held = sids != RS_NULL_SID
                used.update(np.unique(sids[held]).tolist())
                mids = np.asarray(body.material_optical_reference_mid)[held]
                shades.update(np.unique(mids).tolist())
            assert used == {0}
            assert len(shades) > 2

    def test_a_darker_tone_darkens_the_bed(self, tmp_path):
        with make_editor(tmp_path) as editor:
            pale = editor.create(pebble_bed.MODULE, {"seed": 3, "tone": 0.0, **QUICK})
            dark = editor.create(pebble_bed.MODULE, {"seed": 3, "tone": 1.0, **QUICK})
            assert _make_mean_shade(dark) > _make_mean_shade(pale)

    def test_weathering_wears_the_bed_airier(self, tmp_path):
        with make_editor(tmp_path) as editor:
            fresh = editor.create(
                pebble_bed.MODULE, {"seed": 5, "weathering": 0.0, **QUICK}
            )
            worn = editor.create(
                pebble_bed.MODULE, {"seed": 5, "weathering": 1.0, **QUICK}
            )
            assert _count_voxels(worn) < _count_voxels(fresh)

    def test_an_occupancy_mask_carves_the_bed(self, tmp_path):
        mask = np.ones((4, 4, 2), dtype=bool)
        mask[:2, :, :] = False
        with make_editor(tmp_path) as editor:
            whole = editor.create(pebble_bed.MODULE, {"seed": 3, **QUICK})
            carved = editor.create(
                pebble_bed.MODULE, {"seed": 3, "occupancy": mask, **QUICK}
            )
            assert _count_voxels(carved) < 0.75 * _count_voxels(whole)

    def test_too_coarse_dials_fail_at_draft_time(self, tmp_path):
        with make_editor(tmp_path) as editor:
            with pytest.raises(Exception):
                editor.create(
                    pebble_bed.MODULE,
                    {
                        "seed": 3,
                        "stone_size": 0.03,
                        "voxel_size": "10cm",
                        "wear_subdivision": 1,
                    },
                )


def _make_mean_shade(draft):
    """Return the mean optical mid over the whole bed, dark end high."""
    total = 0.0
    count = 0
    for structure in get_structures(draft).values():
        body = structure.bodies[0]
        held = np.asarray(body.material_reference_sid) != RS_NULL_SID
        mids = np.asarray(body.material_optical_reference_mid)[held]
        total += float(mids.sum())
        count += mids.size
    return total / max(count, 1)


def _count_voxels(draft):
    """Return how many voxels the whole bed keeps."""
    total = 0
    for structure in get_structures(draft).values():
        sids = np.asarray(structure.bodies[0].material_reference_sid)
        total += int((sids != RS_NULL_SID).sum())
    return total
