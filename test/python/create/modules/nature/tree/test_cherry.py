# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the cherry: a quick build inside its caps."""

import numpy as np

from rise_origin import create as cr
from rise_origin.create.modules.nature.tree import cherry as kind

QUICK = {"height": 3.5, "season": "spring"}
QUICK.setdefault("trunk_voxel_size", "5cm")
QUICK.setdefault("canopy_voxel_size", "10cm")


def make_editor(tmp_path):
    return cr.Editor(
        "world",
        session_dir=str(tmp_path / "session"),
        cache=cr.RealizationCache(str(tmp_path / "cache")),
    )


def gather_leaf_mids(tree):
    mids = set()
    for body in tree.bodies:
        sids = np.asarray(body.material_reference_sid)
        used = sids == 1
        if used.any():
            values = np.asarray(body.material_optical_reference_mid)
            mids |= set(values[used].tolist())
    return mids


class TestCherry:
    def test_a_quick_build_stays_inside_the_caps(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(kind.MODULE, QUICK)
            tree = draft.rscs[0]
            assert 0 < len(tree.bodies) < kind.BODY_CAP
            wood = sum(
                int((np.asarray(body.material_reference_sid) == 0).sum())
                for body in tree.bodies
            )
            assert wood > 0

    def test_spring_wears_blossom_mids(self, tmp_path):
        from rise_origin.create.modules.material.nature import plant

        with make_editor(tmp_path) as editor:
            draft = editor.create(kind.MODULE, QUICK)
            mids = gather_leaf_mids(draft.rscs[0])
            assert mids
            assert mids <= set(plant.BLOSSOM_MIDS)
