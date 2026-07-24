# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the baobab: a quick build inside its caps."""

import numpy as np

from rise_origin import create as cr
from rise_origin.create.modules.nature.tree import baobab as kind

QUICK = {"height": 5.0}
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


class TestBaobab:
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
