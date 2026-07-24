# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the cumulus: a small opaque white build."""

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import cloud
from rise_origin.create.modules.nature.cloud import cumulus

# The smallest allowed dials, so the build is tiny and fast.
SMALL = {"size": 400.0, "voxel_size": "50m"}


def make_editor(tmp_path):
    return cr.Editor(
        "world",
        session_dir=str(tmp_path / "session"),
        cache=cr.RealizationCache(str(tmp_path / "cache")),
    )


class TestCumulus:
    def test_builds_an_opaque_white_cloud(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(cumulus.MODULE, SMALL)
            struct = draft.rscs[0]
            assert list(struct.material_references) == [cloud.CLOUD]
            assert struct.is_fixed
            assert draft.requires[0].material_reference == "cloud"
