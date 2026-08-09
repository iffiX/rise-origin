# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the cloud material package: white and rainy shells, order, purity."""

import hashlib
import os
import tempfile

from rise_origin import create as cr
from rise_origin.create.modules.material.nature import cloud
from rise_origin.rsc import essence


def make_editor(tmp_path, name="world"):
    return cr.Editor(
        name,
        session_dir=str(tmp_path / "session"),
        cache=cr.RealizationCache(str(tmp_path / "cache")),
    )


def make_optics_by_name(draft):
    return {
        rsc.name: rsc
        for rsc in draft.rscs
        if isinstance(rsc, essence.MaterialOpticalPropertySimpleMicrofacet)
    }


def make_rsc_digests(draft):
    digests = []
    for node in draft.iterate_subtree_nodes():
        for rsc in node.rscs:
            handle, path = tempfile.mkstemp(suffix=".rscb")
            os.close(handle)
            try:
                rsc.save(path)
                with open(path, "rb") as saved:
                    digests.append(hashlib.sha256(saved.read()).hexdigest())
            finally:
                os.remove(path)
    return digests


class TestCloudPackage:
    def test_the_package_releases_the_white_and_rainy_materials(self, tmp_path):
        with make_editor(tmp_path) as editor:
            draft = editor.create(cloud.MODULE)
            names = [rsc.name for rsc in draft.rscs]
            assert cloud.CLOUD in names
            for reference in cloud.CLOUD_RAIN_LAYER_REFERENCES:
                assert reference in names
            references = [cloud.CLOUD_OPTICAL_REFERENCES]
            references += list(cloud.CLOUD_RAIN_LAYER_OPTICAL_REFERENCES)
            for one in references:
                for reference in one:
                    assert reference in names

    def test_the_shell_counts_agree(self):
        assert len(cloud.CLOUD_OPTICAL_REFERENCES) == cloud.CLOUD_LAYER_COUNT == 1
        assert cloud.CLOUD_RAIN_LAYER_COUNT == len(cloud.CLOUD_RAIN_LAYERS)
        assert len(cloud.CLOUD_RAIN_LAYER_REFERENCES) == cloud.CLOUD_RAIN_LAYER_COUNT
        assert all(len(one) == 1 for one in cloud.CLOUD_RAIN_LAYER_OPTICAL_REFERENCES)

    def test_the_white_material_is_opaque(self, tmp_path):
        with make_editor(tmp_path) as editor:
            optics = make_optics_by_name(editor.create(cloud.MODULE))
            white = optics[cloud.CLOUD_OPTICAL_REFERENCES[0]]
            assert not white.is_transparent
            assert white.emission_radiance == 0.0

    def test_the_rainy_shells_grade_to_an_opaque_core(self, tmp_path):
        with make_editor(tmp_path) as editor:
            optics = make_optics_by_name(editor.create(cloud.MODULE))
            for (
                name,
                _albedo,
                _roughness,
                is_transparent,
                _ior,
            ) in cloud.CLOUD_RAIN_LAYERS:
                optic = optics[f"{cloud.MODULE}:{name}#optic"]
                assert optic.is_transparent == is_transparent
                assert optic.emission_radiance == 0.0
            core = cloud.CLOUD_RAIN_LAYERS[-1]
            assert not core[3]

    def test_a_double_build_is_byte_identical(self, tmp_path):
        with make_editor(tmp_path / "a") as first:
            one = make_rsc_digests(first.create(cloud.MODULE))
        with make_editor(tmp_path / "b") as second:
            two = make_rsc_digests(second.create(cloud.MODULE))
        assert one == two
