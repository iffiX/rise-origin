# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the draft plan view: boxes, clusters, text, and the image."""

import numpy as np
import pytest
from PIL import Image

from rise_origin import create as cr


class FakeAccess:
    """The minimal access wall a detached tree needs."""

    def __init__(self):
        self._next_uid = 0

    def allocate_uid(self):
        self._next_uid += 1
        return self._next_uid

    def is_session_node(self, node):
        return False


@pytest.fixture
def access():
    return FakeAccess()


def make_box_draft(access, name, size=(1.0, 1.0, 1.0), material_class="static"):
    return cr.Draft(access, name, boxes=[cr.Box((0, 0, 0), size, material_class)])


class TestPlanView:
    def make_world(self, access):
        world = cr.Draft(access, "world")
        world.insert(
            make_box_draft(access, "house", size=(6, 4, 3)), cr.Pose((0, 0, 0))
        )
        world.insert(
            make_box_draft(access, "shed", size=(2, 2, 2)), cr.Pose((10, 0, 0))
        )
        return world

    def test_boxes_and_bounds(self, access):
        view = self.make_world(access).get_plan_view((-1, -1, 0), (13, 5, 2.5))
        assert len(view.boxes) == 2
        bounds = view.get_bounds()
        assert np.allclose([bounds.min.x, bounds.max.x], [0, 12], atol=1e-6)

    def test_cut_flag_reads_the_band_top(self, access):
        view = self.make_world(access).get_plan_view((-1, -1, 0), (13, 5, 2.5))
        cut = {box.node.name: box.is_cut for box in view.boxes}
        assert cut["house"] and not cut["shed"]

    def test_clusters_split_disjoint_content(self, access):
        view = self.make_world(access).get_plan_view((-1, -1, 0), (13, 5, 2.5))
        clusters = view.get_sub_clusters()
        assert len(clusters) == 2
        assert [c.nodes[0].name for c in clusters] == ["house", "shed"]

    def test_descriptions_are_deterministic_sentences(self, access):
        view = self.make_world(access).get_plan_view((-1, -1, 0), (13, 5, 2.5))
        text = view.to_description()
        assert "house: 6.00 by 4.00 m" in text
        assert text == view.to_description()
        clusters = view.to_sub_clusters_description()
        assert "cluster 1" in clusters and "shed" in clusters

    def test_to_image_returns_a_pil_image(self, access):
        view = self.make_world(access).get_plan_view((-1, -1, 0), (13, 5, 2.5))
        image = view.to_image(width=400)
        assert isinstance(image, Image.Image)
        assert image.width == 400 and image.height > 0
        colors = set(image.get_flattened_data())
        assert len(colors) > 2
