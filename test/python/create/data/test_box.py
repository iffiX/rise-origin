# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the cover box: corners, posed bounds, and the json round trip."""

import math

import numpy as np

from rise_origin import RQuat3rf, RVec3rf
from rise_origin import create as cr


def make_quat_about_z(angle_rad: float) -> RQuat3rf:
    return RQuat3rf.unit().from_rotation_vector(RVec3rf(0.0, 0.0, angle_rad))


class TestBox:
    def test_aabb_in_root_of_rotated_box(self):
        box = cr.Box((0, 0, 0), (2, 1, 1))
        pose = cr.Pose((10.0, 0.0, 0.0), make_quat_about_z(math.pi / 2))
        lo, hi = box.get_aabb_in_root(pose)
        assert np.allclose(lo, [9.0, 0.0, 0.0], atol=1e-6)
        assert np.allclose(hi, [10.0, 2.0, 1.0], atol=1e-6)

    def test_corners_count_and_extent(self):
        corners = cr.Box((-1, -2, -3), (1, 2, 3)).get_corners()
        assert corners.shape == (8, 3)
        assert np.allclose(corners.min(axis=0), [-1, -2, -3])
        assert np.allclose(corners.max(axis=0), [1, 2, 3])

    def test_json_round_trip(self):
        box = cr.Box((0, 0, 0), (1, 2, 3), "mixed")
        loaded = cr.Box.from_json(box.to_json())
        assert loaded == box
