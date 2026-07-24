# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the pose: composition, inversion, and the json round trip."""

import math

import numpy as np

from rise_origin import RQuat3rf, RVec3rf
from rise_origin import create as cr


def make_quat_about_z(angle_rad: float) -> RQuat3rf:
    return RQuat3rf.unit().from_rotation_vector(RVec3rf(0.0, 0.0, angle_rad))


class TestPose:
    def test_ctor_coerces_sequences(self):
        pose = cr.Pose((1, 2, 3), (0, 0, 0, 1))
        assert isinstance(pose.relative_position, RVec3rf)
        assert isinstance(pose.orientation, RQuat3rf)
        assert pose.relative_position == RVec3rf(1, 2, 3)

    def test_defaults_are_identity(self):
        pose = cr.Pose()
        assert pose.relative_position == RVec3rf.zero()
        assert pose.apply((1, 2, 3)) == RVec3rf(1, 2, 3)

    def test_compose_matches_manual_math(self):
        parent = cr.Pose((1.0, 0.0, 0.0), make_quat_about_z(math.pi / 2))
        child = cr.Pose((1.0, 0.0, 0.0), make_quat_about_z(math.pi / 2))
        composed = parent.compose(child)
        position = composed.relative_position
        assert np.allclose(
            [position.x, position.y, position.z], [1.0, 1.0, 0.0], atol=1e-6
        )
        rotated = composed.orientation.rotate_vec3d(RVec3rf(1.0, 0.0, 0.0))
        assert np.allclose(
            [rotated.x, rotated.y, rotated.z], [-1.0, 0.0, 0.0], atol=1e-6
        )

    def test_invert_round_trips(self):
        pose = cr.Pose((0.4, -2.0, 1.5), make_quat_about_z(0.7))
        round_trip = pose.compose(pose.invert())
        position = round_trip.relative_position
        assert np.allclose(
            [position.x, position.y, position.z], [0.0, 0.0, 0.0], atol=1e-6
        )
        point = RVec3rf(3.0, 1.0, -2.0)
        back = pose.invert().apply(pose.apply(point))
        assert np.allclose([back.x, back.y, back.z], [3.0, 1.0, -2.0], atol=1e-5)

    def test_json_round_trip(self):
        pose = cr.Pose((0.4, -2.0, 1.5), make_quat_about_z(0.7))
        loaded = cr.Pose.from_json(pose.to_json())
        assert loaded == pose

    def test_identity_survives_invert_compose(self):
        relative = cr.Pose().invert().compose(cr.Pose())
        assert relative.is_identity()
        assert not cr.Pose((0.1, 0.0, 0.0)).is_identity()
        assert not cr.Pose((0, 0, 0), make_quat_about_z(0.2)).is_identity()
