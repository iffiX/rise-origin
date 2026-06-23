# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests that the basic value types construct from Python sequences (binding coercion)."""

from rise_origin import RBound3ri, RColor3b, RColor3f, RQuat3rf, RVec3rf


def test_vec3_from_sequence():
    vec = RVec3rf((1.0, 2.0, 3.0))
    assert (vec.x, vec.y, vec.z) == (1.0, 2.0, 3.0)


def test_quat_from_sequence():
    quat = RQuat3rf((0.0, 0.0, 0.0, 1.0))
    assert (quat.x, quat.y, quat.z, quat.w) == (0.0, 0.0, 0.0, 1.0)


def test_color3b_from_sequence():
    color = RColor3b((10, 20, 30))
    assert (color.r, color.g, color.b) == (10, 20, 30)


def test_color3f_from_sequence():
    color = RColor3f((0.1, 0.2, 0.3))
    assert abs(color.r - 0.1) < 1e-6


def test_bound3ri_from_sequence():
    # Order is (min_x, min_y, min_z, max_x, max_y, max_z).
    bound = RBound3ri((0, 1, 2, 3, 4, 5))
    assert (bound.min_x, bound.min_y, bound.min_z) == (0, 1, 2)
    assert (bound.max_x, bound.max_y, bound.max_z) == (3, 4, 5)
