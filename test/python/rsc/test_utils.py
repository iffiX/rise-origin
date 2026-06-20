# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc._utils tuple coercers."""

from rise_origin.rsc import _utils


def test_vec3_coercion_and_passthrough():
    vec = _utils._vec3rf((1.0, 2.0, 3.0))
    assert (vec.x, vec.y, vec.z) == (1.0, 2.0, 3.0)
    assert _utils._vec3rf(vec) is vec


def test_quat_coercion():
    quat = _utils._quat3rf((0.0, 0.0, 0.0, 1.0))
    assert (quat.x, quat.y, quat.z, quat.w) == (0.0, 0.0, 0.0, 1.0)


def test_color3b_coercion():
    color = _utils._color3b((10, 20, 30))
    assert (color.r, color.g, color.b) == (10, 20, 30)


def test_color4f_coercion():
    color = _utils._color4f((0.1, 0.2, 0.3, 1.0))
    assert abs(color.a - 1.0) < 1e-6


def test_bound3ri_coercion():
    # Order is (min_x, min_y, min_z, max_x, max_y, max_z).
    bound = _utils._bound3ri((0, 1, 2, 3, 4, 5))
    assert (bound.min_x, bound.min_y, bound.min_z) == (0, 1, 2)
    assert (bound.max_x, bound.max_y, bound.max_z) == (3, 4, 5)
