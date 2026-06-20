# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.resource.texture."""

import pickle

import pytest

import rise_origin as rs
from rise_origin.rsc.resource import TextureConstant, TextureGrid


def test_texture_constant_value():
    texture = TextureConstant("sky", value=(0.5, 0.6, 0.7, 1.0))
    assert texture.name == "sky"
    assert abs(texture.value.r - 0.5) < 1e-6


def test_texture_grid_defaults():
    grid = TextureGrid("ground", line_color=(0.2, 0.2, 0.2, 1.0))
    assert grid.is_meters is True
    assert grid.x_spacing == pytest.approx(1.0)
    assert grid.line_width == pytest.approx(0.05)


def test_texture_in_resource_variant_pickles():
    variant = rs.RSC_ResourceVariant(TextureConstant("c", value=(0.0, 0.0, 0.0, 1.0)))
    restored = pickle.loads(pickle.dumps(variant))
    assert restored.value.name == "c"
