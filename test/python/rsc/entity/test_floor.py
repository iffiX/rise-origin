# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.entity.floor."""

import numpy as np
import pytest

from rise_origin.rsc.entity import FloorArray, FloorFlat


def test_floor_flat_fields():
    floor = FloorFlat("ground", height=0.1, material_reference="dirt")
    assert floor.height == pytest.approx(0.1)
    assert floor.material_reference == "dirt"


def test_floor_array_from_function_shape():
    floor = FloorArray.from_function(
        lambda x, y: 0.0 * x, x_size=4.0, y_size=4.0, x_values=8, y_values=6
    )
    assert (floor.x_values, floor.y_values) == (8, 6)
    assert len(floor.height) == 48


def test_floor_array_from_heightmap_infers_shape():
    grid = np.zeros((6, 8))  # (y_values, x_values)
    floor = FloorArray.from_heightmap(grid, x_size=4.0, y_size=4.0)
    assert (floor.x_values, floor.y_values) == (8, 6)
    assert len(floor.height) == 48
