# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for shallow_copy / deep_copy on RSC types."""

import numpy as np
import pytest

from rise_origin import rsc


def _body():
    occ = np.zeros((2, 1, 1), dtype=bool)
    occ[0, 0, 0] = True
    occ[1, 0, 0] = True
    return rsc.entity.StructureVoxelBody.from_grid(
        occ, material_sid=0, segment_sid=0, voxel_size=0.01
    )


def _structure():
    return rsc.entity.StructureVoxel("hand", material_references=["m"], bodies=[_body()])


def test_shallow_copy_applies_overrides():
    s = _structure()
    s2 = s.shallow_copy(name="hand_2", origin_position=(3.0, 0.0, 0.0))
    assert s2.name == "hand_2"
    assert s2.origin_position.x == pytest.approx(3.0)
    # The original is untouched by the override.
    assert s.name == "hand"
    assert s.origin_position.x == pytest.approx(0.0)


def test_shallow_copy_shares_bodies():
    s = _structure()
    s2 = s.shallow_copy(name="hand_2")
    # A shallow copy shares the body node, so mutating one is seen by the other.
    s2.bodies[0].voxel_size = 0.5
    assert s.bodies[0].voxel_size == pytest.approx(0.5)


def test_deep_copy_is_independent():
    s = _structure()
    s3 = s.deep_copy(name="hand_3")
    assert s3.name == "hand_3"
    s3.bodies[0].voxel_size = 0.5
    # A deep copy owns its own bodies, so the original keeps its value.
    assert s.bodies[0].voxel_size == pytest.approx(0.01)


def test_unknown_override_raises():
    s = _structure()
    with pytest.raises(TypeError):
        s.shallow_copy(does_not_exist=1)


def test_world_deep_copy_preserves_and_isolates():
    s = _structure()
    hands = [s.shallow_copy(name=f"hand_{i}", origin_position=(3.0 * i, 0.0, 0.0)) for i in range(3)]
    world = rsc.World("w", essences=[rsc.essence.Material("m")], entities=hands)
    assert [e.name for e in world.entities] == ["hand_0", "hand_1", "hand_2"]

    clone = world.deep_copy()
    assert [e.name for e in clone.entities] == ["hand_0", "hand_1", "hand_2"]
    clone.entities[0].origin_position = (99.0, 0.0, 0.0)
    assert world.entities[0].origin_position.x == pytest.approx(0.0)
    assert clone.entities[0].origin_position.x == pytest.approx(99.0)
