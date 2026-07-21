# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""RSC reference sharing: a Python object's identity is one C++ shared node, end to end.

Assigning the same wrapper into two places shares one C++ object, reads hand back that same
wrapper, mutation aliases, and the archive stores it once. Distinct constructions stay distinct,
and deep_copy still isolates. Floats are compared with approx since fields are single precision.
"""

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import rise_origin as rs
from _serialization_common import body, pool_type_names, state
from rise_origin import rsc


def _voxel_body_count(node):
    return pool_type_names(state(node)).count("RSC_StructureVoxelBody")


# ======================================================================================
# Object identity is preserved across assignment and read-back
# ======================================================================================
def test_read_back_returns_the_same_wrapper():
    b = body(3)
    s = rsc.entity.StructureVoxel("s", material_references=["m"], bodies=[b])
    assert s.bodies[0] is b


def test_same_body_in_two_structures_is_one_object():
    b = body(3)
    s1 = rsc.entity.StructureVoxel("a", material_references=["m"], bodies=[b])
    s2 = rsc.entity.StructureVoxel("b", material_references=["m"], bodies=[b])
    assert s1.bodies[0] is b
    assert s2.bodies[0] is b
    assert s1.bodies[0] is s2.bodies[0]
    # Mutation through one is seen through the other and the original.
    s1.bodies[0].voxel_size = 0.7
    assert s2.bodies[0].voxel_size == pytest.approx(0.7)
    assert b.voxel_size == pytest.approx(0.7)


def test_append_shares():
    b = body(2)
    s = rsc.entity.StructureVoxel("s", material_references=["m"])
    s.bodies.append(b)
    assert s.bodies[0] is b
    b.voxel_size = 0.3
    assert s.bodies[0].voxel_size == pytest.approx(0.3)


def test_setitem_replaces_by_reference():
    keep, other = body(2), body(2, voxel_size=0.05)
    s = rsc.entity.StructureVoxel("s", material_references=["m"], bodies=[keep])
    s.bodies[0] = other
    assert s.bodies[0] is other
    other.voxel_size = 0.4
    assert s.bodies[0].voxel_size == pytest.approx(0.4)


def test_insert_shares_and_orders():
    first, inserted = body(2), body(2, voxel_size=0.05)
    s = rsc.entity.StructureVoxel("s", material_references=["m"], bodies=[first])
    s.bodies.insert(0, inserted)
    assert s.bodies[0] is inserted
    assert s.bodies[1] is first


# ======================================================================================
# Distinct constructions stay distinct
# ======================================================================================
def test_distinct_bodies_do_not_alias():
    s1 = rsc.entity.StructureVoxel("a", material_references=["m"], bodies=[body(3)])
    s2 = rsc.entity.StructureVoxel("b", material_references=["m"], bodies=[body(3)])
    assert s1.bodies[0] is not s2.bodies[0]
    s1.bodies[0].voxel_size = 0.7
    assert s2.bodies[0].voxel_size != pytest.approx(0.7)


# ======================================================================================
# Embedded RSCRef members share by identity too
# ======================================================================================
def test_record_read_back_identity():
    s = rsc.entity.StructureVoxel("s", material_references=["m"])
    assert s.record is s.record


def test_assigned_record_is_shared():
    s = rsc.entity.StructureVoxel("s", material_references=["m"])
    record = rsc.entity.StructureVoxelRecord()
    s.record = record
    assert s.record is record


# ======================================================================================
# Copy semantics are unchanged by the holder
# ======================================================================================
def test_shallow_copy_shares_children():
    b = body(3)
    s = rsc.entity.StructureVoxel("s", material_references=["m"], bodies=[b])
    clone = s.shallow_copy(name="s2")
    assert clone.bodies[0] is s.bodies[0]


def test_deep_copy_isolates_children():
    b = body(3)
    s = rsc.entity.StructureVoxel("s", material_references=["m"], bodies=[b])
    clone = s.deep_copy(name="s2")
    assert clone.bodies[0] is not s.bodies[0]
    clone.bodies[0].voxel_size = 0.9
    assert s.bodies[0].voxel_size == pytest.approx(0.01)


# ======================================================================================
# Sharing reaches the archive as one pooled node
# ======================================================================================
def test_shared_body_is_pooled_once():
    shared = body(4)
    entities = [
        rsc.entity.StructureVoxel(f"h{i}", material_references=["m"], bodies=[shared])
        for i in range(5)
    ]
    world = rsc.World("shared", essences=[rsc.essence.Material("m")], entities=entities)
    assert _voxel_body_count(world) == 1


def test_distinct_bodies_are_pooled_each():
    entities = [
        rsc.entity.StructureVoxel(f"u{i}", material_references=["m"], bodies=[body(4)])
        for i in range(5)
    ]
    world = rsc.World("unique", essences=[rsc.essence.Material("m")], entities=entities)
    assert _voxel_body_count(world) == 5
