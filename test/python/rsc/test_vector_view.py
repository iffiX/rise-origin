# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Value-vector members are uniform write-through list views (rspyBindVector).

A string vector, a value-struct vector, and a nested string vector all behave like a live Python
list: append/extend/insert/setitem/delitem/clear mutate the object in place, whole-list assignment
replaces, a nested element yields a recursive subview whose mutation also writes through, and a
bound value element hands back a reference so field writes land on the stored element.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import rise_origin as rs
from rise_origin import rsc


# ======================================================================================
# String vector (material_references)
# ======================================================================================
def test_string_vector_append_writes_through():
    s = rs.RSC_StructureVoxel()
    s.material_references.append("a")
    s.material_references.append("b")
    assert list(s.material_references) == ["a", "b"]


def test_string_vector_extend_and_insert():
    s = rs.RSC_StructureVoxel()
    s.material_references.extend(["a", "b"])
    s.material_references.insert(0, "z")
    assert list(s.material_references) == ["z", "a", "b"]


def test_string_vector_setitem_delitem_clear():
    s = rs.RSC_StructureVoxel()
    s.material_references = ["a", "b", "c"]
    s.material_references[1] = "B"
    assert list(s.material_references) == ["a", "B", "c"]
    del s.material_references[0]
    assert list(s.material_references) == ["B", "c"]
    s.material_references.clear()
    assert list(s.material_references) == []


def test_string_vector_whole_list_assignment():
    s = rs.RSC_StructureVoxel()
    s.material_references = ["x", "y", "z"]
    assert list(s.material_references) == ["x", "y", "z"]
    s.material_references = ("p", "q")
    assert list(s.material_references) == ["p", "q"]


def test_string_vector_len_iter_eq_repr():
    s = rs.RSC_StructureVoxel()
    s.material_references = ["a", "b"]
    assert len(s.material_references) == 2
    assert [x for x in s.material_references] == ["a", "b"]
    assert s.material_references == ["a", "b"]
    assert repr(s.material_references) == "['a', 'b']"


# ======================================================================================
# Nested string vector (material_optical_references) with recursive subview
# ======================================================================================
def test_nested_inner_mutation_writes_through():
    s = rs.RSC_StructureVoxel()
    s.material_optical_references = [["a"], ["b", "c"]]
    s.material_optical_references[0].append("z")
    s.material_optical_references[1][0] = "B"
    assert [list(inner) for inner in s.material_optical_references] == [["a", "z"], ["B", "c"]]


def test_nested_element_is_a_recursive_view():
    s = rs.RSC_StructureVoxel()
    s.material_optical_references = [["a"]]
    inner = s.material_optical_references[0]
    assert type(inner).__name__ == "RSVectorView"
    assert list(inner) == ["a"]


def test_nested_whole_assignment():
    s = rs.RSC_StructureVoxel()
    s.material_optical_references = [["a", "b"], []]
    assert [list(inner) for inner in s.material_optical_references] == [["a", "b"], []]


# ======================================================================================
# Value-struct vector (ray_directions) hands back element references
# ======================================================================================
def test_value_struct_vector_append_and_len():
    r = rs.RSC_RaySensorFree()
    r.ray_directions = []
    r.ray_directions.append(rs.RVec3rf(1.0, 0.0, 0.0))
    r.ray_directions.append(rs.RVec3rf(0.0, 1.0, 0.0))
    assert len(r.ray_directions) == 2


def test_value_struct_element_field_writes_through():
    r = rs.RSC_RaySensorFree()
    r.ray_directions = [rs.RVec3rf(1.0, 0.0, 0.0)]
    r.ray_directions[0].x = 5.0
    assert r.ray_directions[0].x == pytest.approx(5.0)


def test_value_struct_whole_assignment():
    r = rs.RSC_RaySensorFree()
    r.ray_directions = [rs.RVec3rf(1.0, 2.0, 3.0), rs.RVec3rf(4.0, 5.0, 6.0)]
    got = [(v.x, v.y, v.z) for v in r.ray_directions]
    assert got == pytest.approx([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])


# ======================================================================================
# Errors
# ======================================================================================
def test_out_of_range_index_raises():
    s = rs.RSC_StructureVoxel()
    s.material_references = ["a"]
    with pytest.raises(IndexError):
        _ = s.material_references[5]


def test_wrong_element_type_raises():
    r = rs.RSC_RaySensorFree()
    with pytest.raises(Exception):
        r.ray_directions = ["not a vector"]
