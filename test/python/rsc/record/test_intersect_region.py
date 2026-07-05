# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.record.intersect_region."""

import pickle

import pytest

import rise_origin as rs
from rise_origin.rsc.record import (
    IntersectRegionCuboid,
    IntersectRegionCuboidRigidBodyLinked,
    IntersectRegionSphere,
    IntersectRegionSphereRigidBodyLinked,
)


def test_cuboid_fields():
    region = IntersectRegionCuboid(
        "gate",
        position=(2.0, 0.0, 0.5),
        size=(0.2, 2.0, 1.0),
        target_controller_reference="referee",
    )
    assert region.name == "gate"
    assert region.position.x == 2.0
    assert region.size.y == 2.0
    assert region.target_controller_reference == "referee"


def test_sphere_fields():
    region = IntersectRegionSphere("zone", position=(0.0, 0.0, 1.0), radius=0.5)
    assert region.radius == 0.5
    assert region.position.z == 1.0


def test_rb_linked_explicit_target():
    region = IntersectRegionCuboidRigidBodyLinked(
        "z", structure_name="dog", body_sid=0, size=(0.1, 0.1, 0.1)
    )
    assert region.structure_name == "dog"
    assert region.body_sid == 0


def test_sphere_rb_linked_fields():
    region = IntersectRegionSphereRigidBodyLinked(
        "zone",
        structure_name="dog",
        body_sid=0,
        segment_bid=0,
        relative_anchor=(0.1, 0.0, 0.0),
        radius=0.5,
    )
    assert region.structure_name == "dog"
    assert region.body_sid == 0
    assert region.relative_anchor.x == pytest.approx(0.1)
    assert region.radius == 0.5


def test_region_in_record_variant_pickles():
    variant = rs.RSC_RecordVariant(IntersectRegionSphere("z", radius=1.0))
    restored = pickle.loads(pickle.dumps(variant))
    assert restored.value.name == "z"


def test_sphere_rb_linked_in_record_variant_pickles():
    variant = rs.RSC_RecordVariant(
        IntersectRegionSphereRigidBodyLinked("z", structure_name="dog", radius=1.0)
    )
    restored = pickle.loads(pickle.dumps(variant))
    assert restored.value.name == "z"
    assert restored.value.structure_name == "dog"
