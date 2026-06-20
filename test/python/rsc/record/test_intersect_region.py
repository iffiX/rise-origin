# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.record.intersect_region."""

import pickle

import rise_origin as rs
from rise_origin.rsc.entity import StructureVoxel, StructureVoxelBody
from rise_origin.rsc.record import (
    IntersectRegionCuboid,
    IntersectRegionCuboidRigidBodyLinked,
    IntersectRegionSphere,
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


def test_rb_linked_attached_to_resolves():
    body = StructureVoxelBody.box(2, 2, 2, voxel_size=0.02)
    StructureVoxel("dog", bodies=[body])
    region = IntersectRegionCuboidRigidBodyLinked(
        "z", attached_to=body, size=(0.1, 0.1, 0.1)
    )
    assert region.structure_name == "dog"
    assert region.body_sid == 0


def test_region_in_record_variant_pickles():
    variant = rs.RSC_RecordVariant(IntersectRegionSphere("z", radius=1.0))
    restored = pickle.loads(pickle.dumps(variant))
    assert restored.value.name == "z"
