# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.entity.structure_voxel: construction, generators, refs, importers."""

import pickle

import numpy as np
import pytest

import rise_origin as rs
from rise_origin import rsc
from rise_origin.rsc.entity import (
    CameraPerspectiveRigidBodyLinked,
    StructureVoxel,
    StructureVoxelBody,
    StructureVoxelConstraint,
)

HINGE = rs.RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT
VS = 0.02


def test_body_box_fills_full_lattice():
    body = StructureVoxelBody.box(4, 3, 2, voxel_size=VS)
    assert (body.x_voxels, body.y_voxels, body.z_voxels) == (4, 3, 2)
    assert len(body.material_reference_sid) == 24
    assert all(m == 0 for m in body.material_reference_sid)


def test_body_sphere_fills_about_half_its_box():
    body = StructureVoxelBody.sphere(6, voxel_size=VS)
    filled = sum(1 for m in body.material_reference_sid if m != rs.RS_NULL_INDEX)
    total = body.x_voxels * body.y_voxels * body.z_voxels
    assert 0.45 < filled / total < 0.58


def test_body_capsule_spans_caps():
    body = StructureVoxelBody.capsule(2, 8, axis="z", voxel_size=VS)
    assert body.z_voxels == 8 + 2 * 2


def test_body_from_sdf_torus_is_nonempty():
    def torus(x, y, z):
        return (np.sqrt(x * x + y * y) - 0.05) ** 2 + z * z - 0.02**2

    body = StructureVoxelBody.from_sdf(
        torus, bounds=((-0.08, -0.08, -0.03), (0.08, 0.08, 0.03)), voxel_size=VS
    )
    assert any(m != rs.RS_NULL_INDEX for m in body.material_reference_sid)


def test_constraint_body_refs_resolve_by_identity():
    torso = StructureVoxelBody.box(6, 4, 3, voxel_size=VS)
    leg = StructureVoxelBody.cylinder(2, 6, voxel_size=VS)
    hip = StructureVoxelConstraint(HINGE, a=torso, b=leg)
    structure = StructureVoxel("dog", bodies=[torso, leg], constraints=[hip])
    assert structure.constraints[0].a_body_sid == 0
    assert structure.constraints[0].b_body_sid == 1
    assert torso._body_sid == 0
    assert torso._structure_name == "dog"


def test_constraint_segment_tuple_ref():
    first = StructureVoxelBody.box(2, 2, 2, voxel_size=VS)
    second = StructureVoxelBody.box(2, 2, 2, voxel_size=VS)
    joint = StructureVoxelConstraint(HINGE, a=(first, 0), b=(second, 1))
    structure = StructureVoxel("s", bodies=[first, second], constraints=[joint])
    assert structure.constraints[0].b_segment_bid == 1


def test_constraint_explicit_int_ids():
    joint = StructureVoxelConstraint(HINGE, a_body_sid=0, b_body_sid=2)
    assert joint.a_body_sid == 0 and joint.b_body_sid == 2


def test_unattached_body_link_raises():
    body = StructureVoxelBody.box(2, 2, 2, voxel_size=VS)
    with pytest.raises(ValueError):
        CameraPerspectiveRigidBodyLinked("eye", attached_to=body)


def test_resolve_size_scalar_dict_and_missing():
    assert StructureVoxel._resolve_size(0.01, "x") == 0.01
    assert StructureVoxel._resolve_size({"a": 0.02, "*": 0.05}, "a") == 0.02
    assert StructureVoxel._resolve_size({"a": 0.02, "*": 0.05}, "z") == 0.05
    with pytest.raises(KeyError):
        StructureVoxel._resolve_size({"a": 0.02}, "z")


def test_structure_in_entity_variant_pickles():
    structure = StructureVoxel(
        "s", bodies=[StructureVoxelBody.box(2, 2, 2, voxel_size=VS)]
    )
    restored = pickle.loads(pickle.dumps(rs.RSC_EntityVariant(structure)))
    assert restored.value.name == "s"


@pytest.mark.voxel
def test_from_mesh_single_body(box_mesh):
    structure = StructureVoxel.from_mesh(box_mesh, voxel_size=VS)
    assert len(structure.bodies) == 1
    assert structure.bodies[0].x_voxels > 0


@pytest.mark.voxel
def test_from_mesh_robust_to_non_watertight(box_mesh):
    holed = box_mesh.copy()
    holed.faces = holed.faces[:-2]  # drop two faces -> non-watertight
    structure = StructureVoxel.from_mesh(holed, voxel_size=VS)
    filled = sum(
        1 for m in structure.bodies[0].material_reference_sid if m != rs.RS_NULL_INDEX
    )
    assert filled > 0


@pytest.mark.importer
def test_from_urdf(urdf_path):
    structure = StructureVoxel.from_urdf(urdf_path, voxel_size=VS)
    assert len(structure.bodies) == 2
    assert len(structure.constraints) == 1
    assert structure.signal.hinge_signal_num == 1


@pytest.mark.importer
def test_from_urdf_per_link_voxel_size(urdf_path):
    structure = StructureVoxel.from_urdf(
        urdf_path, voxel_size={"upper": 0.01, "*": 0.04}
    )
    assert sorted(round(b.voxel_size, 3) for b in structure.bodies) == [0.01, 0.04]


@pytest.mark.importer
def test_from_mjcf(mjcf_path):
    structure = StructureVoxel.from_mjcf(mjcf_path, voxel_size=VS)
    assert len(structure.bodies) == 2
    assert len(structure.constraints) == 1


@pytest.mark.importer
def test_from_gltf_multi_node(glb_path):
    structure = StructureVoxel.from_gltf(glb_path, voxel_size=VS)
    assert len(structure.bodies) == 2


@pytest.mark.importer
def test_from_vox_palette_to_material(vox_path):
    structure = StructureVoxel.from_vox(vox_path, voxel_size=VS)
    body = structure.bodies[0]
    assert (body.x_voxels, body.y_voxels, body.z_voxels) == (2, 1, 1)
    assert list(body.material_reference_sid) == [0, 1]
