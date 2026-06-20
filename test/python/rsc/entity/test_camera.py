# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.entity.camera."""

import pytest

from rise_origin.rsc.entity import (
    CameraPerspectiveFree,
    CameraPerspectiveRecord,
    CameraPerspectiveRigidBodyLinked,
    StructureVoxel,
    StructureVoxelBody,
)


def test_free_camera_fields():
    camera = CameraPerspectiveFree(
        "cam", origin_position=(-0.5, 0.0, 0.3), image_width=128
    )
    assert camera.name == "cam"
    assert camera.image_width == 128
    assert abs(camera.origin_position.x + 0.5) < 1e-6


def test_camera_embedded_record():
    camera = CameraPerspectiveFree(
        "cam", record=CameraPerspectiveRecord("sight", has_rgb=True)
    )
    assert camera.record.has_rgb is True


def test_rigidbodylinked_attached_to_resolves():
    body = StructureVoxelBody.box(2, 2, 2, voxel_size=0.02)
    StructureVoxel("dog", bodies=[body])
    eye = CameraPerspectiveRigidBodyLinked(
        "eye", attached_to=body, relative_anchor=(0.1, 0, 0)
    )
    assert eye.structure_name == "dog"
    assert eye.body_sid == 0


def test_rigidbodylinked_explicit_ids():
    eye = CameraPerspectiveRigidBodyLinked(
        "eye", structure_name="dog", body_sid=1, segment_bid=2
    )
    assert eye.body_sid == 1
    assert eye.segment_bid == 2
