# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Shared fixtures and oracles for the RSC file and archive serialization tests.

The archive bytes returned by ``__getstate__`` are the canonical, deterministic, positional
serialization of an RSC node (the same bytes pickle uses). Two nodes with equal fields and equal
sharing structure produce equal archive bytes, so byte equality of ``state(a) == state(b)`` is the
strongest available equality oracle. The one caveat is that a node built in Python and the same node
after a file round-trip can differ by a single benign normalization (a unit quaternion's w component
settles by one ulp on read), so the round-trip oracle compares two loaded nodes to each other, never
a loaded node to its in-memory original.
"""

import struct

import numpy as np

import rise_origin as rs
from rise_origin import rsc

_CONSTRAINT = rs.RSE_StructureVoxelConstraintType


def body(voxels=4, voxel_size=0.01, material_sid=0, segment_sid=0):
    """Build a solid cubic voxel body of the given side length."""
    occupancy = np.ones((voxels, voxels, voxels), dtype=bool)
    return rsc.entity.StructureVoxelBody.from_occupancy(
        occupancy,
        material_sid=material_sid,
        segment_sid=segment_sid,
        voxel_size=voxel_size,
    )


def rich_world():
    """Build a world exercising every serialized member kind.

    The world holds a voxel structure with two bodies, a hinge constraint, a shallow copy that shares
    those bodies, a height field floor, perspective and stereo cameras, a ray sensor with a rank two
    direction array, a cosmic background, materials and textures, and a simulation carrying a non
    default denoise variant.
    """
    constraint = rsc.entity.StructureVoxelConstraint(
        _CONSTRAINT.RSE_STRUCTURE_VOXEL_HINGE_JOINT,
        a_body_sid=0,
        b_body_sid=1,
        hinge_min=-1.0,
        hinge_max=1.0,
    )
    structure = rsc.entity.StructureVoxel(
        "s1",
        material_references=["steel"],
        bodies=[body(4), body(3, 0.02)],
        constraints=[constraint],
    )
    structure_copy = structure.shallow_copy(name="s2")
    floor = rsc.entity.FloorArray(
        "floor",
        material_reference="steel",
        x_size=4.0,
        y_size=4.0,
        x_values=2,
        y_values=2,
        height=(0.0, 0.1, 0.2, 0.3),
    )
    camera = rsc.entity.CameraPerspectiveFree("cam", image_width=64, image_height=48)
    stereo = rsc.entity.CameraStereoFree("stereo", image_width=64, image_height=48)
    ray = rsc.entity.RaySensorFree(
        "ray",
        ray_directions=[rs.RVec3rf(1.0, 0.0, 0.0), rs.RVec3rf(0.0, 1.0, 0.0), rs.RVec3rf(0.0, 0.0, 1.0)],
        max_range=50.0,
    )
    sky = rsc.entity.CosmicBackgroundUniform("sky", intensity=0.8)
    simulation = rsc.simulation.Simulation(
        seed=11,
        vision=rsc.simulation.Vision(
            denoise=rsc.simulation.VisionDenoiseSVGF(sigma_color=2.0, temporal_alpha=0.1)
        ),
    )
    return rsc.World(
        "rich",
        resources=[
            rsc.resource.TextureConstant("t", value=(0.5, 0.6, 0.7)),
            rsc.resource.TextureGrid("g", x_spacing=0.5),
        ],
        essences=[rsc.essence.Material("steel")],
        entities=[structure, structure_copy, floor, camera, stereo, ray, sky],
        records=[rsc.record.IntersectRegionSphere("zone", radius=2.0)],
        signals=[rsc.signal.Force("push", structure_name="s1")],
        recorders=[rsc.recorder.RecorderHDF5("rec", path="/tmp/o.h5")],
        controllers=[rsc.controller.ControllerCallback("brain")],
        simulation=simulation,
    )


def state(node):
    """Return a node's canonical archive bytes, the strongest equality oracle."""
    return node.__getstate__()


def pool_entry_count(archive_bytes):
    """Return the number of shared objects in an archive byte stream."""
    return struct.unpack_from("<Q", archive_bytes, 0)[0]


def pool_type_names(archive_bytes):
    """Return the leading run of shared object type names in an archive byte stream."""
    count = struct.unpack_from("<Q", archive_bytes, 0)[0]
    offset = 8
    names = []
    for _ in range(count):
        length = struct.unpack_from("<Q", archive_bytes, offset)[0]
        if length > 64:
            break
        candidate = archive_bytes[offset + 8 : offset + 8 + length]
        if not all(32 <= character < 127 for character in candidate):
            break
        names.append(candidate.decode())
        offset += 8 + length
    return names
