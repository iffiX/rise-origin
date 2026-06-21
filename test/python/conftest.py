# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Shared fixtures for the rise_origin Python test suite.

Asset fixtures are synthesized on disk per session rather than committed as
binaries, so importer tests stay hermetic and reviewable.
"""

import struct

import numpy as np
import pytest


@pytest.fixture(scope="session")
def box_mesh():
    """A watertight 0.1 m cube ``trimesh.Trimesh``."""
    trimesh = pytest.importorskip("trimesh")
    return trimesh.creation.box(extents=(0.1, 0.1, 0.1))


@pytest.fixture(scope="session")
def vox_path(tmp_path_factory):
    """A 2x1x1 MagicaVoxel file: two voxels with palette indices 1 and 2."""

    def chunk(chunk_id, content):
        return chunk_id + struct.pack("<ii", len(content), 0) + content

    size = chunk(b"SIZE", struct.pack("<iii", 2, 1, 1))
    voxels = chunk(b"XYZI", struct.pack("<i", 2) + bytes([0, 0, 0, 1, 1, 0, 0, 2]))
    main = chunk(b"MAIN", b"") + size + voxels
    path = tmp_path_factory.mktemp("assets") / "blob.vox"
    path.write_bytes(b"VOX " + struct.pack("<i", 150) + main)
    return str(path)


@pytest.fixture(scope="session")
def urdf_path(tmp_path_factory):
    """A two-link revolute-jointed URDF (box base + cylinder upper arm)."""
    path = tmp_path_factory.mktemp("assets") / "arm.urdf"
    path.write_text("""<?xml version="1.0"?>
<robot name="arm">
  <link name="base">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="upper">
    <collision>
      <origin xyz="0 0 0.2"/>
      <geometry><cylinder radius="0.05" length="0.4"/></geometry>
    </collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="2"/>
  </joint>
</robot>
""")
    return str(path)


@pytest.fixture(scope="session")
def mjcf_path(tmp_path_factory):
    """A two-body hinge-jointed MJCF (box base + capsule upper arm)."""
    path = tmp_path_factory.mktemp("assets") / "arm.xml"
    path.write_text("""<mujoco model="mjarm">
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom name="g0" type="box" size="0.1 0.1 0.1"/>
      <body name="upper" pos="0 0 0.2">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0 0 0.3" size="0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
    return str(path)


@pytest.fixture(scope="session")
def glb_path(tmp_path_factory):
    """A two-mesh GLB scene (a box node and an offset sphere node)."""
    trimesh = pytest.importorskip("trimesh")
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
        node_name="cube",
        geom_name="cube",
    )
    offset = np.eye(4)
    offset[:3, 3] = (0.3, 0.0, 0.0)
    scene.add_geometry(
        trimesh.creation.icosphere(radius=0.05),
        node_name="ball",
        geom_name="ball",
        transform=offset,
    )
    path = tmp_path_factory.mktemp("assets") / "scene.glb"
    scene.export(str(path))
    return str(path)


@pytest.fixture(scope="session")
def step_path(tmp_path_factory):
    """A two-solid STEP file: a box and an offset sphere."""
    cq = pytest.importorskip("cadquery")
    assembly = cq.Assembly()
    assembly.add(cq.Workplane().box(0.1, 0.1, 0.1), name="box")
    assembly.add(
        cq.Workplane().transformed(offset=(0.3, 0, 0)).sphere(0.05), name="ball"
    )
    path = tmp_path_factory.mktemp("assets") / "parts.step"
    assembly.export(str(path))
    return str(path)
