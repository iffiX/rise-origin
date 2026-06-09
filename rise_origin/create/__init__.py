# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Lightweight, one-line builders for RiseOrigin configuration objects.

The raw ``RS_*Config`` API is verbose and exposes two friction points: opaque
enums, and pybind vectors that can only be extended (never assigned). The
``rise_origin.create`` builders wrap those away so you can assemble a scene in
a few readable lines:

    import rise_origin as rs
    from rise_origin.create import (
        StructureCreator, MaterialCreator, FloorCreator,
        CameraCreator, RecordCreator,
    )

    config = rs.RS_Config()
    config.material_configs.append(MaterialCreator.rigid("rigid"))
    config.floor_config = FloorCreator.flat()

    sc = StructureCreator("ball", voxel_size=0.01)
    sc.add_sphere_body(5, material="rigid", relative_origin_position=(0.0, 0.0, 0.2))
    config.structure_configs.append(sc.build())

    config.camera_configs.append(
        CameraCreator.free("cam", origin_position=(-0.5, 0, 0.3))
    )
    config.record_configs.append(RecordCreator.structure("rec", structure_name="ball"))

Each builder documents the units and valid ranges of its arguments. Kwarg
names mirror C++ field names; inside a per-joint method the redundant
joint-type prefix is dropped (e.g. ``signal_sid`` in ``add_hinge_joint`` maps
to C++ ``hinge_signal_sid``).
"""

from rise_origin.create.camera import CameraCreator
from rise_origin.create.floor import FloorCreator, FloorHeightMapCreator, RenderSpec as FloorRenderSpec
from rise_origin.create.material import MaterialCreator, RenderSpec as MaterialRenderSpec
from rise_origin.create.record import RecordCreator
from rise_origin.create.recorder import RecorderCreator
from rise_origin.create.signal import SignalCreator
from rise_origin.create.structure import (
    BodyRef,
    CompositeBodyRef,
    SegmentRef,
    StructureCreator,
)
from rise_origin.create.utils import Color, Quat, Vec3, VoxelRange

__all__ = [
    "StructureCreator",
    "BodyRef",
    "SegmentRef",
    "CompositeBodyRef",
    "MaterialCreator",
    "MaterialRenderSpec",
    "FloorCreator",
    "FloorHeightMapCreator",
    "FloorRenderSpec",
    "CameraCreator",
    "RecordCreator",
    "SignalCreator",
    "RecorderCreator",
    "Color",
    "Vec3",
    "Quat",
    "VoxelRange",
]
