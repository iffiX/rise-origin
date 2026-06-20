# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Entity characterizations.

An entity is an object that participates in the simulation: a voxel structure, a
perspective camera, a floor, or a cosmic background. Structures and cameras
carry embedded records and signals; bodies expose geometric named constructors
(``box`` / ``sphere`` / ``cylinder`` / ``capsule`` / ``from_grid`` / ``from_sdf``
/ ``from_mesh``).
"""

from rise_origin import (
    RSC_CameraBase,
    RSC_CameraPerspectiveBase,
    RSC_CosmicBackgroundBase,
    RSC_EntityBase,
    RSC_FloorBase,
    RSC_StructureBase,
)

from rise_origin.rsc.entity.camera import (
    CameraPerspectiveFree,
    CameraPerspectiveRecord,
    CameraPerspectiveRigidBodyLinked,
    CameraPerspectiveSignal,
)
from rise_origin.rsc.entity.cosmic_background import (
    CosmicBackgroundGradient,
    CosmicBackgroundHosekWilkie,
    CosmicBackgroundUniform,
)
from rise_origin.rsc.entity.floor import FloorArray, FloorFlat
from rise_origin.rsc.entity.structure_voxel import (
    BodyRef,
    VoxelSizeSpec,
    StructureVoxel,
    StructureVoxelBody,
    StructureVoxelConstraint,
    StructureVoxelRecord,
    StructureVoxelSignal,
)

#: Abstract base of every entity characterization (carries name).
EntityBase = RSC_EntityBase

#: Abstract base of the structure entities.
StructureBase = RSC_StructureBase

#: Abstract base of the camera entities.
CameraBase = RSC_CameraBase

#: Abstract base of the perspective cameras.
CameraPerspectiveBase = RSC_CameraPerspectiveBase

#: Abstract base of the floor entities.
FloorBase = RSC_FloorBase

#: Abstract base of the cosmic-background entities.
CosmicBackgroundBase = RSC_CosmicBackgroundBase

__all__ = [
    "BodyRef",
    "VoxelSizeSpec",
    "EntityBase",
    "StructureBase",
    "CameraBase",
    "CameraPerspectiveBase",
    "FloorBase",
    "CosmicBackgroundBase",
    "StructureVoxel",
    "StructureVoxelBody",
    "StructureVoxelConstraint",
    "StructureVoxelRecord",
    "StructureVoxelSignal",
    "CameraPerspectiveFree",
    "CameraPerspectiveRigidBodyLinked",
    "CameraPerspectiveRecord",
    "CameraPerspectiveSignal",
    "FloorFlat",
    "FloorArray",
    "CosmicBackgroundUniform",
    "CosmicBackgroundGradient",
    "CosmicBackgroundHosekWilkie",
]
