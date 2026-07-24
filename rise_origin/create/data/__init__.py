# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The value types every create signature uses, one file per type family.

Math values are the engine-bound types used directly: ``RVec3rf``,
``RVec2rf``, ``RQuat3rf``, ``RMat3rf``, and ``RS_Version``.
"""

from .dir import Dir, DirName
from .pose import Pose
from .box import AABB, Box, Face, MaterialClass
from .path import Path
from .material import MaterialRequire
from .module_reference import ModuleReference
from .identity import Identity
from .update import (
    CompileMemo,
    CompileMemoEntity,
    WorldUpdate,
    WorldUpdateInstruction,
)
from .annotation import (
    Annotation,
    AnnotationNote,
    AnnotationPath,
    AnnotationPoint,
    AnnotationRegion,
    AnnotationView,
)
from .result import (
    AlongJitter,
    Distance,
    Filters,
    FloorSample,
    GridJitter,
    LeanResult,
    NearbyHit,
    Overlap,
    PlacementReport,
    RayHit,
    RingJitter,
    SurfaceAnchor,
)

__all__ = [
    "Dir",
    "DirName",
    "Pose",
    "AABB",
    "Box",
    "Face",
    "MaterialClass",
    "Path",
    "MaterialRequire",
    "Identity",
    "ModuleReference",
    "CompileMemo",
    "CompileMemoEntity",
    "WorldUpdate",
    "WorldUpdateInstruction",
    "Annotation",
    "AnnotationNote",
    "AnnotationPath",
    "AnnotationPoint",
    "AnnotationRegion",
    "AnnotationView",
    "AlongJitter",
    "Distance",
    "Filters",
    "FloorSample",
    "GridJitter",
    "LeanResult",
    "NearbyHit",
    "Overlap",
    "PlacementReport",
    "RayHit",
    "RingJitter",
    "SurfaceAnchor",
]
