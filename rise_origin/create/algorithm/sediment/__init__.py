# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The sediment engine: one voxel volume, three composable phases.

A :class:`Volume` is the only state: a world-anchored voxel grid whose
channels say what material each cell is, which segment it belongs to,
and what structural marks it carries. Generators produce volumes,
fracture classes split them into segments, and erosion classes remove
material; a rock or a terrain differs only in which classes run and at
what cell size.
"""

from .volume import Volume
from .generator import (
    ColumnGenerator,
    GrainGenerator,
    StrataGenerator,
    StrataGeneratorLayer,
    UniformGenerator,
    VeinGenerator,
)
from .fracture import (
    ChunkFracture,
    FractureFragment,
    PlaneFracture,
    PlateFracture,
    SeamFracture,
)
from .pile import PileGenerator, SegmentTone
from .erosion import BallErosion, DetailErosion, PileErosion

__all__ = [
    "Volume",
    "UniformGenerator",
    "StrataGeneratorLayer",
    "StrataGenerator",
    "GrainGenerator",
    "ColumnGenerator",
    "VeinGenerator",
    "PileGenerator",
    "SegmentTone",
    "FractureFragment",
    "PlaneFracture",
    "SeamFracture",
    "ChunkFracture",
    "PlateFracture",
    "BallErosion",
    "PileErosion",
    "DetailErosion",
]
