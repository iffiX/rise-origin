# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Shared generation helpers the first-party kinds apply inline.

Nothing here is system vocabulary: a module using none of these is exactly as
much a citizen. Every random draw flows through a handed-in :class:`Rng`.
"""

from .grid_span import GridSpan
from .field import (
    CellNoise,
    CellSample,
    FBMNoiseField,
    Field,
    GradientNoiseField,
    RidgedNoiseField,
    TurbulenceNoiseField,
    ValueNoiseField,
)
from .cloud import (
    Blob,
    BlobCloud,
    BlobCloudParams,
    CloudBody,
    CloudSnapshot,
    CloudVoxelizer,
)
from .tree import (
    TreeBark,
    TreeBarkBands,
    TreeBarkFurrows,
    TreeBarkPatches,
    TreeBarkPlates,
    TreeBody,
    TreeBodyBudget,
    TreeLeaf,
    TreeVoxelizer,
    WeberPennGraph,
    WeberPennParams,
    WeberPennStem,
    WeberPennTree,
)

__all__ = [
    "GridSpan",
    "CellNoise",
    "CellSample",
    "FBMNoiseField",
    "Field",
    "GradientNoiseField",
    "RidgedNoiseField",
    "TurbulenceNoiseField",
    "ValueNoiseField",
    "Blob",
    "BlobCloud",
    "BlobCloudParams",
    "CloudBody",
    "CloudSnapshot",
    "CloudVoxelizer",
    "TreeBark",
    "TreeBarkBands",
    "TreeBarkFurrows",
    "TreeBarkPatches",
    "TreeBarkPlates",
    "TreeBody",
    "TreeBodyBudget",
    "TreeLeaf",
    "TreeVoxelizer",
    "WeberPennGraph",
    "WeberPennParams",
    "WeberPennStem",
    "WeberPennTree",
]
