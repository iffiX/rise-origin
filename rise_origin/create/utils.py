# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Shared type aliases for the ``rise_origin.create`` builders."""

from typing import Tuple

__all__ = ["Color", "Vec3", "Quat", "VoxelRange"]

#: RGB colour, 0-255 per channel: ``(r, g, b)``.
Color = Tuple[int, int, int]

#: 3D vector / position in meters: ``(x, y, z)``.
Vec3 = Tuple[float, float, float]

#: Orientation quaternion in ``(x, y, z, w)`` order.
Quat = Tuple[float, float, float, float]

#: Inclusive voxel index box: ``(min_x, max_x, min_y, max_y, min_z, max_z)``.
VoxelRange = Tuple[int, int, int, int, int, int]
