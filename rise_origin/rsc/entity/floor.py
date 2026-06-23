# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Floor entities: the static ground surface.

Two kinds are available: an infinite flat plane, and a finite height-field
("array") floor whose elevation is sampled from a grid. Both reference a
material for appearance and contact.
"""

from typing import Callable

import numpy as np

from rise_origin import RSC_FloorArray, RSC_FloorFlat

__all__ = ["FloorFlat", "FloorArray"]


class FloorFlat(RSC_FloorFlat):
    """An infinite flat floor plane at ``z = height``, centered on the origin.

    Args:
        name: Unique entity name.
        material_reference: Name of the material applied to the floor.
        height: Plane elevation along Z, in meters.
        uv_scale: World-meters-per-UV-tile for texturing the plane.
    """

    def __init__(
        self,
        name: str = "floor",
        *,
        material_reference: str = "",
        height: float = 0.0,
        uv_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.material_reference = material_reference
        self.height = height
        self.uv_scale = uv_scale


class FloorArray(RSC_FloorArray):
    """A finite height-field floor sampled from a row-major elevation grid.

    The grid spans ``x_size`` by ``y_size`` meters centered on the origin, with
    ``x_values`` by ``y_values`` samples; ``height`` lists the elevations in
    row-major order (length ``x_values * y_values``). Repeat flags tile the
    field beyond its extent along each axis.

    Args:
        name: Unique entity name.
        material_reference: Name of the material applied to the floor.
        x_size: Field extent along X, in meters.
        y_size: Field extent along Y, in meters.
        x_values: Number of samples along X.
        y_values: Number of samples along Y.
        is_x_repeat: Whether the field tiles along X beyond its extent.
        is_y_repeat: Whether the field tiles along Y beyond its extent.
        height: Row-major elevation samples in meters
            (length ``x_values * y_values``).
    """

    def __init__(
        self,
        name: str = "floor",
        *,
        material_reference: str = "",
        x_size: float = 0.0,
        y_size: float = 0.0,
        x_values: int = 0,
        y_values: int = 0,
        is_x_repeat: bool = False,
        is_y_repeat: bool = False,
        height: "tuple[float, ...]" = (),
    ) -> None:
        super().__init__()
        self.name = name
        self.material_reference = material_reference
        self.x_size = x_size
        self.y_size = y_size
        self.x_values = x_values
        self.y_values = y_values
        self.is_x_repeat = is_x_repeat
        self.is_y_repeat = is_y_repeat
        self.height = height

    @classmethod
    def from_function(
        cls,
        fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        x_size: float,
        y_size: float,
        x_values: int,
        y_values: int,
        name: str = "floor",
        material_reference: str = "",
        is_x_repeat: bool = False,
        is_y_repeat: bool = False,
    ) -> "FloorArray":
        """Build a height-field floor by sampling ``fn`` over the grid.

        ``fn`` is evaluated, vectorized, at each sample's world ``(x, y)`` (the
        field is centered on the origin, spanning ``x_size`` by ``y_size``) and
        returns the elevation in meters.

        Args:
            fn: Elevation function taking two ``(y_values, x_values)`` coordinate
                arrays (world meters) and returning a height array (meters).
            x_size: Field extent along X, in meters.
            y_size: Field extent along Y, in meters.
            x_values: Number of samples along X.
            y_values: Number of samples along Y.
            name: Unique entity name.
            material_reference: Name of the material applied to the floor.
            is_x_repeat: Whether the field tiles along X beyond its extent.
            is_y_repeat: Whether the field tiles along Y beyond its extent.

        Returns:
            A populated :class:`FloorArray`.
        """
        xs = -x_size / 2.0 + (np.arange(x_values) + 0.5) * (x_size / x_values)
        ys = -y_size / 2.0 + (np.arange(y_values) + 0.5) * (y_size / y_values)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        heights = np.asarray(fn(gx, gy), dtype=float)
        return cls(
            name,
            material_reference=material_reference,
            x_size=x_size,
            y_size=y_size,
            x_values=x_values,
            y_values=y_values,
            is_x_repeat=is_x_repeat,
            is_y_repeat=is_y_repeat,
            height=heights.ravel(order="C").tolist(),
        )

    @classmethod
    def from_heightmap(
        cls,
        heightmap,
        *,
        x_size: float,
        y_size: float,
        name: str = "floor",
        material_reference: str = "",
        is_x_repeat: bool = False,
        is_y_repeat: bool = False,
    ) -> "FloorArray":
        """Build a height-field floor from a 2D elevation grid.

        Sample counts are inferred from the grid shape, which is interpreted as
        ``(y_values, x_values)`` (row-major, X fastest).

        Args:
            heightmap: ``(y_values, x_values)`` array of elevations in meters.
            x_size: Field extent along X, in meters.
            y_size: Field extent along Y, in meters.
            name: Unique entity name.
            material_reference: Name of the material applied to the floor.
            is_x_repeat: Whether the field tiles along X beyond its extent.
            is_y_repeat: Whether the field tiles along Y beyond its extent.

        Returns:
            A populated :class:`FloorArray`.
        """
        grid = np.asarray(heightmap, dtype=float)
        y_values, x_values = grid.shape
        return cls(
            name,
            material_reference=material_reference,
            x_size=x_size,
            y_size=y_size,
            x_values=int(x_values),
            y_values=int(y_values),
            is_x_repeat=is_x_repeat,
            is_y_repeat=is_y_repeat,
            height=grid.ravel(order="C").tolist(),
        )
