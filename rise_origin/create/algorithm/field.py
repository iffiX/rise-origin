# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Fields: functions over space, sampled one array pass at a time.

A field answers what value sits at each position, computed from the
position and its constructor members alone, with no stored data. Fields
give randomness that is coherent in space: nearby points get similar
values, varying over a feature size in meters, so the same features
appear at every voxel size, just sharper. Fields are ingredients, never
outputs; combining them is plain array math at the call site.

Every field samples either loose ``(n, 3)`` positions or a whole
:class:`GridSpan`, whose cell centers are built by broadcasting the
three coordinate lines so no point cloud is ever stored.
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..protocol.rng import Rng
from .dtype import float32, float64, int64, make_torch_dtype
from .grid_span import GridSpan

__all__ = [
    "Field",
    "ValueNoiseField",
    "GradientNoiseField",
    "FBMNoiseField",
    "RidgedNoiseField",
    "TurbulenceNoiseField",
    "CellNoise",
    "CellSample",
]

_HASH_MASK = 0xFFFFFFFF


class Field:
    """One scalar function over space; calling the instance samples it.

    A field instance holds its dials and its stream-derived salt as
    members set once, so it prints, pickles for the worker pool, and
    samples the same values on every machine. Bodies are elementwise
    math and fixed-order folds only, so a field is deterministic per
    device with no special flags.

    Args:
        dtype: The float dtype the field computes and answers in, from
            the algorithm dtype vocabulary.
    """

    def __init__(self, dtype: object = float32) -> None:
        self.dtype = dtype

    def __call__(
        self, points_or_grid: Union[np.ndarray, GridSpan], device: Optional[str] = None
    ) -> Union[np.ndarray, object]:
        """Return one value per position.

        Args:
            points_or_grid: The positions to sample at (unit: m): an
                ``(n, 3)`` array answered as ``(n,)`` values, or a
                :class:`GridSpan` answered as a ``cell_num``-shaped array
                of cell-center values.
            device: None samples in numpy on the host; a torch device
                name returns a torch tensor living there.
        """
        px, py, pz, xp = Field._make_parts(points_or_grid, device, self.dtype)
        return self._sample(px, py, pz, xp)

    def _sample(self, px: object, py: object, pz: object, xp: object) -> object:
        """Return the field values of prepared coordinate parts.

        Args:
            px: The x coordinates in this field's dtype, broadcastable
                against the other parts.
            py: The y coordinates.
            pz: The z coordinates.
            xp: The array library the parts live in, numpy or torch.
        """
        raise NotImplementedError

    @staticmethod
    def _make_parts(
        points_or_grid: Union[np.ndarray, GridSpan],
        device: Optional[str],
        dtype: object,
    ):
        """Return broadcastable x, y, z parts and their array library."""
        if isinstance(points_or_grid, GridSpan):
            lines = points_or_grid.make_axes(device)
            if device is None:
                xp = np
                cast = [line.astype(dtype) for line in lines]
            else:
                import torch

                xp = torch
                cast = [line.to(make_torch_dtype(dtype)) for line in lines]
            return (
                cast[0][:, None, None],
                cast[1][None, :, None],
                cast[2][None, None, :],
                xp,
            )
        if device is None:
            points = np.asarray(points_or_grid, dtype=dtype)
            return points[:, 0], points[:, 1], points[:, 2], np
        import torch

        points = torch.as_tensor(
            points_or_grid, dtype=make_torch_dtype(dtype), device=device
        )
        return points[:, 0], points[:, 1], points[:, 2], torch

    @staticmethod
    def _cast(values: object, dtype: object, xp: object) -> object:
        """Return the values in a dtype, in the array library they live in."""
        if xp is np:
            return values.astype(dtype)
        return values.to(make_torch_dtype(dtype))

    @staticmethod
    def _make_lattice_bits(
        ix: object, iy: object, iz: object, salt: int, xp: object
    ) -> object:
        """Return deterministic 32-bit lattice hashes carried in int64.

        The 32-bit wraparound of the reference hash is carried in 64-bit
        integers under an explicit mask, which numpy and torch share.
        """
        h = (
            ((ix & _HASH_MASK) * 73856093) & _HASH_MASK
            ^ ((iy & _HASH_MASK) * 19349663) & _HASH_MASK
            ^ ((iz & _HASH_MASK) * 83492791) & _HASH_MASK
            ^ (salt & _HASH_MASK)
        )
        h = h ^ (h >> 13)
        h = (h * 0x5BD1E995) & _HASH_MASK
        h = h ^ (h >> 15)
        return h

    @staticmethod
    def _hash_lattice(
        ix: object, iy: object, iz: object, salt: int, dtype: object, xp: object
    ) -> object:
        """Return deterministic lattice values in [0, 1)."""
        bits = Field._make_lattice_bits(ix, iy, iz, salt, xp)
        return Field._cast(bits, dtype, xp) / float(2**32)

    @staticmethod
    def _draw_salt(rng_or_salt: Union[Rng, int], path: str) -> int:
        """Return the lattice salt drawn from a stream or passed plain."""
        if isinstance(rng_or_salt, Rng):
            return int(rng_or_salt.split(path).integers(0, 2**31))
        return int(rng_or_salt)


class ValueNoiseField(Field):
    """Smooth value noise in [-1, 1] with a feature size in meters.

    The lattice values come from an integer hash salted by the stream, so
    the same stream always yields the same field on every machine, with
    no stored permutation table. A worker process rebuilds the exact
    field from the plain ``salt`` member.

    Args:
        rng_or_salt: The random stream the lattice salt is drawn from, or
            the plain salt of an existing field.
        feature_size: The feature size (unit: m).
        dtype: The float dtype the field computes and answers in, from
            the algorithm dtype vocabulary.
    """

    def __init__(
        self,
        rng_or_salt: Union[Rng, int],
        feature_size: float,
        dtype: object = float64,
    ) -> None:
        super().__init__(dtype)
        self.salt = Field._draw_salt(rng_or_salt, "value_noise")
        self.feature_size = float(feature_size)

    def _sample(self, px: object, py: object, pz: object, xp: object) -> object:
        sx = px / self.feature_size
        sy = py / self.feature_size
        sz = pz / self.feature_size
        fx, fy, fz = xp.floor(sx), xp.floor(sy), xp.floor(sz)
        bx = Field._cast(fx, int64, xp)
        by = Field._cast(fy, int64, xp)
        bz = Field._cast(fz, int64, xp)
        ux, uy, uz = sx - fx, sy - fy, sz - fz
        wx = ux * ux * (3.0 - 2.0 * ux)
        wy = uy * uy * (3.0 - 2.0 * uy)
        wz = uz * uz * (3.0 - 2.0 * uz)
        total = (px + py + pz) * 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner = Field._hash_lattice(
                        bx + dx, by + dy, bz + dz, self.salt, self.dtype, xp
                    )
                    weight = (
                        (wx if dx else 1.0 - wx)
                        * (wy if dy else 1.0 - wy)
                        * (wz if dz else 1.0 - wz)
                    )
                    total = total + corner * weight
        return total * 2.0 - 1.0


class GradientNoiseField(Field):
    """Smooth gradient noise in [-1, 1] with a feature size in meters.

    The improved noise of Perlin 2002: a unit gradient from the twelve
    cube edge directions hangs on every lattice corner, corner dot
    products blend under the quintic fade ``6t^5 - 15t^4 + 10t^3``, and
    the result scales by the unit-gradient bound so the range is exactly
    [-1, 1]. The quintic fade keeps second derivatives continuous, so
    surfaces displaced by this field have clean normals at any cell size.

    Args:
        rng_or_salt: The random stream the lattice salt is drawn from, or
            the plain salt of an existing field.
        feature_size: The feature size (unit: m).
        dtype: The float dtype the field computes and answers in, from
            the algorithm dtype vocabulary.
    """

    _EDGE_DIRECTIONS = tuple(
        (x / math.sqrt(2.0), y / math.sqrt(2.0), z / math.sqrt(2.0))
        for x, y, z in (
            (1, 1, 0),
            (-1, 1, 0),
            (1, -1, 0),
            (-1, -1, 0),
            (1, 0, 1),
            (-1, 0, 1),
            (1, 0, -1),
            (-1, 0, -1),
            (0, 1, 1),
            (0, -1, 1),
            (0, 1, -1),
            (0, -1, -1),
        )
    )
    _RANGE_SCALE = 2.0 / math.sqrt(3.0)

    def __init__(
        self,
        rng_or_salt: Union[Rng, int],
        feature_size: float,
        dtype: object = float32,
    ) -> None:
        super().__init__(dtype)
        self.salt = Field._draw_salt(rng_or_salt, "gradient_noise")
        self.feature_size = float(feature_size)

    def _sample(self, px: object, py: object, pz: object, xp: object) -> object:
        sx = px / self.feature_size
        sy = py / self.feature_size
        sz = pz / self.feature_size
        fx, fy, fz = xp.floor(sx), xp.floor(sy), xp.floor(sz)
        bx = Field._cast(fx, int64, xp)
        by = Field._cast(fy, int64, xp)
        bz = Field._cast(fz, int64, xp)
        ux, uy, uz = sx - fx, sy - fy, sz - fz
        wx = ux * ux * ux * (ux * (ux * 6.0 - 15.0) + 10.0)
        wy = uy * uy * uy * (uy * (uy * 6.0 - 15.0) + 10.0)
        wz = uz * uz * uz * (uz * (uz * 6.0 - 15.0) + 10.0)
        table = self._make_direction_table(xp, px)
        total = (px + py + pz) * 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    bits = Field._make_lattice_bits(
                        bx + dx, by + dy, bz + dz, self.salt, xp
                    )
                    side = bits % 12
                    dot = (
                        table[0][side] * (ux - dx)
                        + table[1][side] * (uy - dy)
                        + table[2][side] * (uz - dz)
                    )
                    weight = (
                        (wx if dx else 1.0 - wx)
                        * (wy if dy else 1.0 - wy)
                        * (wz if dz else 1.0 - wz)
                    )
                    total = total + dot * weight
        return total * self._RANGE_SCALE

    def _make_direction_table(self, xp: object, like: object):
        """Return the edge direction parts as arrays beside the samples."""
        parts = tuple(zip(*self._EDGE_DIRECTIONS))
        if xp is np:
            return tuple(np.asarray(part, dtype=self.dtype) for part in parts)
        import torch

        return tuple(
            torch.as_tensor(part, dtype=like.dtype, device=like.device)
            for part in parts
        )


class _LayerNoiseField(Field):
    """Gradient noise layers summed under one shaping rule.

    Each layer halves the feature size of the one before and scales by
    ``gain``; the shaped sum divides by the weight total so the range of
    the shaping rule is kept.

    Args:
        rng_or_salt: The random stream each layer's salt is drawn from,
            or the plain base salt of an existing field.
        feature_size: The feature size of the coarsest layer (unit: m).
        layer_num: How many layers to sum.
        gain: The weight each finer layer scales by.
        dtype: The float dtype the field computes and answers in, from
            the algorithm dtype vocabulary.
    """

    def __init__(
        self,
        rng_or_salt: Union[Rng, int],
        feature_size: float,
        layer_num: int = 4,
        gain: float = 0.5,
        dtype: object = float32,
    ) -> None:
        super().__init__(dtype)
        base_salt = Field._draw_salt(rng_or_salt, "layer_noise")
        self.feature_size = float(feature_size)
        self.layer_num = int(layer_num)
        self.gain = float(gain)
        self.layers = tuple(
            GradientNoiseField(
                (base_salt + 0x632BE59B * index) & _HASH_MASK,
                feature_size / (2.0**index),
                dtype,
            )
            for index in range(self.layer_num)
        )

    def _sample(self, px: object, py: object, pz: object, xp: object) -> object:
        total = (px + py + pz) * 0.0
        scale = 0.0
        for index, layer in enumerate(self.layers):
            weight = self.gain**index
            shaped = self._shape_layer(layer._sample(px, py, pz, xp), xp)
            total = total + shaped * weight
            scale += weight
        return total / scale

    def _shape_layer(self, values: object, xp: object) -> object:
        """Return one layer's values under this field's shaping rule."""
        raise NotImplementedError


class FBMNoiseField(_LayerNoiseField):
    """Layered gradient noise in [-1, 1]: natural rough variation.

    Args:
        rng_or_salt: The random stream each layer's salt is drawn from,
            or the plain base salt of an existing field.
        feature_size: The feature size of the coarsest layer (unit: m).
        layer_num: How many layers to sum.
        gain: The weight each finer layer scales by.
        dtype: The float dtype the field computes and answers in, from
            the algorithm dtype vocabulary.
    """

    def _shape_layer(self, values: object, xp: object) -> object:
        return values


class RidgedNoiseField(_LayerNoiseField):
    """Layered ridged noise in [0, 1]: sharp crests for rocky shapes.

    Args:
        rng_or_salt: The random stream each layer's salt is drawn from,
            or the plain base salt of an existing field.
        feature_size: The feature size of the coarsest layer (unit: m).
        layer_num: How many layers to sum.
        gain: The weight each finer layer scales by.
        dtype: The float dtype the field computes and answers in, from
            the algorithm dtype vocabulary.
    """

    def _shape_layer(self, values: object, xp: object) -> object:
        return 1.0 - xp.abs(values)


class TurbulenceNoiseField(_LayerNoiseField):
    """Layered folded noise in [0, 1]: the marble vein and bed warp shape.

    Args:
        rng_or_salt: The random stream each layer's salt is drawn from,
            or the plain base salt of an existing field.
        feature_size: The feature size of the coarsest layer (unit: m).
        layer_num: How many layers to sum.
        gain: The weight each finer layer scales by.
        dtype: The float dtype the field computes and answers in, from
            the algorithm dtype vocabulary.
    """

    def _shape_layer(self, values: object, xp: object) -> object:
        return xp.abs(values)


@dataclass(frozen=True)
class CellSample:
    """One cellular query's answers, arrays shaped like the positions.

    Attributes:
        nearest_distance: The distance to the nearest feature point
            (unit: feature size).
        second_distance: The distance to the second nearest feature
            point (unit: feature size).
        cell_id: The 32-bit hash naming the owning feature point's cell,
            carried in int64.
    """

    nearest_distance: object
    second_distance: object
    cell_id: object


class CellNoise:
    """Cellular noise: distances to scattered points and the owner cell.

    One feature point hangs jittered inside every lattice cell; a query
    scans the 3x3x3 cell neighborhood and answers the nearest and
    second nearest distances plus the id of the owning cell, which
    colors granite grains and picks a mineral per stone. Not a scalar
    field: the answer is a :class:`CellSample`.

    Args:
        rng_or_salt: The random stream the lattice salts are drawn from,
            or the plain base salt of an existing instance.
        feature_size: The cell size (unit: m).
        dtype: The float dtype distances are computed and answered in,
            from the algorithm dtype vocabulary.
    """

    _PART_SALTS = (0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35)

    def __init__(
        self,
        rng_or_salt: Union[Rng, int],
        feature_size: float,
        dtype: object = float32,
    ) -> None:
        self.dtype = dtype
        self.salt = Field._draw_salt(rng_or_salt, "cell_noise")
        self.feature_size = float(feature_size)

    def __call__(
        self, points_or_grid: Union[np.ndarray, GridSpan], device: Optional[str] = None
    ) -> CellSample:
        """Return the cellular answers per position.

        Args:
            points_or_grid: The positions to sample at (unit: m): an
                ``(n, 3)`` array or a :class:`GridSpan` sampled at its
                cell centers.
            device: None samples in numpy on the host; a torch device
                name returns torch tensors living there.
        """
        px, py, pz, xp = Field._make_parts(points_or_grid, device, self.dtype)
        sx = px / self.feature_size
        sy = py / self.feature_size
        sz = pz / self.feature_size
        bx = Field._cast(xp.floor(sx), int64, xp)
        by = Field._cast(xp.floor(sy), int64, xp)
        bz = Field._cast(xp.floor(sz), int64, xp)
        big = (px + py + pz) * 0.0 + 1.0e9
        nearest = big
        second = big
        owner = bx * 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cx, cy, cz = bx + dx, by + dy, bz + dz
                    ox = Field._hash_lattice(
                        cx, cy, cz, self.salt ^ self._PART_SALTS[0], self.dtype, xp
                    )
                    oy = Field._hash_lattice(
                        cx, cy, cz, self.salt ^ self._PART_SALTS[1], self.dtype, xp
                    )
                    oz = Field._hash_lattice(
                        cx, cy, cz, self.salt ^ self._PART_SALTS[2], self.dtype, xp
                    )
                    gx = sx - (Field._cast(cx, self.dtype, xp) + ox)
                    gy = sy - (Field._cast(cy, self.dtype, xp) + oy)
                    gz = sz - (Field._cast(cz, self.dtype, xp) + oz)
                    distance = xp.sqrt(gx * gx + gy * gy + gz * gz)
                    bits = Field._make_lattice_bits(cx, cy, cz, self.salt, xp)
                    is_nearest = distance < nearest
                    second = xp.where(is_nearest, nearest, xp.minimum(second, distance))
                    nearest = xp.minimum(nearest, distance)
                    owner = xp.where(is_nearest, bits, owner)
        return CellSample(nearest, second, owner)
