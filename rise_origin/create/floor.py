# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""One-line builders for :class:`RS_FloorConfig` plus heightmap generators.

A floor is either absent, a flat plane at ``z = 0``, or an array (heightmap)
floor whose heights are given in meters along Z. Heightmap floors can tile
infinitely in X and Y.

Every builder takes a ``render`` argument that selects how the surface looks
(see :data:`RenderSpec`): an RGB ``Color`` for a flat colour, or a mapping of
grid parameters for a ruled grid.

Example::

    from rise_origin.create import FloorCreator, FloorHeightMapCreator
    config.floor_config = FloorCreator.flat(render=(120, 120, 120))

    # Grid floor: pass a mapping of grid parameters instead of a colour.
    config.floor_config = FloorCreator.flat(
        render={"x_spacing": 0.5, "y_spacing": 0.5, "line_color": (30, 30, 30)}
    )

    hills = FloorHeightMapCreator.wave(40, 40, amplitude=0.05)
    config.floor_config = FloorCreator.array(
        hills, size=0.4, is_x_repeat=True, is_y_repeat=True
    )
"""

from collections.abc import Mapping
from typing import Any, Optional, Union

import numpy as np

from rise_origin import (
    RColor3b,
    RS_ArrayFloorDataConfig,
    RS_OpaqueFloorRenderConfig,
    RS_OpaqueGridFloorRenderConfig,
    RS_TransparentFloorRenderConfig,
    RS_FloorConfig,
    RSE_FloorDataType,
    RSE_FloorRenderType,
)
from rise_origin.create.utils import Color

__all__ = [
    "FloorCreator",
    "FloorHeightMapCreator",
    "RenderSpec",
]

#: How a floor surface is rendered. One of:
#:
#: * an RGB ``Color`` tuple — a flat opaque constant colour;
#: * a mapping carrying an ``ior`` key — a refractive transparent surface, with
#:   optional ``color`` (per-channel attenuation RGB), ``ior`` and ``roughness``;
#:   see :func:`_transparent_render_config` for defaults;
#: * a mapping carrying a ``color`` key (and no ``ior``) — a flat opaque surface,
#:   with optional ``roughness`` and ``metalness``; see
#:   :func:`_opaque_render_config` for defaults;
#: * any other mapping — a ruled grid, with optional ``x_spacing`` / ``y_spacing``
#:   (meters between lines), ``line_width`` (meters), ``fill_color`` (cell RGB),
#:   ``line_color`` (line RGB), ``roughness`` and ``metalness``; see
#:   :func:`_opaque_grid_render_config` for defaults.
RenderSpec = Union[Color, Mapping[str, Any]]


def _opaque_render_config(
    color: Color = (255, 255, 255),
    roughness: float = 1.0,
    metalness: float = 0.0,
) -> RS_OpaqueFloorRenderConfig:
    """Build an opaque floor render config from a colour, roughness and metalness."""
    render = RS_OpaqueFloorRenderConfig()
    render.color = RColor3b(*color)
    render.roughness = float(roughness)
    render.metalness = float(metalness)
    return render


def _opaque_grid_render_config(
    x_spacing: float = 1.0,
    y_spacing: float = 1.0,
    line_width: float = 0.05,
    fill_color: Color = (255, 255, 255),
    line_color: Color = (0, 0, 0),
    roughness: float = 1.0,
    metalness: float = 0.0,
) -> RS_OpaqueGridFloorRenderConfig:
    """Build a grid floor render config from spacings, width, RGB tuples and shading."""
    render = RS_OpaqueGridFloorRenderConfig()
    render.x_spacing = float(x_spacing)
    render.y_spacing = float(y_spacing)
    render.line_width = float(line_width)
    render.fill_color = RColor3b(*fill_color)
    render.line_color = RColor3b(*line_color)
    render.roughness = float(roughness)
    render.metalness = float(metalness)
    return render


def _transparent_render_config(
    color: Color = (255, 255, 255),
    ior: float = 1.5,
    roughness: float = 0.0,
) -> RS_TransparentFloorRenderConfig:
    """Build a transparent floor render config from a colour, IOR and roughness."""
    render = RS_TransparentFloorRenderConfig()
    render.color = RColor3b(*color)
    render.ior = float(ior)
    render.roughness = float(roughness)
    return render


def _apply_render(floor: RS_FloorConfig, render: RenderSpec) -> None:
    """Populate ``floor.render_config`` from a :data:`RenderSpec`.

    A mapping with an ``ior`` key is a transparent surface; a mapping with a bare
    ``color`` key is an opaque flat surface (so it can carry ``roughness`` /
    ``metalness``); any other mapping is grid parameters; a plain ``Color`` tuple
    is an opaque flat surface.
    """
    if isinstance(render, Mapping):
        if "ior" in render:
            floor.render_config.type = RSE_FloorRenderType.RSE_FLOOR_RENDER_TRANSPARENT
            floor.render_config.config = _transparent_render_config(**render)
        elif "color" in render:
            floor.render_config.type = RSE_FloorRenderType.RSE_FLOOR_RENDER_OPAQUE
            floor.render_config.config = _opaque_render_config(**render)
        else:
            floor.render_config.type = RSE_FloorRenderType.RSE_FLOOR_RENDER_OPAQUE_GRID
            floor.render_config.config = _opaque_grid_render_config(**render)
    else:
        floor.render_config.type = RSE_FloorRenderType.RSE_FLOOR_RENDER_OPAQUE
        floor.render_config.config = _opaque_render_config(render)


class FloorCreator:
    """Factory of :class:`RS_FloorConfig` presets.

    Each method returns a complete floor config (data + render). Assign the
    result directly to ``config.floor_config``.
    """

    @staticmethod
    def none(render: RenderSpec = (120, 120, 120)) -> RS_FloorConfig:
        """Floor disabled; nothing for structures to rest on.

        Args:
            render: Render spec (unused while disabled, kept for consistency).
                See :data:`RenderSpec`.

        Returns:
            A floor config of type ``RSE_FLOOR_DATA_NONE``.
        """
        floor = RS_FloorConfig()
        floor.data_config.type = RSE_FloorDataType.RSE_FLOOR_DATA_NONE
        # NONE/FLAT floors carry no data payload; the config setter only accepts
        # an RS_ArrayFloorDataConfig, so leave data_config.config at its default.
        _apply_render(floor, render)
        return floor

    @staticmethod
    def flat(render: RenderSpec = (120, 120, 120)) -> RS_FloorConfig:
        """Infinite flat floor at ``z = 0``.

        Args:
            render: Constant colour or grid spec; see :data:`RenderSpec`.

        Returns:
            A floor config of type ``RSE_FLOOR_DATA_FLAT``.
        """
        floor = RS_FloorConfig()
        floor.data_config.type = RSE_FloorDataType.RSE_FLOOR_DATA_FLAT
        _apply_render(floor, render)
        return floor

    @staticmethod
    def array(
        height: np.ndarray,
        size: Optional[float] = None,
        *,
        x_size: Optional[float] = None,
        y_size: Optional[float] = None,
        is_x_repeat: bool = False,
        is_y_repeat: bool = False,
        render: RenderSpec = (120, 120, 120),
    ) -> RS_FloorConfig:
        """Array (heightmap) floor from a 2D array of heights in meters.

        Args:
            height: 2D array of shape ``[y_values, x_values]``; each entry is a
                height in meters along Z.
            size: Convenience for square tiles; sets both ``x_size`` and
                ``y_size`` (meters). Provide this or both ``x_size``/``y_size``.
            x_size: Physical tile extent in X (meters). Must be > 0.
            y_size: Physical tile extent in Y (meters). Must be > 0.
            is_x_repeat: If ``True``, the tile repeats infinitely along X. Maps
                to ``RS_ArrayFloorDataConfig.is_x_repeat``.
            is_y_repeat: If ``True``, the tile repeats infinitely along Y. Maps
                to ``RS_ArrayFloorDataConfig.is_y_repeat``.
            render: Constant colour or grid spec; see :data:`RenderSpec`.

        Returns:
            A floor config of type ``RSE_FLOOR_DATA_ARRAY``.

        Raises:
            ValueError: If ``height`` is not 2D, or sizing arguments are missing.
        """
        height = np.asarray(height, dtype=np.float64)
        if height.ndim != 2:
            raise ValueError("height must be a 2D array of shape [y_values, x_values]")
        if size is not None:
            x_size = y_size = size
        if x_size is None or y_size is None:
            raise ValueError("provide size, or both x_size and y_size")

        y_values, x_values = height.shape

        data = RS_ArrayFloorDataConfig()
        data.x_size = float(x_size)
        data.y_size = float(y_size)
        data.x_values = int(x_values)
        data.y_values = int(y_values)
        data.is_x_repeat = is_x_repeat
        data.is_y_repeat = is_y_repeat
        # height is an opaque bound std::vector<float>; extend in place (it cannot
        # be reassigned). Row-major (y outer, x inner) matches the engine layout.
        data.height.extend(height.ravel(order="C").tolist())

        floor = RS_FloorConfig()
        floor.data_config.type = RSE_FloorDataType.RSE_FLOOR_DATA_ARRAY
        floor.data_config.config = data
        _apply_render(floor, render)
        return floor


class FloorHeightMapCreator:
    """Generators of 2D heightmaps (meters) for array floors.

    Each method returns a ``[y_values, x_values]`` float array suitable for
    :meth:`FloorCreator.array`.
    """

    @staticmethod
    def wave(
        x_values: int,
        y_values: int,
        *,
        amplitude: float = 0.05,
        period: int = 10,
    ) -> np.ndarray:
        """Generate a smooth 2D sinusoidal wave heightmap (meters).

        The pattern is seamless when ``x_values``/``y_values`` are multiples of
        ``period``, so it tiles cleanly with ``repeat=True``.

        Args:
            x_values: Number of columns (X samples).
            y_values: Number of rows (Y samples).
            amplitude: Peak height in meters.
            period: Wave period in samples.

        Returns:
            A ``[y_values, x_values]`` float array of heights in meters.
        """
        xs = np.arange(x_values)
        ys = np.arange(y_values)
        gx, gy = np.meshgrid(xs, ys)
        wave = np.sin(2.0 * np.pi * gx / period) * np.sin(2.0 * np.pi * gy / period)
        return amplitude * wave

    @staticmethod
    def mountain(
        x_values: int,
        y_values: int,
        *,
        max_height: float = 0.3,
        flat_radius: float = 3.0,
        slope: float = 0.05,
    ) -> np.ndarray:
        """Generate a funnel-shaped mountain heightmap rising from a flat center.

        Args:
            x_values: Number of columns (X samples).
            y_values: Number of rows (Y samples).
            max_height: Maximum rim height in meters.
            flat_radius: Radius (in samples) of the flat central region.
            slope: Height increase per sample beyond ``flat_radius``.

        Returns:
            A ``[y_values, x_values]`` float array of heights in meters.
        """
        cx, cy = x_values / 2.0, y_values / 2.0
        xs = np.arange(x_values)
        ys = np.arange(y_values)
        gx, gy = np.meshgrid(xs, ys)
        dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        height = np.clip((dist - flat_radius) * slope, 0.0, max_height)
        return height

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        """Perlin's smootherstep easing curve ``6t^5 - 15t^4 + 10t^3``."""
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _perlin_octave(x_values: int, y_values: int, cells: float, rng) -> np.ndarray:
        """Compute one octave of 2D Perlin gradient noise.

        Args:
            x_values: Number of columns to sample.
            y_values: Number of rows to sample.
            cells: Number of gradient-grid cells spanning the sampled region.
            rng: A ``numpy.random.Generator`` supplying the gradient directions.

        Returns:
            A ``[y_values, x_values]`` float array of noise values.
        """
        grid = int(np.ceil(cells)) + 1
        angles = rng.uniform(0.0, 2.0 * np.pi, size=(grid + 1, grid + 1))
        grad_x = np.cos(angles)
        grad_y = np.sin(angles)

        sample_x = np.linspace(0.0, cells, x_values, endpoint=False)
        sample_y = np.linspace(0.0, cells, y_values, endpoint=False)
        fx, fy = np.meshgrid(sample_x, sample_y)

        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        tx = fx - x0
        ty = fy - y0

        # Dot product of each corner's gradient with the offset vector to the sample.
        n00 = grad_x[y0, x0] * tx + grad_y[y0, x0] * ty
        n10 = grad_x[y0, x1] * (tx - 1.0) + grad_y[y0, x1] * ty
        n01 = grad_x[y1, x0] * tx + grad_y[y1, x0] * (ty - 1.0)
        n11 = grad_x[y1, x1] * (tx - 1.0) + grad_y[y1, x1] * (ty - 1.0)

        u = FloorHeightMapCreator._fade(tx)
        v = FloorHeightMapCreator._fade(ty)
        nx0 = n00 * (1.0 - u) + n10 * u
        nx1 = n01 * (1.0 - u) + n11 * u
        return nx0 * (1.0 - v) + nx1 * v

    @staticmethod
    def perlin(
        x_values: int,
        y_values: int,
        *,
        max_height: float = 0.3,
        scale: float = 6.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        seed: int = 0,
    ) -> np.ndarray:
        """Generate a fractal (multi-octave) Perlin-noise terrain heightmap.

        Produces natural rolling terrain, min-max normalized into ``[0, max_height]``.
        Note this is not seamlessly tileable (use :meth:`wave` for
        ``repeat=True`` floors).

        Args:
            x_values: Number of columns (X samples).
            y_values: Number of rows (Y samples).
            max_height: Peak terrain height in meters.
            scale: Base feature size, in gradient cells across the map (larger =
                finer features).
            octaves: Number of noise layers summed for fractal detail.
            persistence: Amplitude falloff per octave (0-1).
            lacunarity: Frequency growth per octave (> 1).
            seed: Random seed for reproducible terrain.

        Returns:
            A ``[y_values, x_values]`` float array of heights in meters.
        """
        rng = np.random.default_rng(seed)
        total = np.zeros((y_values, x_values), dtype=np.float64)
        amplitude = 1.0
        frequency = 1.0
        max_amplitude = 0.0
        for _ in range(max(1, octaves)):
            total += amplitude * FloorHeightMapCreator._perlin_octave(
                x_values, y_values, scale * frequency, rng
            )
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        total /= max_amplitude

        span = total.max() - total.min()
        if span > 0.0:
            total = (total - total.min()) / span
        return max_height * total
