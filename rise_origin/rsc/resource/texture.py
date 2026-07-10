# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Texture resources: reusable color sources referenced by materials and floors.

A texture is named and referenced by other characterizations (e.g. a material's
``albedo_texture_reference``). The simplest is a flat constant color; the grid
texture procedurally paints a ruled grid useful for ground planes and scale
reference.
"""

from rise_origin import RSC_TextureConstant, RSC_TextureGrid

from rise_origin import RColor3f

__all__ = ["TextureConstant", "TextureGrid"]


class TextureConstant(RSC_TextureConstant):
    """A flat, single-color texture.

    Args:
        name: Unique resource name used by texture references.
        value: Constant RGB color, 0-1 per channel.
    """

    def __init__(
        self,
        name: str,
        *,
        value: RColor3f = RColor3f.zero(),
    ) -> None:
        super().__init__()
        self.name = name
        self.value = value


class TextureGrid(RSC_TextureGrid):
    """A procedural ruled-grid texture.

    Lines of ``line_color`` and width ``line_width`` are drawn over a ``fill_color``
    background at ``x_spacing`` / ``y_spacing`` intervals. Spacings and width are
    interpreted in world meters when ``is_meters`` is true, otherwise in UV units.

    Args:
        name: Unique resource name used by texture references.
        fill_color: Background RGB color, 0-1 per channel.
        line_color: Grid-line RGB color, 0-1 per channel.
        is_meters: If true, spacings and line width are in world meters; if
            false, in UV units.
        x_spacing: Grid spacing along U (meters or UV per ``is_meters``).
        y_spacing: Grid spacing along V (meters or UV per ``is_meters``).
        line_width: Grid-line width (meters or UV per ``is_meters``).
    """

    def __init__(
        self,
        name: str,
        *,
        fill_color: RColor3f = RColor3f.zero(),
        line_color: RColor3f = RColor3f.zero(),
        is_meters: bool = True,
        x_spacing: float = 1.0,
        y_spacing: float = 1.0,
        line_width: float = 0.05,
    ) -> None:
        super().__init__()
        self.name = name
        self.fill_color = fill_color
        self.line_color = line_color
        self.is_meters = is_meters
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.line_width = line_width
