# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Resource characterizations.

A resource is a named, shareable asset that other characterizations reference by
name -- currently textures (color sources for materials and floors).

Example::

    from rise_origin import rsc

    sky = rsc.resource.TextureConstant("sky", value=(0.5, 0.7, 0.9, 1.0))
    grid = rsc.resource.TextureGrid("ground", line_color=(0.2, 0.2, 0.2, 1.0))
"""

from rise_origin import RSC_ResourceBase, RSC_TextureBase

from rise_origin.rsc.resource.texture import TextureConstant, TextureGrid

#: Abstract base of every resource characterization (carries name).
ResourceBase = RSC_ResourceBase

#: Abstract base of the texture resources.
TextureBase = RSC_TextureBase

__all__ = [
    "ResourceBase",
    "TextureBase",
    "TextureConstant",
    "TextureGrid",
]
