# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Floor modes — every floor layout crossed with every surface style.

What you'll learn:

* Pick a floor *layout* as an entity: a flat infinite plane
  (:class:`rsc.entity.FloorFlat`), a finite heightmap (array) tile, or that same
  tile repeated infinitely in X/Y (:class:`rsc.entity.FloorArray`).
* Pick a floor *surface* through its material's optical essence: an opaque
  constant colour, a ruled opaque grid (no ``ior``), or a refractive transparent
  surface (``is_transparent`` with ``ior``). Roughness and metalness ride in a
  packed texture, roughness in red and metalness in green.

The scene is built from two independent "pick ONE" blocks in :func:`build` — one
for the surface and one for the layout — so commenting lines on/off walks through
all nine combinations (3 layouts x 3 surfaces).

Run it::

    python examples/render/floor.py
    RISE_MODE=record RISE_OUT=floor.mp4 python examples/render/floor.py
"""

import math

import numpy as np

from rise_origin import rsc

import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E401,E702 — find sibling _common
import _common

VOXEL = 0.02
CUBE = 4  # cube edge (voxels) -> 0.08 m props

# Heightmap (array) floor parameters.
GRID_N = 48  # samples per side
GRID_SIZE = 1.2  # physical tile extent (m), centred on the origin
GRID_AMP = 0.04  # wave peak height (m)
GRID_PERIODS = 4  # whole wave periods across the tile, so it tiles seamlessly

SKY = (170, 200, 235)  # soft blue, so the transparent floor has a tint to bend

# A few colourful props for scale and to sit on the floor.
PROP_COLORS = [
    _common.PALETTE["blue_green"],
    _common.PALETTE["amber"],
    _common.PALETTE["tiger_orange"],
]


# --- The three floor surfaces (resources + optical essence per surface) -----

# Opaque: a flat matte colour (an RGB tuple).
OPAQUE_SURFACE = (200, 198, 192)

# Opaque grid: ruled lines over a fill, rendered as a sharp metal.
GRID_SURFACE = {
    "x_spacing": 0.2,
    "y_spacing": 0.2,
    "line_width": 0.01,
    "fill_color": (210, 208, 202),
    "line_color": (40, 40, 40),
    "roughness": 0.0,
    "metalness": 1.0,
}

# Transparent: a faint tint the surface refracts through.
TRANSPARENT_SURFACE = {"color": (220, 235, 255), "ior": 1.5, "roughness": 0.0}


def opaque_surface(surface, resources, essences) -> None:
    """Matte opaque ground: a constant albedo at the engine's default finish."""
    resources.append(
        rsc.resource.TextureConstant("ground#albedo", value=_common._rgb(surface))
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "ground#optic",
            material_reference="ground",
            albedo_texture_reference="ground#albedo",
        )
    )


def grid_surface(surface, resources, essences) -> None:
    """Ruled grid ground: a grid albedo plus a roughness/metalness texture."""
    resources.append(
        rsc.resource.TextureGrid(
            "ground#albedo",
            fill_color=_common._rgb(surface["fill_color"]),
            line_color=_common._rgb(surface["line_color"]),
            x_spacing=surface["x_spacing"],
            y_spacing=surface["y_spacing"],
            line_width=surface["line_width"],
        )
    )
    resources.append(
        rsc.resource.TextureConstant(
            "ground#rm", value=(surface["roughness"], surface["metalness"], 0.0)
        )
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "ground#optic",
            material_reference="ground",
            albedo_texture_reference="ground#albedo",
            roughness_metalness_texture_reference="ground#rm",
        )
    )


def transparent_surface(surface, resources, essences) -> None:
    """Refractive ground: a smooth transparent surface with an index of refraction."""
    resources.append(
        rsc.resource.TextureConstant(
            "ground#albedo", value=_common._rgb(surface["color"])
        )
    )
    resources.append(
        rsc.resource.TextureConstant(
            "ground#rm", value=(surface["roughness"], 0.0, 0.0)
        )
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "ground#optic",
            material_reference="ground",
            albedo_texture_reference="ground#albedo",
            roughness_metalness_texture_reference="ground#rm",
            is_transparent=True,
            ior=surface["ior"],
        )
    )


# --- The three floor layouts ------------------------------------------------


def flat_floor() -> rsc.entity.FloorFlat:
    """Infinite flat plane at ``z = 0``."""
    return rsc.entity.FloorFlat("floor", material_reference="ground")


def array_floor(*, repeat: bool) -> rsc.entity.FloorArray:
    """Heightmap (array) floor shaped by a cosine wave.

    With ``repeat`` the single tile tiles infinitely in X and Y; otherwise only
    the centred ``GRID_SIZE`` tile exists. The wave completes a whole number of
    periods across the tile, so its value matches at opposite edges and the field
    tiles without a seam.
    """
    wavenumber = 2.0 * math.pi * GRID_PERIODS / GRID_SIZE
    return rsc.entity.FloorArray.from_function(
        lambda gx, gy: GRID_AMP * np.cos(wavenumber * gx) * np.cos(wavenumber * gy),
        x_size=GRID_SIZE,
        y_size=GRID_SIZE,
        x_values=GRID_N,
        y_values=GRID_N,
        name="floor",
        material_reference="ground",
        is_x_repeat=repeat,
        is_y_repeat=repeat,
    )


def build():
    """Build the floor-test scene (one floor + a few rigid props).

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    resources = []
    essences = [
        rsc.essence.Material("ground"),
        rsc.essence.MaterialPhysicalPropertyFixed(
            "ground#phys", material_reference="ground"
        ),
    ]

    # --- Pick ONE floor surface (comment the other two out) -----------------
    # opaque_surface(OPAQUE_SURFACE, resources, essences)
    grid_surface(GRID_SURFACE, resources, essences)
    # transparent_surface(TRANSPARENT_SURFACE, resources, essences)
    # ------------------------------------------------------------------------

    # --- Pick ONE floor layout (comment the other two out) ------------------
    floor = flat_floor()
    # floor = array_floor(repeat=False)
    # floor = array_floor(repeat=True)
    # ------------------------------------------------------------------------

    resources.append(
        rsc.resource.TextureConstant("sky#color", value=_common._rgb(SKY))
    )
    sky = rsc.entity.CosmicBackgroundUniform("sky", texture_reference="sky#color")
    entities = [sky, floor]

    material_names = ["ground"]
    for i, color in enumerate(PROP_COLORS):
        name = f"prop{i}"
        material_names.append(name)
        resources.append(
            rsc.resource.TextureConstant(f"{name}#albedo", value=_common._rgb(color))
        )
        essences.append(rsc.essence.Material(name))
        essences.append(
            rsc.essence.MaterialPhysicalPropertySoft(
                f"{name}#phys",
                material_reference=name,
                elastic_modulus=5.0e5,
                density=1000.0,
                poissons_ratio=0.35,
            )
        )
        essences.append(
            rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
                f"{name}#optic",
                material_reference=name,
                albedo_texture_reference=f"{name}#albedo",
            )
        )

    for i, first in enumerate(material_names):
        for second in material_names[i:]:
            pair = f"{first}_{second}"
            essences.append(
                rsc.essence.MaterialPairPropertyCoulombFriction(
                    f"{pair}#friction",
                    first_material_reference=first,
                    second_material_reference=second,
                    u_static=1.0,
                    u_dynamic=0.8,
                )
            )
            essences.append(
                rsc.essence.MaterialPairPropertyRestitution(
                    f"{pair}#restitution",
                    first_material_reference=first,
                    second_material_reference=second,
                    restitution=0.1,
                )
            )

    size = CUBE * VOXEL
    span = (len(PROP_COLORS) - 1) * 0.18
    for i in range(len(PROP_COLORS)):
        y = i * 0.18 - span / 2
        entities.append(
            rsc.entity.StructureVoxel(
                f"prop{i}",
                origin_position=(0.0, y, 0.02),
                material_references=[f"prop{i}"],
                bodies=[
                    rsc.entity.StructureVoxelBody.box(
                        CUBE,
                        CUBE,
                        CUBE,
                        voxel_size=VOXEL,
                        relative_origin_position=(-size / 2, -size / 2, 0.0),
                    )
                ],
            )
        )

    simulation = rsc.simulation.Simulation(
        environment=rsc.simulation.Environment(gravity_acceleration=-9.81),
        runtime=rsc.simulation.Runtime(log_level="info"),
    )
    return rsc.World(
        "floor",
        simulation=simulation,
        resources=resources,
        essences=essences,
        entities=entities,
    )


def main():
    """Build the scene and launch it in the selected run mode (no control)."""
    _common.launch(
        build(),
        None,
        camera_position=(-0.9, 0.0, 0.38),
        camera_focus=(0.0, 0.0, 0.0),
        samples_per_pixel=16,
        max_depth=4,
    )


if __name__ == "__main__":
    main()
