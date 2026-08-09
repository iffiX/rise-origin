# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Sky modes — three world backgrounds over a mirror-smooth reflective floor.

What you'll learn:

* Switch between the three cosmic-background entities: a flat constant colour
  (:class:`rsc.entity.CosmicBackgroundUniform` reading a constant texture), a
  zenith/horizon gradient with a sun lobe
  (:class:`rsc.entity.CosmicBackgroundGradient`), and the physically-based
  Hosek-Wilkie analytic sky (:class:`rsc.entity.CosmicBackgroundHosekWilkie`).
* Build an opaque *reflective* floor: an opaque material optical essence with low
  roughness and full metalness, so the sky and the props mirror in the ground,
  the clearest way to judge each sky mode.

To compare the three skies, comment one of the three lines in :func:`build`
(under "Pick ONE sky") on and the others off, then re-run.

Run it::

    python examples/render/sky.py
    RISE_MODE=record RISE_OUT=sky.mp4 python examples/render/sky.py
"""

from rise_origin import rsc

import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E401,E702 — find sibling _common
import _common

VOXEL = 0.02
CUBE = 4  # cube edge (voxels) -> 0.08 m props

# Three colourful props sitting on the mirror floor, to give the sky something
# to reflect and the reflection something to carry.
PROP_COLORS = [
    _common.PALETTE["blue_green"],
    _common.PALETTE["amber"],
    _common.PALETTE["tiger_orange"],
]


def reflective_floor(
    resources, essences, *, color=(150, 150, 155), roughness=0.04, metalness=1.0
):
    """Flat opaque floor tuned to a near-mirror metallic surface.

    The floor's look is its material's optical essence: a constant albedo plus a
    packed roughness/metalness texture (roughness in red, metalness in green).
    Small roughness gives sharp reflections; full metalness picks a metallic lobe.

    Returns:
        A :class:`rsc.entity.FloorFlat` referencing the ``ground`` material.
    """
    resources.append(
        rsc.resource.TextureConstant("ground#albedo", value=_common._rgb(color))
    )
    resources.append(
        rsc.resource.TextureConstant(
            "ground#rm", value=(roughness, metalness, 0.0)
        )
    )
    essences.append(rsc.essence.Material("ground"))
    essences.append(
        rsc.essence.MaterialPhysicalPropertyFixed(
            "ground#phys", material_reference="ground"
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
    return rsc.entity.FloorFlat("floor", material_reference="ground")


# --- The three sky entities -------------------------------------------------


def constant_color_sky(resources):
    """Flat single-colour background (no horizon, no sun)."""
    resources.append(
        rsc.resource.TextureConstant("sky#color", value=_common._rgb((135, 178, 230)))
    )
    return rsc.entity.CosmicBackgroundUniform("sky", texture_reference="sky#color")


def gradient_sky(resources):
    """Zenith-to-horizon gradient with a soft specular sun lobe."""
    return rsc.entity.CosmicBackgroundGradient(
        "sky",
        zenith_color=(45, 105, 200),  # deep blue overhead
        horizon_color=(232, 230, 214),  # pale near the horizon
        sun_direction=(0.3, 0.4, -1.0),  # must be non-zero
        sun_color=(255, 248, 224),
    )


def hosek_wilkie_sky(resources):
    """Physically-based analytic sky (Hosek-Wilkie); sun points up at +Z."""
    return rsc.entity.CosmicBackgroundHosekWilkie(
        "sky",
        sun_direction=(0.3, 0.4, 0.866),  # toward the sun
        sun_color=(255, 245, 220),
        turbidity=3.0,  # [1, 10]; 1 clear, 10 hazy
        ground_color=(128, 128, 128),  # albedo bounced into sky
        radiance_scale=1.0,
    )


def build():
    """Build the sky-test scene (reflective floor + a few rigid props).

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    resources = []
    essences = []
    floor = reflective_floor(resources, essences)

    # --- Pick ONE sky (comment the other two out) ---------------------------
    # sky = constant_color_sky(resources)
    # sky = gradient_sky(resources)
    sky = hosek_wilkie_sky(resources)
    # ------------------------------------------------------------------------

    entities = [sky, floor]

    size = CUBE * VOXEL
    span = (len(PROP_COLORS) - 1) * 0.14
    for i, color in enumerate(PROP_COLORS):
        name = f"prop{i}"
        resources.append(
            rsc.resource.TextureConstant(f"{name}#albedo", value=_common._rgb(color))
        )
        essences.append(rsc.essence.Material(name))
        essences.append(
            rsc.essence.MaterialPhysicalPropertyFixed(
                f"{name}#phys", material_reference=name
            )
        )
        essences.append(
            rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
                f"{name}#optic",
                material_reference=name,
                albedo_texture_reference=f"{name}#albedo",
            )
        )
        y = i * 0.14 - span / 2
        entities.append(
            rsc.entity.StructureVoxel(
                name,
                origin_position=(0.0, y, 0.0),
                material_references=[name],
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
        vision=rsc.simulation.Vision(denoise=rsc.simulation.VisionDenoiseAtrous()),
        runtime=rsc.simulation.Runtime(log_level="info"),
    )
    return rsc.World(
        "sky",
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
        camera_position=(-0.85, 0.0, 0.22),
        camera_focus=(0.0, 0.0, 0.04),
        samples_per_pixel=1,
    )


if __name__ == "__main__":
    main()
