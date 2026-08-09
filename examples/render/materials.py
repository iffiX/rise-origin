# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Material modes — a front row of solid cubes and a back row of nested cubes.

What you'll learn:

* Reach the three optical looks through one essence,
  :class:`rsc.essence.MaterialOpticalPropertySimpleMicrofacet`: an opaque lobe
  with roughness and metalness in a packed texture, a refractive transparent lobe
  (``is_transparent`` with ``ior``), and an emissive lobe
  (``emission_texture_reference`` with ``emission_radiance``).
* See light travel through nested materials: the back row wraps a transparent
  glass shell around a different-material core, so a ray refracts into the glass,
  crosses the inner interface, and shades or refracts again.

Front row (solid): diffuse, smooth metallic, transparent, emissive.
Back row (glass shell around a core): metallic core, diffuse core, denser-glass
core, and a transparent core with a hollow air pocket at its centre. Each cube is
a ``5 x 5 x 5`` voxel block; cores are ``3 x 3 x 3``.

Run it::

    python examples/render/materials.py
    RISE_MODE=record RISE_OUT=materials.mp4 python examples/render/materials.py
"""

import numpy as np

from rise_origin import RS_NULL_INDEX, rsc

import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E401,E702 — find sibling _common
import _common

VOXEL = 0.02
CUBE = 5  # cube edge (voxels) -> 0.10 m cubes
CORE = 3  # core edge (voxels) -> 0.06 m cores
SPACING = 0.16  # centre spacing between neighbouring cubes (m)
ROW_GAP = 0.20  # front-to-back spacing between the two rows (m)

GROUND_COLOR = (205, 203, 197)  # diffuse off-white grey


def _fixed_material(name, resources, essences) -> None:
    """A material handle plus an immovable (world-anchored) physical property."""
    essences.append(rsc.essence.Material(name))
    essences.append(
        rsc.essence.MaterialPhysicalPropertyFixed(
            f"{name}#phys", material_reference=name
        )
    )


def opaque(name, resources, essences, *, color, roughness, metalness) -> None:
    """Opaque lobe with the given roughness and metalness."""
    _fixed_material(name, resources, essences)
    resources.append(
        rsc.resource.TextureConstant(f"{name}#albedo", value=_common._rgb(color))
    )
    resources.append(
        rsc.resource.TextureConstant(
            f"{name}#rm", value=(roughness, metalness, 0.0)
        )
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            f"{name}#optic",
            material_reference=name,
            albedo_texture_reference=f"{name}#albedo",
            roughness_metalness_texture_reference=f"{name}#rm",
        )
    )


def transparent(name, resources, essences, *, color, ior, roughness) -> None:
    """Refractive transparent lobe; ``color`` is the per-channel tint."""
    _fixed_material(name, resources, essences)
    resources.append(
        rsc.resource.TextureConstant(f"{name}#albedo", value=_common._rgb(color))
    )
    resources.append(
        rsc.resource.TextureConstant(f"{name}#rm", value=(roughness, 0.0, 0.0))
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            f"{name}#optic",
            material_reference=name,
            albedo_texture_reference=f"{name}#albedo",
            roughness_metalness_texture_reference=f"{name}#rm",
            is_transparent=True,
            ior=ior,
        )
    )


def emissive(name, resources, essences, *, color, radiance) -> None:
    """Emissive lobe; emitted radiance is ``color * radiance``."""
    _fixed_material(name, resources, essences)
    resources.append(
        rsc.resource.TextureConstant(f"{name}#emit", value=_common._rgb(color))
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            f"{name}#optic",
            material_reference=name,
            emission_texture_reference=f"{name}#emit",
            emission_radiance=radiance,
        )
    )


def _materials(resources, essences):
    """Build the render-test materials (mechanics fixed, varied optics)."""
    opaque(
        "diffuse",
        resources,
        essences,
        color=(200, 70, 70),
        roughness=1.0,
        metalness=0.0,
    )
    opaque(
        "metallic",
        resources,
        essences,
        color=(220, 222, 226),
        roughness=0.05,
        metalness=1.0,
    )
    transparent(
        "glass", resources, essences, color=(230, 240, 255), ior=1.5, roughness=0.05
    )
    emissive("emissive", resources, essences, color=(255, 200, 120), radiance=10.0)
    transparent(
        "dense_glass",
        resources,
        essences,
        color=(255, 232, 232),
        ior=2.4,
        roughness=0.0,
    )
    transparent(
        "core_glass",
        resources,
        essences,
        color=(232, 255, 236),
        ior=1.33,
        roughness=0.0,
    )


def _solid_cube(name, material, center):
    """A single ``CUBE`` voxel cube of one material at world ``center``."""
    size = CUBE * VOXEL
    return rsc.entity.StructureVoxel(
        name,
        origin_position=(center[0], center[1], center[2]),
        material_references=[material],
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


def _nested_cube(name, shell, core, center, air_center=False):
    """A glass ``shell`` cube wrapping a ``CORE`` cube of ``core`` material.

    The shell is segment 0 and the core is segment 1, so the two materials live in
    separate segments as the engine requires. With ``air_center`` the very middle
    voxel is left empty, leaving a hollow air pocket inside the core.
    """
    size = CUBE * VOXEL
    materials = np.zeros((CUBE, CUBE, CUBE), dtype=np.int64)  # shell -> material 0
    segments = np.zeros((CUBE, CUBE, CUBE), dtype=np.int64)  # shell -> segment 0

    lo = (CUBE - CORE) // 2
    hi = lo + CORE
    materials[lo:hi, lo:hi, lo:hi] = 1  # core -> material 1
    segments[lo:hi, lo:hi, lo:hi] = 1  # core -> segment 1

    if air_center:
        mid = CUBE // 2
        materials[mid, mid, mid] = RS_NULL_INDEX  # hollow air pocket
        segments[mid, mid, mid] = RS_NULL_INDEX

    body = rsc.entity.StructureVoxelBody.from_grid(
        materials,
        segment_grid=segments,
        voxel_size=VOXEL,
        relative_origin_position=(-size / 2, -size / 2, 0.0),
    )
    return rsc.entity.StructureVoxel(
        name,
        origin_position=(center[0], center[1], center[2]),
        material_references=[shell, core],
        bodies=[body],
    )


def build():
    """Build the two-row material-test scene.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    resources = [
        rsc.resource.TextureConstant(
            "ground#albedo", value=_common._rgb(GROUND_COLOR)
        ),
        rsc.resource.TextureConstant("sky#color", value=_common._rgb((255, 255, 255))),
    ]
    essences = [
        rsc.essence.Material("ground"),
        rsc.essence.MaterialPhysicalPropertyFixed(
            "ground#phys", material_reference="ground"
        ),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "ground#optic",
            material_reference="ground",
            albedo_texture_reference="ground#albedo",
        ),
    ]
    _materials(resources, essences)

    # A pure white background gives the glass clean highlights to refract.
    sky = rsc.entity.CosmicBackgroundUniform("sky", texture_reference="sky#color")
    entities = [sky, rsc.entity.FloorFlat("floor", material_reference="ground")]

    front = ["diffuse", "metallic", "glass", "emissive"]
    # Each back cube is (core material, hollow air pocket).
    back = [
        ("metallic", False),
        ("diffuse", False),
        ("dense_glass", False),
        ("core_glass", True),
    ]

    span = (len(front) - 1) * SPACING
    for i, material in enumerate(front):
        y = i * SPACING - span / 2
        entities.append(
            _solid_cube(f"front_{material}", material, (-ROW_GAP / 2, y, 0.0))
        )
    for i, (core, air) in enumerate(back):
        y = i * SPACING - span / 2
        entities.append(
            _nested_cube(
                f"back_{i}", "glass", core, (ROW_GAP / 2, y, 0.0), air_center=air
            )
        )

    simulation = rsc.simulation.Simulation(
        environment=rsc.simulation.Environment(gravity_acceleration=-9.81),
        # ReSTIR is disabled for this multi-refraction glass scene.
        vision=rsc.simulation.Vision(
            restir=rsc.simulation.VisionReSTIR(is_enabled=False)
        ),
        runtime=rsc.simulation.Runtime(log_level="info"),
    )
    return rsc.World(
        "materials",
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
        camera_position=(-0.85, 0.0, 0.45),
        camera_focus=(0.0, 0.0, 0.04),
        samples_per_pixel=128,
        max_depth=10,
    )


if __name__ == "__main__":
    main()
