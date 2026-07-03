# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Minimal VR scene: a grid floor and one cube, presented to an OpenXR headset.

What you'll learn:

* Build a tiny world (grid floor + a single rigid cube).
* Drive it through an :class:`~rise_origin.RS_WorldSession` and a
  :class:`~rise_origin.RS_WorldSessionComponentVR`, which injects a stereo camera
  and presents it to whatever OpenXR runtime is active (on Linux this is
  Monado/WiVRn). The headset's head pose drives the stereo camera each step.

The control frequency is raised to the headset display rate: each control step
renders one stereo frame and consumes one head pose, and the runtime reprojects
from the submitted depth between steps.

Requires a core built with ``RISE_ORIGIN_BUILD_CORE_WITH_VR=ON`` and a running
OpenXR runtime with a headset attached.

Run it::

    python examples/basic/vr.py
"""

import rise_origin as rs
from rise_origin import rsc

VOXEL = 0.02
CUBE = 4  # cube edge in voxels -> 0.08 m
DISPLAY_HZ = 72.0  # control rate = headset display rate

FLOOR_FILL = (210 / 255.0, 208 / 255.0, 202 / 255.0)
FLOOR_LINE = (40 / 255.0, 40 / 255.0, 40 / 255.0)
CUBE_COLOR = (33 / 255.0, 158 / 255.0, 188 / 255.0)
SKY_COLOR = (170 / 255.0, 200 / 255.0, 235 / 255.0)


def build() -> "rsc.World":
    """Build the VR scene: a grid floor, one cube, and a uniform sky."""
    resources = [
        rs_grid := rsc.resource.TextureGrid(
            "ground#albedo",
            fill_color=FLOOR_FILL,
            line_color=FLOOR_LINE,
            x_spacing=1,
            y_spacing=1,
            line_width=0.05,
        ),
        rsc.resource.TextureConstant("ground#rm", value=(0.6, 0.0, 0.0)),
        rsc.resource.TextureConstant("cube#albedo", value=CUBE_COLOR),
        rsc.resource.TextureConstant("sky#color", value=SKY_COLOR),
    ]

    essences = [
        rsc.essence.Material("ground"),
        rsc.essence.MaterialPhysicalPropertyFixed("ground#phys", material_reference="ground"),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "ground#optic",
            material_reference="ground",
            albedo_texture_reference="ground#albedo",
            roughness_metalness_texture_reference="ground#rm",
        ),
        rsc.essence.Material("cube"),
        rsc.essence.MaterialPhysicalPropertySoft(
            "cube#phys",
            material_reference="cube",
            elastic_modulus=5.0e5,
            density=1000.0,
            poissons_ratio=0.35,
        ),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "cube#optic",
            material_reference="cube",
            albedo_texture_reference="cube#albedo",
        ),
    ]

    for first, second in (("ground", "cube"), ("cube", "cube")):
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
    entities = [
        rsc.entity.CosmicBackgroundUniform("sky", texture_reference="sky#color"),
        rsc.entity.FloorFlat("floor", material_reference="ground"),
        rsc.entity.StructureVoxel(
            "cube",
            origin_position=(0.0, 0.0, 0.02),
            material_references=["cube"],
            bodies=[
                rsc.entity.StructureVoxelBody.box(
                    CUBE,
                    CUBE,
                    CUBE,
                    voxel_size=VOXEL,
                    relative_origin_position=(-size / 2, -size / 2, 0.0),
                )
            ],
        ),
    ]

    simulation = rsc.simulation.Simulation(
        environment=rsc.simulation.Environment(gravity_acceleration=-9.81),
        runtime=rsc.simulation.Runtime(log_level="debug"),
        control=rsc.simulation.Control(frequency=DISPLAY_HZ),
        vision=rsc.simulation.Vision(
            #restir=rsc.simulation.VisionReSTIR(is_enabled=True),
            denoise=rsc.simulation.VisionDenoiseSVGF(atrous_iterations=3),
        ),
    )
    return rsc.World(
        "vr",
        simulation=simulation,
        resources=resources,
        essences=essences,
        entities=entities,
    )


def main() -> None:
    # """Start the world and present its stereo camera to the headset until it exits."""
    engine = rs.RiseOrigin("cuda:0")
    handle, callback = engine.hello(build())
    session = rs.RS_WorldSession(handle, callback)
    session.add_component(
        "vr",
        rs.RS_WorldSessionComponentVR("vr-camera"),
        scope=rs.RSE_WorldSessionScope.RSE_WORLD_SESSION_SCOPE_FULL,
    )
    session.run()
    # import os
    # import sys
    # sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # import _common
    # _common.launch(build(), None, camera_focus=(0.0, 0.0, 0.08))

if __name__ == "__main__":
    main()
