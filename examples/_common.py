# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Shared helpers for the rise-origin example gallery.

Every example characterizes a world with ``rise_origin.rsc`` and shows it with
``rise_origin.runtime.viewer.Viewer``. These helpers cut the boilerplate the
examples have in common: a standard ground plane and sky, the material wiring
(each material is a handle plus a physical property, an optical property, a
color texture, and the pairwise friction/restitution between every material),
and a launcher that picks the run mode from environment variables so the same
script can open a window or render gallery media.

Environment variables read by :func:`launch`:

* ``RISE_DEVICE`` sets the device string (default ``"cuda:0"``; empty for the
  default device).
* ``RISE_MODE`` selects ``"interactive"`` (default), ``"record"``, or
  ``"headless"``.
* ``RISE_OUT`` is the output mp4 path for record mode (default ``"out.mp4"``).
* ``RISE_DURATION`` is the seconds to record or run in non-interactive modes
  (default 8).
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import rise_origin as rs
from rise_origin import rsc
from rise_origin.runtime.viewer import Viewer

# A clean, off-white sky keeps the scene calm and the colors readable.
SKY = (236, 234, 228)

# Neutral light floor.
FLOOR_COLOR = (206, 204, 198)

# Black grid lines on the floor, for a sense of scale and motion.
GRID_LINE_COLOR = (0, 0, 0)

# Shared "Refreshing Summer Fun" palette.
PALETTE = {
    "white": (235, 235, 235),
    "sky_blue": (142, 202, 230),
    "blue_green": (33, 158, 188),
    "deep_blue": (2, 48, 71),
    "amber": (255, 183, 3),
    "tiger_orange": (251, 133, 0),
}
PALETTE_CYCLE = list(PALETTE.values())


@dataclass
class MaterialSpec:
    """A compact description of one material the gallery turns into rsc essences.

    One spec expands into a material handle, a physical property (rigid/soft/
    fixed), an optical property, and a constant-color albedo texture. The
    pairwise friction and restitution between every pair of specs are generated
    by :func:`make_world`.

    Attributes:
        name: Unique material name; structures reference it by this name.
        color: Byte RGB surface color, 0-255 per channel.
        kind: Physics kind, one of ``"rigid"``, ``"soft"``, ``"fixed"``, or
            ``"rigid_fixed"`` (an anchored rigid body).
        density: Mass density in kg/m^3 (rigid and soft).
        elastic_modulus: Young's modulus in pascals (soft only).
        fail_stress: Stress in pascals at which soft links break; ``None`` keeps
            the engine default (effectively unbreakable).
        poissons_ratio: Poisson's ratio (soft only).
        min_expansion: Minimum actuation expansion factor (soft only).
        max_expansion: Maximum actuation expansion factor (soft only).
        friction: ``(u_static, u_dynamic)`` friction coefficients; pair friction
            is the average of the two materials in contact.
        restitution: Bounciness in [0, 1]; pair restitution is the average.
        roughness: Surface roughness in [0, 1]; ``None`` keeps the engine's matte
            default and adds no roughness/metalness texture.
        metalness: Metalness in [0, 1]; only used when ``roughness`` is set.
        is_transparent: Whether the surface refracts (clear glass-like).
        ior: Index of refraction, used when ``is_transparent``.
        emission_color: Byte RGB glow color; ``None`` is non-emissive.
        emission_intensity: Scalar multiplier on the emission color.
    """

    name: str
    color: Tuple[int, int, int]
    kind: str = "rigid"
    density: float = 1000.0
    elastic_modulus: float = 5.0e5
    fail_stress: Optional[float] = None
    poissons_ratio: float = 0.35
    min_expansion: float = -1.0
    max_expansion: float = 1.0
    friction: Tuple[float, float] = (1.0, 0.8)
    restitution: float = 0.1
    roughness: Optional[float] = None
    metalness: float = 0.0
    is_transparent: bool = False
    ior: float = 1.5
    emission_color: Optional[Tuple[int, int, int]] = None
    emission_intensity: float = 0.0


def make_world(
        name: str,
        *,
        materials: Sequence[MaterialSpec] = (),
        structures: Sequence = (),
        signals: Sequence = (),
        controllers: Sequence = (),
        records: Sequence = (),
        essences: Sequence = (),
        resources: Sequence = (),
        gravity: float = -9.81,
        control_hz: float = 30.0,
        record_hz: float = 30.0,
        time: Optional["rsc.simulation.Time"] = None,
        soft_body: Optional["rsc.simulation.SoftBody"] = None,
        rigid_body: Optional["rsc.simulation.RigidBody"] = None,
        contact: Optional["rsc.simulation.Contact"] = None,
        sky: Optional[object] = None,
        has_floor: bool = True,
) -> rsc.World:
    """Build a world with a ground plane, sky, materials, and your structures.

    Args:
        name: Unique world name.
        materials: Material specs to wire into essences and textures.
        structures: Entity builders to place in the world (e.g. structures).
        signals: Signal builders for the world.
        controllers: Controller builders for the world.
        records: Record builders for the world.
        essences: Pre-built essences to add alongside the material specs (e.g.
            the per-link materials an importer extracted).
        resources: Pre-built resources to add alongside the material textures.
        gravity: Gravitational acceleration along Z in m/s^2 (negative is down).
        control_hz: Control callback frequency in Hz.
        record_hz: Record sampling frequency in Hz.
        time: Optional :class:`rsc.simulation.Time` to override the time step.
        soft_body: Optional :class:`rsc.simulation.SoftBody` solver config.
        rigid_body: Optional :class:`rsc.simulation.RigidBody` solver config.
        contact: Optional :class:`rsc.simulation.Contact` broad-phase config.
        sky: A cosmic-background entity to use as the sky; defaults to a calm
            gradient with a soft overhead sun.
        has_floor: Whether to add the grid ground plane and its material.

    Returns:
        A populated :class:`rsc.World`.
    """
    resources = list(resources)
    # Caller-supplied properties may reference MaterialSpec handles generated
    # below (for example, per-voxel optical atlas entries). Keep their relative
    # order, but append them only after all generated material handles exist.
    trailing_essences = list(essences)
    essences = []
    entities = [sky if sky is not None else _default_sky()]

    specs = list(materials)
    if has_floor:
        resources.append(
            rsc.resource.TextureGrid(
                "ground#grid",
                fill_color=_rgb(FLOOR_COLOR),
                line_color=_rgb(GRID_LINE_COLOR),
                x_spacing=1.0,
                y_spacing=1.0,
                line_width=0.02,
            )
        )
        entities.append(rsc.entity.FloorFlat("floor", material_reference="ground"))
        specs.insert(0, _GROUND)

    for spec in specs:
        albedo = "ground#grid" if spec is _GROUND else f"{spec.name}#albedo"
        if spec is not _GROUND:
            resources.append(
                rsc.resource.TextureConstant(albedo, value=_rgb(spec.color))
            )
        rm = ""
        if spec is not _GROUND and spec.roughness is not None:
            rm = f"{spec.name}#rm"
            resources.append(
                rsc.resource.TextureConstant(
                    rm, value=(spec.roughness, spec.metalness, 0.0)
                )
            )
        emit = ""
        if spec is not _GROUND and spec.emission_color is not None:
            emit = f"{spec.name}#emit"
            resources.append(
                rsc.resource.TextureConstant(emit, value=_rgb(spec.emission_color))
            )
        essences.append(rsc.essence.Material(spec.name))
        essences.extend(_physical_properties(spec))
        essences.append(
            rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
                f"{spec.name}#optic",
                material_reference=spec.name,
                albedo_texture_reference=albedo,
                roughness_metalness_texture_reference=rm,
                is_transparent=spec.is_transparent,
                ior=spec.ior,
                emission_texture_reference=emit,
                emission_intensity=spec.emission_intensity,
            )
        )

    essences.extend(_pair_properties(specs))
    essences.extend(trailing_essences)
    entities.extend(structures)
    simulation = rsc.simulation.Simulation(
        environment=rsc.simulation.Environment(gravity_acceleration=gravity),
        control=rsc.simulation.Control(frequency=control_hz),
        record=rsc.simulation.Record(frequency=record_hz),
        time=time,
        soft_body=soft_body,
        rigid_body=rigid_body,
        contact=contact,
        vision=rsc.simulation.Vision(
            # restir=rsc.simulation.VisionReSTIR(is_enabled=False),
            denoise=rsc.simulation.VisionDenoiseSVGF(),
        ),
        runtime=rsc.simulation.Runtime(log_level="info"),
    )
    return rsc.World(
        name,
        simulation=simulation,
        resources=resources,
        essences=essences,
        entities=entities,
        signals=signals,
        controllers=controllers,
        records=records,
    )


def controller_by_name(controllers, name: str):
    """Return the controller with the given name from a callback's list, or None."""
    for controller in controllers:
        if controller.name() == name:
            return controller
    return None


def launch(
        world: rsc.World, callback: Optional[Callable] = None, **viewer_kwargs
) -> None:
    """Run an example, choosing the mode from environment variables.

    Opens an interactive window by default; set ``RISE_MODE=record`` to render
    an mp4 (``RISE_OUT``) or ``RISE_MODE=headless`` to run without a window.

    Args:
        world: The fully characterized world to show.
        callback: Per-step control callback ``callback(time, controllers)``.
        **viewer_kwargs: Forwarded to
            :class:`rise_origin.runtime.viewer.Viewer` (e.g. ``camera_focus``,
            ``samples_per_pixel``, ``width``, ``height``).
    """
    device = os.environ.get("RISE_DEVICE", "cuda:0") or None
    mode = os.environ.get("RISE_MODE", "interactive")
    duration = float(os.environ.get("RISE_DURATION", "8"))

    if "camera_position" not in viewer_kwargs:
        viewer_kwargs["camera_position"] = (-0.6, 0.0, 0.4)
    viewer = Viewer(instance=(device, world), callback=callback, **viewer_kwargs)
    if mode == "record":
        viewer.run_record(os.environ.get("RISE_OUT", "out.mp4"), duration=duration)
    elif mode == "headless":
        viewer.run_headless(duration=duration)
    else:
        viewer.run_interactive()
    viewer.wait()


# The fixed ground material, paired against every structure material in contact.
_GROUND = MaterialSpec("ground", color=FLOOR_COLOR, kind="fixed", friction=(1.0, 0.9))


def _rgb(color: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert a byte RGB color to a float RGB tuple, 0-1 per channel."""
    red, green, blue = color
    return (red / 255.0, green / 255.0, blue / 255.0)


def _default_sky():
    """A calm zenith-to-horizon gradient sky with a soft overhead sun."""
    return rsc.entity.CosmicBackgroundGradient(
        zenith_color=(175, 210, 235),
        horizon_color=SKY,
        sun_direction=(0.5, 0.35, 1.0),
        sun_color=(255, 246, 232),
    )


def _physical_properties(spec: MaterialSpec):
    """Build the physical-property essence(s) for one material spec.

    A spec yields one property, except ``"rigid_fixed"``, which attaches both a
    rigid and a fixed property so the engine forms an anchored (immovable) rigid
    body, the static frame a motor hinge reacts against.
    """
    if spec.kind == "rigid":
        return [
            rsc.essence.MaterialPhysicalPropertyRigid(
                f"{spec.name}#phys", material_reference=spec.name, density=spec.density
            )
        ]
    if spec.kind == "fixed":
        return [
            rsc.essence.MaterialPhysicalPropertyFixed(
                f"{spec.name}#phys", material_reference=spec.name
            )
        ]
    if spec.kind == "soft":
        return [
            rsc.essence.MaterialPhysicalPropertySoft(
                f"{spec.name}#phys",
                material_reference=spec.name,
                elastic_modulus=spec.elastic_modulus,
                fail_stress=spec.fail_stress,
                density=spec.density,
                poissons_ratio=spec.poissons_ratio,
                min_expansion=spec.min_expansion,
                max_expansion=spec.max_expansion,
            )
        ]
    if spec.kind == "rigid_fixed":
        return [
            rsc.essence.MaterialPhysicalPropertyRigid(
                f"{spec.name}#phys_rigid",
                material_reference=spec.name,
                density=spec.density,
            ),
            rsc.essence.MaterialPhysicalPropertyFixed(
                f"{spec.name}#phys_fixed", material_reference=spec.name
            ),
        ]
    raise ValueError(
        "material kind must be 'rigid', 'soft', 'fixed', or 'rigid_fixed', "
        f"got {spec.kind!r}"
    )


def _pair_properties(specs: Sequence[MaterialSpec]):
    """Friction and restitution essences for every unordered pair of materials."""
    essences = []
    for i, first in enumerate(specs):
        for second in specs[i:]:
            pair = f"{first.name}_{second.name}"
            u_static = (first.friction[0] + second.friction[0]) / 2.0
            u_dynamic = (first.friction[1] + second.friction[1]) / 2.0
            essences.append(
                rsc.essence.MaterialPairPropertyCoulombFriction(
                    f"{pair}#friction",
                    first_material_reference=first.name,
                    second_material_reference=second.name,
                    u_static=u_static,
                    u_dynamic=u_dynamic,
                )
            )
            essences.append(
                rsc.essence.MaterialPairPropertyRestitution(
                    f"{pair}#restitution",
                    first_material_reference=first.name,
                    second_material_reference=second.name,
                    restitution=(first.restitution + second.restitution) / 2.0,
                )
            )
    return essences
