# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""One reading of where a voxel convex lens brings light to a focus.

A plano-convex lens is painted into voxels. A small emissive ball stands in
front of it and one camera stands far behind, looking back through the glass
at the aperture.

The camera's own rays are gathered by the lens onto a point in front of it,
so the aperture lights up brightly when the ball sits at that point. That is
the camera's conjugate. Sweeping the ball through it and taking the brightest
aperture measures where the lens focuses.

Reading the aperture as lit area rather than as a brightness is deliberate.
Every ray that finds the ball returns the same radiance, so the reading is a
count of rays that arrived, which needs no scale and cannot be flattened by
a tone curve.

This file takes one reading. Use sweep.py to run the sweep and fit the peak.

Run one reading::

    OPTICS_SOURCE=180 python examples/physics_bench/optics/focal_position/main.py
"""

import math
import os
import sys
from typing import Optional, Tuple

import numpy as np

from rise_origin import rsc

EXAMPLES_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
sys.path.insert(0, EXAMPLES_ROOT)
import _common  # noqa: E402

# The bench, in voxels. A plano-convex has one curved face and one flat one,
# so a single staircase surface is under test.
VOXEL = 0.001
LENS_RADIUS = float(os.environ.get("OPTICS_RADIUS", "100.0"))
LENS_INDEX = float(os.environ.get("OPTICS_INDEX", "1.5"))
LENS_SEMI_APERTURE = float(os.environ.get("OPTICS_APERTURE", "30.0"))

# The camera stands far enough back that its cone barely opens across the
# aperture, and near enough that the conjugate stays well clear of f.
CAMERA_DISTANCE = 1500.0
SOURCE_DISTANCE = float(os.environ.get("OPTICS_SOURCE", "180.0"))
SOURCE_RADIUS = 3.0

CAMERA_PIXELS = 48
CAMERA_SAMPLES = int(os.environ.get("OPTICS_SAMPLES", "256"))

# Kept low so no pixel clips. A clipped frame turns the reading into "how
# many pixels saw anything", which peaks when the lens is out of focus.
SOURCE_EMISSION = float(os.environ.get("OPTICS_EMISSION", "0.05"))

IS_SMOOTH = os.environ.get("OPTICS_SMOOTH", "1") != "0"
HAS_LENS = os.environ.get("OPTICS_NO_LENS", "") == ""

CONTROLLER = "optics#brain"
GLASS_MATERIAL = "optics.glass"
SOURCE_MATERIAL = "optics.source"


def get_focal_length() -> float:
    """Paraxial focal length of the plano-convex lens, in voxels."""
    return LENS_RADIUS / (LENS_INDEX - 1.0)


def get_paraxial_conjugate() -> float:
    """Where the paraxial formula puts the camera's conjugate, in voxels."""
    return 1.0 / (1.0 / get_focal_length() - 1.0 / CAMERA_DISTANCE)


def get_lens_thickness() -> float:
    """Axial thickness that clears the curved face, in voxels."""
    sag = LENS_RADIUS - math.sqrt(
        LENS_RADIUS * LENS_RADIUS - LENS_SEMI_APERTURE * LENS_SEMI_APERTURE
    )
    return float(math.ceil(sag) + 4)


def get_camera_view_degrees() -> float:
    """View angle that frames the aperture with dark room around it."""
    half = math.degrees(math.atan(LENS_SEMI_APERTURE / CAMERA_DISTANCE))
    return 2.6 * half


def _refract(direction, normal, eta):
    """Snell's law in vectors, on a stack of rays.

    Args:
        direction: Unit directions, shaped (n, 3).
        normal: Unit normals turned to oppose each direction, shaped (n, 3).
        eta: Index being left over index being entered.

    Returns:
        Unit outgoing directions, and a mask of the rays that did not turn
        back by total internal reflection.
    """
    cos_in = -np.sum(normal * direction, axis=1)
    k = 1.0 - eta * eta * (1.0 - cos_in * cos_in)
    is_through = k >= 0.0
    out = eta * direction + ((eta * cos_in - np.sqrt(np.clip(k, 0.0, None)))[:, None]) * normal
    length = np.linalg.norm(out, axis=1, keepdims=True)
    length[length == 0.0] = 1.0
    return out / length, is_through


def get_exact_conjugate(ray_num: int = 8001) -> Tuple[float, float]:
    """Trace the ideal lens with no voxels and find where it gathers light.

    The curved face is the sphere of radius ``LENS_RADIUS`` centred on the
    axis at ``x = LENS_RADIUS``, so its vertex sits at the origin and the
    glass lies inside it. Rays leave the camera, cross the flat face, cross
    the curved face, and run on. Spherical aberration and the finite aperture
    are inside this answer rather than left over as error, which is what the
    paraxial formula cannot do.

    Args:
        ray_num: How many rays to lay across the aperture.

    Returns:
        The conjugate distance in voxels, and the share of aperture rays that
        pass within the ball radius there.
    """
    thickness = get_lens_thickness()
    centre = np.array([LENS_RADIUS, 0.0, 0.0])
    camera = np.array([thickness + CAMERA_DISTANCE, 0.0, 0.0])

    heights = np.linspace(-LENS_SEMI_APERTURE, LENS_SEMI_APERTURE, ray_num)
    heights = heights[np.abs(heights) > 1e-9]
    entry = np.stack(
        [np.full(heights.shape, thickness), heights, np.zeros_like(heights)], axis=1
    )
    direction = entry - camera
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    flat_normal = np.tile(np.array([1.0, 0.0, 0.0]), (len(heights), 1))
    inside, is_through = _refract(direction, flat_normal, 1.0 / LENS_INDEX)

    # The curved face, as the crossing of the sphere the glass fills.
    offset = entry - centre
    b = np.sum(offset * inside, axis=1)
    c = np.sum(offset * offset, axis=1) - LENS_RADIUS * LENS_RADIUS
    disc = b * b - c
    is_through &= disc >= 0.0
    root = np.sqrt(np.clip(disc, 0.0, None))
    travel = np.where(-b - root > 0.0, -b - root, -b + root)
    exit_point = entry + inside * travel[:, None]

    normal = (exit_point - centre) / LENS_RADIUS
    normal = np.where((np.sum(normal * inside, axis=1) > 0.0)[:, None], -normal, normal)
    outgoing, is_out = _refract(inside, normal, LENS_INDEX)
    is_through &= is_out
    is_through &= np.abs(exit_point[:, 1]) <= LENS_SEMI_APERTURE

    exit_point = exit_point[is_through]
    outgoing = outgoing[is_through]
    # Weight by height so each annulus of the aperture counts by its area.
    weight = np.abs(exit_point[:, 1])

    paraxial = get_paraxial_conjugate()
    best_distance, best_share = float("nan"), -1.0
    for distance in np.linspace(0.3 * paraxial, 2.0 * paraxial, 2001):
        height = (
            exit_point[:, 1]
            + outgoing[:, 1] * (-distance - exit_point[:, 0]) / outgoing[:, 0]
        )
        share = float(
            weight[np.abs(height) <= SOURCE_RADIUS].sum() / max(weight.sum(), 1e-12)
        )
        if share > best_share:
            best_share, best_distance = share, float(distance)
    return best_distance, best_share


def _make_material(name: str, color, *, index: Optional[float], emission: float):
    """One material and the textures it points at.

    Args:
        name: What the material is called.
        color: Byte RGB the surface shows.
        index: Refractive index, or ``None`` for an opaque surface.
        emission: How brightly it burns, 0 for a surface that only reflects.
    """
    albedo = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    resources = [
        rsc.resource.TextureConstant(f"{name}#albedo", value=albedo),
        rsc.resource.TextureConstant(f"{name}#rough", value=(0.0, 0.0, 0.0)),
    ]
    essences = [
        rsc.essence.Material(name),
        rsc.essence.MaterialPhysicalPropertyFixed(
            f"{name}#phys", material_reference=name
        ),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            f"{name}#optic",
            material_reference=name,
            albedo_texture_reference=f"{name}#albedo",
            roughness_metalness_texture_reference=f"{name}#rough",
            is_transparent=index is not None,
            is_smooth_surface=index is not None and IS_SMOOTH,
            ior=index if index is not None else 1.0,
            emission_texture_reference=f"{name}#albedo" if emission > 0 else "",
            emission_radiance=emission,
        ),
    ]
    return essences, resources


def _make_lens_grid() -> np.ndarray:
    """Paint the plano-convex lens, as a material id per cell."""
    thickness = int(get_lens_thickness())
    span = int(math.ceil(LENS_SEMI_APERTURE)) * 2 + 3
    middle = span // 2
    grid = np.full((thickness, span, span), -1, dtype=np.int64)
    for ix in range(thickness):
        # Cell centres, so the surface falls where the sphere really is.
        x = ix + 0.5
        for iy in range(span):
            y = iy - middle
            for iz in range(span):
                z = iz - middle
                if y * y + z * z > LENS_SEMI_APERTURE * LENS_SEMI_APERTURE:
                    continue
                dx = x - LENS_RADIUS
                if dx * dx + y * y + z * z <= LENS_RADIUS * LENS_RADIUS:
                    grid[ix, iy, iz] = 0
    return grid


def _make_source_grid() -> np.ndarray:
    """Paint the small emissive ball, as a material id per cell."""
    radius = int(SOURCE_RADIUS)
    span = radius * 2 + 1
    axis = np.arange(span) - radius
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    ball = (xx * xx + yy * yy + zz * zz) <= radius * radius
    return np.where(ball, 0, -1).astype(np.int64)


def build():
    """Build the bench: lens, ball in front, camera far behind."""
    glass_essences, glass_resources = _make_material(
        GLASS_MATERIAL, (250, 252, 255), index=LENS_INDEX, emission=0.0
    )
    source_essences, source_resources = _make_material(
        SOURCE_MATERIAL, (255, 255, 255), index=None, emission=SOURCE_EMISSION
    )

    grid = _make_lens_grid()
    middle = grid.shape[1] // 2
    lens = rsc.entity.StructureVoxel(
        "lens",
        origin_position=(0.0, 0.0, 0.0),
        material_references=[GLASS_MATERIAL],
        is_fixed=True,
        bodies=[
            rsc.entity.StructureVoxelBody.from_grid(
                grid,
                segment_grid=grid.copy(),
                voxel_size=VOXEL,
                relative_origin_position=(0.0, -middle * VOXEL, -middle * VOXEL),
            )
        ],
    )

    source_grid = _make_source_grid()
    radius = source_grid.shape[0] // 2
    source = rsc.entity.StructureVoxel(
        "source",
        origin_position=(-SOURCE_DISTANCE * VOXEL, 0.0, 0.0),
        material_references=[SOURCE_MATERIAL],
        is_fixed=True,
        bodies=[
            rsc.entity.StructureVoxelBody.from_grid(
                source_grid,
                segment_grid=source_grid.copy(),
                voxel_size=VOXEL,
                relative_origin_position=(
                    -radius * VOXEL,
                    -radius * VOXEL,
                    -radius * VOXEL,
                ),
            )
        ],
    )

    camera = rsc.entity.CameraPerspectiveFree(
        "probe",
        origin_position=((get_lens_thickness() + CAMERA_DISTANCE) * VOXEL, 0.0, 0.0),
        orientation=(0.0, 0.0, 1.0, 0.0),
        image_width=CAMERA_PIXELS,
        image_height=CAMERA_PIXELS,
        samples_per_pixel=CAMERA_SAMPLES,
        max_depth=8,
        field_of_view=get_camera_view_degrees(),
        record=rsc.entity.CameraPerspectiveRecord(
            "probe#rec", has_rgb=True, target_controller_reference=CONTROLLER
        ),
    )

    structures = ([lens] if HAS_LENS else []) + [source, camera]
    return _common.make_world(
        "focal_position",
        essences=glass_essences + source_essences,
        resources=glass_resources + source_resources,
        structures=structures,
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
        has_floor=False,
        gravity=0.0,
        sky=rsc.entity.CosmicBackgroundGradient(
            zenith_color=(0, 0, 0),
            horizon_color=(0, 0, 0),
            sun_direction=(0.0, 0.0, 1.0),
            sun_color=(0, 0, 0),
        ),
    )


def main():
    """Take one reading and print it as a parsable line."""
    os.environ.setdefault("RISE_MODE", "headless")
    os.environ.setdefault("RISE_DURATION", "1.2")

    frames = []

    def control(time, controllers):
        controller = _common.controller_by_name(controllers, CONTROLLER)
        if controller is None or time < 0.5:
            return
        for record in controller.records():
            if record.is_valid() and record.width() > 1:
                frames.append(np.array(record.rgb.pixel(), copy=True))

    _common.launch(build(), control)

    if not frames:
        print(f"RESULT source={SOURCE_DISTANCE:.2f} flux=nan peak=nan clipped=nan")
        return

    image = np.mean(np.stack(frames[-8:]), axis=0)
    luminance = (
        0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    )
    print(
        f"RESULT source={SOURCE_DISTANCE:.2f} flux={float(luminance.mean()):.5f} "
        f"peak={float(luminance.max()):.3f} "
        f"clipped={float((luminance >= 254.0).mean()):.4f} "
        f"index={LENS_INDEX} radius={LENS_RADIUS} smooth={int(IS_SMOOTH)} "
        f"lens={int(HAS_LENS)}"
    )


if __name__ == "__main__":
    main()
