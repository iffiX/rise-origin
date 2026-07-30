# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The stratus field: many wide low slabs scattered in a box.

The shape lives in ``algorithm/cloud/blob_cloud.py`` (Bouthors and Neyret
2004); this module scatters many single stratus, each its own structure,
inside a ``span`` box in meters. ``size`` sets how wide each, ``span`` the
box, ``cloud_num`` how many, ``is_rainy`` the material, and ``voxel_size``
the voxel resolution.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm import BlobCloudParams

from ...material.nature import cloud
from .build import CloudBuild

MODULE = "rise_origin.nature.cloud.stratus_field"

# The size in meters the base blob values describe, the scale the dials
# ride.
REFERENCE_SIZE = 2000.0


class _StratusFieldShape:
    """The stratus field dial-to-parameter mapping."""

    @staticmethod
    def make_params(size: float) -> BlobCloudParams:
        """Return the blob values scaled to one cloud's size.

        Args:
            size: The width of one cloud (unit: m).
        """
        scale = size / REFERENCE_SIZE
        return BlobCloudParams(
            root_num=10,
            root_radius_min=250.0 * scale,
            root_radius_max=400.0 * scale,
            root_plane_jitter_scale=1.5,
            root_rise_scale=0.0,
            level_num=3,
            radius_ratio=1.0 / 3.0,
            radius_jitter=0.3,
            flatten_min=0.85,
            flatten_max=1.0,
            furrow=0.5,
            enlarge=10.0,
            blend_by_level=(0.0, 0.0, 1.0),
            floor_height=100.0 * scale,
            floor_stiffness=80.0 * scale,
        )


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        size=cr.Float(
            800.0, 3000.0, default=1600.0, doc="The width of each cloud (unit: m)."
        ),
        span=cr.Bounds(
            -10000.0,
            10000.0,
            default=((-8000.0, -8000.0, -200.0), (8000.0, 8000.0, 200.0)),
            doc="The box the clouds scatter in (unit: m).",
        ),
        cloud_num=cr.Int(1, 200, default=20, doc="How many clouds the field scatters."),
        is_rainy=cr.Bool(
            default=False,
            doc="Whether the clouds take the translucent rainy material "
            "instead of opaque white.",
        ),
        voxel_size=cr.VoxelSize(
            default="20m", min="10m", max="50m", doc="The emitted voxel edge."
        ),
    ),
    dependencies=(f"{cloud.MODULE}@^1",),
    tags=("nature", "cloud", "stratus", "field"),
)
def stratus_field(editor_access, settings):
    """A scattered field of wide low stratus slabs, white or rainy grey.

    ``size`` sets each cloud's width, ``span`` the box the clouds scatter
    in, ``cloud_num`` how many, ``is_rainy`` the material, and
    ``voxel_size`` the resolution. Each cloud is its own fixed structure,
    so the scene can move or drop one on its own.
    """
    return CloudBuild.make_field_draft(
        editor_access,
        settings,
        kind_name="stratus_field",
        params=_StratusFieldShape.make_params(settings.size),
        span=settings.span,
        cloud_num=settings.cloud_num,
        voxel_size=settings.voxel_size,
        is_rainy=settings.is_rainy,
    )
