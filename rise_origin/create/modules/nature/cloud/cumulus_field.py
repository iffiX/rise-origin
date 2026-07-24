# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The cumulus field: many small puffy clouds scattered in a box.

The shape lives in ``algorithm/cloud/blob_cloud.py`` (Bouthors and Neyret
2004); this module scatters many single cumulus, each its own structure,
inside a ``span`` box in meters. ``cloud_num`` sets how many, ``size``
how wide each, ``is_rainy`` the opaque white or translucent grey
material, and ``voxel_size`` the voxel resolution.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm import BlobCloudParams

from ...material.nature import cloud
from .build import CloudBuild

MODULE = "rise_origin.nature.cloud.cumulus_field"

# The size in meters the paper's section 5 values describe; the size dial
# scales every blob length off this reference.
REFERENCE_SIZE = 1400.0


class _CumulusFieldShape:
    """The cumulus field dial-to-parameter mapping."""

    @staticmethod
    def make_params(size: float) -> BlobCloudParams:
        """Return the paper's blob values scaled to one cloud's size.

        Args:
            size: The width of one cloud (unit: m).
        """
        scale = size / REFERENCE_SIZE
        return BlobCloudParams(
            root_num=6,
            root_radius_min=200.0 * scale,
            root_radius_max=450.0 * scale,
            root_rise_scale=0.6,
            level_num=2,
            radius_ratio=0.5,
            radius_jitter=0.3,
            flatten_min=0.0,
            flatten_max=0.2,
            furrow=0.2,
            enlarge=10.0,
            blend_by_level=(0.0, 1.0),
            floor_height=100.0 * scale,
            floor_stiffness=100.0 * scale,
        )


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        size=cr.Float(400.0, 1500.0, default=400.0),
        span=cr.Bounds(
            -10000.0,
            10000.0,
            default=((-8000.0, -8000.0, -400.0), (8000.0, 8000.0, 400.0)),
        ),
        cloud_num=cr.Int(1, 400, default=100),
        is_rainy=cr.Bool(default=False),
        voxel_size=cr.VoxelSize(default="20m", min="10m", max="50m"),
    ),
    dependencies=(f"{cloud.MODULE}@^1",),
    tags=("nature", "cloud", "cumulus", "field"),
)
def cumulus_field(editor_access, settings):
    """A scattered field of small cumulus, opaque white or rainy grey.

    ``span`` sets the box the clouds scatter in, ``cloud_num`` how many,
    ``size`` each cloud's width, ``is_rainy`` the material, and
    ``voxel_size`` the resolution. Each cloud is its own fixed structure,
    so the scene can move or drop one on its own.
    """
    return CloudBuild.make_field_draft(
        editor_access,
        settings,
        kind_name="cumulus_field",
        params=_CumulusFieldShape.make_params(settings.size),
        span=settings.span,
        cloud_num=settings.cloud_num,
        voxel_size=settings.voxel_size,
        is_rainy=settings.is_rainy,
    )
