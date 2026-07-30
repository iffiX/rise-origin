# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The cumulus cloud: one puffy fair-weather cauliflower.

The shape lives in ``algorithm/cloud/blob_cloud.py`` (Bouthors and Neyret
2004); this module holds the paper's section 5 blob values, scaled by
``size`` so the same cloud resolves at any size. ``size`` sets the
cloud's width, ``voxel_size`` its voxel resolution.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm import BlobCloudParams

from ...material.nature import cloud
from .build import CloudBuild

MODULE = "rise_origin.nature.cloud.cumulus"

# The size in meters the paper's section 5 values describe; the size dial
# scales every blob length off this reference.
REFERENCE_SIZE = 1400.0


class _CumulusShape:
    """The cumulus dial-to-parameter mapping."""

    @staticmethod
    def make_params(size: float) -> BlobCloudParams:
        """Return the paper's blob values scaled to the size dial.

        Every length is the paper's section 5 value times the size over
        the reference size, so the blob ratios are the paper's at any
        cloud size.

        Args:
            size: The width of the cloud (unit: m).
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
        size=cr.Float(400.0, 2000.0, default=700.0, doc="The cloud width (unit: m)."),
        voxel_size=cr.VoxelSize(
            default="20m", min="10m", max="50m", doc="The emitted voxel edge."
        ),
    ),
    dependencies=(f"{cloud.MODULE}@^1",),
    tags=("nature", "cloud", "cumulus"),
)
def cumulus(editor_access, settings):
    """A single puffy fair-weather cumulus, opaque white.

    ``size`` sets the cloud width and ``voxel_size`` its resolution.
    The cloud is fixed and hangs where the scene places it.
    """
    return CloudBuild.make_draft(
        editor_access,
        settings,
        kind_name="cumulus",
        params=_CumulusShape.make_params(settings.size),
        voxel_size=settings.voxel_size,
    )
