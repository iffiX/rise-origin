# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The stratus cloud: one wide low flattened cauliflower slab.

The shape lives in ``algorithm/cloud/blob_cloud.py`` (Bouthors and Neyret
2004); this module reuses the cumulus blob model with strongly flattened
root bubbles scattered wide, so the cloud reads as a low slab rather than
a puffy dome. ``size`` sets its width, ``voxel_size`` its resolution.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm import BlobCloudParams

from ...material.nature import cloud
from .build import CloudBuild

MODULE = "rise_origin.nature.cloud.stratus"

# The size in meters the base blob values describe, the scale the dials
# ride.
REFERENCE_SIZE = 2000.0


class _StratusShape:
    """The stratus dial-to-parameter mapping."""

    @staticmethod
    def make_params(size: float) -> BlobCloudParams:
        """Return the blob values scaled to the size dial.

        Args:
            size: The width of the cloud (unit: m).
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
        size=cr.Float(1000.0, 4000.0, default=2000.0, doc="The cloud width (unit: m)."),
        voxel_size=cr.VoxelSize(
            default="20m", min="10m", max="50m", doc="The emitted voxel edge."
        ),
    ),
    dependencies=(f"{cloud.MODULE}@^1",),
    tags=("nature", "cloud", "stratus"),
)
def stratus(editor_access, settings):
    """A single wide low stratus slab, opaque grey-white.

    ``size`` sets the cloud width and ``voxel_size`` its resolution.
    The cloud is fixed and hangs where the scene places it.
    """
    return CloudBuild.make_draft(
        editor_access,
        settings,
        kind_name="stratus",
        params=_StratusShape.make_params(settings.size),
        voxel_size=settings.voxel_size,
    )
