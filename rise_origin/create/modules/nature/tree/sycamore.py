# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The sycamore: a big broad tree in camouflage flaking bark.

The thick pale trunk flakes in mottled patch layers, the signature of
a plane tree; the wide crown carries big lobed leaves.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm import (
    TreeBark,
    TreeBarkPatches,
    TreeLeaf,
    WeberPennParams,
)

from ...material.nature import plant
from .build import TreeBuild

MODULE = "rise_origin.nature.tree.sycamore"

SEASON_LEAF_DENSITY = {"spring": 0.85, "summer": 1.0, "autumn": 0.9, "winter": 0.0}

LEAF_AREA_BUDGET = 360.0
BODY_CAP = 150


class _SycamoreShape:
    """The sycamore's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the broad thick-trunk sycamore preset under the dials.

        Args:
            settings: The validated sycamore settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=2,
            base_size=0.12,
            scale=settings.height,
            levels=3,
            ratio=0.022,
            ratio_power=1.3,
            taper=(0.88, 0.95, 1.0),
            flare=1.3,
            base_splits=2,
            seg_splits=(0.25, 0.12, 0.0),
            split_angle=(28.0, 18.0, 0.0),
            split_angle_v=(8.0, 8.0, 0.0),
            length=(1.0, 0.5, 0.3),
            length_v=(0.0, 0.08, 0.05),
            curve_res=(10, 8, 4),
            curve=(0.0, 30.0, -20.0),
            curve_back=(0.0, -50.0, 0.0),
            curve_v=(120.0, 130.0, 90.0),
            down_angle=(0.0, 42.0, 40.0),
            down_angle_v=(0.0, -35.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 77.0),
            branches=(1, int(11 + 6 * age), int(9 + 6 * age)),
            leaves=int(round((18 + 9 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.25,
            attraction_up=0.5,
            max_stems=6000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the sycamore's camouflage patch bark.

        Args:
            settings: The validated sycamore settings.
        """
        return TreeBark(
            detail=TreeBarkPatches(
                depth=0.018,
                cell=0.3,
                optical_mid_levels=(3, 2, 0, 4),
            ),
            grain_amplitude=0.006,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the sycamore's big lobed leaf.

        Args:
            settings: The validated sycamore settings.
        """
        return TreeLeaf(outline="maple_star", length=0.22, width_ratio=0.9)


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(8.0, 20.0, default=14.0),
        age=cr.Float(0.0, 1.0, default=0.7),
        lean=cr.Float(0.0, 0.6, default=0.0),
        season=cr.Choice("spring", "summer", "autumn", "winter", default="summer"),
        leaf_fall=cr.Float(0.0, 1.0, default=0.0),
        is_leaf_soft=cr.Bool(default=False),
        trunk_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
        canopy_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
    ),
    dependencies=(f"{plant.MODULE}@^1",),
    tags=("nature", "tree", "sycamore", "plane"),
)
def sycamore(editor_access, settings):
    """A sycamore, big and broad on a thick camouflage-flaked trunk.

    ``season`` picks the leaf optical mids; winter stands bare.
    ``is_leaf_soft`` lets the world sway the canopy and ``leaf_fall``
    keeps that share of the naturally loose leaf parts unattached.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="sycamore",
        params=_SycamoreShape.make_params(settings),
        bark=_SycamoreShape.make_bark(settings),
        leaf=_SycamoreShape.make_leaf(settings),
        leaf_optical_mids=plant.SEASON_LEAF_MIDS[settings.season],
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
        tile_extent=3.0,
    )
