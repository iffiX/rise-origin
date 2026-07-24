# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The birch: slim and airy on papery white bark.

The white bark carries dark flecks and peeling patches; the light
crown turns gold in autumn.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm import (
    TreeBark,
    TreeBarkBands,
    TreeLeaf,
    WeberPennParams,
)

from ...material.nature import plant
from .build import TreeBuild

MODULE = "rise_origin.nature.tree.birch"

SEASON_LEAF_DENSITY = {"spring": 0.85, "summer": 1.0, "autumn": 0.9, "winter": 0.0}

LEAF_AREA_BUDGET = 96.0
BODY_CAP = 100


class _BirchShape:
    """The birch's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the slim airy birch preset under the dials.

        Args:
            settings: The validated birch settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=7,
            base_size=0.4,
            scale=settings.height,
            levels=3,
            ratio=0.009,
            ratio_power=1.2,
            taper=(0.75, 0.9, 1.0),
            flare=0.4,
            seg_splits=(0.1, 0.1, 0.0),
            split_angle=(15.0, 15.0, 0.0),
            split_angle_v=(5.0, 5.0, 0.0),
            length=(1.0, 0.32, 0.45),
            length_v=(0.0, 0.05, 0.1),
            curve_res=(11, 5, 3),
            curve=(0.0, -20.0, -20.0),
            curve_v=(18.0, 60.0, 80.0),
            down_angle=(0.0, 50.0, 38.0),
            down_angle_v=(0.0, -40.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 77.0),
            branches=(1, int(14 + 8 * age), int(12 + 7 * age)),
            leaves=int(round((24 + 10 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.1,
            attraction_up=0.6,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the birch's white bark with dark flecks and peel.

        Args:
            settings: The validated birch settings.
        """
        return TreeBark(
            detail=TreeBarkBands(
                depth=0.012,
                length=0.16,
                wobble=0.2,
                coverage=0.3,
            ),
            grain_amplitude=0.008,
            peel_coverage=0.18,
            peel_cell=0.3,
            optical_mid_base=plant.WOOD_BIRCH_MID,
            optical_mid_light=plant.WOOD_BIRCH_MID,
            optical_mid_crevice=1,
            optical_mid_peel=1,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the birch's small pointed leaf.

        Args:
            settings: The validated birch settings.
        """
        return TreeLeaf(outline="ellipse", length=0.11, width_ratio=0.75)

    @staticmethod
    def make_leaf_mids(settings) -> tuple:
        """Return the season's leaf optical mids; birch turns gold in autumn.

        Args:
            settings: The validated birch settings.
        """
        if settings.season == "autumn":
            return plant.GOLD_MIDS
        return plant.SEASON_LEAF_MIDS[settings.season]


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(4.0, 16.0, default=11.0),
        age=cr.Float(0.0, 1.0, default=0.7),
        lean=cr.Float(0.0, 0.6, default=0.0),
        season=cr.Choice("spring", "summer", "autumn", "winter", default="summer"),
        leaf_fall=cr.Float(0.0, 1.0, default=0.0),
        is_leaf_soft=cr.Bool(default=False),
        trunk_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
        canopy_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
    ),
    dependencies=(f"{plant.MODULE}@^1",),
    tags=("nature", "tree", "birch"),
)
def birch(editor_access, settings):
    """A birch, slim white trunk under an airy light crown.

    ``season`` picks the leaf optical mids, gold in autumn and bare in winter;
    ``is_leaf_soft`` lets the world sway the canopy and ``leaf_fall``
    keeps that share of the naturally loose leaf parts unattached.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="birch",
        params=_BirchShape.make_params(settings),
        bark=_BirchShape.make_bark(settings),
        leaf=_BirchShape.make_leaf(settings),
        leaf_optical_mids=_BirchShape.make_leaf_mids(settings),
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
