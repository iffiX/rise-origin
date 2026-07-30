# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The willow: the Weber-Penn paper's weeping willow, approximated.

The paper shapes its weeping willow with envelope pruning; this preset
reaches the same silhouette through strong downward attraction and long
drooping second-level stems instead, a documented deviation. Fronds
sweep low, held just above the ground by the growth clearance band.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm import (
    TreeBark,
    TreeBarkFurrows,
    TreeLeaf,
    WeberPennParams,
)

from ...material.nature import plant
from .build import TreeBuild

MODULE = "rise_origin.nature.tree.willow"

SEASON_LEAF_DENSITY = {"spring": 0.9, "summer": 1.0, "autumn": 0.9, "winter": 0.0}

LEAF_AREA_BUDGET = 112.0
BODY_CAP = 100


class _WillowShape:
    """The willow's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the weeping willow preset under the dials.

        Args:
            settings: The validated willow settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=3,
            base_size=0.05,
            scale=settings.height,
            levels=3,
            ratio=0.02,
            ratio_power=2.0,
            taper=(0.85, 1.0, 1.0),
            flare=0.75,
            base_splits=2,
            seg_splits=(0.15, 0.2, 0.15),
            split_angle=(6.0, 30.0, 45.0),
            split_angle_v=(0.0, 10.0, 20.0),
            length=(0.85, 0.55, 0.9),
            length_v=(0.0, 0.1, 0.1),
            curve_res=(6, 9, 8),
            curve=(0.0, 20.0, 10.0),
            curve_back=(0.0, 45.0, 0.0),
            curve_v=(25.0, 45.0, 20.0),
            down_angle=(0.0, 52.0, 30.0),
            down_angle_v=(0.0, 15.0, 10.0),
            rotate=(0.0, -120.0, -120.0, 140.0),
            branches=(1, int(10 + 5 * age), int(6 + 4 * age)),
            leaves=int(round((26 + 12 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.1,
            attraction_up=-2.2,
            ground_clearance=0.25,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the willow's ropey gray-brown bark.

        Args:
            settings: The validated willow settings.
        """
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.034,
                width=0.05,
                wobble_length=0.8,
            ),
            moss_coverage=0.2,
            moss_height=2.0,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the willow's long narrow leaf.

        Args:
            settings: The validated willow settings.
        """
        return TreeLeaf(outline="strip", length=0.15, width_ratio=0.22)

    @staticmethod
    def make_leaf_mids(settings) -> tuple:
        """Return the season's leaf optical mids; willow turns gold in autumn.

        Args:
            settings: The validated willow settings.
        """
        if settings.season == "autumn":
            return plant.GOLD_MIDS
        return plant.SEASON_LEAF_MIDS[settings.season]


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(
            5.0, 18.0, default=11.0, doc="The tree's overall height (unit: m)."
        ),
        age=cr.Float(
            0.0,
            1.0,
            default=0.7,
            doc="The age of the tree, from young and sparse at 0 to full "
            "and mature at 1.",
        ),
        lean=cr.Float(
            0.0,
            0.6,
            default=0.0,
            doc="How far the tree tilts off vertical, toward a random "
            "horizontal direction (unit: radians).",
        ),
        season=cr.Choice(
            "spring",
            "summer",
            "autumn",
            "winter",
            default="summer",
            doc="Picks the leaf color for the season, gold in autumn and "
            "bare in winter.",
        ),
        leaf_fall=cr.Float(
            0.0,
            1.0,
            default=0.0,
            doc="The share of the naturally loose leaf parts kept "
            "unattached and ready to shed.",
        ),
        is_leaf_soft=cr.Bool(
            default=False,
            doc="True swaps the fronds onto the soft leaf material so the "
            "world can simulate and sway them.",
        ),
        trunk_voxel_size=cr.VoxelSize(
            default="2cm",
            min="1cm",
            max="10cm",
            doc="The emitted wood voxel edge.",
        ),
        canopy_voxel_size=cr.VoxelSize(
            default="2cm",
            min="1cm",
            max="10cm",
            doc="The emitted leaf voxel edge.",
        ),
    ),
    dependencies=(f"{plant.MODULE}@^1",),
    tags=("nature", "tree", "willow"),
)
def willow(editor_access, settings):
    """A weeping willow, long fronds sweeping toward the ground.

    Settings:
        height: The tree's overall height (unit: m).
        age: The age of the tree, from young and sparse at 0 to full
            and mature at 1.
        lean: How far the tree tilts off vertical, toward a random
            horizontal direction (unit: radians).
        season: Picks the leaf color for the season, gold in autumn
            and bare in winter.
        leaf_fall: The share of the naturally loose leaf parts kept
            unattached and ready to shed.
        is_leaf_soft: True swaps the fronds onto the soft leaf
            material so the world can simulate and sway them.
        trunk_voxel_size: The emitted wood voxel edge.
        canopy_voxel_size: The emitted leaf voxel edge.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="willow",
        params=_WillowShape.make_params(settings),
        bark=_WillowShape.make_bark(settings),
        leaf=_WillowShape.make_leaf(settings),
        leaf_optical_mids=_WillowShape.make_leaf_mids(settings),
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
