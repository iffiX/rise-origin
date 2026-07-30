# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The aspen: the Weber-Penn paper's quaking aspen, golden in autumn.

The preset follows the paper's appendix column for the quaking aspen,
scaled to the voxel world's stem budget; the round fluttering leaves
turn gold in autumn.
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

MODULE = "rise_origin.nature.tree.aspen"

SEASON_LEAF_DENSITY = {"spring": 0.85, "summer": 1.0, "autumn": 0.9, "winter": 0.0}

LEAF_AREA_BUDGET = 104.0
BODY_CAP = 100


class _AspenShape:
    """The aspen's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the paper's aspen preset under the dials.

        Args:
            settings: The validated aspen settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=7,
            base_size=0.4,
            scale=settings.height,
            levels=3,
            ratio=0.015,
            ratio_power=1.2,
            taper=(0.75, 0.9, 1.0),
            flare=0.6,
            length=(1.0, 0.3, 0.6),
            length_v=(0.0, 0.0, 0.1),
            curve_res=(8, 5, 3),
            curve=(0.0, -40.0, -40.0),
            curve_v=(20.0, 50.0, 75.0),
            down_angle=(0.0, 60.0, 45.0),
            down_angle_v=(0.0, -50.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 77.0),
            branches=(1, int(20 + 10 * age), int(10 + 6 * age)),
            leaves=int(round((22 + 10 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.15,
            attraction_up=0.5,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the aspen's pale smooth bark.

        Args:
            settings: The validated aspen settings.
        """
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.016,
                width=0.09,
                wobble_length=1.0,
            ),
            grain_amplitude=0.008,
            optical_mid_base=plant.WOOD_BIRCH_MID,
            optical_mid_light=plant.WOOD_BIRCH_MID,
            optical_mid_crevice=1,
            moss_coverage=0.06,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the aspen's round fluttering leaf.

        Args:
            settings: The validated aspen settings.
        """
        return TreeLeaf(outline="hex", length=0.14, width_ratio=0.95)

    @staticmethod
    def make_leaf_mids(settings) -> tuple:
        """Return the season's leaf optical mids; aspen turns gold in autumn.

        Args:
            settings: The validated aspen settings.
        """
        if settings.season == "autumn":
            return plant.GOLD_MIDS
        return plant.SEASON_LEAF_MIDS[settings.season]


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(
            6.0, 20.0, default=13.0, doc="The tree's overall height (unit: m)."
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
            doc="True swaps the canopy onto the soft leaf material so the "
            "world can simulate and sway it.",
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
    tags=("nature", "tree", "aspen"),
)
def aspen(editor_access, settings):
    """A quaking aspen, slim and pale with a shivering round crown.

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
        is_leaf_soft: True swaps the canopy onto the soft leaf
            material so the world can simulate and sway it.
        trunk_voxel_size: The emitted wood voxel edge.
        canopy_voxel_size: The emitted leaf voxel edge.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="aspen",
        params=_AspenShape.make_params(settings),
        bark=_AspenShape.make_bark(settings),
        leaf=_AspenShape.make_leaf(settings),
        leaf_optical_mids=_AspenShape.make_leaf_mids(settings),
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
