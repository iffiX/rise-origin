# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The acacia: the flat umbrella crown of the savanna.

Forked stems rise and flatten out, carrying foliage only in a thin
layer at the top of the crown.
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

MODULE = "rise_origin.nature.tree.acacia"

SEASON_LEAF_DENSITY = {"spring": 0.9, "summer": 1.0, "autumn": 0.8, "winter": 0.0}

LEAF_AREA_BUDGET = 80.0
BODY_CAP = 100


class _AcaciaShape:
    """The acacia's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the umbrella acacia preset under the dials.

        Args:
            settings: The validated acacia settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=6,
            base_size=0.6,
            scale=settings.height,
            levels=3,
            ratio=0.02,
            ratio_power=1.2,
            taper=(0.75, 0.9, 1.0),
            flare=0.9,
            base_splits=2,
            seg_splits=(0.4, 0.25, 0.0),
            split_angle=(32.0, 22.0, 0.0),
            split_angle_v=(8.0, 8.0, 0.0),
            length=(1.0, 0.65, 0.3),
            length_v=(0.0, 0.08, 0.05),
            curve_res=(6, 6, 3),
            curve=(0.0, 15.0, 10.0),
            curve_back=(0.0, 20.0, 0.0),
            curve_v=(35.0, 40.0, 40.0),
            down_angle=(0.0, 65.0, 50.0),
            down_angle_v=(0.0, 10.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 77.0),
            branches=(1, int(9 + 5 * age), int(14 + 6 * age)),
            leaves=int(round((26 + 10 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.65,
            attraction_up=0.9,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the acacia's rough dark bark.

        Args:
            settings: The validated acacia settings.
        """
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.028,
                width=0.05,
                wobble_length=0.5,
            ),
            moss_coverage=0.0,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the acacia's fine feathery-leaf patch.

        Args:
            settings: The validated acacia settings.
        """
        return TreeLeaf(outline="ellipse", length=0.09, width_ratio=0.8)


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(
            3.0, 10.0, default=6.0, doc="The tree's overall height (unit: m)."
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
            doc="Picks the leaf color for the season; the crown stands "
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
            doc="True swaps the crown onto the soft leaf material so the "
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
    tags=("nature", "tree", "acacia", "savanna"),
)
def acacia(editor_access, settings):
    """An acacia, forked stems under a flat umbrella crown.

    Settings:
        height: The tree's overall height (unit: m).
        age: The age of the tree, from young and sparse at 0 to full
            and mature at 1.
        lean: How far the tree tilts off vertical, toward a random
            horizontal direction (unit: radians).
        season: Picks the leaf color for the season; the crown stands
            bare in winter.
        leaf_fall: The share of the naturally loose leaf parts kept
            unattached and ready to shed.
        is_leaf_soft: True swaps the crown onto the soft leaf material
            so the world can simulate and sway it.
        trunk_voxel_size: The emitted wood voxel edge.
        canopy_voxel_size: The emitted leaf voxel edge.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="acacia",
        params=_AcaciaShape.make_params(settings),
        bark=_AcaciaShape.make_bark(settings),
        leaf=_AcaciaShape.make_leaf(settings),
        leaf_optical_mids=plant.SEASON_LEAF_MIDS[settings.season],
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
