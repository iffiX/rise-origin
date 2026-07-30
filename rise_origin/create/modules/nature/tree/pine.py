# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The pine: a conical evergreen with whorled branches.

Branches ring the trunk in whorls the way real pines grow, dipping
away and lifting at the tips; the needles stay green through every
season.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm import (
    TreeBark,
    TreeBarkPlates,
    TreeLeaf,
    WeberPennParams,
)

from ...material.nature import plant
from .build import TreeBuild

MODULE = "rise_origin.nature.tree.pine"

LEAF_AREA_BUDGET = 104.0
BODY_CAP = 100


class _PineShape:
    """The pine's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the whorled pine preset under the dials.

        Args:
            settings: The validated pine settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=0,
            base_size=0.35,
            scale=settings.height,
            levels=3,
            ratio=0.013,
            ratio_power=1.4,
            taper=(0.85, 0.95, 1.0),
            flare=0.5,
            whorl_branches=(1, 5, 1),
            length=(1.0, 0.32, 0.35),
            length_v=(0.0, 0.03, 0.05),
            curve_res=(12, 5, 3),
            curve=(0.0, -20.0, -15.0),
            curve_back=(0.0, 30.0, 0.0),
            curve_v=(15.0, 25.0, 30.0),
            down_angle=(0.0, 82.0, 50.0),
            down_angle_v=(0.0, 8.0, 15.0),
            rotate=(0.0, 140.0, 140.0, 120.0),
            branches=(1, int(9 + 5 * age), int(7 + 4 * age)),
            leaves=int(30 + 14 * age),
            leaf_begin=0.25,
            attraction_up=0.35,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the pine's plated bark.

        Args:
            settings: The validated pine settings.
        """
        return TreeBark(
            detail=TreeBarkPlates(
                depth=0.028,
                width=0.06,
                length=0.1,
                stagger=0.3,
                jitter=0.7,
                shift=0.6,
            ),
            moss_coverage=0.08,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the pine's needle cluster strip.

        Args:
            settings: The validated pine settings.
        """
        return TreeLeaf(outline="strip", length=0.12, width_ratio=0.3)


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
            doc="Has no effect on the crown color; pines keep their green "
            "needles in every season.",
        ),
        leaf_fall=cr.Float(
            0.0,
            1.0,
            default=0.0,
            doc="The share of the naturally loose needle clusters kept "
            "unattached and ready to shed.",
        ),
        is_leaf_soft=cr.Bool(
            default=False,
            doc="True swaps the needles onto the soft leaf material so "
            "the world can simulate and sway them.",
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
    tags=("nature", "tree", "pine", "evergreen"),
)
def pine(editor_access, settings):
    """A pine, conical with whorled branches and evergreen needles.

    Settings:
        height: The tree's overall height (unit: m).
        age: The age of the tree, from young and sparse at 0 to full
            and mature at 1.
        lean: How far the tree tilts off vertical, toward a random
            horizontal direction (unit: radians).
        season: Has no effect on the crown color; pines keep their
            green needles in every season.
        leaf_fall: The share of the naturally loose needle clusters
            kept unattached and ready to shed.
        is_leaf_soft: True swaps the needles onto the soft leaf
            material so the world can simulate and sway them.
        trunk_voxel_size: The emitted wood voxel edge.
        canopy_voxel_size: The emitted leaf voxel edge.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="pine",
        params=_PineShape.make_params(settings),
        bark=_PineShape.make_bark(settings),
        leaf=_PineShape.make_leaf(settings),
        leaf_optical_mids=plant.NEEDLE_MIDS,
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
