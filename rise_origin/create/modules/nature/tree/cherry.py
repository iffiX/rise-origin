# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The cherry: a small spreading tree that blossoms in spring.

Spring swaps the whole crown onto pink and white blossom optical mids; the
smooth banded bark stays low and forked.
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

MODULE = "rise_origin.nature.tree.cherry"

SEASON_LEAF_DENSITY = {"spring": 1.1, "summer": 1.0, "autumn": 0.85, "winter": 0.0}

LEAF_AREA_BUDGET = 88.0
BODY_CAP = 100


class _CherryShape:
    """The cherry's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the low spreading cherry preset under the dials.

        Args:
            settings: The validated cherry settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=1,
            base_size=0.2,
            scale=settings.height,
            levels=3,
            ratio=0.016,
            ratio_power=1.15,
            taper=(0.7, 0.85, 1.0),
            flare=0.5,
            base_splits=1,
            seg_splits=(0.25, 0.2, 0.0),
            split_angle=(20.0, 22.0, 0.0),
            split_angle_v=(8.0, 8.0, 0.0),
            length=(1.0, 0.45, 0.5),
            length_v=(0.0, 0.08, 0.1),
            curve_res=(6, 5, 4),
            curve=(0.0, -25.0, -20.0),
            curve_v=(40.0, 70.0, 60.0),
            down_angle=(0.0, 48.0, 42.0),
            down_angle_v=(0.0, -40.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 77.0),
            branches=(1, int(12 + 6 * age), int(9 + 5 * age)),
            leaves=int(round((16 + 8 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.15,
            attraction_up=0.4,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the cherry's smooth banded bark.

        Args:
            settings: The validated cherry settings.
        """
        return TreeBark(
            detail=TreeBarkBands(
                depth=0.012,
                length=0.22,
                wobble=0.25,
                coverage=0.9,
            ),
            grain_amplitude=0.008,
            moss_coverage=0.05,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the cherry's small oval leaf or blossom cluster.

        Args:
            settings: The validated cherry settings.
        """
        return TreeLeaf(outline="ellipse", length=0.1, width_ratio=0.7)

    @staticmethod
    def make_leaf_mids(settings) -> tuple:
        """Return the season's optical mids; spring is the blossom show.

        Args:
            settings: The validated cherry settings.
        """
        if settings.season == "spring":
            return plant.BLOSSOM_MIDS
        return plant.SEASON_LEAF_MIDS[settings.season]


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(
            2.5, 8.0, default=5.0, doc="The tree's overall height (unit: m)."
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
            default="spring",
            doc="Picks the crown color for the season: pink blossoms in "
            "spring, green in summer, color in autumn, bare in winter.",
        ),
        leaf_fall=cr.Float(
            0.0,
            1.0,
            default=0.0,
            doc="The share of the naturally loose crown parts kept "
            "unattached, drifting like petals once the world runs.",
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
    tags=("nature", "tree", "cherry", "blossom"),
)
def cherry(editor_access, settings):
    """A cherry, low and spreading, pink with blossom in spring.

    Settings:
        height: The tree's overall height (unit: m).
        age: The age of the tree, from young and sparse at 0 to full
            and mature at 1.
        lean: How far the tree tilts off vertical, toward a random
            horizontal direction (unit: radians).
        season: Picks the crown color for the season: pink blossoms in
            spring, green in summer, color in autumn, bare in winter.
        leaf_fall: The share of the naturally loose crown parts kept
            unattached, drifting like petals once the world runs.
        is_leaf_soft: True swaps the crown onto the soft leaf material
            so the world can simulate and sway it.
        trunk_voxel_size: The emitted wood voxel edge.
        canopy_voxel_size: The emitted leaf voxel edge.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="cherry",
        params=_CherryShape.make_params(settings),
        bark=_CherryShape.make_bark(settings),
        leaf=_CherryShape.make_leaf(settings),
        leaf_optical_mids=_CherryShape.make_leaf_mids(settings),
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
