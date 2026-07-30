# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The tupelo: the Weber-Penn paper's black tupelo, scarlet in autumn.

The preset follows the paper's appendix column for the black tupelo,
with its four stem levels scaled to the voxel world's stem budget; the
tall tapered crown turns scarlet in autumn.
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

MODULE = "rise_origin.nature.tree.tupelo"

SEASON_LEAF_DENSITY = {"spring": 0.85, "summer": 1.0, "autumn": 0.9, "winter": 0.0}

LEAF_AREA_BUDGET = 104.0
BODY_CAP = 100


class _TupeloShape:
    """The tupelo's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the paper's tupelo preset under the dials.

        Args:
            settings: The validated tupelo settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=4,
            base_size=0.2,
            scale=settings.height,
            levels=4,
            ratio=0.015,
            ratio_power=1.3,
            taper=(0.8, 0.9, 1.0, 1.0),
            flare=1.0,
            length=(1.0, 0.3, 0.6, 0.4),
            length_v=(0.0, 0.05, 0.1, 0.0),
            curve_res=(6, 6, 5, 4),
            curve=(0.0, 0.0, -10.0, 0.0),
            curve_v=(40.0, 90.0, 150.0, 0.0),
            down_angle=(0.0, 60.0, 30.0, 45.0),
            down_angle_v=(0.0, -40.0, 10.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 140.0, 140.0),
            branches=(1, int(18 + 8 * age), int(8 + 4 * age), int(4 + 2 * age)),
            leaves=int(round((11 + 5 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.1,
            attraction_up=0.5,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the tupelo's blocky dark bark.

        Args:
            settings: The validated tupelo settings.
        """
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.028,
                width=0.06,
                wobble_length=0.5,
            ),
            moss_coverage=0.12,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the tupelo's glossy oval leaf.

        Args:
            settings: The validated tupelo settings.
        """
        return TreeLeaf(outline="ellipse", length=0.16, width_ratio=0.55)


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(
            8.0, 15.0, default=13.0, doc="The tree's overall height (unit: m)."
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
            doc="Picks the leaf color for the season, scarlet in autumn "
            "and bare in winter.",
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
    tags=("nature", "tree", "tupelo"),
)
def tupelo(editor_access, settings):
    """A black tupelo, a tall tapered crown over straight limbs.

    Settings:
        height: The tree's overall height (unit: m).
        age: The age of the tree, from young and sparse at 0 to full
            and mature at 1.
        lean: How far the tree tilts off vertical, toward a random
            horizontal direction (unit: radians).
        season: Picks the leaf color for the season, scarlet in
            autumn and bare in winter.
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
        kind_name="tupelo",
        params=_TupeloShape.make_params(settings),
        bark=_TupeloShape.make_bark(settings),
        leaf=_TupeloShape.make_leaf(settings),
        leaf_optical_mids=plant.SEASON_LEAF_MIDS[settings.season],
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
