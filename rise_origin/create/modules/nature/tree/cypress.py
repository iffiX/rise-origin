# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The cypress: a narrow evergreen column, hedge-able.

Short upswept branches hug a cylindrical envelope from the ground up,
the tight columnar silhouette of a Mediterranean cypress.
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

MODULE = "rise_origin.nature.tree.cypress"

LEAF_AREA_BUDGET = 96.0
BODY_CAP = 100


class _CypressShape:
    """The cypress's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the columnar cypress preset under the dials.

        Args:
            settings: The validated cypress settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=3,
            base_size=0.04,
            scale=settings.height,
            levels=3,
            ratio=0.012,
            ratio_power=1.1,
            taper=(0.85, 1.0, 1.0),
            flare=0.4,
            length=(1.0, 0.16, 0.3),
            length_v=(0.0, 0.02, 0.05),
            curve_res=(12, 4, 3),
            curve=(0.0, -20.0, -20.0),
            curve_v=(10.0, 40.0, 50.0),
            down_angle=(0.0, 38.0, 35.0),
            down_angle_v=(0.0, 10.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 120.0),
            branches=(1, int(36 + 14 * age), int(9 + 6 * age)),
            leaves=int(30 + 12 * age),
            leaf_begin=0.0,
            attraction_up=0.8,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the cypress's thin fibrous bark.

        Args:
            settings: The validated cypress settings.
        """
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.02,
                width=0.045,
                wobble_length=0.6,
            ),
            moss_coverage=0.05,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the cypress's small scale-leaf patch.

        Args:
            settings: The validated cypress settings.
        """
        return TreeLeaf(outline="ellipse", length=0.07, width_ratio=0.6)


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(
            3.0, 12.0, default=7.0, doc="The tree's overall height (unit: m)."
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
            doc="Has no effect on the crown color; cypresses keep their "
            "green scale leaves in every season.",
        ),
        leaf_fall=cr.Float(
            0.0,
            1.0,
            default=0.0,
            doc="The share of the naturally loose foliage parts kept "
            "unattached and ready to shed.",
        ),
        is_leaf_soft=cr.Bool(
            default=False,
            doc="True swaps the scale leaves onto the soft leaf material "
            "so the world can simulate and sway them.",
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
    tags=("nature", "tree", "cypress", "evergreen"),
)
def cypress(editor_access, settings):
    """A cypress, a narrow evergreen column from the ground up.

    Settings:
        height: The tree's overall height (unit: m).
        age: The age of the tree, from young and sparse at 0 to full
            and mature at 1.
        lean: How far the tree tilts off vertical, toward a random
            horizontal direction (unit: radians).
        season: Has no effect on the crown color; cypresses keep their
            green scale leaves in every season.
        leaf_fall: The share of the naturally loose foliage parts kept
            unattached and ready to shed.
        is_leaf_soft: True swaps the scale leaves onto the soft leaf
            material so the world can simulate and sway them.
        trunk_voxel_size: The emitted wood voxel edge.
        canopy_voxel_size: The emitted leaf voxel edge.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="cypress",
        params=_CypressShape.make_params(settings),
        bark=_CypressShape.make_bark(settings),
        leaf=_CypressShape.make_leaf(settings),
        leaf_optical_mids=plant.NEEDLE_MIDS,
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
