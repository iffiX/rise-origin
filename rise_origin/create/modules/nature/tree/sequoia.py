# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The sequoia: a giant conical evergreen with deep furrowed bark.

The largest tree of the catalog, capped at the large class; coarser
default voxel sizes keep its budgets sane. Short level branches skirt
a massive flared column.
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

MODULE = "rise_origin.nature.tree.sequoia"

LEAF_AREA_BUDGET = 104.0
BODY_CAP = 100


class _SequoiaShape:
    """The sequoia's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the giant conifer preset under the dials.

        Args:
            settings: The validated sequoia settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=0,
            base_size=0.3,
            scale=settings.height,
            levels=3,
            ratio=0.02,
            ratio_power=1.4,
            taper=(0.85, 0.95, 1.0),
            flare=1.9,
            length=(1.0, 0.18, 0.3),
            length_v=(0.0, 0.02, 0.05),
            curve_res=(12, 4, 3),
            curve=(0.0, -30.0, -10.0),
            curve_v=(10.0, 30.0, 30.0),
            down_angle=(0.0, 85.0, 55.0),
            down_angle_v=(0.0, 8.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 120.0),
            branches=(1, int(30 + 10 * age), int(9 + 4 * age)),
            leaves=int(26 + 10 * age),
            leaf_begin=0.15,
            attraction_up=0.3,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the sequoia's deep soft-fissured bark.

        Args:
            settings: The validated sequoia settings.
        """
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.044,
                width=0.09,
                wobble_length=1.0,
            ),
            grain_amplitude=0.012,
            moss_coverage=0.08,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the sequoia's scale-needle strip.

        Args:
            settings: The validated sequoia settings.
        """
        return TreeLeaf(outline="strip", length=0.1, width_ratio=0.35)


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(
            10.0, 15.0, default=14.0, doc="The tree's overall height (unit: m)."
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
            0.3,
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
            doc="Has no effect on the crown color; sequoias keep their "
            "green scale needles in every season.",
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
            doc="True swaps the scale needles onto the soft leaf material "
            "so the world can simulate and sway them.",
        ),
        trunk_voxel_size=cr.VoxelSize(
            default="5cm",
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
    tags=("nature", "tree", "sequoia", "evergreen"),
)
def sequoia(editor_access, settings):
    """A sequoia, a giant flared column skirted by short level limbs.

    Settings:
        height: The tree's overall height (unit: m).
        age: The age of the tree, from young and sparse at 0 to full
            and mature at 1.
        lean: How far the tree tilts off vertical, toward a random
            horizontal direction (unit: radians).
        season: Has no effect on the crown color; sequoias keep their
            green scale needles in every season.
        leaf_fall: The share of the naturally loose needle clusters
            kept unattached and ready to shed.
        is_leaf_soft: True swaps the scale needles onto the soft leaf
            material so the world can simulate and sway them.
        trunk_voxel_size: The emitted wood voxel edge.
        canopy_voxel_size: The emitted leaf voxel edge.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="sequoia",
        params=_SequoiaShape.make_params(settings),
        bark=_SequoiaShape.make_bark(settings),
        leaf=_SequoiaShape.make_leaf(settings),
        leaf_optical_mids=plant.NEEDLE_MIDS,
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
