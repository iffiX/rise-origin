# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The spruce: a tight evergreen cone with drooping branchlets.

Dense whorls hold nearly level branches whose branchlets hang down,
the classic spruce skirt; the needles stay green through every season.
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

MODULE = "rise_origin.nature.tree.spruce"

LEAF_AREA_BUDGET = 112.0
BODY_CAP = 100


class _SpruceShape:
    """The spruce's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the tight-cone spruce preset under the dials.

        Args:
            settings: The validated spruce settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=0,
            base_size=0.08,
            scale=settings.height,
            levels=3,
            ratio=0.012,
            ratio_power=1.3,
            taper=(0.9, 1.0, 1.0),
            flare=0.6,
            whorl_branches=(1, 6, 1),
            length=(1.0, 0.28, 0.3),
            length_v=(0.0, 0.03, 0.05),
            curve_res=(14, 5, 3),
            curve=(0.0, -35.0, -25.0),
            curve_back=(0.0, 10.0, 0.0),
            curve_v=(10.0, 25.0, 25.0),
            down_angle=(0.0, 78.0, 68.0),
            down_angle_v=(0.0, 8.0, 12.0),
            rotate=(0.0, 140.0, 140.0, 120.0),
            branches=(1, int(10 + 5 * age), int(8 + 5 * age)),
            leaves=int(34 + 14 * age),
            leaf_begin=0.1,
            attraction_up=0.2,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the spruce's flaky thin bark.

        Args:
            settings: The validated spruce settings.
        """
        return TreeBark(
            detail=TreeBarkPlates(
                depth=0.02,
                width=0.05,
                length=0.07,
                stagger=0.25,
                jitter=0.8,
                shift=0.7,
            ),
            moss_coverage=0.1,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the spruce's short needle strip.

        Args:
            settings: The validated spruce settings.
        """
        return TreeLeaf(outline="strip", length=0.09, width_ratio=0.35)


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(6.0, 22.0, default=14.0),
        age=cr.Float(0.0, 1.0, default=0.7),
        lean=cr.Float(0.0, 0.6, default=0.0),
        season=cr.Choice("spring", "summer", "autumn", "winter", default="summer"),
        leaf_fall=cr.Float(0.0, 1.0, default=0.0),
        is_leaf_soft=cr.Bool(default=False),
        trunk_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
        canopy_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
    ),
    dependencies=(f"{plant.MODULE}@^1",),
    tags=("nature", "tree", "spruce", "evergreen"),
)
def spruce(editor_access, settings):
    """A spruce, a tight cone of whorls with drooping branchlets.

    ``season`` never bares the crown, since spruces keep their needles;
    ``is_leaf_soft`` lets the world sway them and ``leaf_fall`` keeps
    that share of the naturally loose needle clusters unattached.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="spruce",
        params=_SpruceShape.make_params(settings),
        bark=_SpruceShape.make_bark(settings),
        leaf=_SpruceShape.make_leaf(settings),
        leaf_optical_mids=plant.NEEDLE_MIDS,
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
        tile_extent=2.8,
    )
