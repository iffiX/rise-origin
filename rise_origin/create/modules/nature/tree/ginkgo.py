# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The ginkgo: fan leaves on upswept limbs, pure gold in autumn.

The fan silhouette is the kind's signature; autumn turns the whole
crown to gold at once.
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

MODULE = "rise_origin.nature.tree.ginkgo"

SEASON_LEAF_DENSITY = {"spring": 0.85, "summer": 1.0, "autumn": 1.0, "winter": 0.0}

LEAF_AREA_BUDGET = 96.0
BODY_CAP = 100


class _GinkgoShape:
    """The ginkgo's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the upswept ginkgo preset under the dials.

        Args:
            settings: The validated ginkgo settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=4,
            base_size=0.25,
            scale=settings.height,
            levels=3,
            ratio=0.013,
            ratio_power=1.25,
            taper=(0.8, 0.9, 1.0),
            flare=0.6,
            length=(1.0, 0.3, 0.35),
            length_v=(0.0, 0.05, 0.05),
            curve_res=(5, 5, 3),
            curve=(0.0, -25.0, -15.0),
            curve_v=(15.0, 45.0, 55.0),
            down_angle=(0.0, 42.0, 38.0),
            down_angle_v=(0.0, -30.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 77.0),
            branches=(1, int(17 + 8 * age), int(16 + 8 * age)),
            leaves=int(round((24 + 10 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.15,
            attraction_up=0.7,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the ginkgo's gray corky bark.

        Args:
            settings: The validated ginkgo settings.
        """
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.026,
                width=0.07,
                wobble_length=0.7,
            ),
            moss_coverage=0.05,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the ginkgo's fan leaf.

        Args:
            settings: The validated ginkgo settings.
        """
        return TreeLeaf(outline="fan", length=0.11, width_ratio=1.15)

    @staticmethod
    def make_leaf_mids(settings) -> tuple:
        """Return the season's optical mids; autumn is the gold show.

        Args:
            settings: The validated ginkgo settings.
        """
        if settings.season == "autumn":
            return plant.GOLD_MIDS
        return plant.SEASON_LEAF_MIDS[settings.season]


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(4.0, 15.0, default=10.0),
        age=cr.Float(0.0, 1.0, default=0.7),
        lean=cr.Float(0.0, 0.6, default=0.0),
        season=cr.Choice("spring", "summer", "autumn", "winter", default="summer"),
        leaf_fall=cr.Float(0.0, 1.0, default=0.0),
        is_leaf_soft=cr.Bool(default=False),
        trunk_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
        canopy_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
    ),
    dependencies=(f"{plant.MODULE}@^1",),
    tags=("nature", "tree", "ginkgo"),
)
def ginkgo(editor_access, settings):
    """A ginkgo, fan leaves on upswept limbs, gold in autumn.

    ``season`` picks the leaf optical mids, all gold in autumn and bare in
    winter; ``is_leaf_soft`` lets the world sway the fans and
    ``leaf_fall`` keeps that share of the naturally loose leaf parts
    unattached.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="ginkgo",
        params=_GinkgoShape.make_params(settings),
        bark=_GinkgoShape.make_bark(settings),
        leaf=_GinkgoShape.make_leaf(settings),
        leaf_optical_mids=_GinkgoShape.make_leaf_mids(settings),
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
