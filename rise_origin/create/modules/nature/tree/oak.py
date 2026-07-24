# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The oak: the Weber-Penn paper's California black oak.

The preset follows the paper's appendix column for the black oak: a
low forking trunk, wide gnarled limbs curving back on themselves, and
a broad rounded crown.
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

MODULE = "rise_origin.nature.tree.oak"

SEASON_LEAF_DENSITY = {"spring": 0.85, "summer": 1.0, "autumn": 0.9, "winter": 0.0}

LEAF_AREA_BUDGET = 112.0
BODY_CAP = 100


class _OakShape:
    """The oak's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the paper's black oak preset under the dials.

        Args:
            settings: The validated oak settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=2,
            base_size=0.08,
            scale=settings.height,
            levels=3,
            ratio=0.018,
            ratio_power=1.35,
            taper=(0.75, 0.9, 1.0),
            flare=1.2,
            base_splits=2,
            seg_splits=(0.35, 0.2, 0.1),
            split_angle=(12.0, 10.0, 10.0),
            split_angle_v=(8.0, 10.0, 10.0),
            length=(1.0, 0.55, 0.25),
            length_v=(0.0, 0.1, 0.05),
            curve_res=(8, 8, 3),
            curve=(0.0, 40.0, 0.0),
            curve_back=(0.0, -70.0, 0.0),
            curve_v=(80.0, 120.0, 30.0),
            down_angle=(0.0, 30.0, 45.0),
            down_angle_v=(0.0, -30.0, 10.0),
            rotate=(0.0, 80.0, 140.0, 140.0),
            branches=(1, int(14 + 7 * age), int(16 + 10 * age)),
            leaves=int(round((20 + 10 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.2,
            attraction_up=0.8,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the oak's deep dark bark, mossy with age.

        Args:
            settings: The validated oak settings.
        """
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.038,
                width=0.06,
                wobble_length=0.5,
            ),
            moss_coverage=0.1 + 0.25 * settings.age,
            moss_height=2.5,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the oak's lobed leaf.

        Args:
            settings: The validated oak settings.
        """
        return TreeLeaf(outline="hex", length=0.16, width_ratio=0.7)


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(5.0, 16.0, default=10.0),
        age=cr.Float(0.0, 1.0, default=0.7),
        lean=cr.Float(0.0, 0.6, default=0.0),
        season=cr.Choice("spring", "summer", "autumn", "winter", default="summer"),
        leaf_fall=cr.Float(0.0, 1.0, default=0.0),
        is_leaf_soft=cr.Bool(default=False),
        trunk_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
        canopy_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
    ),
    dependencies=(f"{plant.MODULE}@^1",),
    tags=("nature", "tree", "oak"),
)
def oak(editor_access, settings):
    """A black oak, low forked with wide gnarled limbs.

    ``season`` picks the leaf optical mids; winter stands bare.
    ``is_leaf_soft`` lets the world sway the canopy and ``leaf_fall``
    keeps that share of the naturally loose leaf parts unattached.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="oak",
        params=_OakShape.make_params(settings),
        bark=_OakShape.make_bark(settings),
        leaf=_OakShape.make_leaf(settings),
        leaf_optical_mids=plant.SEASON_LEAF_MIDS[settings.season],
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
