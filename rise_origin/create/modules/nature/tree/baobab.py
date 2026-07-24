# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The baobab: a fat bare trunk under a sparse flat crown.

Most of the height is the massive bottle trunk; a few stubby limbs
carry a thin scatter of leaves at the very top.
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

MODULE = "rise_origin.nature.tree.baobab"

SEASON_LEAF_DENSITY = {"spring": 0.9, "summer": 1.0, "autumn": 0.7, "winter": 0.0}

LEAF_AREA_BUDGET = 64.0
BODY_CAP = 100


class _BaobabShape:
    """The baobab's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the bottle-trunk baobab preset under the dials.

        Args:
            settings: The validated baobab settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=6,
            base_size=0.65,
            scale=settings.height,
            levels=3,
            ratio=0.08,
            ratio_power=1.0,
            taper=(0.65, 0.8, 1.0),
            flare=1.6,
            length=(1.0, 0.42, 0.5),
            length_v=(0.0, 0.05, 0.05),
            curve_res=(4, 4, 3),
            curve=(0.0, -30.0, -20.0),
            curve_v=(20.0, 50.0, 60.0),
            down_angle=(0.0, 55.0, 45.0),
            down_angle_v=(0.0, 15.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 77.0),
            branches=(1, int(9 + 3 * age), int(9 + 4 * age)),
            leaves=int(round((30 + 10 * age) * SEASON_LEAF_DENSITY[settings.season])),
            leaf_begin=0.3,
            attraction_up=0.3,
            max_stems=2000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the baobab's smooth folded bark.

        Args:
            settings: The validated baobab settings.
        """
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.024,
                width=0.16,
                wobble_length=1.4,
            ),
            grain_amplitude=0.008,
            moss_coverage=0.0,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the baobab's small compound-leaf patch.

        Args:
            settings: The validated baobab settings.
        """
        return TreeLeaf(outline="ellipse", length=0.15, width_ratio=0.7)


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(4.0, 14.0, default=7.0),
        age=cr.Float(0.0, 1.0, default=0.7),
        lean=cr.Float(0.0, 0.3, default=0.0),
        season=cr.Choice("spring", "summer", "autumn", "winter", default="summer"),
        leaf_fall=cr.Float(0.0, 1.0, default=0.0),
        is_leaf_soft=cr.Bool(default=False),
        trunk_voxel_size=cr.VoxelSize(default="5cm", min="1cm", max="10cm"),
        canopy_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
    ),
    dependencies=(f"{plant.MODULE}@^1",),
    tags=("nature", "tree", "baobab"),
)
def baobab(editor_access, settings):
    """A baobab, a massive bottle trunk under a sparse flat crown.

    ``season`` picks the sparse leaf optical mids; winter stands bare.
    ``is_leaf_soft`` lets the world sway what little foliage there is
    and ``leaf_fall`` keeps that share of the naturally loose parts
    unattached.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="baobab",
        params=_BaobabShape.make_params(settings),
        bark=_BaobabShape.make_bark(settings),
        leaf=_BaobabShape.make_leaf(settings),
        leaf_optical_mids=plant.SEASON_LEAF_MIDS[settings.season],
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
