# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The maple: a Weber-Penn tree with seasonal optical mids.

The growth model lives in ``algorithm/tree/weber_penn.py`` and the shared
sweep in ``algorithm/tree/voxelizer.py``; this module only maps its
friendly dials onto an internal preset and picks the maple's bark and
leaf silhouette (tmp-docs/create-look-design/nature/tree/design.md
records the mapping). Low ``gnarl`` with one trunk reads as an upright
red maple; high ``gnarl`` with several trunks reads as a gnarled Japanese
maple.
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

MODULE = "rise_origin.nature.tree.maple"

# Leaf density factor per season; winter grows no leaves at all.
SEASON_LEAF_DENSITY = {"spring": 0.85, "summer": 1.0, "autumn": 0.9, "winter": 0.0}

# Test-asserted voxel caps at the default dials (design doc budgets).
LIMB_VOXEL_CAP = 300_000
CANOPY_VOXEL_CAP = 400_000

# Leaf cells one tree may stamp; the voxelizer thins leaves evenly past it.
LEAF_AREA_BUDGET = 112.0

# Test-asserted body cap per tree at any dials.
BODY_CAP = 100


class _MapleShape:
    """The maple's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the Weber-Penn parameters the dials modulate.

        Args:
            settings: The validated maple settings.
        """
        gnarl = settings.gnarl
        age = settings.age
        spread = settings.spread
        return WeberPennParams(
            shape=2 if gnarl > 0.6 else 1,
            base_size=0.3 - 0.05 * gnarl,
            scale=settings.height,
            levels=3 if age > 0.35 else 2,
            ratio=0.015 * (0.7 + 0.6 * age) + 0.01 * gnarl,
            ratio_power=1.1,
            taper=(0.75, 0.85, 1.0),
            flare=0.4 + 0.6 * age + (0.4 if settings.trunk_count > 1 else 0.0),
            base_splits=settings.trunk_count - 1,
            length=(1.0, 0.30 + 0.25 * spread, 0.5, 0.4),
            length_v=(0.0, 0.05, 0.1, 0.0),
            seg_splits=(0.15 + 0.5 * gnarl, 0.2 + 0.3 * gnarl, 0.0),
            split_angle=(20.0 + 15.0 * gnarl, 25.0, 0.0),
            split_angle_v=(5.0 + 10.0 * gnarl, 10.0, 0.0),
            curve_res=(8, 5, 4),
            curve=(0.0, -30.0, -25.0),
            curve_v=(
                20.0 + 70.0 * gnarl,
                60.0 + 60.0 * gnarl,
                80.0 + 40.0 * gnarl,
            ),
            down_angle=(0.0, 45.0 + 25.0 * spread, 40.0),
            down_angle_v=(0.0, -50.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 77.0),
            branches=(
                1,
                int((14 + 12 * age) * (1.0 - 0.45 * gnarl)),
                int((8 + 8 * age) * (1.0 - 0.45 * gnarl)),
                0,
            ),
            leaves=int(
                round(
                    (12.0 + 8.0 * age)
                    * (1.0 - 0.3 * gnarl)
                    * SEASON_LEAF_DENSITY[settings.season]
                )
            ),
            leaf_begin=0.25 + 0.25 * gnarl,
            attraction_up=0.6 - 1.0 * gnarl,
            ground_clearance=0.5,
            max_stems=3000,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the maple's bark detail; gnarled trees grow mossier.

        Args:
            settings: The validated maple settings.
        """
        gnarl = settings.gnarl
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.032,
                width=0.05,
                wobble_length=0.7,
            ),
            grain_amplitude=0.01,
            moss_coverage=0.15 + 0.45 * gnarl,
            moss_height=1.5 + 2.0 * gnarl,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the maple leaf silhouette; spread widens the leaves.

        Args:
            settings: The validated maple settings.
        """
        return TreeLeaf(
            outline="maple_star",
            length=0.18 + 0.06 * settings.spread,
            width_ratio=0.85,
        )


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(3.0, 15.0, default=8.0),
        age=cr.Float(0.0, 1.0, default=0.7),
        gnarl=cr.Float(0.0, 1.0, default=0.2),
        spread=cr.Float(0.0, 1.0, default=0.5),
        lean=cr.Float(0.0, 0.6, default=0.0),
        trunk_count=cr.Int(1, 5, default=1),
        season=cr.Choice("spring", "summer", "autumn", "winter", default="summer"),
        leaf_fall=cr.Float(0.0, 1.0, default=0.0),
        is_leaf_soft=cr.Bool(default=False),
        trunk_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
        canopy_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
    ),
    dependencies=(f"{plant.MODULE}@^1",),
    tags=("nature", "tree", "maple"),
)
def maple(editor_access, settings):
    """A maple tree, upright to gnarled, across four seasons.

    ``gnarl`` twists the stems and loosens the splits, ``spread`` trades
    height for crown width, ``trunk_count`` grows several stems from the
    base, ``lean`` tilts the whole tree, and ``season`` picks the leaf
    optical mids; winter stands bare. ``is_leaf_soft`` swaps the canopy onto the
    soft leaf material and lets the world simulate it; ``leaf_fall``
    keeps that share of the naturally loose leaf parts unattached, so a
    soft tree sheds them when the world starts running.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="maple",
        params=_MapleShape.make_params(settings),
        bark=_MapleShape.make_bark(settings),
        leaf=_MapleShape.make_leaf(settings),
        leaf_optical_mids=plant.SEASON_LEAF_MIDS[settings.season],
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
