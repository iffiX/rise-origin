# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The coconut palm: one curving trunk under a fan of drooping fronds.

A palm is a two-level tree here: the trunk, and the fronds as its only
branches crowded at the very top, each frond carrying leaflets in two
alternating rows.
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

MODULE = "rise_origin.nature.tree.coconut_palm"

LEAF_AREA_BUDGET = 80.0
BODY_CAP = 100


class _PalmShape:
    """The palm's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the single-trunk palm preset under the dials.

        Args:
            settings: The validated palm settings.
        """
        age = settings.age
        return WeberPennParams(
            shape=3,
            base_size=0.92,
            scale=settings.height,
            levels=2,
            ratio=0.014,
            ratio_power=1.0,
            taper=(0.25, 0.9),
            flare=0.35,
            length=(1.0, 0.42),
            length_v=(0.0, 0.06),
            curve_res=(10, 8),
            curve=(6.0, 55.0),
            curve_v=(18.0, 18.0),
            down_angle=(0.0, 32.0, 35.0),
            down_angle_v=(0.0, 12.0, 8.0),
            rotate=(0.0, 140.0, -10.0),
            branches=(1, int(13 + 5 * age)),
            leaves=int(52 + 16 * age),
            leaf_begin=0.1,
            attraction_up=-0.9,
            max_stems=200,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the palm's ringed fibrous trunk surface.

        Args:
            settings: The validated palm settings.
        """
        return TreeBark(
            detail=TreeBarkPlates(
                depth=0.02,
                width=0.09,
                length=0.1,
                stagger=0.5,
                jitter=0.5,
            ),
            grain_amplitude=0.012,
            moss_coverage=0.0,
        )

    @staticmethod
    def make_leaf(settings) -> TreeLeaf:
        """Return the palm's long leaflet strip.

        Args:
            settings: The validated palm settings.
        """
        return TreeLeaf(outline="strip", length=0.3, width_ratio=0.09)


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(
            3.0, 12.0, default=8.0, doc="The tree's overall height (unit: m)."
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
            default=0.1,
            doc="How far the trunk tilts off vertical, giving it a "
            "seaside slant (unit: radians).",
        ),
        season=cr.Choice(
            "spring",
            "summer",
            "autumn",
            "winter",
            default="summer",
            doc="Has no effect on the crown color; palms keep their green "
            "fronds in every season.",
        ),
        leaf_fall=cr.Float(
            0.0,
            1.0,
            default=0.0,
            doc="The share of the naturally loose leaflet parts kept "
            "unattached and ready to shed.",
        ),
        is_leaf_soft=cr.Bool(
            default=False,
            doc="True swaps the fronds onto the soft leaf material so the "
            "world can simulate and sway them.",
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
    tags=("nature", "tree", "palm", "evergreen"),
)
def coconut_palm(editor_access, settings):
    """A coconut palm, one curving trunk under drooping fronds.

    Settings:
        height: The tree's overall height (unit: m).
        age: The age of the tree, from young and sparse at 0 to full
            and mature at 1.
        lean: How far the trunk tilts off vertical, giving it a
            seaside slant (unit: radians).
        season: Has no effect on the crown color; palms keep their
            green fronds in every season.
        leaf_fall: The share of the naturally loose leaflet parts kept
            unattached and ready to shed.
        is_leaf_soft: True swaps the fronds onto the soft leaf
            material so the world can simulate and sway them.
        trunk_voxel_size: The emitted wood voxel edge.
        canopy_voxel_size: The emitted leaf voxel edge.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="coconut_palm",
        params=_PalmShape.make_params(settings),
        bark=_PalmShape.make_bark(settings),
        leaf=_PalmShape.make_leaf(settings),
        leaf_optical_mids=plant.SEASON_LEAF_MIDS["summer"],
        max_leaf_area=LEAF_AREA_BUDGET,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.canopy_voxel_size,
        shed_fraction=settings.leaf_fall,
    )
