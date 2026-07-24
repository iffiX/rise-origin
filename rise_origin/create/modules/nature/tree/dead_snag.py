# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The dead snag: a bare weathered trunk with snapped-off limbs.

The trunk and limbs break short with ragged splintered tips, the gray
dead wood shows through peeled bark patches, and no leaf material is
declared at all. The snag stays a static fixture; making its limbs
individually breakable waits on engine joints, tracked by the TODO in
``algorithm/tree/voxelizer.py``.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm import TreeBark, TreeBarkFurrows, WeberPennParams

from ...material.nature import plant
from .build import TreeBuild

MODULE = "rise_origin.nature.tree.dead_snag"

LIMB_VOXEL_CAP = 250_000
BODY_CAP = 100


class _SnagShape:
    """The snag's dial-to-parameter mapping."""

    @staticmethod
    def make_params(settings) -> WeberPennParams:
        """Return the broken dead trunk preset under the dials.

        Args:
            settings: The validated snag settings.
        """
        decay = settings.decay
        return WeberPennParams(
            shape=2,
            base_size=0.25,
            scale=settings.height,
            levels=3,
            ratio=0.02,
            ratio_power=1.2,
            taper=(0.7, 0.85, 1.0),
            flare=1.0,
            seg_splits=(0.3, 0.2, 0.0),
            split_angle=(18.0, 15.0, 0.0),
            split_angle_v=(10.0, 10.0, 0.0),
            length=(1.0, 0.3, 0.25),
            length_v=(0.0, 0.08, 0.05),
            curve_res=(6, 5, 3),
            curve=(0.0, -20.0, -15.0),
            curve_v=(70.0, 90.0, 70.0),
            down_angle=(0.0, 45.0, 45.0),
            down_angle_v=(0.0, 15.0, 10.0),
            rotate=(0.0, 140.0, 140.0, 77.0),
            branches=(1, int(10 - 4 * decay), int(5 - 3 * decay)),
            break_fraction=(
                0.75 - 0.35 * decay,
                0.55 - 0.25 * decay,
                0.0,
            ),
            leaves=0,
            attraction_up=0.1,
            max_stems=1500,
        )

    @staticmethod
    def make_bark(settings) -> TreeBark:
        """Return the weathered dead bark; decay peels more of it away.

        Args:
            settings: The validated snag settings.
        """
        decay = settings.decay
        return TreeBark(
            detail=TreeBarkFurrows(
                depth=0.036,
                width=0.06,
                wobble_length=0.5,
            ),
            grain_amplitude=0.012,
            peel_coverage=0.25 + 0.35 * decay,
            peel_cell=0.5,
            moss_coverage=0.06,
            optical_mid_base=plant.WOOD_DEAD_MID,
            optical_mid_light=plant.WOOD_PEEL_MID,
            optical_mid_crevice=3,
            optical_mid_peel=plant.WOOD_PEEL_MID,
        )


@cr.module(
    name=MODULE,
    version="1.0.0",
    settings=cr.settings(
        height=cr.Float(2.0, 12.0, default=7.0),
        decay=cr.Float(0.0, 1.0, default=0.6),
        lean=cr.Float(0.0, 0.6, default=0.05),
        trunk_voxel_size=cr.VoxelSize(default="2cm", min="1cm", max="10cm"),
    ),
    dependencies=(f"{plant.MODULE}@^1",),
    tags=("nature", "tree", "dead", "snag"),
)
def dead_snag(editor_access, settings):
    """A dead snag, a broken weathered trunk with splintered stubs.

    ``decay`` shortens the breaks and peels more bark off the gray
    dead wood; there are no leaves in any season, and the snag stands
    as a static fixture.
    """
    return TreeBuild.make_draft(
        editor_access,
        settings,
        kind_name="dead_snag",
        params=_SnagShape.make_params(settings),
        bark=_SnagShape.make_bark(settings),
        leaf=None,
        leaf_optical_mids=(),
        max_leaf_area=0.0,
        trunk_voxel_size=settings.trunk_voxel_size,
        canopy_voxel_size=settings.trunk_voxel_size,
        is_leaf_declared=False,
    )
