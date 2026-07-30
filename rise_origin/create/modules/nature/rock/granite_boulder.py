# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The granite boulder: hard igneous rock, salt-and-pepper grains.

Granite is one material, not layers: the grain optical mid pattern
mixes feldspar, quartz, and mica tones per noise cell, so the surface
reads as speckled mineral crystals rather than strata. An
equidimensional fracture cuts a blocky fragment and spheroidal wear
rounds its corners slowly, because granite is hard, leaving the
weathered tor look. The tone dial runs gray to pink by raising the
potassium feldspar share.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm.sediment import (
    BallErosion,
    DetailErosion,
    GrainGenerator,
)

from ...material.nature import mineral
from .build import RockBuild

MODULE = "rise_origin.nature.rock.granite_boulder"

# Test-asserted voxel cap at the default dials (design doc budget).
STONE_VOXEL_CAP = 120_000

# The one material; volume material id 1.
MATERIAL_NAMES = ("granite",)
GRANITE_MATERIAL_ID = 1

# The grain mix is weighted by slot repetition over this many slots,
# so the cell pattern draws each mineral in its natural proportion.
GRAIN_SLOT_NUM = 40
QUARTZ_SLOT_NUM = 8
BIOTITE_SLOT_NUM = 2
# The potassium feldspar takes up to this share of the feldspar slots
# at full tone; zero tone is a gray granite with no pink.
MAX_ORTHOCLASE_SHARE = 0.5

# A grain never falls below this many emitted voxels, so speckle
# survives the emission downsample however coarse the voxels are.
MIN_GRAIN_VOXEL_NUM = 2.0

# A cluster is at least this many grains across, so patches always hold
# several crystals rather than shrinking to one.
MIN_CLUSTER_GRAIN_NUM = 3.0

# The within-mineral shade dither varies over about this many voxels,
# so a crystal patch reads as fine grain rather than a flat color.
DITHER_VOXEL_NUM = 1.2

# How hard a region biases toward one mineral; the same mineral groups
# without one filling a whole small stone.
CLUSTER_STRENGTH = 0.4

# Spheroidal weight: granite exfoliates, rounding corners without
# honeycomb pitting, so the pit term stays zero.
CURVATURE_WEIGHT = 1.4
WEAR_PACE = 0.3

# Granite holds its blocky shape, so it loses little depth: the base
# share plus a modest weathering share, as fractions of the stone size.
WEAR_DEPTH_BASE_SHARE = 0.02
WEAR_DEPTH_WEATHERING_SHARE = 0.14
REFERENCE_HARDNESS = 1.0
MIN_WEAR_STEP_NUM = 4


class _GraniteShape:
    """The granite boulder's dial-to-piece mapping."""

    @staticmethod
    def make_grain_mids(settings) -> tuple:
        """Return the ordered mineral groups the tone dial sets.

        Each entry is one mineral's shade group; the mix is ordered
        plagioclase, orthoclase, quartz, biotite, so the generator's
        cluster field slides a window over it and the minerals group
        into patches. Slot repetition sets each mineral's share, and
        the tone dial raises the pink orthoclase.

        Args:
            settings: The validated granite boulder settings.
        """
        orthoclase_slot_num = round(
            (GRAIN_SLOT_NUM - QUARTZ_SLOT_NUM - BIOTITE_SLOT_NUM)
            * MAX_ORTHOCLASE_SHARE
            * settings.tone
        )
        plagioclase_slot_num = (
            GRAIN_SLOT_NUM - QUARTZ_SLOT_NUM - BIOTITE_SLOT_NUM - orthoclase_slot_num
        )
        return (
            (mineral.GRANITE_PLAGIOCLASE_MIDS,) * plagioclase_slot_num
            + (mineral.GRANITE_ORTHOCLASE_MIDS,) * orthoclase_slot_num
            + (mineral.GRANITE_QUARTZ_MIDS,) * QUARTZ_SLOT_NUM
            + (mineral.GRANITE_BIOTITE_MIDS,) * BIOTITE_SLOT_NUM
        )

    @staticmethod
    def make_erosion(settings, rng: "cr.Rng") -> BallErosion:
        """Return the slow spheroidal wear that rounds the corners.

        Args:
            settings: The validated granite boulder settings.
            rng: The random stream the durability sway draws from.
        """
        depth = (
            WEAR_DEPTH_BASE_SHARE + WEAR_DEPTH_WEATHERING_SHARE * settings.weathering
        ) * settings.size
        cell_size = settings.voxel_size / settings.wear_subdivision
        step_num = max(
            MIN_WEAR_STEP_NUM,
            int(round(depth / cell_size * REFERENCE_HARDNESS / WEAR_PACE)),
        )
        return BallErosion(
            rng,
            step_num=step_num,
            ball_radius=0.14 * settings.size,
            wear_pace=WEAR_PACE,
            flat_weight=1.0,
            round_weight=CURVATURE_WEIGHT,
            pit_weight=0.0,
            hardness_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_HARDNESS
            ),
            durability_sway_ratio=0.4,
            durability_feature_size=0.3 * settings.size,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the gritty surface carving over the worn shape.

        Args:
            settings: The validated granite boulder settings.
            rng: The random stream the carving draws from.
        """
        return DetailErosion(
            rng,
            depth=settings.voxel_size,
            dent_spacing=0.05,
            dent_width=2.2 * settings.voxel_size,
        )


@cr.module(
    name=MODULE,
    version="1.0.0",
    is_deterministic=False,
    settings=cr.settings(
        size=cr.Float(0.2, 3.0, default=0.8, doc="The target diameter (unit: m)."),
        weathering=cr.Float(
            0.0,
            1.0,
            default=0.35,
            doc="Trades a fresh blocky fragment at 0 for a "
            "corestone-rounded shape at 1.",
        ),
        flatness=cr.Float(
            0.0,
            1.0,
            default=0.15,
            doc="How much the stone flattens along its short side, 0 to 1; "
            "low keeps granite equidimensional.",
        ),
        elongation=cr.Float(
            0.0,
            1.0,
            default=0.15,
            doc="How much the stone stretches along its long side, 0 to 1.",
        ),
        tone=cr.Float(
            0.0,
            1.0,
            default=0.3,
            doc="Runs the grain mix from gray granite at 0 to pink "
            "potassium-feldspar granite at 1.",
        ),
        grain_size=cr.Float(
            0.02, 0.15, default=0.06, doc="The mineral crystal size (unit: m)."
        ),
        cluster_size=cr.Float(
            0.05,
            0.6,
            default=0.15,
            doc="The size of the same-mineral patches (unit: m).",
        ),
        voxel_size=cr.VoxelSize(
            default="2cm", min="1cm", max="10cm", doc="The emitted voxel edge."
        ),
        wear_subdivision=cr.Int(
            1,
            10,
            default=4,
            doc="The unitless ratio of internal wear cells per emitted voxel edge.",
        ),
    ),
    dependencies=(f"{mineral.MODULE}@^1",),
    tags=("nature", "rock", "granite_boulder"),
)
def granite_boulder(editor_access, settings):
    """A granite boulder, one dynamic rigid body.

    Settings:
        size: The target diameter (unit: m).
        weathering: Trades a fresh blocky fragment at 0 for a
            corestone-rounded shape at 1.
        flatness: How much the stone flattens along its short side, 0
            to 1; low keeps granite equidimensional.
        elongation: How much the stone stretches along its long side,
            0 to 1.
        tone: Runs the grain mix from gray granite at 0 to pink
            potassium-feldspar granite at 1.
        grain_size: The mineral crystal size (unit: m).
        cluster_size: The size of the same-mineral patches (unit: m).
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
    """
    rng = cr.Rng(settings.seed)
    grid, device, box_size = RockBuild.make_boulder_grid(
        settings,
        "granite_boulder",
        settings.size,
        settings.flatness,
        settings.elongation,
    )
    grain_feature_size = max(
        settings.grain_size, MIN_GRAIN_VOXEL_NUM * settings.voxel_size
    )
    cluster_feature_size = max(
        settings.cluster_size, MIN_CLUSTER_GRAIN_NUM * grain_feature_size
    )
    volume = GrainGenerator(
        grid,
        str(device),
        rng.split("generator"),
        material_id=GRANITE_MATERIAL_ID,
        grain_mids=_GraniteShape.make_grain_mids(settings),
        grain_feature_size=grain_feature_size,
        cluster_feature_size=cluster_feature_size,
        cluster_strength=CLUSTER_STRENGTH,
        dither_feature_size=DITHER_VOXEL_NUM * settings.voxel_size,
        shape="box",
        box_size=box_size,
    ).generate()
    return RockBuild.make_boulder_draft(
        editor_access,
        settings,
        kind_name="granite_boulder",
        material_names=MATERIAL_NAMES,
        volume=volume,
        rng=rng,
        size=settings.size,
        flatness=settings.flatness,
        elongation=settings.elongation,
        erosion=_GraniteShape.make_erosion(settings, rng.split("erosion")),
        detail=_GraniteShape.make_detail(settings, rng.split("detail")),
    )
