# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The sandstone boulder: layered desert rock, fresh-cut to river-worn.

The sediment engine does the work: strata generation lays alternating
soft body layers and hard resistant bands with real Navajo/Entrada
tones, a plane fracture cuts the fresh block, and ball erosion rounds
it under the dialed transport. Ledges appear on their own where the
soft layers wear faster than the hard bands
(tmp-docs/create-look-design/nature/rock/design.md records the
mapping).
"""

import math

from rise_origin import create as cr
from rise_origin.create.algorithm.sediment import (
    BallErosion,
    DetailErosion,
    StrataGenerator,
    StrataGeneratorLayer,
)

from ...material.nature import mineral
from .build import MARGIN_VOXEL_NUM, RockBuild

MODULE = "rise_origin.nature.rock.sandstone_boulder"

# Test-asserted voxel cap at the default dials (design doc budget).
STONE_VOXEL_CAP = 120_000

# The materials in job order; volume material ids start at 1.
MATERIAL_NAMES = ("sandstone_soft", "sandstone_hard")
SOFT_MATERIAL_ID = 1
HARD_MATERIAL_ID = 2

# Bloore curvature weight: the round term of the tumbling law.
CURVATURE_WEIGHT = 1.2

# Sheltered pockets deepen under this weight, the sandstone hollow.
PIT_WEIGHT = 0.5

# Wear collected by a fully exposed corner per step.
WEAR_PACE = 0.2

# The wear depth the dials buy, as shares of the stone size: every
# stone loses the base share and weathering buys the rest, so the same
# dials wear the same physical depth at any resolution.
WEAR_DEPTH_BASE_SHARE = 0.05
WEAR_DEPTH_WEATHERING_SHARE = 0.22

# The soft hardness the step count is calibrated against.
REFERENCE_HARDNESS = 0.45

# The full light-to-dark tone runs; tone_bias slides along them, so
# the tone dial is a continuous position rather than three tiers.
SOFT_RUN = tuple(mid for tier in mineral.SANDSTONE_SOFT_TONE_MIDS for mid in tier)
HARD_RUN = tuple(mid for tier in mineral.SANDSTONE_HARD_TONE_MIDS for mid in tier)

# Sandstone is dominantly soft body with thin hard resistant bands;
# each material takes its own share of the thickness.
SOFT_THICKNESS_SHARE = 0.78
HARD_MEAN_RATIO = 0.3
MIN_LAYER_VOXEL_NUM = 1.25

# Per-layer tone jitter and drift, kept small so layers vary without
# leaving the tone the dial set.
TONE_LAYER_JITTER = 0.08
TONE_SPREAD = 0.12

# Boundary wander shares of the stone, each capped to its layer.
SOFT_WARP_SHARE = 0.04
HARD_WARP_SHARE = 0.02
MAX_BOUNDARY_WARP_SHARE = 0.2

# Internal laminae in the soft layers: subtle tone shifts spaced well
# apart, so they read as faint layering, not bold pinstripes.
LAMINATION_VOXEL_NUM = 9.0
LAMINATION_STRENGTH = 0.06

# The lamination_tilt dial tilts the laminae up to this angle (unit: radians).
MAX_LAMINATION_ANGLE = 0.5

# Every stone weathers at least this many steps so no face stays a
# perfect plane.
MIN_WEAR_STEP_NUM = 4


class _SandstoneShape:
    """The sandstone boulder's dial-to-piece mapping."""

    @staticmethod
    def make_layers(settings, rng: "cr.Rng") -> tuple:
        """Return interlayered soft layers and thin hard resistant bands.

        The shared strata thickness model gives each material its own
        realistic layer rhythm; the layers interleave soft, hard, soft, so
        every boundary is a real facies change. The tone dial sets
        where each layer sits in its material's light-to-dark run, and
        the soft layers carry cross-layered laminae.

        Args:
            settings: The validated sandstone boulder settings.
            rng: The random stream the layers draw from.
        """
        height = settings.size * (1.0 - 0.6 * settings.flatness)
        margin = 2.0 * MARGIN_VOXEL_NUM * settings.voxel_size
        total = math.ceil((height + margin) / settings.voxel_size) * settings.voxel_size
        min_thickness = MIN_LAYER_VOXEL_NUM * settings.voxel_size
        soft_total = total * SOFT_THICKNESS_SHARE
        soft_thicknesses = StrataGenerator.make_layer_thicknesses(
            rng.split("soft"), soft_total, settings.layer_thickness, min_thickness
        )
        hard_thicknesses = StrataGenerator.make_layer_thicknesses(
            rng.split("hard"),
            total - soft_total,
            HARD_MEAN_RATIO * settings.layer_thickness,
            min_thickness,
        )
        lamination_angle = MAX_LAMINATION_ANGLE * settings.lamination_tilt
        appearance_rng = rng.split("appearance")
        layers = []
        for index in range(max(len(soft_thicknesses), len(hard_thicknesses))):
            if index < len(soft_thicknesses):
                layers.append(
                    _SandstoneShape._make_layer(
                        settings,
                        appearance_rng.split(f"soft/{index}"),
                        SOFT_MATERIAL_ID,
                        soft_thicknesses[index],
                        SOFT_RUN,
                        SOFT_WARP_SHARE,
                        lamination_angle,
                    )
                )
            if index < len(hard_thicknesses):
                layers.append(
                    _SandstoneShape._make_layer(
                        settings,
                        appearance_rng.split(f"hard/{index}"),
                        HARD_MATERIAL_ID,
                        hard_thicknesses[index],
                        HARD_RUN,
                        HARD_WARP_SHARE,
                        0.0,
                    )
                )
        return tuple(layers)

    @staticmethod
    def _make_layer(
        settings, rng, material_id, thickness, run, warp_share, lamination_angle
    ) -> StrataGeneratorLayer:
        """Return one layer with its tone position and internal structure."""
        tone_bias = min(
            max(
                settings.tone + rng.uniform(-TONE_LAYER_JITTER, TONE_LAYER_JITTER), 0.0
            ),
            1.0,
        )
        warp_size = min(
            warp_share * settings.size * rng.uniform(0.5, 1.0),
            MAX_BOUNDARY_WARP_SHARE * thickness,
        )
        is_soft = material_id == SOFT_MATERIAL_ID
        return StrataGeneratorLayer(
            material_id=material_id,
            thickness=thickness,
            optical_mids=run,
            tone_bias=tone_bias,
            tone_spread=TONE_SPREAD,
            feature_size=0.4 * settings.size,
            warp_size=warp_size,
            warp_feature_size=0.8 * settings.size,
            lamination_feature_size=(
                LAMINATION_VOXEL_NUM * settings.voxel_size if is_soft else 0.0
            ),
            lamination_strength=LAMINATION_STRENGTH if is_soft else 0.0,
            lamination_angle=lamination_angle if is_soft else 0.0,
        )

    @staticmethod
    def make_erosion(settings, rng: "cr.Rng") -> BallErosion:
        """Return the tumbling wear the weathering dial buys.

        Args:
            settings: The validated sandstone boulder settings.
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
            ball_radius=0.12 * settings.size,
            wear_pace=WEAR_PACE,
            flat_weight=1.0,
            round_weight=CURVATURE_WEIGHT,
            pit_weight=PIT_WEIGHT,
            hardness_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_HARDNESS
            ),
            cavernous_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_CAVERNOUS
            ),
            durability_sway_ratio=0.6,
            durability_feature_size=0.25 * settings.size,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the surface carving that breaks the planar faces.

        Args:
            settings: The validated sandstone boulder settings.
            rng: The random stream the carving draws from.
        """
        return DetailErosion(
            rng,
            depth=settings.voxel_size,
            dent_spacing=0.25 * settings.size,
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
            default=0.4,
            doc="Trades a fresh angular fragment at 0 for a river-worn " "shape at 1.",
        ),
        flatness=cr.Float(
            0.0,
            1.0,
            default=0.2,
            doc="How much the stone flattens along its short side, 0 to 1; "
            "picks the Zingg class with elongation.",
        ),
        elongation=cr.Float(
            0.0,
            1.0,
            default=0.3,
            doc="How much the stone stretches along its long side, 0 to 1.",
        ),
        tone=cr.Float(
            0.0,
            1.0,
            default=0.35,
            doc="Runs the real palette from pale quarry stone at 0 to "
            "iron-stained dark stone at 1.",
        ),
        layer_thickness=cr.Float(
            0.05, 0.5, default=0.15, doc="The nominal strata spacing (unit: m)."
        ),
        fold=cr.Float(
            0.0,
            1.0,
            default=0.3,
            doc="How wavy the strata bend, 0 flat to 1 strongly folded.",
        ),
        lamination_tilt=cr.Float(
            0.0,
            1.0,
            default=0.2,
            doc="How steeply the internal laminae tilt into cross-layers, 0 "
            "flat to 1 dune foresets.",
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
    tags=("nature", "rock", "sandstone_boulder"),
)
def sandstone_boulder(editor_access, settings):
    """A layered sandstone boulder, one dynamic rigid body.

    Settings:
        size: The target diameter (unit: m).
        weathering: Trades a fresh angular fragment at 0 for a
            river-worn shape at 1.
        flatness: How much the stone flattens along its short side, 0
            to 1; picks the Zingg class with elongation.
        elongation: How much the stone stretches along its long side,
            0 to 1.
        tone: Runs the real palette from pale quarry stone at 0 to
            iron-stained dark stone at 1.
        layer_thickness: The nominal strata spacing (unit: m).
        fold: How wavy the strata bend, 0 flat to 1 strongly folded.
        lamination_tilt: How steeply the internal laminae tilt into
            cross-layers, 0 flat to 1 dune foresets.
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
    """
    rng = cr.Rng(settings.seed)
    grid, device, box_size = RockBuild.make_boulder_grid(
        settings,
        "sandstone_boulder",
        settings.size,
        settings.flatness,
        settings.elongation,
    )
    volume = StrataGenerator(
        grid,
        str(device),
        rng.split("generator"),
        _SandstoneShape.make_layers(settings, rng.split("layers")),
        shape="box",
        fold_size=0.35 * settings.fold * settings.size,
        fold_feature_size=0.55 * settings.size,
        box_size=box_size,
    ).generate()
    return RockBuild.make_boulder_draft(
        editor_access,
        settings,
        kind_name="sandstone_boulder",
        material_names=MATERIAL_NAMES,
        volume=volume,
        rng=rng,
        size=settings.size,
        flatness=settings.flatness,
        elongation=settings.elongation,
        erosion=_SandstoneShape.make_erosion(settings, rng.split("erosion")),
        detail=_SandstoneShape.make_detail(settings, rng.split("detail")),
    )
