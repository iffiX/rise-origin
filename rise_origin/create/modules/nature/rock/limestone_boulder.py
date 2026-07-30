# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The limestone boulder: pale layered rock, weathered into karst pits.

Limestone is one soft carbonate laid in subtle layers: strata generation
drifts the body tone gently from layer to layer, and the layering dial sets
whether the layers barely shift (massive) or read clearly (well-layered).
Ball erosion then rounds the block and hollows karst pits, because
limestone dissolves in rain, so its cavernous weight runs high.
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

MODULE = "rise_origin.nature.rock.limestone_boulder"

# Test-asserted voxel cap at the default dials (design doc budget).
STONE_VOXEL_CAP = 120_000

# The one material; volume material id 1.
MATERIAL_NAMES = ("limestone",)
LIMESTONE_MATERIAL_ID = 1

# The body tone run, warm buff through white to cool gray.
BODY_RUN = mineral.LIMESTONE_BODY_MIDS

# How far the layering dial jitters each layer's tone around the tone
# dial, and how far the tone drifts within a layer; both kept small so a
# stone's tones stay close to each other.
MAX_LAYER_TONE_JITTER = 0.2
TONE_SPREAD = 0.3
MIN_LAYER_VOXEL_NUM = 1.25

# Real limestone breaks along its nodules and fossils, not clean
# planes, so the fracture faces wander by this much at this nodule
# scale, giving the blocky stone its rough surfaces.
FRACTURE_ROUGHNESS_SHARE = 0.045
FRACTURE_ROUGHNESS_FEATURE_SHARE = 0.16

# A limestone block is broken more than once, so a few chunks knock off
# its corners after the first cut, turning the convex block into a
# complex non-convex stone; the reach stays modest so a chunk never
# bites clean through a thin stone.
CHUNK_NUM = 3
CHUNK_RADIUS_SHARE = 0.22

# Ball wear rounds the sharp fracture edges hard so no right-angle
# corner survives, on a ball wide enough to soften whole corners; the
# flat term stays low so the faces barely recede and the stone keeps
# its size while the corners round. Its durability sway is the
# differential weathering that leaves hard nodules proud and dissolves
# the soft matrix back, and the karst dial drives the pit term.
FLAT_WEIGHT = 0.4
CURVATURE_WEIGHT = 2.0
BALL_RADIUS_SHARE = 0.18
PIT_WEIGHT = 0.9
WEAR_PACE = 0.3
WEAR_DEPTH_BASE_SHARE = 0.08
WEAR_DEPTH_WEATHERING_SHARE = 0.22
REFERENCE_HARDNESS = 0.5
MIN_WEAR_STEP_NUM = 6
DURABILITY_SWAY_RATIO = 0.8
DURABILITY_FEATURE_SHARE = 0.2

# The karst carve peppers the surface with small dense dissolution
# pits, a fine detail rather than a reshaping: a narrow feature so the
# pits stay a few voxels wide, a depth tied to the emitted voxel so a
# pit is always about one voxel deep at any resolution and never
# punches a hole or sheds a loose edge voxel, and a high dent share so
# the pits sit close together. The karst dial's real pocking runs
# through the ball wear, not this detail.
KARST_HOLE_SPACING = 0.04
KARST_HOLE_DEPTH_VOXEL_NUM = 1.0
KARST_HOLE_SHARE = 0.7


class _LimestoneShape:
    """The limestone boulder's dial-to-piece mapping."""

    @staticmethod
    def make_layers(settings, rng: "cr.Rng") -> tuple:
        """Return the subtle limestone layers, one carbonate material.

        Args:
            settings: The validated limestone boulder settings.
            rng: The random stream the layer tones draw from.
        """
        height = settings.size * (1.0 - 0.6 * settings.flatness)
        margin = 2.0 * MARGIN_VOXEL_NUM * settings.voxel_size
        total = math.ceil((height + margin) / settings.voxel_size) * settings.voxel_size
        min_thickness = MIN_LAYER_VOXEL_NUM * settings.voxel_size
        thicknesses = StrataGenerator.make_layer_thicknesses(
            rng.split("thickness"), total, settings.layer_thickness, min_thickness
        )
        jitter = MAX_LAYER_TONE_JITTER * settings.layering
        appearance_rng = rng.split("appearance")
        layers = []
        for index, thickness in enumerate(thicknesses):
            layer_rng = appearance_rng.split(str(index))
            tone_bias = min(
                max(settings.tone + layer_rng.uniform(-jitter, jitter), 0.0), 1.0
            )
            layers.append(
                StrataGeneratorLayer(
                    material_id=LIMESTONE_MATERIAL_ID,
                    thickness=thickness,
                    optical_mids=BODY_RUN,
                    tone_bias=tone_bias,
                    tone_spread=TONE_SPREAD,
                    feature_size=settings.mottle_size,
                    warp_size=0.03 * settings.size,
                    warp_feature_size=0.8 * settings.size,
                )
            )
        return tuple(layers)

    @staticmethod
    def make_erosion(settings, rng: "cr.Rng") -> BallErosion:
        """Return the ball wear that softens edges and pits karst.

        The wear stays light so the blocky shape holds; its durability
        sway is the differential weathering that roughens the surface,
        and the pit term hollows the karst pockets.

        Args:
            settings: The validated limestone boulder settings.
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
            ball_radius=BALL_RADIUS_SHARE * settings.size,
            wear_pace=WEAR_PACE,
            flat_weight=FLAT_WEIGHT,
            round_weight=CURVATURE_WEIGHT,
            pit_weight=PIT_WEIGHT * settings.karst,
            hardness_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_HARDNESS
            ),
            cavernous_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_CAVERNOUS
            ),
            durability_sway_ratio=DURABILITY_SWAY_RATIO,
            durability_feature_size=DURABILITY_FEATURE_SHARE * settings.size,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the coarse karst carve that hollows big dissolution pits.

        Args:
            settings: The validated limestone boulder settings.
            rng: The random stream the carving draws from.
        """
        return DetailErosion(
            rng,
            depth=KARST_HOLE_DEPTH_VOXEL_NUM * settings.voxel_size,
            dent_spacing=KARST_HOLE_SPACING,
            dent_width=2.2 * settings.voxel_size,
            dent_share=KARST_HOLE_SHARE,
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
            default=0.45,
            doc="Trades a fresh block at 0 for a worn shape at 1.",
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
            default=0.2,
            doc="How much the stone stretches along its long side, 0 to 1.",
        ),
        tone=cr.Float(
            0.0,
            1.0,
            default=0.25,
            doc="Runs the body from warm buff at 0 through white to cool " "gray at 1.",
        ),
        layering=cr.Float(
            0.0,
            1.0,
            default=0.4,
            doc="How clearly the layers read, 0 massive to 1 well-layered.",
        ),
        karst=cr.Float(
            0.0,
            1.0,
            default=0.6,
            doc="How deeply rain pits the surface, 0 smooth to 1 pocked.",
        ),
        layer_thickness=cr.Float(
            0.05, 0.5, default=0.18, doc="The nominal layer spacing (unit: m)."
        ),
        mottle_size=cr.Float(
            0.03,
            0.2,
            default=0.08,
            doc="The size of the within-layer tone mottling (unit: m).",
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
    tags=("nature", "rock", "limestone_boulder"),
)
def limestone_boulder(editor_access, settings):
    """A limestone boulder, one dynamic rigid body.

    Settings:
        size: The target diameter (unit: m).
        weathering: Trades a fresh block at 0 for a worn shape at 1.
        flatness: How much the stone flattens along its short side, 0
            to 1; picks the Zingg class with elongation.
        elongation: How much the stone stretches along its long side,
            0 to 1.
        tone: Runs the body from warm buff at 0 through white to cool
            gray at 1, so a low tone reads buff and white and a high
            tone gray and white.
        layering: How clearly the layers read, 0 massive to 1 well-layered.
        karst: How deeply rain pits the surface, 0 smooth to 1 pocked.
        layer_thickness: The nominal layer spacing (unit: m).
        mottle_size: The size of the within-layer tone mottling (unit: m).
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
    """
    rng = cr.Rng(settings.seed)
    grid, device, box_size = RockBuild.make_boulder_grid(
        settings,
        "limestone_boulder",
        settings.size,
        settings.flatness,
        settings.elongation,
    )
    volume = StrataGenerator(
        grid,
        str(device),
        rng.split("generator"),
        _LimestoneShape.make_layers(settings, rng.split("layers")),
        shape="box",
        box_size=box_size,
    ).generate()
    return RockBuild.make_boulder_draft(
        editor_access,
        settings,
        kind_name="limestone_boulder",
        material_names=MATERIAL_NAMES,
        volume=volume,
        rng=rng,
        size=settings.size,
        flatness=settings.flatness,
        elongation=settings.elongation,
        erosion=_LimestoneShape.make_erosion(settings, rng.split("erosion")),
        detail=_LimestoneShape.make_detail(settings, rng.split("detail")),
        surface_roughness=FRACTURE_ROUGHNESS_SHARE * settings.size,
        roughness_feature_size=FRACTURE_ROUGHNESS_FEATURE_SHARE * settings.size,
        chunk_num=CHUNK_NUM,
        chunk_radius=CHUNK_RADIUS_SHARE * settings.size,
    )
