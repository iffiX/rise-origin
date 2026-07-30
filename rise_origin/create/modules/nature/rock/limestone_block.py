# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The limestone block: a broken pale mass pitted into karst.

The block is the fixed hero form of the limestone boulder: strata
generation drifts the subtle carbonate layers over the whole box, an
occupancy grid can carve the box into any rough shape, a jittered
seam cut breaks the mass, and ball erosion opens the seams while the
karst dial hollows dissolution pits over the faces.
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

MODULE = "rise_origin.nature.rock.limestone_block"

# Test-asserted voxel cap at the default dials (design doc budget).
FORMATION_VOXEL_CAP = 900_000

# The one material; volume material id 1.
MATERIAL_NAMES = ("limestone",)
LIMESTONE_MATERIAL_ID = 1

# The body tone run, warm buff through white to cool gray.
BODY_RUN = mineral.LIMESTONE_BODY_MIDS

# How far the layering dial jitters each layer's tone around the tone
# dial, and how far the tone drifts within a layer; both kept small so
# a block's tones stay close to each other.
MAX_LAYER_TONE_JITTER = 0.2
TONE_SPREAD = 0.3
MIN_LAYER_VOXEL_NUM = 1.25

# Ball wear at block scale: light flat recession, hard corner
# rounding, and the karst dial driving the pit term, with depth and
# ball radius taken as shares of the block's shortest side.
FLAT_WEIGHT = 1.0
CURVATURE_WEIGHT = 2.0
BALL_RADIUS_SHARE = 0.14
PIT_WEIGHT = 0.9
WEAR_PACE = 0.3
WEAR_DEPTH_BASE_SHARE = 0.04
WEAR_DEPTH_WEATHERING_SHARE = 0.1
REFERENCE_HARDNESS = 0.5
MIN_WEAR_STEP_NUM = 6
DURABILITY_SWAY_RATIO = 0.8
DURABILITY_FEATURE_SHARE = 0.2

# The karst carve peppers the surface with small dense dissolution
# pits, a fine detail rather than a reshaping.
KARST_HOLE_SPACING = 0.04
KARST_HOLE_DEPTH_VOXEL_NUM = 1.0
KARST_HOLE_SHARE = 0.7


class _LimestoneShape:
    """The limestone block's dial-to-piece mapping."""

    @staticmethod
    def make_reference_size(settings) -> float:
        """Return the block's shortest side, the wear reference length.

        Args:
            settings: The validated limestone block settings.
        """
        return min(settings.width, settings.depth, settings.height)

    @staticmethod
    def make_layers(settings, rng: "cr.Rng") -> tuple:
        """Return the subtle limestone layers, one carbonate material.

        Args:
            settings: The validated limestone block settings.
            rng: The random stream the layer tones draw from.
        """
        reference = _LimestoneShape.make_reference_size(settings)
        margin = 2.0 * MARGIN_VOXEL_NUM * settings.voxel_size
        total = (
            math.ceil((settings.height + margin) / settings.voxel_size)
            * settings.voxel_size
        )
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
                    warp_size=0.03 * reference,
                    warp_feature_size=0.8 * reference,
                )
            )
        return tuple(layers)

    @staticmethod
    def make_erosion(settings, rng: "cr.Rng") -> BallErosion:
        """Return the wear that opens seams, rounds corners, and pits.

        Args:
            settings: The validated limestone block settings.
            rng: The random stream the durability sway draws from.
        """
        reference = _LimestoneShape.make_reference_size(settings)
        depth = (
            WEAR_DEPTH_BASE_SHARE + WEAR_DEPTH_WEATHERING_SHARE * settings.weathering
        ) * reference
        cell_size = settings.voxel_size / settings.wear_subdivision
        step_num = max(
            MIN_WEAR_STEP_NUM,
            int(round(depth / cell_size * REFERENCE_HARDNESS / WEAR_PACE)),
        )
        return BallErosion(
            rng,
            step_num=step_num,
            ball_radius=BALL_RADIUS_SHARE * reference,
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
            durability_feature_size=DURABILITY_FEATURE_SHARE * reference,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the karst carve that hollows small dissolution pits.

        Args:
            settings: The validated limestone block settings.
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
        width=cr.Float(
            0.5, 8.0, default=2.5, doc="The block length along x (unit: m)."
        ),
        depth=cr.Float(
            0.5, 8.0, default=1.2, doc="The block length along y (unit: m)."
        ),
        height=cr.Float(0.5, 12.0, default=1.8, doc="The block height (unit: m)."),
        occupancy=cr.Array3D(
            default=None,
            doc="An optional 3D bool grid stretched over the box; absent "
            "cells carve the block away and the edges round at grid-cell "
            "scale.",
        ),
        seam_spacing=cr.Float(
            0.3,
            2.5,
            default=0.8,
            doc="The nominal seam piece spacing the mass breaks along (unit: m).",
        ),
        seam_width=cr.Float(
            0.0,
            0.2,
            default=0.05,
            doc="How wide the seams between pieces read (unit: m); finer "
            "than one voxel leaves the mass uncracked.",
        ),
        weathering=cr.Float(
            0.0,
            1.0,
            default=0.35,
            doc="How far wear carries the surface back inside the given "
            "shape, 0 "
            "fresh-cut to 1 heavily worn.",
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
            6,
            default=1,
            doc="The unitless ratio of internal wear cells per emitted voxel edge.",
        ),
        is_fixed=cr.Bool(
            default=True,
            doc="True stands the block as static scenery; False marshals it "
            "as one dynamic rigid body.",
        ),
    ),
    dependencies=(f"{mineral.MODULE}@^1",),
    tags=("nature", "rock", "limestone_block"),
)
def limestone_block(editor_access, settings):
    """A seamed limestone block, fixed scenery for cliffs and hero rock.

    Settings:
        width: The block length along x (unit: m).
        depth: The block length along y (unit: m).
        height: The block height (unit: m).
        occupancy: An optional 3D bool grid stretched over the box,
            each dimension divided independently; absent cells carve
            the block away, and the carved edges round and roughen at
            grid-cell scale instead of following the cell walls.
        seam_spacing: The nominal seam piece spacing the mass breaks
            along (unit: m).
        seam_width: How wide the seams between pieces read (unit: m);
            finer than one voxel leaves the mass uncracked.
        weathering: How far wear carries the surface back inside the
            given shape, 0 fresh-cut to 1 heavily worn.
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
        is_fixed: True stands the block as static scenery; False
            marshals it as one dynamic rigid body.
    """
    rng = cr.Rng(settings.seed)
    box_size = (settings.width, settings.depth, settings.height)
    grid, device = RockBuild.make_box_grid(
        settings, "limestone_block", box_size, settings.seam_spacing
    )
    volume = StrataGenerator(
        grid,
        str(device),
        rng.split("generator"),
        _LimestoneShape.make_layers(settings, rng.split("layers")),
        shape="box",
        box_size=box_size,
        box_occupancy=settings.occupancy,
    ).generate()
    return RockBuild.make_block_draft(
        editor_access,
        settings,
        kind_name="limestone_block",
        material_names=MATERIAL_NAMES,
        volume=volume,
        rng=rng,
        box_size=box_size,
        seam_spacing=settings.seam_spacing,
        seam_width=settings.seam_width,
        erosion=_LimestoneShape.make_erosion(settings, rng.split("erosion")),
        detail=_LimestoneShape.make_detail(settings, rng.split("detail")),
    )
