# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The sandstone block: a broken layered mass for cliffs and hero rock.

The block is the fixed hero form of the sandstone boulder: strata
generation lays the same soft layers and hard resistant bands over the
whole box, an occupancy grid can carve the box into any rough shape, a
jittered seam cut breaks the mass, and ball erosion opens the seams
into cracks while ledges appear where soft layers wear faster.
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

MODULE = "rise_origin.nature.rock.sandstone_block"

# Test-asserted voxel cap at the default dials (design doc budget).
FORMATION_VOXEL_CAP = 900_000

# The materials in job order; volume material ids start at 1.
MATERIAL_NAMES = ("sandstone_soft", "sandstone_hard")
SOFT_MATERIAL_ID = 1
HARD_MATERIAL_ID = 2

# The full light-to-dark tone runs; tone_bias slides along them.
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

# Boundary wander shares of the block reference size, capped per layer.
SOFT_WARP_SHARE = 0.04
HARD_WARP_SHARE = 0.02
MAX_BOUNDARY_WARP_SHARE = 0.2

# Internal laminae in the soft layers: subtle tone shifts spaced well
# apart, so they read as faint layering, not bold pinstripes.
LAMINATION_VOXEL_NUM = 9.0
LAMINATION_STRENGTH = 0.06

# The lamination_tilt dial tilts the laminae up to this angle (unit: radians).
MAX_LAMINATION_ANGLE = 0.5

# Ball wear at block scale, with depth and ball radius taken as shares
# of the block's shortest side, so ledges and open seams scale with the
# block. The flat term carries the whole face back, and it has to clear
# the curvature term's reading of a flat wall or faces stand untouched
# and the wear depth buys nothing.
FLAT_WEIGHT = 1.0
CURVATURE_WEIGHT = 1.2
PIT_WEIGHT = 0.5
WEAR_PACE = 0.2
WEAR_DEPTH_BASE_SHARE = 0.01
WEAR_DEPTH_WEATHERING_SHARE = 0.12
BALL_RADIUS_SHARE = 0.1
REFERENCE_HARDNESS = 0.45
MIN_WEAR_STEP_NUM = 4
SEAM_RATE_SCALE = 6.0
DURABILITY_SWAY_RATIO = 0.5
DURABILITY_FEATURE_SHARE = 0.2

# The surface carving that breaks the planar faces.
DENT_SPACING_SHARE = 0.25
DENT_WIDTH_VOXEL_NUM = 2.2


class _SandstoneShape:
    """The sandstone block's dial-to-piece mapping."""

    @staticmethod
    def make_reference_size(settings) -> float:
        """Return the block's shortest side, the wear reference length.

        Args:
            settings: The validated sandstone block settings.
        """
        return min(settings.width, settings.depth, settings.height)

    @staticmethod
    def make_layers(settings, rng: "cr.Rng") -> tuple:
        """Return interlayered soft layers and thin hard resistant bands.

        The shared strata thickness model gives each material its own
        realistic layer rhythm; the layers interleave soft, hard, soft,
        so every boundary is a real facies change.

        Args:
            settings: The validated sandstone block settings.
            rng: The random stream the layers draw from.
        """
        reference = _SandstoneShape.make_reference_size(settings)
        margin = 2.0 * MARGIN_VOXEL_NUM * settings.voxel_size
        total = (
            math.ceil((settings.height + margin) / settings.voxel_size)
            * settings.voxel_size
        )
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
                        SOFT_WARP_SHARE * reference,
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
                        HARD_WARP_SHARE * reference,
                        0.0,
                    )
                )
        return tuple(layers)

    @staticmethod
    def make_erosion(settings, rng: "cr.Rng") -> BallErosion:
        """Return the weathering that wears the block back.

        The flat term carries whole faces back while the curvature term
        takes the corners first and the seams open into cracks, so the
        block breaks into worn pieces instead of shrinking as a box.

        Args:
            settings: The validated sandstone block settings.
            rng: The random stream the durability sway draws from.
        """
        reference = _SandstoneShape.make_reference_size(settings)
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
            pit_weight=PIT_WEIGHT,
            hardness_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_HARDNESS
            ),
            cavernous_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_CAVERNOUS
            ),
            durability_sway_ratio=DURABILITY_SWAY_RATIO,
            durability_feature_size=DURABILITY_FEATURE_SHARE * reference,
            seam_rate_scale=SEAM_RATE_SCALE,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the surface carving that breaks the planar faces.

        Args:
            settings: The validated sandstone block settings.
            rng: The random stream the carving draws from.
        """
        reference = _SandstoneShape.make_reference_size(settings)
        return DetailErosion(
            rng,
            depth=settings.voxel_size,
            dent_spacing=DENT_SPACING_SHARE * reference,
            dent_width=DENT_WIDTH_VOXEL_NUM * settings.voxel_size,
        )

    @staticmethod
    def _make_layer(
        settings, rng, material_id, thickness, run, warp_size, lamination_angle
    ) -> StrataGeneratorLayer:
        """Return one layer with its tone position and internal structure."""
        reference = _SandstoneShape.make_reference_size(settings)
        tone_bias = min(
            max(
                settings.tone + rng.uniform(-TONE_LAYER_JITTER, TONE_LAYER_JITTER), 0.0
            ),
            1.0,
        )
        bounded_warp = min(
            warp_size * rng.uniform(0.5, 1.0), MAX_BOUNDARY_WARP_SHARE * thickness
        )
        is_soft = material_id == SOFT_MATERIAL_ID
        return StrataGeneratorLayer(
            material_id=material_id,
            thickness=thickness,
            optical_mids=run,
            tone_bias=tone_bias,
            tone_spread=TONE_SPREAD,
            feature_size=0.4 * reference,
            warp_size=bounded_warp,
            warp_feature_size=0.8 * reference,
            lamination_feature_size=(
                LAMINATION_VOXEL_NUM * settings.voxel_size if is_soft else 0.0
            ),
            lamination_strength=LAMINATION_STRENGTH if is_soft else 0.0,
            lamination_angle=lamination_angle if is_soft else 0.0,
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
            "shape, 0 fresh-cut to 1 heavily worn.",
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
    tags=("nature", "rock", "sandstone_block"),
)
def sandstone_block(editor_access, settings):
    """A seamed sandstone block, fixed scenery for cliffs and hero rock.

    Settings:
        width: The block length along x (unit: m).
        depth: The block length along y (unit: m).
        height: The block height (unit: m).
        occupancy: An optional 3D bool grid stretched over the box,
            each dimension divided independently; absent cells carve
            the block away, and the block only ever wears back inside
            what the grid gives it.
        seam_spacing: The nominal seam piece spacing the mass breaks
            along (unit: m).
        seam_width: How wide the seams between pieces read (unit: m);
            finer than one voxel leaves the mass uncracked.
        weathering: How far wear carries the surface back inside the
            given shape, 0 fresh-cut to 1 heavily worn.
        tone: Runs the real palette from pale quarry stone at 0 to
            iron-stained dark stone at 1.
        layer_thickness: The nominal strata spacing (unit: m).
        fold: How wavy the strata bend, 0 flat to 1 strongly folded.
        lamination_tilt: How steeply the internal laminae tilt into
            cross-layers, 0 flat to 1 dune foresets.
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
        is_fixed: True stands the block as static scenery; False
            marshals it as one dynamic rigid body.
    """
    rng = cr.Rng(settings.seed)
    box_size = (settings.width, settings.depth, settings.height)
    grid, device = RockBuild.make_box_grid(
        settings, "sandstone_block", box_size, settings.seam_spacing
    )
    reference = _SandstoneShape.make_reference_size(settings)
    volume = StrataGenerator(
        grid,
        str(device),
        rng.split("generator"),
        _SandstoneShape.make_layers(settings, rng.split("layers")),
        shape="box",
        fold_size=0.35 * settings.fold * reference,
        fold_feature_size=0.55 * reference,
        box_size=box_size,
        box_occupancy=settings.occupancy,
    ).generate()
    return RockBuild.make_block_draft(
        editor_access,
        settings,
        kind_name="sandstone_block",
        material_names=MATERIAL_NAMES,
        volume=volume,
        rng=rng,
        box_size=box_size,
        seam_spacing=settings.seam_spacing,
        seam_width=settings.seam_width,
        erosion=_SandstoneShape.make_erosion(settings, rng.split("erosion")),
        detail=_SandstoneShape.make_detail(settings, rng.split("detail")),
    )
