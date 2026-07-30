# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The marble block: a broken veined mass for cliffs and hero rock.

The block is the fixed hero form of the marble boulder: vein
generation threads the bright body with the same branching network
over the whole box, an occupancy grid can carve the box into any
rough shape, a jittered seam cut breaks the mass, and gentle wear
opens the seams and smooths the corners into the tumbled sheen.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm.sediment import (
    BallErosion,
    DetailErosion,
    VeinGenerator,
)

from ...material.nature import mineral
from .build import RockBuild

MODULE = "rise_origin.nature.rock.marble_block"

# Test-asserted voxel cap at the default dials (design doc budget).
FORMATION_VOXEL_CAP = 900_000

# The one material; volume material id 1.
MATERIAL_NAMES = ("marble",)
MARBLE_MATERIAL_ID = 1

# The vein network cells sit this far apart, as a share of the block's
# shortest side; wide cells keep the network sparse so its veins stay
# near one voxel thin and read as a web rather than filling the mass.
VEIN_SIZE_SHARE = 0.35

# This share of the network walls stays a vein; the rest drop out, so
# the veins read as partial paths instead of a complete cell division.
VEIN_PATH_SHARE = 0.55
BOLD_VEIN_PATH_SHARE = 0.35

# The network veins step this far along the tone run away from the
# body, so a white stone veins dark and a black one veins light; the
# bold veins carry the whole way to the far end, and the short step
# keeps the two apart wherever the body sits.
VEIN_TONE_STEP_NUM = 2

# Gentle wear at block scale: marble smooths rather than pits, with
# depth and ball radius taken as shares of the block's shortest side.
CURVATURE_WEIGHT = 1.2
WEAR_PACE = 0.3
WEAR_DEPTH_BASE_SHARE = 0.02
WEAR_DEPTH_WEATHERING_SHARE = 0.09
BALL_RADIUS_SHARE = 0.11
REFERENCE_HARDNESS = 0.6
MIN_WEAR_STEP_NUM = 6

# The light surface carving over the worn shape.
DENT_SPACING = 0.1
DENT_WIDTH_VOXEL_NUM = 2.2


class _MarbleShape:
    """The marble block's dial-to-piece mapping."""

    @staticmethod
    def make_reference_size(settings) -> float:
        """Return the block's shortest side, the wear reference length.

        Args:
            settings: The validated marble block settings.
        """
        return min(settings.width, settings.depth, settings.height)

    @staticmethod
    def make_tone_mids(tone: float) -> tuple:
        """Return the body tone and its two vein tones.

        The dial picks one step of the tone run for the whole body, and
        the veins step away from it toward the far end, so a white
        stone veins dark and a black one veins light.

        Args:
            tone: Where the body sits on the run, 0 white to 1 black.
        """
        last = mineral.MARBLE_TONE_LENGTH - 1
        body = int(round(min(max(tone, 0.0), 1.0) * last))
        away = -1 if body > last / 2.0 else 1
        network = min(max(body + away * VEIN_TONE_STEP_NUM, 0), last)
        bold = 0 if away < 0 else last
        return body, (network, bold)

    @staticmethod
    def make_erosion(settings, rng: "cr.Rng") -> BallErosion:
        """Return the gentle wear that opens seams and smooths corners.

        Args:
            settings: The validated marble block settings.
            rng: The random stream the durability sway draws from.
        """
        reference = _MarbleShape.make_reference_size(settings)
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
            flat_weight=1.0,
            round_weight=CURVATURE_WEIGHT,
            pit_weight=0.0,
            hardness_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_HARDNESS
            ),
            durability_sway_ratio=0.35,
            durability_feature_size=0.3 * reference,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the light surface carving over the worn shape.

        Args:
            settings: The validated marble block settings.
            rng: The random stream the carving draws from.
        """
        return DetailErosion(
            rng,
            depth=settings.voxel_size,
            dent_spacing=DENT_SPACING,
            dent_width=DENT_WIDTH_VOXEL_NUM * settings.voxel_size,
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
        tone=cr.Float(
            0.0,
            1.0,
            default=0.0,
            doc="The one tone the whole block carries, 0 white Carrara to "
            "1 black Marquina; the veins take the far end of the run.",
        ),
        weathering=cr.Float(
            0.0,
            1.0,
            default=0.35,
            doc="How far wear carries the surface back inside the given "
            "shape, 0 "
            "fresh-cut to 1 heavily worn.",
        ),
        vein_thickness=cr.Float(
            0.0,
            1.0,
            default=0.3,
            doc="How bold the veins are, 0 hairline feathery to 1 broad "
            "and dominant.",
        ),
        vein_winding=cr.Float(
            0.0,
            1.0,
            default=0.6,
            doc="How far the veins wander, 0 straight to 1 sinuous.",
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
    tags=("nature", "rock", "marble_block"),
)
def marble_block(editor_access, settings):
    """A seamed marble block, fixed scenery for cliffs and hero rock.

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
        tone: The one tone the whole block carries, 0 white Carrara to
            1 black Marquina; the veins take the far end of the run.
        weathering: How far wear carries the surface back inside the
            given shape, 0 fresh-cut to 1 heavily worn.
        vein_thickness: How bold the veins are, 0 hairline feathery to
            1 broad and dominant.
        vein_winding: How far the veins wander, 0 straight to 1 sinuous.
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
        is_fixed: True stands the block as static scenery; False
            marshals it as one dynamic rigid body.
    """
    rng = cr.Rng(settings.seed)
    box_size = (settings.width, settings.depth, settings.height)
    grid, device = RockBuild.make_box_grid(
        settings, "marble_block", box_size, settings.seam_spacing
    )
    reference = _MarbleShape.make_reference_size(settings)
    body_mid, vein_mids = _MarbleShape.make_tone_mids(settings.tone)
    volume = VeinGenerator(
        grid,
        str(device),
        rng.split("generator"),
        material_id=MARBLE_MATERIAL_ID,
        body_mid=body_mid,
        vein_mids=vein_mids,
        vein_feature_size=VEIN_SIZE_SHARE * reference,
        vein_share=0.035 + 0.03 * settings.vein_thickness,
        bold_vein_share=0.03 + 0.03 * settings.vein_thickness,
        vein_path_share=VEIN_PATH_SHARE,
        bold_vein_path_share=BOLD_VEIN_PATH_SHARE,
        winding=settings.vein_winding,
        shape="box",
        box_size=box_size,
        box_occupancy=settings.occupancy,
    ).generate()
    return RockBuild.make_block_draft(
        editor_access,
        settings,
        kind_name="marble_block",
        material_names=MATERIAL_NAMES,
        volume=volume,
        rng=rng,
        box_size=box_size,
        seam_spacing=settings.seam_spacing,
        seam_width=settings.seam_width,
        erosion=_MarbleShape.make_erosion(settings, rng.split("erosion")),
        detail=_MarbleShape.make_detail(settings, rng.split("detail")),
    )
