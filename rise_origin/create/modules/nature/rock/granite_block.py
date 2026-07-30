# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The granite block: a broken speckled mass for cliffs and hero rock.

The block is the fixed hero form of the granite boulder: grain
generation fills the whole box with the same mineral speckle, an
occupancy grid can carve the box into any rough shape, a jittered
seam cut breaks the mass, and slow spheroidal wear opens the seams
into cracks and rounds the corners into the weathered tor look.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm.sediment import (
    BallErosion,
    DetailErosion,
    GrainGenerator,
)

from ...material.nature import mineral
from .build import RockBuild

MODULE = "rise_origin.nature.rock.granite_block"

# Test-asserted voxel cap at the default dials (design doc budget).
FORMATION_VOXEL_CAP = 900_000

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

# Spheroidal wear at block scale: granite exfoliates, rounding corners
# without honeycomb pitting, with depth and ball radius taken as
# shares of the block's shortest side.
CURVATURE_WEIGHT = 1.4
WEAR_PACE = 0.3
WEAR_DEPTH_BASE_SHARE = 0.015
WEAR_DEPTH_WEATHERING_SHARE = 0.08
BALL_RADIUS_SHARE = 0.12
REFERENCE_HARDNESS = 1.0
MIN_WEAR_STEP_NUM = 4

# The gritty surface carving over the worn shape.
DENT_SPACING = 0.05
DENT_WIDTH_VOXEL_NUM = 2.2


class _GraniteShape:
    """The granite block's dial-to-piece mapping."""

    @staticmethod
    def make_reference_size(settings) -> float:
        """Return the block's shortest side, the wear reference length.

        Args:
            settings: The validated granite block settings.
        """
        return min(settings.width, settings.depth, settings.height)

    @staticmethod
    def make_grain_mids(settings) -> tuple:
        """Return the ordered mineral groups the tone dial sets.

        Each entry is one mineral's shade group; the mix is ordered
        plagioclase, orthoclase, quartz, biotite, so the generator's
        cluster field slides a window over it and the minerals group
        into patches. Slot repetition sets each mineral's share, and
        the tone dial raises the pink orthoclase.

        Args:
            settings: The validated granite block settings.
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
        """Return the slow spheroidal wear that opens seams and corners.

        Args:
            settings: The validated granite block settings.
            rng: The random stream the durability sway draws from.
        """
        reference = _GraniteShape.make_reference_size(settings)
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
            durability_sway_ratio=0.4,
            durability_feature_size=0.3 * reference,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the gritty surface carving over the worn shape.

        Args:
            settings: The validated granite block settings.
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
    tags=("nature", "rock", "granite_block"),
)
def granite_block(editor_access, settings):
    """A seamed granite block, fixed scenery for cliffs and hero rock.

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
        tone: Runs the grain mix from gray granite at 0 to pink
            potassium-feldspar granite at 1.
        grain_size: The mineral crystal size (unit: m).
        cluster_size: The size of the same-mineral patches (unit: m).
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
        is_fixed: True stands the block as static scenery; False
            marshals it as one dynamic rigid body.
    """
    rng = cr.Rng(settings.seed)
    box_size = (settings.width, settings.depth, settings.height)
    grid, device = RockBuild.make_box_grid(
        settings, "granite_block", box_size, settings.seam_spacing
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
        box_occupancy=settings.occupancy,
    ).generate()
    return RockBuild.make_block_draft(
        editor_access,
        settings,
        kind_name="granite_block",
        material_names=MATERIAL_NAMES,
        volume=volume,
        rng=rng,
        box_size=box_size,
        seam_spacing=settings.seam_spacing,
        seam_width=settings.seam_width,
        erosion=_GraniteShape.make_erosion(settings, rng.split("erosion")),
        detail=_GraniteShape.make_detail(settings, rng.split("detail")),
    )
