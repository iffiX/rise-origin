# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The pebble bed: a river bed of rounded stones resting on each other.

The pile generator packs the bed with stones grown into contact, pile
wear rounds each one and opens the gaps between them while holding the
contact points, and the marshal wakes the stones near the surface as
moving bodies over a still base. Every stone is river stone, and each
one carries a single shade off that material's gray run.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm.sediment import (
    DetailErosion,
    PileErosion,
    PileGenerator,
    SegmentTone,
)

from ...material.nature import mineral
from .build import RockBuild

MODULE = "rise_origin.nature.rock.pebble_bed"

# Test-asserted voxel cap at the default dials (design doc budget).
BED_VOXEL_CAP = 250_000

# The one material; volume material id 1.
MATERIAL_NAMES = ("river_stone",)
MATERIAL_SHARES = (1.0,)

# The run every stone draws its one shade from, light end first.
TONE_ROWS = ((), mineral.RIVER_STONE_TONE_MIDS)

# How far one stone's shade swings from its neighbors', and how wide the
# neighborhoods that share a shade run, as a share of the stone size.
SHADE_SWING_RATIO = 0.55
SHADE_DRIFT_SIZE_NUM = 6.0

# Pile wear. The flat term must stay above about a tenth of the round
# term or flat surfaces clamp to no wear at all, since a flat surface
# reads about 0.6 of its ball as its own stone. The ball is asked for a
# third of a stone, which the wear's own stencil cap holds down to a few
# cells on a coarse bed.
FLAT_WEIGHT = 0.4
ROUND_WEIGHT = 1.2
WEAR_PACE = 0.25
BALL_RADIUS_SHARE = 0.35

# How far wear carries a stone's surface back, as a share of the stone
# size; the reference hardness is river stone's own, so the step count
# lands on that depth.
WEAR_DEPTH_BASE_SHARE = 0.04
WEAR_DEPTH_WEATHERING_SHARE = 0.08
REFERENCE_HARDNESS = 0.9
MIN_WEAR_STEP_NUM = 4

# Both stones of a pair retreat, so the gap between them comes out twice
# this wide. Emission rounds a cell to whichever side holds it, so a gap
# under about one voxel closes again and the bed reads as one welded
# mass; wear therefore never carries back less than this, however small
# a share of the stone that is.
MIN_WEAR_DEPTH_VOXEL_NUM = 0.6

# The seeds per lattice cell at a full size mix, the paper's Figure 12.
MAX_SEED_NUM = 6

# The flatness dial runs the paper's coefficients from its round rocks
# of Figure 11, all at 1, to its flat plates of Figure 13, 5, 5 and 1.
FLAT_COEFFICIENT = 5.0

# The light surface carving over the worn stones.
DENT_SPACING_SHARE = 1.2
DENT_WIDTH_VOXEL_NUM = 2.0


class _PebbleShape:
    """The pebble bed's dial-to-piece mapping."""

    @staticmethod
    def make_generator(settings, grid, device, box_size, rng: "cr.Rng"):
        """Return the packed pile the bed is carved from.

        Args:
            settings: The validated pebble bed settings.
            grid: The internal grid the pile is grown on.
            device: The torch device the pile lives on.
            box_size: The (x, y, z) bed edge lengths (unit: m).
            rng: The random stream the seed bank draws from.
        """
        return PileGenerator(
            grid,
            str(device),
            rng,
            material_shares=MATERIAL_SHARES,
            stone_size=settings.stone_size,
            size_spread_ratio=0.2 + 0.6 * settings.size_mix,
            seed_num_high=1 + int(round((MAX_SEED_NUM - 1) * settings.size_mix)),
            alpha=1.0 + (FLAT_COEFFICIENT - 1.0) * settings.flatness,
            beta=1.0 + (FLAT_COEFFICIENT - 1.0) * settings.flatness,
            gamma=1.0,
            shape="box",
            box_size=box_size,
            box_occupancy=settings.occupancy,
        )

    @staticmethod
    def make_erosion(settings, rng: "cr.Rng") -> PileErosion:
        """Return the wear that rounds the stones and opens the gaps.

        Args:
            settings: The validated pebble bed settings.
            rng: The random stream the durability sway draws from.
        """
        depth = max(
            (WEAR_DEPTH_BASE_SHARE + WEAR_DEPTH_WEATHERING_SHARE * settings.weathering)
            * settings.stone_size,
            MIN_WEAR_DEPTH_VOXEL_NUM * settings.voxel_size,
        )
        cell_size = settings.voxel_size / settings.wear_subdivision
        step_num = max(
            MIN_WEAR_STEP_NUM,
            int(
                round(
                    depth / cell_size * REFERENCE_HARDNESS / (FLAT_WEIGHT * WEAR_PACE)
                )
            ),
        )
        return PileErosion(
            rng,
            step_num=step_num,
            ball_radius=BALL_RADIUS_SHARE * settings.stone_size,
            wear_pace=WEAR_PACE,
            flat_weight=FLAT_WEIGHT,
            round_weight=ROUND_WEIGHT,
            hardness_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_HARDNESS
            ),
            durability_sway_ratio=0.35,
            durability_feature_size=0.5 * settings.stone_size,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the light surface carving over the worn stones.

        Args:
            settings: The validated pebble bed settings.
            rng: The random stream the carving draws from.
        """
        return DetailErosion(
            rng,
            depth=settings.voxel_size,
            dent_spacing=DENT_SPACING_SHARE * settings.stone_size,
            dent_width=DENT_WIDTH_VOXEL_NUM * settings.voxel_size,
        )


@cr.module(
    name=MODULE,
    version="1.0.0",
    is_deterministic=False,
    settings=cr.settings(
        width=cr.Float(0.3, 6.0, default=1.2, doc="The bed length along x (unit: m)."),
        depth=cr.Float(0.3, 6.0, default=1.2, doc="The bed length along y (unit: m)."),
        thickness=cr.Float(
            0.06,
            1.0,
            default=0.16,
            doc="How deep the bed of stones lies (unit: m); about one "
            "stone deep reads as a bed lying on the ground.",
        ),
        stone_size=cr.Float(
            0.02,
            0.5,
            default=0.12,
            doc="The size one stone grows to (unit: m); keep it about "
            "ten voxels or more, or the gaps between stones cannot be "
            "drawn and the bed reads as one welded mass.",
        ),
        size_mix=cr.Float(
            0.0,
            1.0,
            default=0.6,
            doc="How widely the stone sizes spread, 0 one size to 1 "
            "cobbles down to gravel.",
        ),
        flatness=cr.Float(
            0.0,
            1.0,
            default=0.3,
            doc="How far the stones flatten, 0 round pebbles to 1 flat " "plates.",
        ),
        tone=cr.Float(
            0.0,
            1.0,
            default=0.4,
            doc="Where the bed settles on the river stone run, 0 pale "
            "green-gray to 1 dark slate; the stones spread around it.",
        ),
        weathering=cr.Float(
            0.0,
            1.0,
            default=0.5,
            doc="How far wear carries each stone's surface back inside "
            "its grown shape, 0 freshly packed to 1 heavily rounded.",
        ),
        moving_depth_share=cr.Float(
            0.0,
            1.0,
            default=0.25,
            doc="How deep below the bed surface the stones come away as "
            "moving bodies, as a share of the bed thickness; 0 leaves "
            "the whole bed still scenery.",
        ),
        occupancy=cr.Array3D(
            default=None,
            doc="An optional 3D bool grid stretched over the bed box; "
            "absent cells carve the bed away and the edges round at "
            "grid-cell scale.",
        ),
        voxel_size=cr.VoxelSize(
            default="1cm", min="1cm", max="10cm", doc="The emitted voxel edge."
        ),
        wear_subdivision=cr.Int(
            1,
            6,
            default=2,
            doc="The unitless ratio of internal wear cells per emitted voxel edge.",
        ),
    ),
    dependencies=(f"{mineral.MODULE}@^1",),
    tags=("nature", "rock", "pebble_bed"),
)
def pebble_bed(editor_access, settings):
    """A river bed of rounded stones, loose on top and still beneath.

    Settings:
        width: The bed length along x (unit: m).
        depth: The bed length along y (unit: m).
        thickness: How deep the bed of stones lies (unit: m); about one
            stone deep reads as a bed lying on the ground.
        stone_size: The size one stone grows to (unit: m); keep it
            about ten voxels or more, or the gaps between stones
            cannot be drawn and the bed reads as one welded mass.
        size_mix: How widely the stone sizes spread, 0 one size to 1
            cobbles down to gravel.
        flatness: How far the stones flatten, 0 round pebbles to 1 flat
            plates.
        tone: Where the bed settles on the river stone run, 0 pale
            green-gray to 1 dark slate; the stones spread around it.
        weathering: How far wear carries each stone's surface back
            inside its grown shape, 0 freshly packed to 1 heavily
            rounded.
        moving_depth_share: How deep below the bed surface the stones
            come away as moving bodies, as a share of the bed
            thickness; 0 leaves the whole bed still scenery, and the
            stones past the bed's moving-body budget stay still
            deepest first.
        occupancy: An optional 3D bool grid stretched over the bed box,
            each dimension divided independently; absent cells carve
            the bed away, and the carved edges round and roughen at
            grid-cell scale instead of following the cell walls.
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells per
            emitted voxel edge.
    """
    rng = cr.Rng(settings.seed)
    box_size = (settings.width, settings.depth, settings.thickness)
    grid, device = RockBuild.make_pile_grid(
        settings, "pebble_bed", box_size, settings.stone_size
    )
    volume = _PebbleShape.make_generator(
        settings, grid, device, box_size, rng.split("generator")
    ).generate()
    SegmentTone.make_tones(
        volume,
        TONE_ROWS,
        settings.tone,
        SHADE_SWING_RATIO,
        SHADE_DRIFT_SIZE_NUM * settings.stone_size,
        rng.split("tones"),
    )
    return RockBuild.make_pile_draft(
        editor_access,
        settings,
        kind_name="pebble_bed",
        material_names=MATERIAL_NAMES,
        volume=volume,
        erosion=_PebbleShape.make_erosion(settings, rng.split("erosion")),
        detail=_PebbleShape.make_detail(settings, rng.split("detail")),
        active_degree=settings.moving_depth_share,
    )
