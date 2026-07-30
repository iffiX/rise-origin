# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The basalt columns: a colonnade of packed gray polygonal columns.

A Voronoi packing extrudes one flat-topped column per cell, tops
stepping along the outline direction; grooves and wavy rings cut the
crevice shadow, each column takes its own weathered gray, and dents
roughen the faces. Fixed scenery.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm.sediment import ColumnGenerator, DetailErosion

from ...material.nature import mineral
from .build import RockBuild

MODULE = "rise_origin.nature.rock.basalt_columns"

# Test-asserted voxel cap at the default dials (design doc budget).
FORMATION_VOXEL_CAP = 600_000

# The one material; volume material id 1.
MATERIAL_NAMES = ("basalt",)
BASALT_MATERIAL_ID = 1

# The columns hash their base tone over the wide weathered-gray run;
# the tone dial runs 0 pale weathered gray to 1 dark fresh basalt, the
# swing keeps the dial ends clearly apart, and the fine dither moves a
# face by about one run step around the column's own mean.
COLUMN_TONE_MIDS = mineral.BASALT_COLUMN_MIDS
TONE_SWING_RATIO = 0.5
DITHER_VOXEL_NUM = 1.5

# A column under this many voxels of cross-section is pruned whole.
MIN_COLUMN_VOXEL_NUM = 50

# How far a column top steps below the trend at most, a share of each
# column's own trend height driven by the unevenness dial; half of it
# is a coherent terrace sway spanning a few columns, half the column's
# own randomness. The top faces then roughen by a couple voxels.
TOP_UNEVEN_SHARE = 0.5
TOP_SWAY_FEATURE_COLUMN_NUM = 3.0
TOP_ROUGH_VOXEL_NUM = 1.5

# The rings segment a column along its length about this many widths
# apart, the fracture rhythm real columns carry, and each wanders up
# and down by this share of the spacing so it reads as a wavy crack.
RING_WIDTH_NUM = 0.8
RING_WAVE_SPACING_SHARE = 0.5

# The bend dial leans the column tops this share of the height, and the
# grid carries this much extra room around the footprint so the leaning
# tops are never clipped.
BEND_MAX_SHARE = 0.15
BEND_MARGIN_SHARE = 0.2

# A light pass of dents roughens the faces: one voxel deep, a couple
# voxels wide, spaced a share of the column width apart.
DENT_DEPTH_VOXEL_NUM = 1.0
DENT_WIDTH_VOXEL_NUM = 2.2
DENT_SPACING_SHARE = 0.4
DENT_SHARE = 0.5


class _ColumnsShape:
    """The basalt columns' dial-to-piece mapping."""

    @staticmethod
    def make_generator(settings, grid, device, rng: "cr.Rng") -> ColumnGenerator:
        """Return the column packing for the box.

        Args:
            settings: The validated basalt columns settings.
            grid: The internal grid the columns fill.
            device: The torch device the volume lives on.
            rng: The random stream the packing draws from.
        """
        slope = _ColumnsShape.make_top_slope(settings.top_outline_dir)
        grid_height = _ColumnsShape.make_grid_height(settings, slope)
        return ColumnGenerator(
            grid,
            str(device),
            rng,
            material_id=BASALT_MATERIAL_ID,
            body_mids=COLUMN_TONE_MIDS,
            box_size=(settings.width, settings.depth, grid_height),
            column_width=settings.column_width,
            top_height=settings.height,
            top_slope=slope,
            top_sway_ratio=TOP_UNEVEN_SHARE * settings.top_unevenness,
            top_sway_feature_size=TOP_SWAY_FEATURE_COLUMN_NUM * settings.column_width,
            top_rough_size=TOP_ROUGH_VOXEL_NUM * settings.voxel_size,
            min_column_area=MIN_COLUMN_VOXEL_NUM * settings.voxel_size**2,
            ring_spacing=RING_WIDTH_NUM * settings.column_width,
            ring_wave_size=RING_WAVE_SPACING_SHARE
            * RING_WIDTH_NUM
            * settings.column_width,
            bend_amount=BEND_MAX_SHARE * settings.bend * settings.height,
            tone_bias=1.0 - settings.tone,
            tone_swing_ratio=TONE_SWING_RATIO,
            dither_feature_size=DITHER_VOXEL_NUM * settings.voxel_size,
        )

    @staticmethod
    def make_top_slope(top_outline_dir):
        """Return the (x, y) top-plane rise per meter from the center.

        A ray shot from the center along the direction traces the top
        outline: moving along the horizontal part of the direction, the
        top rises by the vertical part, so the plane's per-meter rise is
        the vertical part spread over the horizontal reach. A direction
        with no horizontal part leaves the tops flat.

        Args:
            top_outline_dir: The (x, y, z) direction the tops rise along.
        """
        reach_x, reach_y, rise = top_outline_dir
        reach_length_sq = reach_x * reach_x + reach_y * reach_y
        if reach_length_sq < 1.0e-9:
            return (0.0, 0.0)
        return (rise * reach_x / reach_length_sq, rise * reach_y / reach_length_sq)

    @staticmethod
    def make_grid_height(settings, top_slope):
        """Return the z the grid must hold for the tilted tops.

        The tops rise off the center height by the slope over the
        footprint half spans, so the grid holds the center height plus
        that rise.

        Args:
            settings: The validated basalt columns settings.
            top_slope: The (x, y) top-plane rise per meter.
        """
        rise = (
            abs(top_slope[0]) * settings.width / 2.0
            + abs(top_slope[1]) * settings.depth / 2.0
        )
        return settings.height + rise

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the light pass of dents that roughens the column faces.

        Args:
            settings: The validated basalt columns settings.
            rng: The random stream the dents draw from.
        """
        return DetailErosion(
            rng,
            depth=DENT_DEPTH_VOXEL_NUM * settings.voxel_size,
            dent_spacing=DENT_SPACING_SHARE * settings.column_width,
            dent_width=DENT_WIDTH_VOXEL_NUM * settings.voxel_size,
            dent_share=DENT_SHARE,
        )


@cr.module(
    name=MODULE,
    version="1.0.0",
    is_deterministic=False,
    settings=cr.settings(
        width=cr.Float(
            0.5, 8.0, default=2.0, doc="The footprint length along x (unit: m)."
        ),
        depth=cr.Float(
            0.5, 8.0, default=1.5, doc="The footprint length along y (unit: m)."
        ),
        height=cr.Float(
            0.5,
            12.0,
            default=3.0,
            doc="The column-top height at the footprint center (unit: m).",
        ),
        column_width=cr.Float(
            0.1, 0.5, default=0.2, doc="How wide the columns grow (unit: m)."
        ),
        tone=cr.Float(
            0.0,
            1.0,
            default=0.5,
            doc="Where the column grays center, 0 pale weathered gray to 1 "
            "dark fresh basalt.",
        ),
        bend=cr.Float(
            0.0,
            1.0,
            default=0.1,
            doc="How far the column tops lean, 0 straight to 1 a strong curve.",
        ),
        top_unevenness=cr.Float(
            0.0,
            1.0,
            default=0.4,
            doc="How far the column tops step below the trend, 0 an even "
            "terrace to 1 deep steps.",
        ),
        top_outline_dir=cr.Vector(
            -8.0,
            8.0,
            default=(1.0, 0.35, 0.5),
            doc="The direction the tops rise along; the horizontal part aims "
            "the rise and the vertical part sets the steepness.",
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
    ),
    dependencies=(f"{mineral.MODULE}@^1",),
    tags=("nature", "rock", "basalt_columns"),
)
def basalt_columns(editor_access, settings):
    """A basalt colonnade, one fixed rock formation.

    Settings:
        width: The footprint length along x (unit: m).
        depth: The footprint length along y (unit: m).
        height: The column-top height at the footprint center (unit: m).
        column_width: How wide the columns grow (unit: m).
        tone: Where the column grays center, 0 pale weathered gray to
            1 dark fresh basalt.
        bend: How far the column tops lean, 0 straight to 1 a strong
            curve.
        top_unevenness: How far the column tops step below the trend,
            0 an even terrace to 1 deep steps.
        top_outline_dir: The direction the tops rise along; a ray from
            the center along it traces the top outline, so its
            horizontal part aims the rise and its vertical part sets the
            steepness. A straight-up direction leaves the tops flat.
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
    """
    rng = cr.Rng(settings.seed)
    margin = BEND_MARGIN_SHARE * settings.height
    slope = _ColumnsShape.make_top_slope(settings.top_outline_dir)
    grid_height = _ColumnsShape.make_grid_height(settings, slope)
    grid, device = RockBuild.make_box_grid(
        settings,
        "basalt_columns",
        (settings.width + 2.0 * margin, settings.depth + 2.0 * margin, grid_height),
        settings.column_width,
    )
    volume = _ColumnsShape.make_generator(
        settings, grid, device, rng.split("generator")
    ).generate()
    return RockBuild.make_volume_draft(
        editor_access,
        settings,
        kind_name="basalt_columns",
        material_names=MATERIAL_NAMES,
        volume=volume,
        detail=_ColumnsShape.make_detail(settings, rng.split("detail")),
        is_fixed=True,
    )
