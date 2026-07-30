# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The slate block: a layered block broken into aligned shed plates.

Tilted layers carry the hue families, plate fracture breaks the block
into thin plates along its weak direction and sheds the exposed ones,
so the faces read as stepped shards; every plate takes its own family
shade, loose parts fall away, and dents roughen the rest.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm.sediment import (
    DetailErosion,
    PlateFracture,
    StrataGenerator,
    StrataGeneratorLayer,
)

from ...material.nature import mineral
from .build import RockBuild

MODULE = "rise_origin.nature.rock.slate_block"

# Test-asserted voxel cap at the default dials (design doc budget).
FORMATION_VOXEL_CAP = 900_000

# The one material; volume material id 1.
MATERIAL_NAMES = ("slate",)
SLATE_MATERIAL_ID = 1

# Layer sizing: mean layer thickness and the share of layers per family; the
# blue-gray family dominates the way real slate quarries do.
LAYER_MEAN_THICKNESS = 0.5
FAMILY_WEIGHTS = (0.5, 0.25, 0.25)

# The layers fold gently so the hue bands wave instead of ruling lines.
FOLD_HEIGHT_SHARE = 0.06
FOLD_FEATURE_SHARE = 1.2

# Plate proportions from the thickness dial: a plate grows this many
# thicknesses wide and long, the shard shape of cross-cracked slate.
PLATE_WIDTH_NUM = 3.0
PLATE_LENGTH_NUM = 6.0

# The raggedness dial weathers the rock: it lowers the Weibull hold
# strength from barely-touched toward heavily broken, and the round
# num bounds the failure cascade, which stops on its first quiet
# round.
STRENGTH_BASE = 1.5
STRENGTH_SPAN = 0.5
SHED_ROUND_NUM = 8

# A light pass of dents roughens the shard faces.
DENT_DEPTH_VOXEL_NUM = 1.0
DENT_WIDTH_VOXEL_NUM = 2.2
DENT_SPACING_SHARE = 1.5
DENT_SHARE = 0.5


class _SlateShape:
    """The slate block's dial-to-piece mapping."""

    @staticmethod
    def make_generator(settings, grid, device, rng: "cr.Rng") -> StrataGenerator:
        """Return the tilted layered fill of the block.

        Args:
            settings: The validated slate block settings.
            grid: The internal grid the block fills.
            device: The torch device the volume lives on.
            rng: The random stream the layers draw from.
        """
        layer_slope = StrataGenerator.make_layer_slope(settings.layer_dir)
        reach = (
            settings.height
            + abs(layer_slope[0]) * settings.width
            + abs(layer_slope[1]) * settings.depth
            + 2.0 * FOLD_HEIGHT_SHARE * settings.height
        )
        thicknesses = StrataGenerator.make_layer_thicknesses(
            rng.split("thickness"),
            reach,
            LAYER_MEAN_THICKNESS,
            2.0 * settings.plate_thickness,
        )
        family_rng = rng.split("family")
        layers = []
        for index, thickness in enumerate(thicknesses):
            roll = family_rng.split(str(index)).uniform(0.0, 1.0)
            family = (
                0
                if roll < FAMILY_WEIGHTS[0]
                else (1 if roll < FAMILY_WEIGHTS[0] + FAMILY_WEIGHTS[1] else 2)
            )
            layers.append(
                StrataGeneratorLayer(
                    material_id=SLATE_MATERIAL_ID,
                    thickness=thickness,
                    optical_mids=mineral.SLATE_FAMILY_MIDS[family],
                    tone_bias=1.0 - settings.tone,
                    tone_spread=0.2,
                    warp_size=0.3 * settings.plate_thickness,
                    warp_feature_size=settings.width,
                )
            )
        return StrataGenerator(
            grid,
            str(device),
            rng.split("generator"),
            layers=tuple(layers),
            shape="box",
            box_size=(settings.width, settings.depth, settings.height),
            box_occupancy=settings.occupancy,
            fold_size=FOLD_HEIGHT_SHARE * settings.height,
            fold_feature_size=FOLD_FEATURE_SHARE * settings.width,
            layer_slope=layer_slope,
        )

    @staticmethod
    def make_fracture(settings, rng: "cr.Rng") -> PlateFracture:
        """Return the plate fracture that breaks and sheds the block.

        Args:
            settings: The validated slate block settings.
            rng: The random stream the plates draw from.
        """
        return PlateFracture(
            rng,
            break_dir=settings.break_dir,
            plate_thickness=settings.plate_thickness,
            plate_width=PLATE_WIDTH_NUM * settings.plate_thickness,
            plate_length=PLATE_LENGTH_NUM * settings.plate_thickness,
            hold_strength=STRENGTH_BASE - STRENGTH_SPAN * settings.raggedness,
            shed_round_num=SHED_ROUND_NUM,
            family_runs=mineral.SLATE_FAMILY_MIDS,
            stain_run=mineral.SLATE_STAIN_MIDS,
            stain_share=0.12 * settings.stain,
            tone_bias=1.0 - settings.tone,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the light pass of dents that roughens the shard faces.

        Args:
            settings: The validated slate block settings.
            rng: The random stream the dents draw from.
        """
        return DetailErosion(
            rng,
            depth=DENT_DEPTH_VOXEL_NUM * settings.voxel_size,
            dent_spacing=DENT_SPACING_SHARE * settings.plate_thickness,
            dent_width=DENT_WIDTH_VOXEL_NUM * settings.voxel_size,
            dent_share=DENT_SHARE,
        )


@cr.module(
    name=MODULE,
    version="1.0.0",
    is_deterministic=False,
    settings=cr.settings(
        width=cr.Float(
            0.5, 8.0, default=3.0, doc="The block length along x (unit: m)."
        ),
        depth=cr.Float(
            0.5, 8.0, default=1.5, doc="The block length along y (unit: m)."
        ),
        height=cr.Float(0.5, 12.0, default=2.5, doc="The block height (unit: m)."),
        occupancy=cr.Array3D(
            default=None,
            doc="An optional 3D bool grid stretched over the box; absent "
            "cells carve the block away and the edges round at grid-cell "
            "scale.",
        ),
        break_dir=cr.Vector(
            -8.0,
            8.0,
            default=(1.0, 0.25, 0.4),
            doc="The weak direction the rock splits along; plates lie thin "
            "along it and their long side runs down its slope.",
        ),
        layer_dir=cr.Vector(
            -8.0,
            8.0,
            default=(1.0, 0.3, 0.35),
            doc="The direction the hue layers rise along; the horizontal "
            "part aims the tilt and the vertical part sets the steepness.",
        ),
        plate_thickness=cr.Float(
            0.04, 0.3, default=0.08, doc="How thick the split plates grow (unit: m)."
        ),
        raggedness=cr.Float(
            0.0,
            1.0,
            default=0.5,
            doc="How deeply the surfaces shed plates, 0 a light peel to 1 "
            "heavily broken.",
        ),
        tone=cr.Float(
            0.0,
            1.0,
            default=0.5,
            doc="Where the plate shades center, 0 pale weathered to 1 dark "
            "fresh rock.",
        ),
        stain=cr.Float(
            0.0,
            1.0,
            default=0.4,
            doc="How much iron rust stains the plates, 0 none to 1 heavy.",
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
    tags=("nature", "rock", "slate_block"),
)
def slate_block(editor_access, settings):
    """A slate block of shed plates, fixed scenery or a dynamic boulder.

    Settings:
        width: The block length along x (unit: m).
        depth: The block length along y (unit: m).
        height: The block height (unit: m).
        occupancy: An optional 3D bool grid stretched over the box,
            each dimension divided independently; absent cells carve
            the block away, and the carved edges round and roughen at
            grid-cell scale instead of following the cell walls.
        break_dir: The weak direction the rock splits along; plates lie
            thin along it and their long side runs down its slope.
        layer_dir: The direction the hue layers rise along; its horizontal
            part aims the tilt and its vertical part sets the
            steepness.
        plate_thickness: How thick the split plates grow (unit: m).
        raggedness: How deeply the surfaces shed plates, 0 a light peel
            to 1 heavily broken.
        tone: Where the plate shades center, 0 pale weathered to 1
            dark fresh rock.
        stain: How much iron rust stains the plates, 0 none to 1 heavy.
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
        is_fixed: True stands the block as static scenery; False
            marshals it as one dynamic rigid body.
    """
    rng = cr.Rng(settings.seed)
    grid, device = RockBuild.make_box_grid(
        settings,
        "slate_block",
        (settings.width, settings.depth, settings.height),
        PLATE_WIDTH_NUM * settings.plate_thickness,
    )
    volume = _SlateShape.make_generator(settings, grid, device, rng).generate()
    fracture = _SlateShape.make_fracture(settings, rng.split("fracture"))
    fracture.fracture(volume)
    fracture.make_plate_tones(volume)
    return RockBuild.make_volume_draft(
        editor_access,
        settings,
        kind_name="slate_block",
        material_names=MATERIAL_NAMES,
        volume=volume,
        detail=_SlateShape.make_detail(settings, rng.split("detail")),
        is_fixed=settings.is_fixed,
        is_loose_dropped=True,
    )
