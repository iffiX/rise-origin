# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The slate boulder: a sharp cleaved chunk of layered slate.

A fracture fragment cuts the chunk out of tilted slate layers, plate
fracture splits it along the weak direction and sheds a few exposed
plates, so the faces read as stepped cleavage; every plate takes its
own family shade and dents roughen the rest. No wear rounding: the
chunk stays sharp, fresh off the parent rock.
"""

import numpy as np

from rise_origin import create as cr
from rise_origin.create.algorithm.sediment import (
    DetailErosion,
    PlaneFracture,
    PlateFracture,
    StrataGenerator,
    StrataGeneratorLayer,
)

from ...material.nature import mineral
from .build import RockBuild

MODULE = "rise_origin.nature.rock.slate_boulder"

# Test-asserted voxel cap at the default dials (design doc budget).
STONE_VOXEL_CAP = 120_000

# The one material; volume material id 1.
MATERIAL_NAMES = ("slate",)
SLATE_MATERIAL_ID = 1

# Layer sizing as a share of the boulder size, and the share of layers
# per family; the blue-gray family dominates like the block.
LAYER_MEAN_SHARE = 0.4
FAMILY_WEIGHTS = (0.5, 0.25, 0.25)

# The layers fold gently so the hue bands wave instead of ruling lines.
FOLD_HEIGHT_SHARE = 0.08
FOLD_FEATURE_SHARE = 1.2

# Plate proportions from the thickness dial: a plate grows this many
# thicknesses wide and long, the shard shape of cross-cracked slate.
PLATE_WIDTH_NUM = 3.0
PLATE_LENGTH_NUM = 6.0

# The raggedness dial lowers the Weibull hold strength; the boulder
# sheds lightly, a fresh chunk rather than a weathered face.
STRENGTH_BASE = 1.8
STRENGTH_SPAN = 0.6
SHED_ROUND_NUM = 5

# A light pass of dents roughens the cleavage faces.
DENT_DEPTH_VOXEL_NUM = 1.0
DENT_WIDTH_VOXEL_NUM = 2.2
DENT_SPACING_SHARE = 1.5
DENT_SHARE = 0.5


class _BoulderShape:
    """The slate boulder's dial-to-piece mapping."""

    @staticmethod
    def make_generator(settings, grid, device, box_size, rng: "cr.Rng"):
        """Return the tilted layered fill of the chunk's box.

        Args:
            settings: The validated slate boulder settings.
            grid: The internal grid the chunk fills.
            device: The torch device the volume lives on.
            box_size: The (x, y, z) box edge lengths (unit: m).
            rng: The random stream the layers draw from.
        """
        layer_slope = StrataGenerator.make_layer_slope(settings.layer_dir)
        reach = (
            box_size[2]
            + abs(layer_slope[0]) * box_size[0]
            + abs(layer_slope[1]) * box_size[1]
            + 2.0 * FOLD_HEIGHT_SHARE * box_size[2]
        )
        thicknesses = StrataGenerator.make_layer_thicknesses(
            rng.split("thickness"),
            reach,
            LAYER_MEAN_SHARE * settings.size,
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
                    warp_feature_size=box_size[0],
                )
            )
        return StrataGenerator(
            grid,
            str(device),
            rng.split("generator"),
            layers=tuple(layers),
            shape="box",
            box_size=box_size,
            fold_size=FOLD_HEIGHT_SHARE * box_size[2],
            fold_feature_size=FOLD_FEATURE_SHARE * box_size[0],
            layer_slope=layer_slope,
        )

    @staticmethod
    def make_fracture(settings, rng: "cr.Rng") -> PlateFracture:
        """Return the plate fracture that splits and sheds the chunk.

        Args:
            settings: The validated slate boulder settings.
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
        """Return the light pass of dents that roughens the faces.

        Args:
            settings: The validated slate boulder settings.
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
        size=cr.Float(0.2, 2.0, default=0.7, doc="The target diameter (unit: m)."),
        flatness=cr.Float(
            0.0,
            1.0,
            default=0.5,
            doc="How much the chunk flattens along its short side, 0 to 1; "
            "slate cleaves slabby, so the default sits high.",
        ),
        elongation=cr.Float(
            0.0,
            1.0,
            default=0.3,
            doc="How much the chunk stretches along its long side, 0 to 1.",
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
            0.02, 0.2, default=0.05, doc="How thick the split plates grow (unit: m)."
        ),
        raggedness=cr.Float(
            0.0,
            1.0,
            default=0.35,
            doc="How deeply the faces shed plates, 0 a clean cut to 1 "
            "heavily stepped.",
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
            default=0.3,
            doc="How much iron rust stains the plates, 0 none to 1 heavy.",
        ),
        voxel_size=cr.VoxelSize(
            default="2cm", min="1cm", max="5cm", doc="The emitted voxel edge."
        ),
        wear_subdivision=cr.Int(
            1,
            6,
            default=1,
            doc="The unitless ratio of internal wear cells per emitted voxel edge.",
        ),
        is_fixed=cr.Bool(
            default=False,
            doc="True stands the chunk as static scenery; False marshals it "
            "as one dynamic rigid body.",
        ),
    ),
    dependencies=(f"{mineral.MODULE}@^1",),
    tags=("nature", "rock", "slate_boulder"),
)
def slate_boulder(editor_access, settings):
    """A sharp slate chunk, one dynamic rigid body by default.

    Settings:
        size: The target diameter (unit: m).
        flatness: How much the chunk flattens along its short side, 0
            to 1; slate cleaves slabby, so the default sits high.
        elongation: How much the chunk stretches along its long side,
            0 to 1.
        break_dir: The weak direction the rock splits along; plates lie
            thin along it and their long side runs down its slope.
        layer_dir: The direction the hue layers rise along; its
            horizontal part aims the tilt and its vertical part sets
            the steepness.
        plate_thickness: How thick the split plates grow (unit: m).
        raggedness: How deeply the faces shed plates, 0 a clean cut to
            1 heavily stepped.
        tone: Where the plate shades center, 0 pale weathered to 1
            dark fresh rock.
        stain: How much iron rust stains the plates, 0 none to 1 heavy.
        voxel_size: The emitted voxel edge (unit: m).
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
        is_fixed: True stands the chunk as static scenery; False
            marshals it as one dynamic rigid body.
    """
    rng = cr.Rng(settings.seed)
    grid, device, box_size = RockBuild.make_boulder_grid(
        settings,
        "slate_boulder",
        settings.size,
        settings.flatness,
        settings.elongation,
    )
    volume = _BoulderShape.make_generator(
        settings, grid, device, box_size, rng
    ).generate()
    fragment = PlaneFracture.make_fragment(
        rng.split("fragment"), settings.size, settings.flatness, settings.elongation
    )
    PlaneFracture((fragment,), np.zeros((1, 3))).fracture(volume)
    fracture = _BoulderShape.make_fracture(settings, rng.split("fracture"))
    fracture.fracture(volume)
    fracture.make_plate_tones(volume)
    return RockBuild.make_volume_draft(
        editor_access,
        settings,
        kind_name="slate_boulder",
        material_names=MATERIAL_NAMES,
        volume=volume,
        detail=_BoulderShape.make_detail(settings, rng.split("detail")),
        is_fixed=settings.is_fixed,
        is_loose_dropped=True,
    )
