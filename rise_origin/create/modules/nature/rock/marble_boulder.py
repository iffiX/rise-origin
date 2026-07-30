# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The marble boulder: bright body threaded by branching veins.

Marble is one recrystallized carbonate whose impurities fold into
winding veins: vein generation lays a bright clouded body and threads
it with a branching vein network whose centers darken into cores, plus
a few bold veins. Ball erosion rounds it gently, since marble takes a
smooth tumbled shape, and its lower roughness gives the faint sheen.
"""

from rise_origin import create as cr
from rise_origin.create.algorithm.sediment import (
    BallErosion,
    DetailErosion,
    VeinGenerator,
)

from ...material.nature import mineral
from .build import RockBuild

MODULE = "rise_origin.nature.rock.marble_boulder"

# Test-asserted voxel cap at the default dials (design doc budget).
STONE_VOXEL_CAP = 120_000

# The one material; volume material id 1.
MATERIAL_NAMES = ("marble",)
MARBLE_MATERIAL_ID = 1

# The vein network cells sit this far apart, as a share of the stone
# size; wide cells keep the network sparse so its veins can stay near
# one voxel thin and still read as a web rather than filling the stone.
VEIN_SIZE_SHARE = 0.35

# This share of the network walls stays a vein; the rest drop out, so
# the veins read as partial paths instead of a complete cell division.
# The bold black veins break harder, so they read as a few sweeping
# strokes rather than an outline.
VEIN_PATH_SHARE = 0.55
BOLD_VEIN_PATH_SHARE = 0.35

# The network veins step this far along the tone run away from the
# body, so a white stone veins dark and a black one veins light; the
# bold veins carry the whole way to the far end, and the short step
# keeps the two apart wherever the body sits.
VEIN_TONE_STEP_NUM = 2

# Marble tumbles to a smooth shape, so it rounds gently.
CURVATURE_WEIGHT = 1.2
WEAR_PACE = 0.3
WEAR_DEPTH_BASE_SHARE = 0.03
WEAR_DEPTH_WEATHERING_SHARE = 0.2
REFERENCE_HARDNESS = 0.6
MIN_WEAR_STEP_NUM = 6


class _MarbleShape:
    """The marble boulder's dial-to-piece mapping."""

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
        """Return the gentle wear that rounds the marble.

        Args:
            settings: The validated marble boulder settings.
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
            ball_radius=0.13 * settings.size,
            wear_pace=WEAR_PACE,
            flat_weight=1.0,
            round_weight=CURVATURE_WEIGHT,
            pit_weight=0.0,
            hardness_rows=RockBuild.make_wear_rows(
                MATERIAL_NAMES, mineral.WEAR_HARDNESS
            ),
            durability_sway_ratio=0.35,
            durability_feature_size=0.3 * settings.size,
        )

    @staticmethod
    def make_detail(settings, rng: "cr.Rng") -> DetailErosion:
        """Return the light surface carving over the worn shape.

        Args:
            settings: The validated marble boulder settings.
            rng: The random stream the carving draws from.
        """
        return DetailErosion(
            rng,
            depth=settings.voxel_size,
            dent_spacing=0.1,
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
            default=0.5,
            doc="Trades a fresh block at 0 for a tumbled shape at 1.",
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
        tone=cr.Float(
            0.0,
            1.0,
            default=0.0,
            doc="The one tone the whole stone carries, 0 white Carrara to "
            "1 black Marquina; the veins take the far end of the run.",
        ),
        wear_subdivision=cr.Int(
            1,
            10,
            default=4,
            doc="The unitless ratio of internal wear cells per emitted voxel edge.",
        ),
    ),
    dependencies=(f"{mineral.MODULE}@^1",),
    tags=("nature", "rock", "marble_boulder"),
)
def marble_boulder(editor_access, settings):
    """A marble boulder, one dynamic rigid body.

    Settings:
        size: The target diameter (unit: m).
        weathering: Trades a fresh block at 0 for a tumbled shape at 1.
        flatness: How much the stone flattens along its short side, 0
            to 1; picks the Zingg class with elongation.
        elongation: How much the stone stretches along its long side,
            0 to 1.
        vein_thickness: How bold the veins are, 0 hairline feathery to
            1 broad and dominant.
        vein_winding: How far the veins wander, 0 straight to 1 sinuous.
        voxel_size: The emitted voxel edge (unit: m).
        tone: The one tone the whole stone carries, 0 white Carrara to
            1 black Marquina; the veins take the far end of the run.
        wear_subdivision: The unitless ratio of internal wear cells
            per emitted voxel edge.
    """
    rng = cr.Rng(settings.seed)
    grid, device, box_size = RockBuild.make_boulder_grid(
        settings,
        "marble_boulder",
        settings.size,
        settings.flatness,
        settings.elongation,
    )
    body_mid, vein_mids = _MarbleShape.make_tone_mids(settings.tone)
    volume = VeinGenerator(
        grid,
        str(device),
        rng.split("generator"),
        material_id=MARBLE_MATERIAL_ID,
        body_mid=body_mid,
        vein_mids=vein_mids,
        vein_feature_size=VEIN_SIZE_SHARE * settings.size,
        vein_share=0.035 + 0.03 * settings.vein_thickness,
        bold_vein_share=0.03 + 0.03 * settings.vein_thickness,
        vein_path_share=VEIN_PATH_SHARE,
        bold_vein_path_share=BOLD_VEIN_PATH_SHARE,
        winding=settings.vein_winding,
        shape="box",
        box_size=box_size,
    ).generate()
    return RockBuild.make_boulder_draft(
        editor_access,
        settings,
        kind_name="marble_boulder",
        material_names=MATERIAL_NAMES,
        volume=volume,
        rng=rng,
        size=settings.size,
        flatness=settings.flatness,
        elongation=settings.elongation,
        erosion=_MarbleShape.make_erosion(settings, rng.split("erosion")),
        detail=_MarbleShape.make_detail(settings, rng.split("detail")),
    )
