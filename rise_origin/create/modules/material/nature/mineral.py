# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The mineral family's material package: rock materials and their optical properties.

Every albedo below is a measured real-world tone, not an invented one:
the sandstone values come from Navajo/Entrada sandstone references
(NPS Zion geology notes, UtahGeology formation pages) converted through
the exact sRGB transfer function, and they sit inside the plausible
rock albedo band. The optical order is contract: structures pick
optical properties by list position, so reordering or removing one is
a new major version of this module; appending is minor. The wear rows
are create-side data the erosion classes gather; the engine never sees
them.
"""

from rise_origin import create as cr
from rise_origin.rsc import essence, resource

MODULE = "rise_origin.material.nature.mineral"

SANDSTONE_SOFT = f"{MODULE}:sandstone_soft"
SANDSTONE_HARD = f"{MODULE}:sandstone_hard"
GRANITE = f"{MODULE}:granite"
LIMESTONE = f"{MODULE}:limestone"
MARBLE = f"{MODULE}:marble"
SLATE = f"{MODULE}:slate"
BASALT = f"{MODULE}:basalt"

# Real anchor tones, linear RGB 0-1, converted from measured sRGB
# references through the exact sRGB transfer (NPS Zion geology notes,
# UtahGeology formation pages). Three tone tiers span real sandstone
# from pale quarry stone to iron-stained Entrada; within a tier the
# tones sit close in value, the way real strata band.
_PALE_BODY_ANCHORS = (
    (0.658, 0.565, 0.434),  # pale cream
    (0.565, 0.413, 0.246),  # pale buff
    (0.527, 0.381, 0.231),  # pale tan
)
_MID_BODY_ANCHORS = (
    (0.503, 0.305, 0.156),  # tan
    (0.552, 0.262, 0.175),  # salmon
    (0.479, 0.314, 0.181),  # warm buff
)
_DARK_BODY_ANCHORS = (
    (0.352, 0.175, 0.084),  # iron tan
    (0.314, 0.127, 0.080),  # dark salmon
    (0.262, 0.072, 0.040),  # red brown
)
_CREVICE_ALBEDO = (0.080, 0.051, 0.037)
_RUST_SPECKLE_ALBEDO = (0.122, 0.045, 0.023)

# How many blended entries one run holds between its real anchors.
_BLEND_RUN_LENGTH = 8

# A hard resistant band is the layer's own tone made a firmer sibling,
# not a separate stripe: kept close in value by a gentle darken and
# nudged toward iron red, so real strata read as laminae, not the
# high-contrast layer-cake look. Raise the darken toward 1 for even
# subtler bands, lower it for bolder ones.
_BAND_DARKEN = 0.78
_BAND_RED_SHIFT = (1.06, 0.9, 0.82)


class _Palette:
    """Blend-run and band-tone construction from real anchor tones."""

    @staticmethod
    def make_run(anchors, length):
        """Return linear-RGB tones interpolated along the anchors.

        Mixing happens in linear RGB, the physically meaningful space
        for reflectance, so every entry lies between two measured
        tones.

        Args:
            anchors: The real anchor tones in run order.
            length: How many entries the run holds.
        """
        run = []
        for index in range(length):
            reach = index / (length - 1) * (len(anchors) - 1)
            low = min(int(reach), len(anchors) - 2)
            mix = reach - low
            first, second = anchors[low], anchors[low + 1]
            run.append(
                tuple(
                    round(first[part] * (1.0 - mix) + second[part] * mix, 3)
                    for part in range(3)
                )
            )
        return tuple(run)

    @staticmethod
    def make_band_anchors(body_anchors):
        """Return band tones close in value to their body anchors.

        Each band tone is its body anchor darkened toward the layer and
        shifted toward iron red, so a hard band sits a small value step
        from the layer it interrupts.

        Args:
            body_anchors: The tier's body anchor tones.
        """
        return tuple(
            tuple(
                round(tone[part] * _BAND_DARKEN * _BAND_RED_SHIFT[part], 3)
                for part in range(3)
            )
            for tone in body_anchors
        )

    @staticmethod
    def make_shades(base, count, swing):
        """Return a short run of slight shade variants of one tone.

        The run spans from ``base`` darkened by ``swing`` to lightened
        by it, so a mineral crystal dithers over a few near-equal tones
        instead of reading as one flat color.

        Args:
            base: The mineral's central tone.
            count: How many shades the run holds.
            swing: How far the shades reach from the base, as a share.
        """
        dark = tuple(base[part] * (1.0 - swing) for part in range(3))
        light = tuple(min(base[part] * (1.0 + swing), 1.0) for part in range(3))
        return _Palette.make_run((dark, light), count)


# Ordered optical albedos; dict order is the contract order. Each tier
# carries a blended body run and a blended band run; the crevice and
# rust speckle tones are substance, never rotated as shades.
SANDSTONE_SOFT_ALBEDOS = {}
SANDSTONE_HARD_ALBEDOS = {}
SANDSTONE_SOFT_TONE_MIDS = []
SANDSTONE_HARD_TONE_MIDS = []
for _tier, _body_anchors in (
    ("pale", _PALE_BODY_ANCHORS),
    ("mid", _MID_BODY_ANCHORS),
    ("dark", _DARK_BODY_ANCHORS),
):
    _band_anchors = _Palette.make_band_anchors(_body_anchors)
    _body_start = len(SANDSTONE_SOFT_ALBEDOS)
    for _index, _tone in enumerate(_Palette.make_run(_body_anchors, _BLEND_RUN_LENGTH)):
        SANDSTONE_SOFT_ALBEDOS[f"{_tier}_body_{_index}"] = _tone
    SANDSTONE_SOFT_TONE_MIDS.append(
        tuple(range(_body_start, _body_start + _BLEND_RUN_LENGTH))
    )
    _band_start = len(SANDSTONE_HARD_ALBEDOS)
    for _index, _tone in enumerate(_Palette.make_run(_band_anchors, _BLEND_RUN_LENGTH)):
        SANDSTONE_HARD_ALBEDOS[f"{_tier}_band_{_index}"] = _tone
    SANDSTONE_HARD_TONE_MIDS.append(
        tuple(range(_band_start, _band_start + _BLEND_RUN_LENGTH))
    )
SANDSTONE_SOFT_ALBEDOS["crevice"] = _CREVICE_ALBEDO
SANDSTONE_HARD_ALBEDOS["rust_speckle"] = _RUST_SPECKLE_ALBEDO
SANDSTONE_SOFT_TONE_MIDS = tuple(SANDSTONE_SOFT_TONE_MIDS)
SANDSTONE_HARD_TONE_MIDS = tuple(SANDSTONE_HARD_TONE_MIDS)
SANDSTONE_SOFT_CREVICE_MID = len(SANDSTONE_SOFT_ALBEDOS) - 1
SANDSTONE_HARD_SPECKLE_MID = len(SANDSTONE_HARD_ALBEDOS) - 1

# The optical reference lists structures pass, in contract order.
SANDSTONE_SOFT_OPTICAL_REFERENCES = tuple(
    f"{SANDSTONE_SOFT}#optic_{name}" for name in SANDSTONE_SOFT_ALBEDOS
)
SANDSTONE_HARD_OPTICAL_REFERENCES = tuple(
    f"{SANDSTONE_HARD}#optic_{name}" for name in SANDSTONE_HARD_ALBEDOS
)

# Granite grain tones, linear RGB from measured mineral references
# (Digital Colony albedo chart anchor at granite gray 0.35-0.40, with
# geology.com and sandatlas.org mineral colors). Granite is one
# salt-and-pepper material: the grain pattern picks a mineral per
# crystal, and each mineral carries a few slight shades so a crystal
# dithers rather than reading as one flat color. Biotite is the sparse
# black speck and crevice is substance for deep cavities, never a
# grain.
_GRANITE_MINERALS = {
    "plagioclase": (0.610, 0.584, 0.515),
    "orthoclase": (0.644, 0.366, 0.275),
    "quartz": (0.386, 0.371, 0.332),
    "biotite": (0.019, 0.018, 0.015),
}
_GRANITE_SHADE_NUM = 3
_GRANITE_SHADE_SWING = 0.12

GRANITE_ALBEDOS = {}
_granite_groups = {}
for _mineral_name, _mineral_base in _GRANITE_MINERALS.items():
    _shade_num = 2 if _mineral_name == "biotite" else _GRANITE_SHADE_NUM
    _group_start = len(GRANITE_ALBEDOS)
    for _shade_index, _shade in enumerate(
        _Palette.make_shades(_mineral_base, _shade_num, _GRANITE_SHADE_SWING)
    ):
        GRANITE_ALBEDOS[f"{_mineral_name}_{_shade_index}"] = _shade
    _granite_groups[_mineral_name] = tuple(
        range(_group_start, _group_start + _shade_num)
    )
GRANITE_ALBEDOS["crevice"] = (0.036, 0.028, 0.019)
GRANITE_OPTICAL_REFERENCES = tuple(
    f"{GRANITE}#optic_{name}" for name in GRANITE_ALBEDOS
)

# Each mineral is a group of its shade mids; the grain generator picks
# a group per crystal and dithers within it.
GRANITE_PLAGIOCLASE_MIDS = _granite_groups["plagioclase"]
GRANITE_ORTHOCLASE_MIDS = _granite_groups["orthoclase"]
GRANITE_QUARTZ_MIDS = _granite_groups["quartz"]
GRANITE_BIOTITE_MIDS = _granite_groups["biotite"]
GRANITE_CREVICE_MID = len(GRANITE_ALBEDOS) - 1

# Limestone body tones, linear RGB (Indiana Limestone Institute grade
# colors, calibrated). Bedded but subtle: the body is one smooth run
# interpolated warm buff to white to cool gray, so warm buff and gray
# never sit adjacent (white always separates them) and a low tone
# reads buff and white, a high tone gray and white. The bright fossil
# fleck, dark mottle, and karst crevice are substance, never drift.
_LIMESTONE_BODY_ANCHORS = (
    (0.69, 0.58, 0.39),  # cream buff
    (0.82, 0.79, 0.72),  # warm white
    (0.45, 0.45, 0.43),  # gray
)
_LIMESTONE_BODY_LENGTH = 8

LIMESTONE_ALBEDOS = {}
for _index, _tone in enumerate(
    _Palette.make_run(_LIMESTONE_BODY_ANCHORS, _LIMESTONE_BODY_LENGTH)
):
    LIMESTONE_ALBEDOS[f"body_{_index}"] = _tone
LIMESTONE_ALBEDOS["fossil_fleck"] = (0.83, 0.77, 0.64)
LIMESTONE_ALBEDOS["dark_mottle"] = (0.17, 0.13, 0.10)
LIMESTONE_ALBEDOS["crevice"] = (0.05, 0.04, 0.03)
LIMESTONE_OPTICAL_REFERENCES = tuple(
    f"{LIMESTONE}#optic_{name}" for name in LIMESTONE_ALBEDOS
)
LIMESTONE_BODY_MIDS = tuple(range(_LIMESTONE_BODY_LENGTH))
LIMESTONE_FOSSIL_MID = _LIMESTONE_BODY_LENGTH
LIMESTONE_MOTTLE_MID = _LIMESTONE_BODY_LENGTH + 1
LIMESTONE_CREVICE_MID = _LIMESTONE_BODY_LENGTH + 2

# Marble tones, linear RGB: one neutral run from the bright Carrara
# white anchored near the measured physicallybased.info marble albedo
# 0.83 down to a Nero Marquina black. A stone takes one step of the run
# for its whole body, the way a quarry block carries one colour, and
# its veins take steps toward the far end so light marble veins dark
# and black marble veins white.
_MARBLE_TONE_ANCHORS = ((0.79, 0.77, 0.73), (0.26, 0.26, 0.27), (0.03, 0.035, 0.04))
MARBLE_TONE_LENGTH = 7
MARBLE_ALBEDOS = {
    f"tone_{_index}": _tone
    for _index, _tone in enumerate(
        _Palette.make_run(_MARBLE_TONE_ANCHORS, MARBLE_TONE_LENGTH)
    )
}
MARBLE_OPTICAL_REFERENCES = tuple(f"{MARBLE}#optic_{name}" for name in MARBLE_ALBEDOS)
MARBLE_TONE_MIDS = tuple(range(MARBLE_TONE_LENGTH))

# Slate tones, linear RGB calibrated to Welsh quarry face references:
# the light ends are the silvery sun-bleached faces a worked slate
# cliff shows, the dark ends the fresh blue-black faces, one run per
# hue family: blue-gray carbon slate (Welsh Ffestiniog), green
# chlorite slate (Westmorland green), neutral silver-gray, and a
# muted iron rust stain. Each run spans the family's fresh dark core
# to its weathered light face, all one length so a plate swaps family
# without changing its shade step.
_SLATE_FAMILY_ANCHORS = {
    "blue_gray": ((0.0420, 0.0520, 0.0660), (0.3600, 0.3900, 0.4300)),
    "green": ((0.0450, 0.0620, 0.0520), (0.3400, 0.3800, 0.3500)),
    "silver": ((0.0550, 0.0580, 0.0620), (0.4200, 0.4300, 0.4500)),
    "rust": ((0.0900, 0.0400, 0.0180), (0.2500, 0.1400, 0.0700)),
}
_SLATE_FAMILY_LENGTH = 6

SLATE_ALBEDOS = {}
SLATE_FAMILY_MIDS = []
for _family, _anchors in _SLATE_FAMILY_ANCHORS.items():
    _family_start = len(SLATE_ALBEDOS)
    for _index, _tone in enumerate(_Palette.make_run(_anchors, _SLATE_FAMILY_LENGTH)):
        SLATE_ALBEDOS[f"{_family}_{_index}"] = _tone
    SLATE_FAMILY_MIDS.append(
        tuple(range(_family_start, _family_start + _SLATE_FAMILY_LENGTH))
    )
SLATE_OPTICAL_REFERENCES = tuple(f"{SLATE}#optic_{name}" for name in SLATE_ALBEDOS)
SLATE_STAIN_MIDS = SLATE_FAMILY_MIDS.pop()
SLATE_FAMILY_MIDS = tuple(SLATE_FAMILY_MIDS)

# Basalt tones, linear RGB from calibrated references (geologyscience.com
# fresh dark gray to black, Giant's Causeway columns). Basalt is fine
# grained with no visible crystals, so the body is a tight dark run the
# grain dither varies only slightly. The lighter exposed gray, the iron
# rust crust, and lichen are substance for weathered surfaces.
_BASALT_BODY_ANCHORS = (
    (0.0176, 0.0194, 0.0222),  # dark
    (0.0343, 0.0369, 0.0423),  # column gray
)
_BASALT_BODY_LENGTH = 5

# The column tone run spans the full weathered-basalt range the
# Reynisfjara and Giant's Causeway columns show: from pale sun-bleached
# gray faces at the light end to dark fresh fracture at the dark end.
# The anchors are neutral gray with a faint cool cast, chosen so the
# light end reads about display 180 and the dark end about display 60,
# the spread real column groups carry; eight steps keep each step small
# so the one-step dither stays a fine grain.
_BASALT_COLUMN_ANCHORS = (
    (0.0500, 0.0500, 0.0530),  # dark fresh fracture
    (0.7500, 0.7500, 0.770),  # pale sun-bleached gray
)
_BASALT_COLUMN_LENGTH = 8

BASALT_ALBEDOS = {}
for _index, _tone in enumerate(
    _Palette.make_run(_BASALT_BODY_ANCHORS, _BASALT_BODY_LENGTH)
):
    BASALT_ALBEDOS[f"body_{_index}"] = _tone
BASALT_ALBEDOS["exposed_gray"] = (0.0685, 0.0685, 0.0704)
BASALT_ALBEDOS["rust_crust"] = (0.1529, 0.1022, 0.0595)
BASALT_ALBEDOS["lichen"] = (0.2747, 0.2664, 0.2086)
for _index, _tone in enumerate(
    _Palette.make_run(_BASALT_COLUMN_ANCHORS, _BASALT_COLUMN_LENGTH)
):
    BASALT_ALBEDOS[f"column_{_index}"] = _tone
BASALT_OPTICAL_REFERENCES = tuple(f"{BASALT}#optic_{name}" for name in BASALT_ALBEDOS)
BASALT_BODY_MIDS = tuple(range(_BASALT_BODY_LENGTH))
BASALT_EXPOSED_MID = _BASALT_BODY_LENGTH
BASALT_RUST_MID = _BASALT_BODY_LENGTH + 1
BASALT_LICHEN_MID = _BASALT_BODY_LENGTH + 2
BASALT_COLUMN_MIDS = tuple(
    range(_BASALT_BODY_LENGTH + 3, _BASALT_BODY_LENGTH + 3 + _BASALT_COLUMN_LENGTH)
)

# Engine physics per material name (unit: kg/m3).
DENSITIES = {
    "sandstone_soft": 2200.0,
    "sandstone_hard": 2500.0,
    "granite": 2650.0,
    "limestone": 2600.0,
    "marble": 2700.0,
    "slate": 2720.0,
    "basalt": 2950.0,
}

# Wear rows the erosion classes gather, keyed by material name.
WEAR_HARDNESS = {
    "sandstone_soft": 0.45,
    "sandstone_hard": 0.85,
    "granite": 1.0,
    "limestone": 0.5,
    "marble": 0.6,
    "slate": 0.8,
    "basalt": 1.0,
}
WEAR_CAVERNOUS = {
    "sandstone_soft": 1.0,
    "sandstone_hard": 0.3,
    "granite": 0.15,
    "limestone": 1.3,
    "marble": 0.25,
    "slate": 0.2,
    "basalt": 0.1,
}

# Surface roughness per material name; most rock is matte, marble
# keeps a slightly smoother unpolished sheen and slate a faint mica
# satin.
ROUGHNESS = {
    "sandstone_soft": 0.9,
    "sandstone_hard": 0.9,
    "granite": 0.9,
    "limestone": 0.9,
    "marble": 0.6,
    "slate": 0.7,
    "basalt": 0.85,
}


@cr.module(
    name=MODULE,
    version="1.0.0",
    tags=(
        "material",
        "mineral",
        "rock",
        "sandstone",
        "granite",
        "limestone",
        "marble",
        "slate",
        "basalt",
    ),
)
def mineral(editor_access, settings):
    """Rock materials with real-world optical properties.

    Releases ``sandstone_soft`` and ``sandstone_hard`` (layered
    Navajo/Entrada tones), ``granite`` (salt-and-pepper mineral
    grains), ``limestone`` (layered pale body), ``marble`` (bright
    veined body), ``slate`` (dark foliated body), and ``basalt`` (dark
    fine-grained body) as rigid materials. Optical order is the
    module's contract; the shade runs, grain mids, vein mids, and
    substance mids ride the exported constants.
    """
    rscs = []
    for material, albedos in (
        (SANDSTONE_SOFT, SANDSTONE_SOFT_ALBEDOS),
        (SANDSTONE_HARD, SANDSTONE_HARD_ALBEDOS),
        (GRANITE, GRANITE_ALBEDOS),
        (LIMESTONE, LIMESTONE_ALBEDOS),
        (MARBLE, MARBLE_ALBEDOS),
        (SLATE, SLATE_ALBEDOS),
        (BASALT, BASALT_ALBEDOS),
    ):
        name = material.split(":")[1]
        rscs.append(
            resource.TextureConstant(
                f"{material}#rm", value=(ROUGHNESS[name], 0.0, 0.0)
            )
        )
        rscs.append(essence.Material(material))
        rscs.append(
            essence.MaterialPhysicalPropertyRigid(
                f"{material}#rigid",
                material_reference=material,
                density=DENSITIES[name],
            )
        )
        for shade, albedo in albedos.items():
            rscs.append(
                resource.TextureConstant(f"{material}#albedo_{shade}", value=albedo)
            )
            rscs.append(
                essence.MaterialOpticalPropertySimpleMicrofacet(
                    f"{material}#optic_{shade}",
                    material_reference=material,
                    albedo_texture_reference=f"{material}#albedo_{shade}",
                    roughness_metalness_texture_reference=f"{material}#rm",
                )
            )
    return cr.Draft(editor_access, "mineral", rscs=rscs)
