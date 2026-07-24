# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The plant family's material package: wood and leaves with their optical properties.

Three materials, every shade and season carried as optical properties
on them. The optical order is contract: structures pick optical
properties by list position (``material_optical_reference_mid``), so
reordering or removing one is
a new major version of this module; appending is minor. Other modules
read the orders and the season subsets from the constants exported here.
"""

from rise_origin import create as cr
from rise_origin.rsc import essence, resource

MODULE = "rise_origin.material.nature.plant"

WOOD = f"{MODULE}:wood"
LEAF = f"{MODULE}:leaf"
LEAF_SOFT = f"{MODULE}:leaf_soft"

# Ordered optical albedos, RGB 0-1; dict order is the contract order.
# bark_crevice and bark_light are appended after 1.0.0, and the birch,
# dead wood, and peel optical properties after 1.1.0, keeping earlier
# mids stable.
WOOD_ALBEDOS = {
    "bark_base": (0.35, 0.27, 0.20),
    "bark_shadow": (0.24, 0.18, 0.13),
    "bark_moss": (0.30, 0.34, 0.18),
    "bark_crevice": (0.16, 0.11, 0.08),
    "bark_light": (0.46, 0.38, 0.30),
    "birch_white": (0.86, 0.84, 0.79),
    "wood_dead": (0.44, 0.42, 0.38),
    "wood_peel": (0.56, 0.47, 0.35),
}
# The needle, blossom, and gold optical properties are appended after
# 1.1.0.
LEAF_ALBEDOS = {
    "spring_light": (0.55, 0.72, 0.35),
    "spring_dark": (0.42, 0.60, 0.28),
    "summer_light": (0.30, 0.52, 0.22),
    "summer_dark": (0.20, 0.38, 0.16),
    "autumn_bright": (0.85, 0.30, 0.12),
    "autumn_deep": (0.65, 0.15, 0.10),
    "autumn_orange": (0.88, 0.50, 0.10),
    "needle_dark": (0.11, 0.28, 0.16),
    "needle_light": (0.18, 0.38, 0.20),
    "blossom_pink": (0.94, 0.72, 0.80),
    "blossom_white": (0.96, 0.90, 0.92),
    "gold_light": (0.90, 0.76, 0.24),
    "gold_deep": (0.78, 0.60, 0.14),
}

# The optical reference lists structures pass, in contract order.
WOOD_OPTICAL_REFERENCES = tuple(f"{WOOD}#optic_{name}" for name in WOOD_ALBEDOS)
LEAF_OPTICAL_REFERENCES = tuple(f"{LEAF}#optic_{name}" for name in LEAF_ALBEDOS)
LEAF_SOFT_OPTICAL_REFERENCES = tuple(
    f"{LEAF_SOFT}#optic_{name}" for name in LEAF_ALBEDOS
)

# Which leaf mids a season dithers over; winter is a bare tree.
SEASON_LEAF_MIDS = {
    "spring": (0, 1),
    "summer": (2, 3),
    "autumn": (4, 5, 6),
    "winter": (),
}

# Mid subsets of the appended optical properties; wood mids index
# WOOD_ALBEDOS.
NEEDLE_MIDS = (7, 8)
BLOSSOM_MIDS = (9, 10)
GOLD_MIDS = (11, 12)
WOOD_BIRCH_MID = 5
WOOD_DEAD_MID = 6
WOOD_PEEL_MID = 7

# Soft leaves stay game-soft so the world step at 10cm holds near 1 ms.
LEAF_SOFT_ELASTIC_MODULUS = 0.4e6
LEAF_SOFT_DENSITY = 500.0


@cr.module(
    name=MODULE,
    version="1.0.0",
    tags=("material", "plant", "wood", "leaf"),
)
def plant(editor_access, settings):
    """Wood and leaf materials with bark and seasonal optical
    properties.

    Releases ``wood`` (fixed) with bark optical properties, ``leaf``
    (fixed) and ``leaf_soft`` (soft) with one optical property per
    season shade. Optical order is
    the module's contract; season subsets ride ``SEASON_LEAF_MIDS``.
    """
    rscs = [
        resource.TextureConstant(f"{WOOD}#rm", value=(0.9, 0.0, 0.0)),
        resource.TextureConstant(f"{LEAF}#rm", value=(0.7, 0.0, 0.0)),
        essence.Material(WOOD),
        essence.MaterialPhysicalPropertyFixed(f"{WOOD}#fixed", material_reference=WOOD),
        essence.Material(LEAF),
        essence.MaterialPhysicalPropertyFixed(f"{LEAF}#fixed", material_reference=LEAF),
        essence.Material(LEAF_SOFT),
        essence.MaterialPhysicalPropertySoft(
            f"{LEAF_SOFT}#soft",
            material_reference=LEAF_SOFT,
            elastic_modulus=LEAF_SOFT_ELASTIC_MODULUS,
            density=LEAF_SOFT_DENSITY,
            poissons_ratio=0.3,
        ),
    ]
    for name, albedo in WOOD_ALBEDOS.items():
        rscs.append(resource.TextureConstant(f"{WOOD}#albedo_{name}", value=albedo))
        rscs.append(
            essence.MaterialOpticalPropertySimpleMicrofacet(
                f"{WOOD}#optic_{name}",
                material_reference=WOOD,
                albedo_texture_reference=f"{WOOD}#albedo_{name}",
                roughness_metalness_texture_reference=f"{WOOD}#rm",
            )
        )
    for name, albedo in LEAF_ALBEDOS.items():
        rscs.append(resource.TextureConstant(f"{LEAF}#albedo_{name}", value=albedo))
        for material in (LEAF, LEAF_SOFT):
            rscs.append(
                essence.MaterialOpticalPropertySimpleMicrofacet(
                    f"{material}#optic_{name}",
                    material_reference=material,
                    albedo_texture_reference=f"{LEAF}#albedo_{name}",
                    roughness_metalness_texture_reference=f"{LEAF}#rm",
                )
            )
    return cr.Draft(editor_access, "plant", rscs=rscs)
