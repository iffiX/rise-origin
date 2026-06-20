# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Material catalogue, render-lobe helpers, and the global segment-id map.

Every distinct material gets its own segment id because the engine validator
enforces "a segment holds at most one material" (see
``RSC_StructureVoxel::_validate``). The :data:`SEG_OF` map is the single
source of truth — stamps reference materials by name and look up the segment id
through it when pasted into a :class:`scene.world.World`.

Each material expands into ``rsc.essence`` records (a material handle, a fixed
physical property, and a microfacet optical property) plus the constant
``rsc.resource`` textures its lobe needs, mirroring what
``examples/render/materials.py`` does. The three lobes (opaque, transparent,
emissive) are produced by the :func:`_opaque` / :func:`_transparent` /
:func:`_emissive` helpers.
"""

from __future__ import annotations

from rise_origin import rsc

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

# ============================================================================
# Render-lobe helpers
# ============================================================================


def _fixed_material(name, resources, essences) -> None:
    """A material handle plus an immovable (world-anchored) physical property."""
    essences.append(rsc.essence.Material(name))
    essences.append(
        rsc.essence.MaterialPhysicalPropertyFixed(
            f"{name}#phys", material_reference=name
        )
    )


def _opaque(name, resources, essences, *, color, roughness, metalness) -> None:
    """Opaque lobe with the given roughness and metalness."""
    _fixed_material(name, resources, essences)
    resources.append(
        rsc.resource.TextureConstant(f"{name}#albedo", value=_common._rgb(color))
    )
    resources.append(
        rsc.resource.TextureConstant(f"{name}#rm", value=(roughness, metalness, 0.0))
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            f"{name}#optic",
            material_reference=name,
            albedo_texture_reference=f"{name}#albedo",
            roughness_metalness_texture_reference=f"{name}#rm",
        )
    )


def _emissive(name, resources, essences, *, color, intensity) -> None:
    """Emissive lobe; emitted radiance is ``color * intensity``."""
    _fixed_material(name, resources, essences)
    resources.append(
        rsc.resource.TextureConstant(f"{name}#emit", value=_common._rgb(color))
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            f"{name}#optic",
            material_reference=name,
            emission_texture_reference=f"{name}#emit",
            emission_intensity=intensity,
        )
    )


def _transparent(name, resources, essences, *, color, ior, roughness) -> None:
    """Refractive transparent lobe; ``color`` is the per-channel tint."""
    _fixed_material(name, resources, essences)
    resources.append(
        rsc.resource.TextureConstant(f"{name}#albedo", value=_common._rgb(color))
    )
    resources.append(
        rsc.resource.TextureConstant(f"{name}#rm", value=(roughness, 0.0, 0.0))
    )
    essences.append(
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            f"{name}#optic",
            material_reference=name,
            albedo_texture_reference=f"{name}#albedo",
            roughness_metalness_texture_reference=f"{name}#rm",
            is_transparent=True,
            ior=ior,
        )
    )


# ============================================================================
# Material name constants
# ----------------------------------------------------------------------------
# Use these everywhere instead of bare string literals so renames are safe.
# ============================================================================

# Opaque structural --------------------------------------------------------
CONCRETE_WALL = "concrete_wall"
CONCRETE_FLOOR = "concrete_floor"
CONCRETE_CEILING = "concrete_ceiling"
METAL_HOUSING = "metal_housing"  # light steel, glossy
METAL_DARK = "metal_dark"  # gunmetal
METAL_INSULATION = "metal_insulation"  # off-white pipe insulation
PAINTED_YELLOW = "painted_yellow"  # hazard / chevrons
PAINTED_RED = "painted_red"  # alarm trim
PAINTED_CYAN = "painted_cyan"  # accent trim
PLASTIC_WHITE = "plastic_white"  # off-white plastic / mattress
PLASTIC_DARK = "plastic_dark"  # dark plastic monitor bodies
RUBBER_BLACK = "rubber_black"  # cushion, mat
FABRIC_BLANKET = "fabric_blanket"  # bunk blanket
FABRIC_LOUNGE_YELLOW = "fabric_lounge_yellow"  # wool-yellow lobby seating
WOOD_DESK = "wood_desk"  # warm walnut reception desk

# Transparent --------------------------------------------------------------
GLASS_TANK = "glass_tank"
DARK_GLASS = "dark_glass"  # smoked glass partition (main hall <-> bio/server)

# Emissive -----------------------------------------------------------------
EM_WHITE = "em_white"  # ceiling strips (cool white, bright)
EM_WARM_PENDANT = "em_warm_pendant"  # lobby decorative pendant (warm amber)
EM_CYAN = "em_cyan"  # monitor screens, status LEDs
EM_AMBER = "em_amber"  # hazard zebra, LCD displays
EM_GREEN = "em_green"  # exit signs, OK status
EM_RED = "em_red"  # alarms, error LEDs
EM_BIO_TEAL = "em_bio_teal"  # specimen tank glow (teal)
EM_BIO_GREEN = "em_bio_green"  # specimen tank glow (green)
EM_BIO_VIOLET = "em_bio_violet"  # specimen tank glow (violet)
EM_REACTOR = "em_reactor"  # reactor inner shell (very bright)
EM_REACTOR_FLOOR_RING = "em_floor_ring"  # reactor floor halo (dim emissive)
EM_SEPIA = "em_sepia"  # Blade-Runner display-cabinet backlight (warm amber)

# ============================================================================
# Segment-id map (single source of truth)
# ============================================================================

#: Ordered material names — index in this tuple == segment id.
MATERIAL_NAMES = (
    CONCRETE_WALL,
    CONCRETE_FLOOR,
    CONCRETE_CEILING,
    METAL_HOUSING,
    METAL_DARK,
    METAL_INSULATION,
    PAINTED_YELLOW,
    PAINTED_RED,
    PAINTED_CYAN,
    PLASTIC_WHITE,
    PLASTIC_DARK,
    RUBBER_BLACK,
    FABRIC_BLANKET,
    FABRIC_LOUNGE_YELLOW,
    WOOD_DESK,
    GLASS_TANK,
    DARK_GLASS,
    EM_WHITE,
    EM_WARM_PENDANT,
    EM_CYAN,
    EM_AMBER,
    EM_GREEN,
    EM_RED,
    EM_BIO_TEAL,
    EM_BIO_GREEN,
    EM_BIO_VIOLET,
    EM_REACTOR,
    EM_REACTOR_FLOOR_RING,
    EM_SEPIA,
)

#: Material name -> segment id. Stamps reference this when pasting.
SEG_OF: dict = {name: i for i, name in enumerate(MATERIAL_NAMES)}


# ============================================================================
# Build the materials
# ============================================================================


def build_materials(resources: list, essences: list) -> None:
    """Append every material the facility uses, all pinned (``is_fixed``).

    The order matches :data:`MATERIAL_NAMES`; resources hold each material's
    constant textures and essences hold its handle plus physical and optical
    properties.
    """
    # Opaque structural --------------------------------------------------
    _opaque(
        CONCRETE_WALL,
        resources,
        essences,
        color=(48, 52, 60),
        roughness=0.92,
        metalness=0.0,
    )
    _opaque(
        CONCRETE_FLOOR,
        resources,
        essences,
        color=(60, 62, 70),
        roughness=0.45,
        metalness=0.0,
    )
    _opaque(
        CONCRETE_CEILING,
        resources,
        essences,
        color=(36, 38, 44),
        roughness=0.95,
        metalness=0.0,
    )
    _opaque(
        METAL_HOUSING,
        resources,
        essences,
        color=(150, 155, 165),
        roughness=0.35,
        metalness=1.0,
    )
    _opaque(
        METAL_DARK,
        resources,
        essences,
        color=(64, 68, 78),
        roughness=0.55,
        metalness=1.0,
    )
    _opaque(
        METAL_INSULATION,
        resources,
        essences,
        color=(200, 195, 180),
        roughness=0.85,
        metalness=0.0,
    )
    _opaque(
        PAINTED_YELLOW,
        resources,
        essences,
        color=(210, 175, 40),
        roughness=0.85,
        metalness=0.0,
    )
    _opaque(
        PAINTED_RED,
        resources,
        essences,
        color=(195, 55, 55),
        roughness=0.85,
        metalness=0.0,
    )
    _opaque(
        PAINTED_CYAN,
        resources,
        essences,
        color=(60, 160, 195),
        roughness=0.85,
        metalness=0.0,
    )
    _opaque(
        PLASTIC_WHITE,
        resources,
        essences,
        color=(215, 215, 205),
        roughness=0.85,
        metalness=0.0,
    )
    _opaque(
        PLASTIC_DARK,
        resources,
        essences,
        color=(28, 30, 36),
        roughness=0.85,
        metalness=0.0,
    )
    _opaque(
        RUBBER_BLACK,
        resources,
        essences,
        color=(14, 14, 18),
        roughness=0.95,
        metalness=0.0,
    )
    _opaque(
        FABRIC_BLANKET,
        resources,
        essences,
        color=(85, 75, 70),
        roughness=0.95,
        metalness=0.0,
    )
    _opaque(
        FABRIC_LOUNGE_YELLOW,
        resources,
        essences,
        color=(210, 180, 110),
        roughness=0.95,
        metalness=0.0,
    )
    _opaque(
        WOOD_DESK,
        resources,
        essences,
        color=(110, 75, 50),
        roughness=0.60,
        metalness=0.0,
    )

    # Transparent --------------------------------------------------------
    _transparent(
        GLASS_TANK, resources, essences, color=(230, 245, 255), ior=1.5, roughness=0.0
    )
    # Dark smoked glass: tinted partition for the main-hall <-> bio/server
    # walls. Tuned half-less dark (and noticeably more transparent) so the
    # lit lab beyond reads clearly through the glass — the user's
    # reference is closer to a smoked-glass conference-room wall than to
    # an opaque privacy partition.
    _transparent(
        DARK_GLASS, resources, essences, color=(170, 185, 200), ior=1.5, roughness=0.01
    )

    # Emissive -----------------------------------------------------------
    # EM_WHITE intensity bumped from 4.0 -> 7.0 (existing industrial strips
    # read too dim in eye-level renders); shape unchanged.
    _emissive(EM_WHITE, resources, essences, color=(240, 240, 255), intensity=15.0)
    _emissive(
        EM_WARM_PENDANT, resources, essences, color=(255, 220, 180), intensity=5.5
    )
    _emissive(EM_CYAN, resources, essences, color=(90, 220, 255), intensity=3.0)
    _emissive(EM_AMBER, resources, essences, color=(255, 180, 60), intensity=3.5)
    _emissive(EM_GREEN, resources, essences, color=(90, 230, 110), intensity=3.0)
    _emissive(EM_RED, resources, essences, color=(255, 60, 60), intensity=3.0)
    _emissive(EM_SEPIA, resources, essences, color=(255, 180, 90), intensity=2.2)
    _emissive(EM_BIO_TEAL, resources, essences, color=(60, 230, 200), intensity=3.5)
    _emissive(EM_BIO_GREEN, resources, essences, color=(120, 240, 130), intensity=3.5)
    _emissive(EM_BIO_VIOLET, resources, essences, color=(180, 110, 240), intensity=3.5)
    _emissive(EM_REACTOR, resources, essences, color=(200, 240, 255), intensity=100.0)
    _emissive(
        EM_REACTOR_FLOOR_RING, resources, essences, color=(255, 160, 50), intensity=20
    )


# ============================================================================
# Emissive-name set (used to count lights)
# ============================================================================

#: Every name that maps to an emissive material — used to count lights.
EMISSIVE_NAMES = frozenset(
    {
        EM_WHITE,
        EM_WARM_PENDANT,
        EM_CYAN,
        EM_AMBER,
        EM_GREEN,
        EM_RED,
        EM_BIO_TEAL,
        EM_BIO_GREEN,
        EM_BIO_VIOLET,
        EM_REACTOR,
        EM_REACTOR_FLOOR_RING,
        EM_SEPIA,
    }
)
