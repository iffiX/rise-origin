# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Essence characterizations.

An essence describes intrinsic properties of matter and the interactions between
materials. A :class:`Material` is a named handle; *unary* property essences
assign physical and optical behavior to one material, while *pair* property
essences characterize the contact interaction between two materials.
"""

from rise_origin import (
    RSC_EssenceBase,
    RSC_MaterialOpticalPropertyBase,
    RSC_MaterialPairPropertyBase,
    RSC_MaterialPhysicalPropertyBase,
    RSC_MaterialUnaryProperty,
)

from rise_origin.rsc.essence.material import (
    Material,
    MaterialOpticalPropertySimpleMicrofacet,
    MaterialPairPropertyCoulombFriction,
    MaterialPairPropertyRestitution,
    MaterialPhysicalPropertyFixed,
    MaterialPhysicalPropertySoft,
    MaterialPhysicalPropertyRigid,
    MaterialPhysicalPropertyAtlas,
)

#: Abstract base of every essence characterization (carries name).
EssenceBase = RSC_EssenceBase

#: Abstract base of properties that reference a single material.
MaterialUnaryProperty = RSC_MaterialUnaryProperty

#: Abstract base of the physical (dynamics) material properties.
MaterialPhysicalPropertyBase = RSC_MaterialPhysicalPropertyBase

#: Abstract base of the optical (appearance) material properties.
MaterialOpticalPropertyBase = RSC_MaterialOpticalPropertyBase

#: Abstract base of properties characterizing a pair of materials.
MaterialPairPropertyBase = RSC_MaterialPairPropertyBase

__all__ = [
    "EssenceBase",
    "MaterialUnaryProperty",
    "MaterialPhysicalPropertyBase",
    "MaterialOpticalPropertyBase",
    "MaterialPairPropertyBase",
    "Material",
    "MaterialPhysicalPropertyFixed",
    "MaterialPhysicalPropertySoft",
    "MaterialPhysicalPropertyRigid",
    "MaterialPhysicalPropertyAtlas",
    "MaterialOpticalPropertySimpleMicrofacet",
    "MaterialPairPropertyCoulombFriction",
    "MaterialPairPropertyRestitution",
]
