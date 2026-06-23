# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Material essences and their properties.

A :class:`Material` is a named handle. Its behavior is assigned by *property*
essences that reference it: a single physical property and a single optical
property per material (unary, referencing one material by name), plus pairwise
properties that characterize the interaction between two materials (friction,
restitution).

Example::

    from rise_origin import rsc

    rsc.essence.Material("rubber")
    rsc.essence.MaterialPhysicalPropertySoft(
        "rubber_phys", material_reference="rubber",
        elastic_modulus=1.0e6, density=1100.0,
    )
    rsc.essence.MaterialPairPropertyCoulombFriction(
        "rubber_ground_mu", first_material_reference="rubber",
        second_material_reference="ground", u_static=0.9, u_dynamic=0.8,
    )
"""

from typing import Optional

from rise_origin import (
    RSC_Material,
    RSC_MaterialOpticalPropertySimpleMicrofacet,
    RSC_MaterialPairPropertyCoulombFriction,
    RSC_MaterialPairPropertyRestitution,
    RSC_MaterialPhysicalPropertyFixed,
    RSC_MaterialPhysicalPropertySoft,
    RSC_MaterialPhysicalPropertyRigid,
    RSC_MaterialPhysicalPropertyAtlas,
)

from rise_origin.rsc._utils import Mat3rf, Vec3rf, _mat3rf

__all__ = [
    "Material",
    "MaterialPhysicalPropertyFixed",
    "MaterialPhysicalPropertySoft",
    "MaterialPhysicalPropertyRigid",
    "MaterialPhysicalPropertyAtlas",
    "MaterialOpticalPropertySimpleMicrofacet",
    "MaterialPairPropertyCoulombFriction",
    "MaterialPairPropertyRestitution",
]


class Material(RSC_Material):
    """A named material handle.

    Properties are assigned by separate property essences that reference this
    material by ``name``.

    Args:
        name: Unique material name.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class MaterialPhysicalPropertyFixed(RSC_MaterialPhysicalPropertyFixed):
    """Marks a material's voxels as immovable (world-anchored).

    Args:
        name: Unique property name.
        material_reference: Name of the material this property applies to.
    """

    def __init__(self, name: str, *, material_reference: str) -> None:
        super().__init__()
        self.name = name
        self.material_reference = material_reference


class MaterialPhysicalPropertySoft(RSC_MaterialPhysicalPropertySoft):
    """Deformable soft-body physics for a material.

    Args:
        name: Unique property name.
        material_reference: Name of the material this property applies to.
        elastic_modulus: Young's modulus in pascals (stiffness).
        fail_stress: Stress in pascals at which links break. ``None`` keeps the
            engine default (effectively unbreakable).
        density: Mass density in kg/m^3.
        poissons_ratio: Poisson's ratio (lateral-to-axial strain coupling).
        min_expansion: Minimum actuation expansion factor (1.0 is rest length).
        max_expansion: Maximum actuation expansion factor (1.0 is rest length).
    """

    def __init__(
        self,
        name: str,
        *,
        material_reference: str,
        elastic_modulus: float = 0.0,
        fail_stress: Optional[float] = None,
        density: float = 0.0,
        poissons_ratio: float = 0.0,
        min_expansion: float = -1.0,
        max_expansion: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.material_reference = material_reference
        self.elastic_modulus = elastic_modulus
        if fail_stress is not None:
            self.fail_stress = fail_stress
        self.density = density
        self.poissons_ratio = poissons_ratio
        self.min_expansion = min_expansion
        self.max_expansion = max_expansion


class MaterialPhysicalPropertyRigid(RSC_MaterialPhysicalPropertyRigid):
    """Rigid-body physics for a material (voxels fused into a rigid segment).

    Args:
        name: Unique property name.
        material_reference: Name of the material this property applies to.
        density: Mass density in kg/m^3.
    """

    def __init__(
        self,
        name: str,
        *,
        material_reference: str,
        density: float = 0.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.material_reference = material_reference
        self.density = density


class MaterialPhysicalPropertyAtlas(RSC_MaterialPhysicalPropertyAtlas):
    """Rigid-body physics with prescribed mass, center of mass, and inertia.

    Overrides the mass, center of mass, and inertia the engine would otherwise
    derive by summing per-voxel density, so a body keeps its true inertials at
    any voxel resolution. Use it for imported bodies whose inertials are known.

    Args:
        name: Unique property name.
        material_reference: Name of the material this property applies to.
        mass: Total mass in kg.
        com_position: Center of mass in the rigid segment's local frame, with
            origin at the segment's minimum bounding-box corner, in meters.
        inertia: 3x3 moment of inertia tensor about the center of mass, in
            body-local axes (kg m^2); a 3x3 or flat nine-element sequence, or an
            ``RMat3rf``.
    """

    def __init__(
        self,
        name: str,
        *,
        material_reference: str,
        mass: float,
        com_position: Vec3rf,
        inertia: Mat3rf,
    ) -> None:
        super().__init__()
        self.name = name
        self.material_reference = material_reference
        self.mass = mass
        self.com_position = com_position
        self.inertia = _mat3rf(inertia)


class MaterialOpticalPropertySimpleMicrofacet(
    RSC_MaterialOpticalPropertySimpleMicrofacet
):
    """Microfacet optical appearance for a material.

    Each texture reference names a resource (e.g. an ``rsc.resource`` texture).
    An empty reference uses the channel's neutral default.

    Args:
        name: Unique property name.
        material_reference: Name of the material this property applies to.
        albedo_texture_reference: Texture resource for base color.
        roughness_metalness_texture_reference: Texture resource packing
            roughness in R and metalness in G.
        is_transparent: Whether the surface refracts (uses ``ior``).
        ior: Index of refraction (used when transparent).
        emission_texture_reference: Texture resource for emitted color.
        emission_intensity: Scalar multiplier on the emission color.
    """

    def __init__(
        self,
        name: str,
        *,
        material_reference: str,
        albedo_texture_reference: str = "",
        roughness_metalness_texture_reference: str = "",
        is_transparent: bool = False,
        ior: float = 1.5,
        emission_texture_reference: str = "",
        emission_intensity: float = 0.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.material_reference = material_reference
        self.albedo_texture_reference = albedo_texture_reference
        self.roughness_metalness_texture_reference = (
            roughness_metalness_texture_reference
        )
        self.is_transparent = is_transparent
        self.ior = ior
        self.emission_texture_reference = emission_texture_reference
        self.emission_intensity = emission_intensity


class MaterialPairPropertyCoulombFriction(RSC_MaterialPairPropertyCoulombFriction):
    """Coulomb friction between two materials in contact.

    Args:
        name: Unique property name.
        first_material_reference: Name of the first material in the pair.
        second_material_reference: Name of the second material in the pair.
        u_static: Static friction coefficient.
        u_dynamic: Dynamic (sliding) friction coefficient.
    """

    def __init__(
        self,
        name: str,
        *,
        first_material_reference: str,
        second_material_reference: str,
        u_static: float = 0.0,
        u_dynamic: float = 0.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.first_material_reference = first_material_reference
        self.second_material_reference = second_material_reference
        self.u_static = u_static
        self.u_dynamic = u_dynamic


class MaterialPairPropertyRestitution(RSC_MaterialPairPropertyRestitution):
    """Bounciness between two materials in contact.

    Args:
        name: Unique property name.
        first_material_reference: Name of the first material in the pair.
        second_material_reference: Name of the second material in the pair.
        restitution: Coefficient of restitution (0 = perfectly inelastic,
            1 = perfectly elastic).
    """

    def __init__(
        self,
        name: str,
        *,
        first_material_reference: str,
        second_material_reference: str,
        restitution: float = 0.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.first_material_reference = first_material_reference
        self.second_material_reference = second_material_reference
        self.restitution = restitution
