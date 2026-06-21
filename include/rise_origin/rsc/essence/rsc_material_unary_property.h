// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_MATERIAL_UNARY_PROPERTY_H
#define RSC_MATERIAL_UNARY_PROPERTY_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_mat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_essence_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Base for any property that targets a single material by name.
 */
class RSC_MaterialUnaryProperty : public RSC_EssenceBase {
   public:
    /// Name of the material this property is associated with.
    std::string material_reference;

    bool operator==(const RSC_MaterialUnaryProperty& other) const {
        return RSC_EssenceBase::operator==(other) && material_reference == other.material_reference;
    }

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_EssenceBase::serialize(s);
        s("material_reference", material_reference);
    }

   protected:
    void _validate() const override {}
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override {}
};

/*****************************************************************
 * Physical properties
 *****************************************************************/
class RSC_MaterialPhysicalPropertyBase : public RSC_MaterialUnaryProperty {
   public:
    bool operator==(const RSC_MaterialPhysicalPropertyBase& other) const {
        return RSC_MaterialUnaryProperty::operator==(other);
    }

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_MaterialUnaryProperty::serialize(s);
    }
};

/**
 * @brief Fixed (static) physical property.
 */
class RSC_MaterialPhysicalPropertyFixed : public RSC_MaterialPhysicalPropertyBase {
   public:
    bool operator==(const RSC_MaterialPhysicalPropertyFixed& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_MaterialPhysicalPropertyBase::serialize(s);
    }

   protected:
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

/**
 * @brief Soft (soft-body) physical property.
 */
class RSC_MaterialPhysicalPropertySoft : public RSC_MaterialPhysicalPropertyBase {
   public:
    /// Elastic modulus of the material (unit: Pa).
    RFloat elastic_modulus = 0;

    /// Maximum failure stress of the material (unit: N/m^2).
    RFloat fail_stress = RS_MAX_FLOAT;

    /// Density of the material (unit: kg/m^3).
    RFloat density = 0;

    /// Poisson's ratio of the material.
    RFloat poissons_ratio = 0;

    /// Min expansion (max contraction) allowed by the material.
    RFloat min_expansion = -1;

    /// Max expansion allowed by the material.
    RFloat max_expansion = 1;

    bool operator==(const RSC_MaterialPhysicalPropertySoft& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_MaterialPhysicalPropertyBase::serialize(s);
        s("elastic_modulus", elastic_modulus);
        s("fail_stress", fail_stress);
        s("density", density);
        s("poissons_ratio", poissons_ratio);
        s("min_expansion", min_expansion);
        s("max_expansion", max_expansion);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

/**
 * @brief Rigid-body physical property.
 */
class RSC_MaterialPhysicalPropertyRigid : public RSC_MaterialPhysicalPropertyBase {
   public:
    /// Density of the material (unit: kg/m^3).
    RFloat density = 0;

    bool operator==(const RSC_MaterialPhysicalPropertyRigid& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_MaterialPhysicalPropertyBase::serialize(s);
        s("density", density);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

/**
 * @brief Atlas (analytic rigid body with prescribed mass/inertia) physical property.
 */
class RSC_MaterialPhysicalPropertyAtlas : public RSC_MaterialPhysicalPropertyBase {
   public:
    /// Total mass of the rigid body (unit: kg).
    RFloat mass = 0;

    /// Center of mass in the rigid-body segment's local frame: origin at the segment's minimum
    /// bounding-box corner, axes the structure frame at identity body orientation (unit: m).
    RVec3rf com_position = RVec3rf::zero();

    /// 3x3 moment of inertia tensor about the center of mass, in body-local axes (unit: kg m^2).
    RMat3rf inertia = RMat3rf::zero();

    bool operator==(const RSC_MaterialPhysicalPropertyAtlas& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_MaterialPhysicalPropertyBase::serialize(s);
        s("mass", mass);
        s("com_position", com_position);
        s("inertia", inertia);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

/*****************************************************************
 * Optical properties
 *****************************************************************/
class RSC_MaterialOpticalPropertyBase : public RSC_MaterialUnaryProperty {
   public:
    bool operator==(const RSC_MaterialOpticalPropertyBase& other) const {
        return RSC_MaterialUnaryProperty::operator==(other);
    }

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_MaterialUnaryProperty::serialize(s);
    }
};

/**
 * @brief Simple microfacet surface BSDF optical property.
 */
class RSC_MaterialOpticalPropertySimpleMicrofacet : public RSC_MaterialOpticalPropertyBase {
   public:
    /// Base color / albedo. Name of a texture resource.
    std::string albedo_texture_reference;

    /// Packed roughness (R channel) and metalness (G channel). Name of a texture resource.
    std::string roughness_metalness_texture_reference;

    /// Whether the material is transparent dielectric.
    bool is_transparent = false;

    /// Index of refraction (1.0 = vacuum, 1.5 = glass).
    RFloat ior = 1.5;

    /// Emission color. Name of a texture resource.
    std::string emission_texture_reference;

    /// Emission intensity (0 = no emission); emitted radiance is the emission color scaled by this.
    RFloat emission_intensity = 0;

    bool operator==(const RSC_MaterialOpticalPropertySimpleMicrofacet& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_MaterialOpticalPropertyBase::serialize(s);
        s("albedo_texture_reference", albedo_texture_reference);
        s("roughness_metalness_texture_reference", roughness_metalness_texture_reference);
        s("is_transparent", is_transparent);
        s("ior", ior);
        s("emission_texture_reference", emission_texture_reference);
        s("emission_intensity", emission_intensity);
    }

   protected:
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_MATERIAL_UNARY_PROPERTY_H
