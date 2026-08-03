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

   protected:
    void _validate(RS_RSCValidator& validator) const override { (void)validator; }
    void _read(const RS_RSCFileTreeReader& tree) override { (void)tree; }
    void _write(RS_RSCFileTreeWriter& tree) const override { (void)tree; }
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
};

/*****************************************************************
 * Physical properties
 *****************************************************************/
class RSC_MaterialPhysicalPropertyBase : public RSC_MaterialUnaryProperty {
   public:
    bool operator==(const RSC_MaterialPhysicalPropertyBase& other) const {
        return RSC_MaterialUnaryProperty::operator==(other);
    }

   protected:
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
};

/**
 * @brief Fixed (static) physical property.
 */
class RSC_MaterialPhysicalPropertyFixed : public RSC_MaterialPhysicalPropertyBase {
   public:
    bool operator==(const RSC_MaterialPhysicalPropertyFixed& other) const;

   protected:
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
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

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Rigid-body physical property.
 */
class RSC_MaterialPhysicalPropertyRigid : public RSC_MaterialPhysicalPropertyBase {
   public:
    /// Density of the material (unit: kg/m^3).
    RFloat density = 0;

    bool operator==(const RSC_MaterialPhysicalPropertyRigid& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
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

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/*****************************************************************
 * Optical properties
 *****************************************************************/
class RSC_MaterialOpticalPropertyBase : public RSC_MaterialUnaryProperty {
   public:
    bool operator==(const RSC_MaterialOpticalPropertyBase& other) const {
        return RSC_MaterialUnaryProperty::operator==(other);
    }

   protected:
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
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

    /// Whether surface faces of this material take a smoothed normal instead of the face normal.
    bool is_smooth_surface = false;

    /// Index of refraction (1.0 = vacuum, 1.5 = glass).
    RFloat ior = 1.5;

    /// Emission color. Name of a texture resource.
    std::string emission_texture_reference;

    /// Emission intensity (0 = no emission); emitted radiance is the emission color scaled by this.
    RFloat emission_intensity = 0;

    bool operator==(const RSC_MaterialOpticalPropertySimpleMicrofacet& other) const;

   protected:
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_MATERIAL_UNARY_PROPERTY_H
