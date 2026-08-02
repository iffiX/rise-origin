// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_MATERIAL_PAIR_PROPERTY_H
#define RSC_MATERIAL_PAIR_PROPERTY_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_essence_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Base class for a material pair property.
 *
 * @details Symmetric: applying to (A, B) is the same as (B, A).
 */
class RSC_MaterialPairPropertyBase : public RSC_EssenceBase {
   public:
    /// Name of the first material.
    std::string first_material_reference;

    /// Name of the second material.
    std::string second_material_reference;

    bool operator==(const RSC_MaterialPairPropertyBase& other) const {
        return RSC_EssenceBase::operator==(other) &&
               first_material_reference == other.first_material_reference &&
               second_material_reference == other.second_material_reference;
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override { (void)validator; }
    void _read(const RS_RSCFileTreeReader& tree) override { (void)tree; }
    void _write(RS_RSCFileTreeWriter& tree) const override { (void)tree; }
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
};

/**
 * @brief Coulomb friction between a material pair.
 */
class RSC_MaterialPairPropertyCoulombFriction : public RSC_MaterialPairPropertyBase {
   public:
    /// Static friction coefficient. Only works when it is voxel against floor,
    /// or when it is rigid against rigid material.
    RFloat u_static = 0;

    /// Dynamic friction coefficient.
    RFloat u_dynamic = 0;

    bool operator==(const RSC_MaterialPairPropertyCoulombFriction& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Restitution between a material pair.
 */
class RSC_MaterialPairPropertyRestitution : public RSC_MaterialPairPropertyBase {
   public:
    /// Coefficient of restitution (0 = perfectly inelastic, 1 = perfectly elastic).
    /// Only works when it is rigid against rigid material.
    RFloat restitution = 0;

    bool operator==(const RSC_MaterialPairPropertyRestitution& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

using RSC_MaterialPairPropertyVariant =
    RS_Variant<RSC_MaterialPairPropertyCoulombFriction, RSC_MaterialPairPropertyRestitution>;

}  // namespace rsc

}  // namespace rise

#endif  // RSC_MATERIAL_PAIR_PROPERTY_H
