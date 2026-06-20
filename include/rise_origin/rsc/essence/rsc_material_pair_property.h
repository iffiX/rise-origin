// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_MATERIAL_PAIR_PROPERTY_H
#define RSC_MATERIAL_PAIR_PROPERTY_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/data/rs_variant.h"
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

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_EssenceBase::serialize(s);
        s("first_material_reference", first_material_reference);
        s("second_material_reference", second_material_reference);
    }

   protected:
    void _validate() const override {}
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override {}
};

/**
 * @brief Coulomb friction between a material pair.
 */
class RSC_MaterialPairPropertyCoulombFriction : public RSC_MaterialPairPropertyBase {
   public:
    /// Static friction coefficient.
    RFloat u_static = 0;

    /// Dynamic friction coefficient.
    RFloat u_dynamic = 0;

    bool operator==(const RSC_MaterialPairPropertyCoulombFriction& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_MaterialPairPropertyBase::serialize(s);
        s("u_static", u_static);
        s("u_dynamic", u_dynamic);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

/**
 * @brief Restitution between a material pair.
 */
class RSC_MaterialPairPropertyRestitution : public RSC_MaterialPairPropertyBase {
   public:
    /// Coefficient of restitution (0 = perfectly inelastic, 1 = perfectly elastic).
    RFloat restitution = 0;

    bool operator==(const RSC_MaterialPairPropertyRestitution& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_MaterialPairPropertyBase::serialize(s);
        s("restitution", restitution);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

using RSC_MaterialPairPropertyVariant =
    RS_Variant<RSC_MaterialPairPropertyCoulombFriction, RSC_MaterialPairPropertyRestitution>;

}  // namespace rsc

}  // namespace rise

#endif  // RSC_MATERIAL_PAIR_PROPERTY_H
