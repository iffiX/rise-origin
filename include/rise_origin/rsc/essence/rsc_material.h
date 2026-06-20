// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_MATERIAL_H
#define RSC_MATERIAL_H

#include "rise_origin/rsc/rsc_base.h"
#include "rsc_essence_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Declaration of a material.
 */
class RSC_Material : public RSC_EssenceBase {
   public:
    bool operator==(const RSC_Material &other) const;

    template <typename Serializer>
    void serialize(Serializer &s) {
        RSC_EssenceBase::serialize(s);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree &rsc_tree) override;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_MATERIAL_H
