// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_ESSENCE_BASE_H
#define RSC_ESSENCE_BASE_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Base class for all essence decription types. An essence is a first-class citizen in RSC.
 * It is used for declaring properties of entities, such as what material they are
 * made of, what physical properties they have, etc.
 *
 * @note An essence is a description of an entity, it is not an entity itself.
 */
class RSC_EssenceBase : public RSC_Base {
   public:
    /// Unique essence name as an id.
    std::string name;

    bool operator==(const RSC_EssenceBase& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        s("name", name);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_ESSENCE_BASE_H
