// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_ENTITY_BASE_H
#define RSC_ENTITY_BASE_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Base class for every entity characterization (things that occupy the world —
 * cameras, floors, structures, cosmic backgrounds, etc.).
 */
class RSC_EntityBase : public RSC_Base {
   public:
    /// Unique entity name within the world.
    std::string name;

    bool operator==(const RSC_EntityBase& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        s("name", name);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_ENTITY_BASE_H
