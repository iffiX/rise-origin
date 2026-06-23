// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_RESOURCE_BASE_H
#define RSC_RESOURCE_BASE_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Base class for all resource types. A resource is a first-class citizen in RSC.
 * It is used for declaring assets that are referenced by other entities, such as textures,
 * etc.
 */
class RSC_ResourceBase : public RSC_Base {
   public:
    /// Unique resource name as an id.
    std::string name;

    bool operator==(const RSC_ResourceBase& other) const;

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

#endif  // RSC_RESOURCE_BASE_H
