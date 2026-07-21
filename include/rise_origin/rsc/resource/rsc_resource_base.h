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

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_RESOURCE_BASE_H
