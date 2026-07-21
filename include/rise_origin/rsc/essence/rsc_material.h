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

   protected:
    void _validate(RS_RSCValidator &validator) const override;
    void _read(const RS_RSCFileTreeReader &tree) override;
    void _write(RS_RSCFileTreeWriter &tree) const override;
    void _deserialize(RS_RSCArchive &archive) override;
    void _serialize(RS_RSCArchive &archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_MATERIAL_H
