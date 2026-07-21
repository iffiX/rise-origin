// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_CONTROLLER_H
#define RSC_CONTROLLER_H

#include "rise_origin/rsc/rsc_base.h"
#include "rsc_controller_base.h"

namespace rise {

namespace rsc {

/**
 * @brief A callback controller: a named I/O container currently only gets passed
 * through the callback interface and nothing more.
 */
class RSC_ControllerCallback : public RSC_ControllerBase {
   public:
    bool operator==(const RSC_ControllerCallback& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_CONTROLLER_H
