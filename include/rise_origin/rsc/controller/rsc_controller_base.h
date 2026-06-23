// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_CONTROLLER_BASE_H
#define RSC_CONTROLLER_BASE_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Base class for all controller types. A controller is a first-class citizen in RSC.
 * It can be used for declaring a component that receives records from the RISE system and
 * send signals to the RISE system.
 */
class RSC_ControllerBase : public RSC_Base {
   public:
    /// Unique controller name as an id.
    std::string name;

    bool operator==(const RSC_ControllerBase& other) const;

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

#endif  // RSC_CONTROLLER_BASE_H
