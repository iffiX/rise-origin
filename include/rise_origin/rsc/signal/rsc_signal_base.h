// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_SIGNAL_BASE_H
#define RSC_SIGNAL_BASE_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Base class for all signal types. A signal is a first-class citizen in RSC.
 * It is used for declaring an input/actuation pathway from a controller into the
 * simulation (force application, prescribed-displacement, etc.).
 */
class RSC_SignalBase : public RSC_Base {
   public:
    /// Unique signal name as an id.
    std::string name;

    /// Name of the controller that produces this signal's values.
    std::string source_controller_name;

    bool operator==(const RSC_SignalBase& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        s("name", name);
        s("source_controller_name", source_controller_name);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_SIGNAL_BASE_H
