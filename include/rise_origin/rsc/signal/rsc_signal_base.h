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
    std::string source_controller_reference;

    bool operator==(const RSC_SignalBase& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_SIGNAL_BASE_H
