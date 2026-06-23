// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_RECORDER_BASE_H
#define RSC_RECORDER_BASE_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Base class for all recorder types. A recorder is a first-class citizen in RSC.
 * It is used for declaring a sink that's capable of saving the records with processed
 * states of entities in the RISE system.
 */
class RSC_RecorderBase : public RSC_Base {
   public:
    /// Unique recorder name as an id.
    std::string name;

    bool operator==(const RSC_RecorderBase& other) const;

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

#endif  // RSC_RECORDER_BASE_H
