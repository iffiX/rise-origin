// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_RECORD_BASE_H
#define RSC_RECORD_BASE_H

#include <string>
#include <vector>

#include "rise_origin/rsc/rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Base class for all record types. A record is a first-class citizen in RSC.
 * It is used for declaring how to save and process the states of entities in the RISE system,
 * so that a controller controlled by the user or a recorder for archiving data can access them.
 */
class RSC_RecordBase : public RSC_Base {
   public:
    /// Unique record name as an id.
    std::string name;

    /// Reference to the controller that this record is sent to.
    std::string target_controller_reference;

    /// Reference to the recorder that this record is sent to.
    std::string target_recorder_reference;

    bool operator==(const RSC_RecordBase& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        s("name", name);
        s("target_controller_reference", target_controller_reference);
        s("target_recorder_reference", target_recorder_reference);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_RECORD_BASE_H
