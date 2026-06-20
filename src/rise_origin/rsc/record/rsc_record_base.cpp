// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/record/rsc_record_base.h"

namespace rise {

namespace rsc {

bool RSC_RecordBase::operator==(const RSC_RecordBase& other) const {
    return RSC_Base::operator==(other) && name == other.name &&
           target_controller_reference == other.target_controller_reference &&
           target_recorder_reference == other.target_recorder_reference;
}

void RSC_RecordBase::_validate() const {
    RSC_EXPECT_NOT_EMPTY(RSC_RecordBase, name);
    if (target_controller_reference.empty() && target_recorder_reference.empty()) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Invalid RSC_RecordBase, at least one of target_controller_reference "
            "and target_recorder_reference must not be empty"
        ));
    }
}

void RSC_RecordBase::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    name = rsc_tree.get<std::string>("Name", name);
    target_controller_reference =
        rsc_tree.get<std::string>("TargetControllerReference", target_controller_reference);
    target_recorder_reference =
        rsc_tree.get<std::string>("TargetRecorderReference", target_recorder_reference);
}

}  // namespace rsc

}  // namespace rise
