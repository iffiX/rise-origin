// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/signal/rsc_signal_base.h"

namespace rise {

namespace rsc {

bool RSC_SignalBase::operator==(const RSC_SignalBase& other) const {
    return RSC_Base::operator==(other) && name == other.name &&
           source_controller_name == other.source_controller_name;
}

void RSC_SignalBase::_validate() const {
    RSC_EXPECT_NOT_EMPTY(RSC_SignalBase, name);
    RSC_EXPECT_NOT_EMPTY(RSC_SignalBase, source_controller_name);
}

void RSC_SignalBase::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    name = rsc_tree.get<std::string>("Name", name);
    source_controller_name =
        rsc_tree.get<std::string>("SourceControllerName", source_controller_name);
}

}  // namespace rsc

}  // namespace rise
