// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/signal/rsc_signal_base.h"

#include "rise_origin/rsc/rs_rsc_validator.h"

namespace rise {

namespace rsc {

bool RSC_SignalBase::operator==(const RSC_SignalBase& other) const {
    return RSC_Base::operator==(other) && name == other.name &&
           source_controller_name == other.source_controller_name;
}

void RSC_SignalBase::_validate(RS_RSCValidator& validator) const {
    validator.submitValidatorIfNotExists("RSC_SignalBase::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_SignalBase, name);
        RSC_EXPECT_NOT_EMPTY(RSC_SignalBase, source_controller_name);
    });
}

void RSC_SignalBase::_read(const RS_RSCPropertyTreeReader& tree) {
    name = tree.get<std::string>("name", name);
    source_controller_name =
        tree.get<std::string>("source_controller_name", source_controller_name);
}

void RSC_SignalBase::_write(RS_RSCPropertyTreeWriter& tree) const {
    tree.add("name", name);
    tree.add("source_controller_name", source_controller_name);
}

}  // namespace rsc

}  // namespace rise
