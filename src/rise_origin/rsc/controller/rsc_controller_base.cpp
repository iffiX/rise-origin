// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/controller/rsc_controller_base.h"

namespace rise {

namespace rsc {

bool RSC_ControllerBase::operator==(const RSC_ControllerBase& other) const {
    return RSC_Base::operator==(other) && name == other.name;
}

void RSC_ControllerBase::_validate() const { RSC_EXPECT_NOT_EMPTY(RSC_ControllerBase, name); }

void RSC_ControllerBase::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    name = rsc_tree.get<std::string>("Name", name);
}

}  // namespace rsc

}  // namespace rise
