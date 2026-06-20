// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/recorder/rsc_recorder_base.h"

namespace rise {

namespace rsc {

bool RSC_RecorderBase::operator==(const RSC_RecorderBase& other) const {
    return RSC_Base::operator==(other) && name == other.name;
}

void RSC_RecorderBase::_validate() const { RSC_EXPECT_NOT_EMPTY(RSC_RecorderBase, name); }

void RSC_RecorderBase::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    name = rsc_tree.get<std::string>("Name", name);
}

}  // namespace rsc

}  // namespace rise
