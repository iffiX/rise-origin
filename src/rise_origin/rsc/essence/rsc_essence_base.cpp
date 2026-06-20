// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/essence/rsc_essence_base.h"

namespace rise {

namespace rsc {

bool RSC_EssenceBase::operator==(const RSC_EssenceBase& other) const {
    return RSC_Base::operator==(other) && name == other.name;
}

void RSC_EssenceBase::_validate() const { RSC_EXPECT_NOT_EMPTY(RSC_EssenceBase, name); }

void RSC_EssenceBase::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    name = rsc_tree.get<std::string>("Name", name);
}

}  // namespace rsc

}  // namespace rise
