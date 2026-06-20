// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/resource/rsc_resource_base.h"

namespace rise {

namespace rsc {

bool RSC_ResourceBase::operator==(const RSC_ResourceBase& other) const {
    return RSC_Base::operator==(other) && name == other.name;
}

void RSC_ResourceBase::_validate() const { RSC_EXPECT_NOT_EMPTY(RSC_ResourceBase, name); }

void RSC_ResourceBase::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    name = rsc_tree.get<std::string>("Name", name);
}

}  // namespace rsc

}  // namespace rise
