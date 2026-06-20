// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_entity_base.h"

namespace rise {

namespace rsc {

bool RSC_EntityBase::operator==(const RSC_EntityBase& other) const {
    return RSC_Base::operator==(other) && name == other.name;
}

void RSC_EntityBase::_validate() const { RSC_EXPECT_NOT_EMPTY(RSC_EntityBase, name); }

void RSC_EntityBase::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    name = rsc_tree.get<std::string>("Name", name);
}

}  // namespace rsc

}  // namespace rise
