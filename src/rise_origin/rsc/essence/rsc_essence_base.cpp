// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/essence/rsc_essence_base.h"

#include "rise_origin/rsc/rs_rsc_validator.h"

namespace rise {

namespace rsc {

bool RSC_EssenceBase::operator==(const RSC_EssenceBase& other) const {
    return RSC_Base::operator==(other) && name == other.name;
}

void RSC_EssenceBase::_validate(RS_RSCValidator& validator) const {
    validator.submitValidatorIfNotExists("RSC_EssenceBase::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_EssenceBase, name);
    });
}

void RSC_EssenceBase::_read(const RS_RSCPropertyTreeReader& tree) {
    name = tree.get<std::string>("name", name);
}

void RSC_EssenceBase::_write(RS_RSCPropertyTreeWriter& tree) const { tree.add("name", name); }

}  // namespace rsc

}  // namespace rise
