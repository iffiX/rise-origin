// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/resource/rsc_resource_base.h"

#include "rise_origin/rsc/rs_rsc_validator.h"

namespace rise {

namespace rsc {

bool RSC_ResourceBase::operator==(const RSC_ResourceBase& other) const {
    return RSC_Base::operator==(other) && name == other.name;
}

void RSC_ResourceBase::_validate(RS_RSCValidator& validator) const {
    validator.submitValidatorIfNotExists("RSC_ResourceBase::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_ResourceBase, name);
    });
}

void RSC_ResourceBase::_read(const RS_RSCPropertyTreeReader& tree) {
    name = tree.get<std::string>("name", name);
}

void RSC_ResourceBase::_write(RS_RSCPropertyTreeWriter& tree) const { tree.add("name", name); }

}  // namespace rsc

}  // namespace rise
