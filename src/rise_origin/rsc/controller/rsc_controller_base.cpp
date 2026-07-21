// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/controller/rsc_controller_base.h"

#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_validator.h"

namespace rise {

namespace rsc {

bool RSC_ControllerBase::operator==(const RSC_ControllerBase& other) const {
    return RSC_Base::operator==(other) && name == other.name;
}

void RSC_ControllerBase::_validate(RS_RSCValidator& validator) const {
    validator.submitValidatorIfNotExists("RSC_ControllerBase::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_ControllerBase, name);
    });
}

void RSC_ControllerBase::_read(const RS_RSCFileTreeReader& tree) {
    name = tree.get<std::string>("name", name);
}

void RSC_ControllerBase::_write(RS_RSCFileTreeWriter& tree) const { tree.add("name", name); }

void RSC_ControllerBase::_deserialize(RS_RSCArchive& archive) { archive.deserialize("name", name); }

void RSC_ControllerBase::_serialize(RS_RSCArchive& archive) const {
    archive.serialize("name", name);
}

}  // namespace rsc

}  // namespace rise
