// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/recorder/rsc_recorder_base.h"

#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_validator.h"

namespace rise {

namespace rsc {

bool RSC_RecorderBase::operator==(const RSC_RecorderBase& other) const {
    return RSC_Base::operator==(other) && name == other.name;
}

void RSC_RecorderBase::_validate(RS_RSCValidator& validator) const {
    validator.submitValidatorIfNotExists("RSC_RecorderBase::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_RecorderBase, name);
    });
}

void RSC_RecorderBase::_read(const RS_RSCFileTreeReader& tree) {
    name = tree.get<std::string>("name", name);
}

void RSC_RecorderBase::_write(RS_RSCFileTreeWriter& tree) const { tree.add("name", name); }

void RSC_RecorderBase::_deserialize(RS_RSCArchive& archive) { archive.deserialize("name", name); }

void RSC_RecorderBase::_serialize(RS_RSCArchive& archive) const { archive.serialize("name", name); }

}  // namespace rsc

}  // namespace rise
