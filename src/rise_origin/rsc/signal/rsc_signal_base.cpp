// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/signal/rsc_signal_base.h"

#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_validator.h"

namespace rise {

namespace rsc {

bool RSC_SignalBase::operator==(const RSC_SignalBase& other) const {
    return RSC_Base::operator==(other) && name == other.name &&
           source_controller_reference == other.source_controller_reference;
}

void RSC_SignalBase::_validate(RS_RSCValidator& validator) const {
    validator.submitValidatorIfNotExists("RSC_SignalBase::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_SignalBase, name);
        RSC_EXPECT_NOT_EMPTY(RSC_SignalBase, source_controller_reference);
    });
}

void RSC_SignalBase::_read(const RS_RSCFileTreeReader& tree) {
    name = tree.get<std::string>("name", name);
    source_controller_reference =
        tree.get<std::string>("source_controller_reference", source_controller_reference);
}

void RSC_SignalBase::_write(RS_RSCFileTreeWriter& tree) const {
    tree.add("name", name);
    tree.add("source_controller_reference", source_controller_reference);
}

void RSC_SignalBase::_deserialize(RS_RSCArchive& archive) {
    archive.deserialize("name", name);
    archive.deserialize("source_controller_reference", source_controller_reference);
}

void RSC_SignalBase::_serialize(RS_RSCArchive& archive) const {
    archive.serialize("name", name);
    archive.serialize("source_controller_reference", source_controller_reference);
}

}  // namespace rsc

}  // namespace rise
