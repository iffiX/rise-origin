// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/controller/rsc_controller.h"

#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_validator.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_ControllerCallback::operator==(const RSC_ControllerCallback& other) const {
    return RSC_ControllerBase::operator==(other);
}

void RSC_ControllerCallback::_validate(RS_RSCValidator& validator) const {
    RSC_ControllerBase::_validate(validator);
}

void RSC_ControllerCallback::_read(const RS_RSCFileTreeReader& tree) {
    RSC_ControllerBase::_read(tree);
}

void RSC_ControllerCallback::_write(RS_RSCFileTreeWriter& tree) const {
    RSC_ControllerBase::_write(tree);
}

void RSC_ControllerCallback::_deserialize(RS_RSCArchive& archive) {
    RSC_ControllerBase::_deserialize(archive);
}

void RSC_ControllerCallback::_serialize(RS_RSCArchive& archive) const {
    RSC_ControllerBase::_serialize(archive);
}

}  // namespace rsc

}  // namespace rise
