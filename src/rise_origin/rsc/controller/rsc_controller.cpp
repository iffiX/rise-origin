// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/controller/rsc_controller.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_ControllerCallback::operator==(const RSC_ControllerCallback& other) const {
    return RSC_ControllerBase::operator==(other);
}

void RSC_ControllerCallback::_validate() const { RSC_ControllerBase::_validate(); }

void RSC_ControllerCallback::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_ControllerBase::_read(rsc_tree);
}

}  // namespace rsc

}  // namespace rise
