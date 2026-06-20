// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/essence/rsc_material.h"

namespace rise {

namespace rsc {

bool RSC_Material::operator==(const RSC_Material& other) const {
    return RSC_EssenceBase::operator==(other);
}

void RSC_Material::_validate() const { RSC_EssenceBase::_validate(); }

void RSC_Material::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_EssenceBase::_read(rsc_tree);
}

}  // namespace rsc

}  // namespace rise
