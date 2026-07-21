// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/essence/rsc_material.h"

#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_validator.h"

namespace rise {

namespace rsc {

bool RSC_Material::operator==(const RSC_Material& other) const {
    return RSC_EssenceBase::operator==(other);
}

void RSC_Material::_validate(RS_RSCValidator& validator) const {
    RSC_EssenceBase::_validate(validator);
}

void RSC_Material::_read(const RS_RSCFileTreeReader& tree) { RSC_EssenceBase::_read(tree); }

void RSC_Material::_write(RS_RSCFileTreeWriter& tree) const { RSC_EssenceBase::_write(tree); }

void RSC_Material::_deserialize(RS_RSCArchive& archive) { RSC_EssenceBase::_deserialize(archive); }

void RSC_Material::_serialize(RS_RSCArchive& archive) const {
    RSC_EssenceBase::_serialize(archive);
}

}  // namespace rsc

}  // namespace rise
