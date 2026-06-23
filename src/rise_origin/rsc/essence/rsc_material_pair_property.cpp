// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/essence/rsc_material_pair_property.h"

#include "rise_origin/rsc/rs_rsc_validator.h"

namespace rise {

namespace rsc {

bool RSC_MaterialPairPropertyCoulombFriction::operator==(
    const RSC_MaterialPairPropertyCoulombFriction &other
) const {
    return RSC_MaterialPairPropertyBase::operator==(other) && u_static == other.u_static &&
           u_dynamic == other.u_dynamic;
}

void RSC_MaterialPairPropertyCoulombFriction::_validate(RS_RSCValidator &validator) const {
    RSC_EssenceBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_MaterialPairPropertyCoulombFriction::_validate", this, [this] {
            RSC_EXPECT_GE(RSC_MaterialPairPropertyCoulombFriction, u_static, 0);
            RSC_EXPECT_LE(RSC_MaterialPairPropertyCoulombFriction, u_static, 1);
            RSC_EXPECT_GE(RSC_MaterialPairPropertyCoulombFriction, u_dynamic, 0);
            RSC_EXPECT_LE(RSC_MaterialPairPropertyCoulombFriction, u_dynamic, 1);
            RSC_EXPECT_NOT_EMPTY(RSC_MaterialPairPropertyCoulombFriction, first_material_reference);
            RSC_EXPECT_NOT_EMPTY(
                RSC_MaterialPairPropertyCoulombFriction, second_material_reference
            );
        }
    );
}

void RSC_MaterialPairPropertyCoulombFriction::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EssenceBase::_read(tree);
    first_material_reference = tree.get("first_material_reference", first_material_reference);
    second_material_reference = tree.get("second_material_reference", second_material_reference);
    u_static = tree.get("u_static", u_static);
    u_dynamic = tree.get("u_dynamic", u_dynamic);
}

void RSC_MaterialPairPropertyCoulombFriction::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EssenceBase::_write(tree);
    tree.add("first_material_reference", first_material_reference);
    tree.add("second_material_reference", second_material_reference);
    tree.add("u_static", u_static);
    tree.add("u_dynamic", u_dynamic);
}

bool RSC_MaterialPairPropertyRestitution::operator==(
    const RSC_MaterialPairPropertyRestitution &other
) const {
    return RSC_MaterialPairPropertyBase::operator==(other) && restitution == other.restitution;
}

void RSC_MaterialPairPropertyRestitution::_validate(RS_RSCValidator &validator) const {
    RSC_EssenceBase::_validate(validator);
    validator.submitValidatorIfNotExists(
        "RSC_MaterialPairPropertyRestitution::_validate", this, [this] {
            RSC_EXPECT_GE(RSC_MaterialPairPropertyRestitution, restitution, 0);
            RSC_EXPECT_LE(RSC_MaterialPairPropertyRestitution, restitution, 1);
            RSC_EXPECT_NOT_EMPTY(RSC_MaterialPairPropertyRestitution, first_material_reference);
            RSC_EXPECT_NOT_EMPTY(RSC_MaterialPairPropertyRestitution, second_material_reference);
        }
    );
}

void RSC_MaterialPairPropertyRestitution::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EssenceBase::_read(tree);
    first_material_reference = tree.get("first_material_reference", first_material_reference);
    second_material_reference = tree.get("second_material_reference", second_material_reference);
    restitution = tree.get("restitution", restitution);
}

void RSC_MaterialPairPropertyRestitution::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EssenceBase::_write(tree);
    tree.add("first_material_reference", first_material_reference);
    tree.add("second_material_reference", second_material_reference);
    tree.add("restitution", restitution);
}

}  // namespace rsc

}  // namespace rise
