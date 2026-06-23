// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_floor.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"

using namespace std;

namespace rise {

namespace rsc {

void RSC_FloorBase::_validate(RS_RSCValidator &validator) const {
    RSC_EntityBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_FloorBase::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_FloorBase, material_reference);
    });
}

void RSC_FloorBase::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_EntityBase::_read(tree);
    if (tree.isChildExist("material_reference")) {
        material_reference = tree.get<string>("material_reference");
    }
}

void RSC_FloorBase::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_EntityBase::_write(tree);
    tree.add("material_reference", material_reference);
}

bool RSC_FloorFlat::operator==(const RSC_FloorFlat &other) const {
    return RSC_FloorBase::operator==(other) && height == other.height && uv_scale == other.uv_scale;
}

void RSC_FloorFlat::_validate(RS_RSCValidator &validator) const {
    RSC_FloorBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_FloorFlat::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_FloorFlat, uv_scale, 0);
    });
}

void RSC_FloorFlat::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_FloorBase::_read(tree);
    height = tree.get("height", height);
    uv_scale = tree.get("uv_scale", uv_scale);
}

void RSC_FloorFlat::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_FloorBase::_write(tree);
    tree.add("height", height);
    tree.add("uv_scale", uv_scale);
}

bool RSC_FloorArray::operator==(const RSC_FloorArray &other) const {
    return RSC_FloorBase::operator==(other) && x_size == other.x_size && y_size == other.y_size &&
           x_values == other.x_values && y_values == other.y_values &&
           is_x_repeat == other.is_x_repeat && is_y_repeat == other.is_y_repeat &&
           height == other.height;
}

void RSC_FloorArray::_validate(RS_RSCValidator &validator) const {
    RSC_FloorBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_FloorArray::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_FloorArray, x_size, 0);
        RSC_EXPECT_GT(RSC_FloorArray, y_size, 0);
        RSC_EXPECT_GT(RSC_FloorArray, x_values, 0);
        RSC_EXPECT_GT(RSC_FloorArray, y_values, 0);
        if (height.size() != x_values * y_values) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Invalid floor, the size of height array must be equal to x_values * y_values"
            ));
        }
    });
}

void RSC_FloorArray::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_FloorBase::_read(tree);
    x_size = tree.get("x_size", x_size);
    y_size = tree.get("y_size", y_size);
    x_values = tree.get("x_values", x_values);
    y_values = tree.get("y_values", y_values);
    is_x_repeat = tree.get("is_x_repeat", is_x_repeat);
    is_y_repeat = tree.get("is_y_repeat", is_y_repeat);
    RSize num = x_values * y_values;
    height = tree.get<std::vector<float>>("height");
    if (height.size() != num) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Floor elevation height array size is {}, required {} (X_Values = {}, Y_Values = {})",
            height.size(),
            num,
            x_values,
            y_values
        ));
    }
}

void RSC_FloorArray::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_FloorBase::_write(tree);
    tree.add("x_size", x_size);
    tree.add("y_size", y_size);
    tree.add("x_values", x_values);
    tree.add("y_values", y_values);
    tree.add("is_x_repeat", is_x_repeat);
    tree.add("is_y_repeat", is_y_repeat);
    tree.add("height", height);
}

}  // namespace rsc

}  // namespace rise
