// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/entity/rsc_floor.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/utils/string/rs_string.h"

using namespace std;

namespace rise {

namespace rsc {

void RSC_FloorBase::_validate() const {
    RSC_EntityBase::_validate();
    RSC_EXPECT_NOT_EMPTY(RSC_FloorBase, material_reference);
}

void RSC_FloorBase::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_EntityBase::_read(rsc_tree);
    if (rsc_tree.isChildExist("MaterialReference")) {
        material_reference = rsc_tree.get<string>("MaterialReference");
    }
}

bool RSC_FloorFlat::operator==(const RSC_FloorFlat &other) const {
    return RSC_FloorBase::operator==(other) && height == other.height && uv_scale == other.uv_scale;
}

void RSC_FloorFlat::_validate() const {
    RSC_FloorBase::_validate();
    RSC_EXPECT_GT(RSC_FloorFlat, uv_scale, 0);
}

void RSC_FloorFlat::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_FloorBase::_read(rsc_tree);
    height = rsc_tree.get("Height", height);
    uv_scale = rsc_tree.get("UvScale", uv_scale);
}

bool RSC_FloorArray::operator==(const RSC_FloorArray &other) const {
    return RSC_FloorBase::operator==(other) && x_size == other.x_size && y_size == other.y_size &&
           x_values == other.x_values && y_values == other.y_values &&
           is_x_repeat == other.is_x_repeat && is_y_repeat == other.is_y_repeat &&
           height == other.height;
}

void RSC_FloorArray::_validate() const {
    RSC_FloorBase::_validate();
    RSC_EXPECT_GT(RSC_FloorArray, x_size, 0);
    RSC_EXPECT_GT(RSC_FloorArray, y_size, 0);
    RSC_EXPECT_GT(RSC_FloorArray, x_values, 0);
    RSC_EXPECT_GT(RSC_FloorArray, y_values, 0);
    if (height.size() != x_values * y_values) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Invalid floor, the size of height array must be equal to x_values * y_values"
        ));
    }
}

void RSC_FloorArray::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_FloorBase::_read(rsc_tree);
    x_size = rsc_tree.get("X_Size", x_size);
    y_size = rsc_tree.get("Y_Size", y_size);
    x_values = rsc_tree.get("X_Values", x_values);
    y_values = rsc_tree.get("Y_Values", y_values);
    is_x_repeat = rsc_tree.get("IsXRepeat", is_x_repeat);
    is_y_repeat = rsc_tree.get("IsYRepeat", is_y_repeat);
    RSize num = x_values * y_values;
    auto split_values = split(strip(rsc_tree.get<string>("Height")), ",");
    if (split_values.size() != num) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Floor elevation height array size is {}, required {} (X_Values = {}, Y_Values = {})",
            split_values.size(),
            num,
            x_values,
            y_values
        ));
    }
    for (const auto &h_str : split_values) {
        height.push_back(static_cast<float>(stod(h_str)));
    }
}

}  // namespace rsc

}  // namespace rise
