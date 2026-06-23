// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/resource/rsc_texture.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/utils/string/rs_string.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_TextureConstant::operator==(const RSC_TextureConstant& other) const {
    return RSC_TextureBase::operator==(other) && value == other.value;
}

void RSC_TextureConstant::_read(const RS_RSCPropertyTreeReader& tree) {
    RSC_ResourceBase::_read(tree);
    value = readColor3f(tree, "value");
}

void RSC_TextureConstant::_write(RS_RSCPropertyTreeWriter& tree) const {
    RSC_ResourceBase::_write(tree);
    writeColor3f(tree, "value", value);
}

bool RSC_TextureGrid::operator==(const RSC_TextureGrid& other) const {
    return RSC_TextureBase::operator==(other) && fill_color == other.fill_color &&
           line_color == other.line_color && is_meters == other.is_meters &&
           x_spacing == other.x_spacing && y_spacing == other.y_spacing &&
           line_width == other.line_width;
}

void RSC_TextureGrid::_validate(RS_RSCValidator& validator) const {
    RSC_ResourceBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_TextureGrid::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_TextureGrid, x_spacing, 0);
        RSC_EXPECT_GT(RSC_TextureGrid, y_spacing, 0);
        RSC_EXPECT_GE(RSC_TextureGrid, line_width, 0);
    });
}

void RSC_TextureGrid::_read(const RS_RSCPropertyTreeReader& tree) {
    RSC_ResourceBase::_read(tree);
    fill_color = readColor3f(tree, "fill_color");
    line_color = readColor3f(tree, "line_color");
    is_meters = tree.get<bool>("is_meters", is_meters);
    x_spacing = tree.get<RFloat>("x_spacing", x_spacing);
    y_spacing = tree.get<RFloat>("y_spacing", y_spacing);
    line_width = tree.get<RFloat>("line_width", line_width);
}

void RSC_TextureGrid::_write(RS_RSCPropertyTreeWriter& tree) const {
    RSC_ResourceBase::_write(tree);
    writeColor3f(tree, "fill_color", fill_color);
    writeColor3f(tree, "line_color", line_color);
    tree.add("is_meters", is_meters);
    tree.add("x_spacing", x_spacing);
    tree.add("y_spacing", y_spacing);
    tree.add("line_width", line_width);
}

}  // namespace rsc

}  // namespace rise
