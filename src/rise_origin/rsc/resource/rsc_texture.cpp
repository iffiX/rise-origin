// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/resource/rsc_texture.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/utils/string/rs_string.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_TextureConstant::operator==(const RSC_TextureConstant& other) const {
    return RSC_TextureBase::operator==(other) && value == other.value;
}

void RSC_TextureConstant::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_ResourceBase::_read(rsc_tree);
    value = readColor3f(rsc_tree.get<string>("Value"));
}

bool RSC_TextureGrid::operator==(const RSC_TextureGrid& other) const {
    return RSC_TextureBase::operator==(other) && fill_color == other.fill_color &&
           line_color == other.line_color && is_meters == other.is_meters &&
           x_spacing == other.x_spacing && y_spacing == other.y_spacing &&
           line_width == other.line_width;
}

void RSC_TextureGrid::_validate() const {
    RSC_ResourceBase::_validate();
    RSC_EXPECT_GT(RSC_TextureGrid, x_spacing, 0);
    RSC_EXPECT_GT(RSC_TextureGrid, y_spacing, 0);
    RSC_EXPECT_GE(RSC_TextureGrid, line_width, 0);
}

void RSC_TextureGrid::_read(const RS_CharacterizationPropertyTree& rsc_tree) {
    RSC_ResourceBase::_read(rsc_tree);
    fill_color = readColor3f(rsc_tree.get<string>("FillColor"));
    line_color = readColor3f(rsc_tree.get<string>("LineColor"));
    is_meters = rsc_tree.get<bool>("IsMeters", is_meters);
    x_spacing = rsc_tree.get<RFloat>("XSpacing", x_spacing);
    y_spacing = rsc_tree.get<RFloat>("YSpacing", y_spacing);
    line_width = rsc_tree.get<RFloat>("LineWidth", line_width);
}

}  // namespace rsc

}  // namespace rise
