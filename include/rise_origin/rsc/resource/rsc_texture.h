// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_TEXTURE_H
#define RSC_TEXTURE_H

#include <string>
#include <vector>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_resource_base.h"

namespace rise {

namespace rsc {

class RSC_TextureBase : public RSC_ResourceBase {
   public:
    bool operator==(const RSC_TextureBase& other) const {
        return RSC_ResourceBase::operator==(other);
    }

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_ResourceBase::serialize(s);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override { (void)validator; }
    void _read(const RS_RSCPropertyTreeReader& tree) override { (void)tree; }
    void _write(RS_RSCPropertyTreeWriter& tree) const override { (void)tree; }
};

class RSC_TextureConstant : public RSC_TextureBase {
   public:
    /// Constant value as RGB float (use R channel for scalar properties like roughness).
    RColor3f value = RColor3f::zero();

    bool operator==(const RSC_TextureConstant& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_TextureBase::serialize(s);
        s("value", value);
    }

   protected:
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A procedural grid pattern. When is_meters is true the pattern is evaluated in world-space
 * meters (a uniform world-aligned grid, e.g. for floors and measurement); when false it is
 * evaluated in surface UV space (per-face 0..1). Lines are centered on multiples of the per-axis
 * spacing and straddled symmetrically by line_width; cells between lines take fill_color.
 */
class RSC_TextureGrid : public RSC_TextureBase {
   public:
    /// Color of the cells between grid lines, as RGB float.
    RColor3f fill_color = RColor3f::zero();

    /// Color of the grid lines, as RGB float.
    RColor3f line_color = RColor3f::zero();

    /// If true, x_spacing/y_spacing/line_width are world-space meters; if false, surface UV units.
    bool is_meters = true;

    /// Spacing between grid lines along X.
    RFloat x_spacing = 1;

    /// Spacing between grid lines along Y.
    RFloat y_spacing = 1;

    /// Width of the grid lines.
    RFloat line_width = 0.05;

    bool operator==(const RSC_TextureGrid& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_TextureBase::serialize(s);
        s("fill_color", fill_color);
        s("line_color", line_color);
        s("is_meters", is_meters);
        s("x_spacing", x_spacing);
        s("y_spacing", y_spacing);
        s("line_width", line_width);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

using RSC_TextureVariant = RS_Variant<RSC_TextureConstant, RSC_TextureGrid>;

}  // namespace rsc

}  // namespace rise

#endif  // RSC_TEXTURE_H
