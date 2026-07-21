// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_FLOOR_H
#define RSC_FLOOR_H

#include <string>
#include <vector>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_entity_base.h"

namespace rise {

namespace rsc {

/*****************************************************************
 * Floor Types
 *****************************************************************/

/**
 * @brief Common base for floor entities: references a material essence.
 */
class RSC_FloorBase : public RSC_EntityBase {
   public:
    /// Name of the material essence to use.
    std::string material_reference;

    bool operator==(const RSC_FloorBase& other) const {
        return RSC_EntityBase::operator==(other) && material_reference == other.material_reference;
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
};

/**
 * @brief An infinite flat floor plane at z = height, centered on the origin in XY.
 */
class RSC_FloorFlat : public RSC_FloorBase {
   public:
    /// Height of the flat floor plane in meters.
    RFloat height = 0;

    /// World-space size (meters) of one UV tile. The plane is infinite, so the material's
    /// texture repeats every uv_scale meters along each axis.
    RFloat uv_scale = 1;

    bool operator==(const RSC_FloorFlat& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A bounded floor plane defined by a height matrix.
 */
class RSC_FloorArray : public RSC_FloorBase {
   public:
    /// Size of the elevated floor in meters in X direction.
    RFloat x_size = 0;

    /// Size of the elevated floor in meters in Y direction.
    RFloat y_size = 0;

    /// Number of values in X direction (columns).
    RSize x_values = 0;

    /// Number of values in Y direction (rows).
    RSize y_values = 0;

    /// Whether the floor repeats in X direction.
    bool is_x_repeat = false;

    /// Whether the floor repeats in Y direction.
    bool is_y_repeat = false;

    /// 2D height matrix of size [y_values, x_values] in meters.
    std::vector<float> height;

    bool operator==(const RSC_FloorArray& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

using RSC_FloorVariant = RS_Variant<RSC_FloorFlat, RSC_FloorArray>;

}  // namespace rsc

}  // namespace rise

#endif  // RSC_FLOOR_H
