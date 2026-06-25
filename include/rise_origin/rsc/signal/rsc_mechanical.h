// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_MECHANICAL_H
#define RSC_MECHANICAL_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/math/rs_bound.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_signal_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Force signal applied to a region of a structure.
 */
class RSC_Force : public RSC_SignalBase {
   public:
    /// Name of the structure to apply the force signal to.
    std::string structure_name;

    /// Voxel range (bound) to apply the force signal to.
    RBound3ri voxel_range;

    /// Structure scope ID of the body to apply force to (default RS_NULL_INDEX for all).
    RIndex body_sid = RS_NULL_INDEX;

    /// Body scope ID of the segment to apply force to (default RS_NULL_INDEX for all).
    RIndex segment_bid = RS_NULL_INDEX;

    bool operator==(const RSC_Force& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_SignalBase::serialize(s);
        s("structure_name", structure_name);
        s("voxel_range", voxel_range);
        s("body_sid", body_sid);
        s("segment_bid", segment_bid);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Torque signal applied to a rigid body of a structure.
 */
class RSC_Torque : public RSC_SignalBase {
   public:
    /// Name of the structure to apply the torque signal to.
    std::string structure_name;

    /// Structure scope ID of the body to apply torque to (default RS_NULL_INDEX for all).
    RIndex body_sid = RS_NULL_INDEX;

    /// Body scope ID of the segment to apply torque to (default RS_NULL_INDEX for all).
    RIndex segment_bid = RS_NULL_INDEX;

    bool operator==(const RSC_Torque& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_SignalBase::serialize(s);
        s("structure_name", structure_name);
        s("body_sid", body_sid);
        s("segment_bid", segment_bid);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Transform signal applied once to the full structure, then reset to identity.
 */
class RSC_Transform : public RSC_SignalBase {
   public:
    /// Name of the structure to apply the transform signal to.
    std::string structure_name;

    bool operator==(const RSC_Transform& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_SignalBase::serialize(s);
        s("structure_name", structure_name);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCPropertyTreeReader& tree) override;
    void _write(RS_RSCPropertyTreeWriter& tree) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_MECHANICAL_H
