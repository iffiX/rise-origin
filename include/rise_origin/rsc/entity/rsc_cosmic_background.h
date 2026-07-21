// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_COSMIC_BACKGROUND_H
#define RSC_COSMIC_BACKGROUND_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_entity_base.h"

namespace rise {

namespace rsc {

class RSC_CosmicBackgroundBase : public RSC_EntityBase {
   public:
    bool operator==(const RSC_CosmicBackgroundBase& other) const {
        return RSC_EntityBase::operator==(other);
    }

   protected:
    void _validate(RS_RSCValidator& validator) const override { (void)validator; }
    void _read(const RS_RSCFileTreeReader& tree) override { (void)tree; }
    void _write(RS_RSCFileTreeWriter& tree) const override { (void)tree; }
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
};

/**
 * @brief A uniform lighting background sampled from a texture resource.
 */
class RSC_CosmicBackgroundUniform : public RSC_CosmicBackgroundBase {
   public:
    /// Reference to the texture resource.
    std::string texture_reference;

    /// Scalar multiplier applied to sampled radiance.
    RFloat intensity = 1.0;

    /// Rotation around the world up axis (radians).
    RFloat rotation_rad = 0.0;

    bool operator==(const RSC_CosmicBackgroundUniform& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A gradient sky blended between a zenith and horizon color, with a sun disk.
 */
class RSC_CosmicBackgroundGradient : public RSC_CosmicBackgroundBase {
   public:
    /// Zenith color; blended with horizon_color by |dir.z|.
    RColor3b zenith_color = RColor3b::zero();

    /// Horizon color; blended with zenith_color by |dir.z|.
    RColor3b horizon_color = RColor3b::zero();

    /// Sun direction in world space; must be non-zero.
    RVec3rf sun_direction = RVec3rf(0.3, 0.4, -1.0);

    /// Sun disk color along sun_direction.
    RColor3b sun_color = RColor3b::zero();

    /// Sun disk angular radius in radians, range (0, pi/2); larger = bigger sun.
    RFloat sun_angular_radius = 0.00465;

    bool operator==(const RSC_CosmicBackgroundGradient& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief A Hosek-Wilkie analytic sky model.
 */
class RSC_CosmicBackgroundHosekWilkie : public RSC_CosmicBackgroundBase {
   public:
    /// Sun direction in world space; must be non-zero. Pointing toward the sun.
    RVec3rf sun_direction = RVec3rf(0.3, 0.4, 0.866);

    /// Sun disk color along sun_direction.
    RColor3b sun_color = RColor3b::zero();

    /// Sun disk angular radius in radians, range (0, pi/2); larger = bigger sun.
    RFloat sun_angular_radius = 0.00465;

    /// Atmospheric turbidity in [1, 10]; 1 is pure Rayleigh, 10 is hazy.
    RFloat turbidity = 3.0;

    /// Ground color in [0, 1] per channel; feeds bounce light back into the sky.
    RColor3b ground_color = RColor3b::zero();

    /// Sky radiance multiplier; scales the final analytic radiance.
    RFloat intensity = 1.0;

    bool operator==(const RSC_CosmicBackgroundHosekWilkie& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_COSMIC_BACKGROUND_H
