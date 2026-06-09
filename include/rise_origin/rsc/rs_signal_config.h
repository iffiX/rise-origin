// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIGNAL_CONFIG_H
#define RS_SIGNAL_CONFIG_H

#include <unordered_map>

#include "rise_origin/utils/rs_any.h"
#include "rise_origin/utils/rs_bound.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_base_config.h"

namespace rise {

class RS_ForceSignalConfig : public RS_BaseConfig {
   public:
    /// Name of the structure to apply the force signal to
    std::string structure_name;

    /// Voxel range (bound) to apply the force signal to
    RBound3ri voxel_range;

    /// Structure scope ID of the body to apply the force signal to
    RIndex body_sid = RS_NULL_INDEX;

    /// Body scope ID of the segment to apply the force signal to
    RIndex segment_bid = RS_NULL_INDEX;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("structure_name", structure_name);
        s("voxel_range", voxel_range);
        s("body_sid", body_sid);
        s("segment_bid", segment_bid);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

class RS_TorqueSignalConfig : public RS_BaseConfig {
   public:
    /// Name of the structure to apply the torque signal to
    std::string structure_name;

    /// Structure scope ID of the body to apply the torque signal to
    RIndex body_sid = RS_NULL_INDEX;

    /// Body scope ID of the segment to apply the torque signal to.
    /// The segment must be a rigid body.
    RIndex segment_bid = RS_NULL_INDEX;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("structure_name", structure_name);
        s("body_sid", body_sid);
        s("segment_bid", segment_bid);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

RS_ENUM(RSE_SignalType, unsigned int, (RSE_SIGNAL_FORCE, 0), (RSE_SIGNAL_TORQUE, 1))

class RS_SignalConfig : public RS_BaseConfig {
   public:
    /// Name of the signal.
    std::string name;

    RSE_SignalType type = RSE_SIGNAL_FORCE;

    RS_Any config;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("name", name);
        s("type", type);
        if (type == RSE_SIGNAL_FORCE) {
            s("config", config.as<RS_ForceSignalConfig>());
        } else if (type == RSE_SIGNAL_TORQUE) {
            s("config", config.as<RS_TorqueSignalConfig>());
        }
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

}  // namespace rise
#endif  // RS_SIGNAL_CONFIG_H
