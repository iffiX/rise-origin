// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RECORDER_CONFIG_H
#define RS_RECORDER_CONFIG_H

#include "rise_origin/utils/rs_any.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_base_config.h"

namespace rise {

class RS_HDF5RecorderConfig : public RS_BaseConfig {
   public:
    std::string path;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("path", path);
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

RS_ENUM(RSE_RecorderType, unsigned int, (RSE_RECORDER_HDF5, 0))

class RS_RecorderConfig : public RS_BaseConfig {
   public:
    RSE_RecorderType type = RSE_RECORDER_HDF5;

    RS_Any config;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("type", type);
        if (type == RSE_RECORDER_HDF5) {
            s("config", config.as<RS_HDF5RecorderConfig>());
        }
    }

   protected:
    void _validate() const override;
    void _read(const RS_ConfigPropertyTree &config_tree) override;
};

}  // namespace rise

#endif  // RS_RECORDER_CONFIG_H
