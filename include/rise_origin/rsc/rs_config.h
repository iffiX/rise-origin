// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONFIG_H
#define RS_CONFIG_H

#include <vector>

#include "rise_origin/utils/rs_mp_serialization.h"
#include "rs_camera_config.h"
#include "rs_floor_config.h"
#include "rs_material_config.h"
#include "rs_record_config.h"
#include "rs_recorder_config.h"
#include "rs_signal_config.h"
#include "rs_simulation_config.h"
#include "rs_structure_config.h"
#include "rs_world_config.h"

namespace rise {

class RS_Config {
   public:
    /// Flag indicating whether the config is validated
    bool is_valid = false;
    std::string invalid_reason;

    /// configs
    RS_SimulationConfig simulation_config;
    RS_FloorConfig floor_config;
    RS_WorldConfig world_config;
    std::vector<RS_MaterialConfig> material_configs;
    std::vector<RS_StructureConfig> structure_configs;
    std::vector<RS_RecordConfig> record_configs;
    std::vector<RS_SignalConfig> signal_configs;
    std::vector<RS_RecorderConfig> recorder_configs;
    std::vector<RS_CameraConfig> camera_configs;

    RS_Config() = default;

    /**
     * @param configs A vector of configuration strings to be parsed,
     *     configuration string coming later will overwrite overridable sections.
     * @param throw_error Whether to throw an error when validation fails. If not,
     *     the validity and the invalid reason will be stored in RS_Config::is_valid and
     *     RS_Config::invalid_reason, respectively.
     */
    explicit RS_Config(const std::vector<std::string> &configs, bool throw_error = true);

    /**
     * @param config_paths A vector of configuration file paths to be parsed,
     *     configuration file coming later will overwrite overridable sections.
     * @param throw_error Whether to throw an error when validation fails. If not,
     *     the validity and the invalid reason will be stored in RS_Config::is_valid and
     *     RS_Config::invalid_reason, respectively.
     */
    void open(const std::vector<std::string> &config_paths, bool throw_error = true);

    [[nodiscard]] std::vector<uint8_t> saveAsBinary() const {
        RS_MessagePackSerializer s;
        s("config_root", *this);
        return s.serialize();
    }

    void loadFromBinary(const std::vector<uint8_t> &buffer) {
        RS_MessagePackDeserializer d;
        d.deserialize(buffer.data(), buffer.size());
        d("config_root", *this);
    }

    /**
     * Validate the configuration, when throw_error is true, throws an error. Otherwise,
     * stores validity in is_valid
     */
    void validate(bool throw_error = true);

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("is_valid", is_valid);
        s("invalid_reason", invalid_reason);
        s("simulation_config", simulation_config);
        s("floor_config", floor_config);
        s("world_config", world_config);
        s("material_configs", material_configs);
        s("structure_configs", structure_configs);
        s("record_configs", record_configs);
        s("signal_configs", signal_configs);
        s("recorder_configs", recorder_configs);
        s("camera_configs", camera_configs);
    }

   private:
    std::vector<std::string> _configs;

    void _parseSettings();
};

}  // namespace rise
#endif  // RS_CONFIG_H
