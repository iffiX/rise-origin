// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_config.h"

using namespace std;

namespace rise {

RS_Config::RS_Config(const vector<string> &configs, bool throw_error) {
    bool is_non_empty = false;
    _configs.assign(configs.begin(), configs.end());
    for (const auto &config : configs) {
        if (!config.empty()) is_non_empty = true;
    }
    if (is_non_empty) {
        _parseSettings();
    } else {
        is_valid = true;
    }
    if (!is_valid && throw_error) {
        RS_THROW(RS_InvalidConfigurationException(invalid_reason));
    }
}

void RS_Config::open(const vector<string> &config_paths, bool throw_error) {
    bool is_non_empty = false;
    for (const auto &config_path : config_paths) {
        ifstream config_file(config_path);
        if (!config_file.is_open())
            RS_THROW(RS_InvalidConfigurationException("Invalid config file path"));
        if (config_file.peek() != ifstream::traits_type::eof()) {
            is_non_empty = true;
        }
        stringstream stream;
        stream << config_file.rdbuf();
        _configs.push_back(stream.str());
    }
    if (is_non_empty) {
        _parseSettings();
    } else {
        is_valid = true;
    }
    if (!is_valid && throw_error) {
        RS_THROW(RS_InvalidConfigurationException(invalid_reason));
    }
}

void RS_Config::validate(bool throw_error) {
    try {
        simulation_config.validate();
        floor_config.validate();
        world_config.validate();
        for (auto &material_config : material_configs) {
            material_config.validate();
        }
        for (auto &structure_config : structure_configs) {
            structure_config.validate();
        }
        for (auto &record_config : record_configs) {
            record_config.validate();
        }
        for (auto &signal_config : signal_configs) {
            signal_config.validate();
        }
        for (auto &recorder_config : recorder_configs) {
            recorder_config.validate();
        }
        for (auto &camera_config : camera_configs) {
            camera_config.validate();
        }
        is_valid = true;
    } catch (RS_InvalidConfigurationException &e) {
        is_valid = false;
        invalid_reason = e.what();
    }

    if (!is_valid && throw_error) {
        RS_THROW(RS_InvalidConfigurationException(invalid_reason));
    }
}

void RS_Config::_parseSettings() {
    try {
        for (auto &config : _configs) {
            if (config.empty()) continue;
            boost::property_tree::ptree c_tree;
            try {
                stringstream config_stream;
                config_stream << config;
                read_xml(config_stream, c_tree);
            } catch (...) {
                RS_THROW(RS_InvalidConfigurationException("Invalid XML input"));
            }

            RS_ConfigPropertyTree config_tree(c_tree);
            if (config_tree.isChildExist("RSC.Simulation")) {
                simulation_config.read(config_tree.getChild("RSC.Simulation"));
            }
            if (config_tree.isChildExist("RSC.Floor")) {
                floor_config.read(config_tree.getChild("RSC.Floor"));
            }
            if (config_tree.isChildExist("RSC.World")) {
                world_config.read(config_tree.getChild("RSC.World"));
            }
            if (config_tree.isChildExist("RSC.Materials")) {
                for (auto &child : config_tree.getChild("RSC.Materials").getChildrenValues()) {
                    RS_MaterialConfig material_config;
                    material_config.read(child);
                    material_configs.emplace_back(material_config);
                }
            }
            if (config_tree.isChildExist("RSC.Structures")) {
                for (auto &child : config_tree.getChild("RSC.Structures").getChildrenValues()) {
                    RS_StructureConfig structure_config;
                    structure_config.read(child);
                    structure_configs.emplace_back(structure_config);
                }
            }
            if (config_tree.isChildExist("RSC.Records")) {
                for (auto &child : config_tree.getChild("RSC.Records").getChildrenValues()) {
                    RS_RecordConfig record_config;
                    record_config.read(child);
                    record_configs.emplace_back(record_config);
                }
            }
            if (config_tree.isChildExist("RSC.Signals")) {
                for (auto &child : config_tree.getChild("RSC.Signals").getChildrenValues()) {
                    RS_SignalConfig signal_config;
                    signal_config.read(child);
                    signal_configs.emplace_back(signal_config);
                }
            }
            if (config_tree.isChildExist("RSC.Recorders")) {
                for (auto &child : config_tree.getChild("RSC.Recorders").getChildrenValues()) {
                    RS_RecorderConfig recorder_config;
                    recorder_config.read(child);
                    recorder_configs.emplace_back(recorder_config);
                }
            }

            if (config_tree.isChildExist("RSC.Cameras")) {
                for (auto &child : config_tree.getChild("RSC.Cameras").getChildrenValues()) {
                    RS_CameraConfig camera_config;
                    camera_config.read(child);
                    camera_configs.emplace_back(camera_config);
                }
            }
        }
        is_valid = true;
    } catch (RS_InvalidConfigurationException &e) {
        is_valid = false;
        invalid_reason = e.what();
    } catch (...) {
        is_valid = false;
        invalid_reason = "Unknown error while parsing settings";
    }
}

}  // namespace rise