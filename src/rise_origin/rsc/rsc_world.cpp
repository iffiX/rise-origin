// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rsc_world.h"

#include <fstream>
#include <sstream>

#include "rise_origin/utils/string/rs_format.h"

using namespace std;

namespace rise {

namespace rsc {

namespace {

template <typename T>
void validateNamed(const char *ctx, const T &v) {
    try {
        v.validate();
    } catch (RS_InvalidCharacterizationException &ex) {
        RS_THROW(RS_InvalidCharacterizationException(
            "while validating {} {}: {}", ctx, v.name, ex.what()
        ));
    }
}

}  // namespace

RSC_World::RSC_World(const vector<string> &rsc_strings, bool throw_error) {
    bool is_non_empty = false;
    _rsc_strings.assign(rsc_strings.begin(), rsc_strings.end());
    for (const auto &rsc_string : rsc_strings) {
        if (!rsc_string.empty()) is_non_empty = true;
    }
    if (is_non_empty) {
        _parseSettings();
    }
    if (!is_valid && throw_error) {
        RS_THROW(RS_InvalidCharacterizationException(invalid_reason));
    }
}

void RSC_World::open(const vector<string> &rsc_paths, bool throw_error) {
    bool is_non_empty = false;
    for (const auto &rsc_path : rsc_paths) {
        ifstream rsc_file(rsc_path);
        if (!rsc_file.is_open()) {
            RS_THROW(RS_DataException("Invalid RSC file path: {}", rsc_path));
        }
        if (rsc_file.peek() != ifstream::traits_type::eof()) {
            is_non_empty = true;
        }
        stringstream stream;
        stream << rsc_file.rdbuf();
        _rsc_strings.push_back(stream.str());
    }

    if (is_non_empty) {
        _parseSettings();
    }

    if (!is_valid && throw_error) {
        RS_THROW(RS_InvalidCharacterizationException(invalid_reason));
    }
}

void RSC_World::validate(bool throw_error) {
    try {
        try {
            simulation.validate();
        } catch (RS_InvalidCharacterizationException &ex) {
            RS_THROW(
                RS_InvalidCharacterizationException("while validating simulation: {}", ex.what())
            );
        }
        for (auto &r : resources) {
            r.visit([](const auto &v) { validateNamed("resource", v); });
        }
        for (auto &e : essences) {
            e.visit([](const auto &v) { validateNamed("essence", v); });
        }
        for (auto &e : entities) {
            e.visit([](const auto &v) { validateNamed("entity", v); });
        }
        for (auto &r : records) {
            r.visit([](const auto &v) { validateNamed("record", v); });
        }
        for (auto &s : signals) {
            s.visit([](const auto &v) { validateNamed("signal", v); });
        }
        for (auto &r : recorders) {
            r.visit([](const auto &v) { validateNamed("recorder", v); });
        }
        for (auto &c : controllers) {
            c.visit([](const auto &v) { validateNamed("controller", v); });
        }
        is_valid = true;
    } catch (RS_InvalidCharacterizationException &e) {
        is_valid = false;
        invalid_reason = e.what();
    }
    if (!is_valid && throw_error) {
        RS_THROW(RS_InvalidCharacterizationException(invalid_reason));
    }
}

bool RSC_World::operator==(const RSC_World &other) const {
    return RSC_Base::operator==(other) && name == other.name && is_valid == other.is_valid &&
           invalid_reason == other.invalid_reason && simulation == other.simulation &&
           resources == other.resources && essences == other.essences &&
           entities == other.entities && records == other.records && signals == other.signals &&
           recorders == other.recorders && controllers == other.controllers;
}

void RSC_World::_validate() const {}

void RSC_World::_read(const RS_CharacterizationPropertyTree & /*rsc_tree*/) {}

void RSC_World::_parseSettings() {
    try {
        for (auto &rsc_string : _rsc_strings) {
            if (rsc_string.empty()) continue;
            boost::property_tree::ptree c_tree;
            try {
                stringstream rsc_stream;
                rsc_stream << rsc_string;
                read_xml(rsc_stream, c_tree);
            } catch (...) {
                RS_THROW(RS_InvalidCharacterizationException("Invalid XML input"));
            }

            RS_CharacterizationPropertyTree rsc_tree(c_tree);

            if (rsc_tree.isChildExist("RSC.Name")) {
                name = rsc_tree.get<string>("RSC.Name");
            }
            if (rsc_tree.isChildExist("RSC.Simulation")) {
                simulation.read(rsc_tree.getChild("RSC.Simulation"));
            }
            if (rsc_tree.isChildExist("RSC.Resources")) {
                for (auto &[tag, child] : rsc_tree.getChild("RSC.Resources").getChildren()) {
                    if (tag == "TextureConstant") {
                        RSC_TextureConstant e;
                        e.read(child, false);
                        resources.emplace_back(e);
                    } else if (tag == "TextureGrid") {
                        RSC_TextureGrid e;
                        e.read(child, false);
                        resources.emplace_back(e);
                    } else {
                        RS_THROW(
                            RS_InvalidCharacterizationException("Unknown resource tag: {}", tag)
                        );
                    }
                }
            }
            if (rsc_tree.isChildExist("RSC.Essences")) {
                for (auto &[tag, child] : rsc_tree.getChild("RSC.Essences").getChildren()) {
                    if (tag == "Material") {
                        RSC_Material e;
                        e.read(child, false);
                        essences.emplace_back(e);
                    } else if (tag == "MaterialPhysicalPropertySoft") {
                        RSC_MaterialPhysicalPropertySoft e;
                        e.read(child, false);
                        essences.emplace_back(e);
                    } else if (tag == "MaterialPhysicalPropertyFixed") {
                        RSC_MaterialPhysicalPropertyFixed e;
                        e.read(child, false);
                        essences.emplace_back(e);
                    } else if (tag == "MaterialPhysicalPropertyRigid") {
                        RSC_MaterialPhysicalPropertyRigid e;
                        e.read(child, false);
                        essences.emplace_back(e);
                    } else if (tag == "MaterialPhysicalPropertyAtlas") {
                        RSC_MaterialPhysicalPropertyAtlas e;
                        e.read(child, false);
                        essences.emplace_back(e);
                    } else if (tag == "MaterialOpticalPropertySimpleMicrofacet") {
                        RSC_MaterialOpticalPropertySimpleMicrofacet e;
                        e.read(child, false);
                        essences.emplace_back(e);
                    } else if (tag == "MaterialPairPropertyCoulombFriction") {
                        RSC_MaterialPairPropertyCoulombFriction e;
                        e.read(child, false);
                        essences.emplace_back(e);
                    } else if (tag == "MaterialPairPropertyRestitution") {
                        RSC_MaterialPairPropertyRestitution e;
                        e.read(child, false);
                        essences.emplace_back(e);
                    } else {
                        RS_THROW(
                            RS_InvalidCharacterizationException("Unknown essence tag: {}", tag)
                        );
                    }
                }
            }
            if (rsc_tree.isChildExist("RSC.Entities")) {
                for (auto &[tag, child] : rsc_tree.getChild("RSC.Entities").getChildren()) {
                    if (tag == "StructureVoxel") {
                        RSC_StructureVoxel e;
                        e.read(child, false);
                        entities.emplace_back(e);
                    } else if (tag == "CameraPerspectiveFree") {
                        RSC_CameraPerspectiveFree e;
                        e.read(child, false);
                        entities.emplace_back(e);
                    } else if (tag == "CameraPerspectiveRigidBodyLinked") {
                        RSC_CameraPerspectiveRigidBodyLinked e;
                        e.read(child, false);
                        entities.emplace_back(e);
                    } else if (tag == "FloorFlat") {
                        RSC_FloorFlat e;
                        e.read(child, false);
                        entities.emplace_back(e);
                    } else if (tag == "FloorArray") {
                        RSC_FloorArray e;
                        e.read(child, false);
                        entities.emplace_back(e);
                    } else if (tag == "CosmicBackgroundUniform") {
                        RSC_CosmicBackgroundUniform e;
                        e.read(child, false);
                        entities.emplace_back(e);
                    } else if (tag == "CosmicBackgroundGradient") {
                        RSC_CosmicBackgroundGradient e;
                        e.read(child, false);
                        entities.emplace_back(e);
                    } else if (tag == "CosmicBackgroundHosekWilkie") {
                        RSC_CosmicBackgroundHosekWilkie e;
                        e.read(child, false);
                        entities.emplace_back(e);
                    } else {
                        RS_THROW(
                            RS_InvalidCharacterizationException("Unknown entity tag: {}", tag)
                        );
                    }
                }
            }
            if (rsc_tree.isChildExist("RSC.Records")) {
                for (auto &[tag, child] : rsc_tree.getChild("RSC.Records").getChildren()) {
                    if (tag == "IntersectRegionCuboid") {
                        RSC_IntersectRegionCuboid e;
                        e.read(child, false);
                        records.emplace_back(e);
                    } else if (tag == "IntersectRegionCuboidRigidBodyLinked") {
                        RSC_IntersectRegionCuboidRigidBodyLinked e;
                        e.read(child, false);
                        records.emplace_back(e);
                    } else if (tag == "IntersectRegionSphere") {
                        RSC_IntersectRegionSphere e;
                        e.read(child, false);
                        records.emplace_back(e);
                    } else {
                        RS_THROW(
                            RS_InvalidCharacterizationException("Unknown record tag: {}", tag)
                        );
                    }
                }
            }
            if (rsc_tree.isChildExist("RSC.Signals")) {
                for (auto &[tag, child] : rsc_tree.getChild("RSC.Signals").getChildren()) {
                    if (tag == "Force") {
                        RSC_Force e;
                        e.read(child, false);
                        signals.emplace_back(e);
                    } else if (tag == "Torque") {
                        RSC_Torque e;
                        e.read(child, false);
                        signals.emplace_back(e);
                    } else {
                        RS_THROW(
                            RS_InvalidCharacterizationException("Unknown signal tag: {}", tag)
                        );
                    }
                }
            }
            if (rsc_tree.isChildExist("RSC.Recorders")) {
                for (auto &[tag, child] : rsc_tree.getChild("RSC.Recorders").getChildren()) {
                    if (tag == "RecorderHDF5") {
                        RSC_RecorderHDF5 e;
                        e.read(child, false);
                        recorders.emplace_back(e);
                    } else {
                        RS_THROW(
                            RS_InvalidCharacterizationException("Unknown recorder tag: {}", tag)
                        );
                    }
                }
            }
            if (rsc_tree.isChildExist("RSC.Controllers")) {
                for (auto &[tag, child] : rsc_tree.getChild("RSC.Controllers").getChildren()) {
                    if (tag == "ControllerCallback") {
                        RSC_ControllerCallback e;
                        e.read(child, false);
                        controllers.emplace_back(e);
                    } else {
                        RS_THROW(
                            RS_InvalidCharacterizationException("Unknown controller tag: {}", tag)
                        );
                    }
                }
            }
        }
        is_valid = true;
    } catch (RS_InvalidCharacterizationException &e) {
        is_valid = false;
        invalid_reason = e.what();
    } catch (...) {
        is_valid = false;
        invalid_reason = "Unknown error while parsing settings";
    }
}

}  // namespace rsc

}  // namespace rise
