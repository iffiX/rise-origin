// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rsc_world.h"

#include "rise_origin/rsc/rs_rsc_file.h"
#include "rise_origin/rsc/rs_rsc_registry.h"

namespace rise {

namespace rsc {

void RSC_World::assembleFrom(const std::vector<std::string> &paths) {
    auto &registry = RS_RSCRegistry::getInstance();
    for (const auto &path : paths) {
        RSCRef<RSC_Base> node = RS_RSCFile::loadAny(path);

        if (auto *world = dynamic_cast<RSC_World *>(node.get())) {
            if (world->simulation) simulation = world->simulation;
            for (const auto &resource : world->resources) resources.push_back(resource);
            for (const auto &essence : world->essences) essences.push_back(essence);
            for (const auto &entity : world->entities) entities.push_back(entity);
            for (const auto &record : world->records) records.push_back(record);
            for (const auto &signal : world->signals) signals.push_back(signal);
            for (const auto &recorder : world->recorders) recorders.push_back(recorder);
            for (const auto &controller : world->controllers) controllers.push_back(controller);
            continue;
        }

        RSC_ResourceVariant resource;
        if (registry.tryAssign(resource, node)) {
            resources.push_back(resource);
            continue;
        }
        RSC_EssenceVariant essence;
        if (registry.tryAssign(essence, node)) {
            essences.push_back(essence);
            continue;
        }
        RSC_EntityVariant entity;
        if (registry.tryAssign(entity, node)) {
            entities.push_back(entity);
            continue;
        }
        RSC_RecordVariant record;
        if (registry.tryAssign(record, node)) {
            records.push_back(record);
            continue;
        }
        RSC_SignalVariant signal;
        if (registry.tryAssign(signal, node)) {
            signals.push_back(signal);
            continue;
        }
        RSC_RecorderVariant recorder;
        if (registry.tryAssign(recorder, node)) {
            recorders.push_back(recorder);
            continue;
        }
        RSC_ControllerVariant controller;
        if (registry.tryAssign(controller, node)) {
            controllers.push_back(controller);
            continue;
        }

        RS_THROW(RS_InvalidCharacterizationException(
            "RSC file {} is not a world or a world component", path
        ));
    }
}

bool RSC_World::operator==(const RSC_World &other) const {
    return RSC_Base::operator==(other) && name == other.name && simulation == other.simulation &&
           resources == other.resources && essences == other.essences &&
           entities == other.entities && records == other.records && signals == other.signals &&
           recorders == other.recorders && controllers == other.controllers;
}

void RSC_World::_validate(RS_RSCValidator &validator) const {
    if (simulation) simulation->validate(validator);
    for (auto &r : resources) r.visit([&](const auto &v) { v.validate(validator); });
    for (auto &e : essences) e.visit([&](const auto &v) { v.validate(validator); });
    for (auto &e : entities) e.visit([&](const auto &v) { v.validate(validator); });
    for (auto &r : records) r.visit([&](const auto &v) { v.validate(validator); });
    for (auto &s : signals) s.visit([&](const auto &v) { v.validate(validator); });
    for (auto &r : recorders) r.visit([&](const auto &v) { v.validate(validator); });
    for (auto &c : controllers) c.visit([&](const auto &v) { v.validate(validator); });
}

void RSC_World::_read(const RS_RSCPropertyTreeReader &tree) {
    auto &registry = RS_RSCRegistry::getInstance();
    name = tree.get<std::string>("name", name);
    simulation = nullptr;
    if (tree.isChildExist("simulation")) {
        simulation = RSC_Simulation();
        simulation->read(tree.getChild("simulation"));
    }
    for (const auto &element : tree.getArray("resources")) {
        RSC_ResourceVariant value;
        registry.read(value, element);
        resources.push_back(value);
    }
    for (const auto &element : tree.getArray("essences")) {
        RSC_EssenceVariant value;
        registry.read(value, element);
        essences.push_back(value);
    }
    for (const auto &element : tree.getArray("entities")) {
        RSC_EntityVariant value;
        registry.read(value, element);
        entities.push_back(value);
    }
    for (const auto &element : tree.getArray("records")) {
        RSC_RecordVariant value;
        registry.read(value, element);
        records.push_back(value);
    }
    for (const auto &element : tree.getArray("signals")) {
        RSC_SignalVariant value;
        registry.read(value, element);
        signals.push_back(value);
    }
    for (const auto &element : tree.getArray("recorders")) {
        RSC_RecorderVariant value;
        registry.read(value, element);
        recorders.push_back(value);
    }
    for (const auto &element : tree.getArray("controllers")) {
        RSC_ControllerVariant value;
        registry.read(value, element);
        controllers.push_back(value);
    }
}

void RSC_World::_write(RS_RSCPropertyTreeWriter &tree) const {
    auto &registry = RS_RSCRegistry::getInstance();
    tree.add("name", name);
    if (simulation) {
        simulation->write(tree.addChild("simulation"));
    }
    {
        auto array = tree.addArray("resources");
        for (const auto &value : resources) registry.write(value, array.add());
    }
    {
        auto array = tree.addArray("essences");
        for (const auto &value : essences) registry.write(value, array.add());
    }
    {
        auto array = tree.addArray("entities");
        for (const auto &value : entities) registry.write(value, array.add());
    }
    {
        auto array = tree.addArray("records");
        for (const auto &value : records) registry.write(value, array.add());
    }
    {
        auto array = tree.addArray("signals");
        for (const auto &value : signals) registry.write(value, array.add());
    }
    {
        auto array = tree.addArray("recorders");
        for (const auto &value : recorders) registry.write(value, array.add());
    }
    {
        auto array = tree.addArray("controllers");
        for (const auto &value : controllers) registry.write(value, array.add());
    }
}

void RSC_World::_deepCopyShared(MemoRef memo) {
    simulation = simulation.deepCopy(memo);
    resources = resources.deepCopy(memo);
    essences = essences.deepCopy(memo);
    entities = entities.deepCopy(memo);
    records = records.deepCopy(memo);
    signals = signals.deepCopy(memo);
    recorders = recorders.deepCopy(memo);
    controllers = controllers.deepCopy(memo);
}

}  // namespace rsc

}  // namespace rise
