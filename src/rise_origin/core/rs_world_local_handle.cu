// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rs_kernel_manager.h"
#include "rs_world_local_handle.h"

using namespace std;
using namespace rise::rsc;

namespace rise {

RS_WorldLocalHandle::RS_WorldLocalHandle(const std::shared_ptr<RS_Simulation>& simulation)
    : _simulation(simulation) {}

std::string RS_WorldLocalHandle::name() { return _simulation->name; }

bool RS_WorldLocalHandle::isPending() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return status == RSE_WORLD_UNINITIALIZED || status == RSE_WORLD_INITIALIZING;
}

bool RS_WorldLocalHandle::isRunning() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return status == RSE_WORLD_RUNNING;
}

bool RS_WorldLocalHandle::isPaused() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return status == RSE_WORLD_PAUSED;
}

bool RS_WorldLocalHandle::isEnd() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return isWorldEnd(status);
}

void RS_WorldLocalHandle::pause() {
    if (waitForRunning()) {
        _simulation->kernel_manager->pauseKernel();
    }
}

void RS_WorldLocalHandle::resume() {
    if (waitForRunning()) {
        _simulation->kernel_manager->resumeKernel();
    }
}

void RS_WorldLocalHandle::terminate() {
    if (waitForRunning()) {
        _simulation->kernel_manager->terminateKernel();
    }
}

RSE_WorldStatus RS_WorldLocalHandle::getStatus() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return status;
}

bool RS_WorldLocalHandle::waitForRunning() { return _simulation->waitForStatus(RSE_WORLD_RUNNING); }

void RS_WorldLocalHandle::waitForEnd() { _simulation->waitForEnd(); }

RS_WorldMessage RS_WorldLocalHandle::waitForMessage(RIndex message_id, RSE_WorldMessageLevel level) {
    return _simulation->waitForMessage(message_id, level);
}

void RS_WorldLocalHandle::beginModificationTransaction() {
    if (waitForRunning()) {
        _simulation->kernel_manager->beginModificationTransaction();
    }
}

void RS_WorldLocalHandle::endModificationTransaction() {
    if (waitForRunning()) {
        _simulation->kernel_manager->endModificationTransaction();
    }
}

RSC_World RS_WorldLocalHandle::getWorld() {
    if (!waitForRunning()) return {};
    RSC_World world = _simulation->kernel_manager->getWorld();
    // Controllers and recorders live on RS_Simulation; merge them in.
    {
        std::lock_guard lock(_simulation->controllers_mutex);
        for (auto& [_, controller] : _simulation->controllers) world.controllers.push_back(controller);
    }
    {
        std::lock_guard lock(_simulation->recorders_mutex);
        for (auto& [_, entry] : _simulation->recorders) world.recorders.push_back(entry.first);
    }
    return world;
}

RIndex RS_WorldLocalHandle::setSimulation(const RSC_Simulation& simulation) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->setSimulationRSC(simulation, id);
    return id;
}

RIndex RS_WorldLocalHandle::addResource(const RSC_ResourceVariant& resource) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->addResource(resource, id);
    return id;
}

RIndex RS_WorldLocalHandle::removeResource(const std::string& name) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->removeResource(name, id);
    return id;
}

RSC_ResourceVariant RS_WorldLocalHandle::findResource(const std::string& name) {
    if (waitForRunning()) return _simulation->kernel_manager->findResource(name);
    return {};
}

RIndex RS_WorldLocalHandle::addEssence(const RSC_EssenceVariant& essence) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->addEssence(essence, id);
    return id;
}

RIndex RS_WorldLocalHandle::updateEssence(
    const std::string& name,
    const RSC_EssenceVariant& essence
) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->updateEssence(name, essence, id);
    return id;
}

RIndex RS_WorldLocalHandle::removeEssence(const std::string& name) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->removeEssence(name, id);
    return id;
}

RSC_EssenceVariant RS_WorldLocalHandle::findEssence(const std::string& name) {
    if (waitForRunning()) return _simulation->kernel_manager->findEssence(name);
    return {};
}

RIndex RS_WorldLocalHandle::addEntity(const RSC_EntityVariant& entity) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->addEntity(entity, id);
    return id;
}

RIndex RS_WorldLocalHandle::removeEntity(const std::string& name) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->removeEntity(name, id);
    return id;
}

RIndex RS_WorldLocalHandle::freezeEntity(const std::string& name) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->freezeEntity(name, id);
    return id;
}

RIndex RS_WorldLocalHandle::unfreezeEntity(const std::string& name) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->unfreezeEntity(name, id);
    return id;
}

RSC_EntityVariant RS_WorldLocalHandle::findEntity(const std::string& name) {
    if (waitForRunning()) return _simulation->kernel_manager->findEntity(name);
    return {};
}

bool RS_WorldLocalHandle::isEntityFrozen(const std::string& name) {
    if (waitForRunning()) return _simulation->kernel_manager->isEntityFrozen(name);
    return false;
}

RIndex RS_WorldLocalHandle::addRecord(const RSC_RecordVariant& record) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->addRecord(record, id);
    return id;
}

RIndex RS_WorldLocalHandle::removeRecord(const std::string& name) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->removeRecord(name, id);
    return id;
}

RSC_RecordVariant RS_WorldLocalHandle::findRecord(const std::string& name) {
    if (waitForRunning()) return _simulation->kernel_manager->findRecord(name);
    return {};
}

RIndex RS_WorldLocalHandle::addSignal(const RSC_SignalVariant& signal) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->addSignal(signal, id);
    return id;
}

RIndex RS_WorldLocalHandle::removeSignal(const std::string& name) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->kernel_manager->removeSignal(name, id);
    return id;
}

RSC_SignalVariant RS_WorldLocalHandle::findSignal(const std::string& name) {
    if (waitForRunning()) return _simulation->kernel_manager->findSignal(name);
    return {};
}

RIndex RS_WorldLocalHandle::addRecorder(const RSC_RecorderVariant& recorder) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->addRecorder(recorder);
    return id;
}

RIndex RS_WorldLocalHandle::removeRecorder(const std::string& name) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->removeRecorder(name);
    return id;
}

RSC_RecorderVariant RS_WorldLocalHandle::findRecorder(const std::string& name) {
    if (waitForRunning()) return _simulation->findRecorder(name);
    return {};
}

RIndex RS_WorldLocalHandle::addController(const RSC_ControllerVariant& controller) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->addController(controller);
    return id;
}

RIndex RS_WorldLocalHandle::removeController(const std::string& name) {
    if (!waitForRunning()) return RS_NULL_INDEX;
    RIndex id = _simulation->allocateMessageId();
    _simulation->removeController(name);
    return id;
}

RSC_ControllerVariant RS_WorldLocalHandle::findController(const std::string& name) {
    if (waitForRunning()) return _simulation->findController(name);
    return {};
}

}  // namespace rise
