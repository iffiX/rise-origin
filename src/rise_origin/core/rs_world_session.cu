// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <unordered_set>
#include <utility>
#include <vector>

#include "rise_origin/core/rs_world_session.h"
#include "rise_origin/core/rs_world_session_component.h"

namespace rise {

/*****************************************************************************
 * RS_WorldSessionHandle
 *****************************************************************************/
void RS_WorldSessionHandle::_requireOwned(
    const std::string &name,
    const std::unordered_set<std::string> &owned
) {
    if (!owned.count(name)) {
        RS_THROW(RS_WorldSessionScopeException(
            "'{}' was not created by this component, so it cannot be modified or deleted", name
        ));
    }
}

std::string RS_WorldSessionHandle::name() { return _raw->name(); }

std::string RS_WorldSessionHandle::getInfo(const std::string &info_name) {
    return _raw->getInfo(info_name);
}

bool RS_WorldSessionHandle::isPending() { return _raw->isPending(); }

bool RS_WorldSessionHandle::isRunning() { return _raw->isRunning(); }

bool RS_WorldSessionHandle::isPaused() { return _raw->isPaused(); }

bool RS_WorldSessionHandle::isEnd() { return _raw->isEnd(); }

void RS_WorldSessionHandle::pause() { _raw->pause(); }

void RS_WorldSessionHandle::resume() { _raw->resume(); }

void RS_WorldSessionHandle::terminate() { _raw->terminate(); }

RSE_WorldStatus RS_WorldSessionHandle::getStatus() { return _raw->getStatus(); }

bool RS_WorldSessionHandle::waitForRunning() { return _raw->waitForRunning(); }

void RS_WorldSessionHandle::waitForEnd() { _raw->waitForEnd(); }

RBigIndex RS_WorldSessionHandle::getNewestMessageIndex() { return _raw->getNewestMessageIndex(); }

RBigIndex RS_WorldSessionHandle::getOldestMessageIndex() { return _raw->getOldestMessageIndex(); }

std::vector<RS_WorldMessage> RS_WorldSessionHandle::getMessagesMatching(
    RSize max_count,
    RBigIndex oldest_index,
    RBigIndex newest_index,
    RSE_WorldMessageLevel level
) {
    return _raw->getMessagesMatching(max_count, oldest_index, newest_index, level);
}

RS_WorldMessage RS_WorldSessionHandle::waitForMessage(RSE_WorldMessageLevel level) {
    return _raw->waitForMessage(level);
}

RSize RS_WorldSessionHandle::getMessageBufferSize() { return _raw->getMessageBufferSize(); }

void RS_WorldSessionHandle::setMessageBufferSize(RSize max_messages) {
    _raw->setMessageBufferSize(max_messages);
}

void RS_WorldSessionHandle::beginModificationTransaction() { _raw->beginModificationTransaction(); }

void RS_WorldSessionHandle::endModificationTransaction() { _raw->endModificationTransaction(); }

rsc::RSC_World RS_WorldSessionHandle::getWorld() { return _raw->getWorld(); }

RS_WorldTask RS_WorldSessionHandle::setSimulation(const rsc::RSC_Simulation &simulation) {
    return _raw->setSimulation(simulation);
}

RS_WorldTask RS_WorldSessionHandle::addResource(const rsc::RSC_ResourceVariant &resource) {
    RS_WorldTask task = _raw->addResource(resource);
    _owned_resources.insert(_nameOf(resource));
    return task;
}

RS_WorldTask RS_WorldSessionHandle::removeResource(const std::string &name) {
    _requireOwned(name, _owned_resources);
    RS_WorldTask task = _raw->removeResource(name);
    _owned_resources.erase(name);
    return task;
}

rsc::RSC_ResourceVariant RS_WorldSessionHandle::findResource(const std::string &name) {
    return _raw->findResource(name);
}

RS_WorldTask RS_WorldSessionHandle::addEssence(const rsc::RSC_EssenceVariant &essence) {
    RS_WorldTask task = _raw->addEssence(essence);
    _owned_essences.insert(_nameOf(essence));
    return task;
}

RS_WorldTask RS_WorldSessionHandle::updateEssence(
    const std::string &name,
    const rsc::RSC_EssenceVariant &essence
) {
    _requireOwned(name, _owned_essences);
    return _raw->updateEssence(name, essence);
}

RS_WorldTask RS_WorldSessionHandle::removeEssence(const std::string &name) {
    _requireOwned(name, _owned_essences);
    RS_WorldTask task = _raw->removeEssence(name);
    _owned_essences.erase(name);
    return task;
}

rsc::RSC_EssenceVariant RS_WorldSessionHandle::findEssence(const std::string &name) {
    return _raw->findEssence(name);
}

RS_WorldTask RS_WorldSessionHandle::addEntity(const rsc::RSC_EntityVariant &entity) {
    RS_WorldTask task = _raw->addEntity(entity);
    _owned_entities.insert(_nameOf(entity));
    return task;
}

RS_WorldTask RS_WorldSessionHandle::removeEntity(const std::string &name) {
    _requireOwned(name, _owned_entities);
    RS_WorldTask task = _raw->removeEntity(name);
    _owned_entities.erase(name);
    return task;
}

RS_WorldTask RS_WorldSessionHandle::freezeEntity(const std::string &name) {
    _requireOwned(name, _owned_entities);
    return _raw->freezeEntity(name);
}

RS_WorldTask RS_WorldSessionHandle::unfreezeEntity(const std::string &name) {
    _requireOwned(name, _owned_entities);
    return _raw->unfreezeEntity(name);
}

rsc::RSC_EntityVariant RS_WorldSessionHandle::findEntity(const std::string &name) {
    return _raw->findEntity(name);
}

RIndex RS_WorldSessionHandle::findEntityGID(const std::string &name) {
    return _raw->findEntityGID(name);
}

std::string RS_WorldSessionHandle::findEntityName(RIndex gid) { return _raw->findEntityName(gid); }

bool RS_WorldSessionHandle::isEntityFrozen(const std::string &name) {
    return _raw->isEntityFrozen(name);
}

RS_WorldTask RS_WorldSessionHandle::addRecord(const rsc::RSC_RecordVariant &record) {
    RS_WorldTask task = _raw->addRecord(record);
    _owned_records.insert(_nameOf(record));
    return task;
}

RS_WorldTask RS_WorldSessionHandle::removeRecord(const std::string &name) {
    _requireOwned(name, _owned_records);
    RS_WorldTask task = _raw->removeRecord(name);
    _owned_records.erase(name);
    return task;
}

rsc::RSC_RecordVariant RS_WorldSessionHandle::findRecord(const std::string &name) {
    return _raw->findRecord(name);
}

RS_WorldTask RS_WorldSessionHandle::addSignal(const rsc::RSC_SignalVariant &signal) {
    RS_WorldTask task = _raw->addSignal(signal);
    _owned_signals.insert(_nameOf(signal));
    return task;
}

RS_WorldTask RS_WorldSessionHandle::removeSignal(const std::string &name) {
    _requireOwned(name, _owned_signals);
    RS_WorldTask task = _raw->removeSignal(name);
    _owned_signals.erase(name);
    return task;
}

rsc::RSC_SignalVariant RS_WorldSessionHandle::findSignal(const std::string &name) {
    return _raw->findSignal(name);
}

RS_WorldTask RS_WorldSessionHandle::addRecorder(const rsc::RSC_RecorderVariant &recorder) {
    RS_WorldTask task = _raw->addRecorder(recorder);
    _owned_recorders.insert(_nameOf(recorder));
    return task;
}

RS_WorldTask RS_WorldSessionHandle::removeRecorder(const std::string &name) {
    _requireOwned(name, _owned_recorders);
    RS_WorldTask task = _raw->removeRecorder(name);
    _owned_recorders.erase(name);
    return task;
}

rsc::RSC_RecorderVariant RS_WorldSessionHandle::findRecorder(const std::string &name) {
    return _raw->findRecorder(name);
}

RS_WorldTask RS_WorldSessionHandle::addController(const rsc::RSC_ControllerVariant &controller) {
    RS_WorldTask task = _raw->addController(controller);
    _owned_controllers.insert(_nameOf(controller));
    return task;
}

RS_WorldTask RS_WorldSessionHandle::removeController(const std::string &name) {
    _requireOwned(name, _owned_controllers);
    RS_WorldTask task = _raw->removeController(name);
    _owned_controllers.erase(name);
    return task;
}

rsc::RSC_ControllerVariant RS_WorldSessionHandle::findController(const std::string &name) {
    return _raw->findController(name);
}

/*****************************************************************************
 * RS_WorldSessionAccess
 *****************************************************************************/
std::vector<std::reference_wrapper<RS_WorldController>> RS_WorldSessionAccess::controllers() {
    std::vector<std::reference_wrapper<RS_WorldController>> mine;
    for (auto &controller : _controllers.get()) {
        if (_handle->_owned_controllers.count(controller.name)) mine.emplace_back(controller);
    }
    return mine;
}

RS_WorldController &RS_WorldSessionAccess::controller(const std::string &name) {
    if (!_handle->_owned_controllers.count(name)) {
        RS_THROW(
            RS_WorldSessionScopeException("controller '{}' is outside this component's scope", name)
        );
    }
    for (auto &controller : _controllers.get()) {
        if (controller.name == name) return controller;
    }
    RS_THROW(RS_WorldSessionScopeException(
        "controller '{}' is in this component's scope but absent from the current step", name
    ));
}

/*****************************************************************************
 * RS_WorldSession
 *****************************************************************************/
RS_WorldSession::RS_WorldSession(
    std::shared_ptr<RS_WorldHandle> handle,
    std::shared_ptr<RS_WorldCallback> callback
)
    : _raw(std::move(handle)), _callback(std::move(callback)) {
    _callback->setCallback([this](RBigFloat time, std::vector<RS_WorldController> controllers) {
        _onCallback(time, std::move(controllers));
    });
    _core_component = _makeCoreComponent();
}

RS_WorldSession::~RS_WorldSession() {
    // Detach components here so their threads and handles tear down before library unload at exit.
    _detachComponents();
    if (_callback) _callback->resetCallback();
}

bool RS_WorldSession::hasComponent(const std::string &name) const {
    return _components.count(name) != 0;
}

void RS_WorldSession::addComponent(
    const std::string &name,
    std::shared_ptr<RS_WorldSessionComponent> component
) {
    component->onAttach(_makeAccess());
    _components.emplace(name, std::move(component));
}

void RS_WorldSession::removeComponent(const std::string &name) {
    auto found = _components.find(name);
    if (found == _components.end()) {
        RS_THROW(RS_NotFoundException("Session component '{}' not found", name));
    }
    found->second->onDetach();
    _components.erase(found);
}

std::shared_ptr<RS_WorldSessionComponentGeneric> RS_WorldSession::getCoreComponent() {
    return _core_component;
}

RS_WorldSessionAccess RS_WorldSession::getAccess() { return _makeAccess(); }

std::shared_ptr<RS_WorldSessionHandle> RS_WorldSession::getHandle() {
    return std::make_shared<RS_WorldSessionHandle>(_raw);
}

void RS_WorldSession::run(bool is_blocking) {
    if (is_blocking) {
        _raw->waitForEnd();
        _detachComponents();
    }
}

RS_WorldSessionAccess RS_WorldSession::_makeAccess() {
    return RS_WorldSessionAccess(
        std::make_shared<RS_WorldSessionHandle>(_raw), _controllers, _broker
    );
}

std::shared_ptr<RS_WorldSessionComponentGeneric> RS_WorldSession::_makeCoreComponent() {
    rsc::RSC_World world = _raw->getWorld();
    auto getNames = [](const auto &items) {
        std::unordered_set<std::string> names;
        for (const auto &item : items) {
            names.insert(item.visit([](const auto &held) -> std::string { return held.name; }));
        }
        return names;
    };
    auto handle = std::make_shared<RS_WorldSessionHandle>(
        _raw,
        getNames(world.resources),
        getNames(world.essences),
        getNames(world.entities),
        getNames(world.records),
        getNames(world.signals),
        getNames(world.recorders),
        getNames(world.controllers)
    );
    auto core = std::make_shared<RS_WorldSessionComponentGeneric>();
    core->onAttach(RS_WorldSessionAccess(std::move(handle), _controllers, _broker));
    return core;
}

void RS_WorldSession::_detachComponents() {
    // Idempotent so run() and the destructor can both call it without detaching twice.
    if (_is_detached) return;
    _is_detached = true;
    _core_component->onDetach();
    for (auto &entry : _components) entry.second->onDetach();
}

void RS_WorldSession::_onCallback(RBigFloat time, std::vector<RS_WorldController> controllers) {
    _controllers = std::move(controllers);

    _core_component->onCallback(time, _pool);
    for (auto &entry : _components) entry.second->onCallback(time, _pool);
    _pool.wait();
}

}  // namespace rise
