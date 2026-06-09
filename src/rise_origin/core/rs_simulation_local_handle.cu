// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rs_kernel_manager.h"
#include "rs_simulation_local_handle.h"
#include "rs_simulation_manager.h"

using namespace fmt;
using namespace std;

namespace rise {

RS_SimulationLocalHandle::RS_SimulationLocalHandle(const std::shared_ptr<RS_Simulation>& simulation)
    : _simulation(simulation) {}

bool RS_SimulationLocalHandle::isPending() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return status == RSE_SIMULATION_UNINITIALIZED || status == RSE_SIMULATION_INITIALIZING;
}

bool RS_SimulationLocalHandle::isRunning() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return status == RSE_SIMULATION_RUNNING;
}

bool RS_SimulationLocalHandle::isPaused() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return status == RSE_SIMULATION_PAUSED;
}

bool RS_SimulationLocalHandle::isEnd() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return isSimulationEnd(status);
}

void RS_SimulationLocalHandle::pause() {
    if (waitForRunning()) {
        _simulation->kernel_manager->pauseKernel();
    }
}

void RS_SimulationLocalHandle::resume() {
    if (waitForRunning()) {
        _simulation->kernel_manager->resumeKernel();
    }
}

void RS_SimulationLocalHandle::terminate() {
    if (waitForRunning()) {
        _simulation->kernel_manager->terminateKernel();
    }
}

RSE_SimulationStatus RS_SimulationLocalHandle::getStatus() {
    auto [status, status_info] = _simulation->getStatusAndStatusInfo();
    return status;
}

bool RS_SimulationLocalHandle::waitForRunning() {
    return _simulation->waitForStatus(RSE_SIMULATION_RUNNING);
}

void RS_SimulationLocalHandle::waitForEnd() { _simulation->waitForEnd(); }

void RS_SimulationLocalHandle::beginModificationTransaction() {
    if (waitForRunning()) {
        _simulation->kernel_manager->beginModificationTransaction();
    }
}

void RS_SimulationLocalHandle::endModificationTransaction() {
    if (waitForRunning()) {
        _simulation->kernel_manager->endModificationTransaction();
    }
}

void RS_SimulationLocalHandle::setSimulationConfig(const RS_SimulationConfig& config) {
    if (waitForRunning()) {
        config.validate();
        _simulation->kernel_manager->setSimulationConfig(config);
    }
}

void RS_SimulationLocalHandle::setWorld(const RS_WorldConfig& config) {
    if (waitForRunning()) {
        config.validate();
        _simulation->kernel_manager->setWorld(config);
    }
}

void RS_SimulationLocalHandle::setFloor(const RS_FloorConfig& config) {
    if (waitForRunning()) {
        config.validate();
        _simulation->kernel_manager->setFloor(config);
    }
}

void RS_SimulationLocalHandle::addMaterial(const RS_MaterialConfig& config) {
    if (waitForRunning()) {
        config.validate();
        _simulation->kernel_manager->addMaterial(config);
    }
}

void RS_SimulationLocalHandle::addRecord(const RS_RecordConfig& config) {
    if (waitForRunning()) {
        config.validate();
        _simulation->kernel_manager->addRecord(config);
    }
}

void RS_SimulationLocalHandle::addSignal(const RS_SignalConfig& config) {
    if (waitForRunning()) {
        config.validate();
        _simulation->kernel_manager->addSignal(config);
    }
}

void RS_SimulationLocalHandle::addStructure(const RS_StructureConfig& config) {
    if (waitForRunning()) {
        config.validate();
        _simulation->kernel_manager->addStructure(config);
    }
}

void RS_SimulationLocalHandle::addCamera(const RS_CameraConfig& config) {
    if (waitForRunning()) {
        config.validate();
        _simulation->kernel_manager->addCamera(config);
    }
}

void RS_SimulationLocalHandle::updateMaterial(const RS_MaterialConfig& config) {
    if (waitForRunning()) {
        config.validate();
        _simulation->kernel_manager->updateMaterial(config);
    }
}

void RS_SimulationLocalHandle::deleteFloor() {
    if (waitForRunning()) {
        _simulation->kernel_manager->deleteFloor();
    }
}

void RS_SimulationLocalHandle::deleteMaterial(RIndex id) {
    if (waitForRunning()) {
        _simulation->kernel_manager->deleteMaterial(id);
    }
}

void RS_SimulationLocalHandle::deleteRecord(RIndex id) {
    if (waitForRunning()) {
        _simulation->kernel_manager->deleteRecord(id);
    }
}

void RS_SimulationLocalHandle::deleteSignal(RIndex id) {
    if (waitForRunning()) {
        _simulation->kernel_manager->deleteSignal(id);
    }
}

void RS_SimulationLocalHandle::deleteStructure(RIndex id) {
    if (waitForRunning()) {
        _simulation->kernel_manager->deleteStructure(id);
    }
}

void RS_SimulationLocalHandle::deleteCamera(RIndex id) {
    if (waitForRunning()) {
        _simulation->kernel_manager->deleteCamera(id);
    }
}

RIndex RS_SimulationLocalHandle::findMaterial(const string& name) {
    if (waitForRunning()) {
        return _simulation->kernel_manager->findMaterial(name);
    }
    return RS_NULL_INDEX;
}

RIndex RS_SimulationLocalHandle::findRecord(const string& name) {
    if (waitForRunning()) {
        return _simulation->kernel_manager->findRecord(name);
    }
    return RS_NULL_INDEX;
}

RIndex RS_SimulationLocalHandle::findSignal(const string& name) {
    if (waitForRunning()) {
        return _simulation->kernel_manager->findSignal(name);
    }
    return RS_NULL_INDEX;
}

RIndex RS_SimulationLocalHandle::findStructure(const string& name) {
    if (waitForRunning()) {
        return _simulation->kernel_manager->findStructure(name);
    }
    return RS_NULL_INDEX;
}

RIndex RS_SimulationLocalHandle::findCamera(const string& name) {
    if (waitForRunning()) {
        return _simulation->kernel_manager->findCamera(name);
    }
    return RS_NULL_INDEX;
}

RS_Config RS_SimulationLocalHandle::getConfig() {
    if (waitForRunning()) {
        return _simulation->kernel_manager->getConfig();
    }
    return {};
}

RS_SimulationConfig RS_SimulationLocalHandle::getSimulationConfig() {
    if (waitForRunning()) {
        return _simulation->kernel_manager->getSimulationConfig();
    }
    return {};
}

RS_FloorConfig RS_SimulationLocalHandle::getFloorConfig() {
    if (waitForRunning()) {
        return _simulation->kernel_manager->getFloorConfig();
    }
    return {};
}

vector<RS_MaterialConfig> RS_SimulationLocalHandle::getMaterialConfigs() {
    if (waitForRunning()) {
        return _simulation->kernel_manager->getMaterialConfigs();
    }
    return {};
}

vector<RS_RecordConfig> RS_SimulationLocalHandle::getRecordConfigs() {
    if (waitForRunning()) {
        return _simulation->kernel_manager->getRecordConfigs();
    }
    return {};
}

vector<RS_SignalConfig> RS_SimulationLocalHandle::getSignalConfigs() {
    if (waitForRunning()) {
        return _simulation->kernel_manager->getSignalConfigs();
    }
    return {};
}

vector<RS_StructureConfig> RS_SimulationLocalHandle::getStructureConfigs() {
    if (waitForRunning()) {
        return _simulation->kernel_manager->getStructureConfigs();
    }
    return {};
}

vector<RS_CameraConfig> RS_SimulationLocalHandle::getCameraConfigs() {
    if (waitForRunning()) {
        return _simulation->kernel_manager->getCameraConfigs();
    }
    return {};
}

}  // namespace rise