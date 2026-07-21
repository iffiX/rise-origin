// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_rsc_registry.h"

#include "rise_origin/rsc/rsc_world.h"

namespace rise {

namespace rsc {

RSC_Base::RSCRef<RSC_Base>
RS_RSCRegistry::make(std::string_view type, const RS_RSCFileTreeReader &data) {
    return getInstance()._make(type, data);
}

const std::string &RS_RSCRegistry::getName(const std::type_info &type) {
    return getInstance()._getName(type);
}

std::map<std::string, std::vector<std::string>> RS_RSCRegistry::getCategory() {
    return getInstance()._getCategory();
}

RSC_Base::RSCRef<RSC_Base> RS_RSCRegistry::getExample(std::string_view type) {
    return getInstance()._getExample(type);
}

RSC_Base::RSCRef<RSC_Base>
RS_RSCRegistry::_make(std::string_view type, const RS_RSCFileTreeReader &data) const {
    auto it = _read_map.find(std::string(type));
    if (it == _read_map.end()) {
        RS_THROW(RS_InvalidCharacterizationException("Unknown RSC type {}", type));
    }
    return it->second(data);
}

const std::string &RS_RSCRegistry::_getName(const std::type_info &type) const {
    auto it = _name_map.find(std::type_index(type));
    if (it == _name_map.end()) {
        RS_THROW(RS_InvalidCharacterizationException("Unregistered RSC type {}", type.name()));
    }
    return it->second;
}

std::map<std::string, std::vector<std::string>> RS_RSCRegistry::_getCategory() const {
    return _category_map;
}

RSC_Base::RSCRef<RSC_Base> RS_RSCRegistry::_getExample(std::string_view type) const {
    auto it = _make_map.find(std::string(type));
    if (it == _make_map.end()) {
        RS_THROW(RS_InvalidCharacterizationException("Unknown RSC type {}", type));
    }
    return it->second();
}

void RS_RSCRegistry::_registerAll() {
    // Embedded
    _register<RSC_StructureVoxelBody>("RSC_StructureVoxelBody", "embedded");
    _register<RSC_StructureVoxelConstraint>("RSC_StructureVoxelConstraint", "embedded");
    _register<RSC_StructureVoxelRecord>("RSC_StructureVoxelRecord", "embedded");
    _register<RSC_StructureVoxelSignal>("RSC_StructureVoxelSignal", "embedded");
    _register<RSC_CameraPerspectiveRecord>("RSC_CameraPerspectiveRecord", "embedded");
    _register<RSC_CameraPerspectiveSignal>("RSC_CameraPerspectiveSignal", "embedded");
    _register<RSC_CameraStereoRecord>("RSC_CameraStereoRecord", "embedded");
    _register<RSC_CameraStereoSignal>("RSC_CameraStereoSignal", "embedded");
    _register<RSC_RaySensorRecord>("RSC_RaySensorRecord", "embedded");
    _register<RSC_RaySensorSignal>("RSC_RaySensorSignal", "embedded");

    // Resource
    _register<RSC_TextureConstant>("RSC_TextureConstant", "resource");
    _register<RSC_TextureGrid>("RSC_TextureGrid", "resource");

    // Essence
    _register<RSC_Material>("RSC_Material", "essence");
    _register<RSC_MaterialPhysicalPropertySoft>("RSC_MaterialPhysicalPropertySoft", "essence");
    _register<RSC_MaterialPhysicalPropertyFixed>("RSC_MaterialPhysicalPropertyFixed", "essence");
    _register<RSC_MaterialPhysicalPropertyRigid>("RSC_MaterialPhysicalPropertyRigid", "essence");
    _register<RSC_MaterialPhysicalPropertyAtlas>("RSC_MaterialPhysicalPropertyAtlas", "essence");
    _register<RSC_MaterialOpticalPropertySimpleMicrofacet>(
        "RSC_MaterialOpticalPropertySimpleMicrofacet", "essence"
    );
    _register<RSC_MaterialPairPropertyCoulombFriction>(
        "RSC_MaterialPairPropertyCoulombFriction", "essence"
    );
    _register<RSC_MaterialPairPropertyRestitution>(
        "RSC_MaterialPairPropertyRestitution", "essence"
    );

    // Entity
    _register<RSC_CosmicBackgroundUniform>("RSC_CosmicBackgroundUniform", "entity");
    _register<RSC_CosmicBackgroundGradient>("RSC_CosmicBackgroundGradient", "entity");
    _register<RSC_CosmicBackgroundHosekWilkie>("RSC_CosmicBackgroundHosekWilkie", "entity");
    _register<RSC_FloorFlat>("RSC_FloorFlat", "entity");
    _register<RSC_FloorArray>("RSC_FloorArray", "entity");
    _register<RSC_StructureVoxel>("RSC_StructureVoxel", "entity");
    _register<RSC_CameraPerspectiveFree>("RSC_CameraPerspectiveFree", "entity");
    _register<RSC_CameraPerspectiveRigidBodyLinked>(
        "RSC_CameraPerspectiveRigidBodyLinked", "entity"
    );
    _register<RSC_CameraStereoFree>("RSC_CameraStereoFree", "entity");
    _register<RSC_CameraStereoRigidBodyLinked>("RSC_CameraStereoRigidBodyLinked", "entity");
    _register<RSC_RaySensorFree>("RSC_RaySensorFree", "entity");
    _register<RSC_RaySensorRigidBodyLinked>("RSC_RaySensorRigidBodyLinked", "entity");

    // Record
    _register<RSC_IntersectRegionCuboid>("RSC_IntersectRegionCuboid", "record");
    _register<RSC_IntersectRegionCuboidRigidBodyLinked>(
        "RSC_IntersectRegionCuboidRigidBodyLinked", "record"
    );
    _register<RSC_IntersectRegionSphere>("RSC_IntersectRegionSphere", "record");
    _register<RSC_IntersectRegionSphereRigidBodyLinked>(
        "RSC_IntersectRegionSphereRigidBodyLinked", "record"
    );

    // Signal
    _register<RSC_Force>("RSC_Force", "signal");
    _register<RSC_Torque>("RSC_Torque", "signal");
    _register<RSC_Transform>("RSC_Transform", "signal");

    // Recorder
    _register<RSC_RecorderHDF5>("RSC_RecorderHDF5", "recorder");

    // Controller
    _register<RSC_ControllerCallback>("RSC_ControllerCallback", "controller");

    // Simulation and World
    _register<RSC_Simulation>("RSC_Simulation", "simulation");
    _register<RSC_World>("RSC_World", "world");
}

}  // namespace rsc

}  // namespace rise
