// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_rsc_registry.h"

#include "rise_origin/rsc/rsc_world.h"

namespace rise {

namespace rsc {

RSC_Base::RSCRef<RSC_Base>
RS_RSCRegistry::make(std::string_view type, const RS_RSCPropertyTreeReader &data) const {
    auto it = _read_map.find(std::string(type));
    if (it == _read_map.end()) {
        RS_THROW(RS_InvalidCharacterizationException("Unknown RSC type {}", type));
    }
    return it->second(data);
}

const std::string &RS_RSCRegistry::getName(const std::type_info &type) const {
    auto it = _name_map.find(std::type_index(type));
    if (it == _name_map.end()) {
        RS_THROW(RS_InvalidCharacterizationException("Unregistered RSC type {}", type.name()));
    }
    return it->second;
}

void RS_RSCRegistry::_registerAll() {
    // Resource
    _register<RSC_TextureConstant>("RSC_TextureConstant");
    _register<RSC_TextureGrid>("RSC_TextureGrid");

    // Essence
    _register<RSC_Material>("RSC_Material");
    _register<RSC_MaterialPhysicalPropertySoft>("RSC_MaterialPhysicalPropertySoft");
    _register<RSC_MaterialPhysicalPropertyFixed>("RSC_MaterialPhysicalPropertyFixed");
    _register<RSC_MaterialPhysicalPropertyRigid>("RSC_MaterialPhysicalPropertyRigid");
    _register<RSC_MaterialPhysicalPropertyAtlas>("RSC_MaterialPhysicalPropertyAtlas");
    _register<RSC_MaterialOpticalPropertySimpleMicrofacet>(
        "RSC_MaterialOpticalPropertySimpleMicrofacet"
    );
    _register<RSC_MaterialPairPropertyCoulombFriction>("RSC_MaterialPairPropertyCoulombFriction");
    _register<RSC_MaterialPairPropertyRestitution>("RSC_MaterialPairPropertyRestitution");

    // Entity
    _register<RSC_CosmicBackgroundUniform>("RSC_CosmicBackgroundUniform");
    _register<RSC_CosmicBackgroundGradient>("RSC_CosmicBackgroundGradient");
    _register<RSC_CosmicBackgroundHosekWilkie>("RSC_CosmicBackgroundHosekWilkie");
    _register<RSC_FloorFlat>("RSC_FloorFlat");
    _register<RSC_FloorArray>("RSC_FloorArray");
    _register<RSC_StructureVoxel>("RSC_StructureVoxel");
    _register<RSC_CameraPerspectiveFree>("RSC_CameraPerspectiveFree");
    _register<RSC_CameraPerspectiveRigidBodyLinked>("RSC_CameraPerspectiveRigidBodyLinked");

    // Record
    _register<RSC_IntersectRegionCuboid>("RSC_IntersectRegionCuboid");
    _register<RSC_IntersectRegionCuboidRigidBodyLinked>("RSC_IntersectRegionCuboidRigidBodyLinked");
    _register<RSC_IntersectRegionSphere>("RSC_IntersectRegionSphere");

    // Signal
    _register<RSC_Force>("RSC_Force");
    _register<RSC_Torque>("RSC_Torque");

    // Recorder
    _register<RSC_RecorderHDF5>("RSC_RecorderHDF5");

    // Controller
    _register<RSC_ControllerCallback>("RSC_ControllerCallback");

    // Simulation and World
    _register<RSC_Simulation>("RSC_Simulation");
    _register<RSC_World>("RSC_World");
}

}  // namespace rsc

}  // namespace rise
