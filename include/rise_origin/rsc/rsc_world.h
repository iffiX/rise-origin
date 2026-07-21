// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_WORLD_H
#define RSC_WORLD_H

#include <string>
#include <vector>

#include "rise_origin/utils/data/rs_variant.h"
#include "rsc_base.h"
#include "rsc_simulation.h"

/// Resources
#include "resource/rsc_texture.h"

/// Essences
#include "essence/rsc_material.h"
#include "essence/rsc_material_pair_property.h"
#include "essence/rsc_material_unary_property.h"

/// Entities
#include "entity/rsc_camera.h"
#include "entity/rsc_cosmic_background.h"
#include "entity/rsc_floor.h"
#include "entity/rsc_ray_sensor.h"
#include "entity/rsc_structure.h"

/// Records
#include "record/rsc_intersect_region.h"

/// Signals
#include "signal/rsc_mechanical.h"

/// I/O
#include "controller/rsc_controller.h"
#include "recorder/rsc_recorder.h"

namespace rise {

namespace rsc {

/**
 * @brief Resource variant
 * @details A resource is the source of data where we can pull from to create entities,
 * essences, etc.
 */
using RSC_ResourceVariant = RS_Variant<RSC_TextureConstant, RSC_TextureGrid>;

/**
 * @brief Essence variant
 * @details An essence in the world describes the properties of materials, compositions,
 * and interaction relationships between them.
 */
using RSC_EssenceVariant = RS_Variant<
    RSC_Material,
    RSC_MaterialPhysicalPropertySoft,
    RSC_MaterialPhysicalPropertyFixed,
    RSC_MaterialPhysicalPropertyRigid,
    RSC_MaterialPhysicalPropertyAtlas,
    RSC_MaterialOpticalPropertySimpleMicrofacet,
    RSC_MaterialPairPropertyCoulombFriction,
    RSC_MaterialPairPropertyRestitution>;

/**
 * @brief Entity variant
 * @details An entity in the world is an object that participates in the simulation.
 * It can be a structure, a camera, a floor, or a cosmic background, etc.
 */
using RSC_EntityVariant = RS_Variant<
    RSC_CosmicBackgroundUniform,
    RSC_CosmicBackgroundGradient,
    RSC_CosmicBackgroundHosekWilkie,
    RSC_FloorFlat,
    RSC_FloorArray,
    RSC_StructureVoxel,
    RSC_CameraPerspectiveFree,
    RSC_CameraPerspectiveRigidBodyLinked,
    RSC_CameraStereoFree,
    RSC_CameraStereoRigidBodyLinked,
    RSC_RaySensorFree,
    RSC_RaySensorRigidBodyLinked>;

/**
 * @brief Record variant
 * @details A record declares how to capture and process entity state so controllers and
 * recorders can consume it.
 */
using RSC_RecordVariant = RS_Variant<
    RSC_IntersectRegionCuboid,
    RSC_IntersectRegionCuboidRigidBodyLinked,
    RSC_IntersectRegionSphere,
    RSC_IntersectRegionSphereRigidBodyLinked>;

/**
 * @brief Signal variant
 * @details A signal is a per-step value an entity reads to drive its behavior (e.g. an applied
 * force or torque). Signals are produced by controllers.
 */
using RSC_SignalVariant = RS_Variant<RSC_Force, RSC_Torque, RSC_Transform>;

/**
 * @brief Recorder variant
 * @details A recorder is an I/O mechanism that records the state of the simulation.
 * It can be a recorder that records the state of the simulation to a file, a recorder that
 * records the state of the simulation to a database, a recorder that records the state of the
 * simulation to a network, etc.
 */
using RSC_RecorderVariant = RS_Variant<RSC_RecorderHDF5>;

/**
 * @brief Controller variant
 * @details A controller is an I/O mechanism that reads the state / controls the behavior of
 * an entity. It can be a controller that gets passed through the in-process world callback,
 * or over network etc.
 */
using RSC_ControllerVariant = RS_Variant<RSC_ControllerCallback>;

/**
 * @brief A world, top-level container of everything.
 */
class RSC_World : public RSC_Base {
   public:
    /// Human-readable world name.
    std::string name;

    /// Simulation-wide settings (time step, solver params, gravity, vision, etc.).
    RSCRef<RSC_Simulation> simulation = RSC_Simulation();

    /// Resources.
    RSCVec<RSC_ResourceVariant> resources;

    /// Essences.
    RSCVec<RSC_EssenceVariant> essences;

    /// Entities.
    RSCVec<RSC_EntityVariant> entities;

    /// Records.
    RSCVec<RSC_RecordVariant> records;

    /// Signals.
    RSCVec<RSC_SignalVariant> signals;

    /// Recorders.
    RSCVec<RSC_RecorderVariant> recorders;

    /// Controllers.
    RSCVec<RSC_ControllerVariant> controllers;

    RSC_World() = default;

    /**
     * @brief Assemble this world from saved RSC files (.rsct or .rscb), read in order.
     * @details Each file holds either a single component (resource, essence, entity, record,
     * signal, recorder, controller), which appends to its section, or a world, which appends its
     * components and replaces the simulation.
     * @throw RS_DataException when a path cannot be opened or parsed.
     */
    void assembleFrom(const std::vector<std::string>& paths);

    bool operator==(const RSC_World& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& reader) override;
    void _write(RS_RSCFileTreeWriter& writer) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
    void _deepCopyShared(MemoRef memo) override;
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_WORLD_H
