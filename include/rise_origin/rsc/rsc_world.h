// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_WORLD_H
#define RSC_WORLD_H

#include <string>
#include <vector>

#include "rise_origin/utils/serialization/rs_mp_serialization.h"
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
    RSC_CameraPerspectiveRigidBodyLinked>;

/**
 * @brief Record variant
 * @details A record declares how to capture and process entity state so controllers and
 * recorders can consume it.
 */
using RSC_RecordVariant = RS_Variant<
    RSC_IntersectRegionCuboid,
    RSC_IntersectRegionCuboidRigidBodyLinked,
    RSC_IntersectRegionSphere>;

/**
 * @brief Signal variant
 * @details A signal is a per-step value an entity reads to drive its behavior (e.g. an applied
 * force or torque). Signals are produced by controllers.
 */
using RSC_SignalVariant = RS_Variant<RSC_Force, RSC_Torque>;

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

    /// Validation state (populated by validate()).
    bool is_valid = false;
    std::string invalid_reason;

    /// Simulation-wide settings (time step, solver params, gravity, vision, etc.).
    RSC_Simulation simulation;

    /// Resources.
    std::vector<RSC_ResourceVariant> resources;

    /// Essences.
    std::vector<RSC_EssenceVariant> essences;

    /// Entities.
    std::vector<RSC_EntityVariant> entities;

    /// Records.
    std::vector<RSC_RecordVariant> records;

    /// Signals.
    std::vector<RSC_SignalVariant> signals;

    /// Recorders.
    std::vector<RSC_RecorderVariant> recorders;

    /// Controllers.
    std::vector<RSC_ControllerVariant> controllers;

    RSC_World() = default;

    /**
     * @brief Construct from a vector of XML characterization strings.
     * @param rsc_strings Later strings override RSC_Simulation of earlier ones, and append
     * to other sections of earlier ones.
     * @param throw_parse_error When false, validation errors populate is_valid / invalid_reason
     * instead of throwing.
     */
    explicit RSC_World(const std::vector<std::string>& rsc_strings, bool throw_parse_error = true);

    /**
     * @brief Load from XML file paths. Same override semantics as the string constructor.
     * @throw RS_DataException if the RSC file paths are invalid, it is not affected by
     * throw_parse_error.
     * @param throw_parse_error When false, validation errors populate is_valid / invalid_reason
     * instead of throwing.
     */
    void open(const std::vector<std::string>& rsc_paths, bool throw_parse_error = true);

    /**
     * @brief Validate the world characterization. When throw_error is true, throws on error;
     * otherwise stores validity in is_valid / invalid_reason.
     */
    void validate(bool throw_error = true);

    bool operator==(const RSC_World& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        s("name", name);
        s("is_valid", is_valid);
        s("invalid_reason", invalid_reason);
        s("simulation", simulation);
        s("resources", resources);
        s("essences", essences);
        s("entities", entities);
        s("records", records);
        s("signals", signals);
        s("recorders", recorders);
        s("controllers", controllers);
    }

   protected:
    void _validate() const override;
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;

   private:
    std::vector<std::string> _rsc_strings;
    void _parseSettings();
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_WORLD_H
