// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_HANDLE_H
#define RS_WORLD_HANDLE_H

#include <string>

#include "rise_origin/rsc/rsc_world.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_world_def.h"

namespace rise {

/**
 * @brief Severity of a world message.
 * @details RSE_WORLD_MESSAGE_ANY_LEVEL is a wait-filter wildcard only; it is never emitted.
 */
RS_ENUM(
    RSE_WorldMessageLevel,
    RIndex,
    (RSE_WORLD_MESSAGE_ANY_LEVEL, 0),
    (RSE_WORLD_MESSAGE_INFO, 1),
    (RSE_WORLD_MESSAGE_WARNING, 2),
    (RSE_WORLD_MESSAGE_ERROR, 3)
);

/**
 * @brief Category of object a create operation produced.
 */
RS_ENUM(
    RSE_WorldMessageCreatedType,
    RIndex,
    (RSE_WORLD_MESSAGE_CREATED_TYPE_NONE, 0),
    (RSE_WORLD_MESSAGE_CREATED_TYPE_ENTITY, 1),
    (RSE_WORLD_MESSAGE_CREATED_TYPE_RESOURCE, 2),
    (RSE_WORLD_MESSAGE_CREATED_TYPE_ESSENCE, 3),
    (RSE_WORLD_MESSAGE_CREATED_TYPE_RECORD, 4),
    (RSE_WORLD_MESSAGE_CREATED_TYPE_SIGNAL, 5)
);

/**
 * @brief Free-form text body of a world message.
 */
struct RS_WorldMessageText {
    std::string text;
};

/**
 * @brief Body reporting the gid a world add operation assigned.
 */
struct RS_WorldMessageCreated {
    RSE_WorldMessageCreatedType type = RSE_WORLD_MESSAGE_CREATED_TYPE_NONE;
    RIndex gid = RS_NULL_INDEX;
    std::string name;
};

/**
 * @brief Typed body of a world message.
 */
using RS_WorldMessageData = RS_Variant<RS_WorldMessageText, RS_WorldMessageCreated>;

/**
 * @brief Asynchronous message emitted by a running world.
 * @details The id matches the RIndex returned by the mutator that originated the work.
 */
struct RS_WorldMessage {
    RIndex id = RS_NULL_INDEX;
    RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO;
    RS_WorldMessageData data;
};

/**
 * @brief Handle to a running world.
 * @details Exposes lifecycle control and name-keyed add/update/remove/find
 * operations over the six RSC category axes (entity, essence, resource,
 * controller, signal, recorder). An atomic sequence of mutations can be 
 * framed by wrapping with begin/endModificationTransaction.
 */
class RS_WorldHandle {
   public:
    virtual ~RS_WorldHandle() = default;

    /*****************************************************************
     * Identity
     *****************************************************************/
    /**
     * @brief Name of the world this handle controls.
     */
    virtual std::string name() = 0;

    /*****************************************************************
     * State
     *****************************************************************/
    /**
     * @brief True while the world is uninitialized or initializing.
     */
    virtual bool isPending() = 0;

    /**
     * @brief True while the world is running.
     */
    virtual bool isRunning() = 0;

    /**
     * @brief True while the world is paused.
     */
    virtual bool isPaused() = 0;

    /**
     * @brief True once the world has ended (terminated or errored).
     */
    virtual bool isEnd() = 0;

    /**
     * @brief Request the world pause.
     */
    virtual void pause() = 0;

    /**
     * @brief Request the world resume.
     */
    virtual void resume() = 0;

    /**
     * @brief Request the world terminate.
     */
    virtual void terminate() = 0;

    /**
     * @brief Current world status.
     */
    virtual RSE_WorldStatus getStatus() = 0;

    /**
     * @brief Block until the world is running; false if it ended first.
     */
    virtual bool waitForRunning() = 0;

    /**
     * @brief Block until the world ends.
     */
    virtual void waitForEnd() = 0;

    /*****************************************************************
     * Messages
     *****************************************************************/
    /**
     * @brief Block until a world message arrives.
     * @details message_id == RS_NULL_INDEX returns the next message matching the level filter;
     * otherwise blocks for the message with that id. RSE_WORLD_MESSAGE_ANY_LEVEL matches any severity;
     * a specific level matches severity >= level.
     */
    virtual RS_WorldMessage waitForMessage(
        RIndex message_id = RS_NULL_INDEX,
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_ANY_LEVEL
    ) = 0;

    /*****************************************************************
     * Modification transactions
     *****************************************************************/
    /**
     * @brief Open a modification transaction framing subsequent mutations.
     */
    virtual void beginModificationTransaction() = 0;

    /**
     * @brief Close the current modification transaction.
     */
    virtual void endModificationTransaction() = 0;

    /*****************************************************************
     * Root
     *****************************************************************/
    /**
     * @brief Snapshot the world as an RSC_World characterization.
     */
    virtual rsc::RSC_World getWorld() = 0;

    /**
     * @brief Apply a new simulation sub-characterization.
     * @return Message id to wait on.
     */
    virtual RIndex setSimulation(const rsc::RSC_Simulation& simulation) = 0;

    /*****************************************************************
     * Resource
     *****************************************************************/
    /**
     * @brief Add a resource.
     * @return Message id to wait on.
     */
    virtual RIndex addResource(const rsc::RSC_ResourceVariant& resource) = 0;

    /**
     * @brief Remove the named resource.
     * @return Message id to wait on.
     */
    virtual RIndex removeResource(const std::string& name) = 0;

    /**
     * @brief Re-emit the named resource's characterization.
     */
    virtual rsc::RSC_ResourceVariant findResource(const std::string& name) = 0;

    /*****************************************************************
     * Essence
     *****************************************************************/
    /**
     * @brief Add an essence.
     * @return Message id to wait on.
     */
    virtual RIndex addEssence(const rsc::RSC_EssenceVariant& essence) = 0;
    /**
     * @brief Update the named essence.
     * @return Message id to wait on.
     */
    virtual RIndex updateEssence(const std::string& name,
                                 const rsc::RSC_EssenceVariant& essence) = 0;

    /**
     * @brief Remove the named essence.
     * @return Message id to wait on.
     */
    virtual RIndex removeEssence(const std::string& name) = 0;

    /**
     * @brief Re-emit the named essence's characterization.
     */
    virtual rsc::RSC_EssenceVariant findEssence(const std::string& name) = 0;

    /*****************************************************************
     * Entity
     *****************************************************************/
    /**
     * @brief Add an entity.
     * @return Message id to wait on.
     */
    virtual RIndex addEntity(const rsc::RSC_EntityVariant& entity) = 0;
    /**
     * @brief Remove the named entity.
     * @return Message id to wait on.
     */
    virtual RIndex removeEntity(const std::string& name) = 0;

    /**
     * @brief Freeze the named entity so it stops simulating or rendering while staying present.
     * @return Message id to wait on.
     */
    virtual RIndex freezeEntity(const std::string& name) = 0;

    /**
     * @brief Unfreeze the named entity so it resumes simulating or rendering.
     * @return Message id to wait on.
     */
    virtual RIndex unfreezeEntity(const std::string& name) = 0;

    /**
     * @brief Re-emit the named entity's characterization.
     */
    virtual rsc::RSC_EntityVariant findEntity(const std::string& name) = 0;

    /**
     * @brief Whether the named entity is currently frozen.
     */
    virtual bool isEntityFrozen(const std::string& name) = 0;

    /*****************************************************************
     * Record
     *****************************************************************/
    /**
     * @brief Add a record.
     * @return Message id to wait on.
     */
    virtual RIndex addRecord(const rsc::RSC_RecordVariant& record) = 0;

    /**
     * @brief Remove the named record.
     * @return Message id to wait on.
     */
    virtual RIndex removeRecord(const std::string& name) = 0;

    /**
     * @brief Re-emit the named record's characterization.
     */
    virtual rsc::RSC_RecordVariant findRecord(const std::string& name) = 0;

    /*****************************************************************
     * Signal
     *****************************************************************/
    /**
     * @brief Add a signal.
     * @return Message id to wait on.
     */
    virtual RIndex addSignal(const rsc::RSC_SignalVariant& signal) = 0;

    /**
     * @brief Remove the named signal.
     * @return Message id to wait on.
     */
    virtual RIndex removeSignal(const std::string& name) = 0;

    /**
     * @brief Re-emit the named signal's characterization.
     */
    virtual rsc::RSC_SignalVariant findSignal(const std::string& name) = 0;

    /*****************************************************************
     * Recorder
     *****************************************************************/
    /**
     * @brief Add a recorder.
     * @return Message id to wait on.
     */
    virtual RIndex addRecorder(const rsc::RSC_RecorderVariant& recorder) = 0;

    /**
     * @brief Remove the named recorder.
     * @return Message id to wait on.
     */
    virtual RIndex removeRecorder(const std::string& name) = 0;

    /**
     * @brief Re-emit the named recorder's characterization.
     */
    virtual rsc::RSC_RecorderVariant findRecorder(const std::string& name) = 0;

    /*****************************************************************
     * Controller
     *****************************************************************/
    /**
     * @brief Add a controller.
     * @return Message id to wait on.
     */
    virtual RIndex addController(const rsc::RSC_ControllerVariant& controller) = 0;

    /**
     * @brief Remove the named controller.
     * @return Message id to wait on.
     */
    virtual RIndex removeController(const std::string& name) = 0;
    
    /**
     * @brief Re-emit the named controller's characterization.
     */
    virtual rsc::RSC_ControllerVariant findController(const std::string& name) = 0;
};

}  // namespace rise

#endif  // RS_WORLD_HANDLE_H
