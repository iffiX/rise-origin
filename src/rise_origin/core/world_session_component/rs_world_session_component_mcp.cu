// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_world_session_component.h"

#ifdef RISE_ORIGIN_BUILD_CORE_WITH_MCP

#include <httplib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/rsc/controller/rsc_controller.h"
#include "rise_origin/rsc/entity/rsc_camera.h"
#include "rise_origin/rsc/entity/rsc_structure.h"
#include "rise_origin/rsc/essence/rsc_material.h"
#include "rise_origin/rsc/essence/rsc_material_unary_property.h"
#include "rise_origin/rsc/record/rsc_intersect_region.h"
#include "rise_origin/rsc/resource/rsc_texture.h"
#include "rise_origin/rsc/rs_rsc_archive.h"
#include "rise_origin/rsc/rs_rsc_file.h"
#include "rise_origin/rsc/rs_rsc_registry.h"
#include "rise_origin/rsc/rsc_world.h"
#include "rise_origin/rsc/signal/rsc_mechanical.h"
#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/rs_logging.h"
#include "rs_world_session_component_vr_data.h"

namespace rise {

using namespace std::chrono_literals;

// Default MCP protocol revision.
constexpr const char *RS_MCP_PROTOCOL_VERSION = "2025-11-25";

// How long a request thread waits for its job before it returns an id to probe.
constexpr auto RS_MCP_INLINE_WAIT = 20ms;

// Names of the structure and the mounted eye that stand in for the AI's body.
constexpr const char *RS_MCP_BODY_NAME = "mcp.body";
constexpr const char *RS_MCP_BODY_EYE_NAME = "mcp.body.eye";

// True when the held type carries a name field, so a nameless part is summarized by type alone.
template <typename, typename = void>
struct RS_MCPHasName : std::false_type {};
template <typename T>
struct RS_MCPHasName<T, std::void_t<decltype(std::declval<const T &>().name)>> : std::true_type {};

/*****************************************************************
 * RS_MCPImpl
 *****************************************************************/
/**
 * @brief The http server and tool state behind the MCP component's pimpl.
 * @details A listen thread owns the http server and answers Model Context Protocol requests. Each
 * tool call is handed to a single executor thread and gets an id at once; the request thread waits
 * a few milliseconds for a fast answer, otherwise it returns the id to probe with get_result. A
 * step task, syncControllers, writes the buffered control signals into their live signals and reads
 * out pending capture records. Every tool is one APIBase struct; those live in their own section at
 * the end and hold all tool behavior, while the methods above them are pure helpers.
 */
class RS_MCPImpl {
   public:
    RS_MCPImpl(
        RS_WorldSessionAccess access,
        std::string host,
        int port,
        std::string token,
        bool is_python_exec_enabled
    )
        : _access(std::move(access)),
          _host(std::move(host)),
          _port(port),
          _token(std::move(token)),
          _is_python_exec_enabled(is_python_exec_enabled) {
        if (logging::get("root")) _logger = logging::get("root")->clone("mcp service");
        _subscribeUserChannels();
        _registerBasicApis();
        _registerConvenienceApis();
        _registerEmbodimentApis();
        _registerControlApis();
        _registerCommunicationApis();
        _server.Post(
            "/mcp",
            httplib::Server::Handler(
                std::bind(
                    &RS_MCPImpl::_handleRequest, this, std::placeholders::_1, std::placeholders::_2
                )
            )
        );
        _is_alive = true;
        _execution_thread = std::thread([this] { _executorThreadFunc(); });
        _listen_thread = std::thread([this] { _listenThreadFunc(); });
    }

    ~RS_MCPImpl() {
        _is_alive = false;
        _server.stop();
        _exec_cv.notify_all();
        if (_listen_thread.joinable()) _listen_thread.join();
        if (_execution_thread.joinable()) _execution_thread.join();
    }

    /**
     * @brief Store a Python authored tool served next to the built in ones.
     */
    void registerTool(
        std::string name,
        std::string description,
        std::string input_schema,
        RS_MCPToolHandler handler
    ) {
        std::scoped_lock lock(_api_mutex);
        _apis[name] = std::make_unique<RegisteredTool>(
            *this, name, std::move(description), std::move(input_schema), std::move(handler)
        );
    }

    /**
     * @brief Drop a registered tool by name.
     */
    void unregisterTool(const std::string &name) {
        std::scoped_lock lock(_api_mutex);
        auto found = _apis.find(name);
        if (found != _apis.end() && found->second->isRegistered()) _apis.erase(found);
    }

    /**
     * @brief Get the names of every registered tool.
     */
    std::vector<std::string> getRegisteredToolNames() const {
        std::scoped_lock lock(_api_mutex);
        std::vector<std::string> names;
        for (const auto &entry : _apis) {
            if (entry.second->isRegistered()) names.push_back(entry.first);
        }
        return names;
    }

    /**
     * @brief Write the buffered control signals into their live signals and read out pending
     * capture records; dispatched onto the session pool each control step.
     */
    void syncControllers() {
        _advanceCubeGlides();
        _advanceCubeAutoMotion();
        _applyControls();
        _readCaptures();
        _readCubeReads();
    }

   private:
    /*****************************************************************
     * API utilities
     *****************************************************************/
    enum class APITaskKind { IMMEDIATE, WORLD, FUTURE };

    struct APITaskState {
        std::mutex mutex;
        std::condition_variable cv;
        bool is_produced = false;
        APITaskKind kind = APITaskKind::IMMEDIATE;
        RS_JSON immediate;
        RS_WorldTask world;
        std::shared_future<RS_JSON> future;
    };

    class APITask {
       public:
        static APITask makeFinished(RS_JSON result) {
            APITask task;
            task._set(APITaskKind::IMMEDIATE, std::move(result), {}, {});
            return task;
        }
        static APITask fromWorldTask(RS_WorldTask world) {
            APITask task;
            task._set(APITaskKind::WORLD, {}, std::move(world), {});
            return task;
        }
        static APITask fromFuture(std::shared_future<RS_JSON> future) {
            APITask task;
            task._set(APITaskKind::FUTURE, {}, {}, std::move(future));
            return task;
        }

        void complete(const APITask &produced) {
            std::scoped_lock produced_lock(produced._state->mutex);
            {
                std::scoped_lock lock(_state->mutex);
                _state->kind = produced._state->kind;
                _state->immediate = produced._state->immediate;
                _state->world = produced._state->world;
                _state->future = produced._state->future;
                _state->is_produced = true;
            }
            _state->cv.notify_all();
        }

        bool waitFor(std::chrono::milliseconds timeout) const {
            std::unique_lock lock(_state->mutex);
            _state->cv.wait_for(lock, timeout, [this] { return _state->is_produced; });
            return _isFinished();
        }

        [[nodiscard]] bool isFinished() const {
            std::scoped_lock lock(_state->mutex);
            return _isFinished();
        }

        [[nodiscard]] RS_JSON get() const {
            std::scoped_lock lock(_state->mutex);
            if (_state->kind == APITaskKind::IMMEDIATE) return _state->immediate;
            if (_state->kind == APITaskKind::WORLD) return _makeWorldResult(_state->world);
            return _state->future.get();
        }

       private:
        std::shared_ptr<APITaskState> _state = std::make_shared<APITaskState>();

        void _set(
            APITaskKind kind,
            RS_JSON immediate,
            RS_WorldTask world,
            std::shared_future<RS_JSON> future
        ) {
            _state->kind = kind;
            _state->immediate = std::move(immediate);
            _state->world = std::move(world);
            _state->future = std::move(future);
            _state->is_produced = true;
        }

        [[nodiscard]] bool _isFinished() const {
            if (!_state->is_produced) return false;
            if (_state->kind == APITaskKind::IMMEDIATE) return true;
            if (_state->kind == APITaskKind::WORLD) return _state->world.isFinished();
            return _state->future.valid() &&
                   _state->future.wait_for(0s) == std::future_status::ready;
        }
    };

    /**
     * @brief The base of every tool: it names itself, shows its schema, and runs on the executor.
     * @details Nested structs reach the impl's private helpers and state through the stored
     * reference.
     */
    struct APIBase {
        RS_MCPImpl &impl;
        explicit APIBase(RS_MCPImpl &impl) : impl(impl) {}
        virtual ~APIBase() = default;
        [[nodiscard]] virtual std::string name() const = 0;
        [[nodiscard]] virtual std::string description() const = 0;
        [[nodiscard]] virtual std::string schema() const = 0;
        [[nodiscard]] virtual bool isRegistered() const = 0;
        virtual APITask call(const RS_JSON &params) = 0;
    };

    /*****************************************************************
     * Control and capture buffers
     *****************************************************************/
    /**
     * @brief A one-shot pose kick waiting to be written into a structure's transform signal.
     */
    struct PendingTransform {
        RVec3rf translation;
        RQuat3rf rotation;
        RVec3rf pivot;
    };

    /**
     * @brief A camera pose waiting to be written into a free camera's signal.
     */
    struct PendingCameraPose {
        RVec3rf position;
        RQuat3rf orientation;
    };

    /**
     * @brief A joint target waiting to be written into a structure's voxel signal.
     */
    struct PendingJoint {
        RIndex index;
        RFloat target;
        bool is_prismatic;
    };

    /**
     * @brief A capture in flight: unfreeze the camera, gather frames over the next steps, refreeze.
     */
    struct CaptureRequest {
        std::string camera;
        std::string controller;
        RSize remaining;
        RSize stride;
        RSize countdown;
        bool was_frozen;
        std::vector<RS_JSON> frames;
        std::shared_ptr<std::promise<RS_JSON>> promise;
    };

    /**
     * @brief Frames a run of world mutations so they land atomically off the control step.
     */
    struct Transaction {
        RS_WorldSessionHandle &handle;
        explicit Transaction(RS_WorldSessionHandle &handle) : handle(handle) {
            handle.beginModificationTransaction();
        }
        ~Transaction() { handle.endModificationTransaction(); }
    };

    /*****************************************************************
     * Signal writers and record readers
     *****************************************************************/
    /**
     * @brief Write one buffered pose kick into a structure's transform signal.
     */
    struct TransformWriter {
        const PendingTransform &value;
        template <typename Signal>
        void operator()(Signal &signal) const {
            if constexpr (std::is_same_v<Signal, RS_WorldTransformSignal>) {
                if (signal.translation_signals.object_num > 0 &&
                    signal.translation_signals.translation) {
                    signal.translation_signals.translation[0] = value.translation;
                }
                if (signal.rotation_signals.object_num > 0 && signal.rotation_signals.rotation) {
                    signal.rotation_signals.rotation[0] = value.rotation;
                    signal.rotation_signals.pivot[0] = value.pivot;
                }
            }
        }
    };

    /**
     * @brief Write one buffered force vector into a structure's force signal.
     */
    struct ForceWriter {
        const RVec3rf &force;
        template <typename Signal>
        void operator()(Signal &signal) const {
            if constexpr (std::is_same_v<Signal, RS_WorldForceSignal>) {
                if (signal.force_signals.object_num > 0 && signal.force_signals.force) {
                    signal.force_signals.force[0] = force;
                }
            }
        }
    };

    /**
     * @brief Write one buffered torque vector into a structure's torque signal.
     */
    struct TorqueWriter {
        const RVec3rf &torque;
        template <typename Signal>
        void operator()(Signal &signal) const {
            if constexpr (std::is_same_v<Signal, RS_WorldTorqueSignal>) {
                if (signal.torque_signals.object_num > 0 && signal.torque_signals.torque) {
                    signal.torque_signals.torque[0] = torque;
                }
            }
        }
    };

    /**
     * @brief Write the buffered joint targets into a structure's voxel signal.
     */
    struct JointWriter {
        const std::vector<PendingJoint> &joints;
        template <typename Signal>
        void operator()(Signal &signal) const {
            if constexpr (std::is_same_v<Signal, RS_WorldStructureVoxelSignal>) {
                for (const PendingJoint &joint : joints) {
                    if (joint.is_prismatic) {
                        if (signal.prismatic_signals.object_num > joint.index &&
                            signal.prismatic_signals.prismatic) {
                            signal.prismatic_signals.prismatic[joint.index] = joint.target;
                        }
                    } else if (signal.hinge_signals.object_num > joint.index &&
                               signal.hinge_signals.hinge) {
                        signal.hinge_signals.hinge[joint.index] = joint.target;
                    }
                }
            }
        }
    };

    /**
     * @brief Write one buffered pose into a free camera's pose signal.
     */
    struct CameraPoseWriter {
        const PendingCameraPose &pose;
        template <typename Signal>
        void operator()(Signal &signal) const {
            if constexpr (std::is_same_v<Signal, RS_WorldCameraPerspectiveSignal>) {
                if (signal.position_signals.object_num > 0 && signal.position_signals.position) {
                    signal.position_signals.position[0] = pose.position;
                }
                if (signal.orientation_signals.object_num > 0 &&
                    signal.orientation_signals.orientation) {
                    signal.orientation_signals.orientation[0] = pose.orientation;
                }
            }
        }
    };

    /**
     * @brief Read the first valid perspective frame out of a controller's records.
     */
    struct CaptureReader {
        bool is_valid = false;
        RSize width = 0;
        RSize height = 0;
        const RColor3b *pixels = nullptr;
        template <typename Record>
        void operator()(Record &record) {
            if constexpr (std::is_same_v<Record, RS_WorldCameraPerspectiveRecord>) {
                if (is_valid || !record.isValid() || record.rgb.pixel == nullptr) return;
                is_valid = true;
                width = record.width;
                height = record.height;
                pixels = record.rgb.pixel;
            }
        }
    };

    /**
     * @brief Turn one part of the held type into a name and a registered type string.
     */
    struct PartSummary {
        template <typename T>
        RS_JSON operator()(const T &held) const {
            RS_JSON part;
            part["type"] = rsc::RS_RSCRegistry::getName(typeid(T));
            if constexpr (RS_MCPHasName<T>::value) part["name"] = held.name;
            return part;
        }
    };

    /**
     * @brief Read a camera's driving controller from its record routing reference.
     */
    struct CameraControllerReader {
        std::string controller;
        template <typename Entity>
        void operator()(const Entity &entity) {
            if constexpr (std::is_base_of_v<rsc::RSC_CameraPerspectiveBase, Entity>) {
                controller = entity.record->target_controller_reference;
            }
        }
    };

    /**
     * @brief Read a voxel structure's driving controller from its signal reference.
     */
    struct VoxelControllerReader {
        std::string controller;
        template <typename Entity>
        void operator()(const Entity &entity) {
            if constexpr (std::is_same_v<Entity, rsc::RSC_StructureVoxel>) {
                controller = entity.signal->source_controller_reference;
            }
        }
    };

    /*****************************************************************
     * Embodiment
     *****************************************************************/
    /**
     * @brief The cube's persistent auto-motion mode, held until a manual verb cancels it.
     */
    enum class CubeAutoMotionMode { NONE, TRACK, ORBIT };

    /**
     * @brief Which live state a pending cube read wants back.
     */
    enum class CubeReadKind { POSE, BOUNDS, SURROUNDING };

    /**
     * @brief The cube body's live world-space pose read from its rigid body record.
     */
    struct CubePose {
        bool is_valid = false;
        RVec3rf com;
        RQuat3rf orientation;
        RVec3rf linear_velocity;
        RVec3rf angular_velocity;
    };

    /**
     * @brief One structure sensed by the body's proximity region this step.
     */
    struct CubeSurroundingEntry {
        RIndex gid;
        RVec3rf closest_offset;
    };

    /**
     * @brief A smooth move of the cube toward a target pose over a span of real seconds.
     * @details Timing is wall-clock and tick-gated: it advances only on sync steps, so a paused
     * world neither stalls it nor makes it jump. A zero duration lands in one step.
     */
    struct CubeGlidePlan {
        bool is_look;
        RVec3rf point;
        RFloat duration;
        bool has_start = false;
        RVec3rf start_position;
        RQuat3rf start_orientation;
        RVec3rf target_position;
        RQuat3rf target_orientation;
        RFloat elapsed = 0.0f;
        std::chrono::steady_clock::time_point last_time;
        std::shared_ptr<std::promise<RS_JSON>> promise;
    };

    /**
     * @brief The state that drives an auto-follow or an auto-orbit each sync step.
     */
    struct CubeAutoMotion {
        CubeAutoMotionMode mode = CubeAutoMotionMode::NONE;
        RIndex target_gid = RS_NULL_INDEX;
        std::string target_name;
        RFloat distance = 0.0f;
        RVec3rf point;
        RFloat radius = 0.0f;
        RFloat height = 0.0f;
        RFloat period = 0.0f;
        RFloat angle = 0.0f;
        bool has_last_time = false;
        std::chrono::steady_clock::time_point last_time;
    };

    /**
     * @brief A pending read of the cube's live state, fulfilled from the record on a sync step.
     */
    struct CubeReadRequest {
        CubeReadKind kind;
        std::shared_ptr<std::promise<RS_JSON>> promise;
    };

    /**
     * @brief Read the cube's live pose out of its rigid body record.
     */
    struct CubeRigidBodyReader {
        CubePose pose;
        template <typename Record>
        void operator()(Record &record) {
            if constexpr (std::is_same_v<Record, RS_WorldStructureVoxelRecord>) {
                if (pose.is_valid || !record.isValid()) return;
                if (record.rigid_bodies.object_num == 0 || record.rigid_bodies.com == nullptr)
                    return;
                pose.is_valid = true;
                pose.com = record.rigid_bodies.com[0];
                pose.orientation = record.rigid_bodies.orientation[0];
                pose.linear_velocity = record.rigid_bodies.linear_velocity[0];
                pose.angular_velocity = record.rigid_bodies.angular_velocity[0];
            }
        }
    };

    /**
     * @brief Read the structures inside the body's proximity region out of its intersect record.
     */
    struct CubeSurroundingReader {
        bool is_valid = false;
        std::vector<CubeSurroundingEntry> entries;
        template <typename Record>
        void operator()(Record &record) {
            if constexpr (std::is_same_v<Record, RS_WorldIntersectRegionRecord>) {
                if (is_valid || !record.isValid() || record.memberships.structure_gid == nullptr)
                    return;
                is_valid = true;
                for (RBigSize i = 0; i < record.memberships.object_num; ++i) {
                    CubeSurroundingEntry entry;
                    entry.gid = record.memberships.structure_gid[i];
                    entry.closest_offset = record.memberships.closest_relative_position[i];
                    entries.push_back(entry);
                }
            }
        }
    };

    /*****************************************************************
     * Members
     *****************************************************************/
    RS_WorldSessionAccess _access;
    std::string _host;
    int _port;
    std::string _token;
    bool _is_python_exec_enabled;

    std::shared_ptr<logging::logger> _logger;

    httplib::Server _server;
    std::atomic<bool> _is_alive{false};
    std::thread _listen_thread;
    std::thread _execution_thread;

    // The whole tool surface, built ins and registered, keyed by name.
    mutable std::mutex _api_mutex;
    std::unordered_map<std::string, std::unique_ptr<APIBase>> _apis;

    // The executor's work queue and the id keyed result table it fills.
    std::mutex _exec_mutex;
    std::condition_variable _exec_cv;
    std::deque<std::tuple<APIBase *, RS_JSON, APITask>> _exec_queue;
    std::mutex _result_mutex;
    std::unordered_map<RBigIndex, APITask> _results;
    RBigIndex _next_id = 0;

    // The control drivers the tools wired and the values waiting for the control step.
    std::mutex _control_mutex;
    std::unordered_set<std::string> _wired_controllers;
    std::unordered_map<std::string, PendingTransform> _pending_transform;
    std::unordered_map<std::string, RVec3rf> _pending_force;
    std::unordered_map<std::string, RVec3rf> _pending_torque;
    std::unordered_map<std::string, PendingCameraPose> _pending_camera_pose;
    std::unordered_map<std::string, std::vector<PendingJoint>> _pending_joint;
    std::vector<std::shared_ptr<CaptureRequest>> _captures;

    // The cube embodiment's live motion state and pending reads, driven on the sync step.
    CubeAutoMotion _cube_auto;
    std::unordered_map<std::string, CubeGlidePlan> _cube_glides;
    std::vector<CubeReadRequest> _cube_reads;
    bool _is_cube_enabled = false;
    RVec3rf _cube_extent;
    RVec3rf _cube_eye_anchor;

    // Latest user state carried on the communication channels, drained on demand.
    std::mutex _user_mutex;
    RVec3rf _user_position;
    bool _has_user_position = false;
    RS_WorldSessionVRUserHandAims _user_hand_aims;
    RS_WorldSessionMessageBroker::Subscription _user_position_subscription;
    RS_WorldSessionMessageBroker::Subscription _user_hand_aims_subscription;

    /*****************************************************************
     * Registration
     *****************************************************************/
    void _registerBasicApis() {
        _addApi(std::make_unique<GetResultApi>(*this));
        _addApi(std::make_unique<PauseApi>(*this));
        _addApi(std::make_unique<ResumeApi>(*this));
        _addApi(std::make_unique<TerminateApi>(*this));
        _addApi(std::make_unique<GetStatusApi>(*this));
        _addApi(std::make_unique<IsRunningApi>(*this));
        _addApi(std::make_unique<IsPausedApi>(*this));
        _addApi(std::make_unique<IsEndApi>(*this));
        _addApi(std::make_unique<GetMessagesMatchingApi>(*this));
        _addApi(std::make_unique<GetWorldApi>(*this));
        _addApi(std::make_unique<GetWorldNameApi>(*this));
        _addApi(std::make_unique<SetSimulationApi>(*this));
        _addApi(std::make_unique<AddResourceApi>(*this));
        _addApi(std::make_unique<RemoveResourceApi>(*this));
        _addApi(std::make_unique<FindResourceApi>(*this));
        _addApi(std::make_unique<AddEssenceApi>(*this));
        _addApi(std::make_unique<UpdateEssenceApi>(*this));
        _addApi(std::make_unique<RemoveEssenceApi>(*this));
        _addApi(std::make_unique<FindEssenceApi>(*this));
        _addApi(std::make_unique<AddEntityApi>(*this));
        _addApi(std::make_unique<RemoveEntityApi>(*this));
        _addApi(std::make_unique<FindEntityApi>(*this));
        _addApi(std::make_unique<FreezeEntityApi>(*this));
        _addApi(std::make_unique<UnfreezeEntityApi>(*this));
        _addApi(std::make_unique<IsEntityFrozenApi>(*this));
        _addApi(std::make_unique<AddRecordApi>(*this));
        _addApi(std::make_unique<RemoveRecordApi>(*this));
        _addApi(std::make_unique<FindRecordApi>(*this));
        _addApi(std::make_unique<AddSignalApi>(*this));
        _addApi(std::make_unique<RemoveSignalApi>(*this));
        _addApi(std::make_unique<FindSignalApi>(*this));
        _addApi(std::make_unique<AddRecorderApi>(*this));
        _addApi(std::make_unique<RemoveRecorderApi>(*this));
        _addApi(std::make_unique<FindRecorderApi>(*this));
        _addApi(std::make_unique<AddControllerApi>(*this));
        _addApi(std::make_unique<RemoveControllerApi>(*this));
        _addApi(std::make_unique<FindControllerApi>(*this));
    }

    void _registerConvenienceApis() {
        _addApi(std::make_unique<GetRsctCategoryApi>(*this));
        _addApi(std::make_unique<GetRsctExampleApi>(*this));
        _addApi(std::make_unique<GetWorldSummaryApi>(*this));
        _addApi(std::make_unique<AddCameraApi>(*this));
    }

    void _registerEmbodimentApis() {
        _addApi(std::make_unique<EnableCubeEmbodimentApi>(*this));
        _addApi(std::make_unique<DisableCubeEmbodimentApi>(*this));
        _addApi(std::make_unique<GetCubePoseApi>(*this));
        _addApi(std::make_unique<GetCubeBoundsApi>(*this));
        _addApi(std::make_unique<GetCubeSurroundingApi>(*this));
        _addApi(std::make_unique<GetCubeTrackingApi>(*this));
        _addApi(std::make_unique<MoveCubeApi>(*this));
        _addApi(std::make_unique<LookCubeAtApi>(*this));
        _addApi(std::make_unique<TeleportCubeToApi>(*this));
        _addApi(std::make_unique<AutoCubeTrackAtApi>(*this));
        _addApi(std::make_unique<AutoCubeOrbitAroundApi>(*this));
        _addApi(std::make_unique<TakeCubeEyeImageApi>(*this));
        _addApi(std::make_unique<TakeCubeEyeBurstApi>(*this));
    }

    void _registerControlApis() {
        _addApi(std::make_unique<ApplyTransformApi>(*this));
        _addApi(std::make_unique<ApplyForceApi>(*this));
        _addApi(std::make_unique<ApplyTorqueApi>(*this));
        _addApi(std::make_unique<SetJointApi>(*this));
        _addApi(std::make_unique<MoveCameraApi>(*this));
        _addApi(std::make_unique<TakeImageApi>(*this));
        _addApi(std::make_unique<TakeBurstApi>(*this));
    }

    void _registerCommunicationApis() {
        _addApi(std::make_unique<GetUserPositionApi>(*this));
        _addApi(std::make_unique<GetUserHandAimsApi>(*this));
        _addApi(std::make_unique<ReadChannelApi>(*this));
    }

    void _addApi(std::unique_ptr<APIBase> api) {
        std::string name = api->name();
        _apis[name] = std::move(api);
    }

    /*****************************************************************
     * Transport
     *****************************************************************/
    void _listenThreadFunc() {
        if (!_server.listen(_host, _port)) {
            RS_LOG(_logger, warn, "MCP server could not bind {}:{}", _host, _port);
        }
    }

    void _handleRequest(const httplib::Request &request, httplib::Response &response) {
        if (!_isTokenValid(request)) {
            response.status = 401;
            response.set_content(R"({"error":"unauthorized"})", "application/json");
            return;
        }
        RS_JSON message = RS_JSON::parse(request.body, nullptr, false);
        RS_JSON reply;
        if (message.is_discarded() || !message.contains("method")) {
            reply = _makeError(message, -32700, "parse error");
        } else {
            std::string method = message.value("method", "");
            if (method == "initialize") {
                reply = _makeResult(message, _getInitializeResult(message));
            } else if (method == "ping") {
                reply = _makeResult(message, RS_JSON::object());
            } else if (method == "tools/list") {
                reply = _makeResult(message, _getToolList());
            } else if (method == "tools/call") {
                reply = _makeResult(message, _callTool(message.value("params", RS_JSON::object())));
            } else if (method.rfind("notifications/", 0) == 0) {
                response.status = 202;
                return;
            } else {
                reply = _makeError(message, -32601, "method not found");
            }
        }
        response.set_content(reply.dump(), "application/json");
    }

    [[nodiscard]] bool _isTokenValid(const httplib::Request &request) const {
        if (_token.empty()) return true;
        return request.get_header_value("Authorization") == "Bearer " + _token;
    }

    [[nodiscard]] RS_JSON _getInitializeResult(const RS_JSON &message) const {
        RS_JSON params = message.value("params", RS_JSON::object());
        RS_JSON result;
        result["protocolVersion"] =
            params.value("protocolVersion", std::string(RS_MCP_PROTOCOL_VERSION));
        result["capabilities"]["tools"] = RS_JSON::object();
        result["serverInfo"]["name"] = "rise-origin";
        result["serverInfo"]["version"] = RS_ENGINE_VERSION;
        return result;
    }

    static RS_JSON _makeResult(const RS_JSON &message, RS_JSON result) {
        RS_JSON reply;
        reply["jsonrpc"] = "2.0";
        reply["id"] = message.is_object() ? message.value("id", RS_JSON()) : RS_JSON();
        reply["result"] = std::move(result);
        return reply;
    }

    static RS_JSON _makeError(const RS_JSON &message, int code, std::string text) {
        RS_JSON reply;
        reply["jsonrpc"] = "2.0";
        reply["id"] = message.is_object() ? message.value("id", RS_JSON()) : RS_JSON();
        reply["error"]["code"] = code;
        reply["error"]["message"] = std::move(text);
        return reply;
    }

    /*****************************************************************
     * Tool listing and dispatch
     *****************************************************************/
    [[nodiscard]] RS_JSON _getToolList() const {
        RS_JSON tools = RS_JSON::array();
        std::scoped_lock lock(_api_mutex);
        for (const auto &entry : _apis) {
            tools.push_back(
                {{"name", entry.first},
                 {"description", entry.second->description()},
                 {"inputSchema", RS_JSON::parse(entry.second->schema(), nullptr, false)}}
            );
        }
        RS_JSON result;
        result["tools"] = std::move(tools);
        return result;
    }

    RS_JSON _callTool(const RS_JSON &params) {
        std::string name = params.value("name", std::string());
        RS_JSON args = params.value("arguments", RS_JSON::object());
        if (name == "get_result") return _getResult(args);

        APIBase *api = nullptr;
        {
            std::scoped_lock lock(_api_mutex);
            auto found = _apis.find(name);
            if (found == _apis.end()) return _makeErrorContent("unknown tool: " + name);
            api = found->second.get();
        }

        RBigIndex id;
        APITask task;
        {
            std::scoped_lock lock(_result_mutex);
            id = _next_id++;
            _results[id] = task;
        }
        _enqueueJob(api, args, task);

        if (task.waitFor(RS_MCP_INLINE_WAIT)) {
            RS_JSON result = task.get();
            _eraseResult(id);
            return result;
        }
        return _makeTextContent(_makePendingJson(id).dump());
    }

    RS_JSON _getResult(const RS_JSON &args) {
        RBigIndex id = args.value("id", RBigIndex(0));
        APITask task;
        {
            std::scoped_lock lock(_result_mutex);
            auto found = _results.find(id);
            if (found == _results.end()) return _makeErrorContent("unknown result id");
            task = found->second;
        }
        if (task.isFinished()) {
            RS_JSON result = task.get();
            _eraseResult(id);
            return result;
        }
        return _makeTextContent(_makePendingJson(id).dump());
    }

    static RS_JSON _makePendingJson(RBigIndex id) {
        RS_JSON pending;
        pending["id"] = id;
        pending["status"] = "pending";
        return pending;
    }

    void _eraseResult(RBigIndex id) {
        std::scoped_lock lock(_result_mutex);
        _results.erase(id);
    }

    /*****************************************************************
     * Executor
     *****************************************************************/
    void _enqueueJob(APIBase *api, RS_JSON args, APITask task) {
        {
            std::scoped_lock lock(_exec_mutex);
            _exec_queue.emplace_back(api, std::move(args), std::move(task));
        }
        _exec_cv.notify_one();
    }

    void _executorThreadFunc() {
        while (true) {
            std::tuple<APIBase *, RS_JSON, APITask> job;
            {
                std::unique_lock lock(_exec_mutex);
                _exec_cv.wait(lock, [&] { return !_is_alive || !_exec_queue.empty(); });
                if (!_is_alive && _exec_queue.empty()) return;
                job = std::move(_exec_queue.front());
                _exec_queue.pop_front();
            }
            auto &[api, args, task] = job;
            try {
                task.complete(api->call(args));
            } catch (const std::exception &error) {
                task.complete(APITask::makeFinished(_makeErrorContent(error.what())));
            }
        }
    }

    /*****************************************************************
     * Result and content helpers
     *****************************************************************/
    static RS_JSON _makeTextContent(const std::string &text) {
        RS_JSON content = RS_JSON::array();
        content.push_back({{"type", "text"}, {"text", text}});
        RS_JSON result;
        result["content"] = std::move(content);
        return result;
    }

    static RS_JSON _makeOkContent() { return _makeTextContent(R"({"ok":true})"); }

    static RS_JSON _makeErrorContent(const std::string &text) {
        RS_JSON result = _makeTextContent(text);
        result["isError"] = true;
        return result;
    }

    static RS_JSON _makeImageContent(std::vector<RS_JSON> images) {
        RS_JSON content = RS_JSON::array();
        for (RS_JSON &image : images) content.push_back(std::move(image));
        RS_JSON result;
        result["content"] = std::move(content);
        return result;
    }

    static RS_JSON _makeWorldResult(const RS_WorldTask &task) {
        RS_WorldTaskData data = task.get();
        RS_JSON out;
        if (data.is<RS_WorldTaskDataCreated>()) {
            const RS_WorldTaskDataCreated &created = data.as<RS_WorldTaskDataCreated>();
            out["ok"] = true;
            out["gid"] = created.gid;
            out["name"] = created.name;
        } else if (data.is<RS_WorldTaskDataFailed>()) {
            out["ok"] = false;
            out["error"] = data.as<RS_WorldTaskDataFailed>().error;
        } else {
            out["ok"] = true;
        }
        return _makeTextContent(out.dump());
    }

    /*****************************************************************
     * Schema and argument helpers
     *****************************************************************/
    static RS_JSON _schema(RS_JSON properties, std::vector<std::string> required = {}) {
        RS_JSON out;
        out["type"] = "object";
        out["properties"] = properties.is_object() ? std::move(properties) : RS_JSON::object();
        if (!required.empty()) out["required"] = required;
        return out;
    }

    static RS_JSON _emptySchema() { return _schema(RS_JSON::object()); }

    static RS_JSON _nameSchema() { return _schema({{"name", _prop("string")}}, {"name"}); }

    static RS_JSON _rsctSchema() { return _schema({{"rsct", _prop("object")}}, {"rsct"}); }

    static RS_JSON _prop(const char *type) {
        RS_JSON out;
        out["type"] = type;
        return out;
    }

    static RS_JSON _propArray(int count) {
        RS_JSON out;
        out["type"] = "array";
        out["items"] = _prop("number");
        out["minItems"] = count;
        out["maxItems"] = count;
        return out;
    }

    static std::string _getName(const RS_JSON &args) { return args.value("name", std::string()); }

    static std::string _getString(const RS_JSON &args, const std::string &key) {
        return args.value(key, std::string());
    }

    static RVec3rf _getVec3(const RS_JSON &args, const std::string &key, RVec3rf fallback) {
        if (!args.contains(key)) return fallback;
        const RS_JSON &value = args.at(key);
        return RVec3rf(
            value.at(0).get<RFloat>(), value.at(1).get<RFloat>(), value.at(2).get<RFloat>()
        );
    }

    static RQuat3rf _getQuat(const RS_JSON &args, const std::string &key, RQuat3rf fallback) {
        if (!args.contains(key)) return fallback;
        const RS_JSON &value = args.at(key);
        return RQuat3rf(
            value.at(0).get<RFloat>(),
            value.at(1).get<RFloat>(),
            value.at(2).get<RFloat>(),
            value.at(3).get<RFloat>()
        );
    }

    static RColor3f _getColor(const RS_JSON &args, const std::string &key, RColor3f fallback) {
        if (!args.contains(key)) return fallback;
        const RS_JSON &value = args.at(key);
        return _makeColor(
            value.at(0).get<RFloat>(), value.at(1).get<RFloat>(), value.at(2).get<RFloat>()
        );
    }

    static RColor3f _makeColor(RFloat red, RFloat green, RFloat blue) {
        RColor3f color;
        color.r = red / 255.0f;
        color.g = green / 255.0f;
        color.b = blue / 255.0f;
        return color;
    }

    static RSE_WorldMessageLevel _getMessageLevel(const std::string &level) {
        if (level == "warn") return RSE_WORLD_MESSAGE_WARNING;
        if (level == "err") return RSE_WORLD_MESSAGE_ERROR;
        return RSE_WORLD_MESSAGE_INFO;
    }

    /*****************************************************************
     * RSC serialization
     *****************************************************************/
    template <typename Variant>
    static RS_JSON _makeVariantContent(const Variant &variant) {
        return _makeTextContent(_variantToJson(variant).dump());
    }

    template <typename Variant>
    static RS_JSON _variantToJson(const Variant &variant) {
        RS_JSON json;
        variant.visit([&json](const auto &held) { json = rsc::RS_RSCArchive::toJSONAny(held); });
        return json;
    }

    template <typename Variant>
    static Variant _jsonToVariant(const RS_JSON &json) {
        auto node = rsc::RS_RSCArchive::fromJSONAny(json);
        Variant variant;
        if (!rsc::RS_RSCRegistry::tryAssign(variant, node)) {
            RS_THROW(
                rsc::RS_RSCArchiveException("JSON holds an RSC type not valid for this section")
            );
        }
        return variant;
    }

    template <typename Node>
    static RS_JSON _nodeToJson(const Node &node) {
        return rsc::RS_RSCArchive::toJSON(node);
    }

    template <typename Node>
    static Node _jsonToNode(const RS_JSON &json) {
        return rsc::RS_RSCArchive::fromJSON<Node>(json);
    }

    template <typename Section>
    static RS_JSON _getSectionJson(const Section &section) {
        RS_JSON parts = RS_JSON::array();
        for (const auto &variant : section) parts.push_back(_variantToJson(variant));
        return parts;
    }

    template <typename Section>
    static RS_JSON _getSectionSummary(const Section &section) {
        RS_JSON parts = RS_JSON::array();
        for (const auto &variant : section) parts.push_back(variant.visit(PartSummary{}));
        return parts;
    }

    static RS_JSON _makeExample(const std::string &type) {
        auto node = rsc::RS_RSCRegistry::getExample(type);
        return rsc::RS_RSCArchive::toJSONAny(*node);
    }

    /*****************************************************************
     * Entity and material builders
     *****************************************************************/
    template <typename Camera>
    static void
    _fillCamera(Camera &camera, const RS_JSON &args, const std::string &controller_name) {
        camera.image_width = args.value("width", 1280);
        camera.image_height = args.value("height", 720);
        camera.samples_per_pixel = args.value("samples", 4);
        camera.max_depth = args.value("max_depth", 3);
        camera.field_of_view = args.value("fov", 60.0);
        camera.record->has_rgb = true;
        camera.record->target_controller_reference = controller_name;
        camera.signal->source_controller_reference = controller_name;
    }

    static void _addGlassMaterial(
        std::shared_ptr<RS_WorldSessionHandle> handle,
        const std::string &material,
        RColor3f color,
        RFloat roughness
    ) {
        rsc::RSC_TextureConstant albedo;
        albedo.name = material + ".albedo";
        albedo.value = color;
        handle->addResource(rsc::RSC_ResourceVariant(albedo));

        rsc::RSC_TextureConstant surface;
        surface.name = material + ".rm";
        surface.value.r = roughness;
        handle->addResource(rsc::RSC_ResourceVariant(surface));

        rsc::RSC_Material handle_essence;
        handle_essence.name = material;
        handle->addEssence(rsc::RSC_EssenceVariant(handle_essence));

        rsc::RSC_MaterialPhysicalPropertyRigid rigid;
        rigid.name = material + ".phys_rigid";
        rigid.material_reference = material;
        rigid.density = 1000.0;
        handle->addEssence(rsc::RSC_EssenceVariant(rigid));

        rsc::RSC_MaterialPhysicalPropertyFixed fixed;
        fixed.name = material + ".phys_fixed";
        fixed.material_reference = material;
        handle->addEssence(rsc::RSC_EssenceVariant(fixed));

        rsc::RSC_MaterialOpticalPropertySimpleMicrofacet optical;
        optical.name = material + ".optic";
        optical.material_reference = material;
        optical.albedo_texture_reference = albedo.name;
        optical.roughness_metalness_texture_reference = surface.name;
        optical.is_transparent = true;
        optical.ior = 1.5;
        handle->addEssence(rsc::RSC_EssenceVariant(optical));
    }

    static rsc::RSC_StructureVoxel _makeCubeStructure(
        const std::string &name,
        const std::string &material,
        RSize size,
        RFloat voxel_size,
        RVec3rf position
    ) {
        rsc::RSC_StructureVoxelBody box;
        box.x_voxels = size;
        box.y_voxels = size;
        box.z_voxels = size;
        box.voxel_size = voxel_size;
        RSize voxel_num = size * size * size;
        box.material_reference_sid.assign(voxel_num, 0);
        box.segment_bid.assign(voxel_num, 0);

        rsc::RSC_StructureVoxel cube;
        cube.name = name;
        cube.material_references.push_back(material);
        cube.is_fixed = false;
        RFloat half = size * voxel_size * 0.5f;
        cube.origin_position = RVec3rf(position.x - half, position.y - half, position.z - half);
        cube.bodies.push_back(box);
        return cube;
    }

    /*****************************************************************
     * Control buffering and application
     *****************************************************************/
    void _queueTransform(const std::string &structure, const RS_JSON &args) {
        _ensureTransformDriver(_access.handle(), structure);
        PendingTransform value;
        value.translation = _getVec3(args, "translation", RVec3rf::zero());
        value.rotation = _getQuat(args, "rotation", RQuat3rf::unit());
        value.pivot = _getVec3(args, "pivot", RVec3rf::zero());
        std::scoped_lock lock(_control_mutex);
        _pending_transform[structure] = value;
    }

    void _queueForce(const std::string &structure, const RS_JSON &args) {
        _ensureForceDriver(_access.handle(), structure);
        std::scoped_lock lock(_control_mutex);
        _pending_force[structure] = _getVec3(args, "vector", RVec3rf::zero());
    }

    void _queueTorque(const std::string &structure, const RS_JSON &args) {
        _ensureTorqueDriver(_access.handle(), structure);
        std::scoped_lock lock(_control_mutex);
        _pending_torque[structure] = _getVec3(args, "vector", RVec3rf::zero());
    }

    void _queueCameraPose(const std::string &camera, const RS_JSON &args) {
        PendingCameraPose pose;
        pose.position = _getVec3(args, "position", RVec3rf::zero());
        pose.orientation = _getQuat(args, "orientation", RQuat3rf::unit());
        std::scoped_lock lock(_control_mutex);
        _pending_camera_pose[camera] = pose;
    }

    void _pushJoint(const std::string &controller, PendingJoint joint) {
        std::scoped_lock lock(_control_mutex);
        _pending_joint[controller].push_back(joint);
    }

    void _ensureTransformDriver(
        std::shared_ptr<RS_WorldSessionHandle> handle,
        const std::string &structure
    ) {
        std::string controller_name = _getDriverName("transform", structure);
        if (!_markWired(controller_name)) return;
        Transaction transaction(*handle);
        _addCallbackController(handle, controller_name);
        rsc::RSC_Transform signal;
        signal.name = controller_name + ".signal";
        signal.structure_name = structure;
        signal.source_controller_reference = controller_name;
        handle->addSignal(rsc::RSC_SignalVariant(signal));
    }

    void _ensureForceDriver(
        std::shared_ptr<RS_WorldSessionHandle> handle,
        const std::string &structure
    ) {
        std::string controller_name = _getDriverName("force", structure);
        if (!_markWired(controller_name)) return;
        Transaction transaction(*handle);
        _addCallbackController(handle, controller_name);
        rsc::RSC_Force signal;
        signal.name = controller_name + ".signal";
        signal.structure_name = structure;
        signal.source_controller_reference = controller_name;
        // TODO: remove hard encoding of body_sid and segment_bid
        signal.body_sid = 0;
        signal.segment_bid = 0;
        handle->addSignal(rsc::RSC_SignalVariant(signal));
    }

    void _ensureTorqueDriver(
        std::shared_ptr<RS_WorldSessionHandle> handle,
        const std::string &structure
    ) {
        std::string controller_name = _getDriverName("torque", structure);
        if (!_markWired(controller_name)) return;
        Transaction transaction(*handle);
        _addCallbackController(handle, controller_name);
        rsc::RSC_Torque signal;
        signal.name = controller_name + ".signal";
        signal.structure_name = structure;
        signal.source_controller_reference = controller_name;
        // TODO: remove hard encoding of body_sid and segment_bid
        signal.body_sid = 0;
        signal.segment_bid = 0;
        handle->addSignal(rsc::RSC_SignalVariant(signal));
    }

    bool _markWired(const std::string &controller_name) {
        std::scoped_lock lock(_control_mutex);
        return _wired_controllers.insert(controller_name).second;
    }

    static void _addCallbackController(
        std::shared_ptr<RS_WorldSessionHandle> handle,
        const std::string &controller_name
    ) {
        rsc::RSC_ControllerCallback controller;
        controller.name = controller_name;
        handle->addController(rsc::RSC_ControllerVariant(controller));
    }

    static std::string _getDriverName(const std::string &kind, const std::string &structure) {
        return "mcp." + kind + "." + structure;
    }

    std::string _getVoxelControllerName(const std::string &structure) {
        VoxelControllerReader reader;
        _access.handle()->findEntity(structure).visit(reader);
        return reader.controller;
    }

    void _applyControls() {
        std::scoped_lock lock(_control_mutex);
        _applyTransforms();
        _applyForces();
        _applyTorques();
        _applyJoints();
        _applyCameraPoses();
    }

    void _applyTransforms() {
        std::vector<std::string> done;
        for (auto &entry : _pending_transform) {
            RS_WorldController *controller =
                _findController(_getDriverName("transform", entry.first));
            if (controller == nullptr) continue;
            TransformWriter writer{entry.second};
            for (auto &signal : controller->signals) signal.visit(writer);
            done.push_back(entry.first);
        }
        for (const std::string &structure : done) _pending_transform.erase(structure);
    }

    void _applyForces() {
        for (auto &entry : _pending_force) {
            RS_WorldController *controller = _findController(_getDriverName("force", entry.first));
            if (controller == nullptr) continue;
            ForceWriter writer{entry.second};
            for (auto &signal : controller->signals) signal.visit(writer);
        }
    }

    void _applyTorques() {
        for (auto &entry : _pending_torque) {
            RS_WorldController *controller = _findController(_getDriverName("torque", entry.first));
            if (controller == nullptr) continue;
            TorqueWriter writer{entry.second};
            for (auto &signal : controller->signals) signal.visit(writer);
        }
    }

    void _applyJoints() {
        std::vector<std::string> done;
        for (auto &entry : _pending_joint) {
            RS_WorldController *controller = _findController(entry.first);
            if (controller == nullptr) continue;
            JointWriter writer{entry.second};
            for (auto &signal : controller->signals) signal.visit(writer);
            done.push_back(entry.first);
        }
        for (const std::string &controller : done) _pending_joint.erase(controller);
    }

    void _applyCameraPoses() {
        std::vector<std::string> done;
        for (auto &entry : _pending_camera_pose) {
            RS_WorldController *controller = _findController(entry.first + ".controller");
            if (controller == nullptr) continue;
            CameraPoseWriter writer{entry.second};
            for (auto &signal : controller->signals) signal.visit(writer);
            done.push_back(entry.first);
        }
        for (const std::string &camera : done) _pending_camera_pose.erase(camera);
    }

    RS_WorldController *_findController(const std::string &name) {
        for (auto &controller : _access.controllers()) {
            if (controller.get().name == name) return &controller.get();
        }
        return nullptr;
    }

    /*****************************************************************
     * Capture buffering and readout
     *****************************************************************/
    std::shared_future<RS_JSON> _queueCapture(const RS_JSON &args, RSize count) {
        std::string camera = args.contains("camera") ? _getString(args, "camera")
                                                     : RS_MCP_BODY_EYE_NAME;
        auto handle = _access.handle();
        auto request = std::make_shared<CaptureRequest>();
        request->camera = camera;
        request->controller = _getCameraControllerName(camera);
        request->remaining = count == 0 ? 1 : count;
        request->stride = args.value("step_stride", 2);
        request->countdown = 0;
        request->was_frozen = handle->isEntityFrozen(camera);
        request->promise = std::make_shared<std::promise<RS_JSON>>();
        std::shared_future<RS_JSON> result = request->promise->get_future().share();
        {
            Transaction transaction(*handle);
            handle->unfreezeEntity(camera);
        }
        std::scoped_lock lock(_control_mutex);
        _captures.push_back(request);
        return result;
    }

    std::string _getCameraControllerName(const std::string &camera) {
        CameraControllerReader reader;
        _access.handle()->findEntity(camera).visit(reader);
        return reader.controller.empty() ? camera + ".controller" : reader.controller;
    }

    void _readCaptures() {
        std::vector<std::shared_ptr<CaptureRequest>> pending;
        {
            std::scoped_lock lock(_control_mutex);
            pending.swap(_captures);
        }
        std::vector<std::shared_ptr<CaptureRequest>> unfinished;
        for (auto &request : pending) {
            if (_advanceCapture(request)) {
                request->promise->set_value(_makeImageContent(std::move(request->frames)));
                if (request->was_frozen) {
                    APIBase *freeze = nullptr;
                    {
                        std::scoped_lock lock(_api_mutex);
                        freeze = _apis.at("freeze_entity").get();
                    }
                    RS_JSON args;
                    args["name"] = request->camera;
                    _enqueueJob(freeze, args, APITask());
                }
            } else {
                unfinished.push_back(request);
            }
        }
        std::scoped_lock lock(_control_mutex);
        for (auto &request : unfinished) _captures.push_back(request);
    }

    bool _advanceCapture(const std::shared_ptr<CaptureRequest> &request) {
        RS_WorldController *controller = _findController(request->controller);
        if (controller == nullptr) return false;
        if (request->countdown > 0) {
            request->countdown -= 1;
            return false;
        }
        CaptureReader reader;
        for (auto &record : controller->records) record.visit(reader);
        if (!reader.is_valid) return false;
        request->frames.push_back(_encodeImage(reader.width, reader.height, reader.pixels));
        request->remaining -= 1;
        request->countdown = request->stride;
        return request->remaining == 0;
    }

    static RS_JSON _encodeImage(RSize width, RSize height, const RColor3b *pixels) {
        int length = 0;
        unsigned char *png = stbi_write_png_to_mem(
            reinterpret_cast<const unsigned char *>(pixels),
            static_cast<int>(width) * 3,
            static_cast<int>(width),
            static_cast<int>(height),
            3,
            &length
        );
        std::string encoded;
        if (png != nullptr) {
            encoded = _encodeBase64(png, static_cast<RBigSize>(length));
            std::free(png);
        }
        RS_JSON image;
        image["type"] = "image";
        image["data"] = std::move(encoded);
        image["mimeType"] = "image/png";
        return image;
    }

    static std::string _encodeBase64(const unsigned char *data, RBigSize length) {
        static const char table[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve(((length + 2) / 3) * 4);
        for (RBigSize i = 0; i < length; i += 3) {
            unsigned int chunk = data[i] << 16;
            if (i + 1 < length) chunk |= data[i + 1] << 8;
            if (i + 2 < length) chunk |= data[i + 2];
            out.push_back(table[(chunk >> 18) & 0x3f]);
            out.push_back(table[(chunk >> 12) & 0x3f]);
            out.push_back(i + 1 < length ? table[(chunk >> 6) & 0x3f] : '=');
            out.push_back(i + 2 < length ? table[chunk & 0x3f] : '=');
        }
        return out;
    }

    /*****************************************************************
     * Embodiment control and reads
     *****************************************************************/
    static APITask _makeCubeNotEnabledErrorTask() {
        return APITask::makeFinished(
            _makeTextContent(R"({"ok":false,"error":"cube embodiment not enabled"})")
        );
    }

    CubePose _readCubePose() {
        CubePose pose;
        RS_WorldController *controller =
            _findController(_getDriverName("transform", RS_MCP_BODY_NAME));
        if (controller == nullptr) return pose;
        CubeRigidBodyReader reader;
        for (auto &record : controller->records) record.visit(reader);
        return reader.pose;
    }

    CubeSurroundingReader _readCubeSurrounding() {
        CubeSurroundingReader reader;
        RS_WorldController *controller =
            _findController(_getDriverName("transform", RS_MCP_BODY_NAME));
        if (controller != nullptr) {
            for (auto &record : controller->records) record.visit(reader);
        }
        return reader;
    }

    RVec3rf _getCubeEyePosition(const CubePose &pose) const {
        return pose.com + pose.orientation.rotateVec3D(_cube_eye_anchor);
    }

    static RQuat3rf _makeUprightLook(RVec3rf forward) {
        RVec3rf look = forward.normalized();
        RQuat3rf base(RVec3rf(1.0f, 0.0f, 0.0f), look);
        RVec3rf up_now = base.toOrthoAxisZ();
        RVec3rf world_up(0.0f, 0.0f, 1.0f);
        RVec3rf up_proj = world_up - look * world_up.dot(look);
        if (up_proj.length2() <= 1e-8f) return base.normalized();
        up_proj = up_proj.normalized();
        RFloat cos_roll = std::clamp(up_now.dot(up_proj), -1.0f, 1.0f);
        RFloat sin_roll = up_now.cross(up_proj).dot(look);
        RFloat roll = std::atan2(sin_roll, cos_roll);
        return (RQuat3rf(roll, look) * base).normalized();
    }

    void
    _writeCubeAbsolute(const CubePose &pose, RVec3rf target_position, RQuat3rf target_orientation) {
        PendingTransform value;
        value.rotation = target_orientation * pose.orientation.conjugate();
        value.pivot = pose.com;
        value.translation = target_position - pose.com;
        _pending_transform[RS_MCP_BODY_NAME] = value;
    }

    void _queueCubeGlide(
        bool is_look,
        RVec3rf point,
        RFloat duration,
        std::shared_ptr<std::promise<RS_JSON>> promise
    ) {
        CubeGlidePlan plan;
        plan.is_look = is_look;
        plan.point = point;
        plan.duration = duration;
        plan.promise = std::move(promise);
        std::scoped_lock lock(_control_mutex);
        _cube_auto.mode = CubeAutoMotionMode::NONE;
        _cube_glides[RS_MCP_BODY_NAME] = std::move(plan);
    }

    void _setCubeTrack(RIndex gid, std::string name, RFloat distance) {
        std::scoped_lock lock(_control_mutex);
        _cube_glides.clear();
        _cube_auto = CubeAutoMotion{};
        _cube_auto.mode = CubeAutoMotionMode::TRACK;
        _cube_auto.target_gid = gid;
        _cube_auto.target_name = std::move(name);
        _cube_auto.distance = distance;
    }

    void _setCubeOrbit(RVec3rf point, RFloat radius, RFloat height, RFloat period) {
        std::scoped_lock lock(_control_mutex);
        _cube_glides.clear();
        _cube_auto = CubeAutoMotion{};
        _cube_auto.mode = CubeAutoMotionMode::ORBIT;
        _cube_auto.point = point;
        _cube_auto.radius = radius;
        _cube_auto.height = height;
        _cube_auto.period = period;
    }

    void _clearCubeMotion() {
        std::scoped_lock lock(_control_mutex);
        _cube_glides.clear();
        _cube_auto.mode = CubeAutoMotionMode::NONE;
    }

    std::shared_future<RS_JSON> _queueCubeRead(CubeReadKind kind) {
        CubeReadRequest request;
        request.kind = kind;
        request.promise = std::make_shared<std::promise<RS_JSON>>();
        std::shared_future<RS_JSON> result = request.promise->get_future().share();
        std::scoped_lock lock(_control_mutex);
        _cube_reads.push_back(std::move(request));
        return result;
    }

    void _advanceCubeGlides() {
        CubePose pose = _readCubePose();
        std::scoped_lock lock(_control_mutex);
        if (_cube_glides.empty() || !pose.is_valid) return;
        auto now = std::chrono::steady_clock::now();
        std::vector<std::string> done;
        for (auto &entry : _cube_glides) {
            CubeGlidePlan &plan = entry.second;
            if (!plan.has_start) {
                plan.has_start = true;
                plan.start_position = pose.com;
                plan.start_orientation = pose.orientation;
                plan.last_time = now;
                if (plan.is_look) {
                    plan.target_position = pose.com;
                    plan.target_orientation =
                        _makeUprightLook(plan.point - _getCubeEyePosition(pose));
                } else {
                    plan.target_position = plan.point;
                    plan.target_orientation = pose.orientation;
                }
            }
            plan.elapsed += std::chrono::duration<RFloat>(now - plan.last_time).count();
            plan.last_time = now;
            RFloat t = plan.duration > 0.0f ? std::clamp(plan.elapsed / plan.duration, 0.0f, 1.0f)
                                            : 1.0f;
            RVec3rf desired_position =
                plan.start_position + (plan.target_position - plan.start_position) * t;
            RQuat3rf desired_orientation = plan.start_orientation.slerp(plan.target_orientation, t);
            _writeCubeAbsolute(pose, desired_position, desired_orientation);
            if (t >= 1.0f) {
                if (plan.promise) plan.promise->set_value(_makeTextContent("arrived"));
                done.push_back(entry.first);
            }
        }
        for (const std::string &key : done) _cube_glides.erase(key);
    }

    void _advanceCubeAutoMotion() {
        CubePose pose = _readCubePose();
        std::scoped_lock lock(_control_mutex);
        if (_cube_auto.mode == CubeAutoMotionMode::NONE || !pose.is_valid) return;
        auto now = std::chrono::steady_clock::now();
        if (_cube_auto.mode == CubeAutoMotionMode::TRACK) {
            CubeSurroundingReader reader = _readCubeSurrounding();
            if (!reader.is_valid) return;
            const CubeSurroundingEntry *hit = nullptr;
            for (const CubeSurroundingEntry &entry : reader.entries) {
                if (entry.gid == _cube_auto.target_gid) {
                    hit = &entry;
                    break;
                }
            }
            if (hit == nullptr) {
                _cube_auto.mode = CubeAutoMotionMode::NONE;
                return;
            }
            RFloat distance = hit->closest_offset.length();
            RVec3rf direction = distance > 1e-5f ? hit->closest_offset * (1.0f / distance)
                                                 : RVec3rf(1.0f, 0.0f, 0.0f);
            RVec3rf target_position = pose.com + direction * (distance - _cube_auto.distance);
            _writeCubeAbsolute(pose, target_position, _makeUprightLook(direction));
        } else {
            if (!_cube_auto.has_last_time) {
                RVec3rf relative = pose.com - _cube_auto.point;
                _cube_auto.angle = std::atan2(relative.y, relative.x);
                _cube_auto.has_last_time = true;
                _cube_auto.last_time = now;
            }
            RFloat delta = std::chrono::duration<RFloat>(now - _cube_auto.last_time).count();
            _cube_auto.last_time = now;
            if (_cube_auto.period > 0.0f)
                _cube_auto.angle += (delta / _cube_auto.period) * 2.0f * RFloat(RS_PI);
            RVec3rf position =
                _cube_auto.point + RVec3rf(
                                       _cube_auto.radius * std::cos(_cube_auto.angle),
                                       _cube_auto.radius * std::sin(_cube_auto.angle),
                                       _cube_auto.height
                                   );
            _writeCubeAbsolute(pose, position, _makeUprightLook(_cube_auto.point - position));
        }
    }

    void _readCubeReads() {
        std::vector<CubeReadRequest> pending;
        {
            std::scoped_lock lock(_control_mutex);
            if (_cube_reads.empty()) return;
            pending.swap(_cube_reads);
        }
        CubePose pose = _readCubePose();
        CubeSurroundingReader surrounding = _readCubeSurrounding();
        std::vector<CubeReadRequest> unready;
        for (CubeReadRequest &request : pending) {
            if (request.kind == CubeReadKind::SURROUNDING) {
                if (!surrounding.is_valid) {
                    unready.push_back(std::move(request));
                    continue;
                }
                request.promise->set_value(_makeCubeSurroundingJson(surrounding.entries));
            } else if (!pose.is_valid) {
                unready.push_back(std::move(request));
            } else if (request.kind == CubeReadKind::POSE) {
                request.promise->set_value(_makeCubePoseJson(pose));
            } else {
                request.promise->set_value(_makeCubeBoundsJson(pose));
            }
        }
        if (!unready.empty()) {
            std::scoped_lock lock(_control_mutex);
            for (CubeReadRequest &request : unready) _cube_reads.push_back(std::move(request));
        }
    }

    RS_JSON _makeCubePoseJson(const CubePose &pose) const {
        RVec3rf forward = pose.orientation.toOrthoAxisX();
        RVec3rf up = pose.orientation.toOrthoAxisZ();
        RVec3rf eye = _getCubeEyePosition(pose);
        RS_JSON out;
        out["position"] = {pose.com.x, pose.com.y, pose.com.z};
        out["orientation"] = {
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        };
        out["size"] = {_cube_extent.x, _cube_extent.y, _cube_extent.z};
        out["linear_velocity"] = {
            pose.linear_velocity.x, pose.linear_velocity.y, pose.linear_velocity.z
        };
        out["angular_velocity"] = {
            pose.angular_velocity.x, pose.angular_velocity.y, pose.angular_velocity.z
        };
        out["eye_position"] = {eye.x, eye.y, eye.z};
        out["forward"] = {forward.x, forward.y, forward.z};
        out["up"] = {up.x, up.y, up.z};
        return _makeTextContent(out.dump());
    }

    RS_JSON _makeCubeBoundsJson(const CubePose &pose) const {
        RVec3rf half = _cube_extent * 0.5f;
        RVec3rf low, high;
        bool is_first = true;
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                for (int sz = -1; sz <= 1; sz += 2) {
                    RVec3rf corner(half.x * sx, half.y * sy, half.z * sz);
                    RVec3rf world = pose.com + pose.orientation.rotateVec3D(corner);
                    if (is_first) {
                        low = world;
                        high = world;
                        is_first = false;
                    } else {
                        low = RVec3rf(
                            std::min(low.x, world.x),
                            std::min(low.y, world.y),
                            std::min(low.z, world.z)
                        );
                        high = RVec3rf(
                            std::max(high.x, world.x),
                            std::max(high.y, world.y),
                            std::max(high.z, world.z)
                        );
                    }
                }
            }
        }
        RS_JSON out;
        out["min"] = {low.x, low.y, low.z};
        out["max"] = {high.x, high.y, high.z};
        return _makeTextContent(out.dump());
    }

    RS_JSON _makeCubeSurroundingJson(const std::vector<CubeSurroundingEntry> &entries) {
        auto handle = _access.handle();
        std::vector<std::pair<RFloat, RS_JSON>> items;
        for (const CubeSurroundingEntry &entry : entries) {
            std::string name = handle->findEntityName(entry.gid);
            if (name.empty() || name == RS_MCP_BODY_NAME) continue;
            RFloat distance = entry.closest_offset.length();
            RS_JSON item;
            item["name"] = name;
            item["type"] = _getEntityType(name);
            item["closest_offset"] = {
                entry.closest_offset.x, entry.closest_offset.y, entry.closest_offset.z
            };
            item["distance"] = distance;
            items.emplace_back(distance, std::move(item));
        }
        std::sort(items.begin(), items.end(), [](const auto &left, const auto &right) {
            return left.first < right.first;
        });
        RS_JSON list = RS_JSON::array();
        for (auto &item : items) list.push_back(std::move(item.second));
        RS_JSON out;
        out["count"] = list.size();
        out["entities"] = std::move(list);
        return _makeTextContent(out.dump());
    }

    RS_JSON _makeCubeTrackingJson() {
        std::scoped_lock lock(_control_mutex);
        RS_JSON out;
        if (_cube_auto.mode == CubeAutoMotionMode::TRACK) {
            out["mode"] = "track";
            out["target"] = _cube_auto.target_name;
            out["distance"] = _cube_auto.distance;
        } else if (_cube_auto.mode == CubeAutoMotionMode::ORBIT) {
            out["mode"] = "orbit";
            out["point"] = {_cube_auto.point.x, _cube_auto.point.y, _cube_auto.point.z};
            out["radius"] = _cube_auto.radius;
            out["height"] = _cube_auto.height;
            out["period"] = _cube_auto.period;
        } else {
            out["mode"] = "none";
        }
        return _makeTextContent(out.dump());
    }

    std::string _getEntityType(const std::string &name) {
        try {
            return _access.handle()
                ->findEntity(name)
                .visit(PartSummary{})
                .value("type", std::string());
        } catch (const std::exception &) {
            return std::string();
        }
    }

    /*****************************************************************
     * Communication draining
     *****************************************************************/
    void _subscribeUserChannels() {
        _user_position_subscription = _access.subscribe<RS_WorldSessionVRUserPosition>(
            RS_VR_CHANNEL_USER_POSITION,
            std::bind(&RS_MCPImpl::_onUserPosition, this, std::placeholders::_1)
        );
        _user_hand_aims_subscription = _access.subscribe<RS_WorldSessionVRUserHandAims>(
            RS_VR_CHANNEL_USER_HAND_AIMS,
            std::bind(&RS_MCPImpl::_onUserHandAims, this, std::placeholders::_1)
        );
    }

    void _onUserPosition(std::shared_ptr<const RS_WorldSessionVRUserPosition> message) {
        std::scoped_lock lock(_user_mutex);
        _user_position = message->position;
        _has_user_position = true;
    }

    void _onUserHandAims(std::shared_ptr<const RS_WorldSessionVRUserHandAims> message) {
        std::scoped_lock lock(_user_mutex);
        _user_hand_aims = *message;
    }

    RS_JSON _drainUserPosition() {
        _user_position_subscription.drain();
        std::scoped_lock lock(_user_mutex);
        if (!_has_user_position) return _makeTextContent(R"({"valid":false})");
        RS_JSON result;
        result["valid"] = true;
        result["position"] = {_user_position.x, _user_position.y, _user_position.z};
        return _makeTextContent(result.dump());
    }

    RS_JSON _drainHandAims() {
        _user_hand_aims_subscription.drain();
        std::scoped_lock lock(_user_mutex);
        RS_JSON result;
        result["left"] = _getHandAimJson(_user_hand_aims.left);
        result["right"] = _getHandAimJson(_user_hand_aims.right);
        return _makeTextContent(result.dump());
    }

    static RS_JSON _getHandAimJson(const RS_WorldSessionVRHandAim &aim) {
        RS_JSON result;
        result["valid"] = aim.is_valid;
        result["origin"] = {aim.aim_origin.x, aim.aim_origin.y, aim.aim_origin.z};
        result["direction"] = {aim.aim_direction.x, aim.aim_direction.y, aim.aim_direction.z};
        result["position"] = {aim.aim_position.x, aim.aim_position.y, aim.aim_position.z};
        return result;
    }

   private:
    /*****************************************************************
     * (a) Basic direct world-handle dispatch
     *****************************************************************/
    struct GetResultApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_result"; }
        std::string description() const override { return "Get a deferred tool result by id."; }
        std::string schema() const override {
            return _schema({{"id", _prop("integer")}}, {"id"}).dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(impl._getResult(params));
        }
    };

    struct PauseApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "pause"; }
        std::string description() const override { return "Pause the world stepping."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            impl._access.handle()->pause();
            return APITask::makeFinished(_makeTextContent("paused"));
        }
    };

    struct ResumeApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "resume"; }
        std::string description() const override { return "Resume the world stepping."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            impl._access.handle()->resume();
            return APITask::makeFinished(_makeTextContent("resumed"));
        }
    };

    struct TerminateApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "terminate"; }
        std::string description() const override { return "End the world."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            impl._access.handle()->terminate();
            return APITask::makeFinished(_makeTextContent("terminating"));
        }
    };

    struct GetStatusApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_status"; }
        std::string description() const override { return "Get the world status and step state."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            auto handle = impl._access.handle();
            RS_JSON status;
            status["status"] = enumToString(handle->getStatus());
            status["is_running"] = handle->isRunning();
            status["is_paused"] = handle->isPaused();
            status["is_end"] = handle->isEnd();
            return APITask::makeFinished(_makeTextContent(status.dump()));
        }
    };

    struct IsRunningApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "is_running"; }
        std::string description() const override { return "Get whether the world is running."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            return APITask::makeFinished(
                _makeTextContent(impl._access.handle()->isRunning() ? "true" : "false")
            );
        }
    };

    struct IsPausedApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "is_paused"; }
        std::string description() const override { return "Get whether the world is paused."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            return APITask::makeFinished(
                _makeTextContent(impl._access.handle()->isPaused() ? "true" : "false")
            );
        }
    };

    struct IsEndApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "is_end"; }
        std::string description() const override { return "Get whether the world has ended."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            return APITask::makeFinished(
                _makeTextContent(impl._access.handle()->isEnd() ? "true" : "false")
            );
        }
    };

    struct GetMessagesMatchingApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_messages_matching"; }
        std::string description() const override { return "Get recent world messages at a level."; }
        std::string schema() const override {
            return _schema({{"max_count", _prop("integer")}, {"level", _prop("string")}}).dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            RSize max_count = params.value("max_count", 50);
            RSE_WorldMessageLevel level = _getMessageLevel(_getString(params, "level"));
            std::vector<RS_WorldMessage> messages =
                impl._access.handle()->getMessagesMatching(max_count, 0, RS_NULL_BIG_INDEX, level);
            RS_JSON out = RS_JSON::array();
            for (const RS_WorldMessage &message : messages) {
                out.push_back(
                    {{"index", message.index},
                     {"level", enumToString(message.level)},
                     {"text", message.text}}
                );
            }
            return APITask::makeFinished(_makeTextContent(out.dump()));
        }
    };

    struct GetWorldApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_world"; }
        std::string description() const override { return "Get the whole world as rsc json."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            rsc::RSC_World world = impl._access.handle()->getWorld();
            RS_JSON out;
            out["name"] = world.name;
            out["simulation"] = _nodeToJson(*world.simulation);
            out["resources"] = _getSectionJson(world.resources);
            out["essences"] = _getSectionJson(world.essences);
            out["entities"] = _getSectionJson(world.entities);
            out["records"] = _getSectionJson(world.records);
            out["signals"] = _getSectionJson(world.signals);
            out["recorders"] = _getSectionJson(world.recorders);
            out["controllers"] = _getSectionJson(world.controllers);
            return APITask::makeFinished(_makeTextContent(out.dump()));
        }
    };

    struct GetWorldNameApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_world_name"; }
        std::string description() const override { return "Get the world name."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            return APITask::makeFinished(_makeTextContent(impl._access.handle()->name()));
        }
    };

    struct SetSimulationApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "set_simulation"; }
        std::string description() const override { return "Set the simulation from rsc json."; }
        std::string schema() const override { return _rsctSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(
                handle->setSimulation(_jsonToNode<rsc::RSC_Simulation>(params.at("rsct")))
            );
        }
    };

    struct AddResourceApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "add_resource"; }
        std::string description() const override { return "Add a resource from rsc json."; }
        std::string schema() const override { return _rsctSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(
                handle->addResource(_jsonToVariant<rsc::RSC_ResourceVariant>(params.at("rsct")))
            );
        }
    };

    struct RemoveResourceApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "remove_resource"; }
        std::string description() const override { return "Remove a resource by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->removeResource(_getName(params)));
        }
    };

    struct FindResourceApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "find_resource"; }
        std::string description() const override { return "Get a resource as rsc json by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(
                _makeVariantContent(impl._access.handle()->findResource(_getName(params)))
            );
        }
    };

    struct AddEssenceApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "add_essence"; }
        std::string description() const override { return "Add an essence from rsc json."; }
        std::string schema() const override { return _rsctSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(
                handle->addEssence(_jsonToVariant<rsc::RSC_EssenceVariant>(params.at("rsct")))
            );
        }
    };

    struct UpdateEssenceApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "update_essence"; }
        std::string description() const override {
            return "Replace an essence by name from rsc json.";
        }
        std::string schema() const override {
            return _schema({{"name", _prop("string")}, {"rsct", _prop("object")}}, {"name", "rsct"})
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->updateEssence(
                _getName(params), _jsonToVariant<rsc::RSC_EssenceVariant>(params.at("rsct"))
            ));
        }
    };

    struct RemoveEssenceApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "remove_essence"; }
        std::string description() const override { return "Remove an essence by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->removeEssence(_getName(params)));
        }
    };

    struct FindEssenceApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "find_essence"; }
        std::string description() const override { return "Get an essence as rsc json by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(
                _makeVariantContent(impl._access.handle()->findEssence(_getName(params)))
            );
        }
    };

    struct AddEntityApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "add_entity"; }
        std::string description() const override { return "Add an entity from rsc json."; }
        std::string schema() const override { return _rsctSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(
                handle->addEntity(_jsonToVariant<rsc::RSC_EntityVariant>(params.at("rsct")))
            );
        }
    };

    struct RemoveEntityApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "remove_entity"; }
        std::string description() const override { return "Remove an entity by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->removeEntity(_getName(params)));
        }
    };

    struct FindEntityApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "find_entity"; }
        std::string description() const override { return "Get an entity as rsc json by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(
                _makeVariantContent(impl._access.handle()->findEntity(_getName(params)))
            );
        }
    };

    struct FreezeEntityApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "freeze_entity"; }
        std::string description() const override { return "Freeze an entity by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->freezeEntity(_getName(params)));
        }
    };

    struct UnfreezeEntityApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "unfreeze_entity"; }
        std::string description() const override { return "Unfreeze an entity by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->unfreezeEntity(_getName(params)));
        }
    };

    struct IsEntityFrozenApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "is_entity_frozen"; }
        std::string description() const override {
            return "Get whether an entity is frozen by name.";
        }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(_makeTextContent(
                impl._access.handle()->isEntityFrozen(_getName(params)) ? "true" : "false"
            ));
        }
    };

    struct AddRecordApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "add_record"; }
        std::string description() const override { return "Add a record from rsc json."; }
        std::string schema() const override { return _rsctSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(
                handle->addRecord(_jsonToVariant<rsc::RSC_RecordVariant>(params.at("rsct")))
            );
        }
    };

    struct RemoveRecordApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "remove_record"; }
        std::string description() const override { return "Remove a record by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->removeRecord(_getName(params)));
        }
    };

    struct FindRecordApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "find_record"; }
        std::string description() const override { return "Get a record as rsc json by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(
                _makeVariantContent(impl._access.handle()->findRecord(_getName(params)))
            );
        }
    };

    struct AddSignalApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "add_signal"; }
        std::string description() const override { return "Add a signal from rsc json."; }
        std::string schema() const override { return _rsctSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(
                handle->addSignal(_jsonToVariant<rsc::RSC_SignalVariant>(params.at("rsct")))
            );
        }
    };

    struct RemoveSignalApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "remove_signal"; }
        std::string description() const override { return "Remove a signal by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->removeSignal(_getName(params)));
        }
    };

    struct FindSignalApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "find_signal"; }
        std::string description() const override { return "Get a signal as rsc json by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(
                _makeVariantContent(impl._access.handle()->findSignal(_getName(params)))
            );
        }
    };

    struct AddRecorderApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "add_recorder"; }
        std::string description() const override { return "Add a recorder from rsc json."; }
        std::string schema() const override { return _rsctSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(
                handle->addRecorder(_jsonToVariant<rsc::RSC_RecorderVariant>(params.at("rsct")))
            );
        }
    };

    struct RemoveRecorderApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "remove_recorder"; }
        std::string description() const override { return "Remove a recorder by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->removeRecorder(_getName(params)));
        }
    };

    struct FindRecorderApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "find_recorder"; }
        std::string description() const override { return "Get a recorder as rsc json by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(
                _makeVariantContent(impl._access.handle()->findRecorder(_getName(params)))
            );
        }
    };

    struct AddControllerApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "add_controller"; }
        std::string description() const override { return "Add a controller from rsc json."; }
        std::string schema() const override { return _rsctSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(
                handle->addController(_jsonToVariant<rsc::RSC_ControllerVariant>(params.at("rsct")))
            );
        }
    };

    struct RemoveControllerApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "remove_controller"; }
        std::string description() const override { return "Remove a controller by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            return APITask::fromWorldTask(handle->removeController(_getName(params)));
        }
    };

    struct FindControllerApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "find_controller"; }
        std::string description() const override { return "Get a controller as rsc json by name."; }
        std::string schema() const override { return _nameSchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(
                _makeVariantContent(impl._access.handle()->findController(_getName(params)))
            );
        }
    };

    /*****************************************************************
     * (b) Convenience utilities
     *****************************************************************/
    struct GetRsctCategoryApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_rsct_category"; }
        std::string description() const override {
            return "List the rsc types available to build, grouped by category.";
        }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            RS_JSON out = rsc::RS_RSCRegistry::getCategory();
            return APITask::makeFinished(_makeTextContent(out.dump()));
        }
    };

    struct GetRsctExampleApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_rsct_example"; }
        std::string description() const override {
            return "Get an example rsc node of a type, in the shape add and find use.";
        }
        std::string schema() const override {
            return _schema({{"type", _prop("string")}}, {"type"}).dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::makeFinished(
                _makeTextContent(_makeExample(_getString(params, "type")).dump())
            );
        }
    };

    struct GetWorldSummaryApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_world_summary"; }
        std::string description() const override {
            return "List the world's parts by category with their names and types.";
        }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            rsc::RSC_World world = impl._access.handle()->getWorld();
            RS_JSON summary;
            summary["name"] = world.name;
            summary["resources"] = _getSectionSummary(world.resources);
            summary["essences"] = _getSectionSummary(world.essences);
            summary["entities"] = _getSectionSummary(world.entities);
            summary["records"] = _getSectionSummary(world.records);
            summary["signals"] = _getSectionSummary(world.signals);
            summary["recorders"] = _getSectionSummary(world.recorders);
            summary["controllers"] = _getSectionSummary(world.controllers);
            return APITask::makeFinished(_makeTextContent(summary.dump()));
        }
    };

    struct AddCameraApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "add_camera"; }
        std::string description() const override {
            return "Add a perspective camera, frozen by default.";
        }
        std::string schema() const override {
            return _schema(
                       {{"name", _prop("string")},
                        {"kind", _prop("string")},
                        {"position", _propArray(3)},
                        {"orientation", _propArray(4)},
                        {"structure", _prop("string")},
                        {"anchor", _propArray(3)},
                        {"width", _prop("integer")},
                        {"height", _prop("integer")},
                        {"fov", _prop("number")},
                        {"samples", _prop("integer")},
                        {"max_depth", _prop("integer")},
                        {"frozen", _prop("boolean")}},
                       {"name"}
            )
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            std::string camera_name = _getName(params);
            std::string controller_name = camera_name + ".controller";
            auto handle = impl._access.handle();
            Transaction transaction(*handle);

            rsc::RSC_ControllerCallback controller;
            controller.name = controller_name;
            handle->addController(rsc::RSC_ControllerVariant(controller));

            if (_getString(params, "kind") == "mounted") {
                rsc::RSC_CameraPerspectiveRigidBodyLinked camera;
                camera.name = camera_name;
                camera.structure_name = _getString(params, "structure");
                camera.body_sid = 0;
                camera.segment_bid = 0;
                camera.relative_anchor = _getVec3(params, "anchor", RVec3rf::zero());
                camera.relative_orientation = _getQuat(params, "orientation", RQuat3rf::unit());
                _fillCamera(camera, params, controller_name);
                handle->addEntity(rsc::RSC_EntityVariant(camera));
            } else {
                rsc::RSC_CameraPerspectiveFree camera;
                camera.name = camera_name;
                camera.origin_position = _getVec3(params, "position", RVec3rf(0.0, 0.0, 0.4));
                camera.orientation = _getQuat(params, "orientation", RQuat3rf::unit());
                _fillCamera(camera, params, controller_name);
                handle->addEntity(rsc::RSC_EntityVariant(camera));
            }
            if (params.value("frozen", true))
                return APITask::fromWorldTask(handle->freezeEntity(camera_name));
            return APITask::makeFinished(_makeOkContent());
        }
    };

    /*****************************************************************
     * (c) Embodiment
     *****************************************************************/
    struct EnableCubeEmbodimentApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "enable_cube_embodiment"; }
        std::string description() const override {
            return "Add the AI's sky-blue glass cube body, fixed and floating, with a mounted "
                   "frozen eye and a proximity sense.";
        }
        std::string schema() const override {
            return _schema(
                       {{"size", _prop("integer")},
                        {"voxel_size", _prop("number")},
                        {"color", _propArray(3)},
                        {"position", _propArray(3)},
                        {"roughness", _prop("number")},
                        {"sensing_radius", _prop("number")}}
            ).dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            if (impl._is_cube_enabled)
                return APITask::makeFinished(
                    _makeTextContent(R"({"ok":false,"error":"cube embodiment already enabled"})")
                );
            RSize size = params.value("size", 8);
            RFloat voxel_size = params.value("voxel_size", 0.03);
            RColor3f color = _getColor(params, "color", _makeColor(142, 202, 230));
            RVec3rf position = _getVec3(params, "position", RVec3rf(0.0, 0.0, 1.0));
            RFloat roughness = params.value("roughness", 0.1);
            RFloat sensing_radius = params.value("sensing_radius", 10.0);

            std::string body = RS_MCP_BODY_NAME;
            std::string material = body + ".glass";
            std::string transform_controller = _getDriverName("transform", body);
            std::string eye_controller = std::string(RS_MCP_BODY_EYE_NAME) + ".controller";
            RFloat half = size * voxel_size * 0.5f;
            RFloat reach = size * voxel_size;

            auto handle = impl._access.handle();
            Transaction transaction(*handle);
            _addGlassMaterial(handle, material, color, roughness);

            rsc::RSC_ControllerCallback body_controller;
            body_controller.name = transform_controller;
            handle->addController(rsc::RSC_ControllerVariant(body_controller));

            rsc::RSC_StructureVoxel cube =
                _makeCubeStructure(body, material, size, voxel_size, position);
            cube.record->rigid_body_record_range =
                rsc::RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_ALL;
            cube.record->target_controller_reference = transform_controller;
            handle->addEntity(rsc::RSC_EntityVariant(cube));

            rsc::RSC_Transform transform_signal;
            transform_signal.name = transform_controller + ".signal";
            transform_signal.structure_name = body;
            transform_signal.source_controller_reference = transform_controller;
            handle->addSignal(rsc::RSC_SignalVariant(transform_signal));
            impl._markWired(transform_controller);

            rsc::RSC_IntersectRegionSphereRigidBodyLinked proximity;
            proximity.name = body + ".proximity";
            proximity.structure_name = body;
            proximity.body_sid = 0;
            proximity.segment_bid = 0;
            proximity.relative_anchor = RVec3rf(half, half, half);
            proximity.radius = sensing_radius;
            proximity.target_controller_reference = transform_controller;
            handle->addRecord(rsc::RSC_RecordVariant(proximity));

            rsc::RSC_ControllerCallback controller;
            controller.name = eye_controller;
            handle->addController(rsc::RSC_ControllerVariant(controller));

            rsc::RSC_CameraPerspectiveRigidBodyLinked eye;
            eye.name = RS_MCP_BODY_EYE_NAME;
            eye.structure_name = body;
            eye.body_sid = 0;
            eye.segment_bid = 0;
            eye.relative_anchor = RVec3rf(reach, 0.0, reach * 0.5);
            eye.image_width = 1024;
            eye.image_height = 1024;
            eye.field_of_view = 70.0;
            eye.record->has_rgb = true;
            eye.record->target_controller_reference = eye_controller;
            eye.signal->source_controller_reference = eye_controller;
            handle->addEntity(rsc::RSC_EntityVariant(eye));

            impl._cube_extent = RVec3rf(reach, reach, reach);
            impl._cube_eye_anchor = eye.relative_anchor;
            impl._is_cube_enabled = true;
            return APITask::fromWorldTask(handle->freezeEntity(RS_MCP_BODY_EYE_NAME));
        }
    };

    struct DisableCubeEmbodimentApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "disable_cube_embodiment"; }
        std::string description() const override { return "Remove the whole embodiment bundle."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            std::string body = RS_MCP_BODY_NAME;
            std::string material = body + ".glass";
            auto handle = impl._access.handle();
            impl._is_cube_enabled = false;
            impl._clearCubeMotion();
            Transaction transaction(*handle);
            handle->removeEntity(RS_MCP_BODY_EYE_NAME);
            handle->removeController(std::string(RS_MCP_BODY_EYE_NAME) + ".controller");
            handle->removeRecord(body + ".proximity");
            handle->removeEntity(body);
            handle->removeEssence(material + ".optic");
            handle->removeEssence(material + ".phys_rigid");
            handle->removeEssence(material + ".phys_fixed");
            handle->removeEssence(material);
            handle->removeResource(material + ".rm");
            handle->removeResource(material + ".albedo");
            std::string transform_controller = _getDriverName("transform", body);
            handle->removeSignal(transform_controller + ".signal");
            handle->removeController(transform_controller);
            {
                std::scoped_lock lock(impl._control_mutex);
                impl._wired_controllers.erase(transform_controller);
            }
            return APITask::makeFinished(_makeOkContent());
        }
    };

    struct GetCubePoseApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_cube_pose"; }
        std::string description() const override {
            return "Get the cube body's live position, orientation, size, and eye direction.";
        }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            return APITask::fromFuture(impl._queueCubeRead(CubeReadKind::POSE));
        }
    };

    struct GetCubeBoundsApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_cube_bounds"; }
        std::string description() const override {
            return "Get the cube body's live world-axis-aligned bounding box.";
        }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            return APITask::fromFuture(impl._queueCubeRead(CubeReadKind::BOUNDS));
        }
    };

    struct GetCubeSurroundingApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_cube_surrounding"; }
        std::string description() const override {
            return "List the structures the body's proximity sense feels, nearest first.";
        }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            return APITask::fromFuture(impl._queueCubeRead(CubeReadKind::SURROUNDING));
        }
    };

    struct GetCubeTrackingApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_cube_tracking"; }
        std::string description() const override {
            return "Get whether the cube is auto-following or auto-orbiting, and on what.";
        }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            return APITask::makeFinished(impl._makeCubeTrackingJson());
        }
    };

    struct MoveCubeApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "move_cube"; }
        std::string description() const override {
            return "Kick the cube body with a one-shot translation, rotation, and pivot.";
        }
        std::string schema() const override {
            return _schema(
                       {{"translation", _propArray(3)},
                        {"rotation", _propArray(4)},
                        {"pivot", _propArray(3)}}
            ).dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            impl._clearCubeMotion();
            impl._queueTransform(RS_MCP_BODY_NAME, params);
            return APITask::makeFinished(_makeTextContent("queued"));
        }
    };

    struct LookCubeAtApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "look_cube_at"; }
        std::string description() const override {
            return "Turn the cube so its eye faces a world point, staying upright.";
        }
        std::string schema() const override {
            return _schema({{"point", _propArray(3)}, {"duration", _prop("number")}}, {"point"})
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            RVec3rf point = _getVec3(params, "point", RVec3rf::zero());
            RFloat duration = params.value("duration", 0.0);
            if (duration <= 0.0f) {
                impl._queueCubeGlide(true, point, 0.0f, nullptr);
                return APITask::makeFinished(_makeTextContent("queued"));
            }
            auto promise = std::make_shared<std::promise<RS_JSON>>();
            std::shared_future<RS_JSON> result = promise->get_future().share();
            impl._queueCubeGlide(true, point, duration, std::move(promise));
            return APITask::fromFuture(result);
        }
    };

    struct TeleportCubeToApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "teleport_cube_to"; }
        std::string description() const override {
            return "Move the cube to a world point, instantly or gliding over a span of seconds.";
        }
        std::string schema() const override {
            return _schema({{"point", _propArray(3)}, {"duration", _prop("number")}}, {"point"})
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            RVec3rf point = _getVec3(params, "point", RVec3rf::zero());
            RFloat duration = params.value("duration", 0.0);
            if (duration <= 0.0f) {
                impl._queueCubeGlide(false, point, 0.0f, nullptr);
                return APITask::makeFinished(_makeTextContent("queued"));
            }
            auto promise = std::make_shared<std::promise<RS_JSON>>();
            std::shared_future<RS_JSON> result = promise->get_future().share();
            impl._queueCubeGlide(false, point, duration, std::move(promise));
            return APITask::fromFuture(result);
        }
    };

    struct AutoCubeTrackAtApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "auto_cube_track_at"; }
        std::string description() const override {
            return "Hold the cube at a distance from a surrounding structure and keep facing it.";
        }
        std::string schema() const override {
            return _schema(
                       {{"target", _prop("string")}, {"distance", _prop("number")}},
                       {"target", "distance"}
            )
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            std::string target = _getString(params, "target");
            RIndex gid = impl._access.handle()->findEntityGID(target);
            if (gid == RS_NULL_INDEX)
                return APITask::makeFinished(_makeErrorContent("unknown target: " + target));
            impl._setCubeTrack(gid, target, params.value("distance", 0.0));
            return APITask::makeFinished(_makeTextContent("tracking"));
        }
    };

    struct AutoCubeOrbitAroundApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "auto_cube_orbit_around"; }
        std::string description() const override {
            return "Circle the cube around a world point, facing it, one turn per period.";
        }
        std::string schema() const override {
            return _schema(
                       {{"point", _propArray(3)},
                        {"radius", _prop("number")},
                        {"height", _prop("number")},
                        {"period", _prop("number")}},
                       {"point", "radius", "period"}
            )
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            RVec3rf point = _getVec3(params, "point", RVec3rf::zero());
            impl._setCubeOrbit(
                point,
                params.value("radius", 0.0),
                params.value("height", 0.0),
                params.value("period", 0.0)
            );
            return APITask::makeFinished(_makeTextContent("orbiting"));
        }
    };

    struct TakeCubeEyeImageApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "take_cube_eye_image"; }
        std::string description() const override {
            return "Capture one PNG frame from the cube's mounted eye.";
        }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            RS_JSON args;
            args["camera"] = RS_MCP_BODY_EYE_NAME;
            return APITask::fromFuture(impl._queueCapture(args, 1));
        }
    };

    struct TakeCubeEyeBurstApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "take_cube_eye_burst"; }
        std::string description() const override {
            return "Capture a filmstrip of PNG frames from the cube's mounted eye.";
        }
        std::string schema() const override {
            return _schema({{"count", _prop("integer")}, {"step_stride", _prop("integer")}}).dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            if (!impl._is_cube_enabled) return _makeCubeNotEnabledErrorTask();
            RS_JSON args = params;
            args["camera"] = RS_MCP_BODY_EYE_NAME;
            return APITask::fromFuture(impl._queueCapture(args, params.value("count", 4)));
        }
    };

    /*****************************************************************
     * (d) Signal send and result record
     *****************************************************************/
    struct ApplyTransformApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "apply_transform"; }
        std::string description() const override {
            return "Kick a structure with a one-shot translation, rotation, and pivot.";
        }
        std::string schema() const override {
            return _schema(
                       {{"structure", _prop("string")},
                        {"translation", _propArray(3)},
                        {"rotation", _propArray(4)},
                        {"pivot", _propArray(3)}},
                       {"structure"}
            )
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            impl._queueTransform(_getString(params, "structure"), params);
            return APITask::makeFinished(_makeTextContent("queued"));
        }
    };

    struct ApplyForceApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "apply_force"; }
        std::string description() const override { return "Apply a force vector to a structure."; }
        std::string schema() const override {
            return _schema(
                       {{"structure", _prop("string")}, {"vector", _propArray(3)}},
                       {"structure", "vector"}
            )
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            impl._queueForce(_getString(params, "structure"), params);
            return APITask::makeFinished(_makeTextContent("queued"));
        }
    };

    struct ApplyTorqueApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "apply_torque"; }
        std::string description() const override { return "Apply a torque vector to a structure."; }
        std::string schema() const override {
            return _schema(
                       {{"structure", _prop("string")}, {"vector", _propArray(3)}},
                       {"structure", "vector"}
            )
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            impl._queueTorque(_getString(params, "structure"), params);
            return APITask::makeFinished(_makeTextContent("queued"));
        }
    };

    struct SetJointApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "set_joint"; }
        std::string description() const override {
            return "Set a structure's hinge or prismatic joint target.";
        }
        std::string schema() const override {
            return _schema(
                       {{"structure", _prop("string")},
                        {"joint", _prop("integer")},
                        {"target", _prop("number")},
                        {"kind", _prop("string")}},
                       {"structure", "joint", "target"}
            )
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            std::string controller = impl._getVoxelControllerName(_getString(params, "structure"));
            if (controller.empty())
                return APITask::makeFinished(
                    _makeErrorContent("structure has no joint controller")
                );
            PendingJoint joint;
            joint.index = params.value("joint", 0);
            joint.target = params.value("target", 0.0);
            joint.is_prismatic = _getString(params, "kind") == "prismatic";
            impl._pushJoint(controller, joint);
            return APITask::makeFinished(_makeTextContent("queued"));
        }
    };

    struct MoveCameraApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "move_camera"; }
        std::string description() const override {
            return "Set a free camera's position and orientation.";
        }
        std::string schema() const override {
            return _schema(
                       {{"name", _prop("string")},
                        {"position", _propArray(3)},
                        {"orientation", _propArray(4)}},
                       {"name"}
            )
                .dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            impl._queueCameraPose(_getName(params), params);
            return APITask::makeFinished(_makeTextContent("queued"));
        }
    };

    struct TakeImageApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "take_image"; }
        std::string description() const override {
            return "Freeze, capture one frame, and refreeze a camera as a PNG image.";
        }
        std::string schema() const override {
            return _schema({{"camera", _prop("string")}}).dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::fromFuture(impl._queueCapture(params, 1));
        }
    };

    struct TakeBurstApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "take_burst"; }
        std::string description() const override {
            return "Capture a filmstrip of frames from a camera over the next steps.";
        }
        std::string schema() const override {
            return _schema(
                       {{"camera", _prop("string")},
                        {"count", _prop("integer")},
                        {"step_stride", _prop("integer")}}
            ).dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            return APITask::fromFuture(impl._queueCapture(params, params.value("count", 4)));
        }
    };

    /*****************************************************************
     * (e) Registered external tools
     *****************************************************************/
    /**
     * @brief A Python authored tool: its call invokes the registered handler with the full handle.
     */
    struct RegisteredTool : APIBase {
        std::string _name;
        std::string _description;
        std::string _schema;
        RS_MCPToolHandler _handler;

        RegisteredTool(
            RS_MCPImpl &impl,
            std::string name,
            std::string description,
            std::string schema,
            RS_MCPToolHandler handler
        )
            : APIBase(impl),
              _name(std::move(name)),
              _description(std::move(description)),
              _schema(std::move(schema)),
              _handler(std::move(handler)) {}

        std::string name() const override { return _name; }
        std::string description() const override { return _description; }
        std::string schema() const override { return _schema; }
        bool isRegistered() const override { return true; }
        APITask call(const RS_JSON &params) override {
            RS_JSON result =
                RS_JSON::parse(_handler(impl._access.handle(), params.dump()), nullptr, false);
            if (result.is_discarded()) {
                return APITask::makeFinished(
                    _makeErrorContent("registered tool returned invalid json")
                );
            }
            return APITask::makeFinished(result);
        }
    };

    /*****************************************************************
     * (f) Communication
     *****************************************************************/
    struct GetUserPositionApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_user_position"; }
        std::string description() const override { return "Get the user's latest world position."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            return APITask::makeFinished(impl._drainUserPosition());
        }
    };

    struct GetUserHandAimsApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "get_hand_aims"; }
        std::string description() const override { return "Get the user's latest hand aim rays."; }
        std::string schema() const override { return _emptySchema().dump(); }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &) override {
            return APITask::makeFinished(impl._drainHandAims());
        }
    };

    struct ReadChannelApi : APIBase {
        using APIBase::APIBase;
        std::string name() const override { return "read_channel"; }
        std::string description() const override {
            return "Get the latest typed value on a known data channel.";
        }
        std::string schema() const override {
            return _schema({{"channel", _prop("string")}}, {"channel"}).dump();
        }
        bool isRegistered() const override { return false; }
        APITask call(const RS_JSON &params) override {
            std::string channel = _getString(params, "channel");
            if (channel == RS_VR_CHANNEL_USER_POSITION)
                return APITask::makeFinished(impl._drainUserPosition());
            if (channel == RS_VR_CHANNEL_USER_HAND_AIMS)
                return APITask::makeFinished(impl._drainHandAims());
            return APITask::makeFinished(_makeErrorContent("unknown channel: " + channel));
        }
    };
};

/*****************************************************************
 * RS_WorldSessionComponentMCP
 *****************************************************************/
RS_WorldSessionComponentMCP::RS_WorldSessionComponentMCP(
    std::string name,
    std::string host,
    int port,
    std::string token,
    bool is_python_exec_enabled
)
    : _name(std::move(name)),
      _host(std::move(host)),
      _port(port),
      _token(std::move(token)),
      _is_python_exec_enabled(is_python_exec_enabled) {}

RS_WorldSessionComponentMCP::~RS_WorldSessionComponentMCP() = default;

void RS_WorldSessionComponentMCP::registerTool(
    std::string name,
    std::string description,
    std::string input_schema,
    RS_MCPToolHandler handler
) {
    if (_impl) {
        _impl->registerTool(
            std::move(name), std::move(description), std::move(input_schema), std::move(handler)
        );
    }
}

void RS_WorldSessionComponentMCP::unregisterTool(const std::string &name) {
    if (_impl) _impl->unregisterTool(name);
}

std::vector<std::string> RS_WorldSessionComponentMCP::getRegisteredToolNames() const {
    return _impl ? _impl->getRegisteredToolNames() : std::vector<std::string>{};
}

void RS_WorldSessionComponentMCP::onAttach(RS_WorldSessionAccess access) {
    _access = access;
    _impl = std::make_unique<RS_MCPImpl>(
        std::move(access), _host, _port, _token, _is_python_exec_enabled
    );
}

void RS_WorldSessionComponentMCP::onCallback(RBigFloat time, RS_ThreadPool &pool) {
    pool.detachTask([this] { _impl->syncControllers(); });
}

void RS_WorldSessionComponentMCP::onDetach() { _impl.reset(); }

}  // namespace rise

#endif  // RISE_ORIGIN_BUILD_CORE_WITH_MCP
