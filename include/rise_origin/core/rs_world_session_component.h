// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_SESSION_COMPONENT_H
#define RS_WORLD_SESSION_COMPONENT_H

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "rise_origin/core/rs_world_session.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

/**
 * @brief Runs a user callback over the controllers in its scope each step.
 */
class RS_WorldSessionComponentGeneric : public RS_WorldSessionComponent {
   public:
    using Callback = std::function<
        void(RBigFloat time, std::vector<std::reference_wrapper<RS_WorldController>> controllers)>;

    explicit RS_WorldSessionComponentGeneric(Callback callback = {});

    /**
     * @brief This component's scoped world handle.
     */
    [[nodiscard]] std::shared_ptr<RS_WorldSessionHandle> getHandle() const;

    /**
     * @brief Set the per-step callback; replaces any previous one.
     */
    void setCallback(Callback callback);

    void onAttach(RS_WorldSessionAccess access) override;
    void onCallback(RBigFloat time, RS_ThreadPool &pool) override;
    void onDetach() override;

   private:
    Callback _callback;
    std::optional<RS_WorldSessionAccess> _access;
};

/**
 * @brief Orbit-camera component: injects one free perspective camera, publishes its RGB frame each
 * step, and drives its pose from accumulated mouse gestures.
 */
class RS_WorldSessionComponentCamera : public RS_WorldSessionComponent {
   public:
    using Callback = std::function<void(RBigFloat time)>;

    RS_WorldSessionComponentCamera(
        std::string name,
        Callback callback = {},
        RVec3rf position = RVec3rf(0.0, 0.0, 0.4),
        RVec3rf focus = RVec3rf(0.0, 0.0, 0.15),
        RSize width = 1280,
        RSize height = 720,
        RSize samples_per_pixel = 4,
        RSize max_depth = 3,
        RFloat field_of_view = 60.0,
        RFloat defocus_angle = 0.0,
        RFloat focus_distance = 1.0,
        RFloat min_distance = 0.05,
        RFloat max_distance = RS_MAX_FLOAT,
        RFloat zoom_factor = 0.95
    );

    void onAttach(RS_WorldSessionAccess access) override;
    void onCallback(RBigFloat time, RS_ThreadPool &pool) override;
    void onDetach() override;

    /**
     * @brief Accumulate a mouse gesture; applied on the next control step.
     */
    void move(RFloat orbit_pitch, RFloat orbit_yaw, RFloat pan_x, RFloat pan_y, RFloat zoom);

    /**
     * @brief Snapshot the latest RGB frame; thread-safe against the step that publishes it.
     */
    std::shared_ptr<const RColor3b[]> frame(RSize &width, RSize &height) const;

   private:
    std::string _name;
    std::string _controller_name;

    Callback _callback;

    RSize _width;
    RSize _height;
    RSize _samples_per_pixel;
    RSize _max_depth;
    RFloat _field_of_view;
    RFloat _defocus_angle;
    RFloat _focus_distance;

    std::optional<RS_WorldSessionAccess> _access;

    // Double-buffered RGB frames: one is published to frame(), the other is filled by the parallel
    // copy jobs; the publish job flips _front between them.
    std::shared_ptr<RColor3b[]> _frames[2];
    RIndex _front = 0;
    mutable std::mutex _frame_mutex;

    // Orbit camera pose (absolute, recomputed from pending gestures each step).
    RVec3rf _position;
    RVec3rf _focus;
    RQuat3rf _orientation;
    RFloat _distance;

    // The range the orbit distance is held in, enforced every step.
    RFloat _min_distance;
    RFloat _max_distance;

    // What one wheel notch scales the orbit distance by.
    RFloat _zoom_factor;

    // Pending gesture deltas accumulated by move() on the GUI thread.
    std::mutex _pending_mutex;
    RFloat _pending_orbit_pitch = 0;
    RFloat _pending_orbit_yaw = 0;
    RFloat _pending_pan_x = 0;
    RFloat _pending_pan_y = 0;
    RFloat _pending_zoom = 0;

    void _copyFrame(RIndex pixel_start, RIndex pixel_end);
    void _publishFrame();
    void _applyPending();
    void _applyZoom();
    void _applyPan();
    void _applyOrbit();
    void _writeCameraSignal();
    void _publishUserView();
    static RQuat3rf _lookAt(const RVec3rf &eye, const RVec3rf &at);
    [[nodiscard]] rsc::RSC_EntityVariant _makeCamera() const;
    [[nodiscard]] rsc::RSC_ControllerVariant _makeController() const;
};

#ifdef RISE_ORIGIN_BUILD_CORE_WITH_VR

/**
 * @brief Heavy OpenXR and Vulkan state, hidden behind a pimpl resolved in the component's source.
 */
class RS_VRImpl;

/**
 * @brief Stereo VR component: drives an OpenXR runtime over a Vulkan binding.
 * @details It owns an RS_VRImpl whose present thread runs the OpenXR lifecycle and connects
 * whenever a headset comes up, at start or later. While a headset is live it injects one stereo
 * camera sized to the per-eye resolution, and each control step converts the latest stereo record
 * into its own color and depth buffers, hands them to the present thread, and writes the latest
 * per-eye head pose and frustum back through the camera signal. When the headset drops it removes
 * the camera and waits for the next connection. The render rate is the world's control rate, and
 * the runtime reprojects from the submitted depth.
 */
class RS_WorldSessionComponentVR : public RS_WorldSessionComponent {
   public:
    RS_WorldSessionComponentVR(
        std::string name,
        RSize samples_per_pixel = 1,
        RSize max_depth = 3,
        RFloat near_depth = 0.05,
        RFloat far_depth = 100.0,
        RFloat move_speed = 2.0,
        RFloat fly_speed = 1.5,
        RFloat turn_speed = 90.0,
        RFloat stick_dead_zone = 0.2,
        RFloat boost = 3.0,
        RFloat hud_follow_tau = 0.25
    );
    ~RS_WorldSessionComponentVR() override;

    /**
     * @brief Set the newtons applied along the aim ray at a full trigger pull.
     */
    void setPushForce(RFloat push_force);

    void onAttach(RS_WorldSessionAccess access) override;
    void onCallback(RBigFloat time, RS_ThreadPool &pool) override;
    void onDetach() override;

   private:
    // Staged presence of the VR characterization in the simulation, advanced only from a handle
    // query.
    enum class VRState { IDLE, ENGAGING, ENGAGED, DISENGAGING };

    // Manipulation lifecycle: engage the targeted structure, carry or push it, then tear it down.
    enum class ManipulationState { IDLE, ENGAGING, ENGAGED, DISENGAGING };

    // The last physical eye pose RS_VRImpl issued to render from, handed back to it so the
    // projection layer is labeled with the physical head.
    struct EyePose {
        RVec3rf position;
        RQuat3rf orientation;
    };

    // Guards against a second VR component; OpenXR drives a single headset session per process.
    static std::atomic<bool> _is_created;

    std::optional<RS_WorldSessionAccess> _access;
    std::unique_ptr<RS_VRImpl> _xr;

    /**** VR main ****/
    std::string _name;
    std::string _vr_controller_name;
    std::string _ray_controller_name[2];
    std::string _ray_sensor_name[2];
    RSize _samples_per_pixel;
    RSize _max_depth;
    RFloat _near_depth;
    RFloat _far_depth;
    RFloat _move_speed;
    RFloat _fly_speed;
    RFloat _turn_speed;
    RFloat _stick_dead_zone;
    RFloat _boost;
    RFloat _hud_follow_tau;
    std::atomic<RFloat> _push_force{50.0};

    // Eye data.
    RSize _eye_width = 0;
    RSize _eye_height = 0;
    // Last eye pose issued to render from, handed back to label the projection layer.
    EyePose _eye_pose[2];

    // Staged presence of the VR characterization, advanced from a per-callback handle query.
    std::atomic<VRState> _vr_state{VRState::IDLE};

    // True while a convert and publish pipeline is in flight, so the borrowed staging is not
    // overwritten in use.
    std::atomic<bool> _is_publishing_frame{false};

    /**** Manipulation ****/
    RIndex _highlight_gid[2] = {RS_NULL_INDEX, RS_NULL_INDEX};
    ManipulationState _manip_state[2] = {ManipulationState::IDLE, ManipulationState::IDLE};
    bool _is_target_frozen[2] = {false, false};
    RIndex _target_gid[2] = {RS_NULL_INDEX, RS_NULL_INDEX};
    RIndex _target_body_sid[2] = {RS_NULL_INDEX, RS_NULL_INDEX};
    RIndex _target_segment_bid[2] = {RS_NULL_INDEX, RS_NULL_INDEX};
    std::string _manipulation_controller_name[2];
    std::string _target_structure[2];
    RVec3rf _hand_prev_position[2];
    RQuat3rf _hand_prev_orientation[2];
    std::vector<RS_WorldTask> _pending_tasks[2];

    void _syncVRCharacterization();
    void _syncManipulationCharacterization(RIndex hand);
    void _convertAndWriteFrame(RIndex eye, RIndex pixel_start, RIndex pixel_end);
    void _commitFrame();
    void _writeCameraSignal();
    void _writeRaySignal();
    void _writeManipulationSignal(RIndex hand);
    void _updateRay();
    void _updateInfo(RBigFloat time);
    void _publishUserView();

    [[nodiscard]] bool
    _readRayHit(RIndex hand, RIndex &gid, RFloat &distance, RIndex &body_sid, RIndex &segment_bid);
    [[nodiscard]] rsc::RSC_EntityVariant _makeCamera() const;
    [[nodiscard]] rsc::RSC_EntityVariant
    _makeRaySensor(std::string name, std::string controller_name) const;
    [[nodiscard]] rsc::RSC_ControllerVariant _makeController(std::string name) const;
};

#endif  // RISE_ORIGIN_BUILD_CORE_WITH_VR

#ifdef RISE_ORIGIN_BUILD_CORE_WITH_MCP

/**
 * @brief The server state hidden behind the MCP component's pimpl.
 */
class RS_MCPImpl;

/**
 * @brief The signature of a Python authored tool: full handle plus argument json in, result json out.
 */
using RS_MCPToolHandler =
    std::function<std::string(std::shared_ptr<RS_WorldSessionHandle> handle, std::string arguments)>;

/**
 * @brief MCP server component that lets an AI client read, change, see, and be embodied in the world.
 * @details It hosts an http server on its own thread that speaks the Model Context Protocol. Each tool
 * call runs on a single executor thread and gets an id; a fast answer returns at once, otherwise the
 * id is probed with get_result. A step task writes the buffered control signals and reads out capture
 * frames. The tool surface has five kinds: a one to one mirror of the engine, server side helpers,
 * signal send and capture, user communication, and Python registered tools.
 */
class RS_WorldSessionComponentMCP : public RS_WorldSessionComponent {
   public:
    RS_WorldSessionComponentMCP(
        std::string name,
        std::string host = "127.0.0.1",
        int port = 8765,
        std::string token = "",
        bool is_python_exec_enabled = false
    );
    ~RS_WorldSessionComponentMCP() override;

    /**
     * @brief Register a Python authored tool served next to the built in ones.
     */
    void registerTool(
        std::string name, std::string description, std::string input_schema, RS_MCPToolHandler handler
    );

    /**
     * @brief Remove a registered tool by name.
     */
    void unregisterTool(const std::string &name);

    /**
     * @brief Get the names of every registered tool.
     */
    std::vector<std::string> getRegisteredToolNames() const;

    void onAttach(RS_WorldSessionAccess access) override;
    void onCallback(RBigFloat time, RS_ThreadPool &pool) override;
    void onDetach() override;

   private:
    std::string _name;
    std::string _host;
    int _port;
    std::string _token;
    bool _is_python_exec_enabled;

    std::optional<RS_WorldSessionAccess> _access;
    std::unique_ptr<RS_MCPImpl> _impl;
};

#endif  // RISE_ORIGIN_BUILD_CORE_WITH_MCP

}  // namespace rise

#endif  // RS_WORLD_SESSION_COMPONENT_H
