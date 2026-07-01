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
        RFloat focus_distance = 1.0
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
 * @details It owns an RS_VRImpl whose present thread runs the OpenXR lifecycle and connects whenever
 * a headset comes up, at start or later. While a headset is live it injects one stereo camera sized
 * to the per-eye resolution, and each control step converts the latest stereo record into its own
 * color and depth buffers, hands them to the present thread, and writes the latest per-eye head pose
 * and frustum back through the camera signal. When the headset drops it removes the camera and waits
 * for the next connection. The render rate is the world's control rate, and the runtime reprojects
 * from the submitted depth.
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
    // Staged presence of the VR characterization in the simulation, advanced only from a handle query.
    enum class VRState { NOT_CONNECTED, ADDING_RSC, CONNECTED, REMOVING_RSC };

    // The last physical eye pose RS_VRImpl issued to render from, handed back to it so the
    // projection layer is labeled with the physical head.
    struct EyePose {
        RVec3rf position;
        RQuat3rf orientation;
    };

    // Guards against a second VR component; OpenXR drives a single headset session per process.
    static std::atomic<bool> _is_created;

    std::string _name;
    std::string _controller_name;
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
    std::atomic<VRState> _vr_state{VRState::NOT_CONNECTED};

    std::optional<RS_WorldSessionAccess> _access;
    std::unique_ptr<RS_VRImpl> _xr;

    // Converted color and copied depth, one buffer per eye.
    std::shared_ptr<RColor4b[]> _frame_rgba[2];
    std::shared_ptr<float[]> _frame_depth[2];
    // True while a convert and publish pipeline is in flight, so buffers are not overwritten in use.
    std::atomic<bool> _is_publishing_frame{false};

    void _syncVRCharacterization();
    void _syncManipulationCharacterization();
    void _convertFrame(RIndex eye, RIndex pixel_start, RIndex pixel_end);
    void _syncFrame();
    void _writeCameraSignal();
    void _writeManipulationSignal();
    void _updateInfo(RBigFloat time);
    [[nodiscard]] std::string
    _pickTarget(const RVec3rf &aim_origin, const RVec3rf &aim_direction);
    [[nodiscard]] rsc::RSC_EntityVariant _makeCamera() const;
    [[nodiscard]] rsc::RSC_ControllerVariant _makeController(std::string name) const;
};

#endif  // RISE_ORIGIN_BUILD_CORE_WITH_VR

}  // namespace rise

#endif  // RS_WORLD_SESSION_COMPONENT_H
