// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_world_session_component.h"

#ifdef RISE_ORIGIN_BUILD_CORE_WITH_VR

#include <vulkan/vulkan.h>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"
#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/core/world_session_component/rs_world_session_component_vr_data.h"
#include "rise_origin/rsc/controller/rsc_controller.h"
#include "rise_origin/rsc/entity/rsc_camera.h"
#include "rise_origin/rsc/entity/rsc_ray_sensor.h"
#include "rise_origin/rsc/signal/rsc_mechanical.h"
#include "rise_origin/utils/math/rs_mat.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_logging.h"

namespace rise {
// Log an OpenXR failure and throw; the present thread catches and recovers to the waiting state.
#define RS_XR_CHECK(call)                                                                      \
    do {                                                                                       \
        XrResult _result = (call);                                                             \
        if (XR_FAILED(_result)) {                                                              \
            RS_LOG(_logger, error, "OpenXR call failed ({}): {}", #call, (int)_result);        \
            RS_THROW(RS_RuntimeException("OpenXR call failed ({}): {}", #call, (int)_result)); \
        }                                                                                      \
    } while (0)

// Log a Vulkan failure and throw.
#define RS_VK_CHECK(call)                                                                      \
    do {                                                                                       \
        VkResult _result = (call);                                                             \
        if (_result != VK_SUCCESS) {                                                           \
            RS_LOG(_logger, error, "Vulkan call failed ({}): {}", #call, (int)_result);        \
            RS_THROW(RS_RuntimeException("Vulkan call failed ({}): {}", #call, (int)_result)); \
        }                                                                                      \
    } while (0)

// Connection stage of the present thread.
RS_ENUM(
    RSE_VRState,
    unsigned int,
    (RSE_VR_STATE_INACTIVE, 0),
    (RSE_VR_STATE_WAITING, 1),
    (RSE_VR_STATE_RUNNING, 2)
)

// Pixels converted per parallel block, per eye.
constexpr RSize RS_VR_CONVERT_BLOCK = 65536;

// The HUD content is laid out at this logical resolution and scaled to the swapchain image size, so
// the image resolution can change without retuning the layout.
constexpr float RS_VR_HUD_REFERENCE_WIDTH = 2688.0f;
constexpr float RS_VR_HUD_REFERENCE_HEIGHT = 448.0f;

// A swapchain image normally releases within a frame, so a wait beyond one second means the
// compositor has stalled; the present thread then recovers the session instead of hanging forever.
constexpr XrDuration RS_VR_SWAPCHAIN_WAIT_TIMEOUT = 1'000'000'000;

// The aim ray atlas is three square cells side by side: beam strip, miss reticle, hit reticle.
constexpr int32_t RS_VR_RAY_CELL = 64;

/*****************************************************************
 * RS_VRImpl
 *****************************************************************/
/**
 * @brief OpenXR and Vulkan state for the stereo VR component, hidden behind the component's pimpl.
 * @details A single present thread owns the whole OpenXR lifecycle: it brings up the instance,
 * waits for a headset to connect at any time, runs the frame loop, and recovers to the waiting
 * state on disconnect or loss. The component fills the eye staging buffers directly each control
 * step and commits through commitFrame, and reads back the latest per-eye pose through getEyes.
 */
class RS_VRImpl {
   public:
    /****************************************************************
     * Types
     ****************************************************************/
    /**
     * @brief Which manipulation gesture colors a hand's beam and reticle.
     */
    enum class RayMode { NEUTRAL, GRAB, PUSH };

    /**
     * @brief One eye's head pose and asymmetric frustum, in rise-origin coordinates.
     */
    struct EyePose {
        /// Eye position after locomotion.
        RVec3rf position;

        /// Eye orientation after locomotion.
        RQuat3rf orientation;

        /// Physical eye position before locomotion.
        RVec3rf raw_position;

        /// Physical eye orientation before locomotion.
        RQuat3rf raw_orientation;

        /// Frustum left angle.
        RFloat angle_left = 0;

        /// Frustum right angle.
        RFloat angle_right = 0;

        /// Frustum up angle.
        RFloat angle_up = 0;

        /// Frustum down angle.
        RFloat angle_down = 0;
    };

    /**
     * @brief The physical pose one eye was rendered from, that the component hands back to label
     * the projection layer with.
     */
    struct FramePose {
        /// Physical eye position, rise-origin coordinates.
        RVec3rf position;

        /// Physical eye orientation, rise-origin coordinates.
        RQuat3rf orientation;
    };

    /**
     * @brief One hand's high-level intent, in rise-origin coordinates.
     */
    struct HandIntent {
        RVec3rf position;
        RQuat3rf orientation;
        RVec3rf aim_origin;
        RQuat3rf aim_orientation;
        RFloat push_amount = 0;
        bool is_aim_valid = false;
        bool is_grab = false;
        bool is_push = false;
    };

    /**
     * @brief The header metrics the component publishes to the HUD each step.
     */
    struct HudInfo {
        RFloat sim_time = 0;
        RFloat world_load = 0;
        bool is_world_load_valid = false;
        RBigSize device_used = 0;
        RBigSize device_total = 0;
        RBigSize host_used = 0;
        RBigSize host_total = 0;
        std::string device_name;
    };

    /****************************************************************
     * Lifecycle
     ****************************************************************/
    RS_VRImpl(
        RFloat near_depth,
        RFloat far_depth,
        RFloat move_speed,
        RFloat fly_speed,
        RFloat turn_speed_degrees,
        RFloat stick_dead_zone,
        RFloat boost,
        RFloat hud_follow_tau,
        RFloat ctrl_ray_beam_width = 0.01,
        RFloat ctrl_ray_beam_length = 0.4,
        RFloat ctrl_ray_reticle_fraction = 0.03,
        RFloat ctrl_ray_reticle_min = 0.02,
        RFloat ctrl_ray_reach = 8.0,
        RFloat ctrl_marker_size = 0.09
    )
        : _near_depth(near_depth),
          _far_depth(far_depth),
          _move_speed(move_speed),
          _fly_speed(fly_speed),
          _turn_speed(turn_speed_degrees * RFloat(RS_PI) / RFloat(180.0)),
          _stick_dead_zone(stick_dead_zone),
          _boost(boost),
          _hud_follow_tau(hud_follow_tau),
          _ctrl_ray_beam_width(ctrl_ray_beam_width),
          _ctrl_ray_beam_length(ctrl_ray_beam_length),
          _ctrl_ray_reticle_fraction(ctrl_ray_reticle_fraction),
          _ctrl_ray_reticle_min(ctrl_ray_reticle_min),
          _ctrl_ray_reach(ctrl_ray_reach),
          _ctrl_marker_size(ctrl_marker_size) {
        if (logging::get("root")) _logger = logging::get("root")->clone("vr service");
        _basis = _makeBasisChange();
        _is_alive = true;
        _vr_present_thread = std::thread([this] { _vrPresentThreadFunc(); });
    }

    ~RS_VRImpl() {
        stop();
        if (_vr_present_thread.joinable()) _vr_present_thread.join();
        _freeSession();
        _freeInstance();
        _drainPending();
    }

    /**
     * @brief Prompt the present thread to stop so a caller can join it promptly.
     * @details xrRequestExitSession unblocks a present thread parked in xrWaitFrame so the loop
     * sees the cleared alive flag and returns before the runtime library is torn down at process
     * exit.
     */
    void stop() {
        _is_alive = false;
        if (_xr_session != XR_NULL_HANDLE) xrRequestExitSession(_xr_session);
    }

    /****************************************************************
     * Queries
     ****************************************************************/
    /**
     * @brief Whether the session is live and the first head pose has been located.
     */
    [[nodiscard]] bool isRunning() const { return _is_running; }

    /**
     * @brief Per-eye render width, valid while running.
     */
    [[nodiscard]] RSize eyeWidth() const { return _eye_width; }

    /**
     * @brief Per-eye render height, valid while running.
     */
    [[nodiscard]] RSize eyeHeight() const { return _eye_height; }

    /****************************************************************
     * Frame
     ****************************************************************/
    /**
     * @brief The mapped color staging buffer for one eye, valid only while a staging write is held.
     */
    [[nodiscard]] RColor4b* getColorStagingBuffer(uint32_t eye) {
        return static_cast<RColor4b*>(_eye_upload[eye].color_mapped);
    }

    /**
     * @brief The mapped depth staging buffer for one eye, or null when the runtime lacks depth.
     */
    [[nodiscard]] float* getDepthStagingBuffer(uint32_t eye) {
        return _has_depth_extension ? static_cast<float*>(_eye_upload[eye].depth_mapped) : nullptr;
    }

    /**
     * @brief Claim the staging buffers for a direct component write, non-blocking.
     * @details Succeeds only when the staging is alive, the present thread has consumed the last
     * frame, and both eye copies have finished, so the component never overwrites a buffer the GPU
     * still reads. A held claim keeps the present thread from freeing the staging on a disconnect.
     * @return False when the staging is unavailable this step, so the caller skips the frame.
     */
    [[nodiscard]] bool beginStagingWrite() {
        std::scoped_lock lock(_staging_mutex);
        if (!_is_staging_alive || _has_pending.load(std::memory_order_acquire)) return false;
        for (uint32_t eye = 0; eye < 2; ++eye) {
            if (vkGetFenceStatus(_vk_device, _eye_upload[eye].fence) != VK_SUCCESS) return false;
        }
        _is_staging_borrowed = true;
        return true;
    }

    /**
     * @brief Release the staging claim so a pending teardown can proceed.
     */
    void endStagingWrite() {
        {
            std::scoped_lock lock(_staging_mutex);
            _is_staging_borrowed = false;
        }
        _staging_cv.notify_all();
    }

    /**
     * @brief Commit the staged stereo frame to the present thread and return at once.
     */
    void commitFrame(
        const FramePose& left_frame_pose,
        const FramePose& right_frame_pose,
        RSize width,
        RSize height
    ) {
        if (width != _eye_width || height != _eye_height) return;
        std::scoped_lock lock(_frame_mutex);
        _pending_frame_pose[0] = left_frame_pose;
        _pending_frame_pose[1] = right_frame_pose;
        _has_frame_pose = true;
        _has_pending.store(true, std::memory_order_release);
    }

    /**
     * @brief The latest located left and right eye, in rise-origin coordinates after locomotion.
     */
    [[nodiscard]] std::pair<EyePose, EyePose> getEyes() {
        std::scoped_lock lock(_eye_mutex);
        EyePose left = _eyes[0];
        EyePose right = _eyes[1];
        left.raw_position = left.position;
        left.raw_orientation = left.orientation;
        right.raw_position = right.position;
        right.raw_orientation = right.orientation;
        left.position = _getPlayPosition(left.position);
        left.orientation = _getPlayOrientation(left.orientation);
        right.position = _getPlayPosition(right.position);
        right.orientation = _getPlayOrientation(right.orientation);
        return {left, right};
    }

    /****************************************************************
     * Input and HUD
     ****************************************************************/
    /**
     * @brief The latest left and right hand intent, in rise-origin world coordinates.
     */
    [[nodiscard]] std::pair<HandIntent, HandIntent> getHandIntents() {
        std::scoped_lock lock(_intent_mutex);
        return {_intents[0], _intents[1]};
    }

    /**
     * @brief Publish the latest header metrics for the HUD to draw.
     */
    void setHudInfo(const HudInfo& info) {
        std::scoped_lock lock(_hud_info_mutex);
        _hud_info = info;
    }

    /**
     * @brief Set one hand's aim ray for the present thread to draw.
     * @details The present thread builds the beam from that live hand each frame; only the reach
     * and the hit state come from the component, so the beam origin never lags the hand.
     */
    void setRay(bool is_left_hand, bool is_active, RFloat length, bool is_hit, RayMode mode) {
        uint32_t hand = is_left_hand ? 0 : 1;
        std::scoped_lock lock(_ctrl_ray_mutex);
        _is_ctrl_ray_active[hand] = is_active;
        _ctrl_ray_length[hand] = length;
        _is_ctrl_ray_hit[hand] = is_hit;
        _ctrl_ray_mode[hand] = mode;
    }

   private:
    /****************************************************************
     * Types
     ****************************************************************/
    // Per-eye Vulkan upload resources, one set reused across frames.
    struct EyeUpload {
        VkBuffer color_staging = VK_NULL_HANDLE;
        VkDeviceMemory color_staging_memory = VK_NULL_HANDLE;
        void* color_mapped = nullptr;
        VkBuffer depth_staging = VK_NULL_HANDLE;
        VkDeviceMemory depth_staging_memory = VK_NULL_HANDLE;
        void* depth_mapped = nullptr;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
    };

    /****************************************************************
     * Data
     ****************************************************************/
    /**** Config ****/
    // Configured depth range, used to fill the depth layer info.
    RFloat _near_depth;
    RFloat _far_depth;
    // Locomotion tuning, set from the constructor; the turn speed is stored in radians per second.
    RFloat _move_speed;
    RFloat _fly_speed;
    RFloat _turn_speed;
    RFloat _stick_dead_zone;
    RFloat _boost;
    // Seconds for the body-locked HUD to catch up to the head heading.
    RFloat _hud_follow_tau;
    // Controller tuning: beam width, reticle half-size as a fraction of the reach with a floor, the
    // beam reach on a miss, and the controller marker diameter, all in meters.
    RFloat _ctrl_ray_beam_width;
    RFloat _ctrl_ray_beam_length;
    RFloat _ctrl_ray_reticle_fraction;
    RFloat _ctrl_ray_reticle_min;
    RFloat _ctrl_ray_reach;
    RFloat _ctrl_marker_size;
    // Logging.
    std::shared_ptr<logging::logger> _logger;

    /**** OpenXR ****/
    // OpenXR handles, created through the runtime.
    XrInstance _xr_instance = XR_NULL_HANDLE;
    XrSystemId _xr_system = XR_NULL_SYSTEM_ID;
    XrSession _xr_session = XR_NULL_HANDLE;
    XrSpace _xr_space = XR_NULL_HANDLE;
    XrSpace _xr_view_space = XR_NULL_HANDLE;
    XrSwapchain _xr_color_swapchain[2]{};
    XrSwapchain _xr_depth_swapchain[2]{};
    // OpenXR Vulkan binding entry points.
    PFN_xrGetVulkanGraphicsRequirements2KHR _xr_get_graphics_requirements = nullptr;
    PFN_xrCreateVulkanInstanceKHR _xr_create_vulkan_instance = nullptr;
    PFN_xrGetVulkanGraphicsDevice2KHR _xr_get_vulkan_device = nullptr;
    PFN_xrCreateVulkanDeviceKHR _xr_create_vulkan_device = nullptr;

    /**** Vulkan ****/
    // Vulkan handles, created through the runtime so they match the headset GPU.
    VkInstance _vk_instance = VK_NULL_HANDLE;
    VkPhysicalDevice _vk_physical_device = VK_NULL_HANDLE;
    VkDevice _vk_device = VK_NULL_HANDLE;
    VkQueue _vk_queue = VK_NULL_HANDLE;
    uint32_t _queue_family_index = 0;
    VkCommandPool _command_pool = VK_NULL_HANDLE;
    std::vector<XrSwapchainImageVulkanKHR> _color_images[2];
    std::vector<XrSwapchainImageVulkanKHR> _depth_images[2];
    EyeUpload _eye_upload[2];

    /**** Input ****/
    // Abstract action set bound to every common profile, plus per-hand grip and aim spaces.
    XrActionSet _xr_action_set = XR_NULL_HANDLE;
    XrAction _xr_grip_pose = XR_NULL_HANDLE;
    XrAction _xr_aim_pose = XR_NULL_HANDLE;
    XrAction _xr_grab = XR_NULL_HANDLE;
    XrAction _xr_push = XR_NULL_HANDLE;
    XrAction _xr_move = XR_NULL_HANDLE;
    XrAction _xr_turn = XR_NULL_HANDLE;
    XrAction _xr_menu = XR_NULL_HANDLE;
    XrAction _xr_boost = XR_NULL_HANDLE;
    XrSpace _xr_grip_space[2]{};
    XrSpace _xr_aim_space[2]{};
    XrPath _xr_hand_path[2]{};
    bool _was_menu_pressed = false;
    // Latest hand intent from the present thread to the component, guarded.
    std::mutex _intent_mutex;
    HandIntent _intents[2];

    /**** Locomotion ****/
    // Play-space offset integrated each present cycle; guarded with the eyes by _eye_mutex.
    RVec3rf _play_position;
    RFloat _play_yaw = 0;
    XrTime _last_input_time = 0;

    /**** HUD ****/
    // Dear ImGui rendered into an OpenXR quad layer; the component only feeds it through
    // setHudInfo.
    XrSwapchain _xr_hud_swapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageVulkanKHR> _hud_images;
    std::vector<VkImageView> _hud_image_views;
    std::vector<VkFramebuffer> _hud_framebuffers;
    VkRenderPass _hud_render_pass = VK_NULL_HANDLE;
    VkDescriptorPool _hud_descriptor_pool = VK_NULL_HANDLE;
    VkCommandBuffer _hud_command_buffer = VK_NULL_HANDLE;
    VkFence _hud_fence = VK_NULL_HANDLE;
    ImFont* _hud_font = nullptr;
    RSize _hud_width = 1344;
    RSize _hud_height = 224;
    bool _is_hud_ready = false;
    bool _has_hud_image = false;
    std::atomic<bool> _is_hud_visible{true};
    // Lazily-followed body frame the HUD is anchored to. Present thread only.
    RFloat _hud_yaw = 0;
    RVec3rf _hud_position;
    bool _has_hud_anchor = false;
    // Latest header metrics from the component to the present thread, guarded.
    std::mutex _hud_info_mutex;
    HudInfo _hud_info;

    /**** Frame handoff ****/
    // Frame pose hand-off to the present thread; the pending flag is atomic for a lock-free poll.
    std::mutex _frame_mutex;
    FramePose _pending_frame_pose[2];
    bool _has_frame_pose = false;
    std::atomic<bool> _has_pending{false};

    /**** Staging borrow ****/
    // Guards the component's direct staging fill against the present thread freeing it on
    // disconnect.
    std::mutex _staging_mutex;
    std::condition_variable _staging_cv;
    bool _is_staging_borrowed = false;
    bool _is_staging_alive = false;

    /**** Controller ****/
    // Per-hand aim ray drawn as re-posed quad layers; reach and hit from setRay, aim and grip from
    // the live physical hand captured in _syncInput.
    std::mutex _ctrl_ray_mutex;
    bool _is_ctrl_ray_active[2] = {false, false};
    bool _is_ctrl_ray_hit[2] = {false, false};
    bool _has_ctrl_ray_aim[2] = {false, false};
    bool _is_ctrl_ray_ready = false;
    bool _has_ctrl_grip_room[2] = {false, false};
    RFloat _ctrl_ray_length[2] = {0, 0};
    RVec3rf _ctrl_ray_origin_room[2];
    RVec3rf _ctrl_ray_dir_room[2];
    RVec3rf _ctrl_grip_origin_room[2];
    RayMode _ctrl_ray_mode[2] = {RayMode::NEUTRAL, RayMode::NEUTRAL};
    XrSwapchain _ctrl_xr_ray_swapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageVulkanKHR> _ctrl_ray_images;

    /**** Eyes ****/
    // Latest located eyes from the present thread to the component, guarded.
    std::mutex _eye_mutex;
    EyePose _eyes[2];
    RQuat3rf _basis;

    /**** Connection state ****/
    // Per-eye size, swapchain formats, and connection state.
    RSize _eye_width = 0;
    RSize _eye_height = 0;
    VkFormat _color_format = VK_FORMAT_R8G8B8A8_SRGB;
    VkFormat _depth_format = VK_FORMAT_D32_SFLOAT;
    bool _has_depth_extension = false;
    bool _has_pico_extension = false;
    bool _has_focus3_extension = false;
    XrSessionState _session_state = XR_SESSION_STATE_UNKNOWN;
    bool _is_session_begun = false;
    bool _is_instance_lost = false;
    // True once an eye image has been uploaded, so the layer can re-submit it. Present thread only.
    bool _has_image = false;
    std::atomic<bool> _is_running{false};

    /**** Present thread ****/
    std::atomic<bool> _is_alive{false};
    std::thread _vr_present_thread;

    /****************************************************************
     * Present loop
     ****************************************************************/
    void _vrPresentThreadFunc() {
        if (!_makeInstance()) {
            RS_LOG(_logger, warn, "No OpenXR runtime available; the VR component stays inactive");
            return;
        }
        RSE_VRState state = RSE_VR_STATE_WAITING;
        while (_is_alive) {
            try {
                if (state == RSE_VR_STATE_WAITING) {
                    if (_is_instance_lost) {
                        _freeInstance();
                        if (!_makeInstance()) break;
                        _is_instance_lost = false;
                    }
                    if (!_getSystem()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        continue;
                    }
                    if (_vk_device == VK_NULL_HANDLE) _makeVulkanDevice();
                    _initSession();
                    state = RSE_VR_STATE_RUNNING;
                } else if (!_pollEvents()) {
                    _freeSession();
                    _is_running = false;
                    _drainPending();
                    state = RSE_VR_STATE_WAITING;
                } else if (_is_session_begun) {
                    _renderFrame();
                } else {
                    _drainPending();
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            } catch (const std::exception& error) {
                RS_LOG(_logger, error, "VR present thread fault: {}; recovering", error.what());
                _freeSession();
                _is_running = false;
                _drainPending();
                state = RSE_VR_STATE_WAITING;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
    }

    /****************************************************************
     * OpenXR bringup
     ****************************************************************/
    bool _makeInstance() {
        uint32_t extension_count = 0;
        if (XR_FAILED(
                xrEnumerateInstanceExtensionProperties(nullptr, 0, &extension_count, nullptr)
            )) {
            return false;
        }
        std::vector<XrExtensionProperties> available(
            extension_count, {XR_TYPE_EXTENSION_PROPERTIES}
        );
        RS_XR_CHECK(xrEnumerateInstanceExtensionProperties(
            nullptr, extension_count, &extension_count, available.data()
        ));

        auto isSupported = [&](const char* name) {
            for (const XrExtensionProperties& property : available) {
                if (std::strcmp(property.extensionName, name) == 0) return true;
            }
            return false;
        };

        if (!isSupported(XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME)) {
            RS_LOG(
                _logger,
                warn,
                "OpenXR runtime lacks XR_KHR_vulkan_enable2; the VR component stays inactive"
            );
            return false;
        }
        std::vector<const char*> enabled{XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
        _has_depth_extension = isSupported(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);
        if (_has_depth_extension) enabled.push_back(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);
        // Stick-based controller profiles behind extensions, bound only when the runtime offers
        // them.
        _has_pico_extension = isSupported(XR_BD_CONTROLLER_INTERACTION_EXTENSION_NAME);
        if (_has_pico_extension) enabled.push_back(XR_BD_CONTROLLER_INTERACTION_EXTENSION_NAME);
        _has_focus3_extension =
            isSupported(XR_HTC_VIVE_FOCUS3_CONTROLLER_INTERACTION_EXTENSION_NAME);
        if (_has_focus3_extension)
            enabled.push_back(XR_HTC_VIVE_FOCUS3_CONTROLLER_INTERACTION_EXTENSION_NAME);

        XrInstanceCreateInfo create_info{XR_TYPE_INSTANCE_CREATE_INFO};
        std::strcpy(create_info.applicationInfo.applicationName, "rise-origin");
        create_info.applicationInfo.applicationVersion = 1;
        std::strcpy(create_info.applicationInfo.engineName, "rise-origin");
        create_info.applicationInfo.engineVersion = 1;
        create_info.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        create_info.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
        create_info.enabledExtensionNames = enabled.data();

        if (XR_FAILED(xrCreateInstance(&create_info, &_xr_instance))) {
            RS_LOG(_logger, warn, "xrCreateInstance failed; the VR component stays inactive");
            return false;
        }

        RS_XR_CHECK(xrGetInstanceProcAddr(
            _xr_instance,
            "xrGetVulkanGraphicsRequirements2KHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&_xr_get_graphics_requirements)
        ));
        RS_XR_CHECK(xrGetInstanceProcAddr(
            _xr_instance,
            "xrCreateVulkanInstanceKHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&_xr_create_vulkan_instance)
        ));
        RS_XR_CHECK(xrGetInstanceProcAddr(
            _xr_instance,
            "xrGetVulkanGraphicsDevice2KHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&_xr_get_vulkan_device)
        ));
        RS_XR_CHECK(xrGetInstanceProcAddr(
            _xr_instance,
            "xrCreateVulkanDeviceKHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&_xr_create_vulkan_device)
        ));
        return true;
    }

    bool _getSystem() {
        XrSystemGetInfo system_info{XR_TYPE_SYSTEM_GET_INFO};
        system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        XrResult result = xrGetSystem(_xr_instance, &system_info, &_xr_system);
        if (result == XR_ERROR_FORM_FACTOR_UNAVAILABLE) return false;
        RS_XR_CHECK(result);
        _getViewConfig();
        return true;
    }

    void _getViewConfig() {
        uint32_t view_count = 0;
        RS_XR_CHECK(xrEnumerateViewConfigurationViews(
            _xr_instance,
            _xr_system,
            XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
            0,
            &view_count,
            nullptr
        ));
        if (view_count != 2) {
            RS_LOG(
                _logger, error, "Expected a stereo view configuration, got {} views", view_count
            );
            RS_THROW(RS_RuntimeException());
        }
        XrViewConfigurationView config[2];
        for (uint32_t eye = 0; eye < 2; ++eye) {
            config[eye] = {XR_TYPE_VIEW_CONFIGURATION_VIEW};
        }
        RS_XR_CHECK(xrEnumerateViewConfigurationViews(
            _xr_instance,
            _xr_system,
            XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
            2,
            &view_count,
            config
        ));
        _eye_width = config[0].recommendedImageRectWidth;
        _eye_height = config[0].recommendedImageRectHeight;
        RS_LOG(_logger, info, "OpenXR per-eye render resolution: {}x{}", _eye_width, _eye_height);
    }

    void _makeVulkanDevice() {
        XrGraphicsRequirementsVulkanKHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
        RS_XR_CHECK(_xr_get_graphics_requirements(_xr_instance, _xr_system, &requirements));

        VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app_info.pApplicationName = "rise-origin";
        app_info.applicationVersion = 1;
        app_info.pEngineName = "rise-origin";
        app_info.engineVersion = 1;
        app_info.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo vk_instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        vk_instance_info.pApplicationInfo = &app_info;

        XrVulkanInstanceCreateInfoKHR xr_instance_info{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
        xr_instance_info.systemId = _xr_system;
        xr_instance_info.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
        xr_instance_info.vulkanCreateInfo = &vk_instance_info;
        VkResult vk_result = VK_SUCCESS;
        RS_XR_CHECK(
            _xr_create_vulkan_instance(_xr_instance, &xr_instance_info, &_vk_instance, &vk_result)
        );
        RS_VK_CHECK(vk_result);

        XrVulkanGraphicsDeviceGetInfoKHR device_get_info{
            XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR
        };
        device_get_info.systemId = _xr_system;
        device_get_info.vulkanInstance = _vk_instance;
        RS_XR_CHECK(_xr_get_vulkan_device(_xr_instance, &device_get_info, &_vk_physical_device));

        _queue_family_index = _getGraphicsQueueFamily();

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = _queue_family_index;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo vk_device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        vk_device_info.queueCreateInfoCount = 1;
        vk_device_info.pQueueCreateInfos = &queue_info;

        XrVulkanDeviceCreateInfoKHR xr_device_info{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
        xr_device_info.systemId = _xr_system;
        xr_device_info.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
        xr_device_info.vulkanPhysicalDevice = _vk_physical_device;
        xr_device_info.vulkanCreateInfo = &vk_device_info;
        RS_XR_CHECK(
            _xr_create_vulkan_device(_xr_instance, &xr_device_info, &_vk_device, &vk_result)
        );
        RS_VK_CHECK(vk_result);

        vkGetDeviceQueue(_vk_device, _queue_family_index, 0, &_vk_queue);

        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = _queue_family_index;
        RS_VK_CHECK(vkCreateCommandPool(_vk_device, &pool_info, nullptr, &_command_pool));
    }

    uint32_t _getGraphicsQueueFamily() {
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(_vk_physical_device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            _vk_physical_device, &family_count, families.data()
        );
        for (uint32_t index = 0; index < family_count; ++index) {
            if (families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) return index;
        }
        RS_LOG(_logger, error, "No Vulkan graphics queue family on the headset GPU");
        RS_THROW(RS_RuntimeException());
        return 0;
    }

    void _initSession() {
        XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
        binding.instance = _vk_instance;
        binding.physicalDevice = _vk_physical_device;
        binding.device = _vk_device;
        binding.queueFamilyIndex = _queue_family_index;
        binding.queueIndex = 0;

        XrSessionCreateInfo session_info{XR_TYPE_SESSION_CREATE_INFO};
        session_info.next = &binding;
        session_info.systemId = _xr_system;
        RS_XR_CHECK(xrCreateSession(_xr_instance, &session_info, &_xr_session));

        XrReferenceSpaceCreateInfo space_info{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
        space_info.poseInReferenceSpace.orientation.w = 1.0f;
        if (XR_FAILED(xrCreateReferenceSpace(_xr_session, &space_info, &_xr_space))) {
            // Not every runtime offers a stage space; fall back to a local space.
            space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
            RS_XR_CHECK(xrCreateReferenceSpace(_xr_session, &space_info, &_xr_space));
        }

        uint32_t format_count = 0;
        RS_XR_CHECK(xrEnumerateSwapchainFormats(_xr_session, 0, &format_count, nullptr));
        std::vector<int64_t> formats(format_count);
        RS_XR_CHECK(
            xrEnumerateSwapchainFormats(_xr_session, format_count, &format_count, formats.data())
        );
        _color_format = _getColorFormat(formats, _logger);
        _depth_format = _getDepthFormat(formats);

        for (uint32_t eye = 0; eye < 2; ++eye) {
            _xr_color_swapchain[eye] = _makeSwapchain(
                static_cast<int64_t>(_color_format),
                XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT,
                _color_images[eye]
            );
            if (_has_depth_extension) {
                _xr_depth_swapchain[eye] = _makeSwapchain(
                    static_cast<int64_t>(_depth_format),
                    XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                        XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT,
                    _depth_images[eye]
                );
            }
            _makeEyeUpload(_eye_upload[eye]);
        }
        _initInput();

        XrReferenceSpaceCreateInfo view_space_info{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        view_space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
        view_space_info.poseInReferenceSpace.orientation.w = 1.0f;
        RS_XR_CHECK(xrCreateReferenceSpace(_xr_session, &view_space_info, &_xr_view_space));
        _initHud();
        _initController();

        // The staging buffers now exist; let the component fill them directly until teardown.
        {
            std::scoped_lock lock(_staging_mutex);
            _is_staging_alive = true;
        }
    }

    XrSwapchain _makeSwapchain(
        int64_t format,
        XrSwapchainUsageFlags usage,
        std::vector<XrSwapchainImageVulkanKHR>& images
    ) {
        XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        info.usageFlags = usage;
        info.format = format;
        info.sampleCount = 1;
        info.width = static_cast<uint32_t>(_eye_width);
        info.height = static_cast<uint32_t>(_eye_height);
        info.faceCount = 1;
        info.arraySize = 1;
        info.mipCount = 1;
        XrSwapchain swapchain = XR_NULL_HANDLE;
        RS_XR_CHECK(xrCreateSwapchain(_xr_session, &info, &swapchain));

        uint32_t image_count = 0;
        RS_XR_CHECK(xrEnumerateSwapchainImages(swapchain, 0, &image_count, nullptr));
        images.assign(image_count, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        RS_XR_CHECK(xrEnumerateSwapchainImages(
            swapchain,
            image_count,
            &image_count,
            reinterpret_cast<XrSwapchainImageBaseHeader*>(images.data())
        ));
        return swapchain;
    }

    void _makeEyeUpload(EyeUpload& upload) {
        const VkDeviceSize color_bytes = static_cast<VkDeviceSize>(_eye_width) * _eye_height * 4;
        const VkDeviceSize depth_bytes =
            static_cast<VkDeviceSize>(_eye_width) * _eye_height * sizeof(float);
        _makeStagingBuffer(
            color_bytes, upload.color_staging, upload.color_staging_memory, upload.color_mapped
        );
        if (_has_depth_extension) {
            _makeStagingBuffer(
                depth_bytes, upload.depth_staging, upload.depth_staging_memory, upload.depth_mapped
            );
        }

        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = _command_pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        RS_VK_CHECK(vkAllocateCommandBuffers(_vk_device, &command_info, &upload.command_buffer));

        // Created signaled so the first upload's wait at the top of the frame returns at once.
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        RS_VK_CHECK(vkCreateFence(_vk_device, &fence_info, nullptr, &upload.fence));
    }

    void _makeStagingBuffer(
        VkDeviceSize bytes,
        VkBuffer& buffer,
        VkDeviceMemory& memory,
        void*& mapped
    ) {
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = bytes;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        RS_VK_CHECK(vkCreateBuffer(_vk_device, &buffer_info, nullptr, &buffer));

        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(_vk_device, buffer, &requirements);

        VkMemoryAllocateInfo allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate_info.allocationSize = requirements.size;
        allocate_info.memoryTypeIndex = _getMemoryType(
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        RS_VK_CHECK(vkAllocateMemory(_vk_device, &allocate_info, nullptr, &memory));
        RS_VK_CHECK(vkBindBufferMemory(_vk_device, buffer, memory, 0));
        RS_VK_CHECK(vkMapMemory(_vk_device, memory, 0, bytes, 0, &mapped));
    }

    uint32_t _getMemoryType(uint32_t type_bits, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(_vk_physical_device, &memory_properties);
        for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
            const bool is_type = (type_bits & (1u << index)) != 0;
            const bool has_properties =
                (memory_properties.memoryTypes[index].propertyFlags & properties) == properties;
            if (is_type && has_properties) return index;
        }
        RS_LOG(_logger, error, "No Vulkan memory type with the requested properties");
        RS_THROW(RS_RuntimeException());
        return 0;
    }

    void _freeSession() {
        // Mark the staging dead and wait out any in-flight component fill before destroying the
        // buffers, so a raw staging pointer the component holds never dangles.
        {
            std::unique_lock lock(_staging_mutex);
            _is_staging_alive = false;
            _staging_cv.wait(lock, [this] { return !_is_staging_borrowed; });
        }
        if (_vk_device != VK_NULL_HANDLE) vkDeviceWaitIdle(_vk_device);

        if (_is_hud_ready) {
            ImGui_ImplVulkan_Shutdown();
            ImGui::DestroyContext();
            _is_hud_ready = false;
        }
        _has_hud_image = false;
        _has_hud_anchor = false;
        _has_frame_pose = false;
        _has_image = false;
        for (VkFramebuffer framebuffer : _hud_framebuffers) {
            if (framebuffer != VK_NULL_HANDLE)
                vkDestroyFramebuffer(_vk_device, framebuffer, nullptr);
        }
        _hud_framebuffers.clear();
        for (VkImageView view : _hud_image_views) {
            if (view != VK_NULL_HANDLE) vkDestroyImageView(_vk_device, view, nullptr);
        }
        _hud_image_views.clear();
        if (_hud_fence != VK_NULL_HANDLE) {
            vkDestroyFence(_vk_device, _hud_fence, nullptr);
            _hud_fence = VK_NULL_HANDLE;
        }
        if (_hud_command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(_vk_device, _command_pool, 1, &_hud_command_buffer);
            _hud_command_buffer = VK_NULL_HANDLE;
        }
        if (_hud_render_pass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(_vk_device, _hud_render_pass, nullptr);
            _hud_render_pass = VK_NULL_HANDLE;
        }
        if (_hud_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(_vk_device, _hud_descriptor_pool, nullptr);
            _hud_descriptor_pool = VK_NULL_HANDLE;
        }
        if (_xr_hud_swapchain != XR_NULL_HANDLE) {
            xrDestroySwapchain(_xr_hud_swapchain);
            _xr_hud_swapchain = XR_NULL_HANDLE;
        }
        _hud_images.clear();
        if (_ctrl_xr_ray_swapchain != XR_NULL_HANDLE) {
            xrDestroySwapchain(_ctrl_xr_ray_swapchain);
            _ctrl_xr_ray_swapchain = XR_NULL_HANDLE;
        }
        _ctrl_ray_images.clear();
        _is_ctrl_ray_ready = false;
        if (_xr_view_space != XR_NULL_HANDLE) {
            xrDestroySpace(_xr_view_space);
            _xr_view_space = XR_NULL_HANDLE;
        }

        for (uint32_t eye = 0; eye < 2; ++eye) {
            EyeUpload& upload = _eye_upload[eye];
            if (upload.fence != VK_NULL_HANDLE) {
                vkDestroyFence(_vk_device, upload.fence, nullptr);
                upload.fence = VK_NULL_HANDLE;
            }
            if (upload.command_buffer != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(_vk_device, _command_pool, 1, &upload.command_buffer);
                upload.command_buffer = VK_NULL_HANDLE;
            }
            if (upload.color_staging != VK_NULL_HANDLE) {
                vkDestroyBuffer(_vk_device, upload.color_staging, nullptr);
                vkFreeMemory(_vk_device, upload.color_staging_memory, nullptr);
                upload.color_staging = VK_NULL_HANDLE;
                upload.color_mapped = nullptr;
            }
            if (upload.depth_staging != VK_NULL_HANDLE) {
                vkDestroyBuffer(_vk_device, upload.depth_staging, nullptr);
                vkFreeMemory(_vk_device, upload.depth_staging_memory, nullptr);
                upload.depth_staging = VK_NULL_HANDLE;
                upload.depth_mapped = nullptr;
            }
            if (_xr_color_swapchain[eye] != XR_NULL_HANDLE) {
                xrDestroySwapchain(_xr_color_swapchain[eye]);
                _xr_color_swapchain[eye] = XR_NULL_HANDLE;
            }
            if (_xr_depth_swapchain[eye] != XR_NULL_HANDLE) {
                xrDestroySwapchain(_xr_depth_swapchain[eye]);
                _xr_depth_swapchain[eye] = XR_NULL_HANDLE;
            }
            _color_images[eye].clear();
            _depth_images[eye].clear();
        }
        for (uint32_t hand = 0; hand < 2; ++hand) {
            if (_xr_grip_space[hand] != XR_NULL_HANDLE) {
                xrDestroySpace(_xr_grip_space[hand]);
                _xr_grip_space[hand] = XR_NULL_HANDLE;
            }
            if (_xr_aim_space[hand] != XR_NULL_HANDLE) {
                xrDestroySpace(_xr_aim_space[hand]);
                _xr_aim_space[hand] = XR_NULL_HANDLE;
            }
        }
        if (_xr_action_set != XR_NULL_HANDLE) {
            xrDestroyActionSet(_xr_action_set);
            _xr_action_set = XR_NULL_HANDLE;
        }
        if (_xr_space != XR_NULL_HANDLE) {
            xrDestroySpace(_xr_space);
            _xr_space = XR_NULL_HANDLE;
        }
        if (_xr_session != XR_NULL_HANDLE) {
            xrDestroySession(_xr_session);
            _xr_session = XR_NULL_HANDLE;
        }
        _is_session_begun = false;
        _last_input_time = 0;
    }

    void _freeInstance() {
        if (_command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(_vk_device, _command_pool, nullptr);
            _command_pool = VK_NULL_HANDLE;
        }
        if (_vk_device != VK_NULL_HANDLE) {
            vkDestroyDevice(_vk_device, nullptr);
            _vk_device = VK_NULL_HANDLE;
        }
        if (_vk_instance != VK_NULL_HANDLE) {
            vkDestroyInstance(_vk_instance, nullptr);
            _vk_instance = VK_NULL_HANDLE;
        }
        if (_xr_instance != XR_NULL_HANDLE) {
            xrDestroyInstance(_xr_instance);
            _xr_instance = XR_NULL_HANDLE;
        }
        _xr_system = XR_NULL_SYSTEM_ID;
    }

    /****************************************************************
     * Frame render
     ****************************************************************/
    bool _pollEvents() {
        XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
        while (xrPollEvent(_xr_instance, &event) == XR_SUCCESS) {
            if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                const auto& changed =
                    *reinterpret_cast<const XrEventDataSessionStateChanged*>(&event);
                _session_state = changed.state;
                if (changed.state == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo begin_info{XR_TYPE_SESSION_BEGIN_INFO};
                    begin_info.primaryViewConfigurationType =
                        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    RS_XR_CHECK(xrBeginSession(_xr_session, &begin_info));
                    _is_session_begun = true;
                } else if (changed.state == XR_SESSION_STATE_STOPPING) {
                    _is_running = false;
                    xrEndSession(_xr_session);
                    return false;
                } else if (changed.state == XR_SESSION_STATE_EXITING ||
                           changed.state == XR_SESSION_STATE_LOSS_PENDING) {
                    _is_running = false;
                    return false;
                }
            } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
                _is_instance_lost = true;
                _is_running = false;
                return false;
            }
            event = {XR_TYPE_EVENT_DATA_BUFFER};
        }
        return true;
    }

    void _renderFrame() {
        XrFrameWaitInfo wait_info{XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState frame_state{XR_TYPE_FRAME_STATE};
        RS_XR_CHECK(xrWaitFrame(_xr_session, &wait_info, &frame_state));

        XrFrameBeginInfo begin_info{XR_TYPE_FRAME_BEGIN_INFO};
        RS_XR_CHECK(xrBeginFrame(_xr_session, &begin_info));

        _syncInput(frame_state.predictedDisplayTime);

        FramePose frame_pose[2];
        bool has_frame_pose = false;
        bool is_new_content = false;
        {
            std::scoped_lock lock(_frame_mutex);
            for (uint32_t eye = 0; eye < 2; ++eye) frame_pose[eye] = _pending_frame_pose[eye];
            has_frame_pose = _has_frame_pose;
            is_new_content = _has_pending.load(std::memory_order_acquire);
        }

        XrCompositionLayerProjectionView projection_views[2]{};
        XrCompositionLayerDepthInfoKHR depth_infos[2]{};
        XrCompositionLayerProjection projection_layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        bool has_views = false;
        if (frame_state.shouldRender == XR_TRUE) {
            has_views = _getViews(
                frame_state.predictedDisplayTime,
                frame_pose,
                has_frame_pose,
                is_new_content,
                projection_views,
                depth_infos,
                projection_layer
            );
        }

        // Once the component-filled staging is copied into the swapchain, release the pending frame
        // so the component can refill it; only when this frame actually consumed new content.
        if (is_new_content && frame_state.shouldRender == XR_TRUE) {
            _has_pending.store(false, std::memory_order_release);
        }

        XrFrameEndInfo end_info{XR_TYPE_FRAME_END_INFO};
        end_info.displayTime = frame_state.predictedDisplayTime;
        end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        const XrCompositionLayerBaseHeader* layers[8];
        uint32_t layer_count = 0;
        if (has_views) {
            layers[layer_count++] =
                reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection_layer);
        }
        // Render the HUD every frame, now that neither the eye upload nor the HUD render stalls the
        // present thread, so the readouts stay current.
        if (frame_state.shouldRender == XR_TRUE && _is_hud_visible) _renderHud();
        // The HUD composites over the world as a translucent quad. It is anchored to a lazily
        // followed body frame so head glances leave it world anchored, which the runtime reprojects
        // crisply, instead of a head-locked quad that fights the reprojection and smears. Declared
        // before the branch so it outlives it through xrEndFrame.
        XrCompositionLayerQuad hud_quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
        XrSpaceLocation head{XR_TYPE_SPACE_LOCATION};
        const bool has_head = _xr_view_space != XR_NULL_HANDLE &&
                              XR_SUCCEEDED(xrLocateSpace(
                                  _xr_view_space, _xr_space, frame_state.predictedDisplayTime, &head
                              )) &&
                              (head.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
                              (head.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0;
        if (_is_hud_visible && _has_hud_image && has_head) {
            hud_quad.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                                  XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT;
            hud_quad.space = _xr_space;
            hud_quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            hud_quad.subImage.swapchain = _xr_hud_swapchain;
            hud_quad.subImage.imageRect.offset = {0, 0};
            hud_quad.subImage.imageRect.extent = {
                static_cast<int32_t>(_hud_width), static_cast<int32_t>(_hud_height)
            };
            RFloat dt = static_cast<RFloat>(frame_state.predictedDisplayPeriod) * RFloat(1e-9);
            hud_quad.pose = _getBodyHudPose(head.pose, dt);
            hud_quad.size = {0.66f, 0.11f};
            layers[layer_count++] =
                reinterpret_cast<const XrCompositionLayerBaseHeader*>(&hud_quad);
        }
        // Each hand's aim ray composites as a beam plus a reticle quad, re-posed every present
        // frame from the live physical hand, so a beam tracks its controller between the slower
        // world frames. Declared before the branch so they outlive it through xrEndFrame.
        // Each hand contributes a beam, a reticle, and a controller marker ball.
        XrCompositionLayerQuad ray_quads[4];
        XrCompositionLayerQuad marker_quads[2];
        for (XrCompositionLayerQuad& quad : ray_quads) quad = {XR_TYPE_COMPOSITION_LAYER_QUAD};
        for (XrCompositionLayerQuad& quad : marker_quads) quad = {XR_TYPE_COMPOSITION_LAYER_QUAD};
        if (_is_ctrl_ray_ready && has_head) {
            for (uint32_t hand = 0; hand < 2; ++hand) {
                XrCompositionLayerQuad& beam = ray_quads[hand * 2];
                XrCompositionLayerQuad& reticle = ray_quads[hand * 2 + 1];
                if (_getControllerRayQuads(hand, head.pose, beam, reticle)) {
                    layers[layer_count++] =
                        reinterpret_cast<const XrCompositionLayerBaseHeader*>(&beam);
                    layers[layer_count++] =
                        reinterpret_cast<const XrCompositionLayerBaseHeader*>(&reticle);
                }
                if (_getControllerMarkerQuad(hand, head.pose, marker_quads[hand])) {
                    layers[layer_count++] =
                        reinterpret_cast<const XrCompositionLayerBaseHeader*>(&marker_quads[hand]);
                }
            }
        }
        end_info.layerCount = layer_count;
        end_info.layers = layer_count > 0 ? layers : nullptr;
        RS_XR_CHECK(xrEndFrame(_xr_session, &end_info));
    }

    bool _getViews(
        XrTime display_time,
        const FramePose* frame_pose,
        bool has_frame_pose,
        bool is_new_content,
        XrCompositionLayerProjectionView* projection_views,
        XrCompositionLayerDepthInfoKHR* depth_infos,
        XrCompositionLayerProjection& projection_layer
    ) {
        XrViewState view_state{XR_TYPE_VIEW_STATE};
        XrViewLocateInfo locate_info{XR_TYPE_VIEW_LOCATE_INFO};
        locate_info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        locate_info.displayTime = display_time;
        locate_info.space = _xr_space;
        XrView views[2]{{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
        uint32_t located = 0;
        RS_XR_CHECK(xrLocateViews(_xr_session, &locate_info, &view_state, 2, &located, views));
        const bool is_pose_valid =
            (view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) != 0 &&
            (view_state.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) != 0;
        if (is_pose_valid) {
            std::scoped_lock lock(_eye_mutex);
            for (uint32_t eye = 0; eye < 2; ++eye) {
                _eyes[eye].position = _getRiseVector(views[eye].pose.position);
                _eyes[eye].orientation = _getRiseOrientation(views[eye].pose.orientation);
                _eyes[eye].angle_left = views[eye].fov.angleLeft;
                _eyes[eye].angle_right = views[eye].fov.angleRight;
                _eyes[eye].angle_up = views[eye].fov.angleUp;
                _eyes[eye].angle_down = views[eye].fov.angleDown;
            }
            _is_running = true;
        }

        // Only upload when the component has a new frame. With no new content the swapchain still
        // holds the last released image, so re-submitting the layer over it lets the runtime
        // reproject the last frame from its own baseline instead of re-warping a freshly uploaded
        // copy every tick, which is both cheaper and steadier.
        if (is_new_content) _has_image = true;
        for (uint32_t eye = 0; eye < 2; ++eye) {
            if (is_new_content) _uploadEye(eye);

            projection_views[eye] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            // Label the layer with the pose this eye's pixels were actually rendered from,
            // recovered from the camera record, so the runtime reprojects from the right baseline
            // and the world does not smear when the head turns fast. Fall back to the located pose
            // until the first rendered frame arrives.
            projection_views[eye].pose = has_frame_pose ? _getLayerPose(frame_pose[eye])
                                                        : views[eye].pose;
            // The camera renders this eye's real asymmetric frustum, fed back through the signal,
            // so submit the runtime's frustum to display the full field of view.
            projection_views[eye].fov = views[eye].fov;
            projection_views[eye].subImage.swapchain = _xr_color_swapchain[eye];
            projection_views[eye].subImage.imageRect.offset = {0, 0};
            projection_views[eye].subImage.imageRect.extent = {
                static_cast<int32_t>(_eye_width), static_cast<int32_t>(_eye_height)
            };

            if (_has_depth_extension) {
                depth_infos[eye] = {XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR};
                depth_infos[eye].subImage.swapchain = _xr_depth_swapchain[eye];
                depth_infos[eye].subImage.imageRect.offset = {0, 0};
                depth_infos[eye].subImage.imageRect.extent = {
                    static_cast<int32_t>(_eye_width), static_cast<int32_t>(_eye_height)
                };
                depth_infos[eye].minDepth = 0.0f;
                depth_infos[eye].maxDepth = 1.0f;
                depth_infos[eye].nearZ = static_cast<float>(_near_depth);
                depth_infos[eye].farZ = static_cast<float>(_far_depth);
                projection_views[eye].next = &depth_infos[eye];
            }
        }

        projection_layer.space = _xr_space;
        projection_layer.viewCount = 2;
        projection_layer.views = projection_views;
        return _has_image;
    }

    // Wait for a swapchain image with a bounded timeout; a timeout throws so the present loop
    // recovers the session rather than blocking the present thread forever on a stalled compositor.
    void _waitSwapchainImage(XrSwapchain swapchain) {
        XrSwapchainImageWaitInfo wait_info{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        wait_info.timeout = RS_VR_SWAPCHAIN_WAIT_TIMEOUT;
        XrResult result = xrWaitSwapchainImage(swapchain, &wait_info);
        if (result == XR_TIMEOUT_EXPIRED) {
            RS_THROW(RS_RuntimeException("OpenXR swapchain wait timed out; recovering session"));
        }
        RS_XR_CHECK(result);
    }

    // Upload one eye's already-filled staging buffer to the swapchain; the component wrote the
    // staging directly, so this only issues the GPU copy.
    void _uploadEye(uint32_t eye) {
        EyeUpload& upload = _eye_upload[eye];

        // Wait on the previous frame's copy; it finished long ago at frame rate, so this does not
        // stall the present thread inside the frame.
        vkWaitForFences(_vk_device, 1, &upload.fence, VK_TRUE, UINT64_MAX);

        uint32_t color_index = 0;
        XrSwapchainImageAcquireInfo acquire_info{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        RS_XR_CHECK(xrAcquireSwapchainImage(_xr_color_swapchain[eye], &acquire_info, &color_index));
        _waitSwapchainImage(_xr_color_swapchain[eye]);

        uint32_t depth_index = 0;
        if (_has_depth_extension) {
            XrSwapchainImageAcquireInfo depth_acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
            RS_XR_CHECK(
                xrAcquireSwapchainImage(_xr_depth_swapchain[eye], &depth_acquire, &depth_index)
            );
            _waitSwapchainImage(_xr_depth_swapchain[eye]);
        }

        _submitUpload(eye, color_index, depth_index);

        XrSwapchainImageReleaseInfo release_info{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        RS_XR_CHECK(xrReleaseSwapchainImage(_xr_color_swapchain[eye], &release_info));
        if (_has_depth_extension) {
            XrSwapchainImageReleaseInfo depth_release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            RS_XR_CHECK(xrReleaseSwapchainImage(_xr_depth_swapchain[eye], &depth_release));
        }
    }

    void _submitUpload(uint32_t eye, uint32_t color_index, uint32_t depth_index) {
        EyeUpload& upload = _eye_upload[eye];
        RS_VK_CHECK(vkResetCommandBuffer(upload.command_buffer, 0));

        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        RS_VK_CHECK(vkBeginCommandBuffer(upload.command_buffer, &begin_info));

        _copyBufferToImage(
            upload.command_buffer,
            upload.color_staging,
            _color_images[eye][color_index].image,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            static_cast<uint32_t>(_eye_width),
            static_cast<uint32_t>(_eye_height)
        );
        if (_has_depth_extension) {
            _copyBufferToImage(
                upload.command_buffer,
                upload.depth_staging,
                _depth_images[eye][depth_index].image,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                static_cast<uint32_t>(_eye_width),
                static_cast<uint32_t>(_eye_height)
            );
        }

        RS_VK_CHECK(vkEndCommandBuffer(upload.command_buffer));

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &upload.command_buffer;
        RS_VK_CHECK(vkResetFences(_vk_device, 1, &upload.fence));
        RS_VK_CHECK(vkQueueSubmit(_vk_queue, 1, &submit_info, upload.fence));
        // The runtime synchronizes against this submission on the shared queue before compositing
        // the released image, so the present thread does not block on the GPU here; the next
        // frame's wait at the top guards the staging buffers.
    }

    void _copyBufferToImage(
        VkCommandBuffer command_buffer,
        VkBuffer buffer,
        VkImage image,
        VkImageAspectFlags aspect,
        VkImageLayout final_layout,
        uint32_t width,
        uint32_t height
    ) {
        VkImageMemoryBarrier to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image = image;
        to_transfer.subresourceRange = {aspect, 0, 1, 0, 1};
        to_transfer.srcAccessMask = 0;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_transfer
        );

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {aspect, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(
            command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
        );

        VkImageMemoryBarrier to_final{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_final.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_final.newLayout = final_layout;
        to_final.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_final.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_final.image = image;
        to_final.subresourceRange = {aspect, 0, 1, 0, 1};
        to_final.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_final.dstAccessMask = 0;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_final
        );
    }

    // Release any pending frame so the component's staging claim succeeds again after a session
    // drop.
    void _drainPending() { _has_pending.store(false, std::memory_order_release); }

    /****************************************************************
     * Input
     ****************************************************************/
    void _initInput() {
        XrActionSetCreateInfo set_info{XR_TYPE_ACTION_SET_CREATE_INFO};
        std::strcpy(set_info.actionSetName, "gameplay");
        std::strcpy(set_info.localizedActionSetName, "Gameplay");
        RS_XR_CHECK(xrCreateActionSet(_xr_instance, &set_info, &_xr_action_set));
        RS_XR_CHECK(xrStringToPath(_xr_instance, "/user/hand/left", &_xr_hand_path[0]));
        RS_XR_CHECK(xrStringToPath(_xr_instance, "/user/hand/right", &_xr_hand_path[1]));

        auto makeAction = [&](const char* name,
                              const char* localized,
                              XrActionType type,
                              bool both_hands,
                              XrAction& action) {
            XrActionCreateInfo info{XR_TYPE_ACTION_CREATE_INFO};
            std::strcpy(info.actionName, name);
            std::strcpy(info.localizedActionName, localized);
            info.actionType = type;
            info.countSubactionPaths = both_hands ? 2 : 0;
            info.subactionPaths = both_hands ? _xr_hand_path : nullptr;
            RS_XR_CHECK(xrCreateAction(_xr_action_set, &info, &action));
        };
        makeAction("grip_pose", "Grip Pose", XR_ACTION_TYPE_POSE_INPUT, true, _xr_grip_pose);
        makeAction("aim_pose", "Aim Pose", XR_ACTION_TYPE_POSE_INPUT, true, _xr_aim_pose);
        makeAction("grab", "Grab", XR_ACTION_TYPE_FLOAT_INPUT, true, _xr_grab);
        makeAction("push", "Push", XR_ACTION_TYPE_FLOAT_INPUT, true, _xr_push);
        makeAction("move", "Move", XR_ACTION_TYPE_VECTOR2F_INPUT, false, _xr_move);
        makeAction("turn", "Turn", XR_ACTION_TYPE_VECTOR2F_INPUT, false, _xr_turn);
        makeAction("menu", "Menu", XR_ACTION_TYPE_BOOLEAN_INPUT, false, _xr_menu);
        makeAction("boost", "Boost", XR_ACTION_TYPE_BOOLEAN_INPUT, false, _xr_boost);

        _suggestProfiles();

        XrSessionActionSetsAttachInfo attach_info{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
        attach_info.countActionSets = 1;
        attach_info.actionSets = &_xr_action_set;
        RS_XR_CHECK(xrAttachSessionActionSets(_xr_session, &attach_info));

        XrActionSpaceCreateInfo space_info{XR_TYPE_ACTION_SPACE_CREATE_INFO};
        space_info.poseInActionSpace.orientation.w = 1.0f;
        for (uint32_t hand = 0; hand < 2; ++hand) {
            space_info.subactionPath = _xr_hand_path[hand];
            space_info.action = _xr_grip_pose;
            RS_XR_CHECK(xrCreateActionSpace(_xr_session, &space_info, &_xr_grip_space[hand]));
            space_info.action = _xr_aim_pose;
            RS_XR_CHECK(xrCreateActionSpace(_xr_session, &space_info, &_xr_aim_space[hand]));
        }
    }

    // Suggest bindings for one interaction profile; a profile the runtime rejects is skipped.
    void _suggestProfile(
        const char* profile,
        const std::vector<std::pair<XrAction, const char*>>& binds
    ) {
        XrPath profile_path;
        if (XR_FAILED(xrStringToPath(_xr_instance, profile, &profile_path))) return;
        std::vector<XrActionSuggestedBinding> bindings;
        bindings.reserve(binds.size());
        for (const auto& bind : binds) {
            XrPath path;
            if (XR_FAILED(xrStringToPath(_xr_instance, bind.second, &path))) continue;
            bindings.push_back({bind.first, path});
        }
        XrInteractionProfileSuggestedBinding suggestion{
            XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING
        };
        suggestion.interactionProfile = profile_path;
        suggestion.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
        suggestion.suggestedBindings = bindings.data();
        if (XR_FAILED(xrSuggestInteractionProfileBindings(_xr_instance, &suggestion))) {
            RS_LOG(_logger, info, "VR: interaction profile not accepted: {}", profile);
        }
    }

    // Bind the abstract actions to every common stick-based profile plus the universal fallback,
    // so the runtime binds whatever device is connected.
    void _suggestProfiles() {
        _suggestProfile(
            "/interaction_profiles/khr/simple_controller",
            {{_xr_grip_pose, "/user/hand/left/input/grip/pose"},
             {_xr_grip_pose, "/user/hand/right/input/grip/pose"},
             {_xr_aim_pose, "/user/hand/left/input/aim/pose"},
             {_xr_aim_pose, "/user/hand/right/input/aim/pose"},
             {_xr_push, "/user/hand/left/input/select/click"},
             {_xr_push, "/user/hand/right/input/select/click"},
             {_xr_menu, "/user/hand/left/input/menu/click"}}
        );
        _suggestProfile(
            "/interaction_profiles/oculus/touch_controller",
            {{_xr_grip_pose, "/user/hand/left/input/grip/pose"},
             {_xr_grip_pose, "/user/hand/right/input/grip/pose"},
             {_xr_aim_pose, "/user/hand/left/input/aim/pose"},
             {_xr_aim_pose, "/user/hand/right/input/aim/pose"},
             {_xr_grab, "/user/hand/left/input/squeeze/value"},
             {_xr_grab, "/user/hand/right/input/squeeze/value"},
             {_xr_push, "/user/hand/left/input/trigger/value"},
             {_xr_push, "/user/hand/right/input/trigger/value"},
             {_xr_move, "/user/hand/left/input/thumbstick"},
             {_xr_turn, "/user/hand/right/input/thumbstick"},
             {_xr_boost, "/user/hand/left/input/thumbstick/click"},
             {_xr_menu, "/user/hand/left/input/menu/click"}}
        );
        _suggestProfile(
            "/interaction_profiles/valve/index_controller",
            {{_xr_grip_pose, "/user/hand/left/input/grip/pose"},
             {_xr_grip_pose, "/user/hand/right/input/grip/pose"},
             {_xr_aim_pose, "/user/hand/left/input/aim/pose"},
             {_xr_aim_pose, "/user/hand/right/input/aim/pose"},
             {_xr_grab, "/user/hand/left/input/squeeze/value"},
             {_xr_grab, "/user/hand/right/input/squeeze/value"},
             {_xr_push, "/user/hand/left/input/trigger/value"},
             {_xr_push, "/user/hand/right/input/trigger/value"},
             {_xr_move, "/user/hand/left/input/thumbstick"},
             {_xr_turn, "/user/hand/right/input/thumbstick"},
             {_xr_boost, "/user/hand/left/input/thumbstick/click"},
             {_xr_menu, "/user/hand/left/input/b/click"}}
        );
        _suggestProfile(
            "/interaction_profiles/microsoft/motion_controller",
            {{_xr_grip_pose, "/user/hand/left/input/grip/pose"},
             {_xr_grip_pose, "/user/hand/right/input/grip/pose"},
             {_xr_aim_pose, "/user/hand/left/input/aim/pose"},
             {_xr_aim_pose, "/user/hand/right/input/aim/pose"},
             {_xr_grab, "/user/hand/left/input/squeeze/click"},
             {_xr_grab, "/user/hand/right/input/squeeze/click"},
             {_xr_push, "/user/hand/left/input/trigger/value"},
             {_xr_push, "/user/hand/right/input/trigger/value"},
             {_xr_move, "/user/hand/left/input/thumbstick"},
             {_xr_turn, "/user/hand/right/input/thumbstick"},
             {_xr_boost, "/user/hand/left/input/thumbstick/click"},
             {_xr_menu, "/user/hand/left/input/menu/click"}}
        );
        if (_has_pico_extension) {
            for (const char* profile :
                 {"/interaction_profiles/bytedance/pico_neo3_controller",
                  "/interaction_profiles/bytedance/pico4_controller"}) {
                _suggestProfile(
                    profile,
                    {{_xr_grip_pose, "/user/hand/left/input/grip/pose"},
                     {_xr_grip_pose, "/user/hand/right/input/grip/pose"},
                     {_xr_aim_pose, "/user/hand/left/input/aim/pose"},
                     {_xr_aim_pose, "/user/hand/right/input/aim/pose"},
                     {_xr_grab, "/user/hand/left/input/squeeze/click"},
                     {_xr_grab, "/user/hand/right/input/squeeze/click"},
                     {_xr_push, "/user/hand/left/input/trigger/value"},
                     {_xr_push, "/user/hand/right/input/trigger/value"},
                     {_xr_move, "/user/hand/left/input/thumbstick"},
                     {_xr_turn, "/user/hand/right/input/thumbstick"},
                     {_xr_boost, "/user/hand/left/input/thumbstick/click"},
                     {_xr_menu, "/user/hand/left/input/menu/click"}}
                );
            }
        }
        if (_has_focus3_extension) {
            _suggestProfile(
                "/interaction_profiles/htc/vive_focus3_controller",
                {{_xr_grip_pose, "/user/hand/left/input/grip/pose"},
                 {_xr_grip_pose, "/user/hand/right/input/grip/pose"},
                 {_xr_aim_pose, "/user/hand/left/input/aim/pose"},
                 {_xr_aim_pose, "/user/hand/right/input/aim/pose"},
                 {_xr_grab, "/user/hand/left/input/squeeze/value"},
                 {_xr_grab, "/user/hand/right/input/squeeze/value"},
                 {_xr_push, "/user/hand/left/input/trigger/value"},
                 {_xr_push, "/user/hand/right/input/trigger/value"},
                 {_xr_move, "/user/hand/left/input/thumbstick"},
                 {_xr_turn, "/user/hand/right/input/thumbstick"},
                 {_xr_menu, "/user/hand/left/input/menu/click"}}
            );
        }
    }

    void _syncInput(XrTime display_time) {
        if (_xr_action_set == XR_NULL_HANDLE) return;
        XrActiveActionSet active{_xr_action_set, XR_NULL_PATH};
        XrActionsSyncInfo sync_info{XR_TYPE_ACTIONS_SYNC_INFO};
        sync_info.countActiveActionSets = 1;
        sync_info.activeActionSets = &active;
        if (XR_FAILED(xrSyncActions(_xr_session, &sync_info))) return;

        RFloat dt = 0;
        if (_last_input_time != 0 && display_time > _last_input_time) {
            dt = static_cast<RFloat>(static_cast<double>(display_time - _last_input_time) * 1e-9);
        }
        _last_input_time = display_time;

        _applyLocomotion(
            _getVec2Action(_xr_move), _getVec2Action(_xr_turn), _getBoolAction(_xr_boost), dt
        );

        bool is_menu = _getBoolAction(_xr_menu);
        if (is_menu && !_was_menu_pressed) _is_hud_visible = !_is_hud_visible;
        _was_menu_pressed = is_menu;

        for (uint32_t hand = 0; hand < 2; ++hand) {
            HandIntent intent;
            RVec3rf grip_position;
            RQuat3rf grip_orientation;
            RVec3rf aim_origin;
            RQuat3rf aim_orientation;
            bool has_grip =
                _locateHand(_xr_grip_space[hand], display_time, grip_position, grip_orientation);
            bool has_aim =
                _locateHand(_xr_aim_space[hand], display_time, aim_origin, aim_orientation);
            {
                std::scoped_lock lock(_eye_mutex);
                if (has_grip) {
                    intent.position = _getPlayPosition(grip_position);
                    intent.orientation = _getPlayOrientation(grip_orientation);
                }
                intent.is_aim_valid = has_aim;
                if (has_aim) {
                    intent.aim_origin = _getPlayPosition(aim_origin);
                    intent.aim_orientation = _getPlayOrientation(aim_orientation);
                }
            }
            intent.push_amount = _getFloatAction(_xr_push, hand);
            intent.is_push = intent.push_amount > RFloat(0.1);
            intent.is_grab = _getFloatAction(_xr_grab, hand) > RFloat(0.5);
            {
                std::scoped_lock lock(_intent_mutex);
                _intents[hand] = intent;
            }

            // Keep the raw physical aim and grip so the ray and marker quads track each controller
            // live; the grip drives the marker even when the aim pose is not tracked.
            _has_ctrl_ray_aim[hand] = has_aim;
            if (has_aim) {
                _ctrl_ray_origin_room[hand] = aim_origin;
                _ctrl_ray_dir_room[hand] = aim_orientation.toOrthoAxisX();
            }
            _has_ctrl_grip_room[hand] = has_grip;
            if (has_grip) _ctrl_grip_origin_room[hand] = grip_position;
        }
    }

    bool _locateHand(
        XrSpace space,
        XrTime display_time,
        RVec3rf& position_out,
        RQuat3rf& orientation_out
    ) {
        if (space == XR_NULL_HANDLE) return false;
        XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
        if (XR_FAILED(xrLocateSpace(space, _xr_space, display_time, &location))) return false;
        // Only output position and return true if actively tracked, not merely valid.
        if (!(location.locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT) ||
            !(location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)) {
            return false;
        }
        position_out = _getRiseVector(location.pose.position);
        orientation_out = _getRiseOrientation(location.pose.orientation);
        return true;
    }

    RVec2rf _getVec2Action(XrAction action) {
        XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
        info.action = action;
        XrActionStateVector2f state{XR_TYPE_ACTION_STATE_VECTOR2F};
        if (XR_FAILED(xrGetActionStateVector2f(_xr_session, &info, &state)) || !state.isActive) {
            return RVec2rf(0, 0);
        }
        return RVec2rf(state.currentState.x, state.currentState.y);
    }

    RFloat _getFloatAction(XrAction action, uint32_t hand) {
        XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
        info.action = action;
        info.subactionPath = _xr_hand_path[hand];
        XrActionStateFloat state{XR_TYPE_ACTION_STATE_FLOAT};
        if (XR_FAILED(xrGetActionStateFloat(_xr_session, &info, &state)) || !state.isActive) {
            return 0;
        }
        return state.currentState;
    }

    bool _getBoolAction(XrAction action) {
        XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
        info.action = action;
        XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
        if (XR_FAILED(xrGetActionStateBoolean(_xr_session, &info, &state)) || !state.isActive) {
            return false;
        }
        return state.currentState == XR_TRUE;
    }

    /****************************************************************
     * Locomotion
     ****************************************************************/
    // Integrate the play-space offset from this cycle's sticks; caller holds no lock.
    void _applyLocomotion(RVec2rf move, RVec2rf turn, bool is_boost, RFloat dt) {
        std::scoped_lock lock(_eye_mutex);
        RVec3rf forward = _getPlayOrientation(_eyes[0].orientation).toOrthoAxisX();
        RVec3rf flat_forward = RVec3rf(forward.x, forward.y, 0);
        if (flat_forward.length() > RFloat(RS_EPS)) flat_forward = flat_forward.normalized();
        RVec3rf flat_right = flat_forward.cross(RVec3rf(0, 0, 1));

        RFloat speed = _move_speed * (is_boost ? _boost : RFloat(1.0));
        if (std::abs(move.x) > _stick_dead_zone || std::abs(move.y) > _stick_dead_zone) {
            _play_position = _play_position + flat_forward * (move.y * speed * dt) +
                             flat_right * (move.x * speed * dt);
        }
        if (std::abs(turn.y) > _stick_dead_zone) {
            _play_position.z += turn.y * _fly_speed * dt;
        }
        // Smooth continuous yaw about the head, so the world pivots around the user.
        if (std::abs(turn.x) > _stick_dead_zone) {
            RVec3rf head_raw = _eyes[0].position;
            RVec3rf head_world = _getPlayPosition(head_raw);
            _play_yaw -= turn.x * _turn_speed * dt;
            _play_position = head_world - _yaw(_play_yaw).rotateVec3D(head_raw);
        }
    }

    // Compose the play-space offset onto a room-space point; caller holds _eye_mutex.
    [[nodiscard]] RVec3rf _getPlayPosition(const RVec3rf& position) const {
        return _play_position + _yaw(_play_yaw).rotateVec3D(position);
    }

    // Compose the play-space yaw onto a room-space orientation; caller holds _eye_mutex.
    [[nodiscard]] RQuat3rf _getPlayOrientation(const RQuat3rf& orientation) const {
        return _yaw(_play_yaw) * orientation;
    }

    static RQuat3rf _yaw(RFloat angle) { return RQuat3rf(angle, RVec3rf(0, 0, 1)); }

    /****************************************************************
     * HUD
     ****************************************************************/
    void _initHud() {
        XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        info.format = static_cast<int64_t>(_color_format);
        info.sampleCount = 1;
        info.width = static_cast<uint32_t>(_hud_width);
        info.height = static_cast<uint32_t>(_hud_height);
        info.faceCount = 1;
        info.arraySize = 1;
        info.mipCount = 1;
        RS_XR_CHECK(xrCreateSwapchain(_xr_session, &info, &_xr_hud_swapchain));
        uint32_t image_count = 0;
        RS_XR_CHECK(xrEnumerateSwapchainImages(_xr_hud_swapchain, 0, &image_count, nullptr));
        _hud_images.assign(image_count, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        RS_XR_CHECK(xrEnumerateSwapchainImages(
            _xr_hud_swapchain,
            image_count,
            &image_count,
            reinterpret_cast<XrSwapchainImageBaseHeader*>(_hud_images.data())
        ));

        // The render pass clears to fully transparent so the panels composite over the world.
        VkAttachmentDescription color{};
        color.format = _color_format;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &reference;
        VkRenderPassCreateInfo pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        pass_info.attachmentCount = 1;
        pass_info.pAttachments = &color;
        pass_info.subpassCount = 1;
        pass_info.pSubpasses = &subpass;
        RS_VK_CHECK(vkCreateRenderPass(_vk_device, &pass_info, nullptr, &_hud_render_pass));

        _hud_image_views.resize(image_count);
        _hud_framebuffers.resize(image_count);
        for (uint32_t i = 0; i < image_count; ++i) {
            VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view_info.image = _hud_images[i].image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = _color_format;
            view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            RS_VK_CHECK(vkCreateImageView(_vk_device, &view_info, nullptr, &_hud_image_views[i]));
            VkFramebufferCreateInfo fb_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fb_info.renderPass = _hud_render_pass;
            fb_info.attachmentCount = 1;
            fb_info.pAttachments = &_hud_image_views[i];
            fb_info.width = static_cast<uint32_t>(_hud_width);
            fb_info.height = static_cast<uint32_t>(_hud_height);
            fb_info.layers = 1;
            RS_VK_CHECK(vkCreateFramebuffer(_vk_device, &fb_info, nullptr, &_hud_framebuffers[i]));
        }

        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 16;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        RS_VK_CHECK(vkCreateDescriptorPool(_vk_device, &pool_info, nullptr, &_hud_descriptor_pool));

        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = _command_pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        RS_VK_CHECK(vkAllocateCommandBuffers(_vk_device, &alloc, &_hud_command_buffer));
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        RS_VK_CHECK(vkCreateFence(_vk_device, &fence_info, nullptr, &_hud_fence));

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.DisplaySize = ImVec2(static_cast<float>(_hud_width), static_cast<float>(_hud_height));
#ifdef RS_VR_HUD_FONT_PATH
        // The bundled Rajdhani tech font, drawn at per-element sizes through PushFont; the layout
        // falls back to the built-in font if the file is missing.
        if (std::FILE* font_file = std::fopen(RS_VR_HUD_FONT_PATH, "rb")) {
            std::fclose(font_file);
            _hud_font = io.Fonts->AddFontFromFileTTF(RS_VR_HUD_FONT_PATH, 64.0f);
        }
#endif
        _applyHudStyle();
        ImGui_ImplVulkan_InitInfo init{};
        init.ApiVersion = VK_API_VERSION_1_1;
        init.Instance = _vk_instance;
        init.PhysicalDevice = _vk_physical_device;
        init.Device = _vk_device;
        init.QueueFamily = _queue_family_index;
        init.Queue = _vk_queue;
        init.DescriptorPool = _hud_descriptor_pool;
        init.MinImageCount = 2;
        init.ImageCount = image_count < 2 ? 2 : image_count;
        init.PipelineInfoMain.RenderPass = _hud_render_pass;
        ImGui_ImplVulkan_Init(&init);
        _is_hud_ready = true;
    }

    // Anchor the HUD to a lazily-followed body frame: it tracks the head position and heading but
    // ignores head pitch, roll, and quick glances, so it stays world anchored during head motion,
    // which the runtime reprojects crisply, and re-centers as the body turns.
    XrPosef _getBodyHudPose(const XrPosef& head, RFloat dt) {
        RQuat3rf head_orientation(
            head.orientation.x, head.orientation.y, head.orientation.z, head.orientation.w
        );
        RVec3rf forward = head_orientation.rotateVec3D(RVec3rf(0, 0, -1));
        RFloat head_yaw = std::atan2(-forward.x, -forward.z);
        RVec3rf head_position(head.position.x, head.position.y, head.position.z);

        if (!_has_hud_anchor) {
            _hud_yaw = head_yaw;
            _hud_position = head_position;
            _has_hud_anchor = true;
        }
        RFloat rate = RS_MIN(dt / _hud_follow_tau, RFloat(1));
        _hud_position = _hud_position + (head_position - _hud_position) * rate;
        RFloat yaw_delta = head_yaw - _hud_yaw;
        while (yaw_delta > RFloat(RS_PI)) yaw_delta -= RFloat(2 * RS_PI);
        while (yaw_delta < -RFloat(RS_PI)) yaw_delta += RFloat(2 * RS_PI);
        _hud_yaw = _hud_yaw + yaw_delta * rate;

        RQuat3rf yaw(_hud_yaw, RVec3rf(0, 1, 0));
        RVec3rf position = _hud_position + yaw.rotateVec3D(RVec3rf(0, -0.28, -0.6));
        RQuat3rf orientation = (yaw * RQuat3rf(-0.2164f, 0.0f, 0.0f, 0.9763f)).normalized();

        XrPosef pose;
        pose.position = {
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z)
        };
        pose.orientation = {
            static_cast<float>(orientation.x),
            static_cast<float>(orientation.y),
            static_cast<float>(orientation.z),
            static_cast<float>(orientation.w)
        };
        return pose;
    }

    // Render the HUD into its quad swapchain image; the latest image is composited every frame.
    void _renderHud() {
        if (!_is_hud_ready) return;
        // Wait on the previous HUD render, finished long ago at this reduced rate, so the command
        // buffer and fence are free to reuse without stalling the present thread.
        vkWaitForFences(_vk_device, 1, &_hud_fence, VK_TRUE, UINT64_MAX);

        uint32_t index = 0;
        XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        if (XR_FAILED(xrAcquireSwapchainImage(_xr_hud_swapchain, &acquire, &index))) return;
        XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        wait.timeout = RS_VR_SWAPCHAIN_WAIT_TIMEOUT;
        if (xrWaitSwapchainImage(_xr_hud_swapchain, &wait) != XR_SUCCESS) {
            XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            xrReleaseSwapchainImage(_xr_hud_swapchain, &release);
            return;
        }

        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(RS_VR_HUD_REFERENCE_WIDTH, RS_VR_HUD_REFERENCE_HEIGHT);
        io.DisplayFramebufferScale = ImVec2(
            static_cast<float>(_hud_width) / RS_VR_HUD_REFERENCE_WIDTH,
            static_cast<float>(_hud_height) / RS_VR_HUD_REFERENCE_HEIGHT
        );
        io.DeltaTime = 1.0f / 90.0f;
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();
        _drawHud();
        ImGui::Render();

        vkResetCommandBuffer(_hud_command_buffer, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(_hud_command_buffer, &begin);
        VkClearValue clear{};
        clear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        pass.renderPass = _hud_render_pass;
        pass.framebuffer = _hud_framebuffers[index];
        pass.renderArea.extent = {
            static_cast<uint32_t>(_hud_width), static_cast<uint32_t>(_hud_height)
        };
        pass.clearValueCount = 1;
        pass.pClearValues = &clear;
        vkCmdBeginRenderPass(_hud_command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), _hud_command_buffer);
        vkCmdEndRenderPass(_hud_command_buffer);
        vkEndCommandBuffer(_hud_command_buffer);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &_hud_command_buffer;
        vkResetFences(_vk_device, 1, &_hud_fence);
        vkQueueSubmit(_vk_queue, 1, &submit, _hud_fence);
        // The runtime synchronizes against this submission before compositing the released image,
        // so the present thread does not block on the GPU here.

        XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(_xr_hud_swapchain, &release);
        _has_hud_image = true;
    }

    // Draw the built-in three-column status panel from the latest HudInfo snapshot. Coordinates are
    // absolute in the zero-padded window, so each element sits at a deliberate position and size.
    void _drawHud() {
        HudInfo info;
        {
            std::scoped_lock lock(_hud_info_mutex);
            info = _hud_info;
        }
        const ImVec4 value_color(0.85f, 0.92f, 0.97f, 1.0f);
        const ImVec4 accent_color(0.25f, 0.85f, 0.95f, 1.0f);
        const ImVec4 dim_color(0.55f, 0.70f, 0.78f, 1.0f);
        const ImVec4 good_color(0.35f, 0.92f, 0.55f, 1.0f);
        const ImVec4 warn_color(0.95f, 0.55f, 0.25f, 1.0f);
        const ImU32 accent = IM_COL32(64, 217, 242, 255);
        const ImU32 rule = IM_COL32(64, 217, 242, 90);
        const ImU32 gauge_dim = IM_COL32(40, 60, 70, 120);

        // Type scale, in pixels, chosen so each readout fills its band without overflowing.
        constexpr float label_size = 58.0f;
        constexpr float sublabel_size = 40.0f;
        constexpr float time_size = 104.0f;
        constexpr float memory_size = 92.0f;
        constexpr float status_size = 60.0f;

        const float width = RS_VR_HUD_REFERENCE_WIDTH;
        const float height = RS_VR_HUD_REFERENCE_HEIGHT;
        const float column_width = width / 4.0f;
        const float margin = 48.0f;

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(width, height));
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNav;
        ImGui::Begin("rise origin monitor", nullptr, flags);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 origin = ImGui::GetWindowPos();

        // Futuristic frame: a top accent rule, column dividers, and corner ticks.
        draw_list->AddLine(
            ImVec2(origin.x + margin, origin.y + 22),
            ImVec2(origin.x + width - margin, origin.y + 22),
            accent,
            3.0f
        );
        for (int i = 1; i < 4; ++i) {
            float x = origin.x + i * column_width;
            draw_list->AddLine(
                ImVec2(x, origin.y + 48), ImVec2(x, origin.y + height - 40), rule, 2.0f
            );
        }
        const float tick = 26.0f;
        draw_list->AddLine(
            ImVec2(origin.x + 16, origin.y + 16),
            ImVec2(origin.x + 16 + tick, origin.y + 16),
            accent,
            3.0f
        );
        draw_list->AddLine(
            ImVec2(origin.x + 16, origin.y + 16),
            ImVec2(origin.x + 16, origin.y + 16 + tick),
            accent,
            3.0f
        );
        draw_list->AddLine(
            ImVec2(origin.x + width - 16 - tick, origin.y + height - 16),
            ImVec2(origin.x + width - 16, origin.y + height - 16),
            accent,
            3.0f
        );
        draw_list->AddLine(
            ImVec2(origin.x + width - 16, origin.y + height - 16 - tick),
            ImVec2(origin.x + width - 16, origin.y + height - 16),
            accent,
            3.0f
        );

        auto text = [&](float x, float y, float size, const ImVec4& color, const char* body) {
            ImGui::PushFont(_hud_font, size);
            ImGui::SetCursorPos(ImVec2(x, y));
            ImGui::TextColored(color, "%s", body);
            ImGui::PopFont();
        };

        const float c0 = margin;
        const float c1 = column_width + margin;
        const float c2 = 2 * column_width + margin;
        const float c3 = 3 * column_width + margin;
        const float bar_width = column_width - 2 * margin;

        text(c0, 40, label_size, accent_color, "SIMULATION");
        text(c0, 140, sublabel_size, dim_color, "WORLD TIME");
        text(c0, 188, time_size, value_color, _getTimeString(info.sim_time).c_str());
        text(c0, 348, status_size, good_color, "NOMINAL");

        // A load at or below real time reads good, above it warns; the gauge fills toward
        // saturation.
        bool is_behind = info.world_load > 1.0f;
        ImVec4 load_color = !info.is_world_load_valid ? dim_color
                            : is_behind               ? warn_color
                                                      : good_color;
        ImU32 load_gauge = is_behind ? IM_COL32(242, 140, 64, 255) : IM_COL32(90, 235, 140, 255);
        RFloat load_fraction = info.is_world_load_valid ? RS_MIN(info.world_load, RFloat(1))
                                                        : RFloat(0);
        text(c1, 40, label_size, accent_color, "WORLD LOAD");
        text(c1, 140, sublabel_size, dim_color, "VS REALTIME");
        text(
            c1,
            196,
            memory_size,
            load_color,
            info.is_world_load_valid ? _getLoadString(info.world_load).c_str() : "N/A"
        );
        ImGui::SetCursorPos(ImVec2(c1, 360));
        _drawSegmentBar(load_fraction, ImVec2(bar_width, 34), load_gauge, gauge_dim);

        text(c2, 40, label_size, accent_color, "GPU MEMORY");
        text(c2, 140, sublabel_size, dim_color, info.device_name.c_str());
        text(
            c2,
            196,
            memory_size,
            value_color,
            _getMemoryString(info.device_used, info.device_total).c_str()
        );
        RFloat device_fraction =
            info.device_total > 0 ? static_cast<RFloat>(info.device_used) / info.device_total : 0;
        ImGui::SetCursorPos(ImVec2(c2, 360));
        _drawSegmentBar(device_fraction, ImVec2(bar_width, 34), accent, gauge_dim);

        text(c3, 40, label_size, accent_color, "HOST MEMORY");
        text(c3, 140, sublabel_size, dim_color, "SYSTEM");
        text(
            c3,
            196,
            memory_size,
            value_color,
            _getMemoryString(info.host_used, info.host_total).c_str()
        );
        RFloat host_fraction =
            info.host_total > 0 ? static_cast<RFloat>(info.host_used) / info.host_total : 0;
        ImGui::SetCursorPos(ImVec2(c3, 360));
        _drawSegmentBar(host_fraction, ImVec2(bar_width, 34), accent, gauge_dim);

        ImGui::End();
    }

    // One translucent, hard-edged, cyan-accented style for the see-through monitor look.
    void _applyHudStyle() {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 0;
        style.FrameRounding = 0;
        style.GrabRounding = 0;
        style.WindowBorderSize = 1;
        style.FrameBorderSize = 1;
        style.WindowPadding = ImVec2(0, 0);
        ImVec4* colors = style.Colors;
        colors[ImGuiCol_WindowBg] = ImVec4(0.04f, 0.06f, 0.08f, 0.35f);
        colors[ImGuiCol_ChildBg] = ImVec4(0.04f, 0.06f, 0.08f, 0.35f);
        colors[ImGuiCol_Border] = ImVec4(0.25f, 0.85f, 0.95f, 1.0f);
        colors[ImGuiCol_Text] = ImVec4(0.80f, 0.86f, 0.92f, 1.0f);
        colors[ImGuiCol_Separator] = ImVec4(0.25f, 0.85f, 0.95f, 0.5f);
    }

    /****************************************************************
     * Controller
     ****************************************************************/
    // Create the controller texture atlas once per session and fill its beam, reticle, and marker
    // cells.
    void _initController() {
        const uint32_t width = 8 * RS_VR_RAY_CELL;
        const uint32_t height = RS_VR_RAY_CELL;

        XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
                          XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
        info.format = static_cast<int64_t>(_color_format);
        info.sampleCount = 1;
        info.width = width;
        info.height = height;
        info.faceCount = 1;
        info.arraySize = 1;
        info.mipCount = 1;
        RS_XR_CHECK(xrCreateSwapchain(_xr_session, &info, &_ctrl_xr_ray_swapchain));
        uint32_t image_count = 0;
        RS_XR_CHECK(xrEnumerateSwapchainImages(_ctrl_xr_ray_swapchain, 0, &image_count, nullptr));
        _ctrl_ray_images.assign(image_count, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        RS_XR_CHECK(xrEnumerateSwapchainImages(
            _ctrl_xr_ray_swapchain,
            image_count,
            &image_count,
            reinterpret_cast<XrSwapchainImageBaseHeader*>(_ctrl_ray_images.data())
        ));

        // Draw the atlas into a host buffer, then copy it into an acquired swapchain image once;
        // the released image stays referenced by the quad layers every frame afterward.
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        _makeStagingBuffer(
            static_cast<VkDeviceSize>(width) * height * 4, staging, staging_memory, mapped
        );
        _drawControllerRayAtlas(static_cast<RColor4b*>(mapped), width, height);

        uint32_t index = 0;
        XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        RS_XR_CHECK(xrAcquireSwapchainImage(_ctrl_xr_ray_swapchain, &acquire, &index));
        _waitSwapchainImage(_ctrl_xr_ray_swapchain);

        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = _command_pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        RS_VK_CHECK(vkAllocateCommandBuffers(_vk_device, &alloc, &command_buffer));
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        RS_VK_CHECK(vkCreateFence(_vk_device, &fence_info, nullptr, &fence));

        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        RS_VK_CHECK(vkBeginCommandBuffer(command_buffer, &begin));
        _copyBufferToImage(
            command_buffer,
            staging,
            _ctrl_ray_images[index].image,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            width,
            height
        );
        RS_VK_CHECK(vkEndCommandBuffer(command_buffer));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        RS_VK_CHECK(vkQueueSubmit(_vk_queue, 1, &submit, fence));
        vkWaitForFences(_vk_device, 1, &fence, VK_TRUE, UINT64_MAX);

        XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        RS_XR_CHECK(xrReleaseSwapchainImage(_ctrl_xr_ray_swapchain, &release));

        vkDestroyFence(_vk_device, fence, nullptr);
        vkFreeCommandBuffers(_vk_device, _command_pool, 1, &command_buffer);
        vkDestroyBuffer(_vk_device, staging, nullptr);
        vkFreeMemory(_vk_device, staging_memory, nullptr);
        _is_ctrl_ray_ready = true;
    }

    // Fill the atlas: a beam strip and reticle per mode color plus a white controller marker. The
    // beam cells are neutral, grab, push; the reticle cells are neutral-miss, neutral-hit, grab,
    // push. The beam width maps across the cell, so the soft edge lives along the cell's columns.
    void _drawControllerRayAtlas(RColor4b* pixel, uint32_t width, uint32_t height) {
        for (uint32_t i = 0; i < width * height; ++i) pixel[i] = RColor4b{0, 0, 0, 0};
        const RColor4b cyan{64, 220, 255, 255};
        const RColor4b amber{255, 170, 40, 255};
        const RColor4b green{60, 230, 90, 255};
        const RColor4b red{240, 70, 60, 255};
        _drawControllerRayBeam(pixel, width, 0 * RS_VR_RAY_CELL, cyan);
        _drawControllerRayBeam(pixel, width, 1 * RS_VR_RAY_CELL, green);
        _drawControllerRayBeam(pixel, width, 2 * RS_VR_RAY_CELL, red);
        _drawControllerRayReticle(pixel, width, 3 * RS_VR_RAY_CELL, cyan);
        _drawControllerRayReticle(pixel, width, 4 * RS_VR_RAY_CELL, amber);
        _drawControllerRayReticle(pixel, width, 5 * RS_VR_RAY_CELL, green);
        _drawControllerRayReticle(pixel, width, 6 * RS_VR_RAY_CELL, red);
        _drawControllerMarker(pixel, width, 7 * RS_VR_RAY_CELL);
    }

    // Draw a soft-edged beam strip into one atlas cell at the given column offset.
    static void
    _drawControllerRayBeam(RColor4b* pixel, uint32_t width, int32_t offset, RColor4b color) {
        const float center = (RS_VR_RAY_CELL - 1) * 0.5f;
        for (int32_t y = 0; y < RS_VR_RAY_CELL; ++y) {
            for (int32_t x = 0; x < RS_VR_RAY_CELL; ++x) {
                float across = 1.0f - std::fabs((x - center) / center);
                color.a = static_cast<RByte>(RS_MIN(across * 2.5f, 1.0f) * 240.0f);
                pixel[y * width + offset + x] = color;
            }
        }
    }

    // Draw a ring plus a center dot into one atlas cell at the given column offset.
    static void
    _drawControllerRayReticle(RColor4b* pixel, uint32_t width, int32_t offset, RColor4b color) {
        const float center = (RS_VR_RAY_CELL - 1) * 0.5f;
        for (int32_t y = 0; y < RS_VR_RAY_CELL; ++y) {
            for (int32_t x = 0; x < RS_VR_RAY_CELL; ++x) {
                float dx = x - center;
                float dy = y - center;
                float radius = std::sqrt(dx * dx + dy * dy);
                bool is_ring = radius >= 22.0f && radius <= 28.0f;
                bool is_dot = radius <= 4.0f;
                if (is_ring || is_dot) pixel[y * width + offset + x] = color;
            }
        }
    }

    // Draw the controller marker: a bright white boundary ring around a faint translucent fill, so
    // the billboard reads as a see-through ball at the controller.
    static void _drawControllerMarker(RColor4b* pixel, uint32_t width, int32_t offset) {
        const float center = (RS_VR_RAY_CELL - 1) * 0.5f;
        for (int32_t y = 0; y < RS_VR_RAY_CELL; ++y) {
            for (int32_t x = 0; x < RS_VR_RAY_CELL; ++x) {
                float dx = x - center;
                float dy = y - center;
                float radius = std::sqrt(dx * dx + dy * dy);
                if (radius > 30.0f) continue;
                RByte alpha = radius >= 25.0f ? 255 : 45;
                pixel[y * width + offset + x] = RColor4b{255, 255, 255, alpha};
            }
        }
    }

    // Build the beam and reticle quads for one hand from its live physical aim, or return false.
    bool _getControllerRayQuads(
        uint32_t hand,
        const XrPosef& head_pose,
        XrCompositionLayerQuad& beam,
        XrCompositionLayerQuad& reticle
    ) {
        bool is_active;
        RFloat length;
        bool is_hit;
        RayMode mode;
        {
            std::scoped_lock lock(_ctrl_ray_mutex);
            is_active = _is_ctrl_ray_active[hand];
            length = _ctrl_ray_length[hand];
            is_hit = _is_ctrl_ray_hit[hand];
            mode = _ctrl_ray_mode[hand];
        }
        // Beam and reticle color by gesture: neutral cyan, grab green, push red.
        int beam_cell = mode == RayMode::GRAB ? 1 : mode == RayMode::PUSH ? 2 : 0;
        int reticle_cell = mode == RayMode::GRAB ? 5 : mode == RayMode::PUSH ? 6 : is_hit ? 4 : 3;
        if (!is_active || !_has_ctrl_ray_aim[hand] || length <= RFloat(0)) return false;
        RVec3rf direction = _ctrl_ray_dir_room[hand];
        if (direction.length() < RFloat(RS_EPS)) return false;
        direction = direction.normalized();

        // The beam is a short stub from the controller, clipped to a near target; the reticle marks
        // the target at the hit distance, or a modest reach when nothing is hit.
        RFloat reticle_distance = is_hit ? length : RS_MIN(length, _ctrl_ray_reach);
        RFloat beam_length = RS_MIN(length, _ctrl_ray_beam_length);
        RVec3rf start = _ctrl_ray_origin_room[hand];
        RVec3rf beam_mid = start + direction * (beam_length * RFloat(0.5));
        RVec3rf reticle_pos = start + direction * reticle_distance;
        RVec3rf head_position = _getRiseVector(head_pose.position);

        // Beam billboard: the long axis runs along the ray, the face turns toward the head.
        RVec3rf beam_right = direction.cross(head_position - beam_mid);
        if (beam_right.length() < RFloat(RS_EPS)) return false;
        beam_right = beam_right.normalized();
        RVec3rf beam_normal = beam_right.cross(direction).normalized();
        beam = _makeControllerRayQuad(
            beam_mid,
            _getControllerBasisOrientation(beam_right, direction, beam_normal),
            _ctrl_ray_beam_width,
            beam_length,
            beam_cell
        );

        // Reticle: a small square at the target facing the head, sized by distance to hold a steady
        // angular size, in the hit cell when the ray lands on a structure.
        RVec3rf reticle_normal = head_position - reticle_pos;
        if (reticle_normal.length() < RFloat(RS_EPS)) reticle_normal = direction * RFloat(-1);
        reticle_normal = reticle_normal.normalized();
        RVec3rf reticle_right = RVec3rf(0, 0, 1).cross(reticle_normal);
        if (reticle_right.length() < RFloat(RS_EPS)) reticle_right = beam_right;
        reticle_right = reticle_right.normalized();
        RVec3rf reticle_up = reticle_normal.cross(reticle_right).normalized();
        RFloat size = RS_MAX(reticle_distance * _ctrl_ray_reticle_fraction, _ctrl_ray_reticle_min);
        reticle = _makeControllerRayQuad(
            reticle_pos,
            _getControllerBasisOrientation(reticle_right, reticle_up, reticle_normal),
            size,
            size,
            reticle_cell
        );
        return true;
    }

    // Build the marker quad for one hand at its live physical grip, a ball facing the head, or
    // return false when the grip is not tracked.
    bool _getControllerMarkerQuad(
        uint32_t hand,
        const XrPosef& head_pose,
        XrCompositionLayerQuad& quad
    ) {
        if (!_has_ctrl_grip_room[hand]) return false;
        RVec3rf position = _ctrl_grip_origin_room[hand];
        RVec3rf normal = _getRiseVector(head_pose.position) - position;
        if (normal.length() < RFloat(RS_EPS)) return false;
        normal = normal.normalized();
        RVec3rf right = RVec3rf(0, 0, 1).cross(normal);
        if (right.length() < RFloat(RS_EPS)) right = RVec3rf(1, 0, 0);
        right = right.normalized();
        RVec3rf up = normal.cross(right).normalized();
        quad = _makeControllerRayQuad(
            position,
            _getControllerBasisOrientation(right, up, normal),
            _ctrl_marker_size,
            _ctrl_marker_size,
            7
        );
        return true;
    }

    // Assemble one world-locked quad layer sampling the given atlas cell.
    XrCompositionLayerQuad _makeControllerRayQuad(
        const RVec3rf& position,
        const RQuat3rf& orientation,
        RFloat width,
        RFloat height,
        int cell
    ) {
        XrCompositionLayerQuad quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
        quad.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                          XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT;
        quad.space = _xr_space;
        quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        quad.subImage.swapchain = _ctrl_xr_ray_swapchain;
        quad.subImage.imageRect.offset = {cell * RS_VR_RAY_CELL, 0};
        quad.subImage.imageRect.extent = {RS_VR_RAY_CELL, RS_VR_RAY_CELL};
        quad.pose.position = _getXrVector(position);
        // The orientation already lives in OpenXR coordinates, so assign it directly; running it
        // through the two-sided head-pose conversion would collapse the quad edge-on.
        quad.pose.orientation = XrQuaternionf{
            static_cast<float>(orientation.x),
            static_cast<float>(orientation.y),
            static_cast<float>(orientation.z),
            static_cast<float>(orientation.w)
        };
        quad.size = {static_cast<float>(width), static_cast<float>(height)};
        return quad;
    }

    // The OpenXR-space orientation whose local x, y, z axes point along the given world-space
    // right, up, and normal. The axes are converted to OpenXR coordinates first so the quad is a
    // proper world-locked billboard instead of an edge-on sliver.
    RQuat3rf _getControllerBasisOrientation(
        const RVec3rf& right,
        const RVec3rf& up,
        const RVec3rf& normal
    ) const {
        XrVector3f x = _getXrVector(right);
        XrVector3f y = _getXrVector(up);
        XrVector3f z = _getXrVector(normal);
        RMat3rf matrix(
            x.x,
            y.x,
            z.x,
            //
            x.y,
            y.y,
            z.y,
            //
            x.z,
            y.z,
            z.z
        );
        RQuat3rf orientation;
        orientation.fromRotationMatrix(matrix);
        return orientation;
    }

    /****************************************************************
     * Coordinates
     ****************************************************************/
    // The OpenXR reference space is right handed with +Y up and the view looking along -Z. The
    // rise-origin world is +Z up and the renderer is right handed too, with image-right equal to
    // -toOrthoAxisY. This proper rotation maps a head pose between the two frames.
    RQuat3rf _makeBasisChange() const {
        RMat3rf matrix(
            0.0,
            0.0,
            -1.0,
            //
            -1.0,
            0.0,
            0.0,
            //
            0.0,
            1.0,
            0.0
        );
        RQuat3rf basis;
        basis.fromRotationMatrix(matrix);
        return basis;
    }

    RVec3rf _getRiseVector(const XrVector3f& vector) const {
        return RVec3rf(-vector.z, -vector.x, vector.y);
    }

    RQuat3rf _getRiseOrientation(const XrQuaternionf& orientation) const {
        RQuat3rf head(orientation.x, orientation.y, orientation.z, orientation.w);
        return (_basis * head * _basis.conjugate()).normalized();
    }

    XrVector3f _getXrVector(const RVec3rf& vector) const {
        return XrVector3f{
            static_cast<float>(-vector.y),
            static_cast<float>(vector.z),
            static_cast<float>(-vector.x)
        };
    }

    XrQuaternionf _getXrOrientation(const RQuat3rf& orientation) const {
        RQuat3rf head = (_basis.conjugate() * orientation * _basis).normalized();
        return XrQuaternionf{
            static_cast<float>(head.x),
            static_cast<float>(head.y),
            static_cast<float>(head.z),
            static_cast<float>(head.w)
        };
    }

    // Convert a physical render pose the component handed back into the OpenXR reference-space pose
    // the projection layer is labeled with, so the runtime reprojects against the physical head.
    XrPosef _getLayerPose(const FramePose& frame_pose) {
        XrPosef pose;
        pose.position = _getXrVector(frame_pose.position);
        pose.orientation = _getXrOrientation(frame_pose.orientation);
        return pose;
    }

    /****************************************************************
     * Static helpers
     ****************************************************************/
    // Pick an RGBA8 color format so the component's RColor4b maps with no swizzle.
    static VkFormat _getColorFormat(
        const std::vector<int64_t>& formats,
        const std::shared_ptr<logging::logger>& logger
    ) {
        const std::array<VkFormat, 2> preferred{VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM};
        for (VkFormat format : preferred) {
            for (int64_t candidate : formats) {
                if (static_cast<VkFormat>(candidate) == format) return format;
            }
        }
        RS_LOG(logger, warn, "VR: no RGBA8 swapchain format offered; colors may be swizzled");
        return formats.empty() ? VK_FORMAT_R8G8B8A8_SRGB : static_cast<VkFormat>(formats[0]);
    }

    // Pick a full-precision float depth format so the copied float32 depth maps with no conversion.
    static VkFormat _getDepthFormat(const std::vector<int64_t>& formats) {
        const std::array<VkFormat, 3> preferred{
            VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM
        };
        for (VkFormat format : preferred) {
            for (int64_t candidate : formats) {
                if (static_cast<VkFormat>(candidate) == format) return format;
            }
        }
        return VK_FORMAT_D32_SFLOAT;
    }

    // Format world time as HH:MM:SS:mmm.
    static std::string _getTimeString(RFloat seconds) {
        long total = static_cast<long>(seconds);
        int milliseconds = static_cast<int>((seconds - static_cast<RFloat>(total)) * 1000.0f);
        char buffer[40];
        std::snprintf(
            buffer,
            sizeof(buffer),
            "%02ld:%02ld:%02ld:%03d",
            total / 3600,
            (total % 3600) / 60,
            total % 60,
            milliseconds
        );
        return buffer;
    }

    // Format a used / total byte pair as gigabytes.
    static std::string _getMemoryString(RBigSize used, RBigSize total) {
        char buffer[32];
        std::snprintf(
            buffer,
            sizeof(buffer),
            "%.2f / %.2f GB",
            static_cast<double>(used) / 1e9,
            static_cast<double>(total) / 1e9
        );
        return buffer;
    }

    // Format the world load as a percentage of real time, such as 10%.
    static std::string _getLoadString(RFloat load) {
        char buffer[16];
        std::snprintf(buffer, sizeof(buffer), "%.0f%%", static_cast<double>(load) * 100.0);
        return buffer;
    }

    // Draw a segmented gauge: a row of small cells filled up to the fraction in the accent color.
    static void _drawSegmentBar(RFloat fraction, ImVec2 size, ImU32 accent, ImU32 dim) {
        constexpr int segment_num = 20;
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        RFloat gap = 2.0f;
        RFloat cell_width = (size.x - gap * (segment_num - 1)) / segment_num;
        int filled = static_cast<int>(fraction * segment_num + 0.5f);
        for (int i = 0; i < segment_num; ++i) {
            RFloat x = origin.x + i * (cell_width + gap);
            draw_list->AddRectFilled(
                ImVec2(x, origin.y),
                ImVec2(x + cell_width, origin.y + size.y),
                i < filled ? accent : dim
            );
        }
        ImGui::Dummy(size);
    }
};

/*****************************************************************
 * RS_WorldSessionComponentVR
 *****************************************************************/
std::atomic<bool> RS_WorldSessionComponentVR::_is_created{false};

RS_WorldSessionComponentVR::RS_WorldSessionComponentVR(
    std::string name,
    RSize samples_per_pixel,
    RSize max_depth,
    RFloat near_depth,
    RFloat far_depth,
    RFloat move_speed,
    RFloat fly_speed,
    RFloat turn_speed,
    RFloat stick_dead_zone,
    RFloat boost,
    RFloat hud_follow_tau
)
    : _name(std::move(name)),
      _vr_controller_name(_name + "#controller"),
      _samples_per_pixel(samples_per_pixel),
      _max_depth(max_depth),
      _near_depth(near_depth),
      _far_depth(far_depth),
      _move_speed(move_speed),
      _fly_speed(fly_speed),
      _turn_speed(turn_speed),
      _stick_dead_zone(stick_dead_zone),
      _boost(boost),
      _hud_follow_tau(hud_follow_tau) {
    bool expected = false;
    if (!_is_created.compare_exchange_strong(expected, true)) {
        RS_THROW(RS_RuntimeException(
            "A VR component already exists; OpenXR drives a single headset session per process"
        ));
    }
    _ray_controller_name[0] = _name + "#ray_left";
    _ray_controller_name[1] = _name + "#ray_right";
    _ray_sensor_name[0] = _name + "#ray_sensor_left";
    _ray_sensor_name[1] = _name + "#ray_sensor_right";
    _manipulation_controller_name[0] = _name + "#manipulation_left";
    _manipulation_controller_name[1] = _name + "#manipulation_right";
}

RS_WorldSessionComponentVR::~RS_WorldSessionComponentVR() { _is_created = false; }

void RS_WorldSessionComponentVR::setPushForce(RFloat push_force) { _push_force = push_force; }

void RS_WorldSessionComponentVR::onAttach(RS_WorldSessionAccess access) {
    _access = std::move(access);
    _xr = std::make_unique<RS_VRImpl>(
        _near_depth,
        _far_depth,
        _move_speed,
        _fly_speed,
        _turn_speed,
        _stick_dead_zone,
        _boost,
        _hud_follow_tau
    );
}

void RS_WorldSessionComponentVR::onCallback(RBigFloat time, RS_ThreadPool& pool) {
    auto intents = _xr->getHandIntents();
    const RS_VRImpl::HandIntent* hand_intents[2] = {&intents.first, &intents.second};
    for (RIndex hand = 0; hand < 2; ++hand) {
        _highlight_gid[hand] = RS_NULL_INDEX;
        if (_vr_state.load() != VRState::ENGAGED) continue;
        if (_manip_state[hand] != ManipulationState::IDLE) {
            _highlight_gid[hand] = _target_gid[hand];
        } else if (hand_intents[hand]->is_aim_valid) {
            RIndex gid = RS_NULL_INDEX;
            RIndex body_sid = RS_NULL_INDEX;
            RIndex segment_bid = RS_NULL_INDEX;
            RFloat distance = 0;
            // A miss leaves gid null, which is the wanted no-highlight value.
            static_cast<void>(_readRayHit(hand, gid, distance, body_sid, segment_bid));
            _highlight_gid[hand] = gid;
        }
    }

    _updateRay();
    if (_vr_state.load() == VRState::ENGAGED) _publishUserState();

    pool.detachTask([this] {
        _syncVRCharacterization();
        if (_vr_state.load() == VRState::ENGAGED) {
            for (RIndex hand = 0; hand < 2; ++hand) _syncManipulationCharacterization(hand);
        }
    });
    pool.detachTask([this, time] { _updateInfo(time); });
    if (_vr_state.load() != VRState::ENGAGED) return;

    // Skip the frame while a publish is in flight or the staging is not free, so the simulation
    // never blocks and no task overwrites staging the present thread still reads. Claiming the
    // staging holds it against a disconnect until the publish task releases it below.
    if (_is_publishing_frame.load() || !_xr->beginStagingWrite()) return;
    _is_publishing_frame.store(true);

    RIndex pixel_num = static_cast<RIndex>(static_cast<RBigSize>(_eye_width) * _eye_height);
    RSize block_num = RS_CEIL_DIV(pixel_num, RS_VR_CONVERT_BLOCK);
    auto convert = std::make_shared<RS_MultiFuture<void>>(pool.submitTaskSequence(
        0, static_cast<RIndex>(2 * block_num), [this, pixel_num, block_num](RIndex i) {
            RIndex eye = i / static_cast<RIndex>(block_num);
            RIndex start = (i % static_cast<RIndex>(block_num)) * RS_VR_CONVERT_BLOCK;
            _convertAndWriteFrame(eye, start, RS_MIN(start + RS_VR_CONVERT_BLOCK, pixel_num));
        }
    ));
    pool.detachTask([this, convert] {
        // Release the staging claim and the publish flag on every exit path, so an exception in
        // convert or commit cannot strand the borrow and hang a disconnect.
        auto guard = std::shared_ptr<void>(nullptr, [this](void*) {
            _xr->endStagingWrite();
            _is_publishing_frame.store(false);
        });
        // Wait for the direct staging fill, commit the frame to release the present thread, then
        // write the camera signal for the next render; one task keeps these in order.
        convert->wait();
        _commitFrame();
        _writeCameraSignal();
        _writeRaySignal();
    });
}

void RS_WorldSessionComponentVR::onDetach() { _xr.reset(); }

void RS_WorldSessionComponentVR::_syncVRCharacterization() {
    bool is_running = _xr->isRunning();
    bool is_resident = _access->handle()->findEntityGID(_name) != RS_NULL_INDEX;
    switch (_vr_state.load()) {
        case VRState::IDLE:
            if (is_running && is_resident) {
                _vr_state.store(VRState::ENGAGED);
            } else if (is_running) {
                _eye_width = _xr->eyeWidth();
                _eye_height = _xr->eyeHeight();
                for (RIndex eye = 0; eye < 2; ++eye) {
                    _eye_pose[eye] = {RVec3rf(0, 0, 0), RQuat3rf(0, 0, 0, 1)};
                }
                _access->handle()->addEntity(_makeCamera());
                _access->handle()->addController(_makeController(_vr_controller_name));
                for (RIndex hand = 0; hand < 2; ++hand) {
                    _access->handle()->addController(_makeController(_ray_controller_name[hand]));
                    _access->handle()->addEntity(
                        _makeRaySensor(_ray_sensor_name[hand], _ray_controller_name[hand])
                    );
                }
                _vr_state.store(VRState::ENGAGING);
            }
            break;
        case VRState::ENGAGING:
            // Wait for the async add to land before rendering, then it is safe to draw and remove.
            if (is_resident) _vr_state.store(VRState::ENGAGED);
            break;
        case VRState::ENGAGED:
            if (!is_running) {
                _access->handle()->removeController(_vr_controller_name);
                _access->handle()->removeEntity(_name);
                for (RIndex hand = 0; hand < 2; ++hand) {
                    _access->handle()->removeEntity(_ray_sensor_name[hand]);
                    _access->handle()->removeController(_ray_controller_name[hand]);
                }
                _vr_state.store(VRState::DISENGAGING);
            } else if (!is_resident) {
                _vr_state.store(VRState::IDLE);
            }
            break;
        case VRState::DISENGAGING:
            // Wait until the entity is truly gone before returning to idle, so no in-flight frame
            // job runs against a half-removed characterization.
            if (!is_resident) _vr_state.store(VRState::IDLE);
            break;
    }
}

void RS_WorldSessionComponentVR::_syncManipulationCharacterization(RIndex hand) {
    auto intents = _xr->getHandIntents();
    const RS_VRImpl::HandIntent& intent = hand == 0 ? intents.first : intents.second;
    bool is_trigger = intent.is_grab || intent.is_push;
    auto handle = _access->handle();
    const std::string& controller_name = _manipulation_controller_name[hand];

    switch (_manip_state[hand]) {
        case ManipulationState::IDLE: {
            if (!is_trigger || !intent.is_aim_valid) break;
            RIndex gid = RS_NULL_INDEX;
            RIndex body_sid = RS_NULL_INDEX;
            RIndex segment_bid = RS_NULL_INDEX;
            RFloat distance = 0;
            if (!_readRayHit(hand, gid, distance, body_sid, segment_bid)) break;
            std::string name = handle->findEntityName(gid);
            if (name.empty() || handle->findEntityGID(name) == RS_NULL_INDEX) break;
            _target_gid[hand] = gid;
            _target_body_sid[hand] = body_sid;
            _target_segment_bid[hand] = segment_bid;
            _target_structure[hand] = name;
            _hand_prev_position[hand] = intent.position;
            _hand_prev_orientation[hand] = intent.orientation;

            rsc::RSC_Transform transform;
            transform.name = controller_name + "#transform";
            transform.source_controller_reference = controller_name;
            transform.structure_name = name;
            rsc::RSC_Force force;
            force.name = controller_name + "#force";
            force.source_controller_reference = controller_name;
            force.structure_name = name;
            force.voxel_range = RBound3ri::highest();
            force.body_sid = body_sid;
            force.segment_bid = segment_bid;

            // One transaction so the controller and both signals land together; hold the add tasks.
            _pending_tasks[hand].clear();
            handle->beginModificationTransaction();
            _pending_tasks[hand].push_back(handle->addController(_makeController(controller_name)));
            _pending_tasks[hand].push_back(handle->addSignal(rsc::RSC_SignalVariant(transform)));
            _pending_tasks[hand].push_back(handle->addSignal(rsc::RSC_SignalVariant(force)));
            handle->endModificationTransaction();
            _manip_state[hand] = ManipulationState::ENGAGING;
            break;
        }
        case ManipulationState::ENGAGING: {
            // Poll the add tasks non-blocking: any failure tears the partial add down; once all
            // succeed the controller and signals are resident for the next step's snapshot.
            bool is_any_failed = false;
            bool is_all_finished = true;
            for (const RS_WorldTask& task : _pending_tasks[hand]) {
                RSE_WorldTaskState state = task.state();
                if (state == RSE_WORLD_TASK_FAIL) is_any_failed = true;
                if (state == RSE_WORLD_TASK_PENDING) is_all_finished = false;
            }
            if (is_any_failed) {
                _manip_state[hand] = ManipulationState::DISENGAGING;
            } else if (is_all_finished) {
                _pending_tasks[hand].clear();
                _manip_state[hand] = is_trigger ? ManipulationState::ENGAGED
                                                : ManipulationState::DISENGAGING;
            }
            break;
        }
        case ManipulationState::ENGAGED: {
            if (is_trigger && handle->findEntityGID(_target_structure[hand]) != RS_NULL_INDEX) {
                // Freeze while grabbing so the transform snaps the body without gravity sag; a push
                // keeps physics live. Freezing a foreign structure needs the full scope, so without
                // it the grab still drags the body and only the anti-sag freeze is skipped.
                bool want_frozen = intent.is_grab &&
                                   handle->scope() == RSE_WORLD_SESSION_SCOPE_FULL;
                if (want_frozen && !_is_target_frozen[hand]) {
                    handle->freezeEntity(_target_structure[hand]);
                    _is_target_frozen[hand] = true;
                } else if (!want_frozen && _is_target_frozen[hand]) {
                    handle->unfreezeEntity(_target_structure[hand]);
                    _is_target_frozen[hand] = false;
                }
                _writeManipulationSignal(hand);
            } else {
                if (_is_target_frozen[hand]) {
                    handle->unfreezeEntity(_target_structure[hand]);
                    _is_target_frozen[hand] = false;
                }
                _manip_state[hand] = ManipulationState::DISENGAGING;
            }
            break;
        }
        case ManipulationState::DISENGAGING: {
            if (_pending_tasks[hand].empty()) {
                // Remove both signals and the controller together; a leaked force would keep
                // pushing.
                handle->beginModificationTransaction();
                _pending_tasks[hand].push_back(
                    handle->removeSignal(controller_name + "#transform")
                );
                _pending_tasks[hand].push_back(handle->removeSignal(controller_name + "#force"));
                _pending_tasks[hand].push_back(handle->removeController(controller_name));
                handle->endModificationTransaction();
                _target_gid[hand] = RS_NULL_INDEX;
                _target_structure[hand].clear();
            }
            bool is_all_finished = true;
            for (const RS_WorldTask& task : _pending_tasks[hand]) {
                if (task.state() == RSE_WORLD_TASK_PENDING) is_all_finished = false;
            }
            if (is_all_finished) {
                _pending_tasks[hand].clear();
                _manip_state[hand] = ManipulationState::IDLE;
            }
            break;
        }
    }
}

void RS_WorldSessionComponentVR::_convertAndWriteFrame(
    RIndex eye,
    RIndex pixel_start,
    RIndex pixel_end
) {
    RColor4b* dst_rgba = _xr->getColorStagingBuffer(eye);
    float* dst_depth = _xr->getDepthStagingBuffer(eye);
    for (auto& controller_ref : _access->controllers()) {
        RS_WorldController& controller = controller_ref.get();
        if (controller.name != _vr_controller_name) continue;
        for (auto& record_variant : controller.records) {
            record_variant.visit([&](auto& record) {
                using Record = std::decay_t<decltype(record)>;
                if constexpr (std::is_same_v<Record, RS_WorldCameraStereoRecord>) {
                    if (!record.isValid() || record.width != _eye_width ||
                        record.height != _eye_height)
                        return;
                    const RColor3b* src_rgb = eye == 0 ? record.left_rgb.pixel
                                                       : record.right_rgb.pixel;
                    const float* src_depth = eye == 0 ? record.left_depth.pixel
                                                      : record.right_depth.pixel;
                    const RIndex* src_gid =
                        (eye == 0 ? record.left_entity_info : record.right_entity_info).entity_gid;
                    // Blend a fixed amber tint over the pixels of either hand's highlighted
                    // structure, resolved once in onCallback.
                    bool is_tint_active =
                        src_gid != nullptr &&
                        (_highlight_gid[0] != RS_NULL_INDEX || _highlight_gid[1] != RS_NULL_INDEX);
                    for (RIndex pixel = pixel_start; pixel < pixel_end; ++pixel) {
                        RColor4b color{src_rgb[pixel].r, src_rgb[pixel].g, src_rgb[pixel].b, 255};
                        if (is_tint_active && (src_gid[pixel] == _highlight_gid[0] ||
                                               src_gid[pixel] == _highlight_gid[1])) {
                            color.r = static_cast<RByte>((static_cast<int>(color.r) + 255) / 2);
                            color.g = static_cast<RByte>((static_cast<int>(color.g) + 170) / 2);
                            color.b = static_cast<RByte>((static_cast<int>(color.b) + 40) / 2);
                        }
                        dst_rgba[pixel] = color;
                    }
                    if (src_depth && dst_depth) {
                        std::memcpy(
                            dst_depth + pixel_start,
                            src_depth + pixel_start,
                            static_cast<RBigSize>(pixel_end - pixel_start) * sizeof(float)
                        );
                    }
                }
            });
        }
    }
}

void RS_WorldSessionComponentVR::_commitFrame() {
    _xr->commitFrame(
        {_eye_pose[0].position, _eye_pose[0].orientation},
        {_eye_pose[1].position, _eye_pose[1].orientation},
        _eye_width,
        _eye_height
    );
}

void RS_WorldSessionComponentVR::_writeCameraSignal() {
    auto eyes = _xr->getEyes();
    auto& left = eyes.first;
    auto& right = eyes.second;
    // Keep the physical pose behind this render instruction so it can label the frame it produces.
    _eye_pose[0] = {left.raw_position, left.raw_orientation};
    _eye_pose[1] = {right.raw_position, right.raw_orientation};
    for (auto& controller_ref : _access->controllers()) {
        RS_WorldController& controller = controller_ref.get();
        if (controller.name != _vr_controller_name) continue;
        for (auto& signal_variant : controller.signals) {
            signal_variant.visit([&](auto& signal) {
                using Signal = std::decay_t<decltype(signal)>;
                if constexpr (std::is_same_v<Signal, RS_WorldCameraStereoSignal>) {
                    if (signal.left_position_signals.object_num > 0 &&
                        signal.left_position_signals.position) {
                        signal.left_position_signals.position[0] = left.position;
                    }
                    if (signal.right_position_signals.object_num > 0 &&
                        signal.right_position_signals.position) {
                        signal.right_position_signals.position[0] = right.position;
                    }
                    if (signal.left_orientation_signals.object_num > 0 &&
                        signal.left_orientation_signals.orientation) {
                        signal.left_orientation_signals.orientation[0] = left.orientation;
                    }
                    if (signal.right_orientation_signals.object_num > 0 &&
                        signal.right_orientation_signals.orientation) {
                        signal.right_orientation_signals.orientation[0] = right.orientation;
                    }
                    if (signal.left_optics_signals.object_num >= 4 &&
                        signal.left_optics_signals.field_of_view) {
                        signal.left_optics_signals.field_of_view[0] = left.angle_left;
                        signal.left_optics_signals.field_of_view[1] = left.angle_right;
                        signal.left_optics_signals.field_of_view[2] = left.angle_up;
                        signal.left_optics_signals.field_of_view[3] = left.angle_down;
                    }
                    if (signal.right_optics_signals.object_num >= 4 &&
                        signal.right_optics_signals.field_of_view) {
                        signal.right_optics_signals.field_of_view[0] = right.angle_left;
                        signal.right_optics_signals.field_of_view[1] = right.angle_right;
                        signal.right_optics_signals.field_of_view[2] = right.angle_up;
                        signal.right_optics_signals.field_of_view[3] = right.angle_down;
                    }
                }
            });
        }
    }
}

void RS_WorldSessionComponentVR::_writeRaySignal() {
    auto intents = _xr->getHandIntents();
    const RS_VRImpl::HandIntent* hands[2] = {&intents.first, &intents.second};
    // Note: for each hand their ray sensor casts along its local +x.
    for (auto& controller_ref : _access->controllers()) {
        RS_WorldController& controller = controller_ref.get();
        RIndex hand = controller.name == _ray_controller_name[0]   ? 0
                      : controller.name == _ray_controller_name[1] ? 1
                                                                   : RS_NULL_INDEX;
        if (hand == RS_NULL_INDEX) continue;
        const RS_VRImpl::HandIntent& intent = *hands[hand];
        for (auto& signal_variant : controller.signals) {
            signal_variant.visit([&](auto& signal) {
                using Signal = std::decay_t<decltype(signal)>;
                if constexpr (std::is_same_v<Signal, RS_WorldRaySensorSignal>) {
                    if (signal.position_signals.object_num > 0 &&
                        signal.position_signals.position) {
                        signal.position_signals.position[0] = intent.aim_origin;
                    }
                    if (signal.orientation_signals.object_num > 0 &&
                        signal.orientation_signals.orientation) {
                        signal.orientation_signals.orientation[0] = intent.aim_orientation;
                    }
                }
            });
        }
    }
}

void RS_WorldSessionComponentVR::_writeManipulationSignal(RIndex hand) {
    auto intents = _xr->getHandIntents();
    const RS_VRImpl::HandIntent& intent = hand == 0 ? intents.first : intents.second;
    // Grab carries the body by the incremental hand delta about the previous grip pose; an identity
    // transform when not grabbing maps every point to itself, so physics keeps running.
    RQuat3rf rotation = intent.is_grab
                            ? intent.orientation * _hand_prev_orientation[hand].conjugate()
                            : RQuat3rf::unit();
    RVec3rf translation = intent.is_grab ? intent.position - _hand_prev_position[hand]
                                         : RVec3rf(0, 0, 0);
    RVec3rf pivot = _hand_prev_position[hand];
    RVec3rf push = intent.is_push ? intent.aim_orientation.toOrthoAxisX() *
                                        (intent.push_amount * _push_force.load())
                                  : RVec3rf(0, 0, 0);

    for (auto& controller_ref : _access->controllers()) {
        RS_WorldController& controller = controller_ref.get();
        if (controller.name != _manipulation_controller_name[hand]) continue;
        for (auto& signal_variant : controller.signals) {
            signal_variant.visit([&](auto& signal) {
                using Signal = std::decay_t<decltype(signal)>;
                if constexpr (std::is_same_v<Signal, RS_WorldTransformSignal>) {
                    if (signal.rotation_signals.object_num > 0 &&
                        signal.rotation_signals.rotation) {
                        signal.rotation_signals.rotation[0] = rotation;
                        signal.rotation_signals.pivot[0] = pivot;
                    }
                    if (signal.translation_signals.object_num > 0 &&
                        signal.translation_signals.translation) {
                        signal.translation_signals.translation[0] = translation;
                    }
                } else if constexpr (std::is_same_v<Signal, RS_WorldForceSignal>) {
                    if (signal.force_signals.object_num > 0 && signal.force_signals.force) {
                        signal.force_signals.force[0] = push;
                    }
                }
            });
        }
    }

    _hand_prev_position[hand] = intent.position;
    _hand_prev_orientation[hand] = intent.orientation;
}

void RS_WorldSessionComponentVR::_updateRay() {
    if (_vr_state.load() != VRState::ENGAGED) {
        _xr->setRay(true, false, 0, false, RS_VRImpl::RayMode::NEUTRAL);
        _xr->setRay(false, false, 0, false, RS_VRImpl::RayMode::NEUTRAL);
        return;
    }
    auto intents = _xr->getHandIntents();
    const RS_VRImpl::HandIntent* hands[2] = {&intents.first, &intents.second};
    for (RIndex hand = 0; hand < 2; ++hand) {
        RIndex gid = RS_NULL_INDEX;
        RIndex body_sid = RS_NULL_INDEX;
        RIndex segment_bid = RS_NULL_INDEX;
        RFloat distance = static_cast<RFloat>(_far_depth);
        bool is_hit = _readRayHit(hand, gid, distance, body_sid, segment_bid);
        RS_VRImpl::RayMode mode = hands[hand]->is_grab   ? RS_VRImpl::RayMode::GRAB
                                  : hands[hand]->is_push ? RS_VRImpl::RayMode::PUSH
                                                         : RS_VRImpl::RayMode::NEUTRAL;
        _xr->setRay(hand == 0, true, distance, is_hit, mode);
    }
}

void RS_WorldSessionComponentVR::_updateInfo(RBigFloat time) {
    if (!_xr) return;
    auto handle = _access->handle();
    auto bytes = [](const std::string& text) -> RBigSize {
        return text.empty() ? 0 : static_cast<RBigSize>(std::strtoull(text.c_str(), nullptr, 10));
    };
    std::string world_load = handle->getInfo("world_load");
    RS_VRImpl::HudInfo info;
    info.sim_time = static_cast<RFloat>(time);
    info.is_world_load_valid = !world_load.empty();
    info.world_load = info.is_world_load_valid
                          ? static_cast<RFloat>(std::strtod(world_load.c_str(), nullptr))
                          : 0;
    info.device_used = bytes(handle->getInfo("device_memory_used"));
    info.device_total = bytes(handle->getInfo("device_memory_capacity"));
    info.host_used = bytes(handle->getInfo("host_memory_used"));
    info.host_total = bytes(handle->getInfo("host_memory_capacity"));
    info.device_name = handle->getInfo("device_name");
    _xr->setHudInfo(info);
}

void RS_WorldSessionComponentVR::_publishUserState() {
    auto [left, right] = _xr->getEyes();
    RS_WorldSessionVRUserPosition user_position;
    user_position.position = (left.position + right.position) * 0.5;
    _access->publish<RS_WorldSessionVRUserPosition>(
        RS_VR_CHANNEL_USER_POSITION,
        std::make_shared<const RS_WorldSessionVRUserPosition>(user_position)
    );

    auto intents = _xr->getHandIntents();
    const RS_VRImpl::HandIntent* hands[2] = {&intents.first, &intents.second};
    RS_WorldSessionVRUserHandAims aims;
    RS_WorldSessionVRHandAim* out[2] = {&aims.left, &aims.right};
    for (RIndex hand = 0; hand < 2; ++hand) {
        RIndex gid = RS_NULL_INDEX;
        RIndex body_sid = RS_NULL_INDEX;
        RIndex segment_bid = RS_NULL_INDEX;
        RFloat distance = static_cast<RFloat>(_far_depth);
        static_cast<void>(_readRayHit(hand, gid, distance, body_sid, segment_bid));
        RVec3rf direction = hands[hand]->aim_orientation.toOrthoAxisX();
        out[hand]->is_valid = hands[hand]->is_aim_valid;
        out[hand]->aim_origin = hands[hand]->aim_origin;
        out[hand]->aim_direction = direction;
        out[hand]->aim_position = hands[hand]->aim_origin + direction * distance;
    }
    _access->publish<RS_WorldSessionVRUserHandAims>(
        RS_VR_CHANNEL_USER_HAND_AIMS, std::make_shared<const RS_WorldSessionVRUserHandAims>(aims)
    );
}

bool RS_WorldSessionComponentVR::_readRayHit(
    RIndex hand,
    RIndex& gid,
    RFloat& distance,
    RIndex& body_sid,
    RIndex& segment_bid
) {
    gid = RS_NULL_INDEX;
    distance = static_cast<RFloat>(_far_depth);
    body_sid = RS_NULL_INDEX;
    segment_bid = RS_NULL_INDEX;
    bool is_hit = false;
    for (auto& controller_ref : _access->controllers()) {
        RS_WorldController& controller = controller_ref.get();
        if (controller.name != _ray_controller_name[hand]) continue;
        for (auto& record_variant : controller.records) {
            record_variant.visit([&](auto& record) {
                using Record = std::decay_t<decltype(record)>;
                if constexpr (std::is_same_v<Record, RS_WorldRaySensorRecord>) {
                    if (!record.isValid() || record.ray_num < 1 || !record.hit.is_hit ||
                        !record.hit.is_hit[0])
                        return;
                    // The beam ends at any surface, so read the distance whatever it hit; the hit
                    // flag and target ids only count a structure, since that is what we manipulate.
                    if (record.hit.distance) distance = record.hit.distance[0];
                    if (!record.entity_info.entity_type ||
                        record.entity_info.entity_type[0] != RSE_WORLD_ENTITY_TYPE_STRUCTURE_VOXEL)
                        return;
                    is_hit = true;
                    if (record.entity_info.entity_gid) gid = record.entity_info.entity_gid[0];
                    if (record.entity_detail.body_sid) body_sid = record.entity_detail.body_sid[0];
                    if (record.entity_detail.segment_bid)
                        segment_bid = record.entity_detail.segment_bid[0];
                }
            });
        }
    }
    return is_hit;
}

rsc::RSC_EntityVariant RS_WorldSessionComponentVR::_makeCamera() const {
    auto [left, right] = _xr->getEyes();
    RFloat ipd = (left.position - right.position).length();

    rsc::RSC_CameraStereoFree camera;
    camera.name = _name;
    camera.image_width = _eye_width;
    camera.image_height = _eye_height;
    camera.samples_per_pixel = _samples_per_pixel;
    camera.max_depth = _max_depth;
    camera.interpupillary_distance = ipd > 0.0 ? ipd : 0.063;
    camera.convergence_distance = 0.0;
    camera.record.has_rgb = true;
    camera.record.has_depth = true;
    camera.record.has_entity_info = true;
    camera.record.is_depth_hyperbolic = true;
    camera.record.hyperbolic_near_depth = _near_depth;
    camera.record.hyperbolic_far_depth = _far_depth;
    camera.record.target_controller_reference = _vr_controller_name;
    camera.signal.source_controller_reference = _vr_controller_name;
    return rsc::RSC_EntityVariant(camera);
}

rsc::RSC_EntityVariant
RS_WorldSessionComponentVR::_makeRaySensor(std::string name, std::string controller_name) const {
    rsc::RSC_RaySensorFree sensor;
    sensor.name = std::move(name);
    // One ray along local +x; the signal drives the sensor orientation so it points along the aim.
    sensor.ray_directions = {RVec3rf(1, 0, 0)};
    sensor.max_range = _far_depth;
    sensor.record.has_hit = true;
    sensor.record.has_entity_info = true;
    sensor.record.has_entity_detail = true;
    sensor.record.target_controller_reference = controller_name;
    sensor.signal.source_controller_reference = controller_name;
    return rsc::RSC_EntityVariant(sensor);
}

rsc::RSC_ControllerVariant RS_WorldSessionComponentVR::_makeController(std::string name) const {
    rsc::RSC_ControllerCallback controller;
    controller.name = std::move(name);
    return rsc::RSC_ControllerVariant(controller);
}

}  // namespace rise

#endif  // RISE_ORIGIN_BUILD_CORE_WITH_VR