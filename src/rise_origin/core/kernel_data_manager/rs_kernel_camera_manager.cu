// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/logic/vision/rs_camera_c.h"
#include "rs_kernel_camera_manager.h"

using namespace std;
using namespace rise::logic;
using namespace rise::device;

namespace rise {

void RS_KernelCameraManager::init(RS_KernelManager* kernel_manager, const RS_CameraConfig& config) {
    if (_camera_gid != RS_NULL_INDEX)
        RS_THROW(RS_RuntimeException("Camera manager cannot be re-initialized twice"));

    _kernel_manager = kernel_manager;
    _config = config;
    _camera_gid = kernel_manager->getCameraNameToGid().at(config.name);

    auto memory_provider = _kernel_manager->getDeviceProvider();
    auto buffer_frame_num = _kernel_manager->getKernelRecordBufferFrameNum();

    RSize pixel_num = 0;
    if (_config.type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE) {
        auto camera_config = _config.config.as<RS_RigidBodyLinkedPerspectiveCameraConfig>();
        auto it = _kernel_manager->getStructureNameToGid().find(camera_config.structure_name);
        if (it == _kernel_manager->getStructureNameToGid().end())
            RS_THROW(RS_InvalidConfigurationException(
                "Structure {} is not registered in the kernel manager", camera_config.structure_name
            ));

        _structure_gid = it->second;
        auto& structure_manager = _kernel_manager->getKernelStructureManagers().at(_structure_gid);

        RIndex rigid_body_sid =
            structure_manager.findRigidBodySID(camera_config.body_sid, camera_config.segment_bid);
        if (rigid_body_sid == RS_NULL_INDEX)
            RS_THROW(RS_InvalidConfigurationException(
                "Rigid body in body {} segment {} of structure {} does not exist",
                camera_config.body_sid,
                camera_config.segment_bid,
                camera_config.structure_name
            ));

        _rigid_body_sid = rigid_body_sid;

        _camera_rb_position = structure_manager.getRigidBodyPositionFromRelativeAnchor(
            camera_config.body_sid, rigid_body_sid, camera_config.relative_anchor
        );
        auto& body_config = structure_manager.getConfig().bodies[camera_config.body_sid];
        _camera_rb_orientation =
            body_config.relative_orientation.inverse() * camera_config.relative_orientation;
        pixel_num = camera_config.image_width * camera_config.image_height;
        _width = camera_config.image_width;
        _height = camera_config.image_height;
        _camera_current_rgb_canvas.init(
            {camera_config.image_height, camera_config.image_width}, memory_provider
        );
        _camera_next_rgb_canvas.init(
            {camera_config.image_height, camera_config.image_width}, memory_provider
        );
    } else {
        auto camera_config = _config.config.as<RS_FreePerspectiveCameraConfig>();
        _camera_origin_position = camera_config.origin_position;
        _camera_orientation = camera_config.orientation;
        pixel_num = camera_config.image_width * camera_config.image_height;
        _width = camera_config.image_width;
        _height = camera_config.image_height;
        _camera_current_rgb_canvas.init(
            {camera_config.image_height, camera_config.image_width}, memory_provider
        );
        _camera_next_rgb_canvas.init(
            {camera_config.image_height, camera_config.image_width}, memory_provider
        );
    }

    // Initialize RGB canvas
    _camera_current_rgb_canvas.fill(RVec3f::zero(), false);
    _camera_next_rgb_canvas.fill(RVec3f::zero(), false);

    // Shared primary AOVs.
    _camera_aux_normal.init({_height, _width}, memory_provider);
    _camera_aux_depth_z.init({_height, _width}, memory_provider);
    _camera_aux_hit_position.init({_height, _width}, memory_provider);
    _camera_aux_object_key.init({_height, _width}, memory_provider);
    _camera_aux_roughness.init({_height, _width}, memory_provider);
    _camera_aux_screen_motion.init({_height, _width}, memory_provider);
    // Restir-only endpoint + chain AOVs.
    _camera_aux_endpoint_hit_position.init({_height, _width}, memory_provider);
    _camera_aux_endpoint_normal.init({_height, _width}, memory_provider);
    _camera_aux_endpoint_object_key.init({_height, _width}, memory_provider);
    _camera_aux_endpoint_albedo.init({_height, _width}, memory_provider);
    _camera_aux_endpoint_roughness.init({_height, _width}, memory_provider);
    _camera_aux_endpoint_metalness.init({_height, _width}, memory_provider);
    _camera_aux_endpoint_intensity.init({_height, _width}, memory_provider);
    _camera_aux_endpoint_material_type.init({_height, _width}, memory_provider);
    _camera_aux_endpoint_screen_motion.init({_height, _width}, memory_provider);
    _camera_aux_chain_throughput.init({_height, _width}, memory_provider);
    _camera_aux_chain_terminal_kind.init({_height, _width}, memory_provider);
    _camera_prev_reservoir_di.init({_height, _width}, memory_provider);
    _camera_current_reservoir_di.init({_height, _width}, memory_provider);
    _camera_prev_reservoir_gi.init({_height, _width}, memory_provider);
    _camera_current_reservoir_gi.init({_height, _width}, memory_provider);
    _camera_spatial_reservoir_di.init({_height, _width}, memory_provider);
    _camera_spatial_reservoir_gi.init({_height, _width}, memory_provider);
    _camera_prev_aux_normal.init({_height, _width}, memory_provider);
    _camera_prev_aux_depth_z.init({_height, _width}, memory_provider);
    _camera_prev_aux_object_key.init({_height, _width}, memory_provider);
    _camera_denoiser_color_history.init({_height, _width}, memory_provider);
    _camera_denoiser_moment_history.init({_height, _width}, memory_provider);
    _camera_denoiser_history_length.init({_height, _width}, memory_provider);
    _camera_denoiser_variance.init({_height, _width}, memory_provider);
    _camera_prev_denoiser_color_history.init({_height, _width}, memory_provider);
    _camera_prev_denoiser_moment_history.init({_height, _width}, memory_provider);
    _camera_prev_denoiser_history_length.init({_height, _width}, memory_provider);

    _camera_aux_normal.fill(RVec3f::zero(), false);
    _camera_aux_depth_z.fill(RFloat(0), false);
    _camera_aux_hit_position.fill(RVec3f::zero(), false);
    _camera_aux_object_key.fill(logic::RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL, false);
    _camera_aux_roughness.fill(RFloat(1), false);
    _camera_aux_screen_motion.fill(RVec2f::zero(), false);
    _camera_aux_endpoint_hit_position.fill(RVec3f::zero(), false);
    _camera_aux_endpoint_normal.fill(RVec3f::zero(), false);
    _camera_aux_endpoint_object_key.fill(logic::RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL, false);
    _camera_aux_endpoint_albedo.fill(RVec3f::zero(), false);
    _camera_aux_endpoint_roughness.fill(RFloat(1), false);
    _camera_aux_endpoint_metalness.fill(RFloat(0), false);
    _camera_aux_endpoint_intensity.fill(RFloat(0), false);
    _camera_aux_endpoint_material_type.fill(RSE_VISION_MATERIAL_OPAQUE, false);
    _camera_aux_endpoint_screen_motion.fill(RVec2f::zero(), false);
    _camera_aux_chain_throughput.fill(RVec3f::one(), false);
    _camera_aux_chain_terminal_kind.fill(RSE_VISION_RESTIR_CHAIN_TERMINAL_SURFACE, false);
    _camera_prev_reservoir_di.fill(logic::RS_ReservoirDI{}, false);
    _camera_current_reservoir_di.fill(logic::RS_ReservoirDI{}, false);
    _camera_prev_reservoir_gi.fill(logic::RS_ReservoirGI{}, false);
    _camera_current_reservoir_gi.fill(logic::RS_ReservoirGI{}, false);
    _camera_spatial_reservoir_di.fill(logic::RS_ReservoirDI{}, false);
    _camera_spatial_reservoir_gi.fill(logic::RS_ReservoirGI{}, false);
    _camera_prev_aux_normal.fill(RVec3f::zero(), false);
    _camera_prev_aux_depth_z.fill(float(0), false);
    _camera_prev_aux_object_key.fill(logic::RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL, false);
    _camera_denoiser_color_history.fill(RVec3f::zero(), false);
    _camera_denoiser_moment_history.fill(RVec2f::zero(), false);
    _camera_denoiser_history_length.fill(RFloat(0), false);
    _camera_denoiser_variance.fill(RFloat(0), false);
    _camera_prev_denoiser_color_history.fill(RVec3f::zero(), false);
    _camera_prev_denoiser_moment_history.fill(RVec2f::zero(), false);
    _camera_prev_denoiser_history_length.fill(RFloat(0), false);

    // create record buffer
    RSize frame_size = RS_SimulationCameraRecord::getRequiredMemorySize(pixel_num);
    _recorder_buffer.init(buffer_frame_num, buffer_frame_num * frame_size, memory_provider);

    // Note: for rigid body linked perspective camera, it is special, it needs to have a controller
    // record buffer, but it does not have a controller or a controller signal buffer
    _controller_buffer.init(frame_size, memory_provider);

    // create signal buffer
    if (hasControllerSignal()) {
        // Note: user interface buffers, use RFloat accuracy
        _host_camera_control_signal_buffer =
            memory_provider->allocateHost(sizeof(RFloat) * 9, true);
        _host_camera_control_signal_buffer.set(0);
    }

    _initKernelData();
}

void RS_KernelCameraManager::free() {
    _freeKernelData();

    _recorder_buffer.free();
    _controller_buffer.free();
    _camera_current_rgb_canvas.free();
    _camera_next_rgb_canvas.free();
    _camera_aux_normal.free();
    _camera_aux_depth_z.free();
    _camera_aux_hit_position.free();
    _camera_aux_object_key.free();
    _camera_aux_roughness.free();
    _camera_aux_screen_motion.free();
    _camera_aux_endpoint_hit_position.free();
    _camera_aux_endpoint_normal.free();
    _camera_aux_endpoint_object_key.free();
    _camera_aux_endpoint_albedo.free();
    _camera_aux_endpoint_roughness.free();
    _camera_aux_endpoint_metalness.free();
    _camera_aux_endpoint_intensity.free();
    _camera_aux_endpoint_material_type.free();
    _camera_aux_endpoint_screen_motion.free();
    _camera_aux_chain_throughput.free();
    _camera_aux_chain_terminal_kind.free();
    _camera_prev_reservoir_di.free();
    _camera_current_reservoir_di.free();
    _camera_prev_reservoir_gi.free();
    _camera_current_reservoir_gi.free();
    _camera_spatial_reservoir_di.free();
    _camera_spatial_reservoir_gi.free();
    _camera_prev_aux_normal.free();
    _camera_prev_aux_depth_z.free();
    _camera_prev_aux_object_key.free();
    _camera_denoiser_color_history.free();
    _camera_denoiser_moment_history.free();
    _camera_denoiser_history_length.free();
    _camera_denoiser_variance.free();
    _camera_prev_denoiser_color_history.free();
    _camera_prev_denoiser_moment_history.free();
    _camera_prev_denoiser_history_length.free();
}

bool RS_KernelCameraManager::hasRecorderRecords() {
    return _config.type == RSE_CAMERA_FREE_PERSPECTIVE ||
           _config.type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE;
}

void RS_KernelCameraManager::syncRecorderRecords() {
    _recorder_buffer.swapAndCopy(true, false, false);
}

void RS_KernelCameraManager::waitForRecorderRecordsSync() { _recorder_buffer.waitForCopyFinish(); }

std::vector<RS_SimulationCameraRecord> RS_KernelCameraManager::getRecorderRecords() {
    vector<RS_SimulationCameraRecord> records;
    for (RIndex i = 0, total = _recorder_buffer.getHostNum(); i < total; ++i) {
        RS_SimulationCameraRecord record;
        record.initFromMemory(_recorder_buffer.getHostChunk(i).ptr(), _width * _height);
        record.camera_gid = _camera_gid;
        record.structure_gid = _structure_gid;
        record.rigid_body_sid = _rigid_body_sid;
        record.width = _width;
        record.height = _height;
        record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
        records.emplace_back(record);
    }
    return records;
}

bool RS_KernelCameraManager::hasControllerSignal() {
    return _config.type == RSE_CAMERA_FREE_PERSPECTIVE;
}

void RS_KernelCameraManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
            // no need to lock kernel as we don't modify kernel data, just pointer contents
            auto& ctx = k_ptr->ctx;

            RS_MemoryPtr device_camera_control_signals =
                ctx.camera_control_signals.data +
                sizeof(RFloat) * _device_camera_control_signal_offset;

            device_camera_control_signals.copyFrom(
                _host_camera_control_signal_buffer, _host_camera_control_signal_buffer.size()
            );
        });
    }
}

RS_SimulationCameraSignal RS_KernelCameraManager::getControllerSignal() {
    if (hasControllerSignal()) {
        return RS_SimulationCameraSignal(
            _host_camera_control_signal_buffer.device(),
            _camera_gid,
            _host_camera_control_signal_buffer.as<RFloat>(),      // position_signals
            _host_camera_control_signal_buffer.as<RFloat>() + 3,  // orientation_signals
            _host_camera_control_signal_buffer.as<RFloat>() + 7,  // focus_distance_signals
            _host_camera_control_signal_buffer.as<RFloat>() + 8,  // field_of_view_signals
            3,                                                    // position_num
            4,                                                    // orientation_num
            1,                                                    // focus_distance_num
            1                                                     // field_of_view_num
        );
    }
    return {};
}

bool RS_KernelCameraManager::hasControllerRecord() {
    return _config.type == RSE_CAMERA_FREE_PERSPECTIVE ||
           _config.type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE;
}

void RS_KernelCameraManager::syncControllerRecord() {
    if (hasControllerRecord()) {
        _controller_buffer.sync(true);
    }
}

RS_SimulationCameraRecord RS_KernelCameraManager::getControllerRecord() {
    RS_SimulationCameraRecord record;
    record.initFromMemory(_controller_buffer.getHostChunk(0).ptr(), _width * _height);
    record.camera_gid = _camera_gid;
    record.structure_gid = _structure_gid;
    record.rigid_body_sid = _rigid_body_sid;
    record.width = _width;
    record.height = _height;
    record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
    return record;
}

std::string RS_KernelCameraManager::getName() const { return _config.name; }

std::string RS_KernelCameraManager::getStructureName() const {
    if (_config.type != RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE) return "";
    auto& camera_config = _config.config.as<RS_RigidBodyLinkedPerspectiveCameraConfig>();
    return camera_config.structure_name;
}

RIndex RS_KernelCameraManager::getStructureGid() const {
    if (_config.type != RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE) return RS_NULL_INDEX;
    return _structure_gid;
}

const RS_CameraConfig& RS_KernelCameraManager::getConfig() const { return _config; }

void RS_KernelCameraManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        lock_guard record_lock(k_ptr->getSyncMutex());

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Context data is modified, needs to update kernel
        k_ptr->markModified();

        auto& ctx = k_ptr->ctx;
        using ContextType = decltype(k_ptr->ctx);
        using CameraMemType = RS_CameraMem<ContextType::FloatMemAccuracy>;

        CameraMemType camera;
        camera.gid = _camera_gid;
        RS_CameraStateUnion state{0};
        state.camera_state.type = _config.type;

        perspective_camera_config::RS_PerspectiveCameraConfig* perspective_config = nullptr;
        if (_config.type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE) {
            auto& camera_config = _config.config.as<RS_RigidBodyLinkedPerspectiveCameraConfig>();

            camera.rigid_body_sid = _rigid_body_sid;
            camera.rigid_body_anchor_rb_position = _camera_rb_position;
            camera.rigid_body_anchor_rb_orientation = _camera_rb_orientation;
            perspective_config = &camera_config;
        } else {
            auto& camera_config = _config.config.as<RS_FreePerspectiveCameraConfig>();
            camera.position = _camera_origin_position;
            camera.orientation = _camera_orientation;
            perspective_config = &camera_config;
        }

        camera.aspect_ratio = static_cast<RFloat>(perspective_config->image_height) /
                              static_cast<RFloat>(perspective_config->image_width);
        camera.image_width = perspective_config->image_width;
        camera.image_height = perspective_config->image_height;

        camera.samples_per_pixel = RS_MAX(perspective_config->samples_per_pixel, RSize(1));
        camera.max_depth = perspective_config->max_depth;

        camera.focus_distance = perspective_config->focus_distance;
        camera.field_of_view = perspective_config->field_of_view;
        camera.defocus_angle = perspective_config->defocus_angle;
        camera.defocus_radius =
            perspective_config->focus_distance *
            tanf(static_cast<float>(perspective_config->defocus_angle * RS_PI) / 180.0f / 2.0f);
        float theta = static_cast<float>(perspective_config->field_of_view * RS_PI) / 180.0f;
        float h = tanf(theta / 2.0f);

        float viewport_height = 2.0f * h * perspective_config->focus_distance;
        float viewport_width = viewport_height / camera.aspect_ratio;

        camera.pixel_height = viewport_height / static_cast<float>(camera.image_height);
        camera.pixel_width = viewport_width / static_cast<float>(camera.image_width);

        camera.state = state.camera_state_packed;

        // Update host layout
        ctx.host_data->camera_layout.add(1, _camera_gid);
        ctx.host_data->camera_info_ray_layout.add(
            camera.image_width * camera.image_height, _camera_gid
        );
        ctx.host_data->camera_rgb_ray_layout.add(
            camera.image_width * camera.image_height * camera.samples_per_pixel, _camera_gid
        );
        ctx.host_data->camera_record_layout.add(1, _camera_gid);
        ctx.host_data->camera_pixel_record_layout.add(
            camera.image_width * camera.image_height, _camera_gid
        );
        ctx.host_data->camera_state_record_layout.add(1, _camera_gid);

        if (hasControllerSignal()) {
            ctx.host_data->camera_control_signal_layout.add(sizeof(RFloat) * 8, _camera_gid);
            _device_camera_control_signal_offset =
                ctx.host_data->camera_control_signal_layout.segmentOffset(
                    ctx.host_data->camera_control_signal_layout.find(_camera_gid)
                );
        }

        // Update device layout
        ctx.host_data->camera_layout.syncDeviceLayout(ctx.camera_layout);
        ctx.host_data->camera_info_ray_layout.syncDeviceLayout(ctx.camera_info_ray_layout);
        ctx.host_data->camera_rgb_ray_layout.syncDeviceLayout(ctx.camera_rgb_ray_layout);
        ctx.host_data->camera_record_layout.syncDeviceLayout(ctx.camera_record_layout);
        ctx.host_data->camera_pixel_record_layout.syncDeviceLayout(ctx.camera_pixel_record_layout);
        ctx.host_data->camera_state_record_layout.syncDeviceLayout(ctx.camera_state_record_layout);
        if (hasControllerSignal()) {
            ctx.host_data->camera_control_signal_layout.syncDeviceLayout(
                ctx.camera_control_signal_layout
            );
        }

        // Write data to device
        if (hasControllerSignal()) {
            camera.camera_control_signal_offset = _device_camera_control_signal_offset;
        }
        ctx.cameras.append(camera);
        ctx.camera_records.append({});
        ctx.camera_recorder_buffers.append(_recorder_buffer);
        ctx.camera_controller_buffers.append(_controller_buffer);
        ctx.camera_current_rgb_canvas.append(_camera_current_rgb_canvas);
        ctx.camera_next_rgb_canvas.append(_camera_next_rgb_canvas);
        ctx.camera_aux_normal.append(_camera_aux_normal);
        ctx.camera_aux_depth_z.append(_camera_aux_depth_z);
        ctx.camera_aux_hit_position.append(_camera_aux_hit_position);
        ctx.camera_aux_object_key.append(_camera_aux_object_key);
        ctx.camera_aux_roughness.append(_camera_aux_roughness);
        ctx.camera_aux_screen_motion.append(_camera_aux_screen_motion);
        ctx.camera_aux_endpoint_hit_position.append(_camera_aux_endpoint_hit_position);
        ctx.camera_aux_endpoint_normal.append(_camera_aux_endpoint_normal);
        ctx.camera_aux_endpoint_object_key.append(_camera_aux_endpoint_object_key);
        ctx.camera_aux_endpoint_albedo.append(_camera_aux_endpoint_albedo);
        ctx.camera_aux_endpoint_roughness.append(_camera_aux_endpoint_roughness);
        ctx.camera_aux_endpoint_metalness.append(_camera_aux_endpoint_metalness);
        ctx.camera_aux_endpoint_intensity.append(_camera_aux_endpoint_intensity);
        ctx.camera_aux_endpoint_material_type.append(_camera_aux_endpoint_material_type);
        ctx.camera_aux_endpoint_screen_motion.append(_camera_aux_endpoint_screen_motion);
        ctx.camera_aux_chain_throughput.append(_camera_aux_chain_throughput);
        ctx.camera_aux_chain_terminal_kind.append(_camera_aux_chain_terminal_kind);
        ctx.camera_prev_reservoir_di.append(_camera_prev_reservoir_di);
        ctx.camera_current_reservoir_di.append(_camera_current_reservoir_di);
        ctx.camera_prev_reservoir_gi.append(_camera_prev_reservoir_gi);
        ctx.camera_current_reservoir_gi.append(_camera_current_reservoir_gi);
        ctx.camera_spatial_reservoir_di.append(_camera_spatial_reservoir_di);
        ctx.camera_spatial_reservoir_gi.append(_camera_spatial_reservoir_gi);
        ctx.camera_prev_aux_normal.append(_camera_prev_aux_normal);
        ctx.camera_prev_aux_depth_z.append(_camera_prev_aux_depth_z);
        ctx.camera_prev_aux_object_key.append(_camera_prev_aux_object_key);
        ctx.camera_denoiser_color_history.append(_camera_denoiser_color_history);
        ctx.camera_denoiser_moment_history.append(_camera_denoiser_moment_history);
        ctx.camera_denoiser_history_length.append(_camera_denoiser_history_length);
        ctx.camera_denoiser_variance.append(_camera_denoiser_variance);
        ctx.camera_prev_denoiser_color_history.append(_camera_prev_denoiser_color_history);
        ctx.camera_prev_denoiser_moment_history.append(_camera_prev_denoiser_moment_history);
        ctx.camera_prev_denoiser_history_length.append(_camera_prev_denoiser_history_length);

        if (hasControllerSignal()) {
            // Resize device signal buffer
            ctx.camera_control_signals.resize(ctx.host_data->camera_control_signal_layout.size());

            // Clear device signal buffer
            RFloat* camera_control_signal = _host_camera_control_signal_buffer.as<RFloat>();

            // position
            camera_control_signal[0] = _camera_origin_position.x;
            camera_control_signal[1] = _camera_origin_position.y;
            camera_control_signal[2] = _camera_origin_position.z;

            // orientation
            camera_control_signal[3] = _camera_orientation.x;
            camera_control_signal[4] = _camera_orientation.y;
            camera_control_signal[5] = _camera_orientation.z;
            camera_control_signal[6] = _camera_orientation.w;

            // focus distance
            camera_control_signal[7] = perspective_config->focus_distance;

            // field of view
            camera_control_signal[8] = perspective_config->field_of_view;

            // copy to device
            RS_MemoryPtr device_camera_control_signals =
                ctx.camera_control_signals.data +
                sizeof(RFloat) * _device_camera_control_signal_offset;
            device_camera_control_signals.copyFrom(
                _host_camera_control_signal_buffer, _host_camera_control_signal_buffer.size()
            );
        }

        // Resolve data dependency
        if (_config.type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE) {
            castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
                p_ptr->template run<RSCF_ResolveRigidBodyLinkedPerspectiveCameraDependency>(
                    {1},
                    *k_ptr,
                    ctx.cameras.size() - 1,
                    ctx.host_data->structure_rigid_body_layout.segmentOffset(
                        ctx.host_data->structure_rigid_body_layout.find(_structure_gid)
                    )
                );
            });
        }

        _is_kernel_data_inited = true;
    });
}

void RS_KernelCameraManager::_freeKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        if (!_is_kernel_data_inited) return;

        lock_guard lock(k_ptr->getSyncMutex());
        auto& ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Context data is modified, needs to update kernel
        k_ptr->markModified();

        // Remove data from device
        RIndex camera_index = ctx.host_data->camera_layout.find(_camera_gid);
        ctx.cameras.erase(camera_index);
        ctx.camera_records.erase(camera_index);
        ctx.camera_recorder_buffers.erase(camera_index);
        ctx.camera_controller_buffers.erase(camera_index);
        ctx.camera_current_rgb_canvas.erase(camera_index);
        ctx.camera_next_rgb_canvas.erase(camera_index);
        ctx.camera_aux_normal.erase(camera_index);
        ctx.camera_aux_depth_z.erase(camera_index);
        ctx.camera_aux_hit_position.erase(camera_index);
        ctx.camera_aux_object_key.erase(camera_index);
        ctx.camera_aux_roughness.erase(camera_index);
        ctx.camera_aux_screen_motion.erase(camera_index);
        ctx.camera_aux_endpoint_hit_position.erase(camera_index);
        ctx.camera_aux_endpoint_normal.erase(camera_index);
        ctx.camera_aux_endpoint_object_key.erase(camera_index);
        ctx.camera_aux_endpoint_albedo.erase(camera_index);
        ctx.camera_aux_endpoint_roughness.erase(camera_index);
        ctx.camera_aux_endpoint_metalness.erase(camera_index);
        ctx.camera_aux_endpoint_intensity.erase(camera_index);
        ctx.camera_aux_endpoint_material_type.erase(camera_index);
        ctx.camera_aux_endpoint_screen_motion.erase(camera_index);
        ctx.camera_aux_chain_throughput.erase(camera_index);
        ctx.camera_aux_chain_terminal_kind.erase(camera_index);
        ctx.camera_prev_reservoir_di.erase(camera_index);
        ctx.camera_current_reservoir_di.erase(camera_index);
        ctx.camera_prev_reservoir_gi.erase(camera_index);
        ctx.camera_current_reservoir_gi.erase(camera_index);
        ctx.camera_spatial_reservoir_di.erase(camera_index);
        ctx.camera_spatial_reservoir_gi.erase(camera_index);
        ctx.camera_prev_aux_normal.erase(camera_index);
        ctx.camera_prev_aux_depth_z.erase(camera_index);
        ctx.camera_prev_aux_object_key.erase(camera_index);
        ctx.camera_denoiser_color_history.erase(camera_index);
        ctx.camera_denoiser_moment_history.erase(camera_index);
        ctx.camera_denoiser_history_length.erase(camera_index);
        ctx.camera_denoiser_variance.erase(camera_index);
        ctx.camera_prev_denoiser_color_history.erase(camera_index);
        ctx.camera_prev_denoiser_moment_history.erase(camera_index);
        ctx.camera_prev_denoiser_history_length.erase(camera_index);

        // Update host layout
        ctx.host_data->camera_layout.remove(ctx.host_data->camera_layout.find(_camera_gid));
        ctx.host_data->camera_info_ray_layout.remove(
            ctx.host_data->camera_info_ray_layout.find(_camera_gid)
        );
        ctx.host_data->camera_rgb_ray_layout.remove(
            ctx.host_data->camera_rgb_ray_layout.find(_camera_gid)
        );
        ctx.host_data->camera_record_layout.remove(
            ctx.host_data->camera_record_layout.find(_camera_gid)
        );
        ctx.host_data->camera_pixel_record_layout.remove(
            ctx.host_data->camera_pixel_record_layout.find(_camera_gid)
        );
        ctx.host_data->camera_state_record_layout.remove(
            ctx.host_data->camera_state_record_layout.find(_camera_gid)
        );
        if (hasControllerSignal()) {
            ctx.host_data->camera_control_signal_layout.remove(
                ctx.host_data->camera_control_signal_layout.find(_camera_gid)
            );
        }

        // Update device layout
        ctx.host_data->camera_layout.syncDeviceLayout(ctx.camera_layout);
        ctx.host_data->camera_info_ray_layout.syncDeviceLayout(ctx.camera_info_ray_layout);
        ctx.host_data->camera_rgb_ray_layout.syncDeviceLayout(ctx.camera_rgb_ray_layout);
        ctx.host_data->camera_record_layout.syncDeviceLayout(ctx.camera_record_layout);
        ctx.host_data->camera_pixel_record_layout.syncDeviceLayout(ctx.camera_pixel_record_layout);
        ctx.host_data->camera_state_record_layout.syncDeviceLayout(ctx.camera_state_record_layout);
        if (hasControllerSignal()) {
            ctx.host_data->camera_control_signal_layout.syncDeviceLayout(
                ctx.camera_control_signal_layout
            );
        }
    });
}

}  // namespace rise