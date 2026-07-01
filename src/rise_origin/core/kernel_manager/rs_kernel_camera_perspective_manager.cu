// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/kernel_manager/rs_kernel_camera_utils.h"
#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/logic/signal/rs_camera_signal_m.h"
#include "rise_origin/logic/utils/rs_launch_c.h"
#include "rise_origin/logic/vision/rs_camera_c.h"
#include "rs_kernel_camera_perspective_manager.h"

using namespace std;
using namespace rise::rsc;
using namespace rise::logic;
using namespace rise::device;

namespace rise {

/*****************************************************************
 * RS_KernelCameraPerspectiveManager
 *****************************************************************/
RS_KernelCameraPerspectiveManager::RS_KernelCameraPerspectiveManager(
    RS_KernelManager* kernel_manager,
    const SupportedRSC& camera_rsc
)
    : _kernel_manager(kernel_manager), _rsc(camera_rsc) {
    _scoped_device_provider =
        std::make_unique<RS_MemoryScopedProvider>(_kernel_manager->getDeviceProvider());
    auto entity_name = _rsc.visit([](auto& c) -> std::string { return c.name; });
    _entity_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_ENTITY, entity_name);
    _record_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_RECORD, entity_name + ":record");
    _signal_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_SIGNAL, entity_name + ":signal");

    auto memory_provider = _scoped_device_provider.get();
    auto buffer_frame_num = _kernel_manager->getKernelRecordBufferFrameNum();

    RSize pixel_num = 0;
    if (_rsc.is<RSC_CameraPerspectiveRigidBodyLinked>()) {
        auto& perspective = _rsc.as<RSC_CameraPerspectiveRigidBodyLinked>();
        auto& structure_managers = _kernel_manager->getKernelStructureVoxelManagers();
        auto sm_it = structure_managers.find(perspective.structure_name);
        if (sm_it == structure_managers.end())
            RS_THROW(RS_InvalidCharacterizationException(
                "Structure {} is not registered in the kernel manager", perspective.structure_name
            ));
        auto& structure_manager = *sm_it->second;

        RIndex rigid_body_sid =
            structure_manager.findRigidBodySID(perspective.body_sid, perspective.segment_bid);
        if (rigid_body_sid == RS_NULL_INDEX)
            RS_THROW(RS_InvalidCharacterizationException(
                "Rigid body in body {} segment {} of structure {} does not exist",
                perspective.body_sid,
                perspective.segment_bid,
                perspective.structure_name
            ));

        _rigid_body_sid = rigid_body_sid;

        _camera_rb_position = structure_manager.getRigidBodyPositionFromRelativeAnchor(
            perspective.body_sid, rigid_body_sid, perspective.relative_anchor
        );
        auto& body_rsc =
            structure_manager.getRSC().as<rsc::RSC_StructureVoxel>().bodies[perspective.body_sid];
        _camera_rb_orientation =
            body_rsc.relative_orientation.inverse() * perspective.relative_orientation;
        pixel_num = perspective.image_width * perspective.image_height;
        _width = perspective.image_width;
        _height = perspective.image_height;
        _camera_current_rgb_canvas.init(
            {perspective.image_height, perspective.image_width}, memory_provider
        );
        _camera_next_rgb_canvas.init(
            {perspective.image_height, perspective.image_width}, memory_provider
        );

        // Reference the structure so it cannot be removed while this camera is alive.
        _structure_reference =
            RS_KernelStructureVoxelReference(_kernel_manager, perspective.structure_name);
    } else {
        auto& perspective = _rsc.as<RSC_CameraPerspectiveFree>();
        _camera_origin_position = perspective.origin_position;
        _camera_orientation = perspective.orientation;
        pixel_num = perspective.image_width * perspective.image_height;
        _width = perspective.image_width;
        _height = perspective.image_height;
        _camera_current_rgb_canvas.init(
            {perspective.image_height, perspective.image_width}, memory_provider
        );
        _camera_next_rgb_canvas.init(
            {perspective.image_height, perspective.image_width}, memory_provider
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
    _camera_aux_chain_terminal_type.init({_height, _width}, memory_provider);
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
    _camera_prev_denoiser_color_history.init({_height, _width}, memory_provider);
    _camera_prev_denoiser_moment_history.init({_height, _width}, memory_provider);
    _camera_prev_denoiser_history_length.init({_height, _width}, memory_provider);
    _camera_denoiser_variance.init({_height, _width}, memory_provider);
    _camera_next_denoiser_variance.init({_height, _width}, memory_provider);

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
    _camera_aux_chain_terminal_type.fill(RSE_VISION_RESTIR_CHAIN_TERMINAL_SURFACE, false);
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
    _camera_prev_denoiser_color_history.fill(RVec3f::zero(), false);
    _camera_prev_denoiser_moment_history.fill(RVec2f::zero(), false);
    _camera_prev_denoiser_history_length.fill(RFloat(0), false);
    _camera_denoiser_variance.fill(RFloat(0), false);
    _camera_next_denoiser_variance.fill(RFloat(0), false);

    // create record buffer
    const auto& record_rsc = _record();
    RBigSize px = RBigSize(pixel_num);
    RBigSize frame_size = RS_WorldCameraPerspectiveRecord::getRequiredMemorySize(
        px,
        record_rsc.has_depth ? px : 0,
        record_rsc.has_normal ? px : 0,
        record_rsc.has_motion ? px : 0,
        record_rsc.has_entity_info ? px : 0,
        1
    );
    _recorder_buffer.init(buffer_frame_num, buffer_frame_num * frame_size, memory_provider);

    // Note: for rigid body linked perspective camera, it is special, it needs to have a controller
    // record buffer, but it does not have a controller or a controller signal buffer
    _controller_buffer.init(frame_size, memory_provider);

    // create signal buffer
    if (hasControllerSignal()) {
        _addSignals();
    }
}

void RS_KernelCameraPerspectiveManager::commit() {
    _kernel_manager->enterKernelDataCriticalZone();
    if (hasControllerSignal()) {
        _initKernelSignalData();
    }
    _initKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;
}

void RS_KernelCameraPerspectiveManager::decommit() {
    if (_is_kernel_data_inited) {
        _kernel_manager->enterKernelDataCriticalZone();
        _freeKernelData();
        if (hasControllerSignal()) {
            _freeKernelSignalData();
        }
        _kernel_manager->exitKernelDataCriticalZone();
        _is_kernel_data_inited = false;
    }
}

void RS_KernelCameraPerspectiveManager::freeze() {
    if (_is_frozen || !_is_kernel_data_inited) return;
    _kernel_manager->enterKernelDataCriticalZone();
    _freezeKernelData(true);
    _kernel_manager->exitKernelDataCriticalZone();
    _is_frozen = true;
}

void RS_KernelCameraPerspectiveManager::unfreeze() {
    if (!_is_frozen || !_is_kernel_data_inited) return;
    _kernel_manager->enterKernelDataCriticalZone();
    _freezeKernelData(false);
    _kernel_manager->exitKernelDataCriticalZone();
    _is_frozen = false;
}

bool RS_KernelCameraPerspectiveManager::isFrozen() const { return _is_frozen; }

void RS_KernelCameraPerspectiveManager::reference() { _use_count += 1; }

void RS_KernelCameraPerspectiveManager::dereference() {
    if (_use_count == 0) {
        RS_THROW(RS_HostExecutionException(
            "Camera '{}' use_count underflow on dereference", getEntityName()
        ));
    }
    _use_count -= 1;
}

RSize RS_KernelCameraPerspectiveManager::getUseCount() const { return _use_count; }

void RS_KernelCameraPerspectiveManager::syncRecorderRecords() {
    _recorder_buffer.swapAndCopy(true, false, false);
}

void RS_KernelCameraPerspectiveManager::waitForRecorderRecordsSync() {
    _recorder_buffer.waitForCopyFinish();
}

std::vector<RS_WorldCameraPerspectiveRecord>
RS_KernelCameraPerspectiveManager::getRecorderRecords() {
    vector<RS_WorldCameraPerspectiveRecord> records;
    const auto& record_rsc = _record();
    RBigSize px = RBigSize(_width) * _height;
    for (RIndex i = 0, total = _recorder_buffer.getHostNum(); i < total; ++i) {
        RS_WorldCameraPerspectiveRecord record;
        record.initFromMemory(
            _recorder_buffer.getHostChunk(i).ptr(),
            px,
            record_rsc.has_depth ? px : 0,
            record_rsc.has_normal ? px : 0,
            record_rsc.has_motion ? px : 0,
            record_rsc.has_entity_info ? px : 0,
            1
        );
        record.camera_gid = _entity_registration.gid();
        record.structure_gid = _structure_reference.gid();
        record.rigid_body_sid = _rigid_body_sid;
        record.width = _width;
        record.height = _height;
        record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
        records.emplace_back(record);
    }
    return records;
}

bool RS_KernelCameraPerspectiveManager::hasControllerSignal() {
    return _rsc.is<RSC_CameraPerspectiveFree>();
}

void RS_KernelCameraPerspectiveManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());
    }
}

RS_WorldCameraPerspectiveSignal RS_KernelCameraPerspectiveManager::getControllerSignal() {
    if (hasControllerSignal()) {
        RS_WorldCameraPerspectiveSignal signal;
        signal.initFromMemory(_host_signal_buffer.ptr(), 1, 1, 1);
        signal.camera_gid = _entity_registration.gid();
        signal.device = _host_signal_buffer.device();
        return signal;
    }
    return {};
}

bool RS_KernelCameraPerspectiveManager::hasControllerRecord() {
    return _rsc.is<RSC_CameraPerspectiveFree>() || _rsc.is<RSC_CameraPerspectiveRigidBodyLinked>();
}

void RS_KernelCameraPerspectiveManager::syncControllerRecord() {
    if (hasControllerRecord()) {
        _controller_buffer.sync(true);
    }
}

RS_WorldCameraPerspectiveRecord RS_KernelCameraPerspectiveManager::getControllerRecord() {
    RS_WorldCameraPerspectiveRecord record;
    const auto& record_rsc = _record();
    RBigSize px = RBigSize(_width) * _height;
    record.initFromMemory(
        _controller_buffer.getHostChunk(0).ptr(),
        px,
        record_rsc.has_depth ? px : 0,
        record_rsc.has_normal ? px : 0,
        record_rsc.has_motion ? px : 0,
        record_rsc.has_entity_info ? px : 0,
        1
    );
    record.camera_gid = _entity_registration.gid();
    record.structure_gid = _structure_reference.gid();
    record.rigid_body_sid = _rigid_body_sid;
    record.width = _width;
    record.height = _height;
    record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
    return record;
}

std::string RS_KernelCameraPerspectiveManager::getEntityName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.name; });
}

std::string RS_KernelCameraPerspectiveManager::getRecordName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.record.name; });
}

std::string RS_KernelCameraPerspectiveManager::getSignalName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.signal.name; });
}

std::string RS_KernelCameraPerspectiveManager::getRecordControllerName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.record.target_controller_reference; });
}

std::string RS_KernelCameraPerspectiveManager::getRecordRecorderName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.record.target_recorder_reference; });
}

std::string RS_KernelCameraPerspectiveManager::getSignalControllerName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.signal.source_controller_reference; });
}

RIndex RS_KernelCameraPerspectiveManager::getEntityGid() const {
    return _entity_registration.gid();
}

RIndex RS_KernelCameraPerspectiveManager::getRecordGid() const {
    return _record_registration.gid();
}

RIndex RS_KernelCameraPerspectiveManager::getSignalGid() const {
    return _signal_registration.gid();
}

std::string RS_KernelCameraPerspectiveManager::getStructureName() const {
    if (!_rsc.is<RSC_CameraPerspectiveRigidBodyLinked>()) return "";
    return _rsc.as<RSC_CameraPerspectiveRigidBodyLinked>().structure_name;
}

RIndex RS_KernelCameraPerspectiveManager::getStructureGid() const {
    if (!_rsc.is<RSC_CameraPerspectiveRigidBodyLinked>()) return RS_NULL_INDEX;
    return _structure_reference.gid();
}

const RS_KernelCameraPerspectiveManager::SupportedRSC&
RS_KernelCameraPerspectiveManager::getRSC() const {
    return _rsc;
}

void RS_KernelCameraPerspectiveManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        auto& ctx = k_ptr->ctx;
        using ContextType = decltype(k_ptr->ctx);
        using CameraMemType = RS_CameraMem<ContextType::FloatMemAccuracy>;

        CameraMemType camera;
        camera.gid = _entity_registration.gid();
        RS_CameraStateUnion state{0};

        const rsc::RSC_CameraPerspectiveBase* perspective_base = nullptr;
        if (_rsc.is<RSC_CameraPerspectiveRigidBodyLinked>()) {
            state.camera_state.type = RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE;
            auto& perspective = _rsc.as<RSC_CameraPerspectiveRigidBodyLinked>();

            camera.rigid_body_sid = _rigid_body_sid;
            camera.rigid_body_anchor_rb_position = _camera_rb_position;
            camera.rigid_body_anchor_rb_orientation = _camera_rb_orientation;
            perspective_base = &perspective;
        } else {
            state.camera_state.type = RSE_CAMERA_FREE_PERSPECTIVE;
            auto& perspective = _rsc.as<RSC_CameraPerspectiveFree>();
            camera.position = _camera_origin_position;
            camera.orientation = _camera_orientation;
            perspective_base = &perspective;
        }
        const auto& record_rsc = perspective_base->record;

        camera.aspect_ratio = static_cast<RFloat>(perspective_base->image_height) /
                              static_cast<RFloat>(perspective_base->image_width);
        camera.image_width = perspective_base->image_width;
        camera.image_height = perspective_base->image_height;

        camera.samples_per_pixel = RS_MAX(perspective_base->samples_per_pixel, RSize(1));
        camera.max_depth = perspective_base->max_depth;

        camera.focus_distance = perspective_base->focus_distance;
        camera.field_of_view = perspective_base->field_of_view;
        camera.defocus_angle = perspective_base->defocus_angle;
        camera.defocus_radius =
            perspective_base->focus_distance *
            tanf(static_cast<float>(perspective_base->defocus_angle * RS_PI) / 180.0f / 2.0f);
        camera.hyperbolic_near_depth = record_rsc.hyperbolic_near_depth;
        camera.hyperbolic_far_depth = record_rsc.hyperbolic_far_depth;

        float theta = static_cast<float>(perspective_base->field_of_view * RS_PI) / 180.0f;
        float h = tanf(theta / 2.0f);

        float viewport_height = 2.0f * h * perspective_base->focus_distance;
        float viewport_width = viewport_height / camera.aspect_ratio;

        camera.pixel_height = viewport_height / static_cast<float>(camera.image_height);
        camera.pixel_width = viewport_width / static_cast<float>(camera.image_width);

        state.camera_state.has_rgb = record_rsc.has_rgb;
        state.camera_state.has_depth = record_rsc.has_depth;
        state.camera_state.has_normal = record_rsc.has_normal;
        state.camera_state.has_motion = record_rsc.has_motion;
        state.camera_state.has_entity_info = record_rsc.has_entity_info;
        state.camera_state.is_depth_hyperbolic = record_rsc.is_depth_hyperbolic;

        camera.state = state.camera_state_packed;

        RSize pixel_count = camera.image_width * camera.image_height;

        // The mono camera occupies sub-camera index 0 of its entity's segment gid block.
        RIndex segment_gid = makeCameraSegmentGid(_entity_registration.gid(), 0);

        // Update host layout
        ctx.host_data->camera_layout.add(1, segment_gid);
        ctx.host_data->camera_info_ray_layout.add(pixel_count, segment_gid);
        ctx.host_data->camera_rgb_ray_layout.add(
            pixel_count * camera.samples_per_pixel, segment_gid
        );
        ctx.host_data->camera_record_layout.add(1, segment_gid);
        ctx.host_data->camera_rgb_record_layout.add(pixel_count, segment_gid);
        ctx.host_data->camera_depth_record_layout.add(
            record_rsc.has_depth ? pixel_count : 0, segment_gid
        );
        ctx.host_data->camera_normal_record_layout.add(
            record_rsc.has_normal ? pixel_count : 0, segment_gid
        );
        ctx.host_data->camera_motion_record_layout.add(
            record_rsc.has_motion ? pixel_count : 0, segment_gid
        );
        ctx.host_data->camera_entity_info_record_layout.add(
            record_rsc.has_entity_info ? pixel_count : 0, segment_gid
        );
        ctx.host_data->camera_state_record_layout.add(1, segment_gid);

        // Update device layout
        ctx.host_data->camera_layout.syncDeviceLayout(ctx.camera_layout);
        ctx.host_data->camera_info_ray_layout.syncDeviceLayout(ctx.camera_info_ray_layout);
        ctx.host_data->camera_rgb_ray_layout.syncDeviceLayout(ctx.camera_rgb_ray_layout);
        ctx.host_data->camera_record_layout.syncDeviceLayout(ctx.camera_record_layout);
        ctx.host_data->camera_rgb_record_layout.syncDeviceLayout(ctx.camera_rgb_record_layout);
        ctx.host_data->camera_depth_record_layout.syncDeviceLayout(ctx.camera_depth_record_layout);
        ctx.host_data->camera_normal_record_layout.syncDeviceLayout(ctx.camera_normal_record_layout
        );
        ctx.host_data->camera_motion_record_layout.syncDeviceLayout(ctx.camera_motion_record_layout
        );
        ctx.host_data->camera_entity_info_record_layout.syncDeviceLayout(
            ctx.camera_entity_info_record_layout
        );
        ctx.host_data->camera_state_record_layout.syncDeviceLayout(ctx.camera_state_record_layout);

        // Write data to device
        if (hasControllerSignal()) {
            camera.camera_signal_handle = _camera_signal;
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
        ctx.camera_aux_chain_terminal_type.append(_camera_aux_chain_terminal_type);
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
        ctx.camera_prev_denoiser_color_history.append(_camera_prev_denoiser_color_history);
        ctx.camera_prev_denoiser_moment_history.append(_camera_prev_denoiser_moment_history);
        ctx.camera_prev_denoiser_history_length.append(_camera_prev_denoiser_history_length);
        ctx.camera_denoiser_variance.append(_camera_denoiser_variance);
        ctx.camera_next_denoiser_variance.append(_camera_next_denoiser_variance);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();

        if (hasControllerSignal()) {
            // Seed initial values into the host signal buffer.
            _host_signal.position_signals.position[0] = _camera_origin_position;
            _host_signal.orientation_signals.orientation[0] = _camera_orientation;
            _host_signal.optics_signals.focus_distance[0] = perspective_base->focus_distance;
            _host_signal.optics_signals.field_of_view[0] = perspective_base->field_of_view;

            _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());
        }

        // Resolve data dependency
        if (_rsc.is<RSC_CameraPerspectiveRigidBodyLinked>()) {
            _kernel_manager->syncKernelIfNeeded();
            castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
                using KernelType = std::remove_pointer_t<decltype(k_ptr)>;

                p_ptr->template run<RSCF_LaunchDirect<
                    RSCF_ResolveRigidBodyLinkedPerspectiveCameraDependency,
                    KernelType>>(
                    {1},
                    ctx.cameras.size() - 1,
                    ctx.host_data->structure_rigid_body_layout.segmentOffset(
                        ctx.host_data->structure_rigid_body_layout.find(_structure_reference.gid())
                    )
                );
            });
        }
    });
}

void RS_KernelCameraPerspectiveManager::_freeKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        if (!_is_kernel_data_inited) return;

        auto& ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Remove data from device
        RIndex segment_gid = makeCameraSegmentGid(_entity_registration.gid(), 0);
        RIndex camera_index = ctx.host_data->camera_layout.find(segment_gid);
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
        ctx.camera_aux_chain_terminal_type.erase(camera_index);
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
        ctx.camera_prev_denoiser_color_history.erase(camera_index);
        ctx.camera_prev_denoiser_moment_history.erase(camera_index);
        ctx.camera_prev_denoiser_history_length.erase(camera_index);
        ctx.camera_denoiser_variance.erase(camera_index);
        ctx.camera_next_denoiser_variance.erase(camera_index);

        // Update host layout
        ctx.host_data->camera_layout.remove(ctx.host_data->camera_layout.find(segment_gid));
        ctx.host_data->camera_info_ray_layout.remove(
            ctx.host_data->camera_info_ray_layout.find(segment_gid)
        );
        ctx.host_data->camera_rgb_ray_layout.remove(
            ctx.host_data->camera_rgb_ray_layout.find(segment_gid)
        );
        ctx.host_data->camera_record_layout.remove(
            ctx.host_data->camera_record_layout.find(segment_gid)
        );
        ctx.host_data->camera_rgb_record_layout.remove(
            ctx.host_data->camera_rgb_record_layout.find(segment_gid)
        );
        ctx.host_data->camera_depth_record_layout.remove(
            ctx.host_data->camera_depth_record_layout.find(segment_gid)
        );
        ctx.host_data->camera_normal_record_layout.remove(
            ctx.host_data->camera_normal_record_layout.find(segment_gid)
        );
        ctx.host_data->camera_motion_record_layout.remove(
            ctx.host_data->camera_motion_record_layout.find(segment_gid)
        );
        ctx.host_data->camera_entity_info_record_layout.remove(
            ctx.host_data->camera_entity_info_record_layout.find(segment_gid)
        );
        ctx.host_data->camera_state_record_layout.remove(
            ctx.host_data->camera_state_record_layout.find(segment_gid)
        );

        // Update device layout
        ctx.host_data->camera_layout.syncDeviceLayout(ctx.camera_layout);
        ctx.host_data->camera_info_ray_layout.syncDeviceLayout(ctx.camera_info_ray_layout);
        ctx.host_data->camera_rgb_ray_layout.syncDeviceLayout(ctx.camera_rgb_ray_layout);
        ctx.host_data->camera_record_layout.syncDeviceLayout(ctx.camera_record_layout);
        ctx.host_data->camera_rgb_record_layout.syncDeviceLayout(ctx.camera_rgb_record_layout);
        ctx.host_data->camera_depth_record_layout.syncDeviceLayout(ctx.camera_depth_record_layout);
        ctx.host_data->camera_normal_record_layout.syncDeviceLayout(ctx.camera_normal_record_layout
        );
        ctx.host_data->camera_motion_record_layout.syncDeviceLayout(ctx.camera_motion_record_layout
        );
        ctx.host_data->camera_entity_info_record_layout.syncDeviceLayout(
            ctx.camera_entity_info_record_layout
        );
        ctx.host_data->camera_state_record_layout.syncDeviceLayout(ctx.camera_state_record_layout);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelCameraPerspectiveManager::_addSignals() {
    RBigSize total_size = RS_WorldCameraPerspectiveSignal::getRequiredMemorySize(1, 1, 1);

    RS_MemoryProvider* memory_provider = _scoped_device_provider.get();
    _host_signal_buffer = memory_provider->allocateHost(total_size, true);
    _device_signal_buffer = memory_provider->allocate(total_size, true);
    _host_signal_buffer.set(0);
    _device_signal_buffer.set(0);

    _host_signal.initFromMemory(_host_signal_buffer.ptr(), 1, 1, 1);
    _host_signal.camera_gid = _entity_registration.gid();
    _device_signal.initFromMemory(_device_signal_buffer.ptr(), 1, 1, 1);
    _device_signal.camera_gid = _entity_registration.gid();
    _device_signal.device = _device_signal_buffer.device();
}

void RS_KernelCameraPerspectiveManager::_initKernelSignalData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        ctx.host_data->camera_signal_layout.add(1, _entity_registration.gid());
        ctx.host_data->camera_signal_layout.syncDeviceLayout(ctx.camera_signal_layout);
        ctx.camera_signals.resize({ctx.host_data->camera_signal_layout.capacity()});
        _camera_signal = ctx.host_data->camera_signal_layout.segmentOffset(
            ctx.host_data->camera_signal_layout.find(_entity_registration.gid())
        );
        RS_CameraSignalMem signal_mem;
        signal_mem.type = RSE_CAMERA_SIGNAL_PERSPECTIVE;
        signal_mem.perspective = _device_signal;
        ctx.camera_signals.writeSlice(&signal_mem, 1, _camera_signal);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelCameraPerspectiveManager::_freeKernelSignalData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        RIndex signal_index = ctx.host_data->camera_signal_layout.find(_entity_registration.gid());
        ctx.host_data->camera_signal_layout.remove(signal_index);
        ctx.host_data->camera_signal_layout.syncDeviceLayout(ctx.camera_signal_layout);
        ctx.camera_signals.resize({ctx.host_data->camera_signal_layout.capacity()});

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelCameraPerspectiveManager::_freezeKernelData(bool is_freeze) {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Toggle every camera layout so render launches skip the camera while frozen; the
        // current canvas is never cleared or retraced, so it holds the last frame.
        bool is_active = !is_freeze;
        RIndex segment_gid = makeCameraSegmentGid(_entity_registration.gid(), 0);
        RIndex camera_index = ctx.host_data->camera_layout.find(segment_gid);
        ctx.host_data->camera_layout.setSegmentIsActive(camera_index, is_active);
        ctx.host_data->camera_info_ray_layout.setSegmentIsActive(
            ctx.host_data->camera_info_ray_layout.find(segment_gid), is_active
        );
        ctx.host_data->camera_rgb_ray_layout.setSegmentIsActive(
            ctx.host_data->camera_rgb_ray_layout.find(segment_gid), is_active
        );
        ctx.host_data->camera_record_layout.setSegmentIsActive(
            ctx.host_data->camera_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->camera_rgb_record_layout.setSegmentIsActive(
            ctx.host_data->camera_rgb_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->camera_depth_record_layout.setSegmentIsActive(
            ctx.host_data->camera_depth_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->camera_normal_record_layout.setSegmentIsActive(
            ctx.host_data->camera_normal_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->camera_motion_record_layout.setSegmentIsActive(
            ctx.host_data->camera_motion_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->camera_entity_info_record_layout.setSegmentIsActive(
            ctx.host_data->camera_entity_info_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->camera_state_record_layout.setSegmentIsActive(
            ctx.host_data->camera_state_record_layout.find(segment_gid), is_active
        );

        ctx.host_data->camera_layout.syncDeviceLayout(ctx.camera_layout);
        ctx.host_data->camera_info_ray_layout.syncDeviceLayout(ctx.camera_info_ray_layout);
        ctx.host_data->camera_rgb_ray_layout.syncDeviceLayout(ctx.camera_rgb_ray_layout);
        ctx.host_data->camera_record_layout.syncDeviceLayout(ctx.camera_record_layout);
        ctx.host_data->camera_rgb_record_layout.syncDeviceLayout(ctx.camera_rgb_record_layout);
        ctx.host_data->camera_depth_record_layout.syncDeviceLayout(ctx.camera_depth_record_layout);
        ctx.host_data->camera_normal_record_layout.syncDeviceLayout(ctx.camera_normal_record_layout
        );
        ctx.host_data->camera_motion_record_layout.syncDeviceLayout(ctx.camera_motion_record_layout
        );
        ctx.host_data->camera_entity_info_record_layout.syncDeviceLayout(
            ctx.camera_entity_info_record_layout
        );
        ctx.host_data->camera_state_record_layout.syncDeviceLayout(ctx.camera_state_record_layout);

        if (hasControllerSignal()) {
            ctx.host_data->camera_signal_layout.setSegmentIsActive(
                ctx.host_data->camera_signal_layout.find(_entity_registration.gid()), is_active
            );
            ctx.host_data->camera_signal_layout.syncDeviceLayout(ctx.camera_signal_layout);
        }

        // Layout active state changed, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

/*****************************************************************
 * RS_KernelCameraPerspectiveReference
 *****************************************************************/
RS_KernelCameraPerspectiveReference::RS_KernelCameraPerspectiveReference(
    RS_KernelManager* kernel_manager,
    std::string name
)
    : _kernel_manager(kernel_manager), _name(std::move(name)) {
    auto& managers = _kernel_manager->getKernelCameraPerspectiveManagers();
    auto it = managers.find(_name);
    if (it == managers.end()) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Camera {} is not registered in the kernel manager", _name
        ));
    }
    it->second->reference();
    _gid = it->second->getEntityGid();
}

RS_KernelCameraPerspectiveReference::~RS_KernelCameraPerspectiveReference() { _release(); }

RS_KernelCameraPerspectiveReference::RS_KernelCameraPerspectiveReference(
    RS_KernelCameraPerspectiveReference&& other
) noexcept
    : _kernel_manager(other._kernel_manager), _name(std::move(other._name)), _gid(other._gid) {
    other._kernel_manager = nullptr;
    other._gid = RS_NULL_INDEX;
}

RS_KernelCameraPerspectiveReference& RS_KernelCameraPerspectiveReference::operator=(
    RS_KernelCameraPerspectiveReference&& other
) noexcept {
    if (this != &other) {
        _release();
        _kernel_manager = other._kernel_manager;
        _name = std::move(other._name);
        _gid = other._gid;
        other._kernel_manager = nullptr;
        other._gid = RS_NULL_INDEX;
    }
    return *this;
}

void RS_KernelCameraPerspectiveReference::_release() {
    if (_kernel_manager == nullptr) return;
    auto& managers = _kernel_manager->getKernelCameraPerspectiveManagers();
    if (auto it = managers.find(_name); it != managers.end()) {
        it->second->dereference();
    }
    _kernel_manager = nullptr;
    _gid = RS_NULL_INDEX;
}

}  // namespace rise
