// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/kernel_manager/rs_kernel_camera_utils.h"
#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/logic/record/rs_camera_record_m.h"
#include "rise_origin/logic/signal/rs_camera_signal_m.h"
#include "rise_origin/logic/utils/rs_launch_c.h"
#include "rise_origin/logic/vision/rs_camera_c.h"
#include "rs_kernel_camera_stereo_manager.h"

using namespace std;
using namespace rise::rsc;
using namespace rise::logic;
using namespace rise::device;

namespace rise {

/*****************************************************************
 * RS_KernelCameraStereoManager
 *****************************************************************/
RS_KernelCameraStereoManager::RS_KernelCameraStereoManager(
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

    if (_rsc.is<RSC_CameraStereoRigidBodyLinked>()) {
        auto& stereo = _rsc.as<RSC_CameraStereoRigidBodyLinked>();
        auto& structure_managers = _kernel_manager->getKernelStructureVoxelManagers();
        auto sm_it = structure_managers.find(stereo.structure_name);
        if (sm_it == structure_managers.end())
            RS_THROW(RS_InvalidCharacterizationException(
                "Structure {} is not registered in the kernel manager", stereo.structure_name
            ));
        auto& structure_manager = *sm_it->second;

        RIndex rigid_body_sid =
            structure_manager.findRigidBodySID(stereo.body_sid, stereo.segment_bid);
        if (rigid_body_sid == RS_NULL_INDEX)
            RS_THROW(RS_InvalidCharacterizationException(
                "Rigid body in body {} segment {} of structure {} does not exist",
                stereo.body_sid,
                stereo.segment_bid,
                stereo.structure_name
            ));

        _rigid_body_sid = rigid_body_sid;

        _camera_rb_position = structure_manager.getRigidBodyPositionFromRelativeAnchor(
            stereo.body_sid, rigid_body_sid, stereo.relative_anchor
        );
        auto& body_rsc =
            structure_manager.getRSC().as<rsc::RSC_StructureVoxel>().bodies[stereo.body_sid];
        _camera_rb_orientation =
            body_rsc.relative_orientation.inverse() * stereo.relative_orientation;

        // Reference the structure so it cannot be removed while this camera is alive.
        _structure_reference =
            RS_KernelStructureVoxelReference(_kernel_manager, stereo.structure_name);
    } else {
        auto& stereo = _rsc.as<RSC_CameraStereoFree>();
        _camera_origin_position = stereo.origin_position;
        _camera_orientation = stereo.orientation;
    }

    const auto& base_rsc = _base();
    _width = base_rsc.image_width;
    _height = base_rsc.image_height;

    // Each eye renders its own perspective-sized canvas and aux buffers.
    for (RIndex eye = 0; eye < 2; ++eye) {
        _camera_current_rgb_canvas[eye].init({_height, _width}, memory_provider);
        _camera_next_rgb_canvas[eye].init({_height, _width}, memory_provider);

        _camera_current_rgb_canvas[eye].fill(RVec3f::zero(), false);
        _camera_next_rgb_canvas[eye].fill(RVec3f::zero(), false);

        _camera_aux_normal[eye].init({_height, _width}, memory_provider);
        _camera_aux_depth_z[eye].init({_height, _width}, memory_provider);
        _camera_aux_hit_position[eye].init({_height, _width}, memory_provider);
        _camera_aux_object_key[eye].init({_height, _width}, memory_provider);
        _camera_aux_roughness[eye].init({_height, _width}, memory_provider);
        _camera_aux_screen_motion[eye].init({_height, _width}, memory_provider);
        _camera_aux_endpoint_hit_position[eye].init({_height, _width}, memory_provider);
        _camera_aux_endpoint_normal[eye].init({_height, _width}, memory_provider);
        _camera_aux_endpoint_object_key[eye].init({_height, _width}, memory_provider);
        _camera_aux_endpoint_albedo[eye].init({_height, _width}, memory_provider);
        _camera_aux_endpoint_roughness[eye].init({_height, _width}, memory_provider);
        _camera_aux_endpoint_metalness[eye].init({_height, _width}, memory_provider);
        _camera_aux_endpoint_emission_radiance[eye].init({_height, _width}, memory_provider);
        _camera_aux_endpoint_material_type[eye].init({_height, _width}, memory_provider);
        _camera_aux_endpoint_screen_motion[eye].init({_height, _width}, memory_provider);
        _camera_aux_chain_throughput[eye].init({_height, _width}, memory_provider);
        _camera_aux_chain_terminal_type[eye].init({_height, _width}, memory_provider);
        _camera_prev_reservoir_di[eye].init({_height, _width}, memory_provider);
        _camera_current_reservoir_di[eye].init({_height, _width}, memory_provider);
        _camera_prev_reservoir_gi[eye].init({_height, _width}, memory_provider);
        _camera_current_reservoir_gi[eye].init({_height, _width}, memory_provider);
        _camera_spatial_reservoir_di[eye].init({_height, _width}, memory_provider);
        _camera_spatial_reservoir_gi[eye].init({_height, _width}, memory_provider);
        _camera_prev_aux_normal[eye].init({_height, _width}, memory_provider);
        _camera_prev_aux_depth_z[eye].init({_height, _width}, memory_provider);
        _camera_prev_aux_object_key[eye].init({_height, _width}, memory_provider);
        _camera_denoiser_rgb_history[eye].init({_height, _width}, memory_provider);
        _camera_denoiser_moment_history[eye].init({_height, _width}, memory_provider);
        _camera_denoiser_history_length[eye].init({_height, _width}, memory_provider);
        _camera_prev_denoiser_rgb_history[eye].init({_height, _width}, memory_provider);
        _camera_prev_denoiser_moment_history[eye].init({_height, _width}, memory_provider);
        _camera_prev_denoiser_history_length[eye].init({_height, _width}, memory_provider);
        _camera_denoiser_variance[eye].init({_height, _width}, memory_provider);
        _camera_next_denoiser_variance[eye].init({_height, _width}, memory_provider);

        _camera_aux_normal[eye].fill(RVec3f::zero(), false);
        _camera_aux_depth_z[eye].fill(RFloat(0), false);
        _camera_aux_hit_position[eye].fill(RVec3f::zero(), false);
        _camera_aux_object_key[eye].fill(logic::RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL, false);
        _camera_aux_roughness[eye].fill(RFloat(1), false);
        _camera_aux_screen_motion[eye].fill(RVec2f::zero(), false);
        _camera_aux_endpoint_hit_position[eye].fill(RVec3f::zero(), false);
        _camera_aux_endpoint_normal[eye].fill(RVec3f::zero(), false);
        _camera_aux_endpoint_object_key[eye].fill(
            logic::RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL, false
        );
        _camera_aux_endpoint_albedo[eye].fill(RS_Vec3DMem<RSE_FLOAT_16>::zero(), false);
        _camera_aux_endpoint_roughness[eye].fill(RFloat(1), false);
        _camera_aux_endpoint_metalness[eye].fill(RFloat(0), false);
        _camera_aux_endpoint_emission_radiance[eye].fill(RFloat(0), false);
        _camera_aux_endpoint_material_type[eye].fill(RSE_VISION_MATERIAL_OPAQUE, false);
        _camera_aux_endpoint_screen_motion[eye].fill(RVec2f::zero(), false);
        _camera_aux_chain_throughput[eye].fill(RVec3f::one(), false);
        _camera_aux_chain_terminal_type[eye].fill(RSE_VISION_RESTIR_CHAIN_TERMINAL_SURFACE, false);
        _camera_prev_reservoir_di[eye].fill(logic::RS_ReservoirDI{}, false);
        _camera_current_reservoir_di[eye].fill(logic::RS_ReservoirDI{}, false);
        _camera_prev_reservoir_gi[eye].fill(logic::RS_ReservoirGI{}, false);
        _camera_current_reservoir_gi[eye].fill(logic::RS_ReservoirGI{}, false);
        _camera_spatial_reservoir_di[eye].fill(logic::RS_ReservoirDI{}, false);
        _camera_spatial_reservoir_gi[eye].fill(logic::RS_ReservoirGI{}, false);
        _camera_prev_aux_normal[eye].fill(RVec3f::zero(), false);
        _camera_prev_aux_depth_z[eye].fill(float(0), false);
        _camera_prev_aux_object_key[eye].fill(
            logic::RS_CAMERA_PIXEL_OBJECT_KEY_MISS_SENTINEL, false
        );
        _camera_denoiser_rgb_history[eye].fill(RS_Vec3DMem<RSE_FLOAT_16>::zero(), false);
        _camera_denoiser_moment_history[eye].fill(RVec2f::zero(), false);
        _camera_denoiser_history_length[eye].fill(RFloat(0), false);
        _camera_prev_denoiser_rgb_history[eye].fill(RS_Vec3DMem<RSE_FLOAT_16>::zero(), false);
        _camera_prev_denoiser_moment_history[eye].fill(RVec2f::zero(), false);
        _camera_prev_denoiser_history_length[eye].fill(RFloat(0), false);
        _camera_denoiser_variance[eye].fill(RFloat(0), false);
        _camera_next_denoiser_variance[eye].fill(RFloat(0), false);
    }

    // Each eye's record buffer holds one unified perspective-style record per frame.
    const auto& record_rsc = _record();
    RBigSize px = RBigSize(_width) * _height;
    RBigSize frame_size = RS_CameraRecordMem::getRequiredMemorySize(
        px,
        record_rsc.has_depth ? px : 0,
        record_rsc.has_normal ? px : 0,
        record_rsc.has_motion ? px : 0,
        record_rsc.has_entity_info ? px : 0,
        1
    );
    for (RIndex eye = 0; eye < 2; ++eye) {
        _recorder_buffer[eye].init(
            buffer_frame_num, buffer_frame_num * frame_size, memory_provider
        );
        _controller_buffer[eye].init(frame_size, memory_provider);
    }

    // create signal buffer
    if (hasControllerSignal()) {
        _addSignals();
    }
}

void RS_KernelCameraStereoManager::commit() {
    _kernel_manager->enterKernelDataCriticalZone();
    if (hasControllerSignal()) {
        _initKernelSignalData();
    }
    _initKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;
}

void RS_KernelCameraStereoManager::decommit() {
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

void RS_KernelCameraStereoManager::freeze() {
    if (_is_frozen || !_is_kernel_data_inited) return;
    _kernel_manager->enterKernelDataCriticalZone();
    _freezeKernelData(true);
    _kernel_manager->exitKernelDataCriticalZone();
    _is_frozen = true;
}

void RS_KernelCameraStereoManager::unfreeze() {
    if (!_is_frozen || !_is_kernel_data_inited) return;
    _kernel_manager->enterKernelDataCriticalZone();
    _freezeKernelData(false);
    _kernel_manager->exitKernelDataCriticalZone();
    _is_frozen = false;
}

bool RS_KernelCameraStereoManager::isFrozen() const { return _is_frozen; }

void RS_KernelCameraStereoManager::reference() { _use_count += 1; }

void RS_KernelCameraStereoManager::dereference() {
    if (_use_count == 0) {
        RS_THROW(RS_HostExecutionException(
            "Camera '{}' use_count underflow on dereference", getEntityName()
        ));
    }
    _use_count -= 1;
}

RSize RS_KernelCameraStereoManager::getUseCount() const { return _use_count; }

void RS_KernelCameraStereoManager::syncRecorderRecords() {
    for (RIndex eye = 0; eye < 2; ++eye) {
        _recorder_buffer[eye].swapAndCopy(true, false, false);
    }
}

void RS_KernelCameraStereoManager::waitForRecorderRecordsSync() {
    for (RIndex eye = 0; eye < 2; ++eye) {
        _recorder_buffer[eye].waitForCopyFinish();
    }
}

std::vector<RS_WorldCameraStereoRecord> RS_KernelCameraStereoManager::getRecorderRecords() {
    vector<RS_WorldCameraStereoRecord> records;
    const auto& record_rsc = _record();
    RBigSize px = RBigSize(_width) * _height;
    RIndex total = _recorder_buffer[0].getHostNum();
    for (RIndex i = 0; i < total; ++i) {
        RS_CameraRecordMem eye_record[2];
        for (RIndex eye = 0; eye < 2; ++eye) {
            eye_record[eye].initFromMemory(
                _recorder_buffer[eye].getHostChunk(i).ptr(),
                px,
                record_rsc.has_depth ? px : 0,
                record_rsc.has_normal ? px : 0,
                record_rsc.has_motion ? px : 0,
                record_rsc.has_entity_info ? px : 0,
                1
            );
        }
        records.emplace_back(_assembleStereoRecord(eye_record[0], eye_record[1]));
    }
    return records;
}

bool RS_KernelCameraStereoManager::hasControllerSignal() { return _rsc.is<RSC_CameraStereoFree>(); }

void RS_KernelCameraStereoManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());
    }
}

RS_WorldCameraStereoSignal RS_KernelCameraStereoManager::getControllerSignal() {
    if (hasControllerSignal()) {
        RS_WorldCameraStereoSignal signal;
        signal.initFromMemory(_host_signal_buffer.ptr(), 1, 1, 4);
        signal.camera_gid = _entity_registration.gid();
        signal.device = _host_signal_buffer.device();
        return signal;
    }
    return {};
}

bool RS_KernelCameraStereoManager::hasControllerRecord() {
    return _rsc.is<RSC_CameraStereoFree>() || _rsc.is<RSC_CameraStereoRigidBodyLinked>();
}

void RS_KernelCameraStereoManager::syncControllerRecord() {
    if (hasControllerRecord()) {
        for (RIndex eye = 0; eye < 2; ++eye) {
            _controller_buffer[eye].sync(true);
        }
    }
}

RS_WorldCameraStereoRecord RS_KernelCameraStereoManager::getControllerRecord() {
    const auto& record_rsc = _record();
    RBigSize px = RBigSize(_width) * _height;
    RS_CameraRecordMem eye_record[2];
    for (RIndex eye = 0; eye < 2; ++eye) {
        eye_record[eye].initFromMemory(
            _controller_buffer[eye].getHostChunk(0).ptr(),
            px,
            record_rsc.has_depth ? px : 0,
            record_rsc.has_normal ? px : 0,
            record_rsc.has_motion ? px : 0,
            record_rsc.has_entity_info ? px : 0,
            1
        );
    }
    return _assembleStereoRecord(eye_record[0], eye_record[1]);
}

std::string RS_KernelCameraStereoManager::getEntityName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.name; });
}

std::string RS_KernelCameraStereoManager::getRecordName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.record->name; });
}

std::string RS_KernelCameraStereoManager::getSignalName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.signal->name; });
}

std::string RS_KernelCameraStereoManager::getRecordControllerName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.record->target_controller_reference; });
}

std::string RS_KernelCameraStereoManager::getRecordRecorderName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.record->target_recorder_reference; });
}

std::string RS_KernelCameraStereoManager::getSignalControllerName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.signal->source_controller_reference; });
}

RIndex RS_KernelCameraStereoManager::getEntityGid() const { return _entity_registration.gid(); }

RIndex RS_KernelCameraStereoManager::getRecordGid() const { return _record_registration.gid(); }

RIndex RS_KernelCameraStereoManager::getSignalGid() const { return _signal_registration.gid(); }

std::string RS_KernelCameraStereoManager::getStructureName() const {
    if (!_rsc.is<RSC_CameraStereoRigidBodyLinked>()) return "";
    return _rsc.as<RSC_CameraStereoRigidBodyLinked>().structure_name;
}

RIndex RS_KernelCameraStereoManager::getStructureGid() const {
    if (!_rsc.is<RSC_CameraStereoRigidBodyLinked>()) return RS_NULL_INDEX;
    return _structure_reference.gid();
}

const RS_KernelCameraStereoManager::SupportedRSC& RS_KernelCameraStereoManager::getRSC() const {
    return _rsc;
}

RS_WorldCameraStereoRecord RS_KernelCameraStereoManager::_assembleStereoRecord(
    RS_CameraRecordMem& left,
    RS_CameraRecordMem& right
) const {
    RS_WorldCameraStereoRecord record;
    record.left_rgb = left.rgb;
    record.right_rgb = right.rgb;
    record.left_depth = left.depth;
    record.right_depth = right.depth;
    record.left_normal = left.normal;
    record.right_normal = right.normal;
    record.left_motion = left.motion;
    record.right_motion = right.motion;
    record.left_entity_info = left.entity_info;
    record.right_entity_info = right.entity_info;
    record.left_camera_state = left.camera_state;
    record.right_camera_state = right.camera_state;
    record.is_valid = left.is_valid && right.is_valid;
    record.step = left.step;
    record.time = left.time;
    record.camera_gid = _entity_registration.gid();
    record.structure_gid = _structure_reference.gid();
    record.rigid_body_sid = _rigid_body_sid;
    record.width = _width;
    record.height = _height;
    record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
    return record;
}

void RS_KernelCameraStereoManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        auto& ctx = k_ptr->ctx;
        using ContextType = decltype(k_ptr->ctx);
        using CameraMemType = RS_CameraMem<ContextType::FloatMemAccuracy>;

        const auto& base_rsc = _base();
        const auto& record_rsc = *base_rsc.record;
        bool is_rigid = _rsc.is<RSC_CameraStereoRigidBodyLinked>();

        // The head pose places the two eyes; for a free camera it is the origin pose, for a
        // rigid-body-linked camera the device resolver overwrites each eye pose from the body.
        RQuat3rf head_orientation = _camera_orientation;
        RVec3rf head_position = _camera_origin_position;
        RVec3rf unit_right = -head_orientation.toOrthoAxisY();
        RFloat half_ipd = RFloat(base_rsc.interpupillary_distance) / RFloat(2);

        RSize pixel_count = base_rsc.image_width * base_rsc.image_height;
        RIndex entity_gid = _entity_registration.gid();

        for (RIndex eye = 0; eye < 2; ++eye) {
            RIndex segment_gid = makeCameraSegmentGid(entity_gid, eye);
            RFloat eye_sign = (eye == 0) ? RFloat(-1) : RFloat(1);

            CameraMemType camera;
            camera.gid = segment_gid;
            camera.sub_camera_index = eye;
            RS_CameraStateUnion state{0};

            if (is_rigid) {
                state.camera_state.type = RSE_CAMERA_RIGID_BODY_LINKED_STEREO;
                camera.rigid_body_sid = _rigid_body_sid;
                camera.rigid_body_anchor_rb_position = _camera_rb_position;
                camera.rigid_body_anchor_rb_orientation = _camera_rb_orientation;
            } else {
                state.camera_state.type = RSE_CAMERA_FREE_STEREO;
                camera.position = head_position + unit_right * (eye_sign * half_ipd);
                camera.orientation = head_orientation;
            }

            camera.saturation_radiance = base_rsc.saturation_radiance;
            camera.aspect_ratio = static_cast<RFloat>(base_rsc.image_height) /
                                  static_cast<RFloat>(base_rsc.image_width);
            camera.image_width = base_rsc.image_width;
            camera.image_height = base_rsc.image_height;

            camera.samples_per_pixel = RS_MAX(base_rsc.samples_per_pixel, RSize(1));
            camera.max_depth = base_rsc.max_depth;

            camera.interpupillary_distance = base_rsc.interpupillary_distance;
            camera.convergence_distance = base_rsc.convergence_distance;
            camera.angle_left = base_rsc.angle_left;
            camera.angle_right = base_rsc.angle_right;
            camera.angle_up = base_rsc.angle_up;
            camera.angle_down = base_rsc.angle_down;
            camera.hyperbolic_near_depth = record_rsc.hyperbolic_near_depth;
            camera.hyperbolic_far_depth = record_rsc.hyperbolic_far_depth;

            state.camera_state.has_rgb = record_rsc.has_rgb;
            state.camera_state.has_depth = record_rsc.has_depth;
            state.camera_state.has_normal = record_rsc.has_normal;
            state.camera_state.has_motion = record_rsc.has_motion;
            state.camera_state.has_entity_info = record_rsc.has_entity_info;
            state.camera_state.is_depth_hyperbolic = record_rsc.is_depth_hyperbolic;
            camera.state = state.camera_state_packed;

            if (hasControllerSignal()) {
                camera.camera_signal_handle = _camera_signal;
            }

            // Update host layout, one segment per eye keyed by the sub-camera segment gid.
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

            // Write this eye's data to the device.
            ctx.cameras.append(camera);
            ctx.camera_records.append(RS_CameraRecordMem{});
            ctx.camera_recorder_buffers.append(_recorder_buffer[eye]);
            ctx.camera_controller_buffers.append(_controller_buffer[eye]);
            ctx.camera_current_rgb_canvas.append(_camera_current_rgb_canvas[eye]);
            ctx.camera_next_rgb_canvas.append(_camera_next_rgb_canvas[eye]);
            ctx.camera_aux_normal.append(_camera_aux_normal[eye]);
            ctx.camera_aux_depth_z.append(_camera_aux_depth_z[eye]);
            ctx.camera_aux_hit_position.append(_camera_aux_hit_position[eye]);
            ctx.camera_aux_object_key.append(_camera_aux_object_key[eye]);
            ctx.camera_aux_roughness.append(_camera_aux_roughness[eye]);
            ctx.camera_aux_screen_motion.append(_camera_aux_screen_motion[eye]);
            ctx.camera_aux_endpoint_hit_position.append(_camera_aux_endpoint_hit_position[eye]);
            ctx.camera_aux_endpoint_normal.append(_camera_aux_endpoint_normal[eye]);
            ctx.camera_aux_endpoint_object_key.append(_camera_aux_endpoint_object_key[eye]);
            ctx.camera_aux_endpoint_albedo.append(_camera_aux_endpoint_albedo[eye]);
            ctx.camera_aux_endpoint_roughness.append(_camera_aux_endpoint_roughness[eye]);
            ctx.camera_aux_endpoint_metalness.append(_camera_aux_endpoint_metalness[eye]);
            ctx.camera_aux_endpoint_emission_radiance.append(
                _camera_aux_endpoint_emission_radiance[eye]
            );
            ctx.camera_aux_endpoint_material_type.append(_camera_aux_endpoint_material_type[eye]);
            ctx.camera_aux_endpoint_screen_motion.append(_camera_aux_endpoint_screen_motion[eye]);
            ctx.camera_aux_chain_throughput.append(_camera_aux_chain_throughput[eye]);
            ctx.camera_aux_chain_terminal_type.append(_camera_aux_chain_terminal_type[eye]);
            ctx.camera_prev_reservoir_di.append(_camera_prev_reservoir_di[eye]);
            ctx.camera_current_reservoir_di.append(_camera_current_reservoir_di[eye]);
            ctx.camera_prev_reservoir_gi.append(_camera_prev_reservoir_gi[eye]);
            ctx.camera_current_reservoir_gi.append(_camera_current_reservoir_gi[eye]);
            ctx.camera_spatial_reservoir_di.append(_camera_spatial_reservoir_di[eye]);
            ctx.camera_spatial_reservoir_gi.append(_camera_spatial_reservoir_gi[eye]);
            ctx.camera_prev_aux_normal.append(_camera_prev_aux_normal[eye]);
            ctx.camera_prev_aux_depth_z.append(_camera_prev_aux_depth_z[eye]);
            ctx.camera_prev_aux_object_key.append(_camera_prev_aux_object_key[eye]);
            ctx.camera_denoiser_rgb_history.append(_camera_denoiser_rgb_history[eye]);
            ctx.camera_denoiser_moment_history.append(_camera_denoiser_moment_history[eye]);
            ctx.camera_denoiser_history_length.append(_camera_denoiser_history_length[eye]);
            ctx.camera_prev_denoiser_rgb_history.append(_camera_prev_denoiser_rgb_history[eye]);
            ctx.camera_prev_denoiser_moment_history.append(
                _camera_prev_denoiser_moment_history[eye]
            );
            ctx.camera_prev_denoiser_history_length.append(
                _camera_prev_denoiser_history_length[eye]
            );
            ctx.camera_denoiser_variance.append(_camera_denoiser_variance[eye]);
            ctx.camera_next_denoiser_variance.append(_camera_next_denoiser_variance[eye]);
        }

        // Update device layout once both eyes' segments are added.
        ctx.host_data->camera_layout.syncDeviceLayout(ctx.camera_layout);
        ctx.host_data->camera_info_ray_layout.syncDeviceLayout(ctx.camera_info_ray_layout);
        ctx.host_data->camera_rgb_ray_layout.syncDeviceLayout(ctx.camera_rgb_ray_layout);
        ctx.host_data->camera_record_layout.syncDeviceLayout(ctx.camera_record_layout);
        ctx.host_data->camera_rgb_record_layout.syncDeviceLayout(ctx.camera_rgb_record_layout);
        ctx.host_data->camera_depth_record_layout.syncDeviceLayout(ctx.camera_depth_record_layout);
        ctx.host_data->camera_normal_record_layout.syncDeviceLayout(
            ctx.camera_normal_record_layout
        );
        ctx.host_data->camera_motion_record_layout.syncDeviceLayout(
            ctx.camera_motion_record_layout
        );
        ctx.host_data->camera_entity_info_record_layout.syncDeviceLayout(
            ctx.camera_entity_info_record_layout
        );
        ctx.host_data->camera_state_record_layout.syncDeviceLayout(ctx.camera_state_record_layout);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();

        if (hasControllerSignal()) {
            // Seed initial per-eye poses into the host signal buffer.
            _host_signal.left_position_signals.position[0] = head_position - unit_right * half_ipd;
            _host_signal.right_position_signals.position[0] = head_position + unit_right * half_ipd;
            _host_signal.left_orientation_signals.orientation[0] = head_orientation;
            _host_signal.right_orientation_signals.orientation[0] = head_orientation;

            // Seed both eyes with the configured frustum angles, ordered left, right, up, down. The
            // VR client overwrites them with the headset's real per-eye angles each step.
            for (RIndex eye = 0; eye < 2; ++eye) {
                auto& optics = (eye == 0) ? _host_signal.left_optics_signals
                                          : _host_signal.right_optics_signals;
                optics.field_of_view[0] = base_rsc.angle_left;
                optics.field_of_view[1] = base_rsc.angle_right;
                optics.field_of_view[2] = base_rsc.angle_up;
                optics.field_of_view[3] = base_rsc.angle_down;
            }

            _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());
        }

        // Resolve data dependency for the rigid-body-linked camera; the resolver places each eye.
        if (is_rigid) {
            _kernel_manager->syncKernelIfNeeded();
            castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
                using KernelType = std::remove_pointer_t<decltype(k_ptr)>;

                RIndex rigid_body_offset = ctx.host_data->structure_rigid_body_layout.segmentOffset(
                    ctx.host_data->structure_rigid_body_layout.find(_structure_reference.gid())
                );
                for (RIndex eye = 0; eye < 2; ++eye) {
                    RIndex camera_index =
                        ctx.host_data->camera_layout.find(makeCameraSegmentGid(entity_gid, eye));
                    p_ptr->template run<RSCF_LaunchDirect<
                        RSCF_ResolveRigidBodyLinkedStereoCameraDependency,
                        KernelType>>({1}, camera_index, rigid_body_offset);
                }
            });
        }
    });
}

void RS_KernelCameraStereoManager::_freeKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        if (!_is_kernel_data_inited) return;

        auto& ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        RIndex entity_gid = _entity_registration.gid();

        // Remove both eyes. Erasing the higher segment first keeps the lower index valid.
        for (RIndex i = 0; i < 2; ++i) {
            RIndex eye = 1 - i;
            RIndex segment_gid = makeCameraSegmentGid(entity_gid, eye);
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
            ctx.camera_aux_endpoint_emission_radiance.erase(camera_index);
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
            ctx.camera_denoiser_rgb_history.erase(camera_index);
            ctx.camera_denoiser_moment_history.erase(camera_index);
            ctx.camera_denoiser_history_length.erase(camera_index);
            ctx.camera_prev_denoiser_rgb_history.erase(camera_index);
            ctx.camera_prev_denoiser_moment_history.erase(camera_index);
            ctx.camera_prev_denoiser_history_length.erase(camera_index);
            ctx.camera_denoiser_variance.erase(camera_index);
            ctx.camera_next_denoiser_variance.erase(camera_index);

            ctx.host_data->camera_layout.remove(camera_index);
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
        }

        // Update device layout
        ctx.host_data->camera_layout.syncDeviceLayout(ctx.camera_layout);
        ctx.host_data->camera_info_ray_layout.syncDeviceLayout(ctx.camera_info_ray_layout);
        ctx.host_data->camera_rgb_ray_layout.syncDeviceLayout(ctx.camera_rgb_ray_layout);
        ctx.host_data->camera_record_layout.syncDeviceLayout(ctx.camera_record_layout);
        ctx.host_data->camera_rgb_record_layout.syncDeviceLayout(ctx.camera_rgb_record_layout);
        ctx.host_data->camera_depth_record_layout.syncDeviceLayout(ctx.camera_depth_record_layout);
        ctx.host_data->camera_normal_record_layout.syncDeviceLayout(
            ctx.camera_normal_record_layout
        );
        ctx.host_data->camera_motion_record_layout.syncDeviceLayout(
            ctx.camera_motion_record_layout
        );
        ctx.host_data->camera_entity_info_record_layout.syncDeviceLayout(
            ctx.camera_entity_info_record_layout
        );
        ctx.host_data->camera_state_record_layout.syncDeviceLayout(ctx.camera_state_record_layout);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelCameraStereoManager::_addSignals() {
    RBigSize total_size = RS_WorldCameraStereoSignal::getRequiredMemorySize(1, 1, 4);

    RS_MemoryProvider* memory_provider = _scoped_device_provider.get();
    _host_signal_buffer = memory_provider->allocateHost(total_size, true);
    _device_signal_buffer = memory_provider->allocate(total_size, true);
    _host_signal_buffer.set(0);
    _device_signal_buffer.set(0);

    _host_signal.initFromMemory(_host_signal_buffer.ptr(), 1, 1, 4);
    _host_signal.camera_gid = _entity_registration.gid();
    _device_signal.initFromMemory(_device_signal_buffer.ptr(), 1, 1, 4);
    _device_signal.camera_gid = _entity_registration.gid();
    _device_signal.device = _device_signal_buffer.device();
}

void RS_KernelCameraStereoManager::_initKernelSignalData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        ctx.host_data->camera_signal_layout.add(1, _entity_registration.gid());
        ctx.host_data->camera_signal_layout.syncDeviceLayout(ctx.camera_signal_layout);
        ctx.camera_signals.resize({ctx.host_data->camera_signal_layout.capacity()});
        _camera_signal = ctx.host_data->camera_signal_layout.segmentOffset(
            ctx.host_data->camera_signal_layout.find(_entity_registration.gid())
        );
        RS_CameraSignalMem signal_mem;
        signal_mem.type = RSE_CAMERA_SIGNAL_STEREO;
        signal_mem.stereo = _device_signal;
        ctx.camera_signals.writeSlice(&signal_mem, 1, _camera_signal);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelCameraStereoManager::_freeKernelSignalData() {
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

void RS_KernelCameraStereoManager::_freezeKernelData(bool is_freeze) {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Toggle both eyes' layouts so render launches skip the camera while frozen; the
        // current canvas is never cleared or retraced, so it holds the last frame.
        bool is_active = !is_freeze;
        RIndex entity_gid = _entity_registration.gid();
        for (RIndex eye = 0; eye < 2; ++eye) {
            RIndex segment_gid = makeCameraSegmentGid(entity_gid, eye);
            ctx.host_data->camera_layout.setSegmentIsActive(
                ctx.host_data->camera_layout.find(segment_gid), is_active
            );
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
        }

        ctx.host_data->camera_layout.syncDeviceLayout(ctx.camera_layout);
        ctx.host_data->camera_info_ray_layout.syncDeviceLayout(ctx.camera_info_ray_layout);
        ctx.host_data->camera_rgb_ray_layout.syncDeviceLayout(ctx.camera_rgb_ray_layout);
        ctx.host_data->camera_record_layout.syncDeviceLayout(ctx.camera_record_layout);
        ctx.host_data->camera_rgb_record_layout.syncDeviceLayout(ctx.camera_rgb_record_layout);
        ctx.host_data->camera_depth_record_layout.syncDeviceLayout(ctx.camera_depth_record_layout);
        ctx.host_data->camera_normal_record_layout.syncDeviceLayout(
            ctx.camera_normal_record_layout
        );
        ctx.host_data->camera_motion_record_layout.syncDeviceLayout(
            ctx.camera_motion_record_layout
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
 * RS_KernelCameraStereoReference
 *****************************************************************/
RS_KernelCameraStereoReference::RS_KernelCameraStereoReference(
    RS_KernelManager* kernel_manager,
    std::string name
)
    : _kernel_manager(kernel_manager), _name(std::move(name)) {
    auto& managers = _kernel_manager->getKernelCameraStereoManagers();
    auto it = managers.find(_name);
    if (it == managers.end()) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Camera {} is not registered in the kernel manager", _name
        ));
    }
    it->second->reference();
    _gid = it->second->getEntityGid();
}

RS_KernelCameraStereoReference::~RS_KernelCameraStereoReference() { _release(); }

RS_KernelCameraStereoReference::RS_KernelCameraStereoReference(
    RS_KernelCameraStereoReference&& other
) noexcept
    : _kernel_manager(other._kernel_manager), _name(std::move(other._name)), _gid(other._gid) {
    other._kernel_manager = nullptr;
    other._gid = RS_NULL_INDEX;
}

RS_KernelCameraStereoReference& RS_KernelCameraStereoReference::operator=(
    RS_KernelCameraStereoReference&& other
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

void RS_KernelCameraStereoReference::_release() {
    if (_kernel_manager == nullptr) return;
    auto& managers = _kernel_manager->getKernelCameraStereoManagers();
    if (auto it = managers.find(_name); it != managers.end()) {
        it->second->dereference();
    }
    _kernel_manager = nullptr;
    _gid = RS_NULL_INDEX;
}

}  // namespace rise
