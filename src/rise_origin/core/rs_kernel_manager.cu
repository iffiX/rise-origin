// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <cmath>
#include <set>

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rise_origin/utils/rs_timer.h"
#include "rs_kernel_manager.h"

using namespace std;
using namespace rise::logic;
using namespace rise::device;

namespace rise {
RS_KernelManager::RS_KernelManager() {
    // Initialize the method lookup table
    _methods = {
        {"pauseKernel", [this](RS_Any arg) { _pauseKernel(); }},
        {"resumeKernel", [this](RS_Any arg) { _resumeKernel(); }},
        {"terminateKernel", [this](RS_Any arg) { _terminateKernel(); }},
        {"beginModificationTransaction", [this](RS_Any arg) { _beginModificationTransaction(); }},
        {"endModificationTransaction", [this](RS_Any arg) { _endModificationTransaction(); }},
        {"setSimulationConfig",
         [this](RS_Any arg) { _setSimulationConfig(arg.as<RS_SimulationConfig>()); }},
        {"setFloor", [this](RS_Any arg) { _setFloor(arg.as<RS_FloorConfig>()); }},
        {"setWorld", [this](RS_Any arg) { _setWorld(arg.as<RS_WorldConfig>()); }},
        {"addMaterial", [this](RS_Any arg) { _addMaterial(arg.as<RS_MaterialConfig>()); }},
        {"addRecord", [this](RS_Any arg) { _addRecord(arg.as<RS_RecordConfig>()); }},
        {"addSignal", [this](RS_Any arg) { _addSignal(arg.as<RS_SignalConfig>()); }},
        {"addStructure", [this](RS_Any arg) { _addStructure(arg.as<RS_StructureConfig>()); }},
        {"addCamera", [this](RS_Any arg) { _addCamera(arg.as<RS_CameraConfig>()); }},
        {"updateMaterial", [this](RS_Any arg) { _updateMaterial(arg.as<RS_MaterialConfig>()); }},
        {"deleteFloor", [this](RS_Any arg) { _deleteFloor(); }},
        {"deleteMaterial", [this](RS_Any arg) { _deleteMaterial(arg.as<RIndex>()); }},
        {"deleteRecord", [this](RS_Any arg) { _deleteRecord(arg.as<RIndex>()); }},
        {"deleteSignal", [this](RS_Any arg) { _deleteSignal(arg.as<RIndex>()); }},
        {"deleteStructure", [this](RS_Any arg) { _deleteStructure(arg.as<RIndex>()); }},
        {"deleteCamera", [this](RS_Any arg) { _deleteCamera(arg.as<RIndex>()); }},
    };
}

void RS_KernelManager::init(
    RS_MemoryProvider* host_provider,
    RS_MemoryProvider* device_provider,
    RSize record_buffer_frame_num
) {
    if (logging::get("root")) _logger = logging::get("root")->clone("kernel manager");
    _thread_pool = make_shared<RS_ThreadPool>();
    _host_provider = host_provider;
    _device_provider = device_provider;
    _kernel_record_buffer_frame_num = record_buffer_frame_num;
    _kernel_synchronization_thread =
        make_unique<thread>(&RS_KernelManager::_kernelSynchronizationThreadFunc, this);
}

void RS_KernelManager::free() {
    // Not inited
    if (!_thread_pool) return;
    _thread_pool = nullptr;
    _device_provider = nullptr;
    _host_provider = nullptr;
    // Signal the synchronization thread to terminate
    _command_queue_cv.notify_one();
    _kernel_synchronization_thread->join();
    _kernel_synchronization_thread = nullptr;

    _material_name_to_gid.clear();
    _record_name_to_gid.clear();
    _signal_name_to_gid.clear();
    _structure_name_to_gid.clear();
    _camera_name_to_gid.clear();

    _record_gid_to_record_type.clear();
    _structure_gid_to_structure_record_gid.clear();
    _structure_gid_to_rigid_body_camera_gids.clear();
}

void RS_KernelManager::initKernel(RIndex kernel_id, RSE_FloatType context_mem_accuracy) {
    RS_Timer timer;

    kernel.markModified();
    kernel.id = kernel_id;
    kernel.host_data = new RS_KernelHostData();
    kernel.ctx_accuracy = context_mem_accuracy;

    _kernel_spatial_manager.init(this);

    castKernelPtrAndCall(&kernel, [&](auto k_ptr) {
        auto& k = *k_ptr;
        k.runtime_states = RS_Value(RS_KernelRuntimeStates(), _device_provider);

        k.ctx.mesh_triangles.init(0, _device_provider);
        k.ctx.mesh_triangle_layout.init(_device_provider);

        k.ctx.structure_link_materials.init(0, _device_provider);
        k.ctx.structure_voxel_materials.init(0, _device_provider);
        k.ctx.structure_links.init(0, _device_provider);
        k.ctx.structure_voxels.init(0, _device_provider);
        k.ctx.structure_rigid_bodies.init(0, _device_provider);
        k.ctx.structure_rigid_body_surface_voxels.init(0, _device_provider);
        k.ctx.structure_constraint_joints.init(0, _device_provider);

        k.ctx.structure_constraint_joint_block_ranges.init(0, _device_provider);

        k.ctx.structure_rigid_contacts.init(0, _device_provider);
        k.ctx.structure_rigid_sorted_contacts.init(0, _device_provider);
        k.ctx.structure_rigid_contact_keys.init(0, _device_provider);
        k.ctx.structure_rigid_sorted_contact_keys.init(0, _device_provider);
        k.ctx.structure_rigid_contact_indices.init(0, _device_provider);
        k.ctx.structure_rigid_sorted_contact_indices.init(0, _device_provider);
        k.ctx.structure_rigid_contact_flags.init(0, _device_provider);
        k.ctx.structure_rigid_contact_prefix_sums.init(0, _device_provider);
        k.ctx.structure_rigid_manifold_deepest_in.init(0, _device_provider);
        k.ctx.structure_rigid_manifold_deepest_out.init(0, _device_provider);
        k.ctx.structure_rigid_manifold_farthest_in.init(0, _device_provider);
        k.ctx.structure_rigid_manifold_farthest_out.init(0, _device_provider);

        k.ctx.structure_emissive_voxel_handles.init(0, _device_provider);

        k.ctx.structure_voxel_bvh_flags.init(0, _device_provider);
        k.ctx.structure_voxel_bvh_prefix_sums.init(0, _device_provider);
        k.ctx.structure_voxel_bvh_active_handles.init(0, _device_provider);

        k.ctx.structure_expansion_signals.init(0, _device_provider);
        k.ctx.structure_hinge_signals.init(0, _device_provider);
        k.ctx.structure_prismatic_signals.init(0, _device_provider);

        k.ctx.structure_link_material_layout.init(_device_provider);
        k.ctx.structure_voxel_material_layout.init(_device_provider);
        k.ctx.structure_link_layout.init(_device_provider);
        k.ctx.structure_voxel_layout.init(_device_provider);
        k.ctx.structure_rigid_body_layout.init(_device_provider);
        k.ctx.structure_rigid_body_surface_voxel_layout.init(_device_provider);
        k.ctx.structure_constraint_joint_layout.init(_device_provider);

        k.ctx.structure_constraint_joint_block_range_layout.init(_device_provider);

        k.ctx.structure_expansion_signal_layout.init(_device_provider);
        k.ctx.structure_hinge_signal_layout.init(_device_provider);
        k.ctx.structure_prismatic_signal_layout.init(_device_provider);

        k.ctx.structure_records.init(0, _device_provider);
        k.ctx.structure_recorder_buffers.init(0, _device_provider);
        k.ctx.structure_controller_buffers.init(0, _device_provider);
        k.ctx.structure_voxel_record_handles.init(0, _device_provider);
        k.ctx.structure_link_record_handles.init(0, _device_provider);
        k.ctx.structure_rigid_body_record_handles.init(0, _device_provider);
        k.ctx.structure_constraint_joint_record_handles.init(0, _device_provider);

        k.ctx.structure_voxel_record_layout.init(_device_provider);
        k.ctx.structure_link_record_layout.init(_device_provider);
        k.ctx.structure_rigid_body_record_layout.init(_device_provider);
        k.ctx.structure_joint_record_layout.init(_device_provider);
        k.ctx.structure_constraint_joint_record_layout.init(_device_provider);

        k.ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_input_layout.init(
            _device_provider
        );
        k.ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_output_layout.init(
            _device_provider
        );

        k.ctx.cameras.init(0, _device_provider);
        k.ctx.camera_records.init(0, _device_provider);
        k.ctx.camera_recorder_buffers.init(0, _device_provider);
        k.ctx.camera_controller_buffers.init(0, _device_provider);
        k.ctx.camera_current_rgb_canvas.init(0, _device_provider);
        k.ctx.camera_next_rgb_canvas.init(0, _device_provider);
        k.ctx.camera_aux_normal.init(0, _device_provider);
        k.ctx.camera_aux_depth_z.init(0, _device_provider);
        k.ctx.camera_aux_hit_position.init(0, _device_provider);
        k.ctx.camera_aux_object_key.init(0, _device_provider);
        k.ctx.camera_aux_roughness.init(0, _device_provider);
        k.ctx.camera_aux_screen_motion.init(0, _device_provider);
        k.ctx.camera_aux_endpoint_hit_position.init(0, _device_provider);
        k.ctx.camera_aux_endpoint_normal.init(0, _device_provider);
        k.ctx.camera_aux_endpoint_object_key.init(0, _device_provider);
        k.ctx.camera_aux_endpoint_albedo.init(0, _device_provider);
        k.ctx.camera_aux_endpoint_roughness.init(0, _device_provider);
        k.ctx.camera_aux_endpoint_metalness.init(0, _device_provider);
        k.ctx.camera_aux_endpoint_intensity.init(0, _device_provider);
        k.ctx.camera_aux_endpoint_material_type.init(0, _device_provider);
        k.ctx.camera_aux_endpoint_screen_motion.init(0, _device_provider);
        k.ctx.camera_aux_chain_throughput.init(0, _device_provider);
        k.ctx.camera_aux_chain_terminal_kind.init(0, _device_provider);
        k.ctx.camera_prev_reservoir_di.init(0, _device_provider);
        k.ctx.camera_current_reservoir_di.init(0, _device_provider);
        k.ctx.camera_prev_reservoir_gi.init(0, _device_provider);
        k.ctx.camera_current_reservoir_gi.init(0, _device_provider);
        k.ctx.camera_spatial_reservoir_di.init(0, _device_provider);
        k.ctx.camera_spatial_reservoir_gi.init(0, _device_provider);
        k.ctx.camera_prev_aux_normal.init(0, _device_provider);
        k.ctx.camera_prev_aux_depth_z.init(0, _device_provider);
        k.ctx.camera_prev_aux_object_key.init(0, _device_provider);
        k.ctx.camera_denoiser_color_history.init(0, _device_provider);
        k.ctx.camera_denoiser_moment_history.init(0, _device_provider);
        k.ctx.camera_denoiser_history_length.init(0, _device_provider);
        k.ctx.camera_denoiser_variance.init(0, _device_provider);
        k.ctx.camera_prev_denoiser_color_history.init(0, _device_provider);
        k.ctx.camera_prev_denoiser_moment_history.init(0, _device_provider);
        k.ctx.camera_prev_denoiser_history_length.init(0, _device_provider);

        k.ctx.camera_control_signals.init(0, _device_provider);

        k.ctx.camera_layout.init(_device_provider);
        k.ctx.camera_info_ray_layout.init(_device_provider);
        k.ctx.camera_rgb_ray_layout.init(_device_provider);
        k.ctx.camera_record_layout.init(_device_provider);
        k.ctx.camera_pixel_record_layout.init(_device_provider);
        k.ctx.camera_state_record_layout.init(_device_provider);

        k.ctx.camera_control_signal_layout.init(_device_provider);

        k.ctx.force_signals.init(0, _device_provider);
        k.ctx.force_signal_layout.init(_device_provider);
        k.ctx.force_rigid.init(0, _device_provider);
        k.ctx.force_soft.init(0, _device_provider);

        k.ctx.torque_signals.init(0, _device_provider);
        k.ctx.torque_signal_layout.init(_device_provider);
        k.ctx.torque_rigid.init(0, _device_provider);

        k.ctx.vision_materials.init(0, _device_provider);
        k.ctx.vision_material_layout.init(_device_provider);

        k.ctx.light_weight.init(0, _device_provider);
        k.ctx.light_cdf.init(0, _device_provider);

        k.ctx.host_data = new RS_ContextHostData();

        k.ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
            .associateInputLayout(k.ctx.host_data->structure_rigid_body_surface_voxel_layout);
        k.ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
            .associateOutputLayout(k.ctx.host_data->structure_rigid_body_layout);
    });

    RS_LOG(_logger, debug, "initKernel takes {}", timer.elapsedTimeString());
    is_kernel_inited.store(true, std::memory_order_release);
}

void RS_KernelManager::freeKernel() {
    if (!kernel.isTerminated()) {
        RS_THROW(RS_HostExecutionException("Kernel is not terminated"));
    }

    RS_Timer timer;

    is_kernel_inited.store(false, std::memory_order_release);

    // Free kernel data managers
    _deleteFloor();
    _deleteWorld();
    for (auto& manager : _kernel_material_managers) manager.second.free();
    for (auto& manager : _kernel_structure_record_managers) manager.second.free();
    for (auto& manager : _kernel_force_managers) manager.second.free();
    for (auto& manager : _kernel_torque_managers) manager.second.free();
    for (auto& manager : _kernel_structure_managers) manager.second.free();
    for (auto& manager : _kernel_camera_managers) manager.second.free();
    _kernel_spatial_manager.free();

    castKernelPtrAndCall(&kernel, [&](auto k_ptr) {
        auto& k = *k_ptr;
        k.runtime_states.free();

        k.ctx.mesh_triangles.free();
        k.ctx.mesh_triangle_layout.free();

        k.ctx.structure_link_materials.free();
        k.ctx.structure_voxel_materials.free();
        k.ctx.structure_links.free();
        k.ctx.structure_voxels.free();
        k.ctx.structure_rigid_bodies.free();
        k.ctx.structure_rigid_body_surface_voxels.free();
        k.ctx.structure_constraint_joints.free();

        k.ctx.structure_constraint_joint_block_ranges.free();

        k.ctx.structure_rigid_contacts.free();
        k.ctx.structure_rigid_sorted_contacts.free();
        k.ctx.structure_rigid_contact_keys.free();
        k.ctx.structure_rigid_sorted_contact_keys.free();
        k.ctx.structure_rigid_contact_indices.free();
        k.ctx.structure_rigid_sorted_contact_indices.free();
        k.ctx.structure_rigid_contact_flags.free();
        k.ctx.structure_rigid_contact_prefix_sums.free();
        k.ctx.structure_rigid_manifold_deepest_in.free();
        k.ctx.structure_rigid_manifold_deepest_out.free();
        k.ctx.structure_rigid_manifold_farthest_in.free();
        k.ctx.structure_rigid_manifold_farthest_out.free();

        k.ctx.structure_emissive_voxel_handles.free();

        k.ctx.structure_voxel_bvh_flags.free();
        k.ctx.structure_voxel_bvh_prefix_sums.free();
        k.ctx.structure_voxel_bvh_active_handles.free();

        k.ctx.structure_expansion_signals.free();
        k.ctx.structure_hinge_signals.free();
        k.ctx.structure_prismatic_signals.free();

        k.ctx.structure_link_material_layout.free();
        k.ctx.structure_voxel_material_layout.free();
        k.ctx.structure_link_layout.free();
        k.ctx.structure_voxel_layout.free();
        k.ctx.structure_rigid_body_layout.free();
        k.ctx.structure_rigid_body_surface_voxel_layout.free();
        k.ctx.structure_constraint_joint_layout.free();

        k.ctx.structure_constraint_joint_block_range_layout.free();

        k.ctx.structure_expansion_signal_layout.free();
        k.ctx.structure_hinge_signal_layout.free();
        k.ctx.structure_prismatic_signal_layout.free();

        k.ctx.structure_records.free();
        k.ctx.structure_recorder_buffers.free();
        k.ctx.structure_controller_buffers.free();
        k.ctx.structure_voxel_record_handles.free();
        k.ctx.structure_link_record_handles.free();
        k.ctx.structure_rigid_body_record_handles.free();
        k.ctx.structure_constraint_joint_record_handles.free();

        k.ctx.structure_voxel_record_layout.free();
        k.ctx.structure_link_record_layout.free();
        k.ctx.structure_rigid_body_record_layout.free();
        k.ctx.structure_joint_record_layout.free();
        k.ctx.structure_constraint_joint_record_layout.free();

        k.ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_input_layout.free();
        k.ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_output_layout.free();

        k.ctx.cameras.free();
        k.ctx.camera_records.free();
        k.ctx.camera_recorder_buffers.free();
        k.ctx.camera_controller_buffers.free();
        k.ctx.camera_current_rgb_canvas.free();
        k.ctx.camera_next_rgb_canvas.free();
        k.ctx.camera_aux_normal.free();
        k.ctx.camera_aux_depth_z.free();
        k.ctx.camera_aux_hit_position.free();
        k.ctx.camera_aux_object_key.free();
        k.ctx.camera_aux_roughness.free();
        k.ctx.camera_aux_screen_motion.free();
        k.ctx.camera_aux_endpoint_hit_position.free();
        k.ctx.camera_aux_endpoint_normal.free();
        k.ctx.camera_aux_endpoint_object_key.free();
        k.ctx.camera_aux_endpoint_albedo.free();
        k.ctx.camera_aux_endpoint_roughness.free();
        k.ctx.camera_aux_endpoint_metalness.free();
        k.ctx.camera_aux_endpoint_intensity.free();
        k.ctx.camera_aux_endpoint_material_type.free();
        k.ctx.camera_aux_endpoint_screen_motion.free();
        k.ctx.camera_aux_chain_throughput.free();
        k.ctx.camera_aux_chain_terminal_kind.free();
        k.ctx.camera_prev_reservoir_di.free();
        k.ctx.camera_current_reservoir_di.free();
        k.ctx.camera_prev_reservoir_gi.free();
        k.ctx.camera_current_reservoir_gi.free();
        k.ctx.camera_spatial_reservoir_di.free();
        k.ctx.camera_spatial_reservoir_gi.free();
        k.ctx.camera_prev_aux_normal.free();
        k.ctx.camera_prev_aux_depth_z.free();
        k.ctx.camera_prev_aux_object_key.free();
        k.ctx.camera_denoiser_color_history.free();
        k.ctx.camera_denoiser_moment_history.free();
        k.ctx.camera_denoiser_history_length.free();
        k.ctx.camera_denoiser_variance.free();
        k.ctx.camera_prev_denoiser_color_history.free();
        k.ctx.camera_prev_denoiser_moment_history.free();
        k.ctx.camera_prev_denoiser_history_length.free();

        k.ctx.camera_control_signals.free();

        k.ctx.camera_layout.free();
        k.ctx.camera_info_ray_layout.free();
        k.ctx.camera_rgb_ray_layout.free();
        k.ctx.camera_record_layout.free();
        k.ctx.camera_pixel_record_layout.free();
        k.ctx.camera_state_record_layout.free();

        k.ctx.camera_control_signal_layout.free();

        k.ctx.force_signals.free();
        k.ctx.force_signal_layout.free();
        k.ctx.force_rigid.free();
        k.ctx.force_soft.free();

        k.ctx.torque_signals.free();
        k.ctx.torque_signal_layout.free();
        k.ctx.torque_rigid.free();

        k.ctx.vision_materials.free();
        k.ctx.vision_material_layout.free();

        k.ctx.light_weight.free();
        k.ctx.light_cdf.free();

        delete k.ctx.host_data;
    });
    delete kernel.host_data;

    RS_LOG(_logger, debug, "freeKernel takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::pauseCommandExecution() {
    // Increase the pause count
    _command_exec_pause_count.fetch_add(1, std::memory_order_acq_rel);

    // wait until manager is no longer in the middle of a command
    // Busy-wait or sleep until the manager has finished its current command
    // i.e. _command_exec_lock == 0
    while (_command_exec_lock.load(std::memory_order_acquire) == 1) {
        // Avoid pure spin if commands can be long; short sleep is typically fine
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RS_KernelManager::resumeCommandExecution() {
    // Decrease the pause count
    _command_exec_pause_count.fetch_sub(1, std::memory_order_acq_rel);
    // No special wait needed. If the count hits 0, the manager can proceed.
}

void RS_KernelManager::syncRecorderRecords() {
    RS_KernelManagerGuard guard(this);
    for (auto& manager : _kernel_structure_record_managers) {
        if (manager.second.hasRecorderRecords()) manager.second.syncRecorderRecords();
    }
    for (auto& manager : _kernel_camera_managers) {
        if (manager.second.hasRecorderRecords()) manager.second.syncRecorderRecords();
    }
}

void RS_KernelManager::waitForRecorderRecordsSync() {
    RS_KernelManagerGuard guard(this);
    for (auto& manager : _kernel_structure_record_managers) {
        if (manager.second.hasRecorderRecords()) manager.second.waitForRecorderRecordsSync();
    }
    for (auto& manager : _kernel_camera_managers) {
        if (manager.second.hasRecorderRecords()) manager.second.waitForRecorderRecordsSync();
    }
}
vector<pair<string, vector<RS_SimulationStructureRecord>>>
RS_KernelManager::getRecorderStructureRecords() {
    RS_KernelManagerGuard guard(this);
    vector<pair<string, vector<RS_SimulationStructureRecord>>> structure_records;

    structure_records.reserve(_kernel_structure_record_managers.size());
    for (auto& pair : _kernel_structure_record_managers) {
        if (!pair.second.hasRecorderRecords()) continue;
        structure_records.emplace_back(
            pair.second.getStructureName(), pair.second.getRecorderRecords()
        );
    }
    return move(structure_records);
}

vector<pair<string, vector<RS_SimulationCameraRecord>>>
RS_KernelManager::getRecorderCameraRecords() {
    RS_KernelManagerGuard guard(this);
    vector<pair<string, vector<RS_SimulationCameraRecord>>> camera_records;

    camera_records.reserve(_kernel_camera_managers.size());
    for (auto& pair : _kernel_camera_managers) {
        if (!pair.second.hasRecorderRecords()) continue;
        camera_records.emplace_back(pair.second.getName(), pair.second.getRecorderRecords());
    }
    return move(camera_records);
}

void RS_KernelManager::syncControllerRecords() {
    RS_KernelManagerGuard guard(this);
    for (auto& manager : _kernel_structure_record_managers) {
        if (manager.second.hasControllerRecord()) {
            manager.second.syncControllerRecord();
        }
    }
    for (auto& manager : _kernel_camera_managers) {
        if (manager.second.hasControllerRecord()) {
            manager.second.syncControllerRecord();
        }
    }
}

void RS_KernelManager::syncControllerSignals() {
    RS_KernelManagerGuard guard(this);
    for (auto& manager : _kernel_structure_managers) {
        if (manager.second.hasControllerSignal()) {
            manager.second.syncControllerSignal();
        }
    }
    for (auto& manager : _kernel_camera_managers) {
        if (manager.second.hasControllerSignal()) {
            manager.second.syncControllerSignal();
        }
    }
    for (auto& manager : _kernel_force_managers) {
        if (manager.second.hasControllerSignal()) {
            manager.second.syncControllerSignal();
        }
    }
    for (auto& manager : _kernel_torque_managers) {
        if (manager.second.hasControllerSignal()) {
            manager.second.syncControllerSignal();
        }
    }
}

vector<RS_SimulationStructureController> RS_KernelManager::getStructureControllers() {
    RS_KernelManagerGuard guard(this);
    vector<RS_SimulationStructureController> structure_controllers;

    for (auto& st_manager : _kernel_structure_managers) {
        RIndex gid = st_manager.first;

        bool has_record =
            _structure_gid_to_structure_record_gid.find(gid) !=
                _structure_gid_to_structure_record_gid.end() &&
            _kernel_structure_record_managers.at(_structure_gid_to_structure_record_gid.at(gid))
                .hasControllerRecord();
        bool has_cameras = _structure_gid_to_rigid_body_camera_gids.find(gid) !=
                           _structure_gid_to_rigid_body_camera_gids.end();

        if (!st_manager.second.hasControllerSignal() && !has_record && !has_cameras) continue;

        RS_SimulationStructureController controller;
        controller.name = st_manager.second.getName();
        if (has_record) {
            controller.structure_record =
                _kernel_structure_record_managers.at(_structure_gid_to_structure_record_gid.at(gid))
                    .getControllerRecord();
        }
        if (has_cameras) {
            for (auto& camera_gid : _structure_gid_to_rigid_body_camera_gids.at(gid)) {
                auto& camera_manager = _kernel_camera_managers.at(camera_gid);
                if (camera_manager.hasControllerRecord()) {
                    controller.camera_records.push_back(camera_manager.getControllerRecord());
                }
            }
        }
        controller.signal = st_manager.second.getControllerSignal();
        structure_controllers.emplace_back(controller);
    }

    return move(structure_controllers);
}

vector<RS_SimulationCameraController> RS_KernelManager::getCameraControllers() {
    RS_KernelManagerGuard guard(this);
    vector<RS_SimulationCameraController> camera_controllers;

    camera_controllers.reserve(_kernel_camera_managers.size());
    for (auto& cam_manager : _kernel_camera_managers) {
        // Only free perspective cameras gets controller, rigid body linked
        // perspective cameras' outputs are redirected to structure.
        if (cam_manager.second.hasControllerSignal()) {
            RS_SimulationCameraController controller;
            controller.name = cam_manager.second.getName();
            controller.camera_record = cam_manager.second.getControllerRecord();
            controller.signal = cam_manager.second.getControllerSignal();
            camera_controllers.emplace_back(controller);
        }
    }
    return move(camera_controllers);
}

vector<RS_SimulationForceController> RS_KernelManager::getForceControllers() {
    RS_KernelManagerGuard guard(this);
    vector<RS_SimulationForceController> force_controllers;

    force_controllers.reserve(_kernel_force_managers.size());
    for (auto& force_manager : _kernel_force_managers) {
        // Always true for force managers.
        if (force_manager.second.hasControllerSignal()) {
            RS_SimulationForceController controller;
            controller.name = force_manager.second.getName();
            controller.signal = force_manager.second.getControllerSignal();
            force_controllers.emplace_back(controller);
        }
    }
    return move(force_controllers);
}

vector<RS_SimulationTorqueController> RS_KernelManager::getTorqueControllers() {
    RS_KernelManagerGuard guard(this);
    vector<RS_SimulationTorqueController> torque_controllers;

    torque_controllers.reserve(_kernel_torque_managers.size());
    for (auto& torque_manager : _kernel_torque_managers) {
        // Always true for torque managers.
        if (torque_manager.second.hasControllerSignal()) {
            RS_SimulationTorqueController controller;
            controller.name = torque_manager.second.getName();
            controller.signal = torque_manager.second.getControllerSignal();
            torque_controllers.emplace_back(controller);
        }
    }
    return move(torque_controllers);
}

void RS_KernelManager::pauseKernel(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("pauseKernel", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _pauseKernel();
    }
}

void RS_KernelManager::resumeKernel(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("resumeKernel", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _resumeKernel();
    }
}

void RS_KernelManager::terminateKernel(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("terminateKernel", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _terminateKernel();
        _command_queue_cv.notify_one();
    }
}

void RS_KernelManager::beginModificationTransaction(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("beginModificationTransaction", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _beginModificationTransaction();
    }
}

void RS_KernelManager::endModificationTransaction(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("endModificationTransaction", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _endModificationTransaction();
    }
}

void RS_KernelManager::setSimulationConfig(const RS_SimulationConfig& config, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("setSimulationConfig", config);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _setSimulationConfig(config);
    }
}

void RS_KernelManager::setFloor(const RS_FloorConfig& config, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("setFloor", config);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _setFloor(config);
    }
}

void RS_KernelManager::setWorld(const RS_WorldConfig& config, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("setWorld", config);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _setWorld(config);
    }
}

void RS_KernelManager::addMaterial(const RS_MaterialConfig& config, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("addMaterial", config);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _addMaterial(config);
    }
}

void RS_KernelManager::addRecord(const RS_RecordConfig& config, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("addRecord", config);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _addRecord(config);
    }
}

void RS_KernelManager::addSignal(const RS_SignalConfig& config, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("addSignal", config);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _addSignal(config);
    }
}

void RS_KernelManager::addStructure(const RS_StructureConfig& config, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("addStructure", config);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _addStructure(config);
    }
}

void RS_KernelManager::addCamera(const RS_CameraConfig& config, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("addCamera", config);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _addCamera(config);
    }
}

void RS_KernelManager::updateMaterial(const RS_MaterialConfig& config, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("updateMaterial", config);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _updateMaterial(config);
    }
}

void RS_KernelManager::deleteFloor(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("deleteFloor", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _deleteFloor();
    }
}

void RS_KernelManager::deleteMaterial(RIndex id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("deleteMaterial", id);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _deleteMaterial(id);
    }
}

void RS_KernelManager::deleteRecord(RIndex id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("deleteRecord", id);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _deleteRecord(id);
    }
}

void RS_KernelManager::deleteSignal(RIndex id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("deleteSignal", id);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _deleteSignal(id);
    }
}

void RS_KernelManager::deleteStructure(RIndex id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("deleteStructure", id);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _deleteStructure(id);
    }
}

void RS_KernelManager::deleteCamera(RIndex id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace("deleteCamera", id);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerGuard guard(this);
        _deleteCamera(id);
    }
}

RIndex RS_KernelManager::findMaterial(const string& name) {
    if (kernel.isTerminated()) return RS_NULL_INDEX;
    RS_KernelManagerGuard guard(this);
    auto it = _material_name_to_gid.find(name);
    return it == _material_name_to_gid.end() ? RS_NULL_INDEX : it->second;
}

RIndex RS_KernelManager::findRecord(const string& name) {
    if (kernel.isTerminated()) return RS_NULL_INDEX;
    RS_KernelManagerGuard guard(this);
    auto it = _record_name_to_gid.find(name);
    return it == _record_name_to_gid.end() ? RS_NULL_INDEX : it->second;
}

RIndex RS_KernelManager::findSignal(const string& name) {
    if (kernel.isTerminated()) return RS_NULL_INDEX;
    RS_KernelManagerGuard guard(this);
    auto it = _signal_name_to_gid.find(name);
    return it == _signal_name_to_gid.end() ? RS_NULL_INDEX : it->second;
}

RIndex RS_KernelManager::findStructure(const string& name) {
    if (kernel.isTerminated()) return RS_NULL_INDEX;
    RS_KernelManagerGuard guard(this);
    auto it = _structure_name_to_gid.find(name);
    return it == _structure_name_to_gid.end() ? RS_NULL_INDEX : it->second;
}

RIndex RS_KernelManager::findCamera(const string& name) {
    if (kernel.isTerminated()) return RS_NULL_INDEX;
    RS_KernelManagerGuard guard(this);
    auto it = _camera_name_to_gid.find(name);
    return it == _camera_name_to_gid.end() ? RS_NULL_INDEX : it->second;
}

RS_Config RS_KernelManager::getConfig() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerGuard guard(this);
    RS_Config config;

    // Call each data manager's getConfig method instead of using stored configs
    config.simulation_config = getSimulationConfig();
    config.floor_config = getFloorConfig();
    config.material_configs = getMaterialConfigs();
    config.record_configs = getRecordConfigs();
    config.signal_configs = getSignalConfigs();
    config.structure_configs = getStructureConfigs();
    config.camera_configs = getCameraConfigs();

    return config;
}

RS_SimulationConfig RS_KernelManager::getSimulationConfig() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerGuard guard(this);
    return _simulation_config;
}

RS_WorldConfig RS_KernelManager::getWorldConfig() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerGuard guard(this);
    return _kernel_world_manager ? _kernel_world_manager->getConfig() : RS_WorldConfig{};
}

RS_FloorConfig RS_KernelManager::getFloorConfig() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerGuard guard(this);
    return _kernel_floor_manager->getConfig();
}

vector<RS_MaterialConfig> RS_KernelManager::getMaterialConfigs() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerGuard guard(this);
    vector<RS_MaterialConfig> material_configs;

    // Iterate through each material manager and collect their configurations
    for (const auto& pair : _kernel_material_managers) {
        material_configs.push_back(pair.second.getConfig());
    }
    return material_configs;
}

vector<RS_RecordConfig> RS_KernelManager::getRecordConfigs() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerGuard guard(this);
    vector<RS_RecordConfig> record_configs;

    // Iterate through each record manager and collect their configurations
    for (const auto& pair : _kernel_structure_record_managers) {
        record_configs.push_back(pair.second.getConfig());
    }
    return record_configs;
}

vector<RS_SignalConfig> RS_KernelManager::getSignalConfigs() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerGuard guard(this);
    vector<RS_SignalConfig> signal_configs;

    // Iterate through each force manager and collect their configurations
    for (const auto& pair : _kernel_force_managers) {
        signal_configs.push_back(pair.second.getConfig());
    }
    // Iterate through each torque manager and collect their configurations
    for (const auto& pair : _kernel_torque_managers) {
        signal_configs.push_back(pair.second.getConfig());
    }
    return signal_configs;
}

std::vector<RS_StructureConfig> RS_KernelManager::getStructureConfigs() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerGuard guard(this);
    vector<RS_StructureConfig> structure_configs;

    // Iterate through each structure manager and collect their configurations
    for (const auto& pair : _kernel_structure_managers) {
        structure_configs.push_back(pair.second.getConfig());
    }
    return structure_configs;
}

vector<RS_CameraConfig> RS_KernelManager::getCameraConfigs() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerGuard guard(this);
    vector<RS_CameraConfig> camera_configs;

    // Iterate through each camera manager and collect their configurations
    for (const auto& pair : _kernel_camera_managers) {
        camera_configs.push_back(pair.second.getConfig());  // Assuming getConfig() exists
    }
    return camera_configs;
}

RS_MemoryProvider* RS_KernelManager::getHostProvider() const { return _host_provider; }

RS_MemoryProvider* RS_KernelManager::getDeviceProvider() const { return _device_provider; }

shared_ptr<logging::logger> RS_KernelManager::getLogger() const { return _logger; }

std::shared_ptr<RS_ThreadPool> RS_KernelManager::getThreadPool() const { return _thread_pool; }

RSize RS_KernelManager::getKernelRecordBufferFrameNum() const {
    return _kernel_record_buffer_frame_num;
}

// Counters
RIndex RS_KernelManager::getMaterialCounter() const { return _material_counter; }

RIndex RS_KernelManager::getRecordCounter() const { return _record_counter; }

RIndex RS_KernelManager::getSignalCounter() const { return _signal_counter; }

RIndex RS_KernelManager::getStructureCounter() const { return _structure_counter; }

RIndex RS_KernelManager::getCameraCounter() const { return _camera_counter; }

// Name maps
const unordered_map<string, RIndex>& RS_KernelManager::getMaterialNameToGid() const {
    return _material_name_to_gid;
}

const unordered_map<string, RIndex>& RS_KernelManager::getRecordNameToGid() const {
    return _record_name_to_gid;
}

const unordered_map<string, RIndex>& RS_KernelManager::getSignalNameToGid() const {
    return _signal_name_to_gid;
}

const unordered_map<string, RIndex>& RS_KernelManager::getStructureNameToGid() const {
    return _structure_name_to_gid;
}

const unordered_map<string, RIndex>& RS_KernelManager::getCameraNameToGid() const {
    return _camera_name_to_gid;
}

// Kernel data managers
RS_KernelSpatialManager& RS_KernelManager::getKernelSpatialManager() {
    return _kernel_spatial_manager;
}

const unique_ptr<RS_KernelFloorManager>& RS_KernelManager::getKernelFloorManager() const {
    return _kernel_floor_manager;
}

const unordered_map<RIndex, RS_KernelMaterialManager>&
RS_KernelManager::getKernelMaterialManagers() const {
    return _kernel_material_managers;
}

const unordered_map<RIndex, RS_KernelStructureRecordManager>&
RS_KernelManager::getKernelStructureRecordManagers() const {
    return _kernel_structure_record_managers;
}

const unordered_map<RIndex, RS_KernelForceManager>&
RS_KernelManager::getKernelForceManagers() const {
    return _kernel_force_managers;
}

const unordered_map<RIndex, RS_KernelTorqueManager>&
RS_KernelManager::getKernelTorqueManagers() const {
    return _kernel_torque_managers;
}

const unordered_map<RIndex, RS_KernelStructureManager>&
RS_KernelManager::getKernelStructureManagers() const {
    return _kernel_structure_managers;
}

const unordered_map<RIndex, RS_KernelCameraManager>&
RS_KernelManager::getKernelCameraManagers() const {
    return _kernel_camera_managers;
}

void RS_KernelManager::_kernelSynchronizationThreadFunc() {
    while (!is_kernel_inited.load(std::memory_order_acquire)) std::this_thread::yield();
    RS_LOG(_logger, debug, "kernel manager sync thread running");
    while (true) {
        unique_lock lock(_command_queue_mutex);
        _command_queue_cv.wait(lock, [this]() {
            return !_command_queue.empty() || kernel.isTerminated();
        });

        if (kernel.isTerminated()) {
            // Termination case 1: synchronized termination
            RS_LOG(_logger, debug, "kernel manager sync thread entering termination");
            while (!_command_queue.empty()) {
                _command_queue.pop();
            }
            RS_LOG(_logger, debug, "kernel manager sync thread terminated");
            break;
        }

        RS_LOG(_logger, debug, "kernel manager sync thread start fetch command");
        auto [function_name, arg] = _command_queue.front();
        _command_queue.pop();

        lock.unlock();

        RS_LOG(
            _logger,
            debug,
            "kernel manager sync thread fetched command: {}, "
            "entering execution barrier",
            function_name
        );

        // Critical region
        while (true) {
            // Tell the world we *intend* to execute
            _command_exec_lock.store(1, std::memory_order_release);

            // We wish the synchronization thread to only execute when it's free
            // And when no higher-priority pause command is issued. If we detect
            // a pause command, release the execution lock, backup and restart
            if (_command_exec_pause_count.load(std::memory_order_acquire) > 0) {
                _command_exec_lock.store(0, std::memory_order_release);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                // We are ready to execute
                break;
            }
        }

        RS_LOG(_logger, debug, "kernel manager sync thread start execution");

        auto it = _methods.find(function_name);
        if (it != _methods.end()) {
            try {
                it->second(arg);
            } catch (const RS_InvalidConfigurationException& e) {
                RS_LOG(
                    _logger,
                    error,
                    "While executing command: {}, "
                    "invalid configuration exception occurred: {}",
                    function_name,
                    e.what()
                );
            }
        } else {
            RS_LOG(_logger, warn, "Unknown command in queue: {}", function_name);
        }

        RS_LOG(_logger, debug, "kernel manager sync thread end execution");

        // Finished executing
        _command_exec_lock.store(0, std::memory_order_release);

        if (function_name == "terminateKernel") {
            // Termination case 2: unsynchronized termination from command queue
            RS_LOG(_logger, debug, "kernel manager sync thread terminated");
            break;
        }
    }
}

void RS_KernelManager::_pauseKernel() {
    lock_guard lock(kernel.getSyncMutex());
    kernel.markModified();
    kernel.pause();
    RS_LOG(_logger, debug, "Pause kernel");
}

void RS_KernelManager::_resumeKernel() {
    lock_guard lock(kernel.getSyncMutex());
    kernel.markModified();
    kernel.resume();
    RS_LOG(_logger, debug, "Resume kernel");
}

void RS_KernelManager::_terminateKernel() {
    if (is_kernel_inited.load(std::memory_order_acquire)) {
        lock_guard lock(kernel.getSyncMutex());
        kernel.markModified();
        kernel.terminate();
    } else {
        // The kernel initialization is not completed, just mark it as terminated
        kernel.terminate();
    }
    RS_LOG(_logger, debug, "Terminate kernel");
}

void RS_KernelManager::_beginModificationTransaction() {
    kernel.lockSyncMutex();
    RS_LOG(_logger, debug, "Begin modification transaction");
}

void RS_KernelManager::_endModificationTransaction() {
    kernel.unlockSyncMutex();
    RS_LOG(_logger, debug, "End modification transaction");
}

void RS_KernelManager::_setSimulationConfig(const RS_SimulationConfig& config) {
    lock_guard lock(kernel.getSyncMutex());

    RS_Timer timer;

    // Sync to make sure all operations in memory provider stream are finished
    _device_provider->sync();

    // Kernel settings is modified, needs to update kernel
    kernel.markModified();

    kernel.seed = config.seed;

    // Integration
    kernel.integration_time_step = config.integration_time_step;
    kernel.integration_dt_frac = config.integration_dt_frac;

    if (config.integration_time_step > 0) {
        kernel.host_data->runtime_states.time_step = config.integration_time_step;
        kernel.host_data->runtime_states.dt =
            config.integration_dt_frac * config.integration_time_step;
        kernel.runtime_states.write(&kernel.host_data->runtime_states);
    }

    // Soft body
    kernel.softbody_internal_damping_z = config.softbody_internal_damping_z;
    kernel.softbody_contact_damping_z = config.softbody_contact_damping_z;
    kernel.softbody_global_damping_z = config.softbody_global_damping_z;

    // Rigid body constraint solver
    kernel.rigidbody_constraint_iterations = config.rigidbody_constraint_iterations;
    kernel.rigidbody_constraint_position_correction_ratio =
        config.rigidbody_constraint_position_correction_ratio;
    kernel.rigidbody_constraint_warm_start_damping_z =
        config.rigidbody_constraint_warm_start_damping_z;
    kernel.rigidbody_constraint_compliance_initial = config.rigidbody_constraint_compliance_initial;
    kernel.rigidbody_constraint_compliance_min_threshold =
        config.rigidbody_constraint_compliance_min_threshold;
    kernel.rigidbody_constraint_compliance_max_threshold =
        config.rigidbody_constraint_compliance_max_threshold;
    kernel.rigidbody_constraint_compliance_auto_adjust_watt_threshold =
        config.rigidbody_constraint_compliance_auto_adjust_watt_threshold;
    kernel.rigidbody_constraint_compliance_auto_adjust_step =
        config.rigidbody_constraint_compliance_auto_adjust_step;

    // Rigid body contact solver
    kernel.rigidbody_contact_iterations = config.rigidbody_contact_iterations;
    kernel.rigidbody_contact_position_correction_iterations =
        config.rigidbody_contact_position_correction_iterations;
    kernel.rigidbody_contact_position_correction_ratio =
        config.rigidbody_contact_position_correction_ratio;
    kernel.rigidbody_contact_warm_start_damping_z = config.rigidbody_contact_warm_start_damping_z;
    kernel.rigidbody_contact_restitution = config.rigidbody_contact_restitution;
    kernel.rigidbody_contact_slop = config.rigidbody_contact_slop;

    // Rigid body damping
    kernel.rigidbody_linear_damping_z = config.rigidbody_linear_damping_z;
    kernel.rigidbody_angular_damping_z = config.rigidbody_angular_damping_z;

    // Vision trace
    kernel.vision_trace_firefly_cap = config.vision_trace_firefly_cap;
    kernel.vision_trace_russian_roulette_start_depth =
        config.vision_trace_russian_roulette_start_depth;

    // Vision ReSTIR DI / GI
    kernel.vision_is_restir_enabled = config.vision_is_restir_enabled;
    kernel.vision_restir_disocclusion_position_threshold =
        config.vision_restir_disocclusion_position_threshold;
    kernel.vision_restir_disocclusion_normal_threshold =
        config.vision_restir_disocclusion_normal_threshold;
    kernel.vision_restir_max_specular_chain = config.vision_restir_max_specular_chain;
    kernel.vision_restir_di_m_cap = config.vision_restir_di_m_cap;
    kernel.vision_restir_di_spatial_neighbors = config.vision_restir_di_spatial_neighbors;
    kernel.vision_restir_di_spatial_radius = config.vision_restir_di_spatial_radius;
    kernel.vision_restir_gi_is_bounce_nee_enabled =
        config.vision_restir_gi_is_bounce_nee_enabled;
    kernel.vision_restir_gi_m_cap = config.vision_restir_gi_m_cap;
    kernel.vision_restir_gi_spatial_neighbors = config.vision_restir_gi_spatial_neighbors;
    kernel.vision_restir_gi_spatial_radius = config.vision_restir_gi_spatial_radius;
    kernel.vision_restir_gi_validation_period = config.vision_restir_gi_validation_period;

    // Vision denoise
    kernel.vision_denoise_type = config.vision_denoise_type;
    kernel.vision_denoise_atrous_iterations = config.vision_denoise_atrous_iterations;
    kernel.vision_denoise_atrous_min_passes = config.vision_denoise_atrous_min_passes;
    kernel.vision_denoise_sigma_color = config.vision_denoise_sigma_color;
    kernel.vision_denoise_sigma_normal = config.vision_denoise_sigma_normal;
    kernel.vision_denoise_sigma_depth = config.vision_denoise_sigma_depth;
    kernel.vision_denoise_temporal_alpha = config.vision_denoise_temporal_alpha;
    kernel.vision_denoise_temporal_alpha_moments =
        config.vision_denoise_temporal_alpha_moments;
    kernel.vision_denoise_temporal_min_history =
        config.vision_denoise_temporal_min_history;

    // Control
    kernel.control_frequency = config.control_frequency;
    kernel.record_frequency = config.record_frequency;

    // Stop condition
    kernel.host_data->stop_condition = config.stop_condition;

    _simulation_config = config;

    RS_LOG(_logger, debug, "setSimulationConfig takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_setWorld(const RS_WorldConfig& config) {
    lock_guard lock(kernel.getSyncMutex());

    RS_Timer timer;

    // Sync to make sure all operations in memory provider stream are finished
    _device_provider->sync();

    // Kernel settings is modified, needs to update kernel
    kernel.markModified();

    if (kernel.world_grav_acc != config.grav_acc) {
        kernel.world_grav_acc = config.grav_acc;

        for (auto& material_manager : _kernel_material_managers) {
            _updateMaterial(material_manager.second.getConfig());
        }
    }

    // Delegate world-render setup to the world manager, which owns the
    // ArHosekSkyModelState lifetime when running the Hosek-Wilkie sky.
    if (_kernel_world_manager == nullptr) {
        _kernel_world_manager = make_unique<RS_KernelWorldManager>();
        _kernel_world_manager->init(this, config);
    } else {
        _kernel_world_manager->setWorld(config);
    }

    RS_LOG(_logger, debug, "setWorld takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_setFloor(const RS_FloorConfig& config) {
    RS_Timer timer;

    _deleteFloor();

    _kernel_floor_manager = make_unique<RS_KernelFloorManager>();
    _kernel_floor_manager->init(this, config);

    RS_LOG(_logger, debug, "setFloor takes {}", timer.elapsedTimeString());
}

RIndex RS_KernelManager::_allocateMaterialGid() {
    for (RIndex tried = 0; tried < RS_SIMULATION_MAX_MATERIAL_NUM; ++tried) {
        if (_material_counter >= RS_SIMULATION_MAX_MATERIAL_NUM) {
            // Reached the upper bound, wrap around and reuse freed gids
            RS_LOG(
                _logger,
                warn,
                "Material gid counter reached upper bound {}, wrapping to reuse freed gids",
                RS_SIMULATION_MAX_MATERIAL_NUM
            );
            _material_counter = 0;
        }
        RIndex gid = _material_counter++;
        if (_kernel_material_managers.find(gid) == _kernel_material_managers.end()) return gid;
    }
    RS_THROW(RS_InvalidConfigurationException(
        "Reached the maximum live material count {}", RS_SIMULATION_MAX_MATERIAL_NUM
    ));
}

RIndex RS_KernelManager::_allocateRecordGid() {
    for (RIndex tried = 0; tried < RS_SIMULATION_MAX_RECORD_NUM; ++tried) {
        if (_record_counter >= RS_SIMULATION_MAX_RECORD_NUM) {
            // Reached the upper bound, wrap around and reuse freed gids
            RS_LOG(
                _logger,
                warn,
                "Record gid counter reached upper bound {}, wrapping to reuse freed gids",
                RS_SIMULATION_MAX_RECORD_NUM
            );
            _record_counter = 0;
        }
        RIndex gid = _record_counter++;
        if (_record_gid_to_record_type.find(gid) == _record_gid_to_record_type.end()) return gid;
    }
    RS_THROW(RS_InvalidConfigurationException(
        "Reached the maximum live record count {}", RS_SIMULATION_MAX_RECORD_NUM
    ));
}

RIndex RS_KernelManager::_allocateSignalGid() {
    // Force and torque signals share the signal gid space
    for (RIndex tried = 0; tried < RS_SIMULATION_MAX_SIGNAL_NUM; ++tried) {
        if (_signal_counter >= RS_SIMULATION_MAX_SIGNAL_NUM) {
            // Reached the upper bound, wrap around and reuse freed gids
            RS_LOG(
                _logger,
                warn,
                "Signal gid counter reached upper bound {}, wrapping to reuse freed gids",
                RS_SIMULATION_MAX_SIGNAL_NUM
            );
            _signal_counter = 0;
        }
        RIndex gid = _signal_counter++;
        if (_kernel_force_managers.find(gid) == _kernel_force_managers.end() &&
            _kernel_torque_managers.find(gid) == _kernel_torque_managers.end())
            return gid;
    }
    RS_THROW(RS_InvalidConfigurationException(
        "Reached the maximum live signal count {}", RS_SIMULATION_MAX_SIGNAL_NUM
    ));
}

RIndex RS_KernelManager::_allocateStructureGid() {
    for (RIndex tried = 0; tried < RS_SIMULATION_MAX_STRUCTURE_NUM; ++tried) {
        if (_structure_counter >= RS_SIMULATION_MAX_STRUCTURE_NUM) {
            // Reached the upper bound, wrap around and reuse freed gids
            RS_LOG(
                _logger,
                warn,
                "Structure gid counter reached upper bound {}, wrapping to reuse freed gids",
                RS_SIMULATION_MAX_STRUCTURE_NUM
            );
            _structure_counter = 0;
        }
        RIndex gid = _structure_counter++;
        if (_kernel_structure_managers.find(gid) == _kernel_structure_managers.end()) return gid;
    }
    RS_THROW(RS_InvalidConfigurationException(
        "Reached the maximum live structure count {}", RS_SIMULATION_MAX_STRUCTURE_NUM
    ));
}

RIndex RS_KernelManager::_allocateCameraGid() {
    for (RIndex tried = 0; tried < RS_SIMULATION_MAX_CAMERA_NUM; ++tried) {
        if (_camera_counter >= RS_SIMULATION_MAX_CAMERA_NUM) {
            // Reached the upper bound, wrap around and reuse freed gids
            RS_LOG(
                _logger,
                warn,
                "Camera gid counter reached upper bound {}, wrapping to reuse freed gids",
                RS_SIMULATION_MAX_CAMERA_NUM
            );
            _camera_counter = 0;
        }
        RIndex gid = _camera_counter++;
        if (_kernel_camera_managers.find(gid) == _kernel_camera_managers.end()) return gid;
    }
    RS_THROW(RS_InvalidConfigurationException(
        "Reached the maximum live camera count {}", RS_SIMULATION_MAX_CAMERA_NUM
    ));
}

void RS_KernelManager::_addMaterial(const RS_MaterialConfig& config) {
    if (_material_name_to_gid.find(config.name) != _material_name_to_gid.end()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Added material has duplicate material name {}", config.name
        ));
    }

    RS_Timer timer;

    RIndex gid = _allocateMaterialGid();
    _material_name_to_gid[config.name] = gid;

    auto& manager = _kernel_material_managers[gid];

    try {
        manager.init(this, config);
    } catch (const RS_InvalidConfigurationException& e) {
        // Clean up the material manager
        _material_name_to_gid.erase(config.name);
        _kernel_material_managers.erase(gid);
        throw e;
    }

    RS_LOG(_logger, debug, "addMaterial takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_addRecord(const RS_RecordConfig& config) {
    if (_record_name_to_gid.find(config.name) != _record_name_to_gid.end()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Added record has duplicate record name {}", config.name
        ));
    }

    RS_Timer timer;

    if (config.type == RSE_RECORD_STRUCTURE) {
        RIndex gid = _allocateRecordGid();

        _record_gid_to_record_type[gid] = RSE_RECORD_STRUCTURE;
        _record_name_to_gid[config.name] = gid;

        auto& manager = _kernel_structure_record_managers[gid];

        try {
            manager.init(this, config);
        } catch (const RS_InvalidConfigurationException& e) {
            // Clean up the record manager
            _record_gid_to_record_type.erase(gid);
            _record_name_to_gid.erase(config.name);
            manager.free();
            _kernel_structure_record_managers.erase(gid);
            throw e;
        }

        _structure_gid_to_structure_record_gid[manager.getStructureGid()] = gid;
    }

    RS_LOG(_logger, debug, "addRecord takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_addSignal(const RS_SignalConfig& config) {
    if (_signal_name_to_gid.find(config.name) != _signal_name_to_gid.end()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Added signal has duplicate signal name {}", config.name
        ));
    }

    RS_Timer timer;

    if (config.type == RSE_SIGNAL_FORCE) {
        RIndex gid = _allocateSignalGid();

        _signal_name_to_gid[config.name] = gid;

        auto& manager = _kernel_force_managers[gid];

        try {
            manager.init(this, config);
        } catch (const RS_InvalidConfigurationException& e) {
            // Clean up the signal manager
            _signal_name_to_gid.erase(config.name);
            manager.free();
            _kernel_force_managers.erase(gid);
            throw e;
        }
    } else if (config.type == RSE_SIGNAL_TORQUE) {
        RIndex gid = _allocateSignalGid();

        _signal_name_to_gid[config.name] = gid;

        auto& manager = _kernel_torque_managers[gid];

        try {
            manager.init(this, config);
        } catch (const RS_InvalidConfigurationException& e) {
            // Clean up the signal manager
            _signal_name_to_gid.erase(config.name);
            manager.free();
            _kernel_torque_managers.erase(gid);
            throw e;
        }
    }

    RS_LOG(_logger, debug, "addSignal takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_addStructure(const RS_StructureConfig& config) {
    if (_structure_name_to_gid.find(config.name) != _structure_name_to_gid.end()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Added structure has duplicate structure name {}", config.name
        ));
    }
    RS_Timer timer;

    RIndex gid = _allocateStructureGid();
    _structure_name_to_gid[config.name] = gid;

    auto& manager = _kernel_structure_managers[gid];

    try {
        manager.init(this, config);
    } catch (const RS_InvalidConfigurationException& e) {
        // Clean up the structure manager
        _structure_name_to_gid.erase(config.name);
        _kernel_structure_managers.erase(gid);
        throw e;
    }

    RS_LOG(_logger, debug, "addStructure takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_addCamera(const RS_CameraConfig& config) {
    if (_camera_name_to_gid.find(config.name) != _camera_name_to_gid.end()) {
        RS_THROW(RS_InvalidConfigurationException(
            "Added camera has duplicate camera name {}", config.name
        ));
    }
    RS_Timer timer;

    RIndex gid = _allocateCameraGid();
    _camera_name_to_gid[config.name] = gid;

    auto& manager = _kernel_camera_managers[gid];
    try {
        manager.init(this, config);
    } catch (const RS_InvalidConfigurationException& e) {
        // Clean up the camera manager
        _camera_name_to_gid.erase(config.name);
        _kernel_camera_managers.erase(gid);
        throw e;
    }

    if (config.type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE)
        _structure_gid_to_rigid_body_camera_gids[manager.getStructureGid()].emplace_back(gid);

    RS_LOG(_logger, debug, "addCamera takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_updateMaterial(const RS_MaterialConfig& config) {
    if (_material_name_to_gid.find(config.name) == _material_name_to_gid.end()) {
        RS_THROW(RS_InvalidConfigurationException(
            "The name of the updated material is not found: {}", config.name
        ));
    }
    RS_Timer timer;
    RIndex material_gid = _material_name_to_gid.at(config.name);
    auto& manager = _kernel_material_managers.at(material_gid);
    manager.update(config);

    for (auto& st_manager : _kernel_structure_managers) {
        /// Find dependent structures, and update materials
        if (st_manager.second.getDependentMaterialGID().count(material_gid) > 0) {
            st_manager.second.updateMaterials();
        }
    }
    RS_LOG(_logger, debug, "updateMaterial takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_deleteFloor() {
    RS_Timer timer;

    // Reset floor config
    if (_kernel_floor_manager != nullptr) {
        _kernel_floor_manager->free();
        _kernel_floor_manager = nullptr;
        RS_LOG(_logger, debug, "deleteFloor takes {}", timer.elapsedTimeString());
    }
}

void RS_KernelManager::_deleteWorld() {
    RS_Timer timer;

    if (_kernel_world_manager != nullptr) {
        _kernel_world_manager->free();
        _kernel_world_manager = nullptr;
        RS_LOG(_logger, debug, "deleteWorld takes {}", timer.elapsedTimeString());
    }
}

void RS_KernelManager::_deleteMaterial(RIndex id) {
    if (_kernel_material_managers.find(id) == _kernel_material_managers.end()) {
        RS_THROW(
            RS_InvalidConfigurationException("Cannot delete material with invalid gid {}", id)
        );
    }

    RS_Timer timer;
    auto& manager = _kernel_material_managers.at(id);

    vector<string> dependent_structures;
    for (auto& p : _kernel_structure_managers) {
        if (p.second.getDependentMaterialGID().count(id) > 0) {
            dependent_structures.push_back(p.second.getConfig().name);
        }
    }
    if (!dependent_structures.empty())
        RS_THROW(RS_InvalidConfigurationException(
            "Material {} cannot be deleted because it is used by structures: {}",
            id,
            fmt::join(dependent_structures, ", ")
        ));

    manager.free();

    string name = manager.getName();
    _material_name_to_gid.erase(name);
    _kernel_material_managers.erase(id);

    RS_LOG(_logger, debug, "deleteMaterial takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_deleteRecord(RIndex id) {
    if (_record_gid_to_record_type.find(id) == _record_gid_to_record_type.end()) {
        RS_THROW(RS_InvalidConfigurationException("Cannot delete record with invalid gid {}", id));
    }

    RS_Timer timer;
    if (_record_gid_to_record_type.at(id) == RSE_RECORD_STRUCTURE) {
        if (_kernel_structure_record_managers.find(id) == _kernel_structure_record_managers.end()) {
            RS_THROW(RS_InvalidConfigurationException(
                "Cannot delete structure record with invalid gid {}", id
            ));
        }

        auto& manager = _kernel_structure_record_managers.at(id);

        string name = manager.getName();
        RIndex structure_gid = manager.getStructureGid();
        manager.free();

        _record_name_to_gid.erase(name);
        _kernel_structure_record_managers.erase(id);
        _structure_gid_to_structure_record_gid.erase(structure_gid);
    }

    RS_LOG(_logger, debug, "deleteRecord takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_deleteSignal(RIndex id) {
    RS_Timer timer;

    if (_kernel_force_managers.find(id) != _kernel_force_managers.end()) {
        auto& manager = _kernel_force_managers.at(id);
        string name = manager.getName();
        manager.free();

        _signal_name_to_gid.erase(name);
        _kernel_force_managers.erase(id);
    } else if (_kernel_torque_managers.find(id) != _kernel_torque_managers.end()) {
        auto& manager = _kernel_torque_managers.at(id);
        string name = manager.getName();
        manager.free();

        _signal_name_to_gid.erase(name);
        _kernel_torque_managers.erase(id);
    } else {
        RS_THROW(RS_InvalidConfigurationException("Cannot delete signal with invalid gid {}", id));
    }

    RS_LOG(_logger, debug, "deleteSignal takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_deleteStructure(RIndex id) {
    if (_kernel_structure_managers.find(id) == _kernel_structure_managers.end()) {
        RS_THROW(
            RS_InvalidConfigurationException("Cannot delete structure with invalid gid {}", id)
        );
    }

    RS_Timer timer;

    auto& manager = _kernel_structure_managers.at(id);

    string name = manager.getName();
    manager.free();

    _structure_name_to_gid.erase(name);
    _kernel_structure_managers.erase(id);

    RS_LOG(_logger, debug, "deleteStructure takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::_deleteCamera(RIndex id) {
    if (_kernel_camera_managers.find(id) == _kernel_camera_managers.end()) {
        RS_THROW(RS_InvalidConfigurationException("Cannot delete camera with invalid gid {}", id));
    }

    RS_Timer timer;

    auto& manager = _kernel_camera_managers.at(id);

    if (manager.getConfig().type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE) {
        auto& current_associated_gids =
            _structure_gid_to_rigid_body_camera_gids[manager.getStructureGid()];
        current_associated_gids.erase(
            std::find(current_associated_gids.begin(), current_associated_gids.end(), id)
        );
    }

    string name = manager.getName();
    manager.free();

    _camera_name_to_gid.erase(name);
    _kernel_camera_managers.erase(id);

    RS_LOG(_logger, debug, "deleteCamera takes {}", timer.elapsedTimeString());
}

}  // namespace rise
