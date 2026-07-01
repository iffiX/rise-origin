// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <algorithm>
#include <set>

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_constant_memory.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rise_origin/utils/rs_system.h"
#include "rise_origin/utils/string/rs_string.h"
#include "rise_origin/utils/time/rs_timer.h"
#include "rs_kernel_manager.h"

using namespace std;
using namespace rise::rsc;
using namespace rise::logic;
using namespace rise::device;

namespace rise {

RS_KernelManager::RS_KernelManager() {
    // Initialize the method lookup table
    _methods = {
        {"pauseKernel", [this](RS_Any arg, RIndex message_id) { _pauseKernel(); }},
        {"resumeKernel", [this](RS_Any arg, RIndex message_id) { _resumeKernel(); }},
        {"terminateKernel", [this](RS_Any arg, RIndex message_id) { _terminateKernel(); }},
        {"beginModificationTransaction",
         [this](RS_Any arg, RIndex message_id) { _beginModificationTransaction(); }},
        {"endModificationTransaction",
         [this](RS_Any arg, RIndex message_id) { _endModificationTransaction(); }},

        {"setSimulationRSC",
         [this](RS_Any arg, RIndex message_id) {
             _setSimulationRSC(arg.as<RSC_Simulation>(), message_id);
         }},

        {"addResource",
         [this](RS_Any arg, RIndex message_id) {
             _addResource(arg.as<RSC_ResourceVariant>(), message_id);
         }},
        {"updateResource",
         [this](RS_Any arg, RIndex message_id) {
             auto& p = arg.as<std::pair<std::string, RSC_ResourceVariant>>();
             _updateResource(p.first, p.second, message_id);
         }},
        {"removeResource",
         [this](RS_Any arg, RIndex message_id) {
             _removeResource(arg.as<std::string>(), message_id);
         }},

        {"addEssence",
         [this](RS_Any arg, RIndex message_id) {
             _addEssence(arg.as<RSC_EssenceVariant>(), message_id);
         }},
        {"updateEssence",
         [this](RS_Any arg, RIndex message_id) {
             auto& p = arg.as<std::pair<std::string, RSC_EssenceVariant>>();
             _updateEssence(p.first, p.second, message_id);
         }},
        {"removeEssence",
         [this](RS_Any arg, RIndex message_id) {
             _removeEssence(arg.as<std::string>(), message_id);
         }},

        {"addEntity",
         [this](RS_Any arg, RIndex message_id) {
             _addEntity(arg.as<RSC_EntityVariant>(), message_id);
         }},
        {"removeEntity",
         [this](RS_Any arg, RIndex message_id) {
             _removeEntity(arg.as<std::string>(), message_id);
         }},
        {"freezeEntity",
         [this](RS_Any arg, RIndex message_id) {
             _freezeEntity(arg.as<std::string>(), message_id);
         }},
        {"unfreezeEntity",
         [this](RS_Any arg, RIndex message_id) {
             _unfreezeEntity(arg.as<std::string>(), message_id);
         }},

        {"addRecord",
         [this](RS_Any arg, RIndex message_id) {
             _addRecord(arg.as<RSC_RecordVariant>(), message_id);
         }},
        {"removeRecord",
         [this](RS_Any arg, RIndex message_id) {
             _removeRecord(arg.as<std::string>(), message_id);
         }},

        {"addSignal",
         [this](RS_Any arg, RIndex message_id) {
             _addSignal(arg.as<RSC_SignalVariant>(), message_id);
         }},
        {"removeSignal", [this](RS_Any arg, RIndex message_id) {
             _removeSignal(arg.as<std::string>(), message_id);
         }},
    };
}

void RS_KernelManager::init(
    RS_MemoryProvider* host_provider,
    RS_MemoryProvider* device_provider,
    RSize record_buffer_frame_num,
    std::function<void(const RS_WorldMessage&)> message_sink
) {
    if (logging::get("root")) _logger = logging::get("root")->clone("kernel manager");
    _thread_pool = make_shared<RS_ThreadPool>(RS_MIN(std::thread::hardware_concurrency(), 32));
    _host_provider = host_provider;
    _device_provider = device_provider;
    _kernel_record_buffer_frame_num = record_buffer_frame_num;
    _message_sink = std::move(message_sink);
    _kernel_synchronization_thread =
        make_unique<thread>(&RS_KernelManager::_kernelSynchronizationThreadFunc, this);

    // Budget the shared structure build cache at a tenth of the host memory free at startup.
    RS_KernelStructureVoxelManager::setBuildCacheByteSize(RS_System::getAvailableMemory() / 10);
}

void RS_KernelManager::free() {
    // Not inited
    if (!_thread_pool) return;
    // Drop cached body builds while the host provider that backs their hashmaps is still alive, so
    // the scoped providers do not outlive it and free into dangling memory at process exit.
    RS_KernelStructureVoxelManager::clearBuildCache();
    _thread_pool = nullptr;
    _device_provider = nullptr;
    _host_provider = nullptr;
    // Signal the synchronization thread to terminate
    _command_queue_cv.notify_one();
    _kernel_synchronization_thread->join();
    _kernel_synchronization_thread = nullptr;
}

void RS_KernelManager::initKernel(RSE_FloatType context_mem_accuracy) {
    RS_Timer timer;

    markKernelModified();
    kernel.host_data = new RS_KernelHostData();
    kernel.ctx_accuracy = context_mem_accuracy;

    _kernel_spatial_manager.init(this);
    _kernel_texture_manager.init(this);
    _kernel_material_manager.init(this);
    // Cosmic background manager is initialised lazily when an
    // RSC_CosmicBackground entity is added through the handle.

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

        k.ctx.structure_voxel_dynamic_flags.init(0, _device_provider);
        k.ctx.structure_voxel_dynamic_prefix_sums.init(0, _device_provider);
        k.ctx.structure_voxel_bvh_dynamic_flags.init(0, _device_provider);
        k.ctx.structure_voxel_bvh_dynamic_prefix_sums.init(0, _device_provider);
        k.ctx.structure_voxel_static_flags.init(0, _device_provider);
        k.ctx.structure_voxel_static_prefix_sums.init(0, _device_provider);

        k.ctx.structure_signals.init(0, _device_provider);

        k.ctx.structure_link_material_layout.init(_device_provider);
        k.ctx.structure_voxel_material_layout.init(_device_provider);
        k.ctx.structure_link_layout.init(_device_provider);
        k.ctx.structure_voxel_layout.init(_device_provider);
        k.ctx.structure_rigid_body_layout.init(_device_provider);
        k.ctx.structure_rigid_body_surface_voxel_layout.init(_device_provider);
        k.ctx.structure_constraint_joint_layout.init(_device_provider);

        k.ctx.structure_constraint_joint_block_range_layout.init(_device_provider);

        k.ctx.structure_signal_layout.init(_device_provider);

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
        k.ctx.camera_aux_chain_terminal_type.init(0, _device_provider);
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
        k.ctx.camera_prev_denoiser_color_history.init(0, _device_provider);
        k.ctx.camera_prev_denoiser_moment_history.init(0, _device_provider);
        k.ctx.camera_prev_denoiser_history_length.init(0, _device_provider);
        k.ctx.camera_denoiser_variance.init(0, _device_provider);
        k.ctx.camera_next_denoiser_variance.init(0, _device_provider);

        k.ctx.camera_signals.init(0, _device_provider);

        k.ctx.camera_layout.init(_device_provider);
        k.ctx.camera_info_ray_layout.init(_device_provider);
        k.ctx.camera_rgb_ray_layout.init(_device_provider);
        k.ctx.camera_record_layout.init(_device_provider);
        k.ctx.camera_rgb_record_layout.init(_device_provider);
        k.ctx.camera_depth_record_layout.init(_device_provider);
        k.ctx.camera_normal_record_layout.init(_device_provider);
        k.ctx.camera_motion_record_layout.init(_device_provider);
        k.ctx.camera_entity_info_record_layout.init(_device_provider);
        k.ctx.camera_state_record_layout.init(_device_provider);

        k.ctx.camera_signal_layout.init(_device_provider);

        k.ctx.vision_materials.init(0, _device_provider);
        k.ctx.vision_material_layout.init(_device_provider);

        k.ctx.light_weight.init(0, _device_provider);
        k.ctx.light_cdf.init(0, _device_provider);

        k.ctx.intersect_regions.init(0, _device_provider);
        k.ctx.intersect_region_counts.init(0, _device_provider);
        k.ctx.intersect_region_offsets.init(0, _device_provider);
        k.ctx.intersect_region_pairs.init(0, _device_provider);
        k.ctx.intersect_region_unique_pairs.init(0, _device_provider);
        k.ctx.intersect_region_keys.init(0, _device_provider);
        k.ctx.intersect_region_sorted_keys.init(0, _device_provider);
        k.ctx.intersect_region_indices.init(0, _device_provider);
        k.ctx.intersect_region_sorted_indices.init(0, _device_provider);
        k.ctx.intersect_region_flags.init(0, _device_provider);
        k.ctx.intersect_region_prefix_sums.init(0, _device_provider);

        k.ctx.force_signals.init(0, _device_provider);
        k.ctx.force_signal_layout.init(_device_provider);
        k.ctx.force_rigid.init(0, _device_provider);
        k.ctx.force_soft.init(0, _device_provider);

        k.ctx.torque_signals.init(0, _device_provider);
        k.ctx.torque_signal_layout.init(_device_provider);
        k.ctx.torque_rigid.init(0, _device_provider);

        k.ctx.transform_signals.init(0, _device_provider);
        k.ctx.transform_signal_layout.init(_device_provider);
        k.ctx.transform_mem.init(0, _device_provider);

        k.ctx.host_data = new RS_ContextHostData();

        k.ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
            .associateInputLayout(k.ctx.host_data->structure_rigid_body_surface_voxel_layout);
        k.ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
            .associateOutputLayout(k.ctx.host_data->structure_rigid_body_layout);

        _configureLayoutGrowthPolicies(k.ctx.host_data);
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

    // Coarse teardown: destroy the kernel data managers. Their destructors release local memory
    // and references without touching the context, which is freed raw below.
    _kernel_force_managers.clear();
    _kernel_torque_managers.clear();
    _kernel_transform_managers.clear();
    _kernel_intersect_region_managers.clear();
    _kernel_structure_voxel_managers.clear();
    _kernel_camera_perspective_managers.clear();
    _kernel_camera_stereo_managers.clear();
    _kernel_floor_manager.reset();

    // Cosmic background must release before the texture manager for texture dependencies.
    _kernel_cosmic_background_manager.reset();

    // Material manager frees before texture manager for texture references.
    _kernel_material_manager.free();

    _kernel_spatial_manager.free();
    _kernel_texture_manager.free();

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

        k.ctx.structure_voxel_dynamic_flags.free();
        k.ctx.structure_voxel_dynamic_prefix_sums.free();
        k.ctx.structure_voxel_bvh_dynamic_flags.free();
        k.ctx.structure_voxel_bvh_dynamic_prefix_sums.free();
        k.ctx.structure_voxel_static_flags.free();
        k.ctx.structure_voxel_static_prefix_sums.free();

        k.ctx.structure_signals.free();

        k.ctx.structure_link_material_layout.free();
        k.ctx.structure_voxel_material_layout.free();
        k.ctx.structure_link_layout.free();
        k.ctx.structure_voxel_layout.free();
        k.ctx.structure_rigid_body_layout.free();
        k.ctx.structure_rigid_body_surface_voxel_layout.free();
        k.ctx.structure_constraint_joint_layout.free();

        k.ctx.structure_constraint_joint_block_range_layout.free();

        k.ctx.structure_signal_layout.free();

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
        k.ctx.camera_aux_chain_terminal_type.free();
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
        k.ctx.camera_prev_denoiser_color_history.free();
        k.ctx.camera_prev_denoiser_moment_history.free();
        k.ctx.camera_prev_denoiser_history_length.free();
        k.ctx.camera_denoiser_variance.free();
        k.ctx.camera_next_denoiser_variance.free();

        k.ctx.camera_signals.free();

        k.ctx.camera_layout.free();
        k.ctx.camera_info_ray_layout.free();
        k.ctx.camera_rgb_ray_layout.free();
        k.ctx.camera_record_layout.free();
        k.ctx.camera_rgb_record_layout.free();
        k.ctx.camera_depth_record_layout.free();
        k.ctx.camera_normal_record_layout.free();
        k.ctx.camera_motion_record_layout.free();
        k.ctx.camera_entity_info_record_layout.free();
        k.ctx.camera_state_record_layout.free();

        k.ctx.camera_signal_layout.free();

        k.ctx.vision_materials.free();
        k.ctx.vision_material_layout.free();

        k.ctx.light_weight.free();
        k.ctx.light_cdf.free();

        k.ctx.intersect_regions.free();
        k.ctx.intersect_region_counts.free();
        k.ctx.intersect_region_offsets.free();
        k.ctx.intersect_region_pairs.free();
        k.ctx.intersect_region_unique_pairs.free();
        k.ctx.intersect_region_keys.free();
        k.ctx.intersect_region_sorted_keys.free();
        k.ctx.intersect_region_indices.free();
        k.ctx.intersect_region_sorted_indices.free();
        k.ctx.intersect_region_flags.free();
        k.ctx.intersect_region_prefix_sums.free();

        k.ctx.force_signals.free();
        k.ctx.force_signal_layout.free();
        k.ctx.force_rigid.free();
        k.ctx.force_soft.free();

        k.ctx.torque_signals.free();
        k.ctx.torque_signal_layout.free();
        k.ctx.torque_rigid.free();

        k.ctx.transform_signals.free();
        k.ctx.transform_signal_layout.free();
        k.ctx.transform_mem.free();

        delete k.ctx.host_data;
    });
    delete kernel.host_data;

    RS_LOG(_logger, debug, "freeKernel takes {}", timer.elapsedTimeString());
}

void RS_KernelManager::pauseCommandExecution() {
    std::unique_lock lock(_command_execution_mutex);
    _command_execution_cv.wait(lock, [this]() { return _command_execution_depth == 0; });
    ++_command_pause_count;
}

void RS_KernelManager::resumeCommandExecution() {
    std::unique_lock lock(_command_execution_mutex);
    if (--_command_pause_count == 0) _command_execution_cv.notify_all();
}

void RS_KernelManager::acquireCriticalDataAccess() { _kernel_data_mutex.lock(); }

void RS_KernelManager::releaseCriticalDataAccess() { _kernel_data_mutex.unlock(); }

/*****************************************************************
 * Kernel Management APIs
 *****************************************************************/
void RS_KernelManager::markKernelModified() { _is_kernel_modified = true; }

bool RS_KernelManager::isKernelModified() const { return _is_kernel_modified; }

void RS_KernelManager::syncKernelIfNeeded() {
    // Nothing to push
    if (!_is_kernel_modified) return;
    // Drain the stream first so the constant-memory write cannot race in-flight launches or pending
    // device mutations.
    _device_provider->sync();
    device::RS_ConstantMemory::write(&kernel, _device_provider->device, sizeof(RS_Kernel));
    // The device now mirrors the host kernel.
    _is_kernel_modified = false;
}

/*****************************************************************
 * Simulation execution APIs
 *****************************************************************/
void RS_KernelManager::syncRecorderRecords() {
    RS_KernelManagerCommandExecutionGuard guard(this);
    for (auto& manager : _kernel_structure_voxel_managers) {
        manager.second->syncRecorderRecords();
    }
    for (auto& manager : _kernel_camera_perspective_managers) {
        manager.second->syncRecorderRecords();
    }
    for (auto& manager : _kernel_camera_stereo_managers) {
        manager.second->syncRecorderRecords();
    }
    for (auto& manager : _kernel_intersect_region_managers) {
        manager.second->syncRecorderRecords();
    }
}

void RS_KernelManager::waitForRecorderRecordsSync() {
    RS_KernelManagerCommandExecutionGuard guard(this);
    for (auto& manager : _kernel_structure_voxel_managers) {
        manager.second->waitForRecorderRecordsSync();
    }
    for (auto& manager : _kernel_camera_perspective_managers) {
        manager.second->waitForRecorderRecordsSync();
    }
    for (auto& manager : _kernel_camera_stereo_managers) {
        manager.second->waitForRecorderRecordsSync();
    }
    for (auto& manager : _kernel_intersect_region_managers) {
        manager.second->waitForRecorderRecordsSync();
    }
}

unordered_map<string, RS_KernelRecorderStructureVoxelBundle>
RS_KernelManager::getRecorderStructureVoxelRecords() {
    RS_KernelManagerCommandExecutionGuard guard(this);
    unordered_map<string, RS_KernelRecorderStructureVoxelBundle> result;

    for (auto& [st_name, st_manager] : _kernel_structure_voxel_managers) {
        string recorder_name = st_manager->getRecordRecorderName();
        if (recorder_name.empty()) continue;
        result[recorder_name].structure_records[st_manager->getEntityName()] =
            st_manager->getRecorderRecords();
    }
    return result;
}

unordered_map<string, RS_KernelRecorderCameraPerspectiveBundle>
RS_KernelManager::getRecorderCameraPerspectiveRecords() {
    RS_KernelManagerCommandExecutionGuard guard(this);
    unordered_map<string, RS_KernelRecorderCameraPerspectiveBundle> result;

    for (auto& [cam_name, cam_manager] : _kernel_camera_perspective_managers) {
        string recorder_name = cam_manager->getRecordRecorderName();
        if (recorder_name.empty()) continue;
        result[recorder_name].camera_records[cam_manager->getEntityName()] =
            cam_manager->getRecorderRecords();
    }
    return result;
}

unordered_map<string, RS_KernelRecorderCameraStereoBundle>
RS_KernelManager::getRecorderCameraStereoRecords() {
    RS_KernelManagerCommandExecutionGuard guard(this);
    unordered_map<string, RS_KernelRecorderCameraStereoBundle> result;

    for (auto& [cam_name, cam_manager] : _kernel_camera_stereo_managers) {
        string recorder_name = cam_manager->getRecordRecorderName();
        if (recorder_name.empty()) continue;
        result[recorder_name].camera_records[cam_manager->getEntityName()] =
            cam_manager->getRecorderRecords();
    }
    return result;
}

unordered_map<string, RS_KernelRecorderIntersectRegionBundle>
RS_KernelManager::getRecorderIntersectRegionRecords() {
    RS_KernelManagerCommandExecutionGuard guard(this);
    unordered_map<string, RS_KernelRecorderIntersectRegionBundle> result;

    for (auto& [region_name, region_manager] : _kernel_intersect_region_managers) {
        string recorder_name = region_manager->getRecordRecorderName();
        if (recorder_name.empty()) continue;
        result[recorder_name].intersect_region_records[region_manager->getRecordName()] =
            region_manager->getRecorderRecords();
    }
    return result;
}

void RS_KernelManager::syncControllerRecords() {
    RS_KernelManagerCommandExecutionGuard guard(this);
    for (auto& manager : _kernel_structure_voxel_managers) {
        if (manager.second->hasControllerRecord()) {
            manager.second->syncControllerRecord();
        }
    }
    for (auto& manager : _kernel_camera_perspective_managers) {
        if (manager.second->hasControllerRecord()) {
            manager.second->syncControllerRecord();
        }
    }
    for (auto& manager : _kernel_camera_stereo_managers) {
        if (manager.second->hasControllerRecord()) {
            manager.second->syncControllerRecord();
        }
    }
    for (auto& manager : _kernel_intersect_region_managers) {
        if (manager.second->hasControllerRecord()) {
            manager.second->syncControllerRecord();
        }
    }
}

void RS_KernelManager::syncControllerSignals() {
    RS_KernelManagerCommandExecutionGuard guard(this);
    for (auto& manager : _kernel_structure_voxel_managers) {
        if (manager.second->hasControllerSignal()) {
            manager.second->syncControllerSignal();
        }
    }
    for (auto& manager : _kernel_camera_perspective_managers) {
        if (manager.second->hasControllerSignal()) {
            manager.second->syncControllerSignal();
        }
    }
    for (auto& manager : _kernel_camera_stereo_managers) {
        if (manager.second->hasControllerSignal()) {
            manager.second->syncControllerSignal();
        }
    }
    for (auto& manager : _kernel_force_managers) {
        if (manager.second->hasControllerSignal()) {
            manager.second->syncControllerSignal();
        }
    }
    for (auto& manager : _kernel_torque_managers) {
        if (manager.second->hasControllerSignal()) {
            manager.second->syncControllerSignal();
        }
    }
    for (auto& manager : _kernel_transform_managers) {
        if (manager.second->hasControllerSignal()) {
            manager.second->syncControllerSignal();
        }
    }
}

unordered_map<string, RS_KernelControllerBundle> RS_KernelManager::getControllerSignalsAndRecords(
    const unordered_set<string>& controller_names
) {
    RS_KernelManagerCommandExecutionGuard guard(this);
    unordered_map<string, RS_KernelControllerBundle> result;

    // Seed registered-but-idle controllers so they still surface.
    for (auto& name : controller_names) result[name];

    // Structure voxel: signal -> its source controller; record -> its target controller.
    for (auto& [st_name, st_manager] : _kernel_structure_voxel_managers) {
        if (st_manager->hasControllerSignal()) {
            const string& cn = st_manager->getSignalControllerName();
            if (!cn.empty()) result[cn].signals.emplace_back(st_manager->getControllerSignal());
        }
        if (st_manager->hasControllerRecord()) {
            const string& cn = st_manager->getRecordControllerName();
            if (!cn.empty()) result[cn].records.emplace_back(st_manager->getControllerRecord());
        }
    }

    // Camera perspective: free cameras carry a signal; all cameras may emit a record.
    for (auto& [cam_name, cam_manager] : _kernel_camera_perspective_managers) {
        if (cam_manager->hasControllerSignal()) {
            const string& cn = cam_manager->getSignalControllerName();
            if (!cn.empty()) result[cn].signals.emplace_back(cam_manager->getControllerSignal());
        }
        if (cam_manager->hasControllerRecord()) {
            const string& cn = cam_manager->getRecordControllerName();
            if (!cn.empty()) result[cn].records.emplace_back(cam_manager->getControllerRecord());
        }
    }

    // Camera stereo: free cameras carry a signal; all cameras may emit a record.
    for (auto& [cam_name, cam_manager] : _kernel_camera_stereo_managers) {
        if (cam_manager->hasControllerSignal()) {
            const string& cn = cam_manager->getSignalControllerName();
            if (!cn.empty()) result[cn].signals.emplace_back(cam_manager->getControllerSignal());
        }
        if (cam_manager->hasControllerRecord()) {
            const string& cn = cam_manager->getRecordControllerName();
            if (!cn.empty()) result[cn].records.emplace_back(cam_manager->getControllerRecord());
        }
    }

    // Force / torque: signal -> its source controller.
    for (auto& [_, fm] : _kernel_force_managers) {
        if (fm->hasControllerSignal()) {
            const string& cn = fm->getRSC().as<RSC_Force>().source_controller_reference;
            if (!cn.empty()) result[cn].signals.emplace_back(fm->getControllerSignal());
        }
    }
    for (auto& [_, tm] : _kernel_torque_managers) {
        if (tm->hasControllerSignal()) {
            const string& cn = tm->getRSC().as<RSC_Torque>().source_controller_reference;
            if (!cn.empty()) result[cn].signals.emplace_back(tm->getControllerSignal());
        }
    }
    for (auto& [_, xm] : _kernel_transform_managers) {
        if (xm->hasControllerSignal()) {
            const string& cn = xm->getRSC().as<RSC_Transform>().source_controller_reference;
            if (!cn.empty()) result[cn].signals.emplace_back(xm->getControllerSignal());
        }
    }

    // Intersect region: record -> its target controller.
    for (auto& [region_name, region_manager] : _kernel_intersect_region_managers) {
        if (region_manager->hasControllerRecord()) {
            const string& cn = region_manager->getRecordControllerName();
            if (!cn.empty()) result[cn].records.emplace_back(region_manager->getControllerRecord());
        }
    }

    return result;
}

void RS_KernelManager::finalizeTransformSignals() {
    for (auto& manager : _kernel_transform_managers) {
        if (manager.second->isStaticRebuildPending()) markStaticVoxelMoved();
        manager.second->clearStaticRebuildPending();
    }
}

void RS_KernelManager::setIntersectRegionRecords(
    const std::vector<logic::RS_IntersectRegionPair>& pairs
) {
    // Critical data access, not command execution: this writes manager-owned record buffers and is
    // called from the executor while it already holds critical data access. Taking the command
    // execution guard here would acquire it under the data guard and deadlock a concurrent command.
    RS_KernelManagerCriticalDataAccessGuard guard(this);

    unordered_map<RIndex, RS_KernelIntersectRegionManager*> manager_by_gid;
    for (auto& [_, rm] : _kernel_intersect_region_managers) {
        manager_by_gid[rm->getRecordGid()] = rm.get();
    }

    // Walk the record-gid-major pairs once; each contiguous run is one record's membership set.
    unordered_set<RIndex> filled;
    RSize n = pairs.size();
    RSize i = 0;
    while (i < n) {
        RIndex gid = pairs[i].record_gid;
        RSize start = i;
        while (i < n && pairs[i].record_gid == gid) i++;
        if (auto it = manager_by_gid.find(gid); it != manager_by_gid.end()) {
            it->second->setIntersectRegionRecord(pairs, start, i - start);
            filled.insert(gid);
        }
    }

    // Records with no membership this step get an empty set.
    for (auto& [gid, manager] : manager_by_gid) {
        if (filled.find(gid) == filled.end()) {
            manager->setIntersectRegionRecord(pairs, 0, 0);
        }
    }
}

std::string RS_KernelManager::purgeStructure(RIndex structure_gid) {
    RS_KernelManagerCommandExecutionGuard guard(this);

    // Resolve the structure name from its gid.
    std::string name;
    for (auto& [structure_name, mgr] : _kernel_structure_voxel_managers) {
        if (mgr->getEntityGid() == structure_gid) {
            name = structure_name;
            break;
        }
    }
    if (name.empty())
        RS_THROW(RS_NotFoundException("Structure with gid {} not found", structure_gid));

    // Gather the dependents that hold a reference on this structure.
    std::vector<std::string> dependent_signals;
    std::vector<std::string> dependent_cameras;
    std::vector<std::string> dependent_records;

    for (auto& [key, mgr] : _kernel_force_managers) {
        if (mgr->getStructureGid() == structure_gid)
            dependent_signals.push_back(mgr->getSignalName());
    }
    for (auto& [key, mgr] : _kernel_torque_managers) {
        if (mgr->getStructureGid() == structure_gid)
            dependent_signals.push_back(mgr->getSignalName());
    }
    for (auto& [key, mgr] : _kernel_transform_managers) {
        if (mgr->getStructureGid() == structure_gid)
            dependent_signals.push_back(mgr->getSignalName());
    }
    for (auto& [key, mgr] : _kernel_camera_perspective_managers) {
        if (mgr->getStructureGid() == structure_gid)
            dependent_cameras.push_back(mgr->getEntityName());
    }
    for (auto& [key, mgr] : _kernel_camera_stereo_managers) {
        if (mgr->getStructureGid() == structure_gid)
            dependent_cameras.push_back(mgr->getEntityName());
    }
    for (auto& [key, mgr] : _kernel_intersect_region_managers) {
        if (mgr->getStructureGid() == structure_gid)
            dependent_records.push_back(mgr->getRecordName());
    }

    RS_LOG(
        _logger,
        warn,
        "Purging structure name: {}, gid: {}, dependent signals: [{}], cameras: [{}], records: "
        "[{}]",
        name,
        structure_gid,
        join(", ", dependent_signals),
        join(", ", dependent_cameras),
        join(", ", dependent_records)
    );

    // Tear down the dependents, then the structure itself.
    for (auto& signal_name : dependent_signals) {
        _removeSignal(signal_name, RS_NULL_INDEX);
        RS_LOG(_logger, warn, "Removed signal: {}", signal_name);
    }
    for (auto& camera_name : dependent_cameras) {
        _removeEntity(camera_name, RS_NULL_INDEX);
        RS_LOG(_logger, warn, "Removed camera: {}", camera_name);
    }
    for (auto& record_name : dependent_records) {
        _removeRecord(record_name, RS_NULL_INDEX);
        RS_LOG(_logger, warn, "Removed record: {}", record_name);
    }
    _removeEntity(name, RS_NULL_INDEX);
    RS_LOG(_logger, warn, "Removed structure: {}", name);

    // Flush the pending changes to the kernel data.
    _flushPendingCommits();

    return name;
}

void RS_KernelManager::markVisionRestirReservoirStale() {
    kernel.vision_is_restir_reservoir_stale = true;
    markKernelModified();
}

bool RS_KernelManager::isVisionRestirReservoirStale() const {
    return kernel.vision_is_restir_reservoir_stale;
}

void RS_KernelManager::clearVisionRestirReservoirStale() {
    kernel.vision_is_restir_reservoir_stale = false;
    markKernelModified();
}

void RS_KernelManager::markVoxelPartitionModified() {
    _kernel_spatial_manager.markVoxelPartitionModified();
}

bool RS_KernelManager::isVoxelPartitionModified() const {
    return _kernel_spatial_manager.isVoxelPartitionModified();
}

void RS_KernelManager::clearVoxelPartitionModified() {
    _kernel_spatial_manager.clearVoxelPartitionModified();
}

void RS_KernelManager::markStaticVoxelMoved() { _kernel_spatial_manager.markStaticVoxelMoved(); }

bool RS_KernelManager::isStaticVoxelMoved() const {
    return _kernel_spatial_manager.isStaticVoxelMoved();
}

void RS_KernelManager::clearStaticVoxelMoved() { _kernel_spatial_manager.clearStaticVoxelMoved(); }

void RS_KernelManager::markFloorTriangleModified() {
    _kernel_spatial_manager.markFloorTriangleModified();
}

bool RS_KernelManager::isFloorTriangleModified() const {
    return _kernel_spatial_manager.isFloorTriangleModified();
}

void RS_KernelManager::clearFloorTriangleModified() {
    _kernel_spatial_manager.clearFloorTriangleModified();
}

/*****************************************************************
 * Handle APIs
 *****************************************************************/
std::string RS_KernelManager::getInfo(const std::string& info_name) {
    if (info_name == "device_memory_used")
        return _device_provider ? _device_provider->getProperty("memory_used") : "";
    if (info_name == "device_memory_capacity")
        return _device_provider ? _device_provider->getProperty("memory_capacity") : "";
    if (info_name == "host_memory_used") return std::to_string(RS_System::getUsedMemory());
    if (info_name == "host_memory_capacity") return std::to_string(RS_System::getTotalMemory());
    if (info_name == "device_name") {
        if (!_device_provider) return "";
        RS_DeviceLocation location = _device_provider->device;
        const char* type_name = "unknown";
        if (location.type == RSE_DEVICE_TYPE_CUDA) type_name = "cuda";
        else if (location.type == RSE_DEVICE_TYPE_CPU) type_name = "cpu";
        return std::string(type_name) + ":" + std::to_string(location.id);
    }
    return "";
}

void RS_KernelManager::pauseKernel(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(RS_NULL_INDEX, "pauseKernel", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _pauseKernel();
    }
}

void RS_KernelManager::resumeKernel(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(RS_NULL_INDEX, "resumeKernel", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _resumeKernel();
    }
}

void RS_KernelManager::terminateKernel(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(RS_NULL_INDEX, "terminateKernel", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _terminateKernel();
    }
}

void RS_KernelManager::beginModificationTransaction(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(RS_NULL_INDEX, "beginModificationTransaction", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _beginModificationTransaction();
    }
}

void RS_KernelManager::endModificationTransaction(bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(RS_NULL_INDEX, "endModificationTransaction", 0);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _endModificationTransaction();
    }
}

void RS_KernelManager::setSimulationRSC(
    const RSC_Simulation& simulation,
    RIndex message_id,
    bool sync
) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "setSimulationRSC", simulation);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _setSimulationRSC(simulation, message_id);
    }
}

const RSC_Simulation& RS_KernelManager::getSimulationRSC() const { return _simulation_rsc; }

void RS_KernelManager::addResource(
    const RSC_ResourceVariant& resource,
    RIndex message_id,
    bool sync
) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "addResource", resource);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _addResource(resource, message_id);
    }
}

void RS_KernelManager::updateResource(
    const std::string& name,
    const RSC_ResourceVariant& resource,
    RIndex message_id,
    bool sync
) {
    if (kernel.isTerminated()) return;
    auto payload = std::make_pair(name, resource);
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "updateResource", payload);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _updateResource(name, resource, message_id);
    }
}

void RS_KernelManager::removeResource(const std::string& name, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "removeResource", name);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _removeResource(name, message_id);
    }
}

RSC_ResourceVariant RS_KernelManager::findResource(const std::string& name) {
    RS_KernelManagerCommandExecutionGuard guard(this);
    auto cfg = _kernel_texture_manager.findTextureRSC(name);
    return cfg.visit([](auto& c) { return RSC_ResourceVariant(c); });
}

void RS_KernelManager::addEssence(const RSC_EssenceVariant& essence, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "addEssence", essence);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _addEssence(essence, message_id);
    }
}

void RS_KernelManager::updateEssence(
    const std::string& name,
    const RSC_EssenceVariant& essence,
    RIndex message_id,
    bool sync
) {
    if (kernel.isTerminated()) return;
    auto payload = std::make_pair(name, essence);
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "updateEssence", payload);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _updateEssence(name, essence, message_id);
    }
}

void RS_KernelManager::removeEssence(const std::string& name, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "removeEssence", name);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _removeEssence(name, message_id);
    }
}

RSC_EssenceVariant RS_KernelManager::findEssence(const std::string& name) {
    RS_KernelManagerCommandExecutionGuard guard(this);
    // Material essence is the only kind of essence currently materialised on
    // the kernel; the material manager owns the name->data lookup.
    if (_kernel_material_manager.findMaterial(name) != RS_NULL_INDEX) {
        const auto* mat = _kernel_material_manager.getMaterialByName(name);
        if (mat != nullptr) return RSC_EssenceVariant(*mat);
    }
    RS_THROW(RS_NotFoundException("Essence '{}' not found", name));
}

void RS_KernelManager::addEntity(const RSC_EntityVariant& entity, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "addEntity", entity);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _addEntity(entity, message_id);
    }
}

void RS_KernelManager::removeEntity(const std::string& name, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "removeEntity", name);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _removeEntity(name, message_id);
    }
}

void RS_KernelManager::freezeEntity(const std::string& name, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "freezeEntity", name);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _freezeEntity(name, message_id);
    }
}

void RS_KernelManager::unfreezeEntity(const std::string& name, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "unfreezeEntity", name);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _unfreezeEntity(name, message_id);
    }
}

bool RS_KernelManager::isEntityFrozen(const std::string& name) {
    RS_KernelManagerCommandExecutionGuard guard(this);
    if (auto it = _kernel_structure_voxel_managers.find(name);
        it != _kernel_structure_voxel_managers.end())
        return it->second->isFrozen();
    if (auto it = _kernel_camera_perspective_managers.find(name);
        it != _kernel_camera_perspective_managers.end())
        return it->second->isFrozen();
    if (auto it = _kernel_camera_stereo_managers.find(name);
        it != _kernel_camera_stereo_managers.end())
        return it->second->isFrozen();
    if (_kernel_floor_manager) {
        bool matched = false;
        _kernel_floor_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) return _kernel_floor_manager->isFrozen();
    }
    if (_kernel_cosmic_background_manager) {
        bool matched = false;
        _kernel_cosmic_background_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) return _kernel_cosmic_background_manager->isFrozen();
    }
    RS_THROW(RS_NotFoundException("Entity '{}' not found", name));
}

RSC_EntityVariant RS_KernelManager::findEntity(const std::string& name) {
    RS_KernelManagerCommandExecutionGuard guard(this);
    if (auto it = _kernel_structure_voxel_managers.find(name);
        it != _kernel_structure_voxel_managers.end())
        return it->second->getRSC().visit([](auto& c) { return RSC_EntityVariant(c); });
    if (auto it = _kernel_camera_perspective_managers.find(name);
        it != _kernel_camera_perspective_managers.end()) {
        return it->second->getRSC().visit([](auto& c) { return RSC_EntityVariant(c); });
    }
    if (auto it = _kernel_camera_stereo_managers.find(name);
        it != _kernel_camera_stereo_managers.end()) {
        return it->second->getRSC().visit([](auto& c) { return RSC_EntityVariant(c); });
    }
    if (_kernel_floor_manager) {
        RSC_EntityVariant result;
        bool found = false;
        _kernel_floor_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) {
                result = RSC_EntityVariant(c);
                found = true;
            }
        });
        if (found) return result;
    }
    if (_kernel_cosmic_background_manager) {
        RSC_EntityVariant result;
        bool found = false;
        _kernel_cosmic_background_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) {
                result = RSC_EntityVariant(c);
                found = true;
            }
        });
        if (found) return result;
    }
    RS_THROW(RS_NotFoundException("Entity '{}' not found", name));
}

RIndex RS_KernelManager::findEntityGID(const std::string& name) {
    RS_KernelManagerCommandExecutionGuard guard(this);
    if (_kernel_floor_manager) {
        bool matched = false;
        _kernel_floor_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) return RS_WORLD_RSC_FLOOR_GID;
    }
    if (_kernel_cosmic_background_manager) {
        bool matched = false;
        _kernel_cosmic_background_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) return RS_WORLD_RSC_COSMIC_BACKGROUND_GID;
    }
    if (auto it = _entity_gids.find(name); it != _entity_gids.end()) return it->second;
    return RS_NULL_INDEX;
}

std::string RS_KernelManager::findEntityName(RIndex gid) {
    RS_KernelManagerCommandExecutionGuard guard(this);
    if (gid == RS_WORLD_RSC_FLOOR_GID && _kernel_floor_manager) {
        std::string name;
        _kernel_floor_manager->getRSC().visit([&](auto& c) { name = c.name; });
        return name;
    }
    if (gid == RS_WORLD_RSC_COSMIC_BACKGROUND_GID && _kernel_cosmic_background_manager) {
        std::string name;
        _kernel_cosmic_background_manager->getRSC().visit([&](auto& c) { name = c.name; });
        return name;
    }
    if (auto it = _entity_names.find(gid); it != _entity_names.end()) return it->second;
    return "";
}

void RS_KernelManager::addRecord(const RSC_RecordVariant& record, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "addRecord", record);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _addRecord(record, message_id);
    }
}

void RS_KernelManager::removeRecord(const std::string& name, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "removeRecord", name);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _removeRecord(name, message_id);
    }
}

RSC_RecordVariant RS_KernelManager::findRecord(const std::string& name) {
    RS_KernelManagerCommandExecutionGuard guard(this);
    for (auto& [_, rm] : _kernel_intersect_region_managers) {
        if (rm->getRecordName() == name) {
            return rm->getRSC().visit([](auto& c) { return RSC_RecordVariant(c); });
        }
    }
    RS_THROW(RS_NotFoundException("Record '{}' not found", name));
}

void RS_KernelManager::addSignal(const RSC_SignalVariant& signal, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "addSignal", signal);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _addSignal(signal, message_id);
    }
}

void RS_KernelManager::removeSignal(const std::string& name, RIndex message_id, bool sync) {
    if (kernel.isTerminated()) return;
    if (!sync) {
        lock_guard lock(_command_queue_mutex);
        _command_queue.emplace(message_id, "removeSignal", name);
        _command_queue_cv.notify_one();
    } else {
        RS_KernelManagerCommandExecutionGuard guard(this);
        _removeSignal(name, message_id);
    }
}

RSC_SignalVariant RS_KernelManager::findSignal(const std::string& name) {
    RS_KernelManagerCommandExecutionGuard guard(this);
    for (auto& [_, fm] : _kernel_force_managers) {
        if (fm->getSignalName() == name) {
            return RSC_SignalVariant(fm->getRSC().as<RSC_Force>());
        }
    }
    for (auto& [_, tm] : _kernel_torque_managers) {
        if (tm->getSignalName() == name) {
            return RSC_SignalVariant(tm->getRSC().as<RSC_Torque>());
        }
    }
    for (auto& [_, xm] : _kernel_transform_managers) {
        if (xm->getSignalName() == name) {
            return RSC_SignalVariant(xm->getRSC().as<RSC_Transform>());
        }
    }
    RS_THROW(RS_NotFoundException("Signal '{}' not found", name));
}

RSC_World RS_KernelManager::getWorld() {
    if (kernel.isTerminated()) return {};
    RS_KernelManagerCommandExecutionGuard guard(this);

    RSC_World world;
    world.simulation = _simulation_rsc;

    // Entities
    for (auto& [_, sm] : _kernel_structure_voxel_managers) {
        sm->getRSC().visit([&](auto& c) { world.entities.emplace_back(c); });
    }
    for (auto& [_, cm] : _kernel_camera_perspective_managers) {
        cm->getRSC().visit([&](auto& c) { world.entities.emplace_back(c); });
    }
    for (auto& [_, cm] : _kernel_camera_stereo_managers) {
        cm->getRSC().visit([&](auto& c) { world.entities.emplace_back(c); });
    }
    if (_kernel_floor_manager) {
        _kernel_floor_manager->getRSC().visit([&](auto& c) { world.entities.emplace_back(c); });
    }
    if (_kernel_cosmic_background_manager) {
        _kernel_cosmic_background_manager->getRSC().visit([&](auto& c) {
            world.entities.emplace_back(c);
        });
    }

    // Essences (materials + unary properties + pair properties).
    for (auto& m : _kernel_material_manager.getMaterialRSCs()) {
        world.essences.emplace_back(m);
    }
    for (auto& p : _kernel_material_manager.getMaterialUnaryPropertyRSCs()) {
        p.visit([&](auto& c) { world.essences.emplace_back(c); });
    }
    for (auto& p : _kernel_material_manager.getMaterialPairPropertyRSCs()) {
        p.visit([&](auto& c) { world.essences.emplace_back(c); });
    }

    // Resources (textures).
    for (auto& r : _kernel_texture_manager.getTextureRSCs()) {
        r.visit([&](auto& c) { world.resources.emplace_back(c); });
    }

    // Signals (free-standing force / torque). Controllers and recorders live on
    // RS_Simulation and are merged in by the handle's getWorld().
    for (auto& [_, fm] : _kernel_force_managers) {
        world.signals.emplace_back(fm->getRSC().as<RSC_Force>());
    }
    for (auto& [_, tm] : _kernel_torque_managers) {
        world.signals.emplace_back(tm->getRSC().as<RSC_Torque>());
    }
    for (auto& [_, xm] : _kernel_transform_managers) {
        world.signals.emplace_back(xm->getRSC().as<RSC_Transform>());
    }

    return world;
}

/****************************************************************
 * Data manager APIs
 ****************************************************************/
RS_MemoryProvider* RS_KernelManager::getHostProvider() const { return _host_provider; }

RS_MemoryProvider* RS_KernelManager::getDeviceProvider() const { return _device_provider; }

shared_ptr<logging::logger> RS_KernelManager::getLogger() const { return _logger; }

std::shared_ptr<RS_ThreadPool> RS_KernelManager::getThreadPool() const { return _thread_pool; }

RSize RS_KernelManager::getKernelRecordBufferFrameNum() const {
    return _kernel_record_buffer_frame_num;
}

RIndex RS_KernelManager::getEntityCounter() const { return _entity_counter; }
RIndex RS_KernelManager::getEssenceCounter() const { return _essence_counter; }
RIndex RS_KernelManager::getResourceCounter() const { return _resource_counter; }
RIndex RS_KernelManager::getRecordCounter() const { return _record_counter; }
RIndex RS_KernelManager::getSignalCounter() const { return _signal_counter; }

namespace {
RIndex registerByName(
    std::unordered_map<std::string, RIndex>& map,
    RIndex& counter,
    const std::string& name,
    const char* kind
) {
    auto [it, inserted] = map.emplace(name, counter);
    if (!inserted) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Duplicate {} name '{}' (already registered with gid {})", kind, name, it->second
        ));
    }
    return counter++;
}

void deregisterByName(
    std::unordered_map<std::string, RIndex>& map,
    const std::string& name,
    const char* kind
) {
    auto it = map.find(name);
    if (it == map.end()) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Cannot deregister {} '{}': not registered", kind, name
        ));
    }
    map.erase(it);
}
}  // namespace

RIndex RS_KernelManager::registerEntity(const string& name) {
    RIndex gid = registerByName(_entity_gids, _entity_counter, name, "entity");
    _entity_names.emplace(gid, name);
    return gid;
}
RIndex RS_KernelManager::registerEssence(const string& name) {
    return registerByName(_essence_gids, _essence_counter, name, "essence");
}
RIndex RS_KernelManager::registerResource(const string& name) {
    return registerByName(_resource_gids, _resource_counter, name, "resource");
}
RIndex RS_KernelManager::registerRecord(const string& name) {
    return registerByName(_record_gids, _record_counter, name, "record");
}
RIndex RS_KernelManager::registerSignal(const string& name) {
    return registerByName(_signal_gids, _signal_counter, name, "signal");
}

void RS_KernelManager::deregisterEntity(const string& name) {
    if (auto it = _entity_gids.find(name); it != _entity_gids.end()) _entity_names.erase(it->second);
    deregisterByName(_entity_gids, name, "entity");
}
void RS_KernelManager::deregisterEssence(const string& name) {
    deregisterByName(_essence_gids, name, "essence");
}
void RS_KernelManager::deregisterResource(const string& name) {
    deregisterByName(_resource_gids, name, "resource");
}
void RS_KernelManager::deregisterRecord(const string& name) {
    deregisterByName(_record_gids, name, "record");
}
void RS_KernelManager::deregisterSignal(const string& name) {
    deregisterByName(_signal_gids, name, "signal");
}

// Kernel data managers — singletons
RS_KernelSpatialManager& RS_KernelManager::getKernelSpatialManager() {
    return _kernel_spatial_manager;
}

RS_KernelTextureManager& RS_KernelManager::getKernelTextureManager() {
    return _kernel_texture_manager;
}

RS_KernelMaterialManager& RS_KernelManager::getKernelMaterialManager() {
    return _kernel_material_manager;
}

const std::unique_ptr<RS_KernelCosmicBackgroundManager>&
RS_KernelManager::getKernelCosmicBackgroundManager() const {
    return _kernel_cosmic_background_manager;
}

const unique_ptr<RS_KernelFloorManager>& RS_KernelManager::getKernelFloorManager() const {
    return _kernel_floor_manager;
}

// Kernel data managers — name-keyed
unordered_map<string, unique_ptr<RS_KernelStructureVoxelManager>>&
RS_KernelManager::getKernelStructureVoxelManagers() {
    return _kernel_structure_voxel_managers;
}

unordered_map<string, unique_ptr<RS_KernelCameraPerspectiveManager>>&
RS_KernelManager::getKernelCameraPerspectiveManagers() {
    return _kernel_camera_perspective_managers;
}

unordered_map<string, unique_ptr<RS_KernelCameraStereoManager>>&
RS_KernelManager::getKernelCameraStereoManagers() {
    return _kernel_camera_stereo_managers;
}

unordered_map<string, unique_ptr<RS_KernelForceManager>>&
RS_KernelManager::getKernelForceManagers() {
    return _kernel_force_managers;
}

unordered_map<string, unique_ptr<RS_KernelTorqueManager>>&
RS_KernelManager::getKernelTorqueManagers() {
    return _kernel_torque_managers;
}

unordered_map<string, unique_ptr<RS_KernelTransformManager>>&
RS_KernelManager::getKernelTransformManagers() {
    return _kernel_transform_managers;
}

unordered_map<string, unique_ptr<RS_KernelIntersectRegionManager>>&
RS_KernelManager::getKernelIntersectRegionManagers() {
    return _kernel_intersect_region_managers;
}

bool RS_KernelManager::isInKernelDataCriticalZone() const {
    return _kernel_data_critical_zone_depth > 0;
}

void RS_KernelManager::enterKernelDataCriticalZone() {
    _kernel_data_mutex.lock();
    _kernel_data_critical_zone_depth += 1;
    // Sync to make sure all operations in memory provider stream are finished.
    if (_kernel_data_critical_zone_depth == 1) _device_provider->sync();
}

void RS_KernelManager::exitKernelDataCriticalZone() {
    _kernel_data_critical_zone_depth -= 1;
    _kernel_data_mutex.unlock();
}

void RS_KernelManager::_kernelSynchronizationThreadFunc() {
    while (!is_kernel_inited.load(std::memory_order_acquire)) std::this_thread::yield();
    RS_LOG(_logger, debug, "Kernel manager sync thread running.");
    while (true) {
        unique_lock lock(_command_queue_mutex);
        _command_queue_cv.wait(lock, [this]() {
            return !_command_queue.empty() || kernel.isTerminated();
        });

        if (kernel.isTerminated()) {
            // Termination case 1: synchronized termination
            RS_LOG(_logger, debug, "Kernel manager sync thread entering termination.");
            while (!_command_queue.empty()) {
                _command_queue.pop();
            }
            RS_LOG(_logger, debug, "Kernel manager sync thread terminated.");
            break;
        }

        RS_LOG(_logger, debug, "Kernel manager sync thread start fetch command.");
        auto [message_id, function_name, arg] = _command_queue.front();
        _command_queue.pop();

        lock.unlock();

        RS_LOG(
            _logger,
            debug,
            "Kernel manager sync thread fetched command: {}, entering execution barrier.",
            function_name
        );

        // Raise the execution depth for the whole command: build and commit. Finders and the
        // callback wait on it, so none of them runs while a command executes.
        CommandExecutionScope execution_scope(this);

        RS_LOG(_logger, debug, "Kernel manager sync thread start execution.");

        // Reset before dispatch; a sub-manager that enters a kernel-data critical zone and throws
        // before exiting leaves this nonzero, marking an unrecoverable failure.
        _kernel_data_critical_zone_depth = 0;

        auto it = _methods.find(function_name);
        if (it == _methods.end()) {
            RS_LOG(_logger, warn, "Unknown command in queue: {}.", function_name);
        } else {
            try {
                it->second(arg, message_id);
                if (!_is_transaction_open) {
                    for (auto& message : _pending_messages)
                        if (message.id != RS_NULL_INDEX && _message_sink) _message_sink(message);
                    _pending_messages.clear();
                }
            } catch (const std::exception& e) {
                RS_LOG(_logger, error, "While executing command {}: {}.", function_name, e.what());
                if (message_id != RS_NULL_INDEX && _message_sink) {
                    _message_sink(
                        {message_id,
                         RSE_WORLD_MESSAGE_ERROR,
                         RS_WorldMessageData{RS_WorldMessageText{e.what()}}}
                    );
                }
                for (auto& message : _pending_messages)
                    if (message.id != RS_NULL_INDEX && _message_sink)
                        _message_sink(
                            {message.id,
                             RSE_WORLD_MESSAGE_ERROR,
                             RS_WorldMessageData{RS_WorldMessageText{
                                 "Aborted because of previous unsuccessful task"
                             }}}
                        );
                _pending_messages.clear();
                // A failure before the kernel-data critical zone is a recoverable characterization
                // error. A failure inside it leaves the context half-mutated and tainted; unwind
                // the zone, mark the kernel terminated, and return so freeKernel does the coarse
                // teardown.
                if (isInKernelDataCriticalZone()) {
                    RS_LOG(
                        _logger,
                        error,
                        "Command {} failed inside kernel-data critical zone; terminating.",
                        function_name
                    );
                    // Mark terminated before releasing the zone so the executor cannot grab the
                    // data mutex and touch the tainted context before it sees the termination.
                    kernel.terminate();
                    while (isInKernelDataCriticalZone()) exitKernelDataCriticalZone();
                    return;
                }
            }
        }

        RS_LOG(_logger, debug, "Kernel manager sync thread end execution.");

        if (function_name == "terminateKernel") {
            // Termination case 2: unsynchronized termination from command queue
            RS_LOG(_logger, debug, "Kernel manager sync thread terminated.");
            break;
        }
    }
}

void RS_KernelManager::_acquireCommandExecution() {
    std::unique_lock lock(_command_execution_mutex);
    if (_command_execution_depth == 0) {
        _command_execution_cv.wait(lock, [this]() { return _command_pause_count == 0; });
    }
    ++_command_execution_depth;
}

void RS_KernelManager::_releaseCommandExecution() {
    std::unique_lock lock(_command_execution_mutex);
    if (--_command_execution_depth == 0) _command_execution_cv.notify_all();
}

void RS_KernelManager::_pauseKernel() {
    RS_KernelManagerCriticalDataAccessGuard guard(this);
    markKernelModified();
    kernel.pause();
    RS_LOG(_logger, debug, "Pause kernel.");
}

void RS_KernelManager::_resumeKernel() {
    RS_KernelManagerCriticalDataAccessGuard guard(this);
    markKernelModified();
    kernel.resume();
    RS_LOG(_logger, debug, "Resume kernel.");
}

void RS_KernelManager::_terminateKernel() {
    if (is_kernel_inited.load(std::memory_order_acquire)) {
        RS_KernelManagerCriticalDataAccessGuard guard(this);
        markKernelModified();
        kernel.terminate();
    } else {
        // The kernel initialization is not completed, just mark it as terminated
        kernel.terminate();
    }
    RS_LOG(_logger, debug, "Terminate kernel.");
}

void RS_KernelManager::_beginModificationTransaction() {
    // Hold command execution across the whole transaction so the uncommitted managers built inside
    // it stay invisible to finders and the callback until the flush.
    _acquireCommandExecution();
    _is_transaction_open = true;
    RS_LOG(_logger, debug, "Begin modification transaction.");
}

void RS_KernelManager::_endModificationTransaction() {
    try {
        _flushPendingCommits();
    } catch (...) {
        _is_transaction_open = false;
        _releaseCommandExecution();
        throw;
    }
    _is_transaction_open = false;
    _releaseCommandExecution();
    RS_LOG(_logger, debug, "End modification transaction.");
}

void RS_KernelManager::_configureLayoutGrowthPolicies(RS_ContextHostData* host_data) {
    std::string capacity_string = _device_provider->getProperty("memory_capacity");
    if (capacity_string.empty()) return;

    RBigSize capacity_bytes = std::stoull(capacity_string);
    if (capacity_bytes == 0) return;

    const bool is_f64 = kernel.ctx_accuracy == RSE_FLOAT_64;

    // Convert the memory budget into an element budget using each layout's own element size, then
    // double the reserve below 40% of the budget and grow by a 2% chunk above it.
    auto apply = [&](RS_PooledLayout& layout, RBigSize element_bytes, RSize minimum_capacity) {
        RSize capacity_elements = static_cast<RSize>(capacity_bytes / element_bytes);
        RSize double_below_threshold = static_cast<RSize>(capacity_elements * 0.4);
        RSize growth_elements = static_cast<RSize>(capacity_elements * 0.02);
        layout.setGrowthPolicy(double_below_threshold, growth_elements, minimum_capacity);
    };

    apply(
        host_data->structure_link_material_layout,
        is_f64 ? sizeof(logic::RS_LinkMaterialMem<RSE_FLOAT_64>)
               : sizeof(logic::RS_LinkMaterialMem<RSE_FLOAT_32>),
        1024
    );
    apply(
        host_data->structure_voxel_material_layout,
        is_f64 ? sizeof(logic::RS_VoxelMaterialMem<RSE_FLOAT_64>)
               : sizeof(logic::RS_VoxelMaterialMem<RSE_FLOAT_32>),
        1024
    );
    apply(
        host_data->structure_link_layout,
        is_f64 ? sizeof(logic::RS_LinkMem<RSE_FLOAT_64>) : sizeof(logic::RS_LinkMem<RSE_FLOAT_32>),
        16384
    );
    apply(
        host_data->structure_voxel_layout,
        is_f64 ? sizeof(logic::RS_VoxelMem<RSE_FLOAT_64>)
               : sizeof(logic::RS_VoxelMem<RSE_FLOAT_32>),
        16384
    );
    apply(
        host_data->structure_rigid_body_layout,
        is_f64 ? sizeof(logic::RS_RigidBodyMem<RSE_FLOAT_64>)
               : sizeof(logic::RS_RigidBodyMem<RSE_FLOAT_32>),
        1024
    );
    apply(
        host_data->structure_rigid_body_surface_voxel_layout,
        is_f64 ? sizeof(logic::RS_RigidBodySurfaceVoxelMem<RSE_FLOAT_64>)
               : sizeof(logic::RS_RigidBodySurfaceVoxelMem<RSE_FLOAT_32>),
        16384
    );
    apply(
        host_data->structure_constraint_joint_layout,
        is_f64 ? sizeof(logic::RS_ConstraintJointMem<RSE_FLOAT_64>)
               : sizeof(logic::RS_ConstraintJointMem<RSE_FLOAT_32>),
        1024
    );

    RS_LOG(
        _logger,
        debug,
        "Layout growth policies configured (memory_capacity={} bytes)",
        capacity_bytes
    );
}

void RS_KernelManager::_setSimulationRSC(const RSC_Simulation& simulation, RIndex message_id) {
    RS_KernelManagerCriticalDataAccessGuard guard(this);
    RS_Timer timer;

    // Sync to make sure all operations in memory provider stream are finished
    _device_provider->sync();

    // Kernel settings are modified, needs to update kernel
    markKernelModified();

    _simulation_rsc = simulation;

    // Seed
    kernel.seed = simulation.seed;

    // Environment
    kernel.world_grav_acc = simulation.environment.gravity_acceleration;

    // Soft body
    kernel.softbody_internal_damping_z = simulation.soft_body.internal_damping_z;
    kernel.softbody_contact_damping_z = simulation.soft_body.contact_damping_z;
    kernel.softbody_global_damping_z = simulation.soft_body.global_damping_z;
    kernel.softbody_integration_frequency_multiplier =
        simulation.soft_body.integration_frequency_multiplier;
    kernel.softbody_link_max_compression = simulation.soft_body.link_max_compression;

    // Rigid body constraint solver
    kernel.rigidbody_constraint_iterations = simulation.rigid_body.constraint_iterations;
    kernel.rigidbody_constraint_position_correction_ratio =
        simulation.rigid_body.constraint_position_correction_ratio;
    kernel.rigidbody_constraint_warm_start_damping_z =
        simulation.rigid_body.constraint_warm_start_damping_z;
    kernel.rigidbody_constraint_stiffness_initial =
        simulation.rigid_body.constraint_stiffness_initial;
    kernel.rigidbody_constraint_stiffness_min_threshold =
        simulation.rigid_body.constraint_stiffness_min_threshold;
    kernel.rigidbody_constraint_stiffness_max_threshold =
        simulation.rigid_body.constraint_stiffness_max_threshold;
    kernel.rigidbody_constraint_stiffness_auto_adjust_watt_threshold =
        simulation.rigid_body.constraint_stiffness_auto_adjust_watt_threshold;
    kernel.rigidbody_constraint_stiffness_auto_adjust_step =
        simulation.rigid_body.constraint_stiffness_auto_adjust_step;

    // Rigid body contact solver
    kernel.rigidbody_contact_iterations = simulation.rigid_body.contact_iterations;
    kernel.rigidbody_contact_position_correction_iterations =
        simulation.rigid_body.contact_position_correction_iterations;
    kernel.rigidbody_contact_position_correction_ratio =
        simulation.rigid_body.contact_position_correction_ratio;
    kernel.rigidbody_contact_warm_start_damping_z =
        simulation.rigid_body.contact_warm_start_damping_z;
    kernel.rigidbody_contact_slop = simulation.rigid_body.contact_slop;

    // Rigid body damping
    kernel.rigidbody_linear_damping_z = simulation.rigid_body.linear_damping_z;
    kernel.rigidbody_angular_damping_z = simulation.rigid_body.angular_damping_z;

    // Vision trace
    kernel.vision_trace_firefly_cap = simulation.vision.trace.firefly_cap;
    kernel.vision_trace_russian_roulette_start_depth =
        simulation.vision.trace.russian_roulette_start_depth;

    // Vision ReSTIR DI / GI
    kernel.vision_is_restir_enabled = simulation.vision.restir.is_enabled;
    kernel.vision_restir_disocclusion_position_threshold =
        simulation.vision.restir.disocclusion_position_threshold;
    kernel.vision_restir_disocclusion_normal_threshold =
        simulation.vision.restir.disocclusion_normal_threshold;
    kernel.vision_restir_max_specular_chain = simulation.vision.restir.max_specular_chain;
    kernel.vision_restir_di_m_cap = simulation.vision.restir.di_m_cap;
    kernel.vision_restir_di_spatial_neighbors = simulation.vision.restir.di_spatial_neighbors;
    kernel.vision_restir_di_spatial_radius = simulation.vision.restir.di_spatial_radius;
    kernel.vision_restir_gi_is_bounce_nee_enabled =
        simulation.vision.restir.is_gi_bounce_nee_enabled;
    kernel.vision_restir_gi_m_cap = simulation.vision.restir.gi_m_cap;
    kernel.vision_restir_gi_spatial_neighbors = simulation.vision.restir.gi_spatial_neighbors;
    kernel.vision_restir_gi_spatial_radius = simulation.vision.restir.gi_spatial_radius;
    kernel.vision_restir_gi_validation_period = simulation.vision.restir.gi_validation_period;

    // Vision denoise
    kernel.vision_denoise_atrous_iterations = 0;
    kernel.vision_denoise_atrous_min_passes = 0;
    kernel.vision_denoise_sigma_color = 0;
    kernel.vision_denoise_sigma_normal = 0;
    kernel.vision_denoise_sigma_depth = 0;
    kernel.vision_denoise_temporal_alpha = 0;
    kernel.vision_denoise_temporal_alpha_moments = 0;
    kernel.vision_denoise_temporal_min_history = 0;
    simulation.vision.denoise.visit([&](auto& d) {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, RSC_SimulationVisionDenoiseNone>) {
            kernel.vision_denoise_type = RSE_VISION_DENOISE_NONE;
        } else if constexpr (std::is_same_v<T, RSC_SimulationVisionDenoiseAtrous>) {
            kernel.vision_denoise_type = RSE_VISION_DENOISE_ATROUS;
            kernel.vision_denoise_atrous_iterations = d.atrous_iterations;
            kernel.vision_denoise_atrous_min_passes = d.atrous_min_passes;
            kernel.vision_denoise_sigma_color = d.sigma_color;
            kernel.vision_denoise_sigma_normal = d.sigma_normal;
            kernel.vision_denoise_sigma_depth = d.sigma_depth;
        } else if constexpr (std::is_same_v<T, RSC_SimulationVisionDenoiseSVGF>) {
            kernel.vision_denoise_type = RSE_VISION_DENOISE_SVGF;
            kernel.vision_denoise_atrous_iterations = d.atrous_iterations;
            kernel.vision_denoise_atrous_min_passes = d.atrous_min_passes;
            kernel.vision_denoise_sigma_color = d.sigma_color;
            kernel.vision_denoise_sigma_normal = d.sigma_normal;
            kernel.vision_denoise_sigma_depth = d.sigma_depth;
            kernel.vision_denoise_temporal_alpha = d.temporal_alpha;
            kernel.vision_denoise_temporal_alpha_moments = d.temporal_alpha_moments;
            kernel.vision_denoise_temporal_min_history = d.temporal_min_history;
        }
    });

    // Control / record frequencies
    kernel.control_frequency = simulation.control.frequency;
    kernel.record_frequency = simulation.record.frequency;

    // Stop condition
    kernel.host_data->stop_condition = simulation.stop_condition;

    RS_LOG(_logger, debug, "setSimulationRSC takes {}.", timer.elapsedTimeString());
    _queueInfo(message_id, RS_WorldMessageText{"setSimulationRSC succeeded"});
}

/*****************************************************************
 * Resource
 *****************************************************************/
void RS_KernelManager::_addResource(const RSC_ResourceVariant& resource, RIndex message_id) {
    std::string name;
    resource.visit([&](auto& r) {
        using T = std::decay_t<decltype(r)>;
        name = r.name;
        if constexpr (std::is_same_v<T, RSC_TextureConstant> ||
                      std::is_same_v<T, RSC_TextureGrid>) {
            _kernel_texture_manager.addTexture(RS_KernelTextureManager::SupportedRSC(r));
        }
    });
    RIndex gid = RS_NULL_INDEX;
    if (auto it = _resource_gids.find(name); it != _resource_gids.end()) gid = it->second;
    _queueInfo(
        message_id, RS_WorldMessageCreated{RSE_WORLD_MESSAGE_CREATED_TYPE_RESOURCE, gid, name}
    );
}

void RS_KernelManager::_updateResource(
    const std::string& name,
    const RSC_ResourceVariant& resource,
    RIndex message_id
) {
    _removeResource(name, RS_NULL_INDEX);
    _addResource(resource, message_id);
}

void RS_KernelManager::_removeResource(const std::string& name, RIndex message_id) {
    // Texture is the only resource kind today; new kinds get added here.
    _kernel_texture_manager.removeTexture(name);
    _queueInfo(message_id, RS_WorldMessageText{format("removeResource {} succeeded", name)});
}

/*****************************************************************
 * Essence
 *****************************************************************/
void RS_KernelManager::_addEssence(const RSC_EssenceVariant& essence, RIndex message_id) {
    std::string name;
    essence.visit([&](auto& c) {
        using T = std::decay_t<decltype(c)>;
        name = c.name;
        if constexpr (std::is_same_v<T, RSC_Material>) {
            _kernel_material_manager.addMaterial(c);
        } else if constexpr (std::is_same_v<T, RSC_MaterialPairPropertyCoulombFriction> ||
                             std::is_same_v<T, RSC_MaterialPairPropertyRestitution>) {
            _kernel_material_manager.addMaterialProperty(
                RS_KernelMaterialManager::PairPropertyVariant(c)
            );
        } else {
            _kernel_material_manager.addMaterialProperty(
                RS_KernelMaterialManager::UnaryPropertyVariant(c)
            );
        }
    });
    RIndex gid = RS_NULL_INDEX;
    if (auto it = _essence_gids.find(name); it != _essence_gids.end()) gid = it->second;
    _queueInfo(
        message_id, RS_WorldMessageCreated{RSE_WORLD_MESSAGE_CREATED_TYPE_ESSENCE, gid, name}
    );
}

void RS_KernelManager::_updateEssence(
    const std::string& name,
    const RSC_EssenceVariant& essence,
    RIndex message_id
) {
    _removeEssence(name, RS_NULL_INDEX);
    _addEssence(essence, message_id);
}

void RS_KernelManager::_removeEssence(const std::string& name, RIndex message_id) {
    if (_kernel_material_manager.findMaterial(name) != RS_NULL_INDEX) {
        _kernel_material_manager.removeMaterial(name);
        _queueInfo(message_id, RS_WorldMessageText{format("removeEssence {} succeeded", name)});
        return;
    }
    if (_kernel_material_manager.hasMaterialProperty(name)) {
        _kernel_material_manager.removeMaterialProperty(name);
        _queueInfo(message_id, RS_WorldMessageText{format("removeEssence {} succeeded", name)});
        return;
    }
    RS_THROW(RS_NotFoundException("Essence '{}' not found", name));
}

/*****************************************************************
 * Entity
 *****************************************************************/
void RS_KernelManager::_addEntity(const RSC_EntityVariant& entity, RIndex message_id) {
    std::string name;
    entity.visit([&](auto& c) {
        using T = std::decay_t<decltype(c)>;
        name = c.name;
        if constexpr (std::is_same_v<T, RSC_StructureVoxel>) {
            auto mgr = std::make_unique<RS_KernelStructureVoxelManager>(this, c);
            _kernel_structure_voxel_managers[c.name] = std::move(mgr);
            _stageCommit(RSE_KERNEL_MANAGER_STRUCTURE_VOXEL, c.name);
        } else if constexpr (std::is_same_v<T, RSC_CameraPerspectiveFree> ||
                             std::is_same_v<T, RSC_CameraPerspectiveRigidBodyLinked>) {
            auto mgr = std::make_unique<RS_KernelCameraPerspectiveManager>(this, c);
            _kernel_camera_perspective_managers[c.name] = std::move(mgr);
            _stageCommit(RSE_KERNEL_MANAGER_CAMERA_PERSPECTIVE, c.name);
        } else if constexpr (std::is_same_v<T, RSC_CameraStereoFree> ||
                             std::is_same_v<T, RSC_CameraStereoRigidBodyLinked>) {
            auto mgr = std::make_unique<RS_KernelCameraStereoManager>(this, c);
            _kernel_camera_stereo_managers[c.name] = std::move(mgr);
            _stageCommit(RSE_KERNEL_MANAGER_CAMERA_STEREO, c.name);
        } else if constexpr (std::is_same_v<T, RSC_FloorFlat> ||
                             std::is_same_v<T, RSC_FloorArray>) {
            if (_kernel_floor_manager) {
                RS_THROW(RS_DuplicateException("Floor entity already exists"));
            }
            _kernel_floor_manager = std::make_unique<RS_KernelFloorManager>(this, c);
            _stageCommit(RSE_KERNEL_MANAGER_FLOOR, c.name);
        } else if constexpr (std::is_same_v<T, RSC_CosmicBackgroundUniform> ||
                             std::is_same_v<T, RSC_CosmicBackgroundGradient> ||
                             std::is_same_v<T, RSC_CosmicBackgroundHosekWilkie>) {
            if (_kernel_cosmic_background_manager) {
                RS_THROW(RS_DuplicateException("Cosmic background entity already exists"));
            }
            _kernel_cosmic_background_manager =
                std::make_unique<RS_KernelCosmicBackgroundManager>(this, c);
            _stageCommit(RSE_KERNEL_MANAGER_COSMIC_BACKGROUND, c.name);
        }
    });
    RIndex gid = RS_NULL_INDEX;
    if (auto it = _entity_gids.find(name); it != _entity_gids.end()) gid = it->second;
    _queueInfo(
        message_id, RS_WorldMessageCreated{RSE_WORLD_MESSAGE_CREATED_TYPE_ENTITY, gid, name}
    );
}

void RS_KernelManager::_removeEntity(const std::string& name, RIndex message_id) {
    if (auto it = _kernel_structure_voxel_managers.find(name);
        it != _kernel_structure_voxel_managers.end()) {
        if (it->second->getUseCount() > 0) {
            RS_THROW(RS_SimulationException(
                "Cannot remove structure '{}': still in use ({} reference(s))",
                name,
                it->second->getUseCount()
            ));
        }
        it->second->decommit();
        _kernel_structure_voxel_managers.erase(it);
        _retractPendingCommit(RSE_KERNEL_MANAGER_STRUCTURE_VOXEL, name);
        _queueInfo(message_id, RS_WorldMessageText{format("removeEntity {} succeeded", name)});
        return;
    }
    if (auto it = _kernel_camera_perspective_managers.find(name);
        it != _kernel_camera_perspective_managers.end()) {
        if (it->second->getUseCount() > 0) {
            RS_THROW(RS_SimulationException(
                "Cannot remove camera '{}': still in use ({} reference(s))",
                name,
                it->second->getUseCount()
            ));
        }
        it->second->decommit();
        _kernel_camera_perspective_managers.erase(it);
        _retractPendingCommit(RSE_KERNEL_MANAGER_CAMERA_PERSPECTIVE, name);
        _queueInfo(message_id, RS_WorldMessageText{format("removeEntity {} succeeded", name)});
        return;
    }
    if (auto it = _kernel_camera_stereo_managers.find(name);
        it != _kernel_camera_stereo_managers.end()) {
        if (it->second->getUseCount() > 0) {
            RS_THROW(RS_SimulationException(
                "Cannot remove camera '{}': still in use ({} reference(s))",
                name,
                it->second->getUseCount()
            ));
        }
        it->second->decommit();
        _kernel_camera_stereo_managers.erase(it);
        _retractPendingCommit(RSE_KERNEL_MANAGER_CAMERA_STEREO, name);
        _queueInfo(message_id, RS_WorldMessageText{format("removeEntity {} succeeded", name)});
        return;
    }
    if (_kernel_floor_manager) {
        bool matched = false;
        _kernel_floor_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) {
            if (_kernel_floor_manager->getUseCount() > 0) {
                RS_THROW(RS_SimulationException(
                    "Cannot remove floor '{}': still in use ({} reference(s))",
                    name,
                    _kernel_floor_manager->getUseCount()
                ));
            }
            _kernel_floor_manager->decommit();
            _kernel_floor_manager.reset();
            _retractPendingCommit(RSE_KERNEL_MANAGER_FLOOR, name);
            _queueInfo(message_id, RS_WorldMessageText{format("removeEntity {} succeeded", name)});
            return;
        }
    }
    if (_kernel_cosmic_background_manager) {
        bool matched = false;
        _kernel_cosmic_background_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) {
            if (_kernel_cosmic_background_manager->getUseCount() > 0) {
                RS_THROW(RS_SimulationException(
                    "Cannot remove cosmic_background '{}': still in use ({} reference(s))",
                    name,
                    _kernel_cosmic_background_manager->getUseCount()
                ));
            }
            _kernel_cosmic_background_manager->decommit();
            _kernel_cosmic_background_manager.reset();
            _retractPendingCommit(RSE_KERNEL_MANAGER_COSMIC_BACKGROUND, name);
            _queueInfo(message_id, RS_WorldMessageText{format("removeEntity {} succeeded", name)});
            return;
        }
    }
    RS_THROW(RS_NotFoundException("Entity '{}' not found", name));
}

void RS_KernelManager::_freezeEntity(const std::string& name, RIndex message_id) {
    if (auto it = _kernel_structure_voxel_managers.find(name);
        it != _kernel_structure_voxel_managers.end()) {
        it->second->freeze();
        _queueInfo(message_id, RS_WorldMessageText{format("freezeEntity {} succeeded", name)});
        return;
    }
    if (auto it = _kernel_camera_perspective_managers.find(name);
        it != _kernel_camera_perspective_managers.end()) {
        it->second->freeze();
        _queueInfo(message_id, RS_WorldMessageText{format("freezeEntity {} succeeded", name)});
        return;
    }
    if (auto it = _kernel_camera_stereo_managers.find(name);
        it != _kernel_camera_stereo_managers.end()) {
        it->second->freeze();
        _queueInfo(message_id, RS_WorldMessageText{format("freezeEntity {} succeeded", name)});
        return;
    }
    if (_kernel_floor_manager) {
        bool matched = false;
        _kernel_floor_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) {
            _kernel_floor_manager->freeze();
            _queueInfo(message_id, RS_WorldMessageText{format("freezeEntity {} succeeded", name)});
            return;
        }
    }
    if (_kernel_cosmic_background_manager) {
        bool matched = false;
        _kernel_cosmic_background_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) {
            _kernel_cosmic_background_manager->freeze();
            _queueInfo(message_id, RS_WorldMessageText{format("freezeEntity {} succeeded", name)});
            return;
        }
    }
    RS_THROW(RS_NotFoundException("Entity '{}' not found", name));
}

void RS_KernelManager::_unfreezeEntity(const std::string& name, RIndex message_id) {
    if (auto it = _kernel_structure_voxel_managers.find(name);
        it != _kernel_structure_voxel_managers.end()) {
        it->second->unfreeze();
        _queueInfo(message_id, RS_WorldMessageText{format("unfreezeEntity {} succeeded", name)});
        return;
    }
    if (auto it = _kernel_camera_perspective_managers.find(name);
        it != _kernel_camera_perspective_managers.end()) {
        it->second->unfreeze();
        _queueInfo(message_id, RS_WorldMessageText{format("unfreezeEntity {} succeeded", name)});
        return;
    }
    if (auto it = _kernel_camera_stereo_managers.find(name);
        it != _kernel_camera_stereo_managers.end()) {
        it->second->unfreeze();
        _queueInfo(message_id, RS_WorldMessageText{format("unfreezeEntity {} succeeded", name)});
        return;
    }
    if (_kernel_floor_manager) {
        bool matched = false;
        _kernel_floor_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) {
            _kernel_floor_manager->unfreeze();
            _queueInfo(
                message_id, RS_WorldMessageText{format("unfreezeEntity {} succeeded", name)}
            );
            return;
        }
    }
    if (_kernel_cosmic_background_manager) {
        bool matched = false;
        _kernel_cosmic_background_manager->getRSC().visit([&](auto& c) {
            if (c.name == name) matched = true;
        });
        if (matched) {
            _kernel_cosmic_background_manager->unfreeze();
            _queueInfo(
                message_id, RS_WorldMessageText{format("unfreezeEntity {} succeeded", name)}
            );
            return;
        }
    }
    RS_THROW(RS_NotFoundException("Entity '{}' not found", name));
}

/*****************************************************************
 * Record
 *****************************************************************/
void RS_KernelManager::_addRecord(const RSC_RecordVariant& record, RIndex message_id) {
    std::string name;
    record.visit([&](auto& r) {
        using T = std::decay_t<decltype(r)>;
        name = r.name;
        if constexpr (std::is_same_v<T, RSC_IntersectRegionCuboid> ||
                      std::is_same_v<T, RSC_IntersectRegionCuboidRigidBodyLinked> ||
                      std::is_same_v<T, RSC_IntersectRegionSphere>) {
            auto mgr = std::make_unique<RS_KernelIntersectRegionManager>(
                this, RS_KernelIntersectRegionManager::SupportedRSC(r)
            );
            _kernel_intersect_region_managers[r.name] = std::move(mgr);
            _stageCommit(RSE_KERNEL_MANAGER_INTERSECT_REGION, r.name);
        }
    });
    RIndex gid = RS_NULL_INDEX;
    if (auto it = _record_gids.find(name); it != _record_gids.end()) gid = it->second;
    _queueInfo(
        message_id, RS_WorldMessageCreated{RSE_WORLD_MESSAGE_CREATED_TYPE_RECORD, gid, name}
    );
}

void RS_KernelManager::_removeRecord(const std::string& name, RIndex message_id) {
    if (auto it = _kernel_intersect_region_managers.find(name);
        it != _kernel_intersect_region_managers.end()) {
        it->second->decommit();
        _kernel_intersect_region_managers.erase(it);
        _retractPendingCommit(RSE_KERNEL_MANAGER_INTERSECT_REGION, name);
        _queueInfo(message_id, RS_WorldMessageText{format("removeRecord {} succeeded", name)});
        return;
    }
    RS_THROW(RS_NotFoundException("Record '{}' not found", name));
}

/*****************************************************************
 * Signal
 *****************************************************************/
void RS_KernelManager::_addSignal(const RSC_SignalVariant& signal, RIndex message_id) {
    std::string name;
    signal.visit([&](auto& s) {
        using T = std::decay_t<decltype(s)>;
        name = s.name;
        if constexpr (std::is_same_v<T, RSC_Force>) {
            auto mgr = std::make_unique<RS_KernelForceManager>(this, s);
            _kernel_force_managers[s.name] = std::move(mgr);
            _stageCommit(RSE_KERNEL_MANAGER_FORCE, s.name);
        } else if constexpr (std::is_same_v<T, RSC_Torque>) {
            auto mgr = std::make_unique<RS_KernelTorqueManager>(this, s);
            _kernel_torque_managers[s.name] = std::move(mgr);
            _stageCommit(RSE_KERNEL_MANAGER_TORQUE, s.name);
        } else if constexpr (std::is_same_v<T, RSC_Transform>) {
            auto mgr = std::make_unique<RS_KernelTransformManager>(this, s);
            _kernel_transform_managers[s.name] = std::move(mgr);
            _stageCommit(RSE_KERNEL_MANAGER_TRANSFORM, s.name);
        }
    });
    RIndex gid = RS_NULL_INDEX;
    if (auto it = _signal_gids.find(name); it != _signal_gids.end()) gid = it->second;
    _queueInfo(
        message_id, RS_WorldMessageCreated{RSE_WORLD_MESSAGE_CREATED_TYPE_SIGNAL, gid, name}
    );
}

void RS_KernelManager::_removeSignal(const std::string& name, RIndex message_id) {
    if (auto it = _kernel_force_managers.find(name); it != _kernel_force_managers.end()) {
        it->second->decommit();
        _kernel_force_managers.erase(it);
        _retractPendingCommit(RSE_KERNEL_MANAGER_FORCE, name);
        _queueInfo(message_id, RS_WorldMessageText{format("removeSignal {} succeeded", name)});
        return;
    }
    if (auto it = _kernel_torque_managers.find(name); it != _kernel_torque_managers.end()) {
        it->second->decommit();
        _kernel_torque_managers.erase(it);
        _retractPendingCommit(RSE_KERNEL_MANAGER_TORQUE, name);
        _queueInfo(message_id, RS_WorldMessageText{format("removeSignal {} succeeded", name)});
        return;
    }
    if (auto it = _kernel_transform_managers.find(name); it != _kernel_transform_managers.end()) {
        it->second->decommit();
        _kernel_transform_managers.erase(it);
        _retractPendingCommit(RSE_KERNEL_MANAGER_TRANSFORM, name);
        _queueInfo(message_id, RS_WorldMessageText{format("removeSignal {} succeeded", name)});
        return;
    }
    RS_THROW(RS_NotFoundException("Signal '{}' not found", name));
}

/*****************************************************************
 * Pending commit staging and flush
 *****************************************************************/
void RS_KernelManager::_stageCommit(RSE_KernelManagerType type, const std::string& name) {
    _pending_commits.push_back({type, name});
    if (!_is_transaction_open) _flushPendingCommits();
}

void RS_KernelManager::_retractPendingCommit(RSE_KernelManagerType type, const std::string& name) {
    _pending_commits.erase(
        std::remove_if(
            _pending_commits.begin(),
            _pending_commits.end(),
            [&](const RS_KernelPendingCommit& pc) { return pc.type == type && pc.name == name; }
        ),
        _pending_commits.end()
    );
}

void RS_KernelManager::_flushPendingCommits() {
    // Textures and materials commit first so structures and floors can resolve their grid handles
    // and vision material slots.
    _kernel_texture_manager.commit();
    _kernel_material_manager.commit();
    for (const auto& pc : _pending_commits) {
        _flushOneCommit(pc.type, pc.name);
    }
    _pending_commits.clear();
}

void RS_KernelManager::_flushOneCommit(RSE_KernelManagerType type, const std::string& name) {
    switch (type) {
        case RSE_KERNEL_MANAGER_STRUCTURE_VOXEL:
            if (auto it = _kernel_structure_voxel_managers.find(name);
                it != _kernel_structure_voxel_managers.end())
                it->second->commit();
            break;
        case RSE_KERNEL_MANAGER_CAMERA_PERSPECTIVE:
            if (auto it = _kernel_camera_perspective_managers.find(name);
                it != _kernel_camera_perspective_managers.end())
                it->second->commit();
            break;
        case RSE_KERNEL_MANAGER_CAMERA_STEREO:
            if (auto it = _kernel_camera_stereo_managers.find(name);
                it != _kernel_camera_stereo_managers.end())
                it->second->commit();
            break;
        case RSE_KERNEL_MANAGER_FLOOR:
            if (_kernel_floor_manager) _kernel_floor_manager->commit();
            break;
        case RSE_KERNEL_MANAGER_COSMIC_BACKGROUND:
            if (_kernel_cosmic_background_manager) _kernel_cosmic_background_manager->commit();
            break;
        case RSE_KERNEL_MANAGER_FORCE:
            if (auto it = _kernel_force_managers.find(name); it != _kernel_force_managers.end())
                it->second->commit();
            break;
        case RSE_KERNEL_MANAGER_TORQUE:
            if (auto it = _kernel_torque_managers.find(name); it != _kernel_torque_managers.end())
                it->second->commit();
            break;
        case RSE_KERNEL_MANAGER_TRANSFORM:
            if (auto it = _kernel_transform_managers.find(name);
                it != _kernel_transform_managers.end())
                it->second->commit();
            break;
        case RSE_KERNEL_MANAGER_INTERSECT_REGION:
            if (auto it = _kernel_intersect_region_managers.find(name);
                it != _kernel_intersect_region_managers.end())
                it->second->commit();
            break;
    }
}

void RS_KernelManager::_queueInfo(RIndex message_id, RS_WorldMessageData data) {
    if (message_id == RS_NULL_INDEX) return;
    _pending_messages.push_back({message_id, RSE_WORLD_MESSAGE_INFO, std::move(data)});
}

/*****************************************************************
 * RS_KernelGidRegistration
 *****************************************************************/
RS_KernelGidRegistration::RS_KernelGidRegistration(
    RS_KernelManager* kernel_manager,
    RSE_KernelGidType type,
    std::string name
)
    : _kernel_manager(kernel_manager), _type(type), _name(std::move(name)) {
    switch (_type) {
        case RSE_KERNEL_GID_ENTITY:
            _gid = _kernel_manager->registerEntity(_name);
            break;
        case RSE_KERNEL_GID_ESSENCE:
            _gid = _kernel_manager->registerEssence(_name);
            break;
        case RSE_KERNEL_GID_RESOURCE:
            _gid = _kernel_manager->registerResource(_name);
            break;
        case RSE_KERNEL_GID_RECORD:
            _gid = _kernel_manager->registerRecord(_name);
            break;
        case RSE_KERNEL_GID_SIGNAL:
            _gid = _kernel_manager->registerSignal(_name);
            break;
    }
}

RS_KernelGidRegistration::~RS_KernelGidRegistration() { _release(); }

RS_KernelGidRegistration::RS_KernelGidRegistration(RS_KernelGidRegistration&& other) noexcept
    : _kernel_manager(other._kernel_manager),
      _type(other._type),
      _name(std::move(other._name)),
      _gid(other._gid) {
    other._kernel_manager = nullptr;
    other._gid = RS_NULL_INDEX;
}

RS_KernelGidRegistration& RS_KernelGidRegistration::operator=(
    RS_KernelGidRegistration&& other
) noexcept {
    if (this != &other) {
        _release();
        _kernel_manager = other._kernel_manager;
        _type = other._type;
        _name = std::move(other._name);
        _gid = other._gid;
        other._kernel_manager = nullptr;
        other._gid = RS_NULL_INDEX;
    }
    return *this;
}

void RS_KernelGidRegistration::_release() {
    if (_kernel_manager == nullptr || _gid == RS_NULL_INDEX) return;
    switch (_type) {
        case RSE_KERNEL_GID_ENTITY:
            _kernel_manager->deregisterEntity(_name);
            break;
        case RSE_KERNEL_GID_ESSENCE:
            _kernel_manager->deregisterEssence(_name);
            break;
        case RSE_KERNEL_GID_RESOURCE:
            _kernel_manager->deregisterResource(_name);
            break;
        case RSE_KERNEL_GID_RECORD:
            _kernel_manager->deregisterRecord(_name);
            break;
        case RSE_KERNEL_GID_SIGNAL:
            _kernel_manager->deregisterSignal(_name);
            break;
    }
    _kernel_manager = nullptr;
    _gid = RS_NULL_INDEX;
}

}  // namespace rise
