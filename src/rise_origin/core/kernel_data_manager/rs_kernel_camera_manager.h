// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_CAMERA_MANAGER_H
#define RS_KERNEL_CAMERA_MANAGER_H

#include <vector>

#include "rise_origin/core/rs_simulation_record.h"
#include "rise_origin/core/rs_simulation_signal.h"
#include "rise_origin/device/data/rs_buffer.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/logic/vision/rs_camera_m.h"
#include "rise_origin/rsc/rs_camera_config.h"

namespace rise {

class RS_KernelManager;

class RS_KernelCameraManager {
   public:
    RS_KernelCameraManager() = default;

    void init(RS_KernelManager *kernel_manager, const RS_CameraConfig &config);

    void free();

    bool hasRecorderRecords();

    void syncRecorderRecords();

    void waitForRecorderRecordsSync();

    std::vector<RS_SimulationCameraRecord> getRecorderRecords();

    bool hasControllerSignal();

    void syncControllerSignal();

    RS_SimulationCameraSignal getControllerSignal();

    bool hasControllerRecord();

    void syncControllerRecord();

    RS_SimulationCameraRecord getControllerRecord();

    std::string getName() const;

    std::string getStructureName() const;

    RIndex getStructureGid() const;

    [[nodiscard]] const RS_CameraConfig &getConfig() const;

   private:
    RIndex _camera_gid = RS_NULL_INDEX;

    /// Linked structure gid, only valid if the camera is attached to a rigid body
    /// in the structure
    RIndex _structure_gid = RS_NULL_INDEX;

    RS_KernelManager *_kernel_manager = nullptr;

    bool _is_kernel_data_inited = false;

    RS_CameraConfig _config;

    RSize _width = 0;

    RSize _height = 0;

    device::RS_DoubleBuffer _recorder_buffer;

    device::RS_DirectBuffer _controller_buffer;

    /// RGB canvas.
    logic::RS_CameraRGBCanvas _camera_current_rgb_canvas;
    logic::RS_CameraRGBCanvas _camera_next_rgb_canvas;

    /// Shared primary AOVs (basic + restir).
    logic::RS_CameraAuxNormal _camera_aux_normal;
    logic::RS_CameraAuxDepthZ _camera_aux_depth_z;
    logic::RS_CameraAuxHitPosition _camera_aux_hit_position;
    logic::RS_CameraAuxObjectKey _camera_aux_object_key;
    logic::RS_CameraAuxRoughness _camera_aux_roughness;
    logic::RS_CameraAuxScreenMotion _camera_aux_screen_motion;
    /// Restir-only endpoint + chain AOVs.
    logic::RS_CameraAuxHitPosition _camera_aux_endpoint_hit_position;
    logic::RS_CameraAuxNormal _camera_aux_endpoint_normal;
    logic::RS_CameraAuxObjectKey _camera_aux_endpoint_object_key;
    logic::RS_CameraAuxAlbedo _camera_aux_endpoint_albedo;
    logic::RS_CameraAuxRoughness _camera_aux_endpoint_roughness;
    logic::RS_CameraAuxMetalness _camera_aux_endpoint_metalness;
    logic::RS_CameraAuxIntensity _camera_aux_endpoint_intensity;
    logic::RS_CameraAuxMaterialType _camera_aux_endpoint_material_type;
    logic::RS_CameraAuxScreenMotion _camera_aux_endpoint_screen_motion;
    logic::RS_CameraAuxChainThroughput _camera_aux_chain_throughput;
    logic::RS_CameraAuxChainTerminalKind _camera_aux_chain_terminal_kind;
    logic::RS_CameraReservoirDI _camera_prev_reservoir_di;
    logic::RS_CameraReservoirDI _camera_current_reservoir_di;
    logic::RS_CameraReservoirGI _camera_prev_reservoir_gi;
    logic::RS_CameraReservoirGI _camera_current_reservoir_gi;
    logic::RS_CameraReservoirDI _camera_spatial_reservoir_di;
    logic::RS_CameraReservoirGI _camera_spatial_reservoir_gi;
    logic::RS_CameraPrevAuxNormal _camera_prev_aux_normal;
    logic::RS_CameraPrevAuxDepthZ _camera_prev_aux_depth_z;
    logic::RS_CameraPrevAuxObjectKey _camera_prev_aux_object_key;
    logic::RS_CameraDenoiserColorHistory _camera_denoiser_color_history;
    logic::RS_CameraDenoiserMomentHistory _camera_denoiser_moment_history;
    logic::RS_CameraDenoiserHistoryLength _camera_denoiser_history_length;
    logic::RS_CameraDenoiserVariance _camera_denoiser_variance;
    logic::RS_CameraDenoiserColorHistory _camera_prev_denoiser_color_history;
    logic::RS_CameraDenoiserMomentHistory _camera_prev_denoiser_moment_history;
    logic::RS_CameraDenoiserHistoryLength _camera_prev_denoiser_history_length;

    /// Signal data
    RSize _device_camera_control_signal_offset = 0;
    RS_MemoryPtr _host_camera_control_signal_buffer;

    /// Camera rigid body relative position, only valid if the camera is attached to a rigid
    /// body in the structure
    RVec3rf _camera_rb_position = RVec3rf::zero();

    /// Camera rigid body relative orientation, only valid if the camera is attached to
    /// a rigid body in the structure
    RQuat3rf _camera_rb_orientation = RQuat3rf::unit();

    /// Camera rigid body structure scope id, only valid if the camera is attached to
    /// a rigid body in the structure
    RIndex _rigid_body_sid = RS_NULL_INDEX;

    /// Camera origin position, only valid if the camera is a free camera
    RVec3rf _camera_origin_position = RVec3rf::zero();

    /// Camera orientation, only valid if the camera is a free camera
    RQuat3rf _camera_orientation = RQuat3rf::unit();

    void _initKernelData();

    void _freeKernelData();
};

}  // namespace rise

#endif  // RS_KERNEL_CAMERA_MANAGER_H
