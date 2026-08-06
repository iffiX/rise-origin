// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_CAMERA_STEREO_MANAGER_H
#define RS_KERNEL_CAMERA_STEREO_MANAGER_H

#include <string>
#include <vector>

#include "rise_origin/core/kernel_manager/rs_kernel_gid_registration.h"
#include "rise_origin/core/kernel_manager/rs_kernel_structure_voxel_manager.h"
#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/device/data/rs_buffer.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/logic/record/rs_camera_record_m.h"
#include "rise_origin/logic/vision/rs_camera_m.h"
#include "rise_origin/rsc/entity/rsc_camera.h"
#include "rise_origin/utils/data/rs_variant.h"

namespace rise {

class RS_KernelManager;

/**
 * @brief Owns one stereo camera (free or rigid-body-linked) as two perspective sub-cameras,
 *        one per eye, each with its own canvas, ReSTIR / denoise aux + history buffers, and
 *        recorder/controller buffer; plus the shared per-eye pose signal.
 * @details Each eye renders exactly like a perspective camera, so the render, denoise, and ReSTIR
 *        kernels stay the perspective path. The host stereo record is assembled by pointing its
 *        left_/right_ fields at the two sub-cameras' fields. Per-camera buffers are kept as arrays
 *        of two, indexed by eye (0 left, 1 right).
 */
class RS_KernelCameraStereoManager {
   public:
    using SupportedRSC =
        RS_Variant<rsc::RSC_CameraStereoFree, rsc::RSC_CameraStereoRigidBodyLinked>;

    /**
     * @brief Bind to the kernel manager, allocate per-eye canvas + aux buffers, and, for the
     *        rigid-body-linked variant, take a reference on the host structure.
     */
    RS_KernelCameraStereoManager(RS_KernelManager *kernel_manager, const SupportedRSC &camera_rsc);

    /**
     * @brief Commit the camera to the kernel context.
     */
    void commit();

    /**
     * @brief Decommit the camera from the kernel context.
     */
    void decommit();

    /**
     * @brief Stop rendering the camera while keeping its canvas at the last frame.
     * @details Deactivates both eyes' layouts so every render launch skips it. Idempotent.
     */
    void freeze();

    /**
     * @brief Resume rendering a frozen camera.
     * @details Reactivates both eyes' layouts. Idempotent.
     */
    void unfreeze();

    /**
     * @brief Whether the camera is currently frozen.
     */
    [[nodiscard]] bool isFrozen() const;

    /**
     * @brief Increment use_count.
     */
    void reference();

    /**
     * @brief Decrement use_count.
     */
    void dereference();

    /**
     * @brief Observe the live use_count.
     */
    RSize getUseCount() const;

    /**
     * @brief Stage the latest device-side frame of both eyes into the recorder
     *        double-buffers; non-blocking.
     */
    void syncRecorderRecords();

    /**
     * @brief Block until the most recent syncRecorderRecords() copy has
     *        completed.
     */
    void waitForRecorderRecordsSync();

    /**
     * @brief Pull the staged recorder records (one entry per frame) from the host buffers,
     *        assembling each frame's two eyes into one stereo record.
     */
    std::vector<RS_WorldCameraStereoRecord> getRecorderRecords();

    /**
     * @brief True iff this camera carries a controllable pose signal (free
     *        stereo camera only).
     */
    bool hasControllerSignal();

    /**
     * @brief Push the host-side control signal (per-eye pose) to the device.
     */
    void syncControllerSignal();

    /**
     * @brief Read the staged controller signal (host snapshot).
     */
    RS_WorldCameraStereoSignal getControllerSignal();

    /**
     * @brief True iff this camera produces a controller record (free or
     *        rigid-body-linked stereo camera).
     */
    bool hasControllerRecord();

    /**
     * @brief Stage the latest controller record (both eyes) into the controller buffers.
     */
    void syncControllerRecord();

    /**
     * @brief Read the staged controller record (host snapshot), assembled from both eyes.
     */
    RS_WorldCameraStereoRecord getControllerRecord();

    /**
     * @brief Camera entity name from the active characterization.
     */
    std::string getEntityName() const;

    /**
     * @brief Name of the recorder record for this camera.
     */
    std::string getRecordName() const;

    /**
     * @brief Name of the controller signal for this camera.
     */
    std::string getSignalName() const;

    /**
     * @brief Name of the controller this camera's record is sent to
     *        (record.target_controller_reference).
     */
    std::string getRecordControllerName() const;

    /**
     * @brief Name of the recorder this camera's record is sent to
     *        (record.target_recorder_reference).
     */
    std::string getRecordRecorderName() const;

    /**
     * @brief Name of the controller that drives this camera's signal
     *        (signal.source_controller_reference).
     */
    std::string getSignalControllerName() const;

    /**
     * @brief Global ID of the camera entity.
     */
    RIndex getEntityGid() const;

    /**
     * @brief Global ID of the recorder record entry.
     */
    RIndex getRecordGid() const;

    /**
     * @brief Global ID of the controller signal entry.
     */
    RIndex getSignalGid() const;

    /**
     * @brief Linked structure name for rigid-body-linked cameras; empty for
     *        free cameras.
     */
    std::string getStructureName() const;

    /**
     * @brief Linked structure gid for rigid-body-linked cameras;
     *        RS_NULL_INDEX for free cameras.
     */
    RIndex getStructureGid() const;

    /**
     * @brief Read-only access to the active camera characterization.
     */
    [[nodiscard]] const SupportedRSC &getRSC() const;

   private:
    RS_KernelManager *_kernel_manager = nullptr;
    std::unique_ptr<RS_MemoryScopedProvider> _scoped_device_provider;

    RS_KernelGidRegistration _entity_registration;
    RS_KernelGidRegistration _record_registration;
    RS_KernelGidRegistration _signal_registration;
    RS_KernelStructureVoxelReference _structure_reference;

    bool _is_kernel_data_inited = false;

    /// True while the camera is frozen, so it stops rendering and its canvas holds the last frame.
    bool _is_frozen = false;

    SupportedRSC _rsc;

    RSize _width = 0;

    RSize _height = 0;

    /// Recorder and controller buffers, one per eye.
    device::RS_DoubleBuffer _recorder_buffer[2];
    device::RS_DirectBuffer _controller_buffer[2];

    /// RGB canvas, one per eye.
    logic::RS_CameraRGBCanvas _camera_current_rgb_canvas[2];
    logic::RS_CameraRGBCanvas _camera_next_rgb_canvas[2];

    /// Shared primary AOVs (basic + restir), one per eye.
    logic::RS_CameraAuxNormal _camera_aux_normal[2];
    logic::RS_CameraAuxDepthZ _camera_aux_depth_z[2];
    logic::RS_CameraAuxHitPosition _camera_aux_hit_position[2];
    logic::RS_CameraAuxObjectKey _camera_aux_object_key[2];
    logic::RS_CameraAuxRoughness _camera_aux_roughness[2];
    logic::RS_CameraAuxScreenMotion _camera_aux_screen_motion[2];
    /// Restir-only endpoint + chain AOVs, one per eye.
    logic::RS_CameraAuxHitPosition _camera_aux_endpoint_hit_position[2];
    logic::RS_CameraAuxNormal _camera_aux_endpoint_normal[2];
    logic::RS_CameraAuxObjectKey _camera_aux_endpoint_object_key[2];
    logic::RS_CameraAuxAlbedo _camera_aux_endpoint_albedo[2];
    logic::RS_CameraAuxRoughness _camera_aux_endpoint_roughness[2];
    logic::RS_CameraAuxMetalness _camera_aux_endpoint_metalness[2];
    logic::RS_CameraAuxIntensity _camera_aux_endpoint_intensity[2];
    logic::RS_CameraAuxMaterialType _camera_aux_endpoint_material_type[2];
    logic::RS_CameraAuxScreenMotion _camera_aux_endpoint_screen_motion[2];
    logic::RS_CameraAuxChainThroughput _camera_aux_chain_throughput[2];
    logic::RS_CameraAuxChainTerminalType _camera_aux_chain_terminal_type[2];
    logic::RS_CameraReservoirDI _camera_prev_reservoir_di[2];
    logic::RS_CameraReservoirDI _camera_current_reservoir_di[2];
    logic::RS_CameraReservoirGI _camera_prev_reservoir_gi[2];
    logic::RS_CameraReservoirGI _camera_current_reservoir_gi[2];
    logic::RS_CameraReservoirDI _camera_spatial_reservoir_di[2];
    logic::RS_CameraReservoirGI _camera_spatial_reservoir_gi[2];
    logic::RS_CameraPrevAuxNormal _camera_prev_aux_normal[2];
    logic::RS_CameraPrevAuxDepthZ _camera_prev_aux_depth_z[2];
    logic::RS_CameraPrevAuxObjectKey _camera_prev_aux_object_key[2];
    logic::RS_CameraDenoiserRGBHistory _camera_denoiser_rgb_history[2];
    logic::RS_CameraDenoiserMomentHistory _camera_denoiser_moment_history[2];
    logic::RS_CameraDenoiserHistoryLength _camera_denoiser_history_length[2];
    logic::RS_CameraDenoiserRGBHistory _camera_prev_denoiser_rgb_history[2];
    logic::RS_CameraDenoiserMomentHistory _camera_prev_denoiser_moment_history[2];
    logic::RS_CameraDenoiserHistoryLength _camera_prev_denoiser_history_length[2];
    logic::RS_CameraDenoiserVariance _camera_denoiser_variance[2];
    logic::RS_CameraDenoiserVariance _camera_next_denoiser_variance[2];

    /// Signal data. Both eyes share one stereo signal (per-eye pose) published into the context's
    /// camera signal array as one entry; each sub-camera reads its eye by its sub_camera_index.
    RHandle _camera_signal = RS_NULL_HANDLE;
    RS_MemoryPtr _host_signal_buffer;
    RS_MemoryPtr _device_signal_buffer;
    RS_WorldCameraStereoSignal _host_signal;
    RS_WorldCameraStereoSignal _device_signal;

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

    /// Live use_count from external consumers. No consumer today, but the API stays uniform.
    RSize _use_count = 0;

    /**
     * @brief Concrete view of this camera's record characterization.
     */
    const rsc::RSC_CameraStereoRecord &_record() const {
        if (_rsc.is<rsc::RSC_CameraStereoRigidBodyLinked>()) {
            return _rsc.as<rsc::RSC_CameraStereoRigidBodyLinked>().record;
        }
        return _rsc.as<rsc::RSC_CameraStereoFree>().record;
    }

    /**
     * @brief Concrete view of this camera's base characterization.
     */
    const rsc::RSC_CameraStereoBase &_base() const {
        if (_rsc.is<rsc::RSC_CameraStereoRigidBodyLinked>()) {
            return _rsc.as<rsc::RSC_CameraStereoRigidBodyLinked>();
        }
        return _rsc.as<rsc::RSC_CameraStereoFree>();
    }

    /** @brief Assemble one host stereo record by pointing its left_/right_ fields at both eyes. */
    RS_WorldCameraStereoRecord
    _assembleStereoRecord(logic::RS_CameraRecordMem &left, logic::RS_CameraRecordMem &right) const;

    /** @brief Allocate and register this camera's shared per-eye-pose signal buffers. */
    void _addSignals();

    /** @brief Build and upload this camera's kernel mem, records, and layouts for both eyes. */
    void _initKernelData();

    /** @brief Release this camera's kernel mem, records, and layouts for both eyes. */
    void _freeKernelData();

    /** @brief Build and upload this camera's controller signal kernel data. */
    void _initKernelSignalData();

    /** @brief Release this camera's controller signal kernel data. */
    void _freeKernelSignalData();

    /** @brief Toggle both eyes' layouts so render launches skip a frozen camera. */
    void _freezeKernelData(bool is_freeze);
};

/**
 * @brief References a camera on construction and dereferences it on destruction.
 */
class RS_KernelCameraStereoReference {
   public:
    RS_KernelCameraStereoReference() = default;
    RS_KernelCameraStereoReference(RS_KernelManager *kernel_manager, std::string name);
    ~RS_KernelCameraStereoReference();

    RS_KernelCameraStereoReference(RS_KernelCameraStereoReference &&other) noexcept;
    RS_KernelCameraStereoReference &operator=(RS_KernelCameraStereoReference &&other) noexcept;
    RS_KernelCameraStereoReference(const RS_KernelCameraStereoReference &) = delete;
    RS_KernelCameraStereoReference &operator=(const RS_KernelCameraStereoReference &) = delete;

    [[nodiscard]] const std::string &name() const { return _name; }
    [[nodiscard]] RIndex gid() const { return _gid; }

   private:
    void _release();

    RS_KernelManager *_kernel_manager = nullptr;
    std::string _name;
    RIndex _gid = RS_NULL_INDEX;
};

}  // namespace rise

#endif  // RS_KERNEL_CAMERA_STEREO_MANAGER_H
