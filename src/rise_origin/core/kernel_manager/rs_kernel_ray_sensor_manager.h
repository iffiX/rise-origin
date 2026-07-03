// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_RAY_SENSOR_MANAGER_H
#define RS_KERNEL_RAY_SENSOR_MANAGER_H

#include <string>
#include <vector>

#include "rise_origin/core/kernel_manager/rs_kernel_gid_registration.h"
#include "rise_origin/core/kernel_manager/rs_kernel_structure_voxel_manager.h"
#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/device/data/rs_buffer.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/logic/ray/rs_ray_sensor_m.h"
#include "rise_origin/rsc/entity/rsc_ray_sensor.h"
#include "rise_origin/utils/data/rs_variant.h"

namespace rise {

class RS_KernelManager;

/**
 * @brief Owns one ray sensor (free or rigid-body-linked), its ray direction buffer, and the
 *        recorder/controller plumbing for hit records and pose signals.
 */
class RS_KernelRaySensorManager {
   public:
    using SupportedRSC = RS_Variant<rsc::RSC_RaySensorFree, rsc::RSC_RaySensorRigidBodyLinked>;

    /**
     * @brief Bind to the kernel manager, allocate the ray direction buffer, and, for the
     *        rigid-body-linked variant, take a reference on the host structure.
     */
    RS_KernelRaySensorManager(RS_KernelManager *kernel_manager, const SupportedRSC &ray_sensor_rsc);

    /**
     * @brief Commit the ray sensor to the kernel context.
     */
    void commit();

    /**
     * @brief Decommit the ray sensor from the kernel context.
     */
    void decommit();

    /**
     * @brief Stop tracing the ray sensor while keeping its last record.
     */
    void freeze();

    /**
     * @brief Resume tracing a frozen ray sensor.
     */
    void unfreeze();

    /**
     * @brief Whether the ray sensor is currently frozen.
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
     * @brief Stage the latest device-side hit record into the recorder double-buffer.
     */
    void syncRecorderRecords();

    /**
     * @brief Block until the most recent syncRecorderRecords() copy has completed.
     */
    void waitForRecorderRecordsSync();

    /**
     * @brief Pull the staged recorder records (one entry per frame) from the host buffer.
     */
    std::vector<RS_WorldRaySensorRecord> getRecorderRecords();

    /**
     * @brief True iff this ray sensor carries a controllable pose signal (free variant only).
     */
    bool hasControllerSignal();

    /**
     * @brief Push the host-side control signal (pose) to the device.
     */
    void syncControllerSignal();

    /**
     * @brief Read the staged controller signal (host snapshot).
     */
    RS_WorldRaySensorSignal getControllerSignal();

    /**
     * @brief True iff this ray sensor produces a controller record (free or rigid-body-linked).
     */
    bool hasControllerRecord();

    /**
     * @brief Stage the latest controller record (hit record) into the controller buffer.
     */
    void syncControllerRecord();

    /**
     * @brief Read the staged controller record (host snapshot).
     */
    RS_WorldRaySensorRecord getControllerRecord();

    /**
     * @brief Ray sensor entity name from the active characterization.
     */
    std::string getEntityName() const;

    /**
     * @brief Name of the recorder record for this ray sensor.
     */
    std::string getRecordName() const;

    /**
     * @brief Name of the controller signal for this ray sensor.
     */
    std::string getSignalName() const;

    /**
     * @brief Name of the controller this ray sensor's record is sent to.
     */
    std::string getRecordControllerName() const;

    /**
     * @brief Name of the recorder this ray sensor's record is sent to.
     */
    std::string getRecordRecorderName() const;

    /**
     * @brief Name of the controller that drives this ray sensor's signal.
     */
    std::string getSignalControllerName() const;

    /**
     * @brief Global ID of the ray sensor entity.
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
     * @brief Linked structure name for rigid-body-linked ray sensors; empty for free.
     */
    std::string getStructureName() const;

    /**
     * @brief Linked structure gid for rigid-body-linked ray sensors; RS_NULL_INDEX for free.
     */
    RIndex getStructureGid() const;

    /**
     * @brief Read-only access to the active ray sensor characterization.
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

    /// True while the ray sensor is frozen, so it stops tracing and holds the last record.
    bool _is_frozen = false;

    SupportedRSC _rsc;

    RSize _ray_num = 0;

    RFloat _max_range = 0;

    device::RS_DoubleBuffer _recorder_buffer;

    device::RS_DirectBuffer _controller_buffer;

    /// Static ray directions relative to the sensor origin.
    device::RS_Array<RVec3rf, 1> _ray_directions;

    /// Signal data. The ray sensor pose signal is backed by per-manager host and device buffers
    /// and published into the context's ray sensor signal array as one entry.
    RHandle _ray_sensor_signal = RS_NULL_HANDLE;
    RS_MemoryPtr _host_signal_buffer;
    RS_MemoryPtr _device_signal_buffer;
    RS_WorldRaySensorSignal _host_signal;
    RS_WorldRaySensorSignal _device_signal;

    /// Sensor rigid body relative position, only valid if linked to a rigid body.
    RVec3rf _ray_sensor_rb_position = RVec3rf::zero();

    /// Sensor rigid body relative orientation, only valid if linked to a rigid body.
    RQuat3rf _ray_sensor_rb_orientation = RQuat3rf::unit();

    /// Sensor rigid body structure scope id, only valid if linked to a rigid body.
    RIndex _rigid_body_sid = RS_NULL_INDEX;

    /// Sensor origin position, only valid if the sensor is a free sensor.
    RVec3rf _origin_position = RVec3rf::zero();

    /// Sensor orientation, only valid if the sensor is a free sensor.
    RQuat3rf _orientation = RQuat3rf::unit();

    /// Live use_count from external consumers.
    RSize _use_count = 0;

    /**
     * @brief Concrete view of this ray sensor's record characterization.
     */
    const rsc::RSC_RaySensorRecord &_record() const {
        if (_rsc.is<rsc::RSC_RaySensorRigidBodyLinked>()) {
            return _rsc.as<rsc::RSC_RaySensorRigidBodyLinked>().record;
        }
        return _rsc.as<rsc::RSC_RaySensorFree>().record;
    }

    /** @brief Allocate and register this ray sensor's controller signal buffers. */
    void _addSignals();

    /** @brief Build and upload this ray sensor's kernel mem, records, and layouts. */
    void _initKernelData();

    /** @brief Release this ray sensor's kernel mem, records, and layouts. */
    void _freeKernelData();

    /** @brief Build and upload this ray sensor's controller signal kernel data. */
    void _initKernelSignalData();

    /** @brief Release this ray sensor's controller signal kernel data. */
    void _freeKernelSignalData();

    /** @brief Toggle every ray sensor layout so trace launches skip a frozen sensor. */
    void _freezeKernelData(bool is_freeze);
};

/**
 * @brief References a ray sensor on construction and dereferences it on destruction.
 */
class RS_KernelRaySensorReference {
   public:
    RS_KernelRaySensorReference() = default;
    RS_KernelRaySensorReference(RS_KernelManager *kernel_manager, std::string name);
    ~RS_KernelRaySensorReference();

    RS_KernelRaySensorReference(RS_KernelRaySensorReference &&other) noexcept;
    RS_KernelRaySensorReference &operator=(RS_KernelRaySensorReference &&other) noexcept;
    RS_KernelRaySensorReference(const RS_KernelRaySensorReference &) = delete;
    RS_KernelRaySensorReference &operator=(const RS_KernelRaySensorReference &) = delete;

    [[nodiscard]] const std::string &name() const { return _name; }
    [[nodiscard]] RIndex gid() const { return _gid; }

   private:
    void _release();

    RS_KernelManager *_kernel_manager = nullptr;
    std::string _name;
    RIndex _gid = RS_NULL_INDEX;
};

}  // namespace rise

#endif  // RS_KERNEL_RAY_SENSOR_MANAGER_H
