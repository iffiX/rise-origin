// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_TORQUE_MANAGER_H
#define RS_KERNEL_TORQUE_MANAGER_H

#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/rsc/signal/rsc_mechanical.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rs_kernel_gid_registration.h"
#include "rs_kernel_structure_voxel_manager.h"

namespace rise {

class RS_KernelManager;

/**
 * @brief Manager for a single torque signal.
 */
class RS_KernelTorqueManager {
   public:
    using SupportedRSC = RS_Variant<rsc::RSC_Torque>;

    /**
     * @brief Bind the signal, reference its emitting structure, and build the host data. Torque
     *        applies to rigid bodies only.
     */
    RS_KernelTorqueManager(
        RS_KernelManager *kernel_manager, const SupportedRSC &torque_rsc
    );

    /**
     * @brief Commit the built signal to the kernel context.
     */
    void commit();

    /**
     * @brief Decommit the signal from the kernel context.
     */
    void decommit();

    /**
     * @brief True iff a controllable signal value is present (true once
     *        initialized; kept for the uniform controller sync loop).
     */
    bool hasControllerSignal();

    /**
     * @brief Push the host-side torque signal buffer to the device.
     */
    void syncControllerSignal();

    /**
     * @brief Read the staged controller signal (host snapshot).
     */
    RS_WorldTorqueSignal getControllerSignal();

    /**
     * @brief Read-only access to the bound signal characterization.
     */
    const SupportedRSC &getRSC() const;

    /**
     * @brief Name of the bound signal.
     */
    std::string getSignalName() const;

    /**
     * @brief Name of the structure this torque targets.
     */
    std::string getStructureName() const;

    /**
     * @brief Global id of the bound signal entry.
     */
    RIndex getSignalGid() const;

    /**
     * @brief Global id of the structure this torque targets.
     */
    RIndex getStructureGid() const;

   private:
    RS_KernelManager *_kernel_manager = nullptr;
    std::unique_ptr<RS_MemoryScopedProvider> _scoped_provider;

    RS_KernelStructureVoxelReference _structure_reference;
    RS_KernelGidRegistration _signal_registration;

    bool _is_kernel_data_inited = false;

    SupportedRSC _rsc;

    // Torque signal data
    RHandle _torque_signal = RS_NULL_HANDLE;
    RS_MemoryPtr _host_signal_buffer;
    RS_MemoryPtr _device_signal_buffer;
    RS_WorldTorqueSignal _host_signal;
    RS_WorldTorqueSignal _device_signal;

    // Torque is rigid-only; the resolved rigid body structure scope ID
    RIndex _rigid_body_sid = RS_NULL_INDEX;

    void _initKernelData();

    void _freeKernelData();
};

}  // namespace rise

#endif  // RS_KERNEL_TORQUE_MANAGER_H
