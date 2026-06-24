// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_FORCE_MANAGER_H
#define RS_KERNEL_FORCE_MANAGER_H

#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/rsc/signal/rsc_mechanical.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_kernel_gid_registration.h"
#include "rs_kernel_structure_voxel_manager.h"

namespace rise {

class RS_KernelManager;

/**
 * @brief Manager for a single force signal.
 */
class RS_KernelForceManager {
   public:
    using SupportedRSC = RS_Variant<rsc::RSC_Force>;

    /**
     * @brief Bind the signal, reference its emitting structure, and build the host data.
     */
    RS_KernelForceManager(RS_KernelManager *kernel_manager, const SupportedRSC &force_rsc);

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
     * @brief Push the host-side force signal buffer to the device.
     */
    void syncControllerSignal();

    /**
     * @brief Read the staged controller signal (host snapshot).
     */
    RS_WorldForceSignal getControllerSignal();

    /**
     * @brief Read-only access to the bound signal characterization.
     */
    const SupportedRSC &getRSC() const;

    /**
     * @brief Name of the bound signal.
     */
    std::string getSignalName() const;

    /**
     * @brief Name of the structure this force targets.
     */
    std::string getStructureName() const;

    /**
     * @brief Global id of the bound signal entry.
     */
    RIndex getSignalGid() const;

    /**
     * @brief Global id of the structure this force targets.
     */
    RIndex getStructureGid() const;

   private:
    RS_KernelManager *_kernel_manager = nullptr;
    std::unique_ptr<RS_MemoryScopedProvider> _scoped_device_provider;

    RS_KernelStructureVoxelReference _structure_reference;
    RS_KernelGidRegistration _signal_registration;

    bool _is_kernel_data_inited = false;

    SupportedRSC _rsc;

    // Force signal data
    RHandle _force_signal = RS_NULL_HANDLE;
    RS_MemoryPtr _host_signal_buffer;
    RS_MemoryPtr _device_signal_buffer;
    RS_WorldForceSignal _host_signal;
    RS_WorldForceSignal _device_signal;

    // Track whether this is for rigid body or soft voxels
    bool _is_rigid = false;
    RIndex _rigid_body_sid = RS_NULL_INDEX;  // Only valid if _is_rigid is true
    RSize _soft_voxel_count = 0;             // Only valid if _is_rigid is false

    void _initKernelData();

    void _freeKernelData();
};

}  // namespace rise

#endif  // RS_KERNEL_FORCE_MANAGER_H
