// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_TRANSFORM_MANAGER_H
#define RS_KERNEL_TRANSFORM_MANAGER_H

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
 * @brief Manager for a single transform signal.
 */
class RS_KernelTransformManager {
   public:
    using SupportedRSC = RS_Variant<rsc::RSC_Transform>;

    /**
     * @brief Bind the signal, reference its emitting structure, and build the host data. Transform
     *        applies to the whole structure.
     */
    RS_KernelTransformManager(RS_KernelManager *kernel_manager, const SupportedRSC &transform_rsc);

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
     * @brief Push the host-side transform signal buffer to the device, then reset it to identity so
     *        the transform applies once.
     */
    void syncControllerSignal();

    /**
     * @brief Read the staged controller signal (host snapshot).
     */
    RS_WorldTransformSignal getControllerSignal();

    /**
     * @brief Whether a synced transform that moves static voxels is not yet applied.
     */
    bool isStaticRebuildPending() const;

    /**
     * @brief Clear the static rebuild pending flag.
     */
    void clearStaticRebuildPending();

    /**
     * @brief Read-only access to the bound signal characterization.
     */
    const SupportedRSC &getRSC() const;

    /**
     * @brief Name of the bound signal.
     */
    std::string getSignalName() const;

    /**
     * @brief Name of the structure this transform targets.
     */
    std::string getStructureName() const;

    /**
     * @brief Global id of the bound signal entry.
     */
    RIndex getSignalGid() const;

    /**
     * @brief Global id of the structure this transform targets.
     */
    RIndex getStructureGid() const;

   private:
    RS_KernelManager *_kernel_manager = nullptr;
    std::unique_ptr<RS_MemoryScopedProvider> _scoped_device_provider;

    RS_KernelStructureVoxelReference _structure_reference;
    RS_KernelGidRegistration _signal_registration;

    bool _is_kernel_data_inited = false;
    bool _is_static_rebuild_pending = false;

    SupportedRSC _rsc;

    // Transform signal data
    RHandle _transform_signal = RS_NULL_HANDLE;
    RS_MemoryPtr _host_signal_buffer;
    RS_MemoryPtr _device_signal_buffer;
    RS_WorldTransformSignal _host_signal;
    RS_WorldTransformSignal _device_signal;

    void _initKernelData();

    void _freeKernelData();

    static bool _isIdentitySignal(const RS_WorldTransformSignal &signal);

    static void _writeIdentitySignal(RS_WorldTransformSignal &signal);
};

}  // namespace rise

#endif  // RS_KERNEL_TRANSFORM_MANAGER_H
