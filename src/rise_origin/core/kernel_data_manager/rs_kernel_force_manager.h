// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_FORCE_MANAGER_H
#define RS_KERNEL_FORCE_MANAGER_H

#include "rise_origin/core/rs_simulation_signal.h"
#include "rise_origin/rsc/rs_signal_config.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_memory.h"

namespace rise {

class RS_KernelManager;

class RS_KernelForceManager {
   public:
    RS_KernelForceManager();

    void init(RS_KernelManager *kernel_manager, const RS_SignalConfig &config);

    void free();

    bool hasControllerSignal();

    void syncControllerSignal();

    RS_SimulationForceSignal getControllerSignal();

    const RS_SignalConfig &getConfig() const;

    std::string getName() const;

   private:
    RS_KernelManager *_kernel_manager = nullptr;

    RIndex _structure_gid = RS_NULL_INDEX;
    RS_SignalConfig _config;

    bool _is_kernel_data_inited = false;

    // Force signal data
    // For device side, the signals directly point to the
    // corresponding signal array region in context
    RSize _device_force_signal_offset = 0;
    RS_MemoryPtr _host_force_signal_buffer;

    // Track whether this is for rigid body or soft voxels
    bool _is_rigid = false;
    RIndex _rigid_body_sid = RS_NULL_INDEX;  // Only valid if _is_rigid is true
    RSize _soft_voxel_count = 0;             // Only valid if _is_rigid is false

    void _initKernelData();

    void _freeKernelData();
};

}  // namespace rise

#endif  // RS_KERNEL_FORCE_MANAGER_H
