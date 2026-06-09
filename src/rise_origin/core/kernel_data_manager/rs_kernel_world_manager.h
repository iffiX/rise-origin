// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_WORLD_MANAGER_H
#define RS_KERNEL_WORLD_MANAGER_H

#include "hosek-wilkie/hosek-wilkie.hpp"
#include "rise_origin/rsc/rs_world_config.h"

namespace rise {

class RS_KernelManager;

/// Owns the lifetime of the cooked Hosek-Wilkie sky state and writes the world
/// render fields onto the kernel. Modelled on RS_KernelFloorManager.
class RS_KernelWorldManager {
   public:
    RS_KernelWorldManager() = default;

    /// Attach to the kernel manager and apply the initial world config.
    void init(RS_KernelManager *kernel_manager, const RS_WorldConfig &world_config);

    /// Release the Hosek-Wilkie cooked state, if any.
    void free();

    /// Re-apply a world config: rewrites the kernel.world_* render fields and
    /// re-cooks the Hosek-Wilkie state when needed (any prior cooked state is
    /// released first).
    void setWorld(const RS_WorldConfig &world_config);

    [[nodiscard]] const RS_WorldConfig &getConfig() const;

   private:
    /// Per-channel Hosek-Wilkie radiance at (cos_theta, cos_gamma), mirroring
    /// the kernel-side _evalHosekWilkieSky formula in double precision.
    static double _evalHosekChannel(
        const ArHosekSkyModelState *state, int ch, double cos_theta, double cos_gamma
    );

    RS_KernelManager *_kernel_manager = nullptr;

    RS_WorldConfig _config;

    ArHosekSkyModelState *_hosek_state = nullptr;
};

}  // namespace rise

#endif  // RS_KERNEL_WORLD_MANAGER_H
