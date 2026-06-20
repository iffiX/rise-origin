// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_COSMIC_BACKGROUND_MANAGER_H
#define RS_KERNEL_COSMIC_BACKGROUND_MANAGER_H

#include <string>

#include "hosek-wilkie/hosek-wilkie.hpp"
#include "rise_origin/core/kernel_manager/rs_kernel_texture_manager.h"
#include "rise_origin/rsc/entity/rsc_cosmic_background.h"
#include "rise_origin/utils/data/rs_variant.h"

namespace rise {

class RS_KernelManager;

/**
 * @brief Owns the cosmic background. Writes ctx.cosmic_background_vision_mem on the kernel
 *        and owns the cooked Hosek-Wilkie sky state. The Uniform background resolves its
 *        texture_reference through the texture manager (constant color or procedural grid).
 */
class RS_KernelCosmicBackgroundManager {
   public:
    using SupportedRSC = RS_Variant<
        rsc::RSC_CosmicBackgroundUniform,
        rsc::RSC_CosmicBackgroundGradient,
        rsc::RSC_CosmicBackgroundHosekWilkie>;

    /**
     * @brief Bind to the kernel manager, reference the Uniform texture, and cook the background
     *        state host side.
     */
    RS_KernelCosmicBackgroundManager(RS_KernelManager *kernel_manager, const SupportedRSC &cosmic_rsc);

    /**
     * @brief Commit the cosmic background to the kernel context.
     */
    void commit();

    /**
     * @brief Decommit the cosmic background from the kernel context.
     */
    void decommit();

    /**
     * @brief Mark the cosmic background frozen. Only records the state for dispatch uniformity.
     */
    void freeze();

    /**
     * @brief Mark the cosmic background unfrozen. Only records the state for dispatch uniformity.
     */
    void unfreeze();

    /**
     * @brief Whether the cosmic background is currently frozen.
     */
    [[nodiscard]] bool isFrozen() const;

    /**
     * @brief Increment use_count (mark this cosmic background as not-removable).
     */
    void reference();

    /**
     * @brief Decrement use_count.
     */
    void dereference();

    /**
     * @brief Observe the live use_count.
     */
    [[nodiscard]] RSize getUseCount() const;

    /**
     * @brief Read-only access to the active background characterization.
     */
    [[nodiscard]] const SupportedRSC &getRSC() const;

   private:
    RS_KernelManager *_kernel_manager = nullptr;

    RS_KernelTextureReference _texture_reference;

    SupportedRSC _rsc;

    bool _is_kernel_data_inited = false;

    /// Frozen state tracked for dispatch uniformity; a cosmic background has no kernel data to pause.
    bool _is_frozen = false;

    /// Live use_count from external consumers.
    RSize _use_count = 0;

    /**
     * @brief Dispatch the active variant to the matching update path.
     */
    void _updateKernelData();

    /**
     * @brief Uniform background: resolve texture_reference through the texture manager to a
     *        constant color (CONSTANT) or a procedural grid handle (GRID).
     */
    void _updateKernelUniformData(const rsc::RSC_CosmicBackgroundUniform &uniform);

    /**
     * @brief Two-tone gradient sky with sun disk.
     */
    void _updateKernelGradientData(const rsc::RSC_CosmicBackgroundGradient &gradient);

    /**
     * @brief Hosek-Wilkie analytic clear-sky; cooks the 27-coefficient state on the host.
     */
    void _updateKernelHosekWilkieData(const rsc::RSC_CosmicBackgroundHosekWilkie &hosek);

    /// Per-channel Hosek-Wilkie radiance at (cos_theta, cos_gamma), mirroring
    /// the kernel-side _evalCosmicBackgroundHosekWilkie formula in double precision.
    static double _evalHosekChannel(
        const ArHosekSkyModelState *state, int ch, double cos_theta, double cos_gamma
    );
};

/**
 * @brief References the cosmic background on construction and dereferences it on destruction.
 */
class RS_KernelCosmicBackgroundReference {
   public:
    RS_KernelCosmicBackgroundReference() = default;
    RS_KernelCosmicBackgroundReference(RS_KernelManager *kernel_manager);
    ~RS_KernelCosmicBackgroundReference();

    RS_KernelCosmicBackgroundReference(RS_KernelCosmicBackgroundReference &&other) noexcept;
    RS_KernelCosmicBackgroundReference &operator=(RS_KernelCosmicBackgroundReference &&other
    ) noexcept;
    RS_KernelCosmicBackgroundReference(const RS_KernelCosmicBackgroundReference &) = delete;
    RS_KernelCosmicBackgroundReference &operator=(const RS_KernelCosmicBackgroundReference &) =
        delete;

   private:
    void _release();

    RS_KernelManager *_kernel_manager = nullptr;
};

}  // namespace rise

#endif  // RS_KERNEL_COSMIC_BACKGROUND_MANAGER_H
