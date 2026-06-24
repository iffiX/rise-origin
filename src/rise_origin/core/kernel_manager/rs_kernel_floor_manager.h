// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_FLOOR_MANAGER_H
#define RS_KERNEL_FLOOR_MANAGER_H

#include <string>
#include <vector>

#include "rise_origin/core/kernel_manager/rs_kernel_material_manager.h"
#include "rise_origin/core/rs_context.h"
#include "rise_origin/logic/mesh/rs_mesh_triangle_m.h"
#include "rise_origin/rsc/entity/rsc_floor.h"
#include "rise_origin/utils/memory/rs_memory.h"

namespace rise {

class RS_KernelManager;

class RS_KernelFloorManager {
   public:
    using SupportedRSC = rsc::RSC_FloorVariant;

    /**
     * @brief Bind to the kernel manager, reference the floor's material, and generate the floor
     *        mesh host side.
     */
    RS_KernelFloorManager(RS_KernelManager *kernel_manager, const SupportedRSC &floor_rsc);

    /**
     * @brief Commit the floor to the kernel context.
     */
    void commit();

    /**
     * @brief Decommit the floor from the kernel context.
     */
    void decommit();

    /**
     * @brief Mark the floor frozen. Only records the state for dispatch uniformity.
     */
    void freeze();

    /**
     * @brief Mark the floor unfrozen. Only records the state for dispatch uniformity.
     */
    void unfreeze();

    /**
     * @brief Whether the floor is currently frozen.
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
    [[nodiscard]] RSize getUseCount() const;

    /**
     * @brief Read-only access to the active floor characterization.
     */
    [[nodiscard]] const SupportedRSC &getRSC() const;

   private:
    RS_KernelManager *_kernel_manager = nullptr;
    std::unique_ptr<RS_MemoryScopedProvider> _scoped_device_provider;

    RS_KernelMaterialReference _material_reference;

    bool _is_kernel_data_inited = false;

    /// Frozen state tracked for dispatch uniformity.
    bool _is_frozen = false;

    /// Live use_count from external consumers.
    RSize _use_count = 0;

    SupportedRSC _rsc;

    RS_FloorMem _floor_ctx;

    /// Floor data built host side and published at commit.
    logic::RSE_VisionFloorDataType _data_type = logic::RSE_VISION_FLOOR_DATA_NONE;
    RFloat _uv_scale = 1;
    std::vector<logic::RS_MeshTriangleMem> _floor_triangles;

    // Publish the prepared floor data into the kernel context.
    void _initKernelData(RHandle vision_material);

    // Clear the floor data from the kernel context.
    void _freeKernelData();

    // 2x upsample with bilinear interpolation. Input samples are treated as cell
    // centers; output uses the same layout at 2x resolution along each axis.
    static void _upsample2x(
        const std::vector<RFloat> &src,
        RSize src_w,
        RSize src_h,
        std::vector<RFloat> &dst,
        RSize dst_w,
        RSize dst_h,
        bool wrap_x,
        bool wrap_y
    );

    // Per-cell unit surface normal from central differences of the height grid.
    static void _computeCellNormals(
        const std::vector<RFloat> &heights,
        RSize w,
        RSize h,
        RFloat cell_x,
        RFloat cell_y,
        bool wrap_x,
        bool wrap_y,
        std::vector<RVec3f> &normals
    );

    // Clamp (or wrap, when repeating) a signed cell index into [0, n).
    static RSize _clampOrWrap(int v, RSize n, bool wrap);

    // Fetch a height sample with edge clamping/wrapping on both axes.
    static RFloat _sample(
        const std::vector<RFloat> &heights,
        int x,
        int y,
        RSize w,
        RSize h,
        bool wrap_x,
        bool wrap_y
    );

    // CCW triangle pair tiling the heightfield, vertices in world space.
    static std::vector<logic::RS_MeshTriangleMem> _generateArrayFloorTriangles(
        const std::vector<RFloat> &heights,
        RSize cols,
        RSize rows,
        RFloat x_size,
        RFloat y_size
    );
};

/**
 * @brief References the floor on construction and dereferences it on destruction.
 */
class RS_KernelFloorReference {
   public:
    RS_KernelFloorReference() = default;
    RS_KernelFloorReference(RS_KernelManager *kernel_manager);
    ~RS_KernelFloorReference();

    RS_KernelFloorReference(RS_KernelFloorReference &&other) noexcept;
    RS_KernelFloorReference &operator=(RS_KernelFloorReference &&other) noexcept;
    RS_KernelFloorReference(const RS_KernelFloorReference &) = delete;
    RS_KernelFloorReference &operator=(const RS_KernelFloorReference &) = delete;

   private:
    void _release();

    RS_KernelManager *_kernel_manager = nullptr;
};

}  // namespace rise

#endif  // RS_KERNEL_FLOOR_MANAGER_H
