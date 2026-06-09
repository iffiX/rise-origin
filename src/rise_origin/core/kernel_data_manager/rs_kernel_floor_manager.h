// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_FLOOR_MANAGER_H
#define RS_KERNEL_FLOOR_MANAGER_H

#include "rise_origin/core/rs_context.h"
#include "rise_origin/logic/mesh/rs_mesh_triangle_m.h"
#include "rise_origin/rsc/rs_floor_config.h"

namespace rise {

class RS_KernelManager;

class RS_KernelFloorManager {
   public:
    RS_KernelFloorManager() = default;

    void init(RS_KernelManager *kernel_manager, const RS_FloorConfig &floor_config);

    void free();

    [[nodiscard]] const RS_FloorConfig &getConfig() const;

   private:
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

    RS_KernelManager *_kernel_manager = nullptr;

    RS_FloorConfig _config;

    RS_ArrayFloorContext _floor_ctx;
};

}  // namespace rise

#endif  // RS_KERNEL_FLOOR_MANAGER_H
