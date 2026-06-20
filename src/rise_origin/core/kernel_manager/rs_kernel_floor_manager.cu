// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/utils/rs_conf.h"
#include "rs_kernel_floor_manager.h"

using namespace std;

namespace rise {

/*****************************************************************
 * RS_KernelFloorManager
 *****************************************************************/
RS_KernelFloorManager::RS_KernelFloorManager(
    RS_KernelManager* kernel_manager, const SupportedRSC& floor_rsc
)
    : _kernel_manager(kernel_manager), _rsc(floor_rsc) {
    _scoped_provider =
        std::make_unique<RS_MemoryScopedProvider>(_kernel_manager->getDeviceProvider());
    auto* provider = _scoped_provider.get();

    // Reference the floor's material so it cannot be removed while the floor is alive.
    auto& material_manager = _kernel_manager->getKernelMaterialManager();
    std::string material_reference =
        floor_rsc.visit([](auto& c) -> std::string { return c.material_reference; });
    _material_reference = RS_KernelMaterialReference(_kernel_manager, material_reference);
    RIndex material_gid = material_manager.findMaterial(material_reference);

    _floor_ctx = RS_FloorMem{};
    _floor_ctx.material_gid = material_gid;

    if (floor_rsc.is<rsc::RSC_FloorFlat>()) {
        auto& flat = floor_rsc.as<rsc::RSC_FloorFlat>();
        _data_type = logic::RSE_VISION_FLOOR_DATA_FLAT;
        _uv_scale = flat.uv_scale;
        _floor_ctx.height = flat.height;
    } else {
        auto& arr = floor_rsc.as<rsc::RSC_FloorArray>();
        _data_type = logic::RSE_VISION_FLOOR_DATA_ARRAY;

        std::vector<RFloat> src_heights(arr.height.begin(), arr.height.end());

        // 2x upsample on host to add smoothness in the precomputed normals.
        RSize src_w = arr.x_values;
        RSize src_h = arr.y_values;
        RSize dst_w = src_w * 2;
        RSize dst_h = src_h * 2;
        std::vector<RFloat> upsampled;
        _upsample2x(
            src_heights, src_w, src_h, upsampled, dst_w, dst_h, arr.is_x_repeat, arr.is_y_repeat
        );

        RFloat cell_x = arr.x_size / RFloat(dst_w);
        RFloat cell_y = arr.y_size / RFloat(dst_h);
        std::vector<RVec3f> normals_host;
        _computeCellNormals(
            upsampled, dst_w, dst_h, cell_x, cell_y, arr.is_x_repeat, arr.is_y_repeat, normals_host
        );

        _floor_ctx.x_size = arr.x_size;
        _floor_ctx.y_size = arr.y_size;
        _floor_ctx.x_values = dst_w;
        _floor_ctx.y_values = dst_h;
        _floor_ctx.inv_cell_x = RFloat(dst_w) / arr.x_size;
        _floor_ctx.inv_cell_y = RFloat(dst_h) / arr.y_size;
        _floor_ctx.is_x_repeat = arr.is_x_repeat;
        _floor_ctx.is_y_repeat = arr.is_y_repeat;
        RFloat hmin = upsampled.empty() ? RFloat(0) : upsampled[0];
        RFloat hmax = hmin;
        for (RFloat h : upsampled) {
            if (h < hmin) hmin = h;
            if (h > hmax) hmax = h;
        }
        _floor_ctx.height_min = hmin;
        _floor_ctx.height_max = hmax;
        _floor_ctx.x_divmod = device::RS_DivMod<RSize>(dst_w);
        _floor_ctx.y_divmod = device::RS_DivMod<RSize>(dst_h);
        // Pick a large multiple of {x,y}_values within RSize range so adding it to any
        // realistic signed cell index produces a non-negative value with the same residue.
        constexpr RSize HALF_MAX = static_cast<RSize>(0x7FFFFFFFu) / 2u;
        _floor_ctx.x_wrap_offset = (HALF_MAX / dst_w) * dst_w;
        _floor_ctx.y_wrap_offset = (HALF_MAX / dst_h) * dst_h;

        _floor_ctx.heights.init({dst_h, dst_w}, provider);
        _floor_ctx.heights.write(upsampled.data());

        _floor_ctx.normals.init({dst_h, dst_w}, provider);
        _floor_ctx.normals.write(normals_host.data());

        _floor_triangles =
            _generateArrayFloorTriangles(upsampled, dst_w, dst_h, arr.x_size, arr.y_size);
    }

    provider->sync();
}

void RS_KernelFloorManager::commit() {
    // The material vision slot is committed before this, so the handle is resolvable.
    RHandle vision_material =
        _kernel_manager->getKernelMaterialManager().findVisionMaterialHandle(
            _material_reference.name()
        );
    _kernel_manager->enterKernelDataCriticalZone();
    _initKernelData(vision_material);
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;

    if (_data_type == logic::RSE_VISION_FLOOR_DATA_ARRAY) {
        _kernel_manager->getKernelSpatialManager().markFloorTriangleModified();
    }
}

void RS_KernelFloorManager::decommit() {
    if (_is_kernel_data_inited) {
        _kernel_manager->enterKernelDataCriticalZone();
        _freeKernelData();
        _kernel_manager->exitKernelDataCriticalZone();
        _is_kernel_data_inited = false;
        _kernel_manager->getKernelSpatialManager().markFloorTriangleModified();
    }
}

void RS_KernelFloorManager::freeze() { _is_frozen = true; }

void RS_KernelFloorManager::unfreeze() { _is_frozen = false; }

bool RS_KernelFloorManager::isFrozen() const { return _is_frozen; }

void RS_KernelFloorManager::reference() { _use_count += 1; }

void RS_KernelFloorManager::dereference() {
    if (_use_count == 0) {
        RS_THROW(RS_HostExecutionException("Floor use_count underflow on dereference"));
    }
    _use_count -= 1;
}

RSize RS_KernelFloorManager::getUseCount() const { return _use_count; }

const RS_KernelFloorManager::SupportedRSC& RS_KernelFloorManager::getRSC() const {
    return _rsc;
}

void RS_KernelFloorManager::_initKernelData(RHandle vision_material) {
    auto* provider = _kernel_manager->getDeviceProvider();
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;
        ctx.floor = _floor_ctx;
        auto& fvm = ctx.floor_vision_mem;
        fvm.data_type = _data_type;
        fvm.vision_material = vision_material;
        fvm.uv_scale = _uv_scale;

        if (!_floor_triangles.empty()) {
            ctx.mesh_triangles.init({static_cast<RSize>(_floor_triangles.size())}, provider);
            ctx.mesh_triangles.write(_floor_triangles.data());
        }

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelFloorManager::_freeKernelData() {
    bool is_array = _rsc.is<rsc::RSC_FloorArray>();

    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;
        ctx.floor_vision_mem.data_type = logic::RSE_VISION_FLOOR_DATA_NONE;

        if (is_array) {
            // The height and normal arrays are owned by the scoped provider.
            ctx.floor.heights = device::RS_Array<RFloat, 2>{};
            ctx.floor.normals = device::RS_Array<RVec3f, 2>{};

            ctx.mesh_triangles.free();
        }

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelFloorManager::_upsample2x(
    const std::vector<RFloat>& src,
    RSize src_w,
    RSize src_h,
    std::vector<RFloat>& dst,
    RSize dst_w,
    RSize dst_h,
    bool wrap_x,
    bool wrap_y
) {
    dst.assign(dst_w * dst_h, RFloat(0));
    for (RSize j = 0; j < dst_h; ++j) {
        RFloat fy = (RFloat(j) + RFloat(0.5)) * RFloat(src_h) / RFloat(dst_h) - RFloat(0.5);
        int y0 = static_cast<int>(std::floor(fy));
        RFloat ty = fy - RFloat(y0);
        RSize jy0 = _clampOrWrap(y0, src_h, wrap_y);
        RSize jy1 = _clampOrWrap(y0 + 1, src_h, wrap_y);
        for (RSize i = 0; i < dst_w; ++i) {
            RFloat fx = (RFloat(i) + RFloat(0.5)) * RFloat(src_w) / RFloat(dst_w) - RFloat(0.5);
            int x0 = static_cast<int>(std::floor(fx));
            RFloat tx = fx - RFloat(x0);
            RSize ix0 = _clampOrWrap(x0, src_w, wrap_x);
            RSize ix1 = _clampOrWrap(x0 + 1, src_w, wrap_x);
            RFloat h00 = src[jy0 * src_w + ix0];
            RFloat h10 = src[jy0 * src_w + ix1];
            RFloat h01 = src[jy1 * src_w + ix0];
            RFloat h11 = src[jy1 * src_w + ix1];
            RFloat h0 = h00 * (RFloat(1) - tx) + h10 * tx;
            RFloat h1 = h01 * (RFloat(1) - tx) + h11 * tx;
            dst[j * dst_w + i] = h0 * (RFloat(1) - ty) + h1 * ty;
        }
    }
}

void RS_KernelFloorManager::_computeCellNormals(
    const std::vector<RFloat>& heights,
    RSize w,
    RSize h,
    RFloat cell_x,
    RFloat cell_y,
    bool wrap_x,
    bool wrap_y,
    std::vector<RVec3f>& normals
) {
    normals.assign(w * h, RVec3f(0, 0, 1));
    for (RSize j = 0; j < h; ++j) {
        for (RSize i = 0; i < w; ++i) {
            RFloat dh_dx =
                (_sample(
                     heights, static_cast<int>(i) + 1, static_cast<int>(j), w, h, wrap_x, wrap_y
                 ) -
                 _sample(
                     heights, static_cast<int>(i) - 1, static_cast<int>(j), w, h, wrap_x, wrap_y
                 )) /
                (RFloat(2) * cell_x);
            RFloat dh_dy =
                (_sample(
                     heights, static_cast<int>(i), static_cast<int>(j) + 1, w, h, wrap_x, wrap_y
                 ) -
                 _sample(
                     heights, static_cast<int>(i), static_cast<int>(j) - 1, w, h, wrap_x, wrap_y
                 )) /
                (RFloat(2) * cell_y);
            RVec3f n(-static_cast<float>(dh_dx), -static_cast<float>(dh_dy), 1.0f);
            float norm = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
            if (norm > 1e-12f) {
                n.x /= norm;
                n.y /= norm;
                n.z /= norm;
            } else {
                n = RVec3f(0, 0, 1);
            }
            normals[j * w + i] = n;
        }
    }
}

RSize RS_KernelFloorManager::_clampOrWrap(int v, RSize n, bool wrap) {
    if (wrap) {
        int m = v % static_cast<int>(n);
        if (m < 0) m += static_cast<int>(n);
        return static_cast<RSize>(m);
    }
    if (v < 0) return 0;
    if (v >= static_cast<int>(n)) return n - 1;
    return static_cast<RSize>(v);
}

RFloat RS_KernelFloorManager::_sample(
    const std::vector<RFloat>& heights,
    int x,
    int y,
    RSize w,
    RSize h,
    bool wrap_x,
    bool wrap_y
) {
    RSize ix = _clampOrWrap(x, w, wrap_x);
    RSize iy = _clampOrWrap(y, h, wrap_y);
    return heights[iy * w + ix];
}

std::vector<logic::RS_MeshTriangleMem> RS_KernelFloorManager::_generateArrayFloorTriangles(
    const std::vector<RFloat>& heights,
    RSize cols,
    RSize rows,
    RFloat x_size,
    RFloat y_size
) {
    std::vector<logic::RS_MeshTriangleMem> triangles;
    if (cols < 2 || rows < 2) return triangles;

    RSize cell_cols = cols - 1;
    RSize cell_rows = rows - 1;
    triangles.resize(2 * cell_cols * cell_rows);

    RFloat half_x = x_size * RFloat(0.5);
    RFloat half_y = y_size * RFloat(0.5);
    RFloat dx = x_size / RFloat(cell_cols);
    RFloat dy = y_size / RFloat(cell_rows);

    for (RSize r = 0; r < cell_rows; ++r) {
        for (RSize c = 0; c < cell_cols; ++c) {
            RVec3f p00(
                float(-half_x + RFloat(c) * dx),
                float(-half_y + RFloat(r) * dy),
                float(heights[r * cols + c])
            );
            RVec3f p10(
                float(-half_x + RFloat(c + 1) * dx),
                float(-half_y + RFloat(r) * dy),
                float(heights[r * cols + (c + 1)])
            );
            RVec3f p01(
                float(-half_x + RFloat(c) * dx),
                float(-half_y + RFloat(r + 1) * dy),
                float(heights[(r + 1) * cols + c])
            );
            RVec3f p11(
                float(-half_x + RFloat(c + 1) * dx),
                float(-half_y + RFloat(r + 1) * dy),
                float(heights[(r + 1) * cols + (c + 1)])
            );

            // CCW so face normals point up (+z) on a flat patch.
            RSize tri_index = 2 * (r * cell_cols + c);

            logic::RS_MeshTriangleMem& tri_a = triangles[tri_index + 0];
            tri_a.vertex_0 = p00;
            tri_a.vertex_1 = p10;
            tri_a.vertex_2 = p11;
            RVec3f e1_a = p10 - p00;
            RVec3f e2_a = p11 - p00;
            tri_a.face_normal = e1_a.cross(e2_a).normalized();
            tri_a.face_tangent = e1_a.normalized();

            logic::RS_MeshTriangleMem& tri_b = triangles[tri_index + 1];
            tri_b.vertex_0 = p00;
            tri_b.vertex_1 = p11;
            tri_b.vertex_2 = p01;
            RVec3f e1_b = p11 - p00;
            RVec3f e2_b = p01 - p00;
            tri_b.face_normal = e1_b.cross(e2_b).normalized();
            tri_b.face_tangent = e1_b.normalized();
        }
    }
    return triangles;
}

/*****************************************************************
 * RS_KernelFloorReference
 *****************************************************************/
RS_KernelFloorReference::RS_KernelFloorReference(RS_KernelManager* kernel_manager)
    : _kernel_manager(kernel_manager) {
    auto& floor = _kernel_manager->getKernelFloorManager();
    if (!floor) {
        RS_THROW(RS_InvalidCharacterizationException("Floor is not registered in the kernel manager"));
    }
    floor->reference();
}

RS_KernelFloorReference::~RS_KernelFloorReference() { _release(); }

RS_KernelFloorReference::RS_KernelFloorReference(RS_KernelFloorReference&& other) noexcept
    : _kernel_manager(other._kernel_manager) {
    other._kernel_manager = nullptr;
}

RS_KernelFloorReference& RS_KernelFloorReference::operator=(RS_KernelFloorReference&& other
) noexcept {
    if (this != &other) {
        _release();
        _kernel_manager = other._kernel_manager;
        other._kernel_manager = nullptr;
    }
    return *this;
}

void RS_KernelFloorReference::_release() {
    if (_kernel_manager == nullptr) return;
    if (auto& floor = _kernel_manager->getKernelFloorManager()) {
        floor->dereference();
    }
    _kernel_manager = nullptr;
}

}  // namespace rise
