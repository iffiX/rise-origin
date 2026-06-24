// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_TEXTURE_MANAGER_H
#define RS_KERNEL_TEXTURE_MANAGER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rise_origin/core/kernel_manager/rs_kernel_gid_registration.h"
#include "rise_origin/rsc/resource/rsc_texture.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

/**
 * @brief Color space of a texture.
 */
RS_ENUM(
    RSE_TextureColorSpace,
    unsigned int,
    (RSE_TEXTURE_COLOR_SPACE_LINEAR, 0),
    (RSE_TEXTURE_COLOR_SPACE_SRGB, 1)
)

/**
 * @brief Storage format of a texture.
 */
RS_ENUM(
    RSE_TextureStorageFormat,
    unsigned int,
    (RSE_TEXTURE_STORAGE_FORMAT_CONSTANT, 0),
    // 1-4 reserved
    (RSE_TEXTURE_STORAGE_FORMAT_GRID, 5)
)

class RS_KernelManager;

/**
 * @brief Pool manager for procedural textures, keyed by name.
 *
 * Lifecycle (uniform reference pattern):
 * - addTexture(texture_rsc) registers the characterization; no device allocation.
 * - referenceTexture(name, color_space, storage_format) binds the color space, does any host
 *   side preload such as image decode, increments use_count, and marks the texture dirty. The
 *   color_space / storage_format arguments are decode hints for image textures; constant and
 *   procedural textures carry an intrinsic format and ignore them.
 * - dereferenceTexture(name) decrements use_count and marks the texture dirty.
 * - commit() uploads the device grid slot for every dirty grid texture with a live reference
 *   and frees it for every dirty grid texture with no reference left.
 * - findTextureHandle(name) reads the grid slot offset, valid after commit().
 * - removeTexture(name) erases the characterization; throws if any consumer still holds a
 * reference.
 *
 * CONSTANT textures never allocate a device slot — callers use isConstant() +
 * getConstant() instead of holding a handle. GRID textures allocate a slot in
 * the context grid table (ctx.texture_grids) and the handle is its offset.
 */
class RS_KernelTextureManager {
   public:
    using SupportedRSC = rsc::RSC_TextureVariant;

    RS_KernelTextureManager() = default;

    void init(RS_KernelManager *kernel_manager);

    void free();

    /**
     * @brief True iff a texture with this name is registered.
     */
    bool hasTexture(const std::string &name) const;

    /**
     * @brief Register a texture characterization under its name. No device upload — the
     *        first referenceTexture() triggers allocation.
     */
    void addTexture(const SupportedRSC &texture_rsc);

    /**
     * @brief Bind the color space, do any host side preload, increment use_count, and mark the
     *        texture dirty. The device grid slot is uploaded by commit(); read its handle with
     *        findTextureHandle(). The caller rolls back with dereferenceTexture() on failure.
     */
    void referenceTexture(
        const std::string &name,
        RSE_TextureColorSpace color_space,
        RSE_TextureStorageFormat storage_format
    );

    /**
     * @brief Decrement use_count and mark the texture dirty. The device slot is freed by the
     *        next commit() once no reference is left. The characterization survives.
     */
    void dereferenceTexture(const std::string &name);

    /**
     * @brief Align device grid storage with host state: upload the slot for every dirty grid
     *        texture with a live reference and free it for every one with no reference left.
     */
    void commit();

    /**
     * @brief Grid slot handle for a referenced grid texture; RS_NULL_HANDLE otherwise. Valid
     *        after commit().
     */
    RHandle findTextureHandle(const std::string &name) const;

    /**
     * @brief Erase the characterization entry. Throws if any consumer still holds a
     *        reference.
     */
    void removeTexture(const std::string &name);

    /**
     * @brief Observe live use_count for a texture name.
     */
    RSize getTextureUseCount(const std::string &name) const;

    /**
     * @brief Snapshot of every registered texture characterization, for getWorld().
     */
    std::vector<SupportedRSC> getTextureRSCs() const;

    /**
     * @brief Single-name characterization lookup, for findResource.
     */
    SupportedRSC findTextureRSC(const std::string &name) const;

    /**
     * @brief True iff a registered texture with this name is a constant.
     */
    bool isConstant(const std::string &name) const;

    /**
     * @brief Fetch the constant value for a registered constant texture.
     */
    RColor3f getConstant(const std::string &name) const;

    /**
     * @brief True iff a registered texture with this name is a procedural grid.
     */
    bool isGrid(const std::string &name) const;

    /**
     * @brief Storage format of a registered texture, used by the material manager to set
     *        the vision material's per-channel sample-input discriminator.
     */
    RSE_TextureStorageFormat getTextureStorageFormat(const std::string &name) const;

   private:
    struct NameEntry {
        /// Whether device storage is out of sync with host state and commit() must reconcile it.
        bool is_dirty = false;
        /// Texture characterization.
        SupportedRSC source_rsc;
        /// Total live references.
        RSize use_count = 0;
        /// Grid slot offset, populated lazily on first reference; RS_NULL_HANDLE for constants.
        RHandle handle = RS_NULL_HANDLE;
        /// Segment id in the grid pool layout (grid only); RS_NULL_INDEX otherwise.
        RIndex grid_id = RS_NULL_INDEX;
        /// Color space bound on first reference.
        RSE_TextureColorSpace color_space = RSE_TEXTURE_COLOR_SPACE_LINEAR;
        /// Held resource gid registration for this texture.
        RS_KernelGidRegistration resource_registration;
    };

    RS_KernelManager *_kernel_manager = nullptr;
    std::unordered_map<std::string, NameEntry> _name_entries;
    RIndex _next_grid_id = 0;

    /// Whether any entry is dirty, so commit() can skip the scan when nothing changed.
    bool _is_dirty = false;

    /// Initialize this manager's context data: the grid pool table and its layout.
    void _initKernelData();

    /// Free this manager's context data.
    void _freeKernelData();

    /// Allocate a device grid slot for grid_id and write the grid params; returns its offset.
    RHandle _uploadGrid(
        const rsc::RSC_TextureGrid &grid,
        RIndex grid_id,
        RSE_TextureColorSpace color_space
    );

    /// Free the device grid slot for grid_id.
    void _releaseGrid(RIndex grid_id);

    /// Decode a color into linear working space per its color space.
    static RColor3f _decodeColor(const RColor3f &color, RSE_TextureColorSpace color_space) {
        return color_space == RSE_TEXTURE_COLOR_SPACE_SRGB ? color.srgbToLinear() : color;
    }
};

/**
 * @brief References a texture on construction and dereferences it on destruction.
 */
class RS_KernelTextureReference {
   public:
    RS_KernelTextureReference() = default;
    RS_KernelTextureReference(
        RS_KernelManager *kernel_manager,
        std::string name,
        RSE_TextureColorSpace color_space,
        RSE_TextureStorageFormat storage_format
    );
    ~RS_KernelTextureReference();

    RS_KernelTextureReference(RS_KernelTextureReference &&other) noexcept;
    RS_KernelTextureReference &operator=(RS_KernelTextureReference &&other) noexcept;
    RS_KernelTextureReference(const RS_KernelTextureReference &) = delete;
    RS_KernelTextureReference &operator=(const RS_KernelTextureReference &) = delete;

    [[nodiscard]] bool isHeld() const { return _kernel_manager != nullptr; }
    [[nodiscard]] const std::string &name() const { return _name; }

   private:
    void _release();

    RS_KernelManager *_kernel_manager = nullptr;
    std::string _name;
};

}  // namespace rise

#endif  // RS_KERNEL_TEXTURE_MANAGER_H
