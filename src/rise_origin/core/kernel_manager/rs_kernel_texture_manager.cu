// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rs_kernel_texture_manager.h"

using namespace std;
using namespace rise::rsc;

namespace rise {

/*****************************************************************
 * RS_KernelTextureManager
 *****************************************************************/
void RS_KernelTextureManager::init(RS_KernelManager *kernel_manager) {
    _kernel_manager = kernel_manager;
    _kernel_manager->enterKernelDataCriticalZone();
    _initKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
}

void RS_KernelTextureManager::free() {
    if (_kernel_manager != nullptr) {
        _kernel_manager->enterKernelDataCriticalZone();

        // Release any grid slots still allocated, then drop all registrations.
        for (auto &[name, entry] : _name_entries) {
            if (entry.grid_id != RS_NULL_INDEX) {
                _releaseGrid(entry.grid_id);
                entry.grid_id = RS_NULL_INDEX;
                entry.handle = RS_NULL_HANDLE;
            }
        }

        _freeKernelData();
        _kernel_manager->exitKernelDataCriticalZone();
        _kernel_manager = nullptr;
    }
    _name_entries.clear();
}

bool RS_KernelTextureManager::hasTexture(const std::string &name) const {
    return _name_entries.count(name) > 0;
}

void RS_KernelTextureManager::addTexture(const SupportedRSC &texture_rsc) {
    auto name = texture_rsc.visit([](const auto &c) -> std::string { return c.name; });
    if (_name_entries.count(name) != 0)
        RS_THROW(RS_InvalidCharacterizationException("Texture {} is already registered", name));
    NameEntry entry;
    entry.source_rsc = texture_rsc;
    entry.resource_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_RESOURCE, name);
    _name_entries.emplace(name, std::move(entry));
}

void RS_KernelTextureManager::referenceTexture(
    const std::string &name,
    RSE_TextureColorSpace color_space,
    RSE_TextureStorageFormat /*storage_format*/
) {
    // storage_format is a decode hint reserved for image textures.
    auto it = _name_entries.find(name);
    if (it == _name_entries.end())
        RS_THROW(RS_InvalidCharacterizationException(
            "Texture {} is not registered in the kernel manager", name
        ));
    NameEntry &entry = it->second;

    // A texture binds to one color space; rebinding under a different one is rejected.
    if (entry.use_count > 0 && entry.color_space != color_space)
        RS_THROW(RS_InvalidCharacterizationException(
            "Texture {} is already bound with color space {}; cannot rebind as {}",
            name,
            enumToString(entry.color_space),
            enumToString(color_space)
        ));

    // Bind the color space and reserve a grid slot id on first live reference; the device upload
    // is deferred to commit(). Keep an existing slot id so a reference cycle within one flush
    // reuses the already uploaded slot.
    if (entry.use_count == 0) {
        entry.color_space = color_space;
        if (entry.source_rsc.is<RSC_TextureGrid>() && entry.grid_id == RS_NULL_INDEX) {
            entry.grid_id = _next_grid_id++;
        }
    }
    entry.use_count += 1;
    entry.is_dirty = true;
    _is_dirty = true;
}

void RS_KernelTextureManager::dereferenceTexture(const std::string &name) {
    auto it = _name_entries.find(name);
    if (it == _name_entries.end()) return;
    NameEntry &entry = it->second;
    if (entry.use_count == 0) return;
    entry.use_count -= 1;
    entry.is_dirty = true;
    _is_dirty = true;
}

void RS_KernelTextureManager::commit() {
    if (!_is_dirty) return;
    _kernel_manager->enterKernelDataCriticalZone();
    for (auto &[name, entry] : _name_entries) {
        if (!entry.is_dirty) continue;
        if (entry.source_rsc.is<RSC_TextureGrid>()) {
            if (entry.use_count > 0 && entry.handle == RS_NULL_HANDLE) {
                entry.handle = _uploadGrid(
                    entry.source_rsc.as<RSC_TextureGrid>(), entry.grid_id, entry.color_space
                );
            } else if (entry.use_count == 0 && entry.grid_id != RS_NULL_INDEX) {
                _releaseGrid(entry.grid_id);
                entry.grid_id = RS_NULL_INDEX;
                entry.handle = RS_NULL_HANDLE;
            }
        }
        entry.is_dirty = false;
    }
    _kernel_manager->exitKernelDataCriticalZone();
    _is_dirty = false;
}

RHandle RS_KernelTextureManager::findTextureHandle(const std::string &name) const {
    auto it = _name_entries.find(name);
    return it == _name_entries.end() ? RS_NULL_HANDLE : it->second.handle;
}

void RS_KernelTextureManager::removeTexture(const std::string &name) {
    auto it = _name_entries.find(name);
    if (it == _name_entries.end()) return;
    if (it->second.use_count > 0)
        RS_THROW(RS_HostExecutionException(
            "Texture {} cannot be removed while it is still referenced", name
        ));
    _name_entries.erase(it);
}

RSize RS_KernelTextureManager::getTextureUseCount(const std::string &name) const {
    auto it = _name_entries.find(name);
    return it == _name_entries.end() ? 0 : it->second.use_count;
}

std::vector<RS_KernelTextureManager::SupportedRSC>
RS_KernelTextureManager::getTextureRSCs() const {
    std::vector<SupportedRSC> texture_rscs;
    texture_rscs.reserve(_name_entries.size());
    for (const auto &[name, entry] : _name_entries) texture_rscs.push_back(entry.source_rsc);
    return texture_rscs;
}

RS_KernelTextureManager::SupportedRSC RS_KernelTextureManager::findTextureRSC(
    const std::string &name
) const {
    auto it = _name_entries.find(name);
    if (it == _name_entries.end())
        RS_THROW(RS_InvalidCharacterizationException(
            "Texture {} is not registered in the kernel manager", name
        ));
    return it->second.source_rsc;
}

bool RS_KernelTextureManager::isConstant(const std::string &name) const {
    auto it = _name_entries.find(name);
    return it != _name_entries.end() && it->second.source_rsc.is<RSC_TextureConstant>();
}

RColor3f RS_KernelTextureManager::getConstant(const std::string &name) const {
    auto it = _name_entries.find(name);
    if (it == _name_entries.end() || !it->second.source_rsc.is<RSC_TextureConstant>())
        RS_THROW(RS_InvalidCharacterizationException("Texture {} is not a constant texture", name));
    return _decodeColor(
        it->second.source_rsc.as<RSC_TextureConstant>().value, it->second.color_space
    );
}

bool RS_KernelTextureManager::isGrid(const std::string &name) const {
    auto it = _name_entries.find(name);
    return it != _name_entries.end() && it->second.source_rsc.is<RSC_TextureGrid>();
}

RSE_TextureStorageFormat RS_KernelTextureManager::getTextureStorageFormat(const std::string &name
) const {
    auto it = _name_entries.find(name);
    if (it == _name_entries.end())
        RS_THROW(RS_InvalidCharacterizationException(
            "Texture {} is not registered in the kernel manager", name
        ));
    if (it->second.source_rsc.is<RSC_TextureGrid>()) return RSE_TEXTURE_STORAGE_FORMAT_GRID;
    return RSE_TEXTURE_STORAGE_FORMAT_CONSTANT;
}

void RS_KernelTextureManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto mem_provider = _kernel_manager->getDeviceProvider();
        auto &ctx = k_ptr->ctx;
        ctx.texture_grids.init(0, mem_provider);
        ctx.texture_grid_layout.init(mem_provider);

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelTextureManager::_freeKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        ctx.texture_grids.free();
        ctx.texture_grid_layout.free();

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

RHandle RS_KernelTextureManager::_uploadGrid(
    const RSC_TextureGrid &grid, RIndex grid_id, RSE_TextureColorSpace color_space
) {
    RHandle handle = RS_NULL_HANDLE;
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        // Update host layout.
        ctx.host_data->texture_grid_layout.add(1, grid_id);
        ctx.host_data->texture_grid_layout.syncDeviceLayout(ctx.texture_grid_layout);
        
        // Update device layout.
        handle = ctx.host_data->texture_grid_layout.segmentOffset(
            ctx.host_data->texture_grid_layout.find(grid_id)
        );

        // Adapt device storage size.
        ctx.texture_grids.resize({ctx.host_data->texture_grid_layout.capacity()});

        // Transfer texture data to device context.
        logic::RS_TextureGridMem grid_mem;
        grid_mem.fill_color = _decodeColor(grid.fill_color, color_space);
        grid_mem.line_color = _decodeColor(grid.line_color, color_space);
        grid_mem.is_meters = grid.is_meters;
        grid_mem.x_spacing = grid.x_spacing;
        grid_mem.y_spacing = grid.y_spacing;
        grid_mem.line_width = grid.line_width;
        ctx.texture_grids.writeSlice(&grid_mem, 1, handle);

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
    return handle;
}

void RS_KernelTextureManager::_releaseGrid(RIndex grid_id) {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        // Update host layout.
        RIndex idx = ctx.host_data->texture_grid_layout.find(grid_id);
        ctx.host_data->texture_grid_layout.remove(idx);

        // Update device layout.
        ctx.host_data->texture_grid_layout.syncDeviceLayout(ctx.texture_grid_layout);

        // Adapt device storage size.
        ctx.texture_grids.resize({ctx.host_data->texture_grid_layout.capacity()});

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

/*****************************************************************
 * RS_KernelTextureReference
 *****************************************************************/
RS_KernelTextureReference::RS_KernelTextureReference(
    RS_KernelManager *kernel_manager,
    std::string name,
    RSE_TextureColorSpace color_space,
    RSE_TextureStorageFormat storage_format
)
    : _kernel_manager(kernel_manager), _name(std::move(name)) {
    _kernel_manager->getKernelTextureManager().referenceTexture(_name, color_space, storage_format);
}

RS_KernelTextureReference::~RS_KernelTextureReference() { _release(); }

RS_KernelTextureReference::RS_KernelTextureReference(RS_KernelTextureReference &&other) noexcept
    : _kernel_manager(other._kernel_manager), _name(std::move(other._name)) {
    other._kernel_manager = nullptr;
}

RS_KernelTextureReference &RS_KernelTextureReference::operator=(RS_KernelTextureReference &&other
) noexcept {
    if (this != &other) {
        _release();
        _kernel_manager = other._kernel_manager;
        _name = std::move(other._name);
        other._kernel_manager = nullptr;
    }
    return *this;
}

void RS_KernelTextureReference::_release() {
    if (_kernel_manager == nullptr) return;
    _kernel_manager->getKernelTextureManager().dereferenceTexture(_name);
    _kernel_manager = nullptr;
}

}  // namespace rise
