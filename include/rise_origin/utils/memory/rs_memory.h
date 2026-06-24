// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MEMORY_H
#define RS_MEMORY_H

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "rise_origin/utils/concurrency/rs_cpu_block.h"
#include "rise_origin/utils/concurrency/rs_thread_pool.h"
#include "rise_origin/utils/memory/rs_device.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_env_var.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/string/rs_format.h"

namespace rise {

class RS_MemoryPtr;
class RS_ConstMemoryPtr;

/// Generates a unique const char* key per call site for use with capture().
/// Produces a compile-time string literal like "rs_simulation_executor.cu:399".
#define RS_CAPTURE_KEY __FILE__ ":" RS_TOSTRING(__LINE__)

RS_ENUM(
    RSE_MemoryProviderType,
    RIndex,
    (RSE_MEMORY_PROVIDER_TYPE_DEFAULT, 0),
    (RSE_MEMORY_PROVIDER_TYPE_CPU, 1),
    (RSE_MEMORY_PROVIDER_TYPE_CUDA, 2),
    (RSE_MEMORY_PROVIDER_TYPE_UNKNOWN, 3)
);

class RS_MemoryProvider {
   public:
    RSE_MemoryProviderType type = RSE_MEMORY_PROVIDER_TYPE_UNKNOWN;
    RS_DeviceLocation device{RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX};

    class RS_MemoryProviderRunConfig {
       public:
        RSize thread_num = RS_MAX_SIZE;
        RSize block_size = RS_AUTO_BLOCK_SIZE;
        bool sync = true;
        RBigSize shared_mem_size = 0;

        [[nodiscard]] bool validate() const {
            return thread_num != RS_MAX_SIZE &&
                   (block_size > 0 || block_size == RS_AUTO_BLOCK_SIZE);
        }

        [[nodiscard]] std::string toString() const {
            return format(
                "RS_MemoryProviderRunConfig(max_threads={}, block_size={}, "
                "sync={}, shared_mem_size={})",
                thread_num,
                block_size,
                sync,
                shared_mem_size
            );
        }
    };

    class RS_MemoryProviderRunGridConfig {
       public:
        RSize grid_dim_x = RS_MAX_SIZE;
        RSize grid_dim_y = RS_MAX_SIZE;
        RSize grid_dim_z = RS_MAX_SIZE;
        RSize block_dim_x = RS_MAX_SIZE;
        RSize block_dim_y = RS_MAX_SIZE;
        RSize block_dim_z = RS_MAX_SIZE;
        bool sync = true;
        RBigSize shared_mem_size = 0;

        [[nodiscard]] bool validate() const {
            return grid_dim_x != RS_MAX_SIZE && grid_dim_y != RS_MAX_SIZE &&
                   grid_dim_z != RS_MAX_SIZE && block_dim_x != RS_MAX_SIZE &&
                   block_dim_y != RS_MAX_SIZE && block_dim_z != RS_MAX_SIZE;
        }
        [[nodiscard]] std::string toString() const {
            return format(
                "RS_MemoryProviderRunGridConfig("
                "grid_dim_x={}, grid_dim_y={}, grid_dim_z={}, "
                "block_dim_x={}, block_dim_y={}, block_dim_z={}, sync={}, shared_mem_size={})",
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
                sync,
                shared_mem_size
            );
        }
    };

    virtual ~RS_MemoryProvider() = default;

    virtual std::string getProperty(const std::string& key) const { return ""; }

    virtual RS_MemoryProvider* base() { return this; }

    virtual void free() = 0;
    virtual RS_MemoryPtr allocate(RBigSize size, bool sync = true) = 0;
    virtual RS_MemoryPtr allocateHost(RBigSize size, bool sync = true) = 0;
    virtual void deallocate(const RS_MemoryPtr& ptr, bool sync = true) = 0;
    virtual void set(
        const RS_MemoryPtr& ptr,
        int value,
        RBigSize size,
        RBigSize offset = 0,
        bool sync = true
    ) = 0;
    void copy(
        const RS_MemoryPtr& dst,
        const RS_MemoryPtr& src,
        RBigSize size,
        bool sync = true,
        bool ordered = true
    );
    virtual void copy(
        const RS_MemoryPtr& dst,
        const RS_ConstMemoryPtr& src,
        RBigSize size,
        bool sync = true,
        bool ordered = true
    ) = 0;
    virtual void
    sync(bool sync_main = true, bool sync_copy_to = true, bool sync_copy_from = true) = 0;

    // All subclasses should have the following template methods

    // Compute function runner:
    // template <
    //     typename ComputeFunction,
    //     typename... Args>
    // void run(RS_MemoryProviderRunConfig config, Args... args) { ... }

    // Compute function grid runner:
    // template <
    //     typename ComputeFunction,
    //     typename... Args>
    // void runGrid(RS_MemoryProviderRunGridConfig config, Args... args) { ... }

    // Captured procedure replay (CUDA graph on GPU, direct execution on CPU):
    // Caches the procedure defined by the lambda and replays it on subsequent calls.
    // Re-captures when dependent args change. Lambda must be the last argument.
    // Key should be generated using RS_CAPTURE_KEY macro for unique per-call-site identity.
    // template <typename... Args>
    // void capture(const char* key, dep0, dep1, ..., Lambda&& fn)
    // { ... }
};

template <typename DerivedType, typename PtrType>
class RS_MemoryPtrBase;

template <typename T>
class RS_IsMemoryPtr {
   private:
    // Helper function to detect RS_MemoryPtrBase inheritance
    // Note: they have to be defined before value
    template <typename U, typename D, typename P>
    static constexpr std::true_type _test(const RS_MemoryPtrBase<D, P>*) {
        return {};
    };

    template <typename U>
    static constexpr std::false_type _test(...) {
        return {};
    };

   public:
    static constexpr bool value = decltype(_test<T>(std::declval<const T*>()))::value;
};

/**
 * @brief Access-safe memory pointer with size, offset, location information and provider
 * information.
 * @note Uses CRTP pattern.
 * @note Memory is managed by the provider. User should explicitly free the
 * memory by calling free(). After free(), values of this memory pointer will
 * be set to invalid values, however, copies of the memory pointer are not
 * modified, they should be considered as invalid, and taken with care.
 */
template <typename DerivedType, typename PtrType>
class RS_MemoryPtrBase {
   public:
    static constexpr bool IsConst = std::is_const_v<std::remove_pointer_t<PtrType>>;
    static_assert(
        std::is_same_v<PtrType, void*> || std::is_same_v<PtrType, const void*>,
        "PtrType must be non-const/const void*"
    );
    using ByteType = std::conditional_t<IsConst, const RByte*, RByte*>;

    PtrType data = nullptr;  /// Pointer to the memory
    RBigSize bytes = 0;      /// Size of the memory from current pointer to allocated end in bytes
    RBigSize offset_bytes = 0;  /// Offset of the data pointer relative to base address in bytes
    RS_DeviceLocation mem_device = {
        RSE_DEVICE_TYPE_UNKNOWN,
        RS_NULL_INDEX
    };  /// Device location of the memory
    RS_MemoryProvider* mem_provider = nullptr;  /// Provider of the memory

    RS_MemoryPtrBase() = default;

    HOST_DEVICE_INLINE_QUALIFIER RS_MemoryPtrBase(
        PtrType ptr,
        RBigSize size,
        RBigSize offset,
        const RS_DeviceLocation& device = {RSE_DEVICE_TYPE_CPU, 0},
        RS_MemoryProvider* provider = nullptr
    )
        : data(ptr),
          bytes(size),
          offset_bytes(offset),
          mem_device(device),
          mem_provider(provider) {}

    RS_MemoryPtrBase(const RS_MemoryPtrBase& other) = default;

    HOST_DEVICE_INLINE_QUALIFIER PtrType ptr() const { return data; }

    HOST_DEVICE_INLINE_QUALIFIER RBigSize size() const { return bytes; }

    HOST_DEVICE_INLINE_QUALIFIER RBigSize offset() const { return offset_bytes; }

    HOST_DEVICE_INLINE_QUALIFIER bool empty() const { return bytes == 0; }

    HOST_DEVICE_INLINE_QUALIFIER RS_MemoryProvider* provider() const { return mem_provider; }

    HOST_DEVICE_INLINE_QUALIFIER RS_DeviceLocation device() const { return mem_device; }

    HOST_DEVICE_INLINE_QUALIFIER explicit operator PtrType() const { return data; }

    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T* as() const {
        return static_cast<T*>(data);
    }

    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const DerivedType& other) const {
        return data == other.data && bytes == other.bytes && offset_bytes == other.offset_bytes &&
               mem_device == other.mem_device && mem_provider == other.mem_provider;
    }

    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const DerivedType& other) const {
        return !(*this == other);
    }

    HOST_DEVICE_INLINE_QUALIFIER bool operator==(void* other) const { return data == other; }

    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(void* other) const { return data != other; }

    HOST_DEVICE_INLINE_QUALIFIER DerivedType operator+(RBigSize offset) const {
        assert((data != nullptr || offset == 0) && "Cannot add non-zero offset to nullptr");
        assert(bytes >= offset && "Cannot add offset greater than memory size");
        return {
            static_cast<ByteType>(data) + offset,
            bytes - offset,
            offset_bytes + offset,
            mem_device,
            mem_provider
        };
    }

    HOST_DEVICE_INLINE_QUALIFIER DerivedType& operator+=(RBigSize offset) {
        assert((data != nullptr || offset == 0) && "Cannot add non-zero offset to nullptr");
        assert(bytes >= offset && "Cannot add offset greater than memory size");
        data = static_cast<ByteType>(data) + offset;
        bytes -= offset;
        offset_bytes += offset;
        return *reinterpret_cast<DerivedType*>(this);
    }

    HOST_DEVICE_INLINE_QUALIFIER DerivedType operator-(RBigSize offset) const {
        assert((data != nullptr || offset == 0) && "Cannot subtract non-zero offset from nullptr");
        assert(offset_bytes >= offset && "Cannot subtract offset greater than memory size");
        return {
            static_cast<ByteType>(data) - offset,
            bytes + offset,
            offset_bytes - offset,
            mem_device,
            mem_provider
        };
    }

    HOST_DEVICE_INLINE_QUALIFIER DerivedType& operator-=(RBigSize offset) {
        assert((data != nullptr || offset == 0) && "Cannot subtract non-zero offset from nullptr");
        assert(offset_bytes >= offset && "Cannot subtract offset greater than memory size");
        data = static_cast<ByteType>(data) - offset;
        bytes += offset;
        offset_bytes -= offset;
        return *reinterpret_cast<DerivedType*>(this);
    }

    template <typename T, typename Index>
    HOST_DEVICE_INLINE_QUALIFIER T& get(Index index) const {
        assert(RBIG_SIZE(sizeof(T)) * index < bytes && "Index out of bounds");
        return static_cast<T*>(data)[index];
    }

    HOST_DEVICE_INLINE_QUALIFIER bool isWeak() const { return mem_provider == nullptr; }

    void set(int value, RBigSize size = RS_MAX_BIG_SIZE, RBigSize offset = 0, bool sync = true) {
        if (size == 0) return;
        if (size == RS_MAX_BIG_SIZE) size = bytes;
        if (mem_provider == nullptr)
            RS_THROW(RS_UninitializedException(
                "Provider is nullptr, cannot set memory with unknown provider"
            ));
        if (size > bytes)
            RS_THROW(RS_OutOfBoundsException(
                "Cannot set memory larger than allowed remaining size {} (offset {})",
                bytes,
                offset_bytes
            ));

        mem_provider->set(*reinterpret_cast<DerivedType*>(this), value, size, offset, sync);
    }

    template <typename MemoryPtr>
    void
    copyFrom(const MemoryPtr& src, RBigSize size, bool sync = true, bool ordered = true) const {
        _resolveCopy(*reinterpret_cast<const DerivedType*>(this), src, size, sync, ordered);
    }

    template <typename MemoryPtr>
    void copyTo(const MemoryPtr& dst, RBigSize size, bool sync = true, bool ordered = true) const {
        _resolveCopy(dst, *reinterpret_cast<const DerivedType*>(this), size, sync, ordered);
    }

   private:
    template <typename MemoryPtrDst, typename MemoryPtrSrc>
    static void _resolveCopy(
        const MemoryPtrDst& dst,
        const MemoryPtrSrc& src,
        RBigSize size,
        bool sync = true,
        bool ordered = true
    ) {
        static_assert(
            RS_IsMemoryPtr<MemoryPtrDst>::value,
            "MemoryPtrDst must be derived from RS_MemoryPtrBase"
        );
        static_assert(
            RS_IsMemoryPtr<MemoryPtrSrc>::value,
            "MemoryPtrSrc must be derived from RS_MemoryPtrBase"
        );

        static_assert(!dst.IsConst, "copy dst must be non-const");
        std::string error_message = format(
            "Copy trace:\n\n"
            "Dst pointer: device={}, index={}, ptr={}, size={}, offset={}\n"
            "Src pointer: device={}, index={}, ptr={}, size={}, offset={}\n"
            "Copy size: {},\n"
            "Sync: {}\n"
            "Ordered: {}\n\n"
            "Exceptions:",
            enumToString(dst.mem_device.type),
            dst.mem_device.id,
            dst.data,
            dst.bytes,
            dst.offset_bytes,
            enumToString(src.mem_device.type),
            src.mem_device.id,
            src.data,
            src.bytes,
            src.offset_bytes,
            size,
            sync,
            ordered
        );
        bool success = false;

        if (src.bytes < size) {
            error_message += format("Copy size is greater than source size\n");
            RS_THROW(RS_OutOfBoundsException(error_message));
        } else if (dst.bytes < size) {
            error_message += format("Copy size is greater than destination size\n");
            RS_THROW(RS_OutOfBoundsException(error_message));
        }

        // Try using provider from dst first
        if (dst.mem_provider != nullptr) {
            try {
                dst.mem_provider->copy(dst, src, size, sync, ordered);
                success = true;
            } catch (const RS_BaseException& e) {
                error_message += format("Exception using dst provider: {}\n", e.what());
            } catch (...) {
                error_message += "Unknown exception using dst provider\n";
            }
        } else {
            error_message += "Dst provider is nullptr and skipped.\n";
        }

        error_message += "\n";

        // If first attempt failed, try using provider from 'src'
        if (!success) {
            if (src.mem_provider != nullptr) {
                try {
                    src.mem_provider->copy(dst, src, size, sync, ordered);
                    success = true;
                } catch (const RS_BaseException& e) {
                    error_message += format("Exception using src provider: {}\n", e.what());
                } catch (...) {
                    error_message += "Unknown exception using src provider\n";
                }
            } else {
                error_message += "Src provider is nullptr and skipped.\n";
            }
        }

        // If both attempts failed, throw exception with detailed info
        if (!success) {
            RS_THROW(RS_HostExecutionException(error_message));
        }
    }
};

class RS_MemoryPtr : public RS_MemoryPtrBase<RS_MemoryPtr, void*> {
   public:
    RS_MemoryPtr() = default;
    /**
     * @brief Construct a new RS_MemoryPtr object.
     * @note If provider is not set, free method will not work.
     * @param ptr Pointer to the memory.
     * @param size Size of the memory.
     * @param device Location of the memory.
     * @param provider Provider of the memory.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_MemoryPtr(
        void* ptr,
        RBigSize size,
        RBigSize offset = 0,
        const RS_DeviceLocation& device = {RSE_DEVICE_TYPE_CPU, 0},
        RS_MemoryProvider* provider = nullptr
    )
        : RS_MemoryPtrBase(ptr, size, offset, device, provider) {}

    void free(bool sync = true) {
        if (data != nullptr && mem_provider != nullptr) {
            mem_provider->deallocate(*this, sync);
            data = nullptr;
            bytes = 0;
            offset_bytes = 0;
            mem_device = {RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX};
            mem_provider = nullptr;
        }
    }
};

class RS_ConstMemoryPtr : public RS_MemoryPtrBase<RS_ConstMemoryPtr, const void*> {
   public:
    RS_ConstMemoryPtr() = default;

    /// No need to be explicit
    HOST_DEVICE_INLINE_QUALIFIER RS_ConstMemoryPtr(const RS_MemoryPtr& other)
        : RS_MemoryPtrBase(
              other.data,
              other.bytes,
              other.offset_bytes,
              other.device(),
              other.provider()
          ) {};

    /**
     * @brief Construct a new RS_ConstMemoryPtr object.
     * @note If provider is not set, free method will not work.
     * @param ptr Pointer to the memory.
     * @param size Size of the memory.
     * @param device Location of the memory.
     * @param provider Provider of the memory.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_ConstMemoryPtr(
        const void* ptr,
        RBigSize size,
        RBigSize offset = 0,
        const RS_DeviceLocation& device = {RSE_DEVICE_TYPE_CPU, 0},
        RS_MemoryProvider* provider = nullptr
    )
        : RS_MemoryPtrBase(ptr, size, offset, device, provider) {}
};

inline void RS_MemoryProvider::copy(
    const RS_MemoryPtr& dst,
    const RS_MemoryPtr& src,
    RBigSize size,
    bool sync,
    bool ordered
) {
    copy(dst, RS_ConstMemoryPtr(src), size, sync, ordered);
}

class RS_MemoryDefaultProvider final : public RS_MemoryProvider {
   public:
    RS_MemoryDefaultProvider() {
        type = RSE_MEMORY_PROVIDER_TYPE_DEFAULT;
        device = {RSE_DEVICE_TYPE_CPU, 0};
    }

    static RS_MemoryDefaultProvider& getInstance() {
        static RS_MemoryDefaultProvider instance;
        return instance;
    }

    void free() override {}

    RS_MemoryPtr allocate(RBigSize size, bool sync) override {
        if (size == 0) return {nullptr, 0, 0, device, this};

        void* ptr = std::malloc(size);
        if (!ptr) {
            RS_THROW(
                RS_DeviceMemoryException("Failed to allocate device memory on the default provider")
            );
        }
        return {ptr, size, 0, device, this};
    }

    RS_MemoryPtr allocateHost(RBigSize size, bool sync = true) override {
        if (size == 0) return {nullptr, 0, 0, device, this};

        void* ptr = std::malloc(size);
        if (!ptr) {
            RS_THROW(
                RS_HostMemoryException("Failed to allocate host memory on the default provider")
            );
        }
        return {ptr, size, 0, device, this};
    }

    void deallocate(const RS_MemoryPtr& ptr, bool sync = true) override {
        if (ptr.provider() != this)
            RS_THROW(RS_HostExecutionException("Cannot free memory allocated by another provider"));

        if (ptr.data) {
            std::free(ptr.data);
        }
    }

    void set(
        const RS_MemoryPtr& ptr,
        int value,
        RBigSize size,
        RBigSize offset = 0,
        bool sync = true
    ) override {
        if (ptr.provider() != this)
            RS_THROW(RS_HostExecutionException("Cannot set memory allocated by another provider"));
        if (ptr.device().type == RSE_DEVICE_TYPE_CPU) {
            std::memset((ptr + offset).data, value, size);
        }
    }

    void copy(
        const RS_MemoryPtr& dst,
        const RS_ConstMemoryPtr& src,
        RBigSize size,
        bool sync = true,
        bool ordered = true
    ) override {
        if (dst.provider() != this && src.provider() != this)
            RS_THROW(
                RS_HostExecutionException("One end of the copy must be allocated by this provider")
            );

        if (dst.device().type != RSE_DEVICE_TYPE_CPU || src.device().type != RSE_DEVICE_TYPE_CPU)
            RS_THROW(RS_HostExecutionException(
                "Cannot perform copy between dst device type {} index {} and src device type "
                "{} "
                "index {}",
                enumToString(dst.device().type),
                dst.device().id,
                enumToString(src.device().type),
                src.device().id
            ));
        std::memcpy(dst.data, src.data, size);
    }

    void
    sync(bool sync_main = true, bool sync_copy_to = true, bool sync_copy_from = true) override {}
};

/**
 * @brief A scoped memory provider that wraps another provider and maintains its own allocation
 * tracking.
 *
 * This provider forwards all operations to an real provider while maintaining its own
 * tracking scope. When destroyed, it automatically frees all allocations made through this
 * scoped provider, without affecting the real provider or allocations made directly
 * through it.
 *
 * Useful for creating temporary allocation scopes where you want automatic cleanup.
 */
class RS_MemoryScopedProvider final : public RS_MemoryProvider {
   public:
    explicit RS_MemoryScopedProvider(RS_MemoryProvider* real) : _real(real) {
        if (real == nullptr)
            RS_THROW(RS_InvalidArgumentException("Scoped provider underlying provider is null"));
        type = real->type;
        device = real->device;
    }

    ~RS_MemoryScopedProvider() override {
        for (auto& p : _tracked) p.free();
    }

    RS_MemoryScopedProvider(const RS_MemoryScopedProvider&) = delete;
    RS_MemoryScopedProvider& operator=(const RS_MemoryScopedProvider&) = delete;

    std::string getProperty(const std::string& key) const override {
        return _real->getProperty(key);
    }

    RS_MemoryProvider* base() override { return _real->base(); }

    void free() override { _real->free(); }

    RS_MemoryPtr allocate(RBigSize size, bool sync = true) override {
        RS_MemoryPtr p = _real->allocate(size, sync);
        if (p.data != nullptr) _tracked.push_back(p);
        return _rebind(p);
    }

    RS_MemoryPtr allocateHost(RBigSize size, bool sync = true) override {
        RS_MemoryPtr p = _real->allocateHost(size, sync);
        if (p.data != nullptr) _tracked.push_back(p);
        return _rebind(p);
    }

    void deallocate(const RS_MemoryPtr& ptr, bool sync = true) override {
        _untrack(ptr.data);
        _real->deallocate(_unbind(ptr), sync);
    }

    void set(
        const RS_MemoryPtr& ptr,
        int value,
        RBigSize size,
        RBigSize offset = 0,
        bool sync = true
    ) override {
        _real->set(_unbind(ptr), value, size, offset, sync);
    }

    void copy(
        const RS_MemoryPtr& dst,
        const RS_ConstMemoryPtr& src,
        RBigSize size,
        bool sync = true,
        bool ordered = true
    ) override {
        _real->copy(_unbind(dst), _unbindConst(src), size, sync, ordered);
    }

    void
    sync(bool sync_main = true, bool sync_copy_to = true, bool sync_copy_from = true) override {
        _real->sync(sync_main, sync_copy_to, sync_copy_from);
    }

   private:
    RS_MemoryProvider* _real = nullptr;
    std::vector<RS_MemoryPtr> _tracked;  /// Stored bound to _real; survivors freed in the dtor.

    RS_MemoryPtr _rebind(const RS_MemoryPtr& p) const {
        if (p.data == nullptr) return p;
        return RS_MemoryPtr(
            p.data,
            p.bytes,
            p.offset_bytes,
            p.mem_device,
            const_cast<RS_MemoryScopedProvider*>(this)
        );
    }

    RS_MemoryPtr _unbind(const RS_MemoryPtr& p) const {
        if (p.provider() != this) return p;
        return RS_MemoryPtr(p.data, p.bytes, p.offset_bytes, p.mem_device, _real);
    }

    RS_ConstMemoryPtr _unbindConst(const RS_ConstMemoryPtr& p) const {
        if (p.provider() != this) return p;
        return RS_ConstMemoryPtr(p.data, p.bytes, p.offset_bytes, p.device(), _real);
    }

    void _untrack(const void* data) {
        for (auto it = _tracked.begin(); it != _tracked.end(); ++it) {
            if (it->data == data) {
                _tracked.erase(it);
                return;
            }
        }
    }
};

}  // namespace rise

#endif  // RS_MEMORY_H
