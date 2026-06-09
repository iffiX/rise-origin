// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MEMORY_H
#define RS_MEMORY_H

#include <cassert>
#include <memory>
#include <vector>

#include "rs_def.h"
#include "rs_device.h"
#include "rs_env_var.h"
#include "rs_exception.h"
#include "rs_format.h"
#include "rs_host_block.h"
#include "rs_thread_pool.h"

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
        RSize shared_mem_size = 0;

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
        RSize shared_mem_size = 0;

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
    virtual void free() = 0;
    virtual RS_MemoryPtr allocate(size_t size, bool sync = true) = 0;
    virtual RS_MemoryPtr allocateHost(size_t size, bool sync = true) = 0;
    virtual void deallocate(const RS_MemoryPtr& ptr, bool sync = true) = 0;
    virtual void
    set(const RS_MemoryPtr& ptr, int value, size_t size, size_t offset = 0, bool sync = true) = 0;
    void copy(
        const RS_MemoryPtr& dst,
        const RS_MemoryPtr& src,
        size_t size,
        bool sync = true,
        bool ordered = true
    );
    virtual void copy(
        const RS_MemoryPtr& dst,
        const RS_ConstMemoryPtr& src,
        size_t size,
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
    using ByteType = std::conditional_t<IsConst, const std::byte*, std::byte*>;

    PtrType data = nullptr;   /// Pointer to the memory
    size_t bytes = 0;         /// Size of the memory from current pointer to allocated end in bytes
    size_t offset_bytes = 0;  /// Offset of the data pointer relative to base address in bytes
    RS_DeviceLocation mem_device = {
        RSE_DEVICE_TYPE_UNKNOWN,
        RS_NULL_INDEX
    };  /// Device location of the memory
    RS_MemoryProvider* mem_provider = nullptr;  /// Provider of the memory

    RS_MemoryPtrBase() = default;

    HOST_DEVICE_INLINE_QUALIFIER RS_MemoryPtrBase(
        PtrType ptr,
        size_t size,
        size_t offset,
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

    HOST_DEVICE_INLINE_QUALIFIER RSize size() const { return bytes; }

    HOST_DEVICE_INLINE_QUALIFIER RSize offset() const { return offset_bytes; }

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

    HOST_DEVICE_INLINE_QUALIFIER DerivedType operator+(size_t offset) const {
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

    HOST_DEVICE_INLINE_QUALIFIER DerivedType& operator+=(size_t offset) {
        assert((data != nullptr || offset == 0) && "Cannot add non-zero offset to nullptr");
        assert(bytes >= offset && "Cannot add offset greater than memory size");
        data = static_cast<ByteType>(data) + offset;
        bytes -= offset;
        offset_bytes += offset;
        return *reinterpret_cast<DerivedType*>(this);
    }

    HOST_DEVICE_INLINE_QUALIFIER DerivedType operator-(size_t offset) const {
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

    HOST_DEVICE_INLINE_QUALIFIER DerivedType& operator-=(size_t offset) {
        assert((data != nullptr || offset == 0) && "Cannot subtract non-zero offset from nullptr");
        assert(offset_bytes >= offset && "Cannot subtract offset greater than memory size");
        data = static_cast<ByteType>(data) - offset;
        bytes += offset;
        offset_bytes -= offset;
        return *reinterpret_cast<DerivedType*>(this);
    }

    template <typename T, typename Index>
    HOST_DEVICE_INLINE_QUALIFIER T& get(Index index) const {
        assert(sizeof(T) * index < bytes && "Index out of bounds");
        return static_cast<T*>(data)[index];
    }

    HOST_DEVICE_INLINE_QUALIFIER bool isWeak() const { return mem_provider == nullptr; }

    void set(
        int value,
        size_t size = std::numeric_limits<size_t>::max(),
        size_t offset = 0,
        bool sync = true
    ) {
        if (size == 0) return;
        if (size == std::numeric_limits<size_t>::max()) size = bytes;
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
    void copyFrom(const MemoryPtr& src, size_t size, bool sync = true, bool ordered = true) const {
        _resolveCopy(*reinterpret_cast<const DerivedType*>(this), src, size, sync, ordered);
    }

    template <typename MemoryPtr>
    void copyTo(const MemoryPtr& dst, size_t size, bool sync = true, bool ordered = true) const {
        _resolveCopy(dst, *reinterpret_cast<const DerivedType*>(this), size, sync, ordered);
    }

   private:
    template <typename MemoryPtrDst, typename MemoryPtrSrc>
    static void _resolveCopy(
        const MemoryPtrDst& dst,
        const MemoryPtrSrc& src,
        size_t size,
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
        size_t size,
        size_t offset = 0,
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
        size_t size,
        size_t offset = 0,
        const RS_DeviceLocation& device = {RSE_DEVICE_TYPE_CPU, 0},
        RS_MemoryProvider* provider = nullptr
    )
        : RS_MemoryPtrBase(ptr, size, offset, device, provider) {}
};

inline void RS_MemoryProvider::copy(
    const RS_MemoryPtr& dst,
    const RS_MemoryPtr& src,
    size_t size,
    bool sync,
    bool ordered
) {
    copy(dst, RS_ConstMemoryPtr(src), size, sync, ordered);
}

class RS_MemoryDefaultProvider final : public RS_MemoryProvider {
   public:
    RS_ThreadPool pool;

    RS_MemoryDefaultProvider() {
        type = RSE_MEMORY_PROVIDER_TYPE_DEFAULT;
        device = {RSE_DEVICE_TYPE_CPU, 0};
    }

    static RS_MemoryDefaultProvider& getInstance() {
        static RS_MemoryDefaultProvider instance;
        return instance;
    }

    void free() override {}

    RS_MemoryPtr allocate(size_t size, bool sync) override {
        if (size == 0) return {nullptr, 0, 0, device, this};

        void* ptr = std::malloc(size);
        if (!ptr) {
            RS_THROW(
                RS_DeviceMemoryException("Failed to allocate device memory on the default provider")
            );
        }
        return {ptr, size, 0, device, this};
    }

    RS_MemoryPtr allocateHost(size_t size, bool sync = true) override {
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
        size_t size,
        size_t offset = 0,
        bool sync = true
    ) override {
        if (ptr.provider() != this)
            RS_THROW(RS_HostExecutionException("Cannot set memory allocated by another provider"));

        pool.wait();
        if (ptr.device().type == RSE_DEVICE_TYPE_CPU) {
            std::memset((ptr + offset).data, value, size);
        }
    }

    void copy(
        const RS_MemoryPtr& dst,
        const RS_ConstMemoryPtr& src,
        size_t size,
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

        pool.wait();
        std::memcpy(dst.data, src.data, size);
    }

    void
    sync(bool sync_main = true, bool sync_copy_to = true, bool sync_copy_from = true) override {
        pool.wait();
    }

    template <typename ComputeFunction, bool Debug = false, typename... Args>
    void run(RS_MemoryProviderRunConfig config, Args... args) {
        if (!config.validate())
            RS_THROW(RS_InvalidArgumentException("Invalid config: {}", config.toString()));

        if (config.thread_num == 0) return;

        ComputeFunction::validate();
        if constexpr (!ComputeFunction::isCPUSupported()) {
            RS_THROW(RS_DeviceExecutionException(
                "Compute function is not supported by the default provider"
            ));
        } else {
            config.block_size = config.block_size == RS_AUTO_BLOCK_SIZE ? RS_HOST_DEFAULT_BLOCK_SIZE
                                                                        : config.block_size;

            RSize block_num = RS_CEIL_DIV(config.thread_num, config.block_size);

            RSize parallel_block_num = pool.size();

            auto host_shared_mem = allocate(
                ComputeFunction::sharedMemLayoutSize() * config.block_size * parallel_block_num,
                true
            );

            auto host_stack_mem =
                allocate(RS_HOST_FIBER_STACK_SIZE * config.block_size * parallel_block_num, true);

            pool.submitLoop(
                    0, block_num, [=](RSize b) {
                        RS_HostBlock host_block;
                        RIndex worker_thread_index =
                            RS_ThreadPool::RS_SynchronizedThreadPoolWorker::index;
                        if constexpr (ComputeFunction::sharedMemLayoutSize() > 0) {
                            host_block.run<ComputeFunction, Args...>(
                                b,
                                config.block_size,
                                RS_MIN(
                                    config.block_size, config.thread_num - b * config.block_size
                                ),
                                RS_HOST_FIBER_STACK_SIZE,
                                (host_stack_mem +
                                 worker_thread_index * config.block_size * RS_HOST_FIBER_STACK_SIZE)
                                    .ptr(),
                                (host_shared_mem + worker_thread_index * config.block_size *
                                                       ComputeFunction::sharedMemLayoutSize())
                                    .ptr(),
                                args...
                            );
                        } else {
                            host_block.run<ComputeFunction, Args...>(
                                b,
                                config.block_size,
                                RS_MIN(
                                    config.block_size, config.thread_num - b * config.block_size
                                ),
                                RS_HOST_FIBER_STACK_SIZE,
                                (host_stack_mem +
                                 worker_thread_index * config.block_size * RS_HOST_FIBER_STACK_SIZE)
                                    .ptr(),
                                static_cast<void*>(nullptr),
                                args...
                            );
                        }
                    }
            ).wait();
            // Since the executor is stateless, we need to wait for all tasks to finish and
            // deallocate the shared memory
            host_shared_mem.free();
            host_stack_mem.free();
        }
    }

    template <typename... Args>
    void capture(const char* key, Args&&... args) {
        auto all = std::forward_as_tuple(std::forward<Args>(args)...);
        std::get<sizeof...(Args) - 1>(all)();
    }
};

/**
 * @brief A class representing a handle to N-dimensional memory managed by a provider.
 * It will be automatically released when all references are released.
 *
 * The memory can reside at any location, is row-ordered and flattened into 1D,
 * it is also copy-by-reference.
 *
 * In contrast, RS_MemoryPtr need user to explicitly free the memory. Also, users can
 * construct RS_MemoryPtr as a weak reference with no valid provider or device,
 * RS_NdMemory does not allow this.
 *
 * @note Can only be used on the host.
 * @tparam T The type of elements stored in the memory.
 */
template <typename T>
class RS_NdMemory {
   public:
    /**
     * @brief A struct representing an accessor for N-dimensional memory.
     *
     * This struct allows accessing elements of N-dimensional memory.
     * It provides functionality for de-referencing, pointer access, and subscripting.
     */
    struct RS_NdMemoryAccessor {
        /// Pointer to the N-dimensional memory.
        const RS_NdMemory* nd_mem = nullptr;
        /// Offset within the memory.
        size_t offset = 0;
        /// Dimension of the memory.
        size_t dim = 0;

        /**
         * @brief Constructs an RS_NdMemoryAccessor object.
         *
         * Constructs an RS_NdMemoryAccessor object with the specified parameters.
         *
         * @param nd_mem Pointer to the N-dimensional memory.
         * @param offset Offset within the memory.
         * @param dim Dimension of the memory.
         */
        RS_NdMemoryAccessor(const RS_NdMemory* nd_mem, size_t offset, size_t dim)
            : nd_mem(nd_mem), offset(offset), dim(dim) {}

        /**
         * @brief Subscript operator for accessing elements recursively.
         *
         * This operator allows accessing elements of N-dimensional memory recursively.
         *
         * @param index Index of the element to access.
         * @return An RS_NdMemoryAccessor object representing the accessed element.
         * @throws RS_OutOfBoundsException if the accessed dimension exceeds the memory's
         * dimensions.
         */
        HOST_INLINE_QUALIFIER RS_NdMemoryAccessor operator[](size_t index) {
            if (dim >= nd_mem->shape.size()) {
                RS_THROW(RS_OutOfBoundsException(
                    "Trying to access dim {} of a {} dim memory", dim + 1, nd_mem->shape.size()
                ));
            } else {
                size_t new_offset = offset + nd_mem->stride[dim] * index;
                return RS_NdMemoryAccessor(nd_mem, new_offset, dim + 1);
            }
        }

        /**
         * @return Pointer to the current memory location.
         */
        T* ptr() { return nd_mem->data.get() + offset; }

        /**
         * @return Pointer to the current memory location.
         */
        T* operator->() { return nd_mem->data.get() + offset; }

        /**
         * @return Reference to the current memory element.
         */
        T& operator*() { return *(nd_mem->data.get() + offset); }

        /**
         * @brief Converts the memory element to a specified type.
         *
         * Eg: Use when you need to access a row of a 2D matrix of size [N, 3]
         * as RS_Vec3D vector.
         *
         * @tparam U The type to convert to.
         * @return Reference to the memory element converted to the specified type.
         */
        template <typename U>
        U& as() {
            return *reinterpret_cast<U*>(nd_mem->data.get() + offset);
        }
    };

    /// Shared pointer to the memory data.
    std::shared_ptr<T> data;

    /// Device location of the memory.
    RS_DeviceLocation device;

    /// Provider of the memory.
    RS_MemoryProvider* provider = nullptr;

    /// Stride of the memory.
    std::vector<size_t> stride;

    /// Shape of the memory.
    std::vector<size_t> shape;

    RS_NdMemory() = default;

    /**
     * @brief Constructs RS_NdMemory with the specified initial shape.
     *
     * @param init_shape Initial shape of the memory.
     * @param provider Memory provider.
     */
    template <typename MemoryProvider = RS_MemoryDefaultProvider>
    explicit RS_NdMemory(
        const std::vector<size_t>& init_shape,
        MemoryProvider& provider = RS_MemoryDefaultProvider::getInstance()
    ) {
        device = provider.device;
        this->provider = &provider;
        resize(init_shape);
    }

    /// Copy constructor for RS_NdMemory.
    RS_NdMemory(const RS_NdMemory& mem) = default;

    /**
     * @brief Overloaded subscript operator for accessing elements.
     *
     * This operator allows accessing elements of N-dimensional memory using subscript notation.
     *
     * @param index Index of the element to access.
     * @return An RS_NdMemoryAccessor object representing the accessed element.
     */
    RS_NdMemoryAccessor operator[](size_t index) const {
        if (device.type != RSE_DEVICE_TYPE_CPU) {
            RS_THROW(RS_DeviceMemoryException(
                "Cannot directly access memory on device type {}", enumToString(device.type)
            ));
        }
        return std::move(RS_NdMemoryAccessor(this, 0, 0)[index]);
    }

    /**
     * @brief Converts the memory to an RS_MemoryPtr object.
     * @note Returned RS_MemoryPtr is weak and free has no effect.
     */
    RS_MemoryPtr toMemoryPtr() { return {data.get(), sizeof(T) * size(), device, nullptr}; }

    /**
     * @brief Fills the memory with a specified value.
     *
     * @param value Value to fill the memory with.
     */
    void fill(const T& value) {
        std::vector<T> fill_data(size(), value);
        copy(fill_data, shape);
    }

    /**
     * @brief Copies data from a (host) vector into the memory and resizes it accordingly.
     *
     * @param new_data Vector containing the data to be copied.
     * @param new_shape Shape of data to be copied.
     */
    void copy(std::vector<T> new_data, const std::vector<size_t>& new_shape) {
        copy(new_data.data(), new_shape);
    }

    /**
     * @brief Copies data from a (host) pointer into the memory and resizes it accordingly.
     *
     * @param new_data Vector containing the data to be copied.
     * @param new_shape Shape of data to be copied.
     */
    void copy(const T* new_data, const std::vector<size_t>& new_shape) {
        resize<false>(new_shape);
        RS_MemoryPtr dst(data.get(), sizeof(T) * size(), device, provider);
        RS_MemoryPtr src(
            const_cast<T*>(new_data), sizeof(T) * size(), {RSE_DEVICE_TYPE_CPU, 0}, nullptr
        );
        dst.copyFrom(src, sizeof(T) * size());
    }

    /**
     * @brief Copies data from another RS_NdMemory object.
     *
     * @param other The source RS_NdMemory object, could be anywhere supported by the provider's
     * copy method.
     */
    void copy(const RS_NdMemory& other) {
        resize<false>(other.shape);
        RS_MemoryPtr dst(data.get(), sizeof(T) * size(), device, provider);
        RS_MemoryPtr src(other.data.get(), sizeof(T) * other.size(), other.device, other.provider);
        dst.copyFrom(src, sizeof(T) * other.size());
    }

    /**
     * @brief Moves the current RS_NdMemory object to the device where the provider resides,
     * creating a new RS_NdMemory object.
     *
     * @param provider Memory provider that resides on the target device.
     * @return A new RS_NdMemory object with moved data on the target device.
     */
    template <typename MemoryProvider>
    RS_NdMemory move(MemoryProvider& provider = RS_MemoryDefaultProvider::getInstance()) {
        RS_NdMemory new_memory;
        new_memory.template resize<false>(shape, provider);
        RS_MemoryPtr dst(new_memory.data.get(), sizeof(T) * size(), new_memory.device, &provider);
        RS_MemoryPtr src(data.get(), sizeof(T) * size(), device, nullptr);
        provider.copy(dst, src, sizeof(T) * size());

        return new_memory;
    }

    /**
     * @brief Resizes the memory to the specified shape.
     *
     * @tparam Copy Flag indicating whether to copy existing data (default is true).
     * @param new_shape New shape of the memory.
     */
    template <bool Copy = true>
    void resize(const std::vector<size_t>& new_shape) {
        if (!new_shape.empty()) {
            size_t new_total_size = 1, old_total_size = shape.empty() ? 0 : 1;
            for (auto& dim_size : new_shape) new_total_size *= dim_size;
            for (auto& dim_size : shape) old_total_size *= dim_size;

            if (new_total_size != 0) {
                auto mem_ptr = provider->allocate(new_total_size * sizeof(T), true);
                T* new_data = static_cast<T*>(mem_ptr.data);
                if (new_data == nullptr) {
                    RS_THROW(RS_DeviceMemoryException(
                        fmt::format(
                            "Failed to allocate memory on device type {} id {}",
                            enumToString(provider->device.type),
                            provider->device.id
                        )
                    ));
                }
                if (old_total_size != 0 && Copy) {
                    mem_ptr.copyFrom(
                        {data.get(), sizeof(T) * old_total_size, 0, device, nullptr},
                        sizeof(T) * RS_MIN(old_total_size, new_total_size)
                    );
                }
                shape = new_shape;
                data.reset(new_data, [&mem_ptr](T*) { mem_ptr.free(); });

            } else {
                shape = new_shape;
                data.reset();
            }
        } else {
            shape.clear();
            data.reset();
        }
        _updateStride();
    }

    /**
     * @brief Checks if the memory is empty.
     *
     * @return True if the memory is empty, false otherwise.
     */
    [[nodiscard]] bool empty() const { return size() == 0; }

    /**
     * @brief Calculates the total size (number of elements) of the memory.
     *
     * @return Total size of the memory.
     */
    [[nodiscard]] size_t size() const {
        size_t total_size = shape.empty() ? 0 : 1;
        for (auto& dim_size : shape) total_size *= dim_size;
        return total_size;
    }

   private:
    /**
     * @brief Updates the stride of the memory based on its shape.
     */
    void _updateStride() {
        stride.resize(shape.size());
        size_t current_stride = 1;
        size_t index = shape.size() - 1;
        for (auto iter = shape.rbegin(); iter != shape.rend(); ++iter, index--) {
            stride[index] = current_stride;
            current_stride *= *iter;
        }
    }
};

}  // namespace rise

#endif  // RS_MEMORY_H
