// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_ND_MEMORY_H
#define RS_ND_MEMORY_H

#include <fmt/ranges.h>

#include <memory>
#include <vector>

#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/string/rs_format.h"

namespace rise {

/*****************************************************************
 * Class Definitions
 *****************************************************************/
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
 * @tparam T The type of elements stored in the memory.
 */
template <typename T>
class RS_NdMemory {
   public:
    /**
     * @brief A struct representing the managed memory pointer.
     */
    struct ManagedMemoryPtr {
        RS_MemoryPtr ptr;
        bool is_managed = true;

        ManagedMemoryPtr(RS_MemoryPtr ptr, bool is_managed = true)
            : ptr(ptr), is_managed(is_managed) {}

        void free() {
            if (is_managed) {
                ptr.free();
            }
        }

        [[nodiscard]] RS_MemoryPtr take() {
            auto ptr = this->ptr;
            this->ptr = {};
            this->is_managed = false;
            return ptr;
        }
    };

    /**
     * @brief A struct representing an accessor for N-dimensional memory.
     *
     * This struct allows accessing elements of N-dimensional memory.
     * It provides functionality for de-referencing, pointer access, and subscripting.
     */
    struct Accessor {
        /// Pointer to the N-dimensional memory.
        const RS_NdMemory* nd_mem = nullptr;
        /// Offset within the memory.
        RBigSize offset = 0;
        /// Dimension of the memory.
        RBigSize dim = 0;

        /**
         * @brief Constructs an Accessor object.
         *
         * Constructs an Accessor object with the specified parameters.
         *
         * @param nd_mem Pointer to the N-dimensional memory.
         * @param offset Offset within the memory.
         * @param dim Dimension of the memory.
         */
        Accessor(const RS_NdMemory* nd_mem, RBigSize offset, RBigSize dim)
            : nd_mem(nd_mem), offset(offset), dim(dim) {}

        /**
         * @brief Subscript operator for accessing elements recursively.
         *
         * This operator allows accessing elements of N-dimensional memory recursively.
         *
         * @param index Index of the element to access.
         * @return An Accessor object representing the accessed element.
         * @throws RS_OutOfBoundsException if the accessed dimension exceeds the memory's
         * dimensions.
         */
        Accessor operator[](RBigSize index) {
            if (dim >= nd_mem->shape.size()) {
                RS_THROW(RS_OutOfBoundsException(
                    "Trying to access dim {} of a {} dim memory", dim + 1, nd_mem->shape.size()
                ));
            } else {
                RBigSize new_offset = offset + nd_mem->stride[dim] * index;
                return Accessor(nd_mem, new_offset, dim + 1);
            }
        }

        /**
         * @return Pointer to the current memory location.
         */
        T* ptr() { return nd_mem->data->ptr.template as<T>() + offset; }

        /**
         * @return Pointer to the current memory location.
         */
        T* operator->() { return nd_mem->data->ptr.template as<T>() + offset; }

        /**
         * @brief Assign a value to the current memory element.
         * @param other The value to assign.
         */
        Accessor& operator=(const T& other) {
            set(other);
            return *this;
        }

        /**
         * @brief Set the value of the current memory element.
         * @param other The value to assign.
         */
        void set(const T& other) {
            auto& data_ptr = nd_mem->data->ptr;
            if (data_ptr.device().type != RSE_DEVICE_TYPE_CPU) {
                (data_ptr + sizeof(T) * offset)
                    .copyFrom(RS_ConstMemoryPtr(&other, sizeof(T)), sizeof(T));
            } else {
                *ptr() = other;
            }
        }

        /**
         * @return Value of the current memory element.
         */
        T get() {
            auto& data_ptr = nd_mem->data->ptr;
            if (data_ptr.device().type != RSE_DEVICE_TYPE_CPU) {
                T result;
                (data_ptr + sizeof(T) * offset).copyTo(RS_MemoryPtr(&result, sizeof(T)), sizeof(T));
                return result;
            } else {
                return *ptr();
            }
        }

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
        U as() {
            auto& data_ptr = nd_mem->data->ptr;
            if (data_ptr.device().type != RSE_DEVICE_TYPE_CPU) {
                U result;
                (data_ptr + sizeof(T) * offset).copyTo(RS_MemoryPtr(&result, sizeof(U)), sizeof(U));
                return result;
            } else {
                return *reinterpret_cast<U*>(data_ptr.template as<T>() + offset);
            }
        }
    };

    /// Shared pointer to the managed memory pointer.
    std::shared_ptr<ManagedMemoryPtr> data;

    /// Stride of the memory.
    std::vector<RBigSize> stride;

    /// Shape of the memory.
    std::vector<RBigSize> shape;

    RS_NdMemory() = default;

    /**
     * @brief Constructs RS_NdMemory with the specified initial 1D size.
     *
     * @param init_size Initial size of the memory.
     * @param provider Memory provider.
     * @param is_host Whether memory is allocated by allocateHost method of the provider
     */
    RS_NdMemory(
        RBigSize init_size,
        RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance(),
        bool is_host = true
    ) {
        RS_MemoryPtr mem = is_host ? provider->allocateHost(sizeof(T) * init_size)
                                   : provider->allocate(sizeof(T) * init_size);
        data =
            std::shared_ptr<ManagedMemoryPtr>(new ManagedMemoryPtr(mem), [](ManagedMemoryPtr* ptr) {
                ptr->free();
                delete ptr;
            });
        shape = {init_size};
        _updateStride();
    }

    /**
     * @brief Constructs RS_NdMemory with the specified initial shape.
     *
     * @param init_shape Initial shape of the memory.
     * @param provider Memory provider.
     * @param is_host Whether memory is allocated by allocateHost method of the provider
     */
    RS_NdMemory(
        const std::vector<RBigSize>& init_shape,
        RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance(),
        bool is_host = true
    ) {
        RBigSize total_size = 1;
        for (auto& dim : init_shape) total_size *= dim;
        RS_MemoryPtr mem = is_host ? provider->allocateHost(sizeof(T) * total_size)
                                   : provider->allocate(sizeof(T) * total_size);
        data =
            std::shared_ptr<ManagedMemoryPtr>(new ManagedMemoryPtr(mem), [](ManagedMemoryPtr* ptr) {
                ptr->free();
                delete ptr;
            });
        shape = init_shape;
        _updateStride();
    }

    /**
     * @brief Constructs RS_NdMemory with the specified initial 1D size and a given RS_MemoryPtr.
     *
     * @param size Initial size of the memory. Can be a slice of the given RS_MemoryPtr.
     * @param mem RS_MemoryPtr object to manage.
     * @param is_managed Whether to manage the RS_MemoryPtr object (free on shared ptr
     * destruction).
     */
    RS_NdMemory(RS_MemoryPtr mem, RBigSize size, bool is_managed = true) {
        if (is_managed && mem.isWeak()) {
            RS_THROW(RS_InvalidArgumentException(
                "Cannot construct a managed RS_NdMemory with a provider-less memory pointer"
            ));
        }
        if (sizeof(T) * size > mem.size()) {
            RS_THROW(RS_InvalidArgumentException(
                "Constructing RS_NdMemory requires store type {} of shape [{}], total needed size "
                "is {}, but memory size is {}",
                typeid(T).name(),
                size,
                sizeof(T) * size,
                mem.size()
            ));
        }

        data = std::shared_ptr<ManagedMemoryPtr>(
            new ManagedMemoryPtr(mem, is_managed), [](ManagedMemoryPtr* ptr) {
                ptr->free();
                delete ptr;
            }
        );
        shape = {size};
        _updateStride();
    }

    /**
     * @brief Constructs RS_NdMemory with the specified initial shape and a given RS_MemoryPtr.
     *
     * @param shape Initial shape of the memory. Can be a slice of the given RS_MemoryPtr.
     * @param mem RS_MemoryPtr object to manage.
     * @param is_managed Whether to manage the RS_MemoryPtr object (free on shared ptr destruction).
     */
    RS_NdMemory(RS_MemoryPtr mem, const std::vector<RBigSize>& shape, bool is_managed = true) {
        if (is_managed && mem.isWeak()) {
            RS_THROW(RS_InvalidArgumentException(
                "Cannot construct a managed RS_NdMemory with a provider-less memory pointer"
            ));
        }

        RBigSize total_size = 1;
        for (auto& dim : shape) total_size *= dim;
        if (sizeof(T) * total_size > mem.size()) {
            RS_THROW(RS_InvalidArgumentException(
                "Constructing RS_NdMemory requires store type {} of shape [{}], total needed size "
                "is {}, but memory size is {}",
                typeid(T).name(),
                fmt::join(shape, ","),
                sizeof(T) * total_size,
                mem.size()
            ));
        }

        data = std::shared_ptr<ManagedMemoryPtr>(
            new ManagedMemoryPtr(mem, is_managed), [](ManagedMemoryPtr* ptr) {
                ptr->free();
                delete ptr;
            }
        );
        this->shape = shape;
        _updateStride();
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
    Accessor operator[](RBigSize index) const {
        if (data->ptr.device().type != RSE_DEVICE_TYPE_CPU) {
            RS_THROW(RS_DeviceMemoryException(
                "Cannot directly access memory on device type {}",
                enumToString(data->ptr.device().type)
            ));
        }
        return std::move(Accessor(this, 0, 0)[index]);
    }

    /**
     * @brief Returns the managed RS_MemoryPtr object.
     */
    [[nodiscard]] RS_MemoryPtr toMemoryPtr() const { return data->ptr; }

    /**
     * @brief Takes the memory pointer and release memory management handled by the RS_NdMemory.
     * @return The RS_MemoryPtr for the field.
     */
    RS_MemoryPtr take() {
        auto ptr = data->take();
        data.reset();
        shape.clear();
        stride.clear();
        return ptr;
    }

    /**
     * @brief Fills the memory with a specified value.
     *
     * @param value Value to fill the memory with.
     */
    void fill(const T& value) {
        std::vector<T> fill_data(size(), value);
        write(fill_data.data());
    }

    /**
     * @brief Writes data from a host vector into the memory and resizes accordingly.
     *
     * @param input Vector containing the data to be copied.
     */
    void write(std::vector<T> input) {
        resize(input.size(), false);
        write(input.data());
    }

    /**
     * @brief Writes data from a host pointer into the memory. The host pointer must be
     * the same size as the memory.
     *
     * @param input Pointer to the data to be copied.
     */
    void write(const T* input) {
        data->ptr.copyFrom(
            RS_MemoryPtr(const_cast<T*>(input), sizeof(T) * size()), sizeof(T) * size()
        );
    }

    /**
     * @brief Writes a slice of data from a host pointer into the memory.
     *
     * @param input Pointer to the data to be copied.
     * @param num Number of elements to copy.
     * @param offset Offset number of elements into the memory to start copying.
     */
    void writeSlice(const T* input, RBigSize num, RBigSize offset) {
        (data->ptr + sizeof(T) * offset)
            .copyFrom(RS_MemoryPtr(const_cast<T*>(input), sizeof(T) * num), sizeof(T) * num);
    }

    /**
     * @brief Reads data from the memory into a host vector.
     *
     * @param output Vector to store the read data.
     */
    void read(std::vector<T>& output) {
        output.resize(size());
        read(output.data());
    }

    /**
     * @brief Reads data from the memory into a host pointer. The host pointer must be
     * the same size as the memory.
     *
     * @param output Pointer to store the read data.
     */
    void read(T* output) {
        data->ptr.copyTo(RS_MemoryPtr(output, sizeof(T) * size()), sizeof(T) * size());
    }

    /**
     * @brief Reads a slice of data from the memory into a host pointer.
     *
     * @param output Pointer to store the read data.
     * @param num Number of elements to read.
     * @param offset Offset number of elements into the memory to start reading.
     */
    void readSlice(T* output, RBigSize num, RBigSize offset) {
        (data->ptr + sizeof(T) * offset)
            .copyTo(RS_MemoryPtr(output, sizeof(T) * num), sizeof(T) * num);
    }

    /**
     * @brief Copies data from another RS_NdMemory object.
     *
     * @param other The source RS_NdMemory object, could be anywhere supported by the provider's
     */
    void copy(const RS_NdMemory& other) {
        resize(other.shape, false);
        data->ptr.copyFrom(other.data->ptr, sizeof(T) * other.size());
    }

    /**
     * @brief Moves the current RS_NdMemory object to the device where the provider resides,
     * creating a new RS_NdMemory object.
     *
     * @param provider Memory provider that resides on the target device.
     * @return A new RS_NdMemory object with moved data on the target device.
     */
    RS_NdMemory move(RS_MemoryProvider* provider, bool is_host = false) {
        RS_NdMemory new_memory(shape, provider, is_host);
        new_memory.copy(*this);
        return new_memory;
    }

    /**
     * @brief Resizes the memory to the specified 1D size.
     *
     * @param new_size New size of the memory.
     * @param copy Flag indicating whether to copy existing data (default is true).
     */
    void resize(RBigSize new_size, bool copy = true) {
        resize(std::vector<RBigSize>{new_size}, copy);
    }

    /**
     * @brief Resizes the memory to the specified shape.
     *
     * @param new_shape New shape of the memory.
     * @param copy Flag indicating whether to copy existing data (default is true).
     */
    void resize(const std::vector<RBigSize>& new_shape, bool copy = true) {
        if (!new_shape.empty()) {
            RBigSize new_total_size = 1, old_total_size = shape.empty() ? 0 : 1;
            for (auto& dim_size : new_shape) new_total_size *= dim_size;
            for (auto& dim_size : shape) old_total_size *= dim_size;

            if (new_total_size != 0) {
                auto mem_ptr = data->ptr.provider()->allocate(new_total_size * sizeof(T));
                if (old_total_size != 0 && copy) {
                    mem_ptr.copyFrom(data->ptr, sizeof(T) * RS_MIN(old_total_size, new_total_size));
                }
                shape = new_shape;
                auto old_mem_ptr = data->ptr;
                old_mem_ptr.free();
                data->ptr = mem_ptr;
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

    [[nodiscard]] RS_DeviceLocation device() const { return data->ptr.device(); }

    [[nodiscard]] RS_MemoryProvider* provider() const { return data->ptr.provider(); }

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
    [[nodiscard]] RBigSize size() const {
        RBigSize total_size = shape.empty() ? 0 : 1;
        for (auto& dim_size : shape) total_size *= dim_size;
        return total_size;
    }

    /// No generic serialization

   private:
    /**
     * @brief Updates the stride of the memory based on its shape.
     */
    void _updateStride() {
        stride.resize(shape.size());
        RBigSize current_stride = 1;
        RBigSize index = shape.size() - 1;
        for (auto iter = shape.rbegin(); iter != shape.rend(); ++iter, index--) {
            stride[index] = current_stride;
            current_stride *= *iter;
        }
    }
};

}  // namespace rise

#endif  // RS_ND_MEMORY_H