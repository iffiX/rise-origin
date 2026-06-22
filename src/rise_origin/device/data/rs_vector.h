// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VECTOR_H
#define RS_VECTOR_H

#include <cassert>

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"
#include "rs_data_util.h"

namespace rise {

namespace device {

class RSCF_VectorFill : public RS_ComputeFunction<RSCF_VectorFill> {
   public:
    template <typename Vector, typename T>
    HOST_DEVICE_INLINE_QUALIFIER static void run(Vector vector, T value, RSize offset = 0) {
        vector[index1D() + offset] = value;
    }
};

/**
 * @brief RS_Vector is a host-friendly container similar to RS_Array<T,1>,
 * but it also tracks used elements and provides erase, insert, and append methods.
 *
 * @tparam T The data type of the elements in the container.
 */
template <typename T>
class RS_Vector {
   public:
    using ObjectType = T;

    /*****************************************************************
     * Data
     *****************************************************************/
    RS_MemoryPtr data;        ///< Pointer to the data stored on the device.
    RSize elem_num = 0;       ///< Capacity: total number of allocated elements.
    RSize used_elem_num = 0;  ///< Number of elements actually in use.

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_Vector() = default;

    /**
     * @brief Constructs an empty vector with the given provider.
     * @param provider Memory provider.
     */
    explicit RS_Vector(RS_MemoryProvider* provider) { init(provider); }

    /**
     * @brief Constructs a vector with the given initial capacity and provider.
     *
     * @param init_capacity Initial capacity of the vector.
     * @param provider Memory provider.
     */
    RS_Vector(RSize init_capacity, RS_MemoryProvider* provider) { init(init_capacity, provider); }

    /**
     * @brief Initialize the vector as an empty vector with the given provider.
     *
     * @param provider Memory provider.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        used_elem_num = 0;
    }

    /**
     * @brief Initialize the vector with the given initial capacity and provider.
     *
     * @param init_capacity Initial capacity of the vector.
     * @param provider Memory provider.
     */
    void init(RSize init_capacity, RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        used_elem_num = 0;
        resize(init_capacity, false);
    }

    /**
     * @brief Copy constructor (shallow copy: only pointer and size info).
     * @param vec RS_Vector to be copied.
     */
    RS_Vector(const RS_Vector& vec) = default;

    /**
     * @brief Frees the GPU memory associated with the vector.
     * @warning It will free the data; all related vectors will be affected.
     */
    void free() {
        data.free();
        elem_num = 0;
        used_elem_num = 0;
    }

    /**
     * @brief Resizes the vector capacity to new_capacity.
     *
     * @param new_capacity The new capacity of the vector.
     * @param copy If true, the existing data is copied to the new allocation (up to used_elem_num).
     */
    void resize(RSize new_capacity, bool copy = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        auto old_data = data;
        RSize old_capacity = elem_num;

        // Update metadata
        elem_num = new_capacity;

        // If the old buffer had no capacity (or was not allocated), just allocate fresh
        auto new_data = old_data.provider()->allocate(sizeof(T) * new_capacity, true);

        // If copy is requested, copy up to the min of old capacity or used_elem_num
        if (copy && old_capacity != 0) {
            const RSize amount_to_copy = RS_MIN(used_elem_num, new_capacity);
            if (amount_to_copy > 0) {
                new_data.copyFrom(old_data, sizeof(T) * amount_to_copy);
            }
        }

        // If new capacity < used_elem_num, we have to clamp used_elem_num
        if (used_elem_num > new_capacity) {
            used_elem_num = new_capacity;
        }

        data = new_data;
        old_data.free();
    }

    /**
     * @brief Fills the entire allocated capacity of the vector with the given value.
     *
     * @param value Value to fill with.
     */
    void fill(const T& value, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        if (elem_num > 0) {
            // If the value is zero, it is faster to just fill memory with 0
            if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
                if (value == static_cast<T>(0)) {
                    data.set(0, data.size(), 0, sync);
                    used_elem_num = elem_num;
                    return;
                }
            }
            used_elem_num = elem_num;
            castProviderPtrAndCall(data.provider(), [&](auto p_ptr) {
                p_ptr->template run<RSCF_VectorFill>(
                    {elem_num, RS_AUTO_BLOCK_SIZE, sync}, *this, value
                );
            });
        }
    }

    /**
     * @brief Fills a slice of the vector with the given value.
     *
     * @param value Value to fill with.
     * @param num The number of elements to be filled.
     * @param offset The starting offset of the slice in the vector.
     */
    void fillSlice(const T& value, RSize num, RSize offset, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_BOUND(
            offset + num <= elem_num, "offset={}, num={}, elem_num={}", offset, num, elem_num
        );

        if (num > 0) {
            used_elem_num = RS_MAX(used_elem_num, offset + num);
            castProviderPtrAndCall(data.provider(), [&](auto p_ptr) {
                p_ptr->template run<RSCF_VectorFill>(
                    {num, RS_AUTO_BLOCK_SIZE, sync}, *this, value, offset
                );
            });
        }
    }

    /**
     * @brief Writes data from the host to the vector. Host data size must be equal to capacity.
     *
     * @param input Pointer to the input data on the host.
     */
    void write(const T* input, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            input != nullptr || elem_num == 0,
            "When writing to a non-empty vector, input pointer must not be null."
        );

        if (elem_num > 0) {
            data.copyFrom(
                RS_ConstMemoryPtr(input, sizeof(T) * elem_num), sizeof(T) * elem_num, sync
            );
        }
        used_elem_num = elem_num;
    }

    /**
     * @brief Write a slice of host data to the device vector.
     *
     * @param input The host data to be written.
     * @param num The number of elements to be written.
     * @param offset The starting offset of the slice in the vector.
     */
    void writeSlice(const T* input, RSize num, RSize offset, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            input != nullptr || num == 0,
            "When writing to a non-empty slice of a vector, input pointer must not be null."
        );
        RS_DATA_EXPECT_BOUND(
            offset + num <= elem_num, "offset={}, num={}, elem_num={}", offset, num, elem_num
        );

        if (num > 0) {
            (data + sizeof(T) * offset)
                .copyFrom(RS_ConstMemoryPtr(input, sizeof(T) * num), sizeof(T) * num, sync);
            // If we wrote beyond the current used_elem_num, update it
            if (offset + num > used_elem_num) {
                used_elem_num = offset + num;
            }
        }
    }

    /**
     * @brief Copies data from another RS_Vector. Resize as needed.
     * (if current capacity is smaller than used elements in the other vector)
     *
     * @param vec RS_Vector to copy data from.
     */
    void copy(const RS_Vector& vec) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(vec.data.provider() != nullptr);

        if (vec.data == data) return;
        if (vec.used_elem_num > elem_num) {
            resize(vec.used_elem_num, false);
        }
        used_elem_num = vec.used_elem_num;
        if (elem_num > 0) {
            data.copyFrom(vec.data, sizeof(T) * used_elem_num);
        }
    }

    /**
     * @brief Copies used data from the device to host.
     *
     * @param output The host buffer to store the data.
     */
    void read(T* output, bool sync = true) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            output != nullptr || used_elem_num == 0,
            "When reading from a non-empty vector, output pointer must not be null."
        );

        if (used_elem_num > 0) {
            data.copyTo(
                RS_MemoryPtr(output, sizeof(T) * used_elem_num), sizeof(T) * used_elem_num, sync
            );
        }
    }

    /**
     * @brief Read a slice of device data to the host.
     *
     * @param output The host buffer to write to.
     * @param num The number of elements to be read.
     * @param offset The starting offset of the slice in the vector.
     */
    void readSlice(T* output, RSize num, RSize offset, bool sync = true) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            output != nullptr || num == 0,
            "When reading a non-empty slice of a vector, output pointer must not be null."
        );
        RS_DATA_EXPECT_BOUND(
            offset + num <= used_elem_num,
            "offset={}, num={}, used_elem_num={}",
            offset,
            num,
            used_elem_num
        );
        if (num > 0) {
            (data + sizeof(T) * offset)
                .copyTo(RS_MemoryPtr(output, sizeof(T) * num), sizeof(T) * num, sync);
        }
    }

    /**
     * @brief Moves data to the specified provider and returns a new RS_Vector.
     *
     * @param provider Target memory provider.
     * @return A new RS_Vector on the specified provider.
     */
    RS_Vector move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        if (elem_num == 0) return RS_Vector(provider);
        RS_Vector new_vec;
        new_vec.data = {nullptr, 0, 0, provider->device, provider};
        new_vec.resize(elem_num, false);
        new_vec.used_elem_num = used_elem_num;
        new_vec.data.copyFrom(data, sizeof(T) * used_elem_num);
        return new_vec;
    }

    /**
     * @brief Inserts a value at a given index.
     *
     * @param value The value to be inserted.
     * @param index The index at which to insert the value.
     */
    void insert(T value, RIndex index) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_BOUND(
            index <= used_elem_num, "index={}, used_elem_num={}", index, used_elem_num
        );

        // Check if we need to resize (expand capacity)
        if (used_elem_num == elem_num) {
            // For simplicity, double the capacity (or allocate at least 1 if empty)
            RSize new_cap = (elem_num == 0) ? 1 : (elem_num * 2);
            resize(new_cap, true);
        }

        // Shift elements in [index, used_elem_num) to the right by 1
        if (index < used_elem_num) {
            auto tmp = data.provider()->allocate(sizeof(T) * (used_elem_num - index), true);
            (data + sizeof(T) * index).copyTo(tmp, sizeof(T) * (used_elem_num - index));
            // Move them one slot over
            (data + sizeof(T) * (index + 1)).copyFrom(tmp, sizeof(T) * (used_elem_num - index));
            tmp.free();
        }
        // Write the new value at index
        (data + sizeof(T) * index).copyFrom(RS_ConstMemoryPtr(&value, sizeof(T)), sizeof(T));
        used_elem_num++;
    }

    /**
     * @brief Appends a value at the end of the used elements.
     *
     * @param value The value to be appended.
     */
    void append(T value) { insert(value, used_elem_num); }

    void erase(RIndex index) { eraseRange(index, index + 1); }

    void clear() {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        used_elem_num = 0;
    }

    void occupy(RIndex new_used_elem_num) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_BOUND(
            new_used_elem_num <= elem_num,
            "new_used_elem_num={}, elem_num={}",
            new_used_elem_num,
            elem_num
        );
        used_elem_num = new_used_elem_num;
    }

    /**
     * @brief Erases elements in the range [begin, end).
     *
     * @param begin The starting index (inclusive).
     * @param end The ending index (exclusive).
     */
    void eraseRange(RIndex begin, RIndex end) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_BOUND(end <= used_elem_num, "end={}, used_elem_num={}", end, used_elem_num);

        if (begin >= end) {
            // No-op
            return;
        }
        RSize num_to_erase = end - begin;
        // Move the tail [end, used_elem_num) down to [begin, ...)
        if (end < used_elem_num) {
            (data + sizeof(T) * begin)
                .copyFrom(data + sizeof(T) * end, sizeof(T) * (used_elem_num - end));
        }
        used_elem_num -= num_to_erase;
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return data.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return data.device(); }

    /**
     * @brief Returns a pointer to the underlying data.
     */
    HOST_DEVICE_INLINE_QUALIFIER T* ptr() { return static_cast<T*>(data.ptr()); }

    /**
     * @brief Returns the capacity (total number of allocated elements).
     */
    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize capacity() const { return elem_num; }

    /**
     * @brief Returns the number of used elements (logical size).
     */
    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize size() const { return used_elem_num; }

    /**
     * @brief Returns true if the used size is 0.
     */
    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return used_elem_num == 0; }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize baseBytes() const { return sizeof(*this); }
    
    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize extraBytes() const { return sizeof(T) * elem_num; }

    HOST_DEVICE_INLINE_QUALIFIER T& back() {
        assert(used_elem_num > 1 && "Cannot get the last element of an empty vector");
        return data.get<T>(used_elem_num - 1);
    }

    HOST_DEVICE_INLINE_QUALIFIER const T& back() const {
        assert(used_elem_num > 1 && "Cannot get the last element of an empty vector");
        return data.get<const T>(used_elem_num - 1);
    }

    HOST_DEVICE_INLINE_QUALIFIER T& front() {
        assert(used_elem_num > 1 && "Cannot get the first element of an empty vector");
        return data.get<T>(0);
    }

    HOST_DEVICE_INLINE_QUALIFIER const T& front() const {
        assert(used_elem_num > 1 && "Cannot get the first element of an empty vector");
        return data.get<const T>(0);
    }

    HOST_DEVICE_INLINE_QUALIFIER T& operator[](RIndex index) { return data.get<T>(index); }

    HOST_DEVICE_INLINE_QUALIFIER const T& operator[](RIndex index) const {
        return data.get<const T>(index);
    }

    /**
     * @brief Get method for the vector.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& get(RIndex index) { return data.get<T>(index); }

    /**
     * @brief Const get method for the vector.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& get(RIndex index) const {
        return data.get<const T>(index);
    }

    /**
     * @brief Set method for the vector.
     */
    HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, const T& value) {
        data.get<T>(index) = value;
    }

    /**
     * @brief Set the value if the value is constructible
     * from the given value type T2.
     */
    template <typename T2>
    HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, const T2& value) {
        static_assert(
            std::is_constructible<T, T2>::value,
            "To use set with T2, Type T must be constructible from type T2."
        );
        data.get<T>(index) = T(value);
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_VECTOR_H
