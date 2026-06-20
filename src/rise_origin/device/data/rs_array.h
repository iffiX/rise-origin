// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_ARRAY_H
#define RS_ARRAY_H

#include <array>
#include <cassert>

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/device/rs_dev_util.h"
#include "rise_origin/device/rs_static_construct.h"
#include "rs_data_util.h"

namespace rise {

namespace device {

class RSCF_ArrayFill : public RS_ComputeFunction<RSCF_ArrayFill> {
   public:
    template <typename Array, typename T>
    HOST_DEVICE_INLINE_QUALIFIER static void run(Array array, T value, RSize offset = 0) {
        array[index1D() + offset] = value;
    }
};

/**
 * @brief RS_Array is a template class for representing fixed dimension,
 * dynamic size, multidimensional arrays on the device.
 *
 * @tparam T The data type of the elements in the array.
 * @tparam Dimension_ The number of dimensions of the array.
 * @tparam IsRowMajor_ Whether the array is row major.
 */
template <typename T, RSize Dimension_, bool IsRowMajor_ = true>
class RS_Array {
   public:
    using ObjectType = T;
    static constexpr RSize Dimension = Dimension_;
    static constexpr bool IsRowMajor = IsRowMajor_;
    /*****************************************************************
     * Data
     *****************************************************************/
    RS_MemoryPtr data;             ///< Pointer to the data stored on the device.
    RSize elem_num = 0;            ///< Total number of elements in the array.
    RSize shape[Dimension] = {0};  ///< Shape of the array in each dimension.

    /**
     * @brief Divisor and modulus for each dimension.
     * @note When array is row major, the divisors are aligned to the right.
     *       When array is column major, the divisors are aligned to the left.
     *
     * Eg: For row major array with shape = {2, 3, 4}, the divisors are {12, 4}.
     *                                          ^  ^   (Divisors)
     *     For column major array with shape = {2, 3, 4}, the divisors are {2, 6}.
     *                                          ^  ^   (Divisors)
     */
    rs_device_std::array<RS_DivMod<RSize>, Dimension - 1> layer_dm;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_Array() = default;

    /**
     * @brief Constructs an empty array with the given provider.
     * @param provider Memory provider.
     */
    explicit RS_Array(RS_MemoryProvider* provider) { init(provider); }

    /**
     * @brief Constructs an array with the given shape and provider.
     *
     * @param init_shape Initial shape of the array.
     * @param provider Memory provider.
     */
    RS_Array(std::array<RSize, Dimension> init_shape, RS_MemoryProvider* provider) {
        init(init_shape, provider);
    }

    /**
     * @brief Constructs an array with the given size provider, only available for
     * 1-dimensional arrays.
     *
     * @param init_size Initial size of the array.
     * @param provider Memory provider.
     */
    RS_Array(RSize init_size, RS_MemoryProvider* provider) {
        static_assert(Dimension == 1, "This method is only available for 1-dimensional arrays.");
        init(init_size, provider);
    }

    /**
     * @brief Copy constructor.
     * @warning It is a shallow copy which only copies pointer and shape.
     * @param array RS_Array to be copied.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Array(const RS_Array& array)
        : data(array.data), elem_num(array.elem_num) {
        for (int i = 0; i < Dimension; i++) shape[i] = array.shape[i];
        for (int i = 0; i < Dimension - 1; i++) layer_dm[i] = array.layer_dm[i];
    }

    /**
     * @brief Initialize the array as an empty array with the given provider.
     *
     * @param provider Memory provider.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        for (RSize& dim_shape : shape) dim_shape = 0;
        for (auto& dm : layer_dm) dm = RS_DivMod<RSize>();
    }

    /**
     * @brief Initialize the array with the given shape and provider.
     *
     * @param init_shape Initial shape of the array.
     * @param provider Memory provider.
     */
    void init(std::array<RSize, Dimension> init_shape, RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        for (RSize& dim_shape : shape) dim_shape = 0;
        for (auto& dm : layer_dm) dm = RS_DivMod<RSize>();
        resize(init_shape, false);
    }

    /**
     * @brief Initialize the array with the given size provider, only available for
     * 1-dimensional arrays.
     *
     * @param init_size Initial size of the array.
     * @param provider Memory provider.
     */
    void init(RSize init_size, RS_MemoryProvider* provider) {
        static_assert(Dimension == 1, "This method is only available for 1-dimensional arrays.");
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        for (RSize& dim_shape : shape) dim_shape = 0;
        for (auto& dm : layer_dm) dm = RS_DivMod<RSize>();
        resize(init_size, false);
    }

    /**
     * @brief Frees the GPU memory associated with the array.
     * @warning It will free the data, all related arrays will be affected.
     */
    void free() {
        data.free();
        elem_num = 0;
        for (RSize& dim_shape : shape) dim_shape = 0;
        for (auto& dm : layer_dm) dm = RS_DivMod<RSize>();
    }

    /**
     * @brief Resizes the array to the new shape.
     *
     * @param new_shape New shape of the array.
     * @param copy If true, the existing data is copied to the new array.
     */
    void resize(std::array<RSize, Dimension> new_shape, bool copy = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        // Check if the new shape is the same as the old shape
        bool need_resize = false;
        for (int i = 0; i < Dimension; i++) {
            if (shape[i] != new_shape[i]) {
                need_resize = true;
            }
            shape[i] = new_shape[i];
        }
        if (need_resize) {
            auto old_data = data;
            RSize old_elem_num = elem_num;
            RSize new_elem_num = 1;
            if constexpr (IsRowMajor) {
                for (int i = Dimension - 1; i >= 0; i--) {
                    new_elem_num *= new_shape[i];
                    shape[i] = new_shape[i];
                    // Divisor is aligned to the right
                    if (i > 0) layer_dm[i - 1].setDivisor(new_elem_num);
                }
            } else {
                for (int i = 0; i < Dimension; i++) {
                    new_elem_num *= new_shape[i];
                    shape[i] = new_shape[i];
                    // Divisor is aligned to the left
                    if (i < Dimension - 1) layer_dm[i].setDivisor(new_elem_num);
                }
            }
            elem_num = new_elem_num;
            auto new_data = old_data.provider()->allocate(sizeof(T) * new_elem_num, true);
            if (copy && new_elem_num != 0 && old_elem_num != 0)
                new_data.copyFrom(old_data, sizeof(T) * RS_MIN(old_elem_num, new_elem_num));
            data = new_data;
            old_data.free();
        }
    }

    /**
     * @brief Resizes the array to the new size, only available for 1-dimensional arrays.
     *
     * @param new_size New size of the array.
     * @param copy If true, the existing data is copied to the new array.
     */
    void resize(RSize new_size, bool copy = true) {
        static_assert(Dimension == 1, "This method is only available for 1-dimensional arrays.");
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        std::array<RSize, 1> new_shape{new_size};
        resize(new_shape, copy);
    }

    /**
     * @brief Fills the array with the given value.
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
                    return;
                }
            }
            castProviderPtrAndCall(data.provider(), [&](auto p_ptr) {
                p_ptr->template run<RSCF_ArrayFill>(
                    {elem_num, RS_AUTO_BLOCK_SIZE, sync}, *this, value
                );
            });
        }
    }

    /**
     * @brief Fills a slice of the array with the given value.
     *
     * @param value Value to fill with.
     * @param num The number of elements to be filled.
     * @param offset The starting offset of the slice in the array.
     */
    void fillSlice(const T& value, RSize num, RSize offset, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        if (num + offset > elem_num) {
            RS_THROW(RS_OutOfBoundsException(
                "Trying to write {} elements starting from {} to an array of size {}",
                num,
                offset,
                elem_num
            ));
        }
        if (num > 0) {
            castProviderPtrAndCall(data.provider(), [&](auto p_ptr) {
                p_ptr->template run<RSCF_ArrayFill>(
                    {num, RS_AUTO_BLOCK_SIZE, sync}, *this, value, offset
                );
            });
        }
    }

    /**
     * @brief Writes data from the host to the array. Host data size must be the same as the array.
     *
     * @param input Pointer to the input data on the host.
     * @param sync If true, the write is synchronous, otherwise it is asynchronous.
     */
    void write(const T* input, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            input != nullptr || elem_num == 0,
            "When writing to non-empty array, input pointer must not be null."
        );

        if (elem_num > 0) {
            data.copyFrom(
                RS_ConstMemoryPtr(input, sizeof(T) * elem_num), sizeof(T) * elem_num, sync
            );
        }
    }

    /**
     * @brief Write a slice of host data to the device array.
     *
     * @param input The host data to be written.
     * @param num The number of elements to be written.
     * @param offset The starting offset of the slice in the array.
     * @param sync If true, the write is synchronous, otherwise it is asynchronous.
     */
    void writeSlice(const T* input, RSize num, RSize offset, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            input != nullptr || num == 0,
            "When writing to a non-empty slice of an array, input pointer must not be null."
        );
        RS_DATA_EXPECT_BOUND(
            num + offset <= elem_num, "num={}, offset={}, elem_num={}", num, offset, elem_num
        );

        if (num > 0) {
            (data + sizeof(T) * offset)
                .copyFrom(RS_ConstMemoryPtr(input, sizeof(T) * num), sizeof(T) * num, sync);
        }
    }

    /**
     * @brief Copies data from another RS_Array. Resize as needed.
     *
     * @param array RS_Array to copy data from.
     */
    void copy(const RS_Array& array) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(array.data.provider() != nullptr);

        if (array.data == data) return;
        std::array<RSize, Dimension> new_shape;
        bool need_resize = false;
        for (int i = 0; i < Dimension; i++) {
            if (shape[i] != array.shape[i]) {
                need_resize = true;
            }
            new_shape[i] = array.shape[i];
            if (i < Dimension - 1) layer_dm[i] = array.layer_dm[i];
        }
        if (need_resize) {
            resize(new_shape, false);
        }
        if (elem_num > 0) {
            data.copyFrom(array.data, sizeof(T) * elem_num);
        }
    }

    /**
     * @brief Copies data from device to host.
     *
     * @param output The address to store the data
     */
    void read(T* output, bool sync = true) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            output != nullptr || elem_num == 0,
            "When reading from non-empty array, output pointer must not be null."
        );

        if (elem_num > 0) {
            data.copyTo(RS_MemoryPtr(output, sizeof(T) * elem_num), sizeof(T) * elem_num, sync);
        }
    }

    /**
     * @brief Read a slice of device data to the host.
     *
     * @param output The host data to write to.
     * @param num The number of elements to be read.
     * @param offset The starting offset of the slice in the array.
     */
    void readSlice(T* output, RSize num, RSize offset, bool sync = true) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            output != nullptr || num == 0,
            "When reading a non-empty slice of an array, output pointer must not be null."
        );
        RS_DATA_EXPECT_BOUND(
            num + offset <= elem_num, "num={}, offset={}, elem_num={}", num, offset, elem_num
        );

        if (num > 0) {
            (data + sizeof(T) * offset)
                .copyTo(RS_MemoryPtr(output, sizeof(T) * num), sizeof(T) * num, sync);
        }
    }

    /// @brief Moves data to specified provider and returns a new RS_Array
    RS_Array move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        if (elem_num == 0) return RS_Array(provider);

        std::array<RSize, Dimension> init_shape;
        for (int i = 0; i < Dimension; i++) init_shape[i] = shape[i];
        RS_Array new_array(init_shape, provider);
        new_array.copy(*this);
        return new_array;
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return data.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return data.device(); }

    /**
     * @brief Returns a pointer to the underlying data.
     */
    HOST_DEVICE_INLINE_QUALIFIER T* ptr() { return static_cast<T*>(data.ptr()); }

    /// @brief Returns the number of elements in the array.
    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize size() const { return elem_num; }

    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return elem_num == 0; }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const { return sizeof(T) * elem_num; }

    /**
     * @brief Checks if the given index is out of bounds.
     *
     * @param index Index to check.
     * @return True if the index is out of bounds, otherwise false.
     */
    HOST_DEVICE_INLINE_QUALIFIER bool isOutOfBound(RIndex index) const { return index >= elem_num; }

    /**
     * @brief Converts N-dimensional coordinates to a linear index.
     *
     * @note For readability over general case.
     *
     * @param coords Variable number of coordinate arguments.
     * @returns The linear index.
     */
    template <RSize D = Dimension_, typename... Coords>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<(D > 3 && sizeof...(Coords) == D), RIndex>
    coordsToIndex(Coords... coords) {
        RIndex index;
        RIndex indices[] = {static_cast<RIndex>(coords)...};

        if constexpr (IsRowMajor) {
            index = indices[D - 1];
            for (int i = D - 2; i >= 0; --i) {
                index += indices[i] * layer_dm[i].divisor;
            }
        } else {
            index = indices[0];
            for (int i = 1; i < D; ++i) {
                index += indices[i] * layer_dm[i - 1].divisor;
            }
        }
        return index;
    }

    /**
     * @brief Converts a linear index to N-dimensional coordinates.
     *
     * @note For readability over general case.
     *
     * @param index Linear index to convert.
     * @param coords References to store the N-dimensional coordinates.
     */
    template <RSize D = Dimension_, typename... Coords>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<(D > 3 && sizeof...(Coords) == D), void>
    indexToCoords(RIndex index, Coords&... coords) {
        RIndex* indices[] = {&coords...};
        RIndex remaining_index = index;
        if constexpr (IsRowMajor) {
            for (int i = 0; i < D - 1; ++i) {
                layer_dm[i].divmod(remaining_index, *indices[i], remaining_index);
            }
            *indices[D - 1] = remaining_index;
        } else {
            for (int i = D - 2; i >= 0; --i) {
                layer_dm[i].divmod(remaining_index, *indices[i + 1], remaining_index);
            }
            *indices[0] = remaining_index;
        }
    }

    /**
     * @brief Converts 2D coordinates (x_0, x_1) to a linear index.
     *
     * @note For readability over general case.
     *
     * @param x_0 x_0-coordinate of first dimension.
     * @param x_1 x_1-coordinate of second dimension.
     * @returns The linear index.
     */
    template <RSize D = Dimension_>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<D == 2, RIndex>
    coordsToIndex(RIndex x_0, RIndex x_1) {
        RIndex index;
        if constexpr (IsRowMajor) {
            index = layer_dm[0].divisor * x_0 + x_1;
        } else {
            index = layer_dm[0].divisor * x_1 + x_0;
        }
        return index;
    }

    /**
     * @brief Converts a linear index to 2D coordinates (x_0, x_1).
     *
     * @note For readability over general case.
     *
     * @param index Linear index to convert.
     * @param x_0 Reference to store the x_0-coordinate.
     * @param x_1 Reference to store the x_1-coordinate.
     */
    template <RSize D = Dimension_>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<D == 2, void>
    indexToCoords(RIndex index, RIndex& x_0, RIndex& x_1) {
        if constexpr (IsRowMajor) {
            layer_dm[0].divmod(index, x_0, x_1);
        } else {
            layer_dm[0].divmod(index, x_1, x_0);
        }
    }

    /**
     * @brief Converts 3D coordinates (x_0, x_1, x_2) to a linear index.
     *
     * @param x_0 x_0-coordinate of first dimension.
     * @param x_1 x_1-coordinate of second dimension.
     * @param x_2 x_2-coordinate of third dimension.
     * @returns The linear index.
     */
    template <RSize D = Dimension_>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<D == 3, RIndex>
    coordsToIndex(RIndex x_0, RIndex x_1, RIndex x_2) {
        RIndex index;
        if constexpr (IsRowMajor) {
            index = layer_dm[0].divisor * x_0 + (layer_dm[1].divisor * x_1 + x_2);
        } else {
            index = layer_dm[1].divisor * x_2 + (layer_dm[0].divisor * x_1 + x_0);
        }
        return index;
    }

    /**
     * @brief Converts a linear index to 3D coordinates (x_0, x_1, x_2).
     *
     * @param index Linear index to convert.
     * @param x_0 Reference to store the x_0-coordinate.
     * @param x_1 Reference to store the x_1-coordinate.
     * @param x_2 Reference to store the x_2-coordinate.
     */
    template <RSize D = Dimension_>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<D == 3, void>
    indexToCoords(RIndex index, RIndex& x_0, RIndex& x_1, RIndex& x_2) {
        RIndex remaining_index;
        if constexpr (IsRowMajor) {
            layer_dm[0].divmod(index, x_0, remaining_index);
            layer_dm[1].divmod(remaining_index, x_1, x_2);
        } else {
            layer_dm[1].divmod(index, x_2, remaining_index);
            layer_dm[0].divmod(remaining_index, x_1, x_0);
        }
    }

    /**
     * @brief Access operator for the array.
     *
     * @param index Index of the element to access.
     * @return Reference to the element at the given index.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& operator[](size_t index) {
        assert(index < elem_num);
        return data.get<T>(index);
    }

    /**
     * @brief Access operator for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @return Reference to the element at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& operator[](
        const rs_device_std::array<RSize, Dimension>& coords
    ) {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        assert(index < elem_num);
        return data.get<T>(index);
    }

    /**
     * @brief Const access operator for the array.
     * @param index Index of the element to access.
     * @return Const reference to the element at the given index.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& operator[](size_t index) const {
        assert(index < elem_num);
        return data.get<const T>(index);
    }

    /**
     * @brief Const access operator for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @return Const reference to the element at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& operator[](
        const rs_device_std::array<RSize, Dimension>& coords
    ) const {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        assert(index < elem_num);
        return data.get<const T>(index);
    }

    /**
     * @brief Get method for the array.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& get(RIndex index) { return data.get<T>(index); }

    /**
     * @brief Get method for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @return Reference to the element at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& get(const rs_device_std::array<RSize, Dimension>& coords) {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        assert(index < elem_num);
        return data.get<T>(index);
    }

    /**
     * @brief Const get method for the array.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& get(RIndex index) const {
        assert(index < elem_num);
        return data.get<const T>(index);
    }

    /**
     * @brief Const get method for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @return Const reference to the element at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& get(
        const rs_device_std::array<RSize, Dimension>& coords
    ) const {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        assert(index < elem_num);
        return data.get<const T>(index);
    }

    /**
     * @brief Set method for the array.
     */
    HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, const T& value) {
        assert(index < elem_num);
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
        assert(index < elem_num);
        data.get<T>(index) = T(value);
    }

    /**
     * @brief Set method for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @param value Value to set at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER void
    set(const rs_device_std::array<RSize, Dimension>& coords, const T& value) {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        assert(index < elem_num);
        data.get<T>(index) = value;
    }

    /**
     * @brief Set the value if the value is constructible
     * from the given value type T2.
     */
    template <typename T2>
    HOST_DEVICE_INLINE_QUALIFIER void
    set(const rs_device_std::array<RSize, Dimension>& coords, const T2& value) {
        static_assert(
            std::is_constructible<T, T2>::value,
            "To use set with T2, Type T must be constructible from type T2."
        );
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        assert(index < elem_num);
        data.get<T>(index) = T(value);
    }

   private:
    template <size_t... Is>
    HOST_DEVICE_INLINE_QUALIFIER RIndex _coordArrayToIndex(
        const rs_device_std::array<RSize, Dimension>& coords,
        rs_device_std::index_sequence<Is...>
    ) {
        return coordsToIndex(coords[Is]...);
    }
};

template <typename T, RSize Dimension>
using RS_RowMajorArray = RS_Array<T, Dimension, true>;

template <typename T, RSize Dimension>
using RS_ColumnMajorArray = RS_Array<T, Dimension, false>;

namespace array_helper {
template <RSize N>
struct NextPowerOfTwo {
    static constexpr RSize value = nextPowerOfTwo(N);
};

template <typename T, T Append, typename Seq>
struct AppendSequence;

template <typename T, T Append, T... Seq>
struct AppendSequence<T, Append, rs_device_std::integer_sequence<T, Seq...>> {
    using type = rs_device_std::integer_sequence<T, Seq..., Append>;
};

template <typename T, T Prepend, typename Seq>
struct PrependSequence;

template <typename T, T Prepend, T... Seq>
struct PrependSequence<T, Prepend, rs_device_std::integer_sequence<T, Seq...>> {
    using type = rs_device_std::integer_sequence<T, Prepend, Seq...>;
};

// Helper that recursively builds the reversed sequence.
template <typename T, T... Ns>
struct ReverseSequenceHelper;

// Base case: an empty sequence reverses to itself.
template <typename T>
struct ReverseSequenceHelper<T> {
    using type = rs_device_std::integer_sequence<T>;
};

// Recursive case: reverse the tail and then append the head at the end.
template <typename T, T First, T... Rest>
struct ReverseSequenceHelper<T, First, Rest...> {
    using type =
        typename AppendSequence<T, First, typename ReverseSequenceHelper<T, Rest...>::type>::type;
};

// Primary template to reverse an integer sequence.
template <typename Seq>
struct ReverseSequence;

template <typename T, T... Ns>
struct ReverseSequence<rs_device_std::integer_sequence<T, Ns...>> {
    using type = typename ReverseSequenceHelper<T, Ns...>::type;
};

// Helper to get the element at a specific index in an integer_sequence
template <typename Seq, RSize Index>
struct GetSequenceElement;

// Specialization for accessing the element at the specified index
template <typename T, T First, T... Rest>
struct GetSequenceElement<rs_device_std::integer_sequence<T, First, Rest...>, 0> {
    static constexpr T value = First;
};

// Recursive case: reduce the index and the sequence
template <typename T, T First, T... Rest, RSize Index>
struct GetSequenceElement<rs_device_std::integer_sequence<T, First, Rest...>, Index> {
    static_assert(sizeof...(Rest) >= Index, "Index out of range");
    static constexpr T value =
        GetSequenceElement<rs_device_std::integer_sequence<T, Rest...>, Index - 1>::value;
};

// Convenience variable template to make it easier to access the value
template <typename Seq, RSize Index>
inline constexpr auto GetSequenceElementV = GetSequenceElement<Seq, Index>::value;

// Helper to compute prefix products
template <typename ProcessedSeq, typename RemainingSeq, typename T, T Product>
struct PrefixMultiplySequenceHelper;

// Specialization for non-empty RemainingSeq
template <typename ProcessedSeq, typename T, T Product, T RemainingFirst, T... RemainingRest>
struct PrefixMultiplySequenceHelper<
    ProcessedSeq,
    rs_device_std::integer_sequence<T, RemainingFirst, RemainingRest...>,
    T,
    Product> {
    // Calculate the new product
    static constexpr T NewProduct = Product * RemainingFirst;

    // Recur by appending NewProduct to ProcessedSeq and reducing RemainingSeq
    using type = typename PrefixMultiplySequenceHelper<
        typename AppendSequence<T, NewProduct, ProcessedSeq>::type,
        rs_device_std::integer_sequence<T, RemainingRest...>,
        T,
        NewProduct>::type;
};

// Base case for empty RemainingSeq
template <typename ProcessedSeq, typename T, T Product>
struct PrefixMultiplySequenceHelper<ProcessedSeq, rs_device_std::integer_sequence<T>, T, Product> {
    using type = ProcessedSeq;
};

// Helper struct for prefix multiply sequence calculation
template <typename Seq>
struct PrefixMultiplySequence;

// Specialization for starting the prefix multiply computation
template <typename T, T... Ns>
struct PrefixMultiplySequence<rs_device_std::integer_sequence<T, Ns...>> {
    using type = typename PrefixMultiplySequenceHelper<
        rs_device_std::integer_sequence<T>,
        rs_device_std::integer_sequence<T, Ns...>,
        T,
        1>::type;
};

// Helper struct to compute multiplication
template <typename Seq>
struct Multiply;

template <RSize First, RSize... Rest>
struct Multiply<rs_device_std::integer_sequence<RSize, First, Rest...>> {
    static constexpr RSize value =
        First * Multiply<rs_device_std::integer_sequence<RSize, Rest...>>::value;
};

// Base case
template <RSize First>
struct Multiply<rs_device_std::integer_sequence<RSize, First>> {
    static constexpr RSize value = First;
};

}  // namespace array_helper

template <RSize... DimensionSizes>
struct RS_StaticArraySizeConfig {
    using SizeSequence = std::integer_sequence<RSize, DimensionSizes...>;
};

template <typename T, typename SizeConfig, bool IsRowMajor = true>
class RS_StaticArray;

template <typename T, bool IsRowMajor_, RSize... DimensionSizes>
class RS_StaticArray<T, RS_StaticArraySizeConfig<DimensionSizes...>, IsRowMajor_> {
   public:
    using ObjectType = T;
    using DimensionSizesSequence = rs_device_std::integer_sequence<RSize, DimensionSizes...>;

    /**
     * @brief The sequence of the sizes of the layers. The length of the sequence is
     * the same as the dimension of the array.
     *
     * @note When array is row-major, the first element is the right most dimension size,
     * and the last element is the total number of elements in the array.
     *
     * @note When array is column-major, the first element is the left most dimension size,
     * and the last element is the total number of elements in the array.
     */
    using LayerSizesSequence = std::conditional_t<
        IsRowMajor_,
        typename array_helper::PrefixMultiplySequence<
            typename array_helper::ReverseSequence<DimensionSizesSequence>::type>::type,
        typename array_helper::PrefixMultiplySequence<DimensionSizesSequence>::type>;

    static constexpr RSize Dimension = sizeof...(DimensionSizes);
    static constexpr bool IsRowMajor = IsRowMajor_;
    static constexpr RSize ElemNum = array_helper::Multiply<DimensionSizesSequence>::value;

    /*****************************************************************
     * Data
     *****************************************************************/
    T data[ElemNum];

    /*****************************************************************
     * Methods
     *****************************************************************/
    /// @brief Assignment operator.
    RS_StaticArray& operator=(const RS_StaticArray&) = default;

    HOST_DEVICE_INLINE_QUALIFIER constexpr T* ptr() { return data; }

    /// @brief Return the total number of used elements in this array.
    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize size() const { return ElemNum; }

    /// @brief Return true if this array is empty.
    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return ElemNum == 0; }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const { return 0; }

    /**
     * @brief Checks if the given coordinates are out of bounds.
     *
     * @param coords Coordinates to check.
     * @return True if the coordinates are out of bounds, otherwise false.
     */
    HOST_DEVICE_INLINE_QUALIFIER bool isOutOfBound(
        const rs_device_std::array<RSize, Dimension>& coords
    ) {
        int is_out_of_bound = 0;
        device::loop<0, Dimension>([&](auto i_const) {
            constexpr RSize i = i_const.value;
            if constexpr (i < Dimension - 1) {
                constexpr RSize dimension_size =
                    array_helper::GetSequenceElementV<DimensionSizesSequence, i>;
                is_out_of_bound |= coords[i] >= dimension_size;
            }
        });
        return is_out_of_bound;
    }

    /**
     * @brief Converts N-dimensional coordinates to a linear index.
     *
     * @param coords Variable number of coordinate arguments.
     * @returns The linear index.
     */
    template <RSize D = sizeof...(DimensionSizes), typename... Coords>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<(D > 3 && sizeof...(Coords) == D), RIndex>
    coordsToIndex(Coords... coords) {
        RIndex indices[] = {static_cast<RIndex>(coords)...};
        RIndex index;

        if constexpr (IsRowMajor) {
            index = indices[D - 1];
            device::loop<0, D - 1>([&](auto i_const) {
                constexpr RSize i = i_const.value;
                if constexpr (i < D - 1) {
                    constexpr RSize layer_size =
                        array_helper::GetSequenceElementV<LayerSizesSequence, i>;
                    RS_StaticDivMod<layer_size>::muladd(index, indices[D - i - 2], index);
                }
            });
        } else {
            index = indices[0];
            device::loop<0, D>([&](auto i_const) {
                constexpr RSize i = i_const.value;
                if constexpr (i < D - 1) {
                    constexpr RSize layer_size =
                        array_helper::GetSequenceElementV<LayerSizesSequence, i>;
                    RS_StaticDivMod<layer_size>::muladd(index, indices[i + 1], index);
                }
            });
        }
        return index;
    }

    /**
     * @brief Converts a linear index to N-dimensional coordinates.
     *
     * @param index Linear index to convert.
     * @param coords References to store the N-dimensional coordinates.
     */
    template <RSize D = sizeof...(DimensionSizes), typename... Coords>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<(D > 3 && sizeof...(Coords) == D), void>
    indexToCoords(RIndex index, Coords&... coords) {
        RIndex* indices[] = {&coords...};
        RIndex remaining_index = index;
        if constexpr (IsRowMajor) {
            device::loop<0, D - 1>([&](auto i_const) {
                constexpr RSize i = i_const.value;
                if constexpr (i < D - 1) {
                    constexpr RSize layer_size =
                        array_helper::GetSequenceElementV<LayerSizesSequence, D - i - 2>;
                    RS_StaticDivMod<layer_size>::divmod(
                        remaining_index, *indices[i], remaining_index
                    );
                }
            });
            *indices[D - 1] = remaining_index;

        } else {
            device::loop<0, D>([&](auto i_const) {
                constexpr RSize i = i_const.value;
                if constexpr (i < D - 1) {
                    constexpr RSize layer_size =
                        array_helper::GetSequenceElementV<LayerSizesSequence, D - i - 2>;
                    RS_StaticDivMod<layer_size>::divmod(
                        remaining_index, *indices[D - i - 1], remaining_index
                    );
                }
            });
            *indices[0] = remaining_index;
        }
    }

    /**
     * @brief Converts 2D coordinates (x_0, x_1) to a linear index.
     *
     * @note For readability over general case.
     *
     * @param x_0 x_0-coordinate of first dimension.
     * @param x_1 x_1-coordinate of second dimension.
     * @returns The linear index.
     */
    template <RSize D = sizeof...(DimensionSizes)>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<D == 2, RIndex>
    coordsToIndex(RIndex x_0, RIndex x_1) {
        RIndex index;
        if constexpr (IsRowMajor) {
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 0>>::muladd(
                index, x_0, x_1
            );
        } else {
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 0>>::muladd(
                index, x_1, x_0
            );
        }
        return index;
    }

    /**
     * @brief Converts a linear index to 2D coordinates (x_0, x_1).
     *
     * @note For readability over general case.
     *
     * @param index Linear index to convert.
     * @param x_0 Reference to store the x_0-coordinate.
     * @param x_1 Reference to store the x_1-coordinate.
     */
    template <RSize D = sizeof...(DimensionSizes)>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<D == 2, void>
    indexToCoords(RIndex index, RIndex& x_0, RIndex& x_1) {
        if constexpr (IsRowMajor) {
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 0>>::divmod(
                index, x_0, x_1
            );
        } else {
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 0>>::divmod(
                index, x_1, x_0
            );
        }
    }

    /**
     * @brief Converts 3D coordinates (x_0, x_1, x_2) to a linear index.
     *
     * @param x_0 x_0-coordinate of first dimension.
     * @param x_1 x_1-coordinate of second dimension.
     * @param x_2 x_2-coordinate of third dimension.
     * @returns The linear index.
     */
    template <RSize D = sizeof...(DimensionSizes)>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<D == 3, RIndex>
    coordsToIndex(RIndex x_0, RIndex x_1, RIndex x_2) {
        RIndex index;
        if constexpr (IsRowMajor) {
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 0>>::muladd(
                index, x_1, x_2
            );
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 1>>::muladd(
                index, x_0, index
            );
        } else {
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 0>>::muladd(
                index, x_1, x_0
            );
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 1>>::muladd(
                index, x_2, index
            );
        }
        return index;
    }

    /**
     * @brief Converts a linear index to 3D coordinates (x_0, x_1, x_2).
     *
     * @param index Linear index to convert.
     * @param x_0 Reference to store the x_0-coordinate.
     * @param x_1 Reference to store the x_1-coordinate.
     * @param x_2 Reference to store the x_2-coordinate.
     */
    template <RSize D = sizeof...(DimensionSizes)>
    HOST_DEVICE_INLINE_QUALIFIER std::enable_if_t<D == 3, void>
    indexToCoords(RIndex index, RIndex& x_0, RIndex& x_1, RIndex& x_2) {
        RIndex remaining_index;
        if constexpr (IsRowMajor) {
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 1>>::divmod(
                index, x_0, remaining_index
            );
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 0>>::divmod(
                remaining_index, x_1, x_2
            );
        } else {
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 1>>::divmod(
                index, x_2, remaining_index
            );
            RS_StaticDivMod<array_helper::GetSequenceElementV<LayerSizesSequence, 0>>::divmod(
                remaining_index, x_1, x_0
            );
        }
    }

    /**
     * @brief Access operator for the array.
     * @param index Index of the element to access.
     * @return Reference to the element at the given index.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& operator[](RIndex index) { return data[index]; }

    /**
     * @brief Access operator for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @return Reference to the element at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& operator[](
        const rs_device_std::array<RSize, Dimension>& coords
    ) {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        return data[index];
    }

    HOST_DEVICE_INLINE_QUALIFIER const T& operator[](RIndex index) const { return data[index]; }

    /**
     * @brief Const access operator for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @return Const reference to the element at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& operator[](
        const rs_device_std::array<RSize, Dimension>& coords
    ) const {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        return data[index];
    }

    /**
     * @brief Get method for the array.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& get(RIndex index) { return data[index]; }

    /**
     * @brief Get method for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @return Reference to the element at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& get(const rs_device_std::array<RSize, Dimension>& coords) {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        return data[index];
    }

    /**
     * @brief Const get method for the array.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& get(RIndex index) const { return data[index]; }

    /**
     * @brief Const get method for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @return Const reference to the element at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& get(
        const rs_device_std::array<RSize, Dimension>& coords
    ) const {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        return data[index];
    }

    /**
     * @brief Set method for the array.
     */
    HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, const T& value) { data[index] = value; }

    /**
     * @brief Set method for the array using multidimensional coordinates.
     * @param coords Array of coordinates.
     * @param value Value to set at the given coordinates.
     */
    HOST_DEVICE_INLINE_QUALIFIER void
    set(const rs_device_std::array<RSize, Dimension>& coords, const T& value) {
        RIndex index = _coordArrayToIndex(coords, rs_device_std::make_index_sequence<Dimension>{});
        data[index] = value;
    }

   private:
    template <size_t... Is>
    HOST_DEVICE_INLINE_QUALIFIER RIndex _coordArrayToIndex(
        const rs_device_std::array<RSize, Dimension>& coords,
        rs_device_std::index_sequence<Is...>
    ) {
        return coordsToIndex(coords[Is]...);
    }
};

template <typename T, RSize... DimensionSizes>
using RS_StaticRowMajorArray = RS_StaticArray<T, RS_StaticArraySizeConfig<DimensionSizes...>, true>;

template <typename T, RSize... DimensionSizes>
using RS_StaticColumnMajorArray =
    RS_StaticArray<T, RS_StaticArraySizeConfig<DimensionSizes...>, false>;

template <typename T, RSize... DimensionSizes>
using RS_StaticRowMajorTwoAlignedArray = RS_StaticArray<
    T,
    RS_StaticArraySizeConfig<array_helper::NextPowerOfTwo<DimensionSizes>::value...>,
    true>;

template <typename T, RSize... DimensionSizes>
using RS_StaticColumnMajorTwoAlignedArray = RS_StaticArray<
    T,
    RS_StaticArraySizeConfig<array_helper::NextPowerOfTwo<DimensionSizes>::value...>,
    false>;

}  // namespace device

}  // namespace rise

#endif  // RS_ARRAY_H
