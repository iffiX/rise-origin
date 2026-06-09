// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SORT_H
#define RS_SORT_H

#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace device {

class RS_SortPairs {
   public:
    RS_MemoryPtr sort_buffer{};
    RS_MemoryProvider* mem_provider = nullptr;

    RS_SortPairs() = default;

    explicit RS_SortPairs(RS_MemoryProvider* provider) { init(provider); }

    void init(RS_MemoryProvider* provider) {
        sort_buffer = provider->allocate(0, true);
        mem_provider = provider;
    }

    void free() {
        sort_buffer.free();
        mem_provider = nullptr;
    }

    /**
     * @brief Sort pairs using merge sort.
     * @tparam KeyType
     * @tparam ValueType
     * @tparam CompareOp
     * @param in_keys Input device memory to be sorted of size element number, has to be on the same
     * device as where the sorter is initialized.
     * @param in_values Input device memory to be sorted of size element number, has to be on the
     * same device as where the sorter is initialized.
     * @param out_keys Output device memory of size element number, result of sorting will be
     * written here.
     * @param out_values Output device memory of size element number, result of sorting will be
     * written here.
     * @param in_len Input element number.
     * @param compare_op The comparison operator that models the Strict Weak Ordering concept.
     */
    template <typename KeyType, typename ValueType, typename CompareOp>
    void sort(
        KeyType* in_keys,
        ValueType* in_values,
        KeyType* out_keys,
        ValueType* out_values,
        RSize in_len,
        CompareOp compare_op
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        sort<KeyType, ValueType>(
            RS_ConstMemoryPtr(in_keys, sizeof(KeyType) * in_len, mem_provider->device, nullptr),
            RS_ConstMemoryPtr(in_values, sizeof(ValueType) * in_len, mem_provider->device, nullptr),
            RS_MemoryPtr(out_keys, sizeof(KeyType) * in_len, mem_provider->device, nullptr),
            RS_MemoryPtr(out_values, sizeof(ValueType) * in_len, mem_provider->device, nullptr),
            in_len,
            compare_op
        );
    }

    /**
     * @brief Sort pairs using merge sort.
     * @tparam KeyType
     * @tparam ValueType
     * @tparam CompareOp
     * @param in_keys Input device array to be sorted of size element number, has to be on the same
     * device as where the sorter is initialized.
     * @param in_values Input device array to be sorted of size element number, has to be on the
     * same device as where the sorter is initialized.
     * @param out_keys Output device array of size element number, result of sorting will be
     * written here.
     * @param out_values Output device array of size element number, result of sorting will be
     * written here.
     * @param compare_op The comparison operator that models the Strict Weak Ordering concept.
     */
    template <typename KeyType, typename ValueType, typename CompareOp>
    void sort(
        RS_Array<KeyType, 1>& in_keys,
        RS_Array<ValueType, 1>& in_values,
        RS_Array<KeyType, 1>& out_keys,
        RS_Array<ValueType, 1>& out_values,
        CompareOp compare_op
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.device() == device() && in_values.device() == device() &&
                out_keys.device() == device() && out_values.device() == device(),
            "All input and output arrays must be on the same device as the sorter."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.size() == in_values.size() && in_keys.size() == out_keys.size() &&
                out_keys.size() == out_values.size(),
            "Sizes must match for in_keys, in_values, out_keys, out_values."
        );
        sort<KeyType, ValueType>(
            in_keys.data, in_values.data, out_keys.data, out_values.data, in_keys.size(), compare_op
        );
    }

    /**
     * @brief Sort pairs using merge sort.
     * @tparam KeyType
     * @tparam ValueType
     * @tparam CompareOp
     * @param in_keys Input device vector to be sorted of size element number, has to be on the same
     * device as where the sorter is initialized.
     * @param in_values Input device vector to be sorted of size element number, has to be on the
     * same device as where the sorter is initialized.
     * @param out_keys Output device vector of size element number, result of sorting will be
     * written here.
     * @param out_values Output device vector of size element number, result of sorting will be
     * written here.
     * @param compare_op The comparison operator that models the Strict Weak Ordering concept.
     */
    template <typename KeyType, typename ValueType, typename CompareOp>
    void sort(
        RS_Vector<KeyType>& in_keys,
        RS_Vector<ValueType>& in_values,
        RS_Vector<KeyType>& out_keys,
        RS_Vector<ValueType>& out_values,
        CompareOp compare_op
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.device() == device() && in_values.device() == device() &&
                out_keys.device() == device() && out_values.device() == device(),
            "All input and output vectors must be on the same device as the sorter."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.size() == in_values.size() && in_keys.size() == out_keys.size() &&
                out_keys.size() == out_values.size(),
            "Sizes must match for in_keys, in_values, out_keys, out_values."
        );
        sort<KeyType, ValueType>(
            in_keys.data, in_values.data, out_keys.data, out_values.data, in_keys.size(), compare_op
        );
        out_keys.occupy(in_keys.size());
        out_values.occupy(in_values.size());
    }

    /**
     * @brief Sort pairs using merge sort.
     * @tparam KeyType
     * @tparam ValueType
     * @tparam CompareOp
     * @param in_keys Input device memory to be sorted of size element number, has to be on the same
     * device as where the sorter is initialized.
     * @param in_values Input device memory to be sorted of size element number, has to be on the
     * same device as where the sorter is initialized.
     * @param out_keys Output device memory of size element number, result of sorting will be
     * written here.
     * @param out_values Output device memory of size element number, result of sorting will be
     * written here.
     * @param in_len Input element number.
     * @param compare_op The comparison operator that models the Strict Weak Ordering concept.
     */
    template <typename KeyType, typename ValueType, typename CompareOp, typename InMemoryPtr>
    std::enable_if_t<
        std::is_same_v<InMemoryPtr, RS_MemoryPtr> || std::is_same_v<InMemoryPtr, RS_ConstMemoryPtr>,
        void>
    sort(
        InMemoryPtr in_keys,
        InMemoryPtr in_values,
        RS_MemoryPtr out_keys,
        RS_MemoryPtr out_values,
        RSize in_len,
        CompareOp compare_op
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.device() == device() && in_values.device() == device() &&
                out_keys.device() == device() && out_values.device() == device(),
            "All input and output memories must be on the same device as the sorter."
        );
        auto* in_ptr_keys = reinterpret_cast<KeyType*>(in_keys.ptr());
        auto* in_ptr_values = reinterpret_cast<ValueType*>(in_values.ptr());
        auto* out_ptr_keys = out_keys.as<KeyType>();
        auto* out_ptr_values = out_values.as<ValueType>();

        if (mem_provider->device.type == RSE_DEVICE_TYPE_CPU) {
            // Because sort is not executed by memory provider, sync the memory provider first
            mem_provider->sync();

            std::vector<RIndex> indices(in_len);
            std::iota(indices.begin(), indices.end(), 0);

            std::sort(
                std::execution::par_unseq,
                indices.begin(),
                indices.end(),
                [&](RIndex lhs, RIndex rhs) {
                    return compare_op(in_ptr_keys[lhs], in_ptr_keys[rhs]);
                }
            );

            for (RIndex i = 0; i < in_len; i++) {
                RIndex src_idx = indices[i];
                out_ptr_keys[i] = in_ptr_keys[src_idx];
                out_ptr_values[i] = in_ptr_values[src_idx];
            }
        }
#ifdef __CUDACC__
        else if (mem_provider->device.type == RSE_DEVICE_TYPE_CUDA) {
            size_t temp_size = 0;
            cub::DeviceMergeSort::SortPairsCopy(
                static_cast<void*>(nullptr),
                temp_size,
                in_ptr_keys,
                out_ptr_keys,
                in_ptr_values,
                out_ptr_values,
                static_cast<int>(in_len),
                compare_op
            );
            if (sort_buffer.size() < temp_size) {
                sort_buffer.free();
                sort_buffer = mem_provider->allocate(temp_size, true);
            }
            castProviderPtrAndCall(mem_provider, [&](auto* p_ptr) {
                if constexpr (std::is_same_v<RS_MemoryCudaProvider*, decltype(p_ptr)>) {
                    cub::DeviceMergeSort::SortPairsCopy(
                        sort_buffer.ptr(),
                        temp_size,
                        in_ptr_keys,
                        out_ptr_keys,
                        in_ptr_values,
                        out_ptr_values,
                        static_cast<int>(in_len),
                        compare_op,
                        p_ptr->stream
                    );
                }
            });

        }
#endif
        else
            RS_THROW(RS_InvalidConfigurationException(
                "Device type {} not supported", enumToString(mem_provider->device.type)
            ));
    }

    [[nodiscard]] RS_DeviceLocation device() const {
        return mem_provider ? mem_provider->device
                            : RS_DeviceLocation(RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX);
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return mem_provider; }
};

class RS_RadixSortPairs {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    RS_MemoryPtr sort_buffer{};
    RS_MemoryProvider* mem_provider = nullptr;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_RadixSortPairs() = default;

    explicit RS_RadixSortPairs(RS_MemoryProvider* provider) { init(provider); }

    void init(RS_MemoryProvider* provider) {
        sort_buffer = provider->allocate(0, true);
        mem_provider = provider;
    }

    void free() {
        sort_buffer.free();
        mem_provider = nullptr;
    }

    /**
     * @brief Sort pairs using radix sort.
     * @tparam KeyType
     * @tparam ValueType
     * @param in_keys Input device memory to be sorted of size element number, has to be on the same
     * device as where the sorter is initialized.
     * @param in_values Input device memory to be sorted of size element number, has to be on the
     * same device as where the sorter is initialized.
     * @param out_keys Output device memory of size element number, result of sorting will be
     * written here.
     * @param out_values Output device memory of size element number, result of sorting will be
     * written here.
     * @param in_len Input element number.
     * @param is_ascending Whether to sort in ascending order.
     */
    template <typename KeyType, typename ValueType>
    void sort(
        KeyType* in_keys,
        ValueType* in_values,
        KeyType* out_keys,
        ValueType* out_values,
        RSize in_len,
        bool is_ascending = true
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        sort<KeyType, ValueType>(
            RS_ConstMemoryPtr(in_keys, sizeof(KeyType) * in_len, mem_provider->device, nullptr),
            RS_ConstMemoryPtr(in_values, sizeof(ValueType) * in_len, mem_provider->device, nullptr),
            RS_MemoryPtr(out_keys, sizeof(KeyType) * in_len, mem_provider->device, nullptr),
            RS_MemoryPtr(out_values, sizeof(ValueType) * in_len, mem_provider->device, nullptr),
            in_len,
            is_ascending
        );
    }

    /**
     * @brief Sort pairs using radix sort.
     * @tparam KeyType
     * @tparam ValueType
     * @param in_keys Input device array to be sorted of size element number, has to be on the same
     * device as where the sorter is initialized.
     * @param in_values Input device array to be sorted of size element number, has to be on the
     * same device as where the sorter is initialized.
     * @param out_keys Output device array of size element number, result of sorting will be
     * written here.
     * @param out_values Output device array of size element number, result of sorting will be
     * written here.
     * @param is_ascending Whether to sort in ascending order.
     */
    template <typename KeyType, typename ValueType>
    void sort(
        RS_Array<KeyType, 1>& in_keys,
        RS_Array<ValueType, 1>& in_values,
        RS_Array<KeyType, 1>& out_keys,
        RS_Array<ValueType, 1>& out_values,
        bool is_ascending = true
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.device() == device() && in_values.device() == device() &&
                out_keys.device() == device() && out_values.device() == device(),
            "All input and output arrays must be on the same device as the sorter."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.size() == in_values.size() && in_keys.size() == out_keys.size() &&
                out_keys.size() == out_values.size(),
            "Sizes must match for in_keys, in_values, out_keys, out_values."
        );
        sort<KeyType, ValueType>(
            in_keys.data,
            in_values.data,
            out_keys.data,
            out_values.data,
            in_keys.size(),
            is_ascending
        );
    }

    /**
     * @brief Sort pairs using radix sort.
     * @tparam KeyType
     * @tparam ValueType
     * @param in_keys Input device vector to be sorted of size element number, has to be on the same
     * device as where the sorter is initialized.
     * @param in_values Input device vector to be sorted of size element number, has to be on the
     * same device as where the sorter is initialized.
     * @param out_keys Output device vector of size element number, result of sorting will be
     * written here.
     * @param out_values Output device vector of size element number, result of sorting will be
     * written here.
     * @param is_ascending Whether to sort in ascending order.
     */
    template <typename KeyType, typename ValueType>
    void sort(
        RS_Vector<KeyType>& in_keys,
        RS_Vector<ValueType>& in_values,
        RS_Vector<KeyType>& out_keys,
        RS_Vector<ValueType>& out_values,
        bool is_ascending = true
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.device() == device() && in_values.device() == device() &&
                out_keys.device() == device() && out_values.device() == device(),
            "All input and output vectors must be on the same device as the sorter."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.size() == in_values.size() && in_keys.size() == out_keys.size() &&
                out_keys.size() == out_values.size(),
            "Sizes must match for in_keys, in_values, out_keys, out_values."
        );
        sort<KeyType, ValueType>(
            in_keys.data,
            in_values.data,
            out_keys.data,
            out_values.data,
            in_keys.size(),
            is_ascending
        );
        out_keys.occupy(in_keys.size());
        out_values.occupy(in_values.size());
    }

    /**
     * @brief Sort pairs using radix sort.
     * @tparam KeyType
     * @tparam ValueType
     * @tparam InMemoryPtr
     * @param in_keys Input device memory to be sorted of size element number, has to be on the same
     * device as where the sorter is initialized.
     * @param in_values Input device memory to be sorted of size element number, has to be on the
     * same device as where the sorter is initialized.
     * @param out_keys Output device memory of size element number, result of sorting will be
     * written here.
     * @param out_values Output device memory of size element number, result of sorting will be
     * written here.
     * @param in_len Input element number.
     * @param is_ascending Whether to sort in ascending order.
     */
    template <typename KeyType, typename ValueType, typename InMemoryPtr>
    std::enable_if_t<
        std::is_same_v<InMemoryPtr, RS_MemoryPtr> || std::is_same_v<InMemoryPtr, RS_ConstMemoryPtr>,
        void>
    sort(
        InMemoryPtr in_keys,
        InMemoryPtr in_values,
        RS_MemoryPtr out_keys,
        RS_MemoryPtr out_values,
        RSize in_len,
        bool is_ascending = true
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_keys.device() == device() && in_values.device() == device() &&
                out_keys.device() == device() && out_values.device() == device(),
            "All input and output memories must be on the same device as the sorter."
        );

        auto* in_ptr_keys = reinterpret_cast<KeyType*>(in_keys.ptr());
        auto* in_ptr_values = reinterpret_cast<ValueType*>(in_values.ptr());
        auto* out_ptr_keys = out_keys.as<KeyType>();
        auto* out_ptr_values = out_values.as<ValueType>();

        if (mem_provider->device.type == RSE_DEVICE_TYPE_CPU) {
            // Because sort is not executed by memory provider, sync the memory provider first
            mem_provider->sync();

            std::vector<RIndex> indices(in_len);
            std::iota(indices.begin(), indices.end(), 0);

            if (is_ascending) {
                std::sort(
                    std::execution::par_unseq,
                    indices.begin(),
                    indices.end(),
                    [&](RIndex lhs, RIndex rhs) { return in_ptr_keys[lhs] < in_ptr_keys[rhs]; }
                );
            } else {
                std::sort(
                    std::execution::par_unseq,
                    indices.begin(),
                    indices.end(),
                    [&](RIndex lhs, RIndex rhs) { return in_ptr_keys[lhs] > in_ptr_keys[rhs]; }
                );
            }

            for (RIndex i = 0; i < in_len; i++) {
                RIndex src_idx = indices[i];
                out_ptr_keys[i] = in_ptr_keys[src_idx];
                out_ptr_values[i] = in_ptr_values[src_idx];
            }
        }
#ifdef __CUDACC__
        else if (mem_provider->device.type == RSE_DEVICE_TYPE_CUDA) {
            if (is_ascending) {
                // ascending sort
                // determine needed memory
                size_t temp_size = 0;
                cub::DeviceRadixSort::SortPairs(
                    static_cast<void*>(nullptr),
                    temp_size,
                    in_ptr_keys,
                    out_ptr_keys,
                    in_ptr_values,
                    out_ptr_values,
                    in_len
                );

                if (sort_buffer.size() < temp_size) {
                    sort_buffer.free();
                    sort_buffer = mem_provider->allocate(temp_size, true);
                }

                castProviderPtrAndCall(mem_provider, [&](auto* p_ptr) {
                    if constexpr (std::is_same_v<RS_MemoryCudaProvider*, decltype(p_ptr)>) {
                        cub::DeviceRadixSort::SortPairs(
                            sort_buffer.ptr(),
                            temp_size,
                            in_ptr_keys,
                            out_ptr_keys,
                            in_ptr_values,
                            out_ptr_values,
                            in_len,
                            0,
                            sizeof(KeyType) * 8,
                            p_ptr->stream
                        );
                    }
                });
            } else {
                // descending sort
                // determine needed memory
                size_t temp_size = 0;
                cub::DeviceRadixSort::SortPairsDescending(
                    nullptr,
                    temp_size,
                    in_ptr_keys,
                    out_ptr_keys,
                    in_ptr_values,
                    out_ptr_values,
                    in_len
                );

                if (sort_buffer.size() < temp_size) {
                    sort_buffer.free();
                    sort_buffer = mem_provider->allocate(temp_size, true);
                }
                castProviderPtrAndCall(mem_provider, [&](auto* p_ptr) {
                    if constexpr (std::is_same_v<RS_MemoryCudaProvider*, decltype(p_ptr)>) {
                        cub::DeviceRadixSort::SortPairsDescending(
                            sort_buffer.ptr(),
                            temp_size,
                            in_ptr_keys,
                            out_ptr_keys,
                            in_ptr_values,
                            out_ptr_values,
                            in_len,
                            0,
                            sizeof(KeyType) * 8,
                            p_ptr->stream
                        );
                    }
                });
            }
            return;
        }

#endif
        else
            RS_THROW(RS_InvalidConfigurationException(
                "Device type {} not supported", enumToString(mem_provider->device.type)
            ));
    }

    [[nodiscard]] RS_DeviceLocation device() const {
        return mem_provider ? mem_provider->device
                            : RS_DeviceLocation(RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX);
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return mem_provider; }
};

}  // namespace device

}  // namespace rise

#endif  // RS_SORT_H
