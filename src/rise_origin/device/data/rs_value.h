// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VALUE_H
#define RS_VALUE_H

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/utils/rs_memory.h"
#include "rs_data_util.h"

namespace rise {

namespace device {

class RSCF_ValueFill : public RS_ComputeFunction<RSCF_ValueFill> {
   public:
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_MemoryPtr data, T v) {
        data.get<T>(0) = v;
    }
};

template <typename T>
class RS_Value {
   public:
    using ObjectType = T;

    /*****************************************************************
     * Data
     *****************************************************************/
    RS_MemoryPtr data;  ///< Pointer to the data stored on the device.

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_Value() = default;

    /**
     * @brief Construct a new RS_Value object with the given provider.
     */
    explicit RS_Value(RS_MemoryProvider* provider) { init(provider); }

    /**
     * @brief Construct a new RS_Value object with the given value and provider.
     */
    RS_Value(const T& value, RS_MemoryProvider* provider) { init(value, provider); }

    /**
     * @brief Initialize the RS_Value object with the default value and the given provider.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);
        data = provider->allocate(sizeof(T), false);
        fill(T());
    }

    /**
     * @brief Initialize the RS_Value object with the given value and provider.
     */
    void init(const T& value, RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);
        data = provider->allocate(sizeof(T), false);
        fill(value);
    }

    /**
     * @brief Free the memory of the RS_Value object.
     * @note This function is no-op when the value is un-initialized with memory provider.
     */
    void free() { data.free(); }

    /**
     * @brief Copy the value from another RS_Value object.
     * @note This function is no-op when the value is un-initialized with memory provider.
     * Or if the source value is un-initialized with memory provider.
     */
    void copy(const RS_Value& value) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(value.data.provider() != nullptr);
        data.copyFrom(value.data, data.size());
    }

    /**
     * @brief Move the value to a new provider
     * @note If original value is un-initialized with provider, the new value will be allocated
     * with the provider, but value will be undefined.
     */
    RS_Value move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_Value new_value(provider);
        new_value.data.copyFrom(data, data.size());
        return new_value;
    }

    /**
     * @brief Fill the value with the given value.
     * @param value Value to fill.
     * @param sync Whether to sync the operation.
     */
    void fill(const T& value, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        castProviderPtrAndCall(data.provider(), [&](auto p_ptr) {
            p_ptr->template run<RSCF_ValueFill>({1, 1, sync}, data, value);
        });
    }

    /**
     * @brief Write the value from the given pointer.
     * @param input Pointer to the value to write.
     * @param sync Whether to sync the operation.
     */
    void write(T* input, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_POINTER_NOT_NULL(input);
        data.copyFrom(RS_ConstMemoryPtr(input, sizeof(T)), sizeof(T), sync);
    }

    /**
     * @brief Read the value to the given pointer.
     * @param output Pointer to the value to read.
     * @param sync Whether to sync the operation.
     */
    void read(T* output, bool sync = true) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_POINTER_NOT_NULL(output);
        data.copyTo(RS_MemoryPtr(output, sizeof(T)), sizeof(T), sync);
    }

    /// @brief Returns the memory provider.
    [[nodiscard]] RS_MemoryProvider* provider() const { return data.provider(); }

    /// @brief Returns the device location.
    [[nodiscard]] RS_DeviceLocation device() const { return data.device(); }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const { return data.size(); }

    /// @brief Returns the value.
    HOST_DEVICE_INLINE_QUALIFIER T& get() const { return data.get<T>(0); }

    /// @brief Sets the value.
    HOST_DEVICE_INLINE_QUALIFIER void set(T value) { data.get<T>(0) = value; }

    /**
     * @brief Sets the value from another type. The type T must be constructible
     * from the type T2.
     * @tparam T2 Type of the value to set.
     * @param value Value to set.
     */
    template <typename T2>
    HOST_DEVICE_INLINE_QUALIFIER void set(T2 value) {
        static_assert(
            std::is_constructible<T, T2>::value,
            "To use set with T2, Type T must be constructible from type T2."
        );
        data.get<T>(0) = T(value);
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_VALUE_H
