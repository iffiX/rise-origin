// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SCAN_H
#define RS_SCAN_H

#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_data_util.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace device {

class RS_Scan {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    RS_MemoryPtr scan_buffer{};
    RS_MemoryProvider* mem_provider = nullptr;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_Scan() = default;

    explicit RS_Scan(RS_MemoryProvider* provider) { init(provider); }

    void init(RS_MemoryProvider* provider) {
        scan_buffer = provider->allocate(0);
        mem_provider = provider;
    }

    void free() {
        scan_buffer.free();
        mem_provider = nullptr;
    }

    /*****************************************************************
     * ExclusiveSum
     *****************************************************************/

    /**
     * @brief Compute exclusive prefix sum (scan) with 0 as the initial value.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @param in_data Input device memory to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device memory of size element number, result of scan will be written
     * here.
     * @param in_len Input element number.
     */
    template <typename InputType, typename OutputType>
    void exclusiveSum(const InputType* in_data, OutputType* out_data, RSize in_len) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        exclusiveSum<InputType, OutputType>(
            RS_ConstMemoryPtr(
                in_data, sizeof(InputType) * in_len, mem_provider->device, mem_provider
            ),
            RS_MemoryPtr(
                out_data, sizeof(OutputType) * in_len, 0, mem_provider->device, mem_provider
            ),
            in_len
        );
    }

    /**
     * @brief Compute exclusive prefix sum (scan) with 0 as the initial value.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @param in_data Input device array to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device array of size element number, result of scan will be written
     * here.
     */
    template <typename InputType, typename OutputType>
    void exclusiveSum(const RS_Array<InputType, 1>& in_data, RS_Array<OutputType, 1>& out_data) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output arrays must be on the same device as the scanner."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_data.size() == out_data.size(), "Sizes must match for in_data and out_data."
        );
        exclusiveSum<InputType, OutputType>(in_data.data, out_data.data, in_data.size());
    }

    /**
     * @brief Compute exclusive prefix sum (scan) with 0 as the initial value.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @param in_data Input device vector to be scanned, of size used_elem_num, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device vector of size used_elem_num, result of scan will be written
     * here.
     */
    template <typename InputType, typename OutputType>
    void exclusiveSum(const RS_Vector<InputType>& in_data, RS_Vector<OutputType>& out_data) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output vectors must be on the same device as the scanner."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_data.size() == out_data.size(), "Sizes must match for in_data and out_data."
        );
        exclusiveSum<InputType, OutputType>(in_data.data, out_data.data, in_data.size());
    }

    /**
     * @brief Compute exclusive prefix sum (scan) with 0 as the initial value.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam InMemoryPtr Input memory pointer type
     * @param in_data Input device memory to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device memory of size element number, result of scan will be written
     * here.
     * @param in_len Input element number.
     */
    template <typename InputType, typename OutputType, typename InMemoryPtr>
    std::enable_if_t<
        std::is_same_v<InMemoryPtr, RS_MemoryPtr> || std::is_same_v<InMemoryPtr, RS_ConstMemoryPtr>,
        void>
    exclusiveSum(InMemoryPtr in_data, RS_MemoryPtr out_data, RSize in_len) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output memories must be on the same device as the scanner."
        );
        if (in_len == 0) return;

        const auto* in_ptr = in_data.template as<InputType>();
        auto* out_ptr = out_data.template as<OutputType>();

        if (mem_provider->device.type == RSE_DEVICE_TYPE_CPU) {
            // Scan is not executed by the memory provider — sync first
            mem_provider->sync();

            // Compute exclusive sum with 0 as initial value
            OutputType sum = static_cast<OutputType>(0);
            for (RIndex i = 0; i < in_len; i++) {
                out_ptr[i] = sum;
                sum += static_cast<OutputType>(in_ptr[i]);
            }
            return;
        }
#ifdef __CUDACC__
        if (mem_provider->device.type == RSE_DEVICE_TYPE_CUDA) {
            size_t temp_size = 0;
            cub::DeviceScan::ExclusiveSum(
                static_cast<void*>(nullptr), temp_size, in_ptr, out_ptr, in_len
            );
            _ensureScanBuffer(temp_size);
            castProviderPtrAndCall(mem_provider, [&](auto* p_ptr) {
                if constexpr (std::is_same_v<RS_MemoryCudaProvider*, decltype(p_ptr)>) {
                    cub::DeviceScan::ExclusiveSum(
                        scan_buffer.ptr(), temp_size, in_ptr, out_ptr, in_len, p_ptr->stream
                    );
                }
            });
            return;
        }
#endif
        RS_THROW(RS_InvalidCharacterizationException(
            "Device type {} not supported", enumToString(mem_provider->device.type)
        ));
    }

    /*****************************************************************
     * InclusiveSum
     *****************************************************************/

    /**
     * @brief Compute inclusive prefix sum (scan).
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @param in_data Input device memory to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device memory of size element number, result of scan will be written
     * here.
     * @param in_len Input element number.
     */
    template <typename InputType, typename OutputType>
    void inclusiveSum(const InputType* in_data, OutputType* out_data, RSize in_len) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        inclusiveSum<InputType, OutputType>(
            RS_ConstMemoryPtr(
                in_data, sizeof(InputType) * in_len, mem_provider->device, mem_provider
            ),
            RS_MemoryPtr(
                out_data, sizeof(OutputType) * in_len, 0, mem_provider->device, mem_provider
            ),
            in_len
        );
    }

    /**
     * @brief Compute inclusive prefix sum (scan).
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @param in_data Input device array to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device array of size element number, result of scan will be written
     * here.
     */
    template <typename InputType, typename OutputType>
    void inclusiveSum(const RS_Array<InputType, 1>& in_data, RS_Array<OutputType, 1>& out_data) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output arrays must be on the same device as the scanner."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_data.size() == out_data.size(), "Sizes must match for in_data and out_data."
        );
        inclusiveSum<InputType, OutputType>(in_data.data, out_data.data, in_data.size());
    }

    /**
     * @brief Compute inclusive prefix sum (scan).
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @param in_data Input device vector to be scanned, of size used_elem_num, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device vector of size used_elem_num, result of scan will be written
     * here.
     */
    template <typename InputType, typename OutputType>
    void inclusiveSum(const RS_Vector<InputType>& in_data, RS_Vector<OutputType>& out_data) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output vectors must be on the same device as the scanner."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_data.size() == out_data.size(), "Sizes must match for in_data and out_data."
        );
        inclusiveSum<InputType, OutputType>(in_data.data, out_data.data, in_data.size());
    }

    /**
     * @brief Compute inclusive prefix sum (scan).
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam InMemoryPtr Input memory pointer type
     * @param in_data Input device memory to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device memory of size element number, result of scan will be written
     * here.
     * @param in_len Input element number.
     */
    template <typename InputType, typename OutputType, typename InMemoryPtr>
    std::enable_if_t<
        std::is_same_v<InMemoryPtr, RS_MemoryPtr> || std::is_same_v<InMemoryPtr, RS_ConstMemoryPtr>,
        void>
    inclusiveSum(InMemoryPtr in_data, RS_MemoryPtr out_data, RSize in_len) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output memories must be on the same device as the scanner."
        );
        if (in_len == 0) return;

        const auto* in_ptr = in_data.template as<InputType>();
        auto* out_ptr = out_data.template as<OutputType>();

        if (mem_provider->device.type == RSE_DEVICE_TYPE_CPU) {
            // Scan is not executed by the memory provider — sync first
            mem_provider->sync();

            // Compute inclusive sum
            OutputType sum = static_cast<OutputType>(0);
            for (RIndex i = 0; i < in_len; i++) {
                sum += static_cast<OutputType>(in_ptr[i]);
                out_ptr[i] = sum;
            }
            return;
        }
#ifdef __CUDACC__
        if (mem_provider->device.type == RSE_DEVICE_TYPE_CUDA) {
            size_t temp_size = 0;
            cub::DeviceScan::InclusiveSum(
                static_cast<void*>(nullptr), temp_size, in_ptr, out_ptr, in_len
            );
            _ensureScanBuffer(temp_size);
            castProviderPtrAndCall(mem_provider, [&](auto* p_ptr) {
                if constexpr (std::is_same_v<RS_MemoryCudaProvider*, decltype(p_ptr)>) {
                    cub::DeviceScan::InclusiveSum(
                        scan_buffer.ptr(), temp_size, in_ptr, out_ptr, in_len, p_ptr->stream
                    );
                }
            });
            return;
        }
#endif
        RS_THROW(RS_InvalidCharacterizationException(
            "Device type {} not supported", enumToString(mem_provider->device.type)
        ));
    }

    /*****************************************************************
     * ExclusiveScan
     *****************************************************************/

    /**
     * @brief Compute exclusive prefix scan with a custom scan operator and initial value.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam ScanOp Scan operator type
     * @tparam InitValueType Initial value type
     * @param in_data Input device memory to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device memory of size element number, result of scan will be written
     * here.
     * @param in_len Input element number.
     * @param scan_op Binary scan operator.
     * @param init_value Initial value for exclusive scan.
     */
    template <typename InputType, typename OutputType, typename ScanOp, typename InitValueType>
    void exclusiveScan(
        const InputType* in_data,
        OutputType* out_data,
        RSize in_len,
        ScanOp scan_op,
        InitValueType init_value
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        exclusiveScan<InputType, OutputType>(
            RS_ConstMemoryPtr(
                in_data, sizeof(InputType) * in_len, mem_provider->device, mem_provider
            ),
            RS_MemoryPtr(
                out_data, sizeof(OutputType) * in_len, 0, mem_provider->device, mem_provider
            ),
            in_len,
            scan_op,
            init_value
        );
    }

    /**
     * @brief Compute exclusive prefix scan with a custom scan operator and initial value.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam ScanOp Scan operator type
     * @tparam InitValueType Initial value type
     * @param in_data Input device array to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device array of size element number, result of scan will be written
     * here.
     * @param scan_op Binary scan operator.
     * @param init_value Initial value for exclusive scan.
     */
    template <typename InputType, typename OutputType, typename ScanOp, typename InitValueType>
    void exclusiveScan(
        const RS_Array<InputType, 1>& in_data,
        RS_Array<OutputType, 1>& out_data,
        ScanOp scan_op,
        InitValueType init_value
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output arrays must be on the same device as the scanner."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_data.size() == out_data.size(), "Sizes must match for in_data and out_data."
        );
        exclusiveScan<InputType, OutputType>(
            in_data.data, out_data.data, in_data.size(), scan_op, init_value
        );
    }

    /**
     * @brief Compute exclusive prefix scan with a custom scan operator and initial value.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam ScanOp Scan operator type
     * @tparam InitValueType Initial value type
     * @param in_data Input device vector to be scanned, of size used_elem_num, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device vector of size used_elem_num, result of scan will be written
     * here.
     * @param scan_op Binary scan operator.
     * @param init_value Initial value for exclusive scan.
     */
    template <typename InputType, typename OutputType, typename ScanOp, typename InitValueType>
    void exclusiveScan(
        const RS_Vector<InputType>& in_data,
        RS_Vector<OutputType>& out_data,
        ScanOp scan_op,
        InitValueType init_value
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output vectors must be on the same device as the scanner."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_data.size() == out_data.size(), "Sizes must match for in_data and out_data."
        );
        exclusiveScan<InputType, OutputType>(
            in_data.data, out_data.data, in_data.size(), scan_op, init_value
        );
    }

    /**
     * @brief Compute exclusive prefix scan with a custom scan operator and initial value.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam InMemoryPtr Input memory pointer type
     * @tparam ScanOp Scan operator type
     * @tparam InitValueType Initial value type
     * @param in_data Input device memory to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device memory of size element number, result of scan will be written
     * here.
     * @param in_len Input element number.
     * @param scan_op Binary scan operator.
     * @param init_value Initial value for exclusive scan.
     */
    template <
        typename InputType,
        typename OutputType,
        typename InMemoryPtr,
        typename ScanOp,
        typename InitValueType>
    std::enable_if_t<
        std::is_same_v<InMemoryPtr, RS_MemoryPtr> || std::is_same_v<InMemoryPtr, RS_ConstMemoryPtr>,
        void>
    exclusiveScan(
        InMemoryPtr in_data,
        RS_MemoryPtr out_data,
        RSize in_len,
        ScanOp scan_op,
        InitValueType init_value
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output memories must be on the same device as the scanner."
        );
        if (in_len == 0) return;

        const auto* in_ptr = in_data.template as<InputType>();
        auto* out_ptr = out_data.template as<OutputType>();

        if (mem_provider->device.type == RSE_DEVICE_TYPE_CPU) {
            // Scan is not executed by the memory provider — sync first
            mem_provider->sync();

            // Compute exclusive scan with custom operator and initial value
            OutputType accumulated = static_cast<OutputType>(init_value);
            for (RIndex i = 0; i < in_len; i++) {
                out_ptr[i] = accumulated;
                accumulated = scan_op(accumulated, static_cast<OutputType>(in_ptr[i]));
            }
            return;
        }
#ifdef __CUDACC__
        if (mem_provider->device.type == RSE_DEVICE_TYPE_CUDA) {
            size_t temp_size = 0;
            cub::DeviceScan::ExclusiveScan(
                static_cast<void*>(nullptr),
                temp_size,
                in_ptr,
                out_ptr,
                scan_op,
                init_value,
                in_len
            );
            _ensureScanBuffer(temp_size);
            castProviderPtrAndCall(mem_provider, [&](auto* p_ptr) {
                if constexpr (std::is_same_v<RS_MemoryCudaProvider*, decltype(p_ptr)>) {
                    cub::DeviceScan::ExclusiveScan(
                        scan_buffer.ptr(),
                        temp_size,
                        in_ptr,
                        out_ptr,
                        scan_op,
                        init_value,
                        in_len,
                        p_ptr->stream
                    );
                }
            });
            return;
        }
#endif
        RS_THROW(RS_InvalidCharacterizationException(
            "Device type {} not supported", enumToString(mem_provider->device.type)
        ));
    }

    /*****************************************************************
     * InclusiveScan
     *****************************************************************/

    /**
     * @brief Compute inclusive prefix scan with a custom scan operator.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam ScanOp Scan operator type
     * @param in_data Input device memory to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device memory of size element number, result of scan will be written
     * here.
     * @param in_len Input element number.
     * @param scan_op Binary scan operator.
     */
    template <typename InputType, typename OutputType, typename ScanOp>
    void
    inclusiveScan(const InputType* in_data, OutputType* out_data, RSize in_len, ScanOp scan_op) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        inclusiveScan<InputType, OutputType>(
            RS_ConstMemoryPtr(
                in_data, sizeof(InputType) * in_len, mem_provider->device, mem_provider
            ),
            RS_MemoryPtr(
                out_data, sizeof(OutputType) * in_len, 0, mem_provider->device, mem_provider
            ),
            in_len,
            scan_op
        );
    }

    /**
     * @brief Compute inclusive prefix scan with a custom scan operator.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam ScanOp Scan operator type
     * @param in_data Input device array to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device array of size element number, result of scan will be written
     * here.
     * @param scan_op Binary scan operator.
     */
    template <typename InputType, typename OutputType, typename ScanOp>
    void inclusiveScan(
        const RS_Array<InputType, 1>& in_data,
        RS_Array<OutputType, 1>& out_data,
        ScanOp scan_op
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output arrays must be on the same device as the scanner."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_data.size() == out_data.size(), "Sizes must match for in_data and out_data."
        );
        inclusiveScan<InputType, OutputType>(in_data.data, out_data.data, in_data.size(), scan_op);
    }

    /**
     * @brief Compute inclusive prefix scan with a custom scan operator.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam ScanOp Scan operator type
     * @param in_data Input device vector to be scanned, of size used_elem_num, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device vector of size used_elem_num, result of scan will be written
     * here.
     * @param scan_op Binary scan operator.
     */
    template <typename InputType, typename OutputType, typename ScanOp>
    void inclusiveScan(
        const RS_Vector<InputType>& in_data,
        RS_Vector<OutputType>& out_data,
        ScanOp scan_op
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output vectors must be on the same device as the scanner."
        );
        RS_DATA_ASSERT_ARGUMENT(
            in_data.size() == out_data.size(), "Sizes must match for in_data and out_data."
        );
        inclusiveScan<InputType, OutputType>(
            in_data.data, out_data.data, in_data.size(), scan_op
        );
    }

    /**
     * @brief Compute inclusive prefix scan with a custom scan operator.
     * @tparam InputType Input data type
     * @tparam OutputType Output data type
     * @tparam InMemoryPtr Input memory pointer type
     * @tparam ScanOp Scan operator type
     * @param in_data Input device memory to be scanned of size element number, has to be on the
     * same device as where the scanner is initialized.
     * @param out_data Output device memory of size element number, result of scan will be written
     * here.
     * @param in_len Input element number.
     * @param scan_op Binary scan operator.
     */
    template <typename InputType, typename OutputType, typename InMemoryPtr, typename ScanOp>
    std::enable_if_t<
        std::is_same_v<InMemoryPtr, RS_MemoryPtr> || std::is_same_v<InMemoryPtr, RS_ConstMemoryPtr>,
        void>
    inclusiveScan(InMemoryPtr in_data, RS_MemoryPtr out_data, RSize in_len, ScanOp scan_op) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in_data.device() == device() && out_data.device() == device(),
            "All input and output memories must be on the same device as the scanner."
        );
        if (in_len == 0) return;

        const auto* in_ptr = in_data.template as<InputType>();
        auto* out_ptr = out_data.template as<OutputType>();

        if (mem_provider->device.type == RSE_DEVICE_TYPE_CPU) {
            // Scan is not executed by the memory provider — sync first
            mem_provider->sync();

            // Compute inclusive scan with custom operator
            out_ptr[0] = static_cast<OutputType>(in_ptr[0]);
            for (RIndex i = 1; i < in_len; i++) {
                out_ptr[i] = scan_op(out_ptr[i - 1], static_cast<OutputType>(in_ptr[i]));
            }
            return;
        }
#ifdef __CUDACC__
        if (mem_provider->device.type == RSE_DEVICE_TYPE_CUDA) {
            size_t temp_size = 0;
            cub::DeviceScan::InclusiveScan(
                static_cast<void*>(nullptr), temp_size, in_ptr, out_ptr, scan_op, in_len
            );
            _ensureScanBuffer(temp_size);
            castProviderPtrAndCall(mem_provider, [&](auto* p_ptr) {
                if constexpr (std::is_same_v<RS_MemoryCudaProvider*, decltype(p_ptr)>) {
                    cub::DeviceScan::InclusiveScan(
                        scan_buffer.ptr(),
                        temp_size,
                        in_ptr,
                        out_ptr,
                        scan_op,
                        in_len,
                        p_ptr->stream
                    );
                }
            });
            return;
        }
#endif
        RS_THROW(RS_InvalidCharacterizationException(
            "Device type {} not supported", enumToString(mem_provider->device.type)
        ));
    }

    [[nodiscard]] RS_DeviceLocation device() const {
        return mem_provider ? mem_provider->device
                            : RS_DeviceLocation(RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX);
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return mem_provider; }

   private:
    void _ensureScanBuffer(size_t needed) {
        if (scan_buffer.bytes >= needed) return;
        scan_buffer.free();
        scan_buffer = mem_provider->allocate(needed);
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_SCAN_H
