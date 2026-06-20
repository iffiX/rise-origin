// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_COPY_H
#define RS_COPY_H

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_soa.h"
#include "rise_origin/device/rs_compute.h"

namespace rise {

namespace device {

class RSCF_CopyFromSoAToArray1DLike : public RS_ComputeFunction<RSCF_CopyFromSoAToArray1DLike> {
   public:
    template <typename T, typename Array1DLike>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Array1DLike array, RS_StructOfArrays<T> soa, RSize array_offset = 0, RSize soa_offset = 0) {
        RIndex index = index1D();
        array[index + array_offset] = soa.get(index + soa_offset);
    }
};

class RSCF_CopyFromArray1DLikeToSoA : public RS_ComputeFunction<RSCF_CopyFromArray1DLikeToSoA> {
   public:
    template <typename T, typename Array1DLike>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_StructOfArrays<T> soa, Array1DLike array, RSize soa_offset = 0, RSize array_offset = 0) {
        RIndex index = index1D();
        soa.set(index + soa_offset, array[index + array_offset]);
    }
};

class RSCF_CopyFromSoAToSoA : public RS_ComputeFunction<RSCF_CopyFromSoAToSoA> {
   public:
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_StructOfArrays<T> dst_soa,
        RS_StructOfArrays<T> src_soa,
        RSize dst_soa_offset = 0,
        RSize src_soa_offset = 0
    ) {
        RIndex index = index1D();
        dst_soa.set(index + dst_soa_offset, src_soa.get(index + src_soa_offset));
    }
};

class RSCF_CopyFromArray1DLikeToArray1DLike
    : public RS_ComputeFunction<RSCF_CopyFromArray1DLikeToArray1DLike> {
   public:
    template <typename Array1DLike>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Array1DLike dst_array, Array1DLike src_array, RSize dst_offset = 0, RSize src_offset = 0) {
        RIndex index = index1D();
        dst_array[index + dst_offset] = src_array[index + src_offset];
    }
};

/**
 * @brief Copy a 2D array into a larger 2D array (for table resize with data preservation).
 *
 * @details Launched with src_dim * src_dim threads. Each thread copies one element
 *          from the source array to the corresponding position in the destination array.
 *          The destination array must be at least as large as the source in both dimensions.
 */
class RSCF_CopyFromArray2DToArray2D : public RS_ComputeFunction<RSCF_CopyFromArray2DToArray2D> {
   public:
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_Array<T, 2> dst, RS_Array<T, 2> src, RSize src_dim) {
        RIndex index = index1D();
        RSize row = index / src_dim;
        RSize col = index % src_dim;
        dst[{row, col}] = src[{row, col}];
    }
};

template <typename T, typename Array1DLike>
void copy(
    RS_StructOfArrays<T> dst_soa,
    Array1DLike src_array,
    RSize copy_num,
    RSize soa_offset = 0,
    RSize array_offset = 0,
    RSize block_size = RS_AUTO_BLOCK_SIZE,
    bool sync = true
) {
    static_assert(!std::is_reference_v<Array1DLike>, "Array1DLike must not be a reference type");

    auto mem_provider = dst_soa.provider();
    castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
        auto copy_src_array = src_array;
        if (src_array.device() != dst_soa.device()) {
            copy_src_array = src_array.move(mem_provider);
            sync = true;
        }
        p_ptr->template run<RSCF_CopyFromArray1DLikeToSoA>(
            {copy_num, block_size, sync}, dst_soa, copy_src_array, soa_offset, array_offset
        );
        if (src_array.device() != dst_soa.device()) {
            copy_src_array.free();
        }
    });
}

template <typename T, typename Array1DLike>
void copy(
    Array1DLike dst_array,
    RS_StructOfArrays<T> src_soa,
    RSize copy_num,
    RSize dst_offset = 0,
    RSize soa_offset = 0,
    RSize block_size = RS_AUTO_BLOCK_SIZE,
    bool sync = true
) {
    static_assert(!std::is_reference_v<Array1DLike>, "Array1DLike must not be a reference type");

    auto mem_provider = dst_array.provider();
    castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
        auto copy_src_soa = src_soa;
        if (src_soa.device() != dst_array.device()) {
            copy_src_soa = src_soa.move(mem_provider);
            sync = true;
        }
        p_ptr->template run<RSCF_CopyFromSoAToArray1DLike>(
            {copy_num, block_size, sync}, dst_array, copy_src_soa, dst_offset, soa_offset
        );
        if (src_soa.device() != dst_array.device()) {
            copy_src_soa.free();
        }
    });
}

template <typename Array1DLike>
void copy(
    Array1DLike dst_array,
    Array1DLike src_array,
    RSize copy_num,
    RSize dst_offset = 0,
    RSize src_offset = 0,
    RSize block_size = RS_AUTO_BLOCK_SIZE,
    bool sync = true
) {
    static_assert(!std::is_reference_v<Array1DLike>, "Array1DLike must not be a reference type");

    auto mem_provider = dst_array.provider();
    castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
        auto copy_src_array = src_array;
        if (src_array.device() != dst_array.device()) {
            copy_src_array = src_array.move(mem_provider);
            sync = true;
        }
        p_ptr->template run<RSCF_CopyFromArray1DLikeToArray1DLike>(
            {copy_num, block_size, sync}, dst_array, copy_src_array, dst_offset, src_offset
        );
        if (src_array.device() != dst_array.device()) {
            copy_src_array.free();
        }
    });
}

template <typename T>
void copy(
    RS_StructOfArrays<T> dst_soa,
    RS_StructOfArrays<T> src_soa,
    RSize copy_num,
    RSize dst_soa_offset = 0,
    RSize src_soa_offset = 0,
    RSize block_size = RS_AUTO_BLOCK_SIZE,
    bool sync = true
) {
    auto mem_provider = dst_soa.provider();
    castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
        auto copy_src_soa = src_soa;
        if (src_soa.device() != dst_soa.device()) {
            copy_src_soa = src_soa.move(mem_provider);
            sync = true;
        }
        p_ptr->template run<RSCF_CopyFromSoAToSoA>(
            {copy_num, block_size, sync}, dst_soa, copy_src_soa, dst_soa_offset, src_soa_offset
        );
        if (src_soa.device() != dst_soa.device()) {
            copy_src_soa.free();
        }
    });
}

/**
 * @brief Copy a 2D array into a (potentially larger) 2D array, preserving data.
 *
 * @param dst Destination 2D array (must be sized >= src in both dimensions)
 * @param src Source 2D array
 * @param src_dim Dimension of the source (square) array
 */
template <typename T>
void copy2D(
    RS_Array<T, 2> dst,
    RS_Array<T, 2> src,
    RSize src_dim,
    RSize block_size = RS_AUTO_BLOCK_SIZE,
    bool sync = true
) {
    if (src_dim == 0) return;
    auto mem_provider = dst.provider();
    castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
        p_ptr->template run<RSCF_CopyFromArray2DToArray2D>(
            {src_dim * src_dim, block_size, sync}, dst, src, src_dim
        );
    });
}

}  // namespace device

}  // namespace rise

#endif  // RS_COPY_H