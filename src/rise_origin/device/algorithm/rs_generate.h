// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_GENERATE_H
#define RS_GENERATE_H

#include "rise_origin/device/rs_compute.h"

namespace rise {

namespace device {
class RSCF_GenerateIndexArray1DLike : public RS_ComputeFunction<RSCF_GenerateIndexArray1DLike> {
   public:
    template <typename Array1DLike>
    HOST_DEVICE_INLINE_QUALIFIER static void run(Array1DLike array) {
        RIndex index = index1D();
        array[index] = index;
    }
};

template <typename Array1DLike>
void generateIndex(Array1DLike array, RSize block_size = RS_AUTO_BLOCK_SIZE, bool sync = true) {
    static_assert(!std::is_reference_v<Array1DLike>, "Array1DLike must not be a reference type");

    if (array.size() == 0) return;
    auto mem_provider = array.provider();
    castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
        p_ptr->template run<RSCF_GenerateIndexArray1DLike>({array.size(), block_size, sync}, array);
    });
}

}  // namespace device

}  // namespace rise
#endif  // RS_GENERATE_H
