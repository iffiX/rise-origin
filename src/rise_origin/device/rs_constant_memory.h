// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONSTANT_MEMORY_H
#define RS_CONSTANT_MEMORY_H

#include <fmt/format.h>

#include "rise_origin/utils/rs_def.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>

#include "rise_origin/device/platform/rs_cuda.h"
extern __constant__ char rs_cuda_constant_memory[];

#else

#include "rise_origin/device/platform/rs_cpu.h"

#endif

extern char rs_host_constant_memory[];

namespace rise {

namespace device {

class RS_ConstantMemory {
   public:
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER static T* as() {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<T*>(rs_cuda_constant_memory);
#else
        return reinterpret_cast<T*>(rs_host_constant_memory);
#endif
    }

    static constexpr RSize size() { return RS_DEVICE_CONSTANT_MEMORY_MAX_SIZE; }

    static void write(void* src, RS_DeviceLocation device, RSize size, RSize offset = 0) {
        if (size == 0) return;
        if (size + offset > RS_DEVICE_CONSTANT_MEMORY_MAX_SIZE) {
            RS_THROW(RS_OutOfBoundsException(
                "Cannot write to constant memory with size {} and offset {}, "
                "max size is {} bytes",
                size,
                offset,
                RS_DEVICE_CONSTANT_MEMORY_MAX_SIZE
            ));
        }
        if (device.type == RSE_DEVICE_TYPE_CPU) {
            if (device.id == RS_NULL_INDEX) {
                RS_THROW(RS_HostExecutionException(
                    "Cannot initialize constant memory of device type {} with invalid id {}",
                    enumToString(device.type),
                    device.id
                ));
            }
            memcpy(rs_host_constant_memory + offset, src, size);
            return;
        }
#ifdef __CUDACC__
        else if (device.type == RSE_DEVICE_TYPE_CUDA) {
            int device_count;
            RcudaGetDeviceCount(&device_count);
            if (device.id >= device_count) {
                RS_THROW(RS_HostExecutionException(
                    "Cannot initialize constant memory of device type {} with invalid id {}",
                    enumToString(device.type),
                    device.id
                ));
            }
            int current_device;
            RcudaGetDevice(&current_device);
            if (device.id != current_device) RcudaSetDevice(device.id);
            RcudaMemcpyToSymbol(rs_cuda_constant_memory, src, size, offset, cudaMemcpyHostToDevice);
            if (device.id != current_device) RcudaSetDevice(current_device);
            return;
        }
#endif
        RS_THROW(RS_HostExecutionException(
            "Cannot write to constant memory of unsupported device type {} id {}",
            enumToString(device.type),
            device.id
        ));
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_CONSTANT_MEMORY_H
