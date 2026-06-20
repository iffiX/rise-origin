// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_PLATFORM_H
#define RS_PLATFORM_H

#include "rise_origin/utils/memory/rs_memory.h"
#include "rs_cpu.h"

#ifdef __CUDACC__
#include <cuda/std/array>
#include <cuda/std/atomic>
#include <cuda/std/tuple>

#include "rs_cuda.h"
#else
#include <array>
#include <atomic>
#include <tuple>

#endif

namespace rise {

namespace device {

#ifdef __CUDACC__
namespace rs_device_std = cuda::std;

inline bool isDeviceValid(const RS_DeviceLocation& device, std::string& error) {
    if (device.type == RSE_DEVICE_TYPE_CPU) {
        if (device.id == RS_NULL_INDEX) {
            error = "CPU device index cannot be NULL_INDEX";
            return false;
        }
        error = "";
        return true;
    }
    if (device.type == RSE_DEVICE_TYPE_CUDA) {
        int count;
        RcudaGetDeviceCount(&count);
        if (device.id >= count) {
            error = fmt::format("Invalid GPU index {}, valid range is [0, {})", device.id, count);
            return false;
        }
        error = "";
        return true;
    }
    error = format(
        "Unsupported device type {}, Supported are RSE_DEVICE_TYPE_CPU, RSE_DEVICE_TYPE_CUDA",
        enumToString(device.type)
    );
    return false;
}

inline std::shared_ptr<RS_MemoryProvider> getMemoryProvider(const RS_DeviceLocation& device) {
    std::string error;
    if (!isDeviceValid(device, error)) {
        RS_THROW(RS_InvalidArgumentException(error));
    }
    if (device.type == RSE_DEVICE_TYPE_CUDA) {
        std::shared_ptr<RS_MemoryCudaProvider> provider(
            new RS_MemoryCudaProvider(device), [](RS_MemoryCudaProvider* p) {
                p->free();
                delete p;
            }
        );
        return provider;
    }
    // RSE_DEVICE_TYPE_CPU
    std::shared_ptr<RS_MemoryCPUProvider> provider(
        new RS_MemoryCPUProvider(0), [](RS_MemoryCPUProvider* p) {
            p->free();
            delete p;
        }
    );
    return provider;
}

template <typename Function>
void castProviderPtrAndCall(RS_MemoryProvider* provider, Function function) {
    RS_MemoryProvider* base = provider->base();
    switch (base->type) {
        case RSE_MEMORY_PROVIDER_TYPE_CUDA:
            function(dynamic_cast<RS_MemoryCudaProvider*>(base));
            break;
        case RSE_MEMORY_PROVIDER_TYPE_CPU:
            function(dynamic_cast<RS_MemoryCPUProvider*>(base));
            break;
        case RSE_MEMORY_PROVIDER_TYPE_DEFAULT:
            function(dynamic_cast<RS_MemoryDefaultProvider*>(base));
            break;
        default:
            RS_THROW(RS_InvalidArgumentException("Invalid memory provider"));
    }
}

#else
namespace rs_device_std = std;

inline bool isDeviceValid(const RS_DeviceLocation& device, std::string& error) {
    if (device.type == RSE_DEVICE_TYPE_CPU) {
        if (device.id == RS_NULL_INDEX) {
            error = "CPU device index cannot be NULL_INDEX";
            return false;
        }
        error = "";
        return true;
    }
    error = format(
        "Unsupported device type {}, Supported is RSE_DEVICE_TYPE_CPU", enumToString(device.type)
    );
    return false;
}

inline std::shared_ptr<RS_MemoryProvider> getMemoryProvider(const RS_DeviceLocation& device) {
    std::string error;
    if (!isDeviceValid(device, error)) {
        RS_THROW(RS_InvalidArgumentException(error));
    }
    // RSE_DEVICE_TYPE_CPU
    std::shared_ptr<RS_MemoryCPUProvider> provider(
        new RS_MemoryCPUProvider(0), [](RS_MemoryCPUProvider* p) {
            p->free();
            delete p;
        }
    );
    return provider;
}

template <typename Function>
void castProviderPtrAndCall(RS_MemoryProvider* provider, Function function) {
    switch (provider->type) {
        case RSE_MEMORY_PROVIDER_TYPE_CPU:
            function(dynamic_cast<RS_MemoryCPUProvider*>(provider));
            break;
        case RSE_MEMORY_PROVIDER_TYPE_DEFAULT:
            function(dynamic_cast<RS_MemoryDefaultProvider*>(provider));
            break;
        default:
            RS_THROW(RS_InvalidArgumentException("Invalid memory provider"));
    }
}
#endif

}  // namespace device

}  // namespace rise

#endif  // RS_PLATFORM_H
