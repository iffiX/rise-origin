// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CUDA_H
#define RS_CUDA_H

#include <cuda.h>

#include <boost/core/demangle.hpp>
#include <iostream>
#include <string>
#include <unordered_map>

#include "rise_origin/device/platform/rs_exec_trace.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_env_var.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/time/rs_timer.h"

#define COLORCODE_RED "\033[0;31m"
#define COLORCODE_BOLD_RED "\033[1;31m\n"
#define COLORCODE_GREEN "\033[0;32m"
#define COLORCODE_BLUE "\033[0;34m"
#define COLORCODE_RESET "\033[0m"

#define CUDA_PRINTF_ASSERT(assert)                                                               \
    if (not(assert)) {                                                                           \
        printf(                                                                                  \
            "Assertion " #assert " failed: @file " __FILE__ " @line " RS_TOSTRING(__LINE__) "\n" \
        );                                                                                       \
    }

#define CUDA_FUNC_PRINT_RESOURCE(func)                                                         \
    {                                                                                          \
        cudaFuncAttributes attr;                                                               \
        cudaFuncGetAttributes(&attr, func);                                                    \
                                                                                               \
        printf(COLORCODE_BLUE "Function: %s " COLORCODE_RESET "\n" COLORCODE_RESET, #func);    \
        printf(                                                                                \
            COLORCODE_BLUE "Constant Memory Size: %zu bytes " COLORCODE_RESET "\n",            \
            attr.constSizeBytes                                                                \
        );                                                                                     \
        printf(                                                                                \
            COLORCODE_BLUE "Shared Memory Size: %zu (req) / %d (allow) bytes " COLORCODE_RESET \
                           "\n",                                                               \
            attr.sharedSizeBytes,                                                              \
            attr.maxDynamicSharedSizeBytes                                                     \
        );                                                                                     \
        printf(                                                                                \
            COLORCODE_BLUE "Local Memory Size: %zu bytes " COLORCODE_RESET "\n",               \
            attr.localSizeBytes                                                                \
        );                                                                                     \
        printf(COLORCODE_BLUE "Number of Registers: %d " COLORCODE_RESET "\n", attr.numRegs);  \
    }

#define CUDA_PRINT_DEVICE_MEMORY(ptr, data_type, offset, size)               \
    {                                                                        \
        data_type *__host_ptr = nullptr;                                     \
        RcudaMallocHost(&__host_ptr, sizeof(data_type) * size);              \
        cudaMemcpy(                                                          \
            __host_ptr,                                                      \
            ((data_type *)ptr) + offset,                                     \
            sizeof(data_type) * size,                                        \
            cudaMemcpyDeviceToHost                                           \
        );                                                                   \
        for (int __i = 0; __i < size; __i++) {                               \
            std::cout << "@" << __i << ": " << __host_ptr[__i] << std::endl; \
        }                                                                    \
        RcudaFreeHost(__host_ptr);                                           \
    }

#define CUDA_PRINT_DEVICE_MEMORY_STREAM(ptr, data_type, offset, size, stream) \
    {                                                                         \
        data_type *__host_ptr = nullptr;                                      \
        RcudaMallocHost(&__host_ptr, sizeof(data_type) * size);               \
        RcudaStreamSynchronize(stream);                                       \
        RcudaMemcpyAsync(                                                     \
            __host_ptr,                                                       \
            ((data_type *)ptr) + offset,                                      \
            sizeof(data_type) * size,                                         \
            cudaMemcpyDeviceToHost,                                           \
            stream                                                            \
        );                                                                    \
        RcudaStreamSynchronize(stream);                                       \
        for (int __i = 0; __i < size; __i++) {                                \
            std::cout << "@" << __i << ": " << __host_ptr[__i] << std::endl;  \
        }                                                                     \
        RcudaFreeHost(__host_ptr);                                            \
    }

#define CUDA_ERROR_CHECK(error_code, function, ...) \
    rise::device::cudaCheckCallStatus((error_code), __FILE__, __LINE__, function, ##__VA_ARGS__)

#define RcudaMemGetInfo(freePtr, totalPtr)                             \
    {                                                                  \
        std::string function = fmt::format(                            \
            "cudaMemGetInfo(freePtr={}, totalPtr={})",                 \
            static_cast<void *>(freePtr),                              \
            static_cast<void *>(totalPtr)                              \
        );                                                             \
        CUDA_ERROR_CHECK(cudaMemGetInfo(freePtr, totalPtr), function); \
    }
#define RcudaDeviceSetLimit(limit, value)                                                          \
    {                                                                                              \
        std::string function =                                                                     \
            fmt::format("cudaDeviceSetLimit(limit={}, value={})", static_cast<int>(limit), value); \
        CUDA_ERROR_CHECK(cudaDeviceSetLimit(limit, value), function);                              \
    }
#define RcudaSetDevice(device)                                                  \
    {                                                                           \
        std::string function = fmt::format("cudaSetDevice(device={})", device); \
        CUDA_ERROR_CHECK(cudaSetDevice(device), function);                      \
    }
#define RcudaGetDevice(deviceNumberPtr)                                               \
    {                                                                                 \
        std::string function = fmt::format(                                           \
            "cudaGetDevice(deviceNumberPtr={})", static_cast<void *>(deviceNumberPtr) \
        );                                                                            \
        CUDA_ERROR_CHECK(cudaGetDevice(deviceNumberPtr), function);                   \
    }
#define RcudaGetDeviceCount(countPtr)                                                      \
    {                                                                                      \
        std::string function =                                                             \
            fmt::format("cudaGetDeviceCount(countPtr={})", static_cast<void *>(countPtr)); \
        CUDA_ERROR_CHECK(cudaGetDeviceCount(countPtr), function);                          \
    }
#define RcudaStreamCreate(streamPtr)                                                       \
    {                                                                                      \
        std::string function =                                                             \
            fmt::format("cudaStreamCreate(streamPtr={})", static_cast<void *>(streamPtr)); \
        CUDA_ERROR_CHECK(cudaStreamCreate(streamPtr), function);                           \
    }
#define RcudaStreamDestroy(stream)                                                    \
    {                                                                                 \
        std::string function =                                                        \
            fmt::format("cudaStreamDestroy(stream={})", static_cast<void *>(stream)); \
        CUDA_ERROR_CHECK(cudaStreamDestroy(stream), function);                        \
    }
#define RcudaMemcpy(dst, src, count, kind)                             \
    {                                                                  \
        std::string function = fmt::format(                            \
            "cudaMemcpy(dst={}, src={}, count={}, kind={})",           \
            static_cast<void *>(dst),                                  \
            static_cast<const void *>(src),                            \
            count,                                                     \
            rise::device::getMemcpyKind(kind)                          \
        );                                                             \
        rise::RS_Timer __exec_timer;                                   \
        cudaMemcpy(dst, src, count, kind);                             \
        CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer); \
    }
#define RcudaMemcpyAsync(dst, src, count, kind, stream)                      \
    {                                                                        \
        std::string function = fmt::format(                                  \
            "cudaMemcpyAsync(dst={}, src={}, count={}, kind={}, stream={})", \
            static_cast<void *>(dst),                                        \
            static_cast<const void *>(src),                                  \
            count,                                                           \
            rise::device::getMemcpyKind(kind),                               \
            static_cast<void *>(stream)                                      \
        );                                                                   \
        rise::RS_Timer __exec_timer;                                         \
        cudaMemcpyAsync(dst, src, count, kind, stream);                      \
        if (RS_EnvironmentVariables::isExecForceSync())        \
            cudaStreamSynchronize(stream);                                   \
        CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer);       \
    }
#define RcudaMemcpyToSymbolAsync(symbol, src, count, offset, stream)                               \
    {                                                                                              \
        std::string function = fmt::format(                                                        \
            "cudaMemcpyToSymbolAsync(symbol={}, src={}, count={}, offset={}, kind={}, stream={})", \
            RS_STRINGIFY(symbol),                                                                  \
            static_cast<const void *>(src),                                                        \
            count,                                                                                 \
            offset,                                                                                \
            "HostToDevice",                                                                        \
            static_cast<void *>(stream)                                                            \
        );                                                                                         \
        rise::RS_Timer __exec_timer;                                                               \
        cudaMemcpyToSymbolAsync(symbol, src, count, offset, cudaMemcpyHostToDevice, stream);       \
        if (RS_EnvironmentVariables::isExecForceSync())                              \
            cudaStreamSynchronize(stream);                                                         \
        CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer);                             \
    }

#define RcudaMemcpyToSymbol(symbol, src, count, offset, kind)          \
    {                                                                  \
        std::string function = fmt::format(                            \
            "cudaMemcpyToSymbol(symbol={}, src={}, count={}, "         \
            "offset={}, kind={})",                                     \
            RS_STRINGIFY(symbol),                                      \
            static_cast<const void *>(src),                            \
            count,                                                     \
            offset,                                                    \
            rise::device::getMemcpyKind(kind)                          \
        );                                                             \
        rise::RS_Timer __exec_timer;                                   \
        cudaMemcpyToSymbol(symbol, src, count, offset, kind);          \
        CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer); \
    }

#define RcudaMemsetAsync(dMem, value, count, stream)                   \
    {                                                                  \
        std::string function = fmt::format(                            \
            "cudaMemsetAsync(dMem={}, value={}, count={}, stream={})", \
            static_cast<void *>(dMem),                                 \
            value,                                                     \
            count,                                                     \
            static_cast<void *>(stream)                                \
        );                                                             \
        rise::RS_Timer __exec_timer;                                   \
        cudaMemsetAsync(dMem, value, count, stream);                   \
        if (RS_EnvironmentVariables::isExecForceSync())  \
            cudaStreamSynchronize(stream);                             \
        CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer); \
    }
#define RcudaMallocAsync(dMemPtr, size, stream)                        \
    {                                                                  \
        std::string function = fmt::format(                            \
            "cudaMallocAsync(dMemPtr={}, size={}, stream={})",         \
            static_cast<void *>(dMemPtr),                              \
            size,                                                      \
            static_cast<void *>(stream)                                \
        );                                                             \
        rise::RS_Timer __exec_timer;                                   \
        cudaMallocAsync(dMemPtr, size, stream);                        \
        if (RS_EnvironmentVariables::isExecForceSync())  \
            cudaStreamSynchronize(stream);                             \
        CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer); \
    }
#define RcudaFreeAsync(dMem, stream)                                       \
    {                                                                      \
        std::string function = fmt::format(                                \
            "cudaFreeAsync(dMem={}, stream={})",                           \
            static_cast<void *>(dMem),                                     \
            static_cast<void *>(stream)                                    \
        );                                                                 \
        if (dMem != nullptr) {                                             \
            rise::RS_Timer __exec_timer;                                   \
            cudaFreeAsync(dMem, stream);                                   \
            if (RS_EnvironmentVariables::isExecForceSync())  \
                cudaStreamSynchronize(stream);                             \
            CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer); \
        }                                                                  \
    }
#define RcudaMallocArray(arrayPtr, desc, width, height)                                         \
    {                                                                                           \
        std::string function = fmt::format(                                                     \
            "cudaMallocArray(arrayPtr={}, desc=(format={}, w={}, x={}, y={}, z={}), width={}, " \
            "height={})",                                                                       \
            static_cast<void *>(arrayPtr),                                                      \
            rise::device::getChannelFormatKind((*desc).f),                                      \
            (*desc).w,                                                                          \
            (*desc).x,                                                                          \
            (*desc).y,                                                                          \
            (*desc).z,                                                                          \
            width,                                                                              \
            height                                                                              \
        );                                                                                      \
        CUDA_ERROR_CHECK(cudaMallocArray(arrayPtr, desc, width, height), function);             \
    }
#define RcudaFreeArray(array)                                                                      \
    {                                                                                              \
        std::string function = fmt::format("cudaFreeArray(array={})", static_cast<void *>(array)); \
        if (array != nullptr) CUDA_ERROR_CHECK(cudaFreeArray(array), function);                    \
    }
#define RcudaMallocHost(hMemPtr, size)                                                \
    {                                                                                 \
        std::string function = fmt::format(                                           \
            "cudaMallocHost(hMemPtr={}, size={})", static_cast<void *>(hMemPtr), size \
        );                                                                            \
        rise::RS_Timer __exec_timer;                                                  \
        CUDA_ERROR_CHECK(cudaMallocHost(hMemPtr, size), function, &__exec_timer);     \
    }
#define RcudaFreeHost(hMem)                                                                     \
    {                                                                                           \
        std::string function = fmt::format("cudaFreeHost(hMem={})", static_cast<void *>(hMem)); \
        if (hMem != nullptr) {                                                                  \
            rise::RS_Timer __exec_timer;                                                        \
            CUDA_ERROR_CHECK(cudaFreeHost(hMem), function, &__exec_timer);                      \
        }                                                                                       \
    }
#define RcudaStreamSynchronize(stream)                                                    \
    {                                                                                     \
        std::string function =                                                            \
            fmt::format("cudaStreamSynchronize(stream={})", static_cast<void *>(stream)); \
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream), function);                        \
    }

#define RcudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)                      \
    {                                                                               \
        std::string function = fmt::format(                                         \
            "cudaMemcpyPeer(dst={}, dstDevice={}, src={}, srcDevice={}, count={})", \
            static_cast<void *>(dst),                                               \
            dstDevice,                                                              \
            static_cast<const void *>(src),                                         \
            srcDevice,                                                              \
            count                                                                   \
        );                                                                          \
        rise::RS_Timer __exec_timer;                                                \
        cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);                      \
        CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer);              \
    }

#define RcudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)              \
    {                                                                                    \
        std::string function = fmt::format(                                              \
            "cudaMemcpyPeerAsync(dst={}, dstDevice={}, src={}, srcDevice={}, count={}, " \
            "stream={})",                                                                \
            static_cast<void *>(dst),                                                    \
            dstDevice,                                                                   \
            static_cast<const void *>(src),                                              \
            srcDevice,                                                                   \
            count,                                                                       \
            static_cast<void *>(stream)                                                  \
        );                                                                               \
        rise::RS_Timer __exec_timer;                                                     \
        cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);              \
        if (RS_EnvironmentVariables::isExecForceSync())                    \
            cudaStreamSynchronize(stream);                                               \
        CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer);                   \
    }

#define RcudaStreamBeginCapture(stream, mode)                                                    \
    {                                                                                            \
        std::string function = fmt::format(                                                      \
            "cudaStreamBeginCapture(stream={}, mode={})", static_cast<void *>(stream), (int)mode \
        );                                                                                       \
        CUDA_ERROR_CHECK(cudaStreamBeginCapture(stream, mode), function);                        \
    }
#define RcudaStreamEndCapture(stream, graphPtr)                             \
    {                                                                       \
        std::string function = fmt::format(                                 \
            "cudaStreamEndCapture(stream={}, graphPtr={})",                 \
            static_cast<void *>(stream),                                    \
            static_cast<void *>(graphPtr)                                   \
        );                                                                  \
        CUDA_ERROR_CHECK(cudaStreamEndCapture(stream, graphPtr), function); \
    }
#define RcudaGraphInstantiate(execPtr, graph, flags)                                               \
    {                                                                                              \
        std::string function = fmt::format(                                                        \
            "cudaGraphInstantiate(execPtr={}, graph={}, flags={})",                                \
            static_cast<void *>(execPtr),                                                          \
            static_cast<void *>(graph),                                                            \
            flags                                                                                  \
        );                                                                                         \
        CUDA_ERROR_CHECK(cudaGraphInstantiate(execPtr, graph, nullptr, nullptr, flags), function); \
    }
#define RcudaGraphLaunch(exec, stream)                                 \
    {                                                                  \
        std::string function = fmt::format(                            \
            "cudaGraphLaunch(exec={}, stream={})",                     \
            static_cast<void *>(exec),                                 \
            static_cast<void *>(stream)                                \
        );                                                             \
        rise::RS_Timer __exec_timer;                                   \
        cudaGraphLaunch(exec, stream);                                 \
        if (RS_EnvironmentVariables::isExecForceSync())  \
            cudaStreamSynchronize(stream);                             \
        CUDA_ERROR_CHECK(cudaGetLastError(), function, &__exec_timer); \
    }

#define CUDA_CHECK_AFTER_CALL(function, grid_size, block_size, smem, stream, timer) \
    {                                                                               \
        cudaStreamSynchronize(stream);                                              \
        std::string function_name = boost::core::demangle(typeid(function).name()); \
        std::string function_signature = fmt::format(                               \
            "{}<<<grid_size={}, block_size={}, smem={}, stream={}>>>",              \
            function_name,                                                          \
            grid_size,                                                              \
            block_size,                                                             \
            smem,                                                                   \
            static_cast<void *>(stream)                                             \
        );                                                                          \
        CUDA_ERROR_CHECK(cudaGetLastError(), function_signature, timer);            \
    }

#define CUDA_MAX_BLOCK_SIZE 1024
#define CUDA_MAX_WARP_SIZE 32

namespace rise {

namespace device {

template <typename T>
std::pair<int, int> getGridAndBlockSize(T func, size_t item_num) {
    int min_grid_size, grid_size, block_size;

    if (item_num < 256) {
        // For small simulations, fit them into 1 block
        // No more than 1024 (cuda max block thread num)
        block_size = 256;
    } else {
        // Dynamically calculate blockSize
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func, 0, item_num);
    }
    grid_size = RS_CEIL_DIV(item_num, block_size);
    block_size = RS_MIN(item_num, block_size);
    return std::make_pair(grid_size, block_size);
}

HOST_QUALIFIER inline void cudaCheckCallStatus(
    cudaError_t code,
    const char *file,
    int line,
    std::string function,
    const RS_Timer *timer = nullptr
) {
    execTracePrint(file, line, function, timer);
    if (code != cudaSuccess) {
        std::string error;
        std::string note =
            !RS_EnvironmentVariables::isExecForceSync()
                ? " Note: CUDA kernels are executing in asynchronous mode, this error may be "
                  "produced "
                  "by a past CUDA call. To accurately check CUDA errors, set the environment "
                  "variable EXEC_FORCE_SYNC = 1."
                : "";
        if (function.empty())
            error = format(
                "{}\n{}({}): CUDA Function Error: {}", note, file, line, cudaGetErrorString(code)
            );
        else
            error = format(
                "{}\nWhile executing function {}\n"
                "{}({}): CUDA Function Error: {}",
                note,
                function,
                file,
                line,
                cudaGetErrorString(code)
            );
        RS_THROW(RS_DeviceExecutionException(error));
    }
}

HOST_QUALIFIER inline const char *getMemcpyKind(const cudaMemcpyKind &kind) {
    if (kind == cudaMemcpyHostToDevice) return "cudaMemcpyHostToDevice";
    if (kind == cudaMemcpyDeviceToHost) return "cudaMemcpyDeviceToHost";
    if (kind == cudaMemcpyDeviceToDevice) return "cudaMemcpyDeviceToDevice";
    if (kind == cudaMemcpyHostToHost) return "cudaMemcpyHostToHost";
    return "UNKNOWN";
}

HOST_QUALIFIER inline const char *getChannelFormatKind(const cudaChannelFormatKind &kind) {
    if (kind == cudaChannelFormatKindSigned) return "cudaChannelFormatKindSigned";
    if (kind == cudaChannelFormatKindUnsigned) return "cudaChannelFormatKindUnsigned";
    if (kind == cudaChannelFormatKindFloat) return "cudaChannelFormatKindFloat";
    if (kind == cudaChannelFormatKindNone) return "cudaChannelFormatKindNone";
    return "UNKNOWN";
}

template <typename ComputeFunction, typename... Args>
__global__ void globalComputeFunctionRunner(RSize max_threads, Args... args) {
    RIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= max_threads) return;
    ComputeFunction::run(args...);
}

template <typename ComputeFunction, typename... Args>
__global__ void globalComputeFunctionDebugRunner(RSize max_threads, Args... args) {
    RIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= max_threads) return;
    ComputeFunction::run(args...);
}

template <typename ComputeFunction, typename... Args>
__global__ void globalComputeFunctionGridRunner(Args... args) {
    ComputeFunction::run(args...);
}

struct RS_CudaGraphCacheBase {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;

    virtual ~RS_CudaGraphCacheBase() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};

template <typename DepTuple>
struct RS_CudaGraphCacheEntry : RS_CudaGraphCacheBase {
    DepTuple stored_deps{};
    bool valid = false;
};

class RS_MemoryCudaProvider final : public RS_MemoryProvider {
   public:
    std::shared_ptr<std::recursive_mutex> mutex = nullptr;

    cudaStream_t stream = nullptr;
    cudaStream_t stream_copy_to = nullptr;
    cudaStream_t stream_copy_from = nullptr;

    RSize stream_event_count = 0;
    RSize stream_copy_to_event_count = 0;
    RSize stream_copy_from_event_count = 0;

    std::unordered_map<const char *, std::unique_ptr<RS_CudaGraphCacheBase>> capture_cache;

    RS_MemoryCudaProvider() {
        type = RSE_MEMORY_PROVIDER_TYPE_CUDA;
        device = {RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX};
    };

    explicit RS_MemoryCudaProvider(
        const RS_DeviceLocation &device,
        cudaStream_t stream = nullptr,
        cudaStream_t stream_copy_to = nullptr,
        cudaStream_t stream_copy_from = nullptr
    ) {
        init(device, stream, stream_copy_to, stream_copy_from);
    };

    void init(
        const RS_DeviceLocation &device,
        cudaStream_t stream = nullptr,
        cudaStream_t stream_copy_to = nullptr,
        cudaStream_t stream_copy_from = nullptr
    ) {
        if (mutex) {
            RS_THROW(RS_ReinitializedException("CUDA provider is already initialized"));
        }

        if (device.type != RSE_DEVICE_TYPE_CUDA) {
            RS_THROW(
                RS_InvalidArgumentException("Invalid device type {}", enumToString(device.type))
            );
        }

        this->device = device;

        type = RSE_MEMORY_PROVIDER_TYPE_CUDA;
        mutex = std::make_shared<std::recursive_mutex>();

        int count;
        RcudaGetDeviceCount(&count);
        if (count <= 0) RS_THROW(RS_DeviceExecutionException("No GPU found."));

        if (device.id >= count)
            RS_THROW(RS_DeviceExecutionException(
                "Invalid GPU index {}, valid range is [0, {})", device.id, count
            ));

        int current_device;
        RcudaGetDevice(&current_device);
        if (device.type == RSE_DEVICE_TYPE_CUDA) RcudaSetDevice(device.id);

        if (stream == nullptr) {
            RcudaStreamCreate(&this->stream);
        } else {
            this->stream = stream;
        }

        if (stream_copy_to == nullptr) {
            RcudaStreamCreate(&this->stream_copy_to);
        } else {
            this->stream_copy_to = stream_copy_to;
        }

        if (stream_copy_from == nullptr) {
            RcudaStreamCreate(&this->stream_copy_from);
        } else {
            this->stream_copy_from = stream_copy_from;
        }

        stream_event_count = 0;
        stream_copy_to_event_count = 0;
        stream_copy_from_event_count = 0;

        RcudaSetDevice(current_device);
    };

    void free() override {
        // Not initialized, no op
        if (!mutex) return;

        mutex->lock();

        RcudaStreamSynchronize(stream);

        // Destroy all cached CUDA graphs before destroying the stream
        capture_cache.clear();

        RcudaStreamDestroy(stream);
        stream = nullptr;
        RcudaStreamSynchronize(stream_copy_to);
        RcudaStreamDestroy(stream_copy_to);
        stream_copy_to = nullptr;
        RcudaStreamSynchronize(stream_copy_from);
        RcudaStreamDestroy(stream_copy_from);
        stream_copy_from = nullptr;

        mutex->unlock();
        mutex.reset();
        device = {RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX};
    }

    std::string getProperty(const std::string &key) const override {
        if (key == "default_auto_block_size")
            return std::to_string(RS_DEVICE_CUDA_DEFAULT_BLOCK_SIZE);
        if (key == "memory_used" || key == "memory_capacity") {
            if (!mutex) return "";
            std::lock_guard<std::recursive_mutex> lock(*mutex.get());
            int current_device;
            RcudaGetDevice(&current_device);
            if (device.id != current_device) RcudaSetDevice(device.id);
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            RcudaMemGetInfo(&free_bytes, &total_bytes);
            if (device.id != current_device) RcudaSetDevice(current_device);
            if (key == "memory_used") return std::to_string(total_bytes - free_bytes);
            return std::to_string(total_bytes);
        }
        return "";
    }

    RS_MemoryPtr allocate(RBigSize size, bool sync = true) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CUDA provider is not initialized"));

        if (size == 0) return {nullptr, 0, 0, device, this};

        std::lock_guard<std::recursive_mutex> lock(*mutex.get());
        void *ptr = nullptr;

        int current_device;
        RcudaGetDevice(&current_device);
        if (device.type == RSE_DEVICE_TYPE_CUDA && device.id != current_device)
            RcudaSetDevice(device.id);

        RcudaMallocAsync(&ptr, size, stream);
        ++stream_event_count;

        if (!ptr) {
            RS_THROW(
                RS_DeviceMemoryException("Failed to allocate device memory on the CUDA provider")
            );
        }

        if (sync) {
            RcudaStreamSynchronize(stream);
            --stream_event_count;
        }

        if (device.type == RSE_DEVICE_TYPE_CUDA && device.id != current_device)
            RcudaSetDevice(current_device);

        if (RS_EnvironmentVariables::isExecForceSync()) {
            RcudaStreamSynchronize(stream);
            --stream_event_count;
        }

        return {ptr, size, 0, device, this};
    }

    RS_MemoryPtr allocateHost(RBigSize size, bool sync = true) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CUDA provider is not initialized"));

        if (size == 0) return {nullptr, 0, 0, {RSE_DEVICE_TYPE_CPU, 0}, this};

        std::lock_guard<std::recursive_mutex> lock(*mutex.get());
        void *ptr = nullptr;
        RcudaMallocHost(&ptr, size);
        if (!ptr) {
            RS_THROW(RS_HostMemoryException("Failed to allocate host memory on the CUDA provider"));
        }
        return {ptr, size, 0, {RSE_DEVICE_TYPE_CPU, 0}, this};
    }

    void deallocate(const RS_MemoryPtr &ptr, bool sync = true) override {
        if (stream == nullptr)
            RS_THROW(RS_HostExecutionException("CUDA provider is not initialized"));

        if (ptr.empty()) return;

        if (ptr.provider() != this)
            RS_THROW(RS_HostExecutionException("Cannot free memory allocated by another provider"));

        std::lock_guard<std::recursive_mutex> lock(*mutex.get());
        if (ptr.device().type == RSE_DEVICE_TYPE_CPU) {
            RcudaFreeHost(ptr.data);
        } else {
            int current_device;
            RcudaGetDevice(&current_device);
            if (ptr.device().id != current_device) RcudaSetDevice(ptr.device().id);

            RcudaFreeAsync(ptr.data, stream);
            ++stream_event_count;

            if (sync || RS_EnvironmentVariables::isExecForceSync()) {
                RcudaStreamSynchronize(stream);
                --stream_event_count;
            }
            if (ptr.device().id != current_device) RcudaSetDevice(current_device);
        }
    }

    void set(
        const RS_MemoryPtr &ptr,
        int value,
        RBigSize size,
        RBigSize offset = 0,
        bool sync = true
    ) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CUDA provider is not initialized"));

        if (size == 0) return;

        if (ptr.provider() != this)
            RS_THROW(RS_HostExecutionException("Cannot set memory allocated by another provider"));

        std::lock_guard<std::recursive_mutex> lock(*mutex.get());
        if (device.type == RSE_DEVICE_TYPE_CPU) {
            std::memset((ptr + offset).data, value, size);
        } else {
            int current_device;
            RcudaGetDevice(&current_device);
            if (device.id != current_device) RcudaSetDevice(device.id);

            RcudaMemsetAsync((ptr + offset).data, value, size, stream);
            ++stream_event_count;

            if (sync || RS_EnvironmentVariables::isExecForceSync()) {
                RcudaStreamSynchronize(stream);
                --stream_event_count;
            }
            if (device.id != current_device) RcudaSetDevice(current_device);
        }
    }

    void copy(
        const RS_MemoryPtr &dst,
        const RS_ConstMemoryPtr &src,
        RBigSize size,
        bool sync = true,
        bool ordered = true
    ) override {
        static std::vector<std::vector<int>> peerAccessCache;
        if (!mutex) RS_THROW(RS_UninitializedException("CUDA provider is not initialized"));

        if (size == 0) return;

        if (dst.provider() != this && src.provider() != this)
            RS_THROW(
                RS_HostExecutionException("One end of the copy must be allocated by this provider")
            );

        if ((dst.device().type != RSE_DEVICE_TYPE_CPU &&
             dst.device().type != RSE_DEVICE_TYPE_CUDA) ||
            (src.device().type != RSE_DEVICE_TYPE_CPU && src.device().type != RSE_DEVICE_TYPE_CUDA))
            RS_THROW(RS_HostExecutionException(
                fmt::format(
                    "Cannot perform copy between dst device type {} index {} and src device type "
                    "{} "
                    "index {}",
                    enumToString(dst.device().type),
                    dst.device().id,
                    enumToString(src.device().type),
                    src.device().id
                )
            ));

        std::lock_guard<std::recursive_mutex> lock(*mutex.get());

        int current_device;
        RcudaGetDevice(&current_device);

        if (device.id != current_device) RcudaSetDevice(device.id);
        if (dst.device().type == RSE_DEVICE_TYPE_CUDA &&
            src.device().type == RSE_DEVICE_TYPE_CUDA) {
            if (dst.device().id == src.device().id) {
                // Direct copy on the same device
                RcudaMemcpyAsync(dst.data, src.data, size, cudaMemcpyDeviceToDevice, stream);
                ++stream_event_count;

                if (sync || RS_EnvironmentVariables::isExecForceSync()) {
                    RcudaStreamSynchronize(stream);
                    --stream_event_count;
                }
            } else {
                int canAccessPeer;

                // Initialize peer access cache if not already done
                if (peerAccessCache.empty()) {
                    int deviceCount;
                    RcudaGetDeviceCount(&deviceCount);
                    peerAccessCache.resize(deviceCount, std::vector<int>(deviceCount, -1));
                }

                // Check if we already queried peer access for these devices
                if (peerAccessCache[dst.device().id][src.device().id] == -1) {
                    cudaDeviceCanAccessPeer(&canAccessPeer, dst.device().id, src.device().id);
                    peerAccessCache[dst.device().id][src.device().id] = canAccessPeer;

                    if (canAccessPeer) {
                        RcudaSetDevice(dst.device().id);
                        cudaDeviceEnablePeerAccess(src.device().id, 0);
                    }
                } else {
                    canAccessPeer = peerAccessCache[dst.device().id][src.device().id];
                }

                // Perform peer copy if possible, otherwise fall back to host copy
                if (canAccessPeer) {
                    RcudaSetDevice(device.id);

                    if (ordered) {
                        RcudaMemcpyPeerAsync(
                            dst.data, dst.device().id, src.data, src.device().id, size, stream
                        );
                        ++stream_event_count;

                        if (sync || RS_EnvironmentVariables::isExecForceSync()) {
                            RcudaStreamSynchronize(stream);
                            --stream_event_count;
                        }
                    } else {
                        if (dst.device() == device) {
                            RcudaMemcpyPeerAsync(
                                dst.data,
                                dst.device().id,
                                src.data,
                                src.device().id,
                                size,
                                stream_copy_to
                            );
                            ++stream_copy_to_event_count;

                            if (sync || RS_EnvironmentVariables::isExecForceSync()) {
                                RcudaStreamSynchronize(stream_copy_to);
                                --stream_copy_to_event_count;
                            }
                        } else {
                            RcudaMemcpyPeerAsync(
                                dst.data,
                                dst.device().id,
                                src.data,
                                src.device().id,
                                size,
                                stream_copy_from
                            );
                            ++stream_copy_from_event_count;

                            if (sync || RS_EnvironmentVariables::isExecForceSync()) {
                                RcudaStreamSynchronize(stream_copy_from);
                                --stream_copy_from_event_count;
                            }
                        }
                    }
                } else {
                    // Allocate temporary host buffer
                    void *hostBuffer;
                    RcudaMallocHost(&hostBuffer, size);

                    // Copy from src device to host
                    RcudaSetDevice(src.device().id);
                    RcudaMemcpy(hostBuffer, src.data, size, cudaMemcpyDeviceToHost);

                    // Copy from host to dst device
                    RcudaSetDevice(dst.device().id);
                    RcudaMemcpy(dst.data, hostBuffer, size, cudaMemcpyHostToDevice);

                    // Free temporary host buffer
                    RcudaFreeHost(hostBuffer);
                }
            }
        } else {
            // Handle other cases such as CPU to GPU or GPU to CPU copies
            cudaMemcpyKind kind;
            if (dst.device().type == RSE_DEVICE_TYPE_CPU) {
                kind = cudaMemcpyDeviceToHost;
            } else {
                kind = cudaMemcpyHostToDevice;
            }
            if (ordered) {
                RcudaMemcpyAsync(dst.data, src.data, size, kind, stream);
                ++stream_event_count;

                if (sync || RS_EnvironmentVariables::isExecForceSync()) {
                    RcudaStreamSynchronize(stream);
                    --stream_event_count;
                }
            } else {
                if (dst.device() == device) {
                    RcudaMemcpyAsync(dst.data, src.data, size, kind, stream_copy_to);
                    ++stream_copy_to_event_count;

                    if (sync || RS_EnvironmentVariables::isExecForceSync()) {
                        RcudaStreamSynchronize(stream_copy_to);
                        --stream_copy_to_event_count;
                    }
                } else {
                    RcudaMemcpyAsync(dst.data, src.data, size, kind, stream_copy_from);
                    ++stream_copy_from_event_count;

                    if (sync || RS_EnvironmentVariables::isExecForceSync()) {
                        RcudaStreamSynchronize(stream_copy_from);
                        --stream_copy_from_event_count;
                    }
                }
            }
        }
        if (device.id != current_device) RcudaSetDevice(current_device);
    }

    void
    sync(bool sync_main = true, bool sync_copy_to = true, bool sync_copy_from = true) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CUDA provider is not initialized"));

        std::lock_guard<std::recursive_mutex> lock(*mutex.get());

        if (sync_main && stream_event_count > 0) {
            RcudaStreamSynchronize(stream);
            stream_event_count = 0;
        }
        if (sync_copy_to && stream_copy_to_event_count > 0) {
            RcudaStreamSynchronize(stream_copy_to);
            stream_copy_to_event_count = 0;
        }
        if (sync_copy_from && stream_copy_from_event_count > 0) {
            RcudaStreamSynchronize(stream_copy_from);
            stream_copy_from_event_count = 0;
        }
    }

    template <typename ComputeFunction, bool Debug = false, typename... Args>
    void run(RS_MemoryProviderRunConfig config, Args... args) {
        if (!mutex) RS_THROW(RS_UninitializedException("CUDA provider is not initialized"));

        if (!config.validate())
            RS_THROW(RS_InvalidArgumentException("Invalid config: {}", config.toString()));

        if (config.thread_num == 0) return;

        ComputeFunction::validate();

        std::lock_guard<std::recursive_mutex> lock(*mutex.get());

        config.block_size =
            config.block_size == RS_AUTO_BLOCK_SIZE
                ? (device.type == RSE_DEVICE_TYPE_CUDA ? RS_DEVICE_CUDA_DEFAULT_BLOCK_SIZE
                                                       : RS_DEVICE_CPU_DEFAULT_BLOCK_SIZE)
                : config.block_size;
        config.shared_mem_size = config.shared_mem_size == 0
                                     ? ComputeFunction::sharedMemLayoutSize() * config.block_size
                                     : config.shared_mem_size;

        if constexpr (!ComputeFunction::isCudaSupported()) {
            RS_THROW(RS_HostExecutionException("Function is not supported on CUDA"));
        } else {
            if (config.shared_mem_size > 65536) {
                RS_THROW(RS_HostExecutionException(
                    fmt::format(
                        "The block size configuration {} "
                        "and block shared memory size {} "
                        "needed per thread exceeds shared memory limit 64KB",
                        config.block_size,
                        config.shared_mem_size
                    )
                ));
            }

            int current_device;
            RcudaGetDevice(&current_device);
            if (device.id != current_device) RcudaSetDevice(device.id);
            RS_Timer exec_timer;
            if constexpr (Debug) {
                globalComputeFunctionDebugRunner<ComputeFunction>
                    <<<RS_CEIL_DIV(config.thread_num, config.block_size),
                       config.block_size,
                       config.shared_mem_size,
                       stream>>>(config.thread_num, args...);
            } else {
                globalComputeFunctionRunner<ComputeFunction>
                    <<<RS_CEIL_DIV(config.thread_num, config.block_size),
                       config.block_size,
                       config.shared_mem_size,
                       stream>>>(config.thread_num, args...);
            }
            ++stream_event_count;

            if (config.sync || RS_EnvironmentVariables::isExecForceSync()) {
                CUDA_CHECK_AFTER_CALL(
                    ComputeFunction,
                    RS_CEIL_DIV(config.thread_num, config.block_size),
                    config.block_size,
                    config.shared_mem_size,
                    stream,
                    &exec_timer
                );
                --stream_event_count;
            }
            if (device.id != current_device) RcudaSetDevice(current_device);
        }
    }

    template <typename ComputeFunction, bool Debug = false, typename... Args>
    void runGrid(RS_MemoryProviderRunGridConfig config, Args... args) {
        if (!mutex) RS_THROW(RS_UninitializedException("CUDA provider is not initialized"));

        if (!config.validate())
            RS_THROW(RS_InvalidArgumentException("Invalid config: {}", config.toString()));

        RSize total_threads_per_block =
            config.block_dim_x * config.block_dim_y * config.block_dim_z;
        RSize total_blocks = config.grid_dim_x * config.grid_dim_y * config.grid_dim_z;

        if (total_threads_per_block == 0 || total_blocks == 0) return;

        ComputeFunction::validate();

        std::lock_guard<std::recursive_mutex> lock(*mutex.get());

        config.shared_mem_size =
            config.shared_mem_size == 0
                ? ComputeFunction::sharedMemLayoutSize() * total_threads_per_block
                : config.shared_mem_size;

        if constexpr (!ComputeFunction::isCudaSupported()) {
            RS_THROW(RS_HostExecutionException("Function is not supported on CUDA"));
        } else {
            if (config.shared_mem_size > 65536) {
                RS_THROW(RS_HostExecutionException(
                    fmt::format(
                        "The block dimension configuration ({}, {}, {}) "
                        "and block shared memory size {} "
                        "needed per thread exceeds shared memory limit 64KB",
                        config.block_dim_x,
                        config.block_dim_y,
                        config.block_dim_z,
                        config.shared_mem_size
                    )
                ));
            }

            int current_device;
            RcudaGetDevice(&current_device);
            if (device.id != current_device) RcudaSetDevice(device.id);

            dim3 grid_dim(config.grid_dim_x, config.grid_dim_y, config.grid_dim_z);
            dim3 block_dim(config.block_dim_x, config.block_dim_y, config.block_dim_z);

            RS_Timer exec_timer;
            globalComputeFunctionGridRunner<ComputeFunction>
                <<<grid_dim, block_dim, config.shared_mem_size, stream>>>(args...);

            ++stream_event_count;

            if (config.sync || RS_EnvironmentVariables::isExecForceSync()) {
                CUDA_CHECK_AFTER_CALL(
                    ComputeFunction,
                    fmt::format(
                        "({}, {}, {})", config.grid_dim_x, config.grid_dim_y, config.grid_dim_z
                    ),
                    fmt::format(
                        "({}, {}, {})", config.block_dim_x, config.block_dim_y, config.block_dim_z
                    ),
                    config.shared_mem_size,
                    stream,
                    &exec_timer
                );
                --stream_event_count;
            }
            if (device.id != current_device) RcudaSetDevice(current_device);
        }
    }

    template <typename... Args>
    void capture(const char *key, Args &&...args) {
        static_assert(sizeof...(Args) >= 1, "capture requires at least a lambda");
        captureImpl(
            key,
            std::forward_as_tuple(std::forward<Args>(args)...),
            std::make_index_sequence<sizeof...(Args) - 1>{}
        );
    }

   private:
    template <typename Tuple, std::size_t... Is>
    void captureImpl(const char *key, Tuple &&all, std::index_sequence<Is...>) {
        if (!mutex) RS_THROW(RS_UninitializedException("CUDA provider is not initialized"));

        std::lock_guard<std::recursive_mutex> lock(*mutex.get());

        auto deps = std::make_tuple(std::get<Is>(all)...);
        auto &&fn = std::get<sizeof...(Is)>(all);

        using DepTuple = decltype(deps);
        using Entry = RS_CudaGraphCacheEntry<DepTuple>;
        auto &slot = capture_cache[key];
        auto *entry = static_cast<Entry *>(slot.get());

        if (entry && entry->valid && entry->stored_deps == deps) {
            // Replay cached graph
            int current_device;
            RcudaGetDevice(&current_device);
            if (device.id != current_device) RcudaSetDevice(device.id);
            RcudaGraphLaunch(entry->exec, stream);
            ++stream_event_count;
            if (device.id != current_device) RcudaSetDevice(current_device);
        } else {
            // Allocate new entry if needed (destroys old graph via unique_ptr)
            slot = std::make_unique<Entry>();
            entry = static_cast<Entry *>(slot.get());

            int current_device;
            RcudaGetDevice(&current_device);
            if (device.id != current_device) RcudaSetDevice(device.id);

            RcudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            fn();
            RcudaStreamEndCapture(stream, &entry->graph);
            RcudaGraphInstantiate(&entry->exec, entry->graph, 0);
            entry->stored_deps = std::move(deps);
            entry->valid = true;

            if (device.id != current_device) RcudaSetDevice(current_device);
        }
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_CUDA_H
