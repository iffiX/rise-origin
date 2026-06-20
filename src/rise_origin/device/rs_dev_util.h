// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_DEV_UTIL_H
#define RS_DEV_UTIL_H

#include <cassert>
#include <type_traits>

#include "rise_origin/utils/meta/rs_constexpr.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/meta/rs_num_traits.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

namespace device {

/**
 * @brief RS_DivMod is a helper class for integer division and modulo operation.
 * There is a fast version for int type and a slow version for other type.
 *
 * @note Adapted from
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/
 * core/providers/cuda/shared_inc/fast_divmod.h
 */

template <typename T>
class RS_DivMod {
   public:
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_DivMod(T d = 1) { setDivisor(d); }

    HOST_DEVICE_INLINE_QUALIFIER void setDivisor(T d) {
        divisor = d == 0 ? 1 : d;
        assert(divisor >= 1 && divisor <= RS_NumTraits<T>::highest());
    }

    HOST_DEVICE_INLINE_QUALIFIER T div(T n) const { return n / divisor; }

    HOST_DEVICE_INLINE_QUALIFIER T mod(T n) const { return n % divisor; }

    HOST_DEVICE_INLINE_QUALIFIER void divmod(T n, T& q, T& r) const {
        q = div(n);
        r = n - q * divisor;
    }

    T divisor = 1;
};

/**
 * @brief The code below is based on section 4 Unsigned division of paper
 * https://gmplib.org/~tege/divcnst-pldi94.pdf The advantage is that div
 * is very simple, then GPU compiler can do loop unroll easily
 * when divmod is called in a loop.
 *
 * @note Adapted from
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/
 * core/providers/cuda/shared_inc/fast_divmod.h
 */
template <>
class RS_DivMod<uint32_t> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_DivMod(uint32_t d = 1) { setDivisor(d); };

    HOST_DEVICE_INLINE_QUALIFIER void setDivisor(uint32_t d) {
        divisor = d == 0 ? 1 : d;
        assert(divisor >= 1 && divisor <= RS_NumTraits<uint32_t>::highest());

        for (l_ = 0; l_ < 32; l_++)
            if ((1U << l_) >= divisor) break;

        uint64_t one = 1;
        uint64_t m = ((one << 32) * ((one << l_) - divisor)) / divisor + 1;
        M_ = static_cast<uint32_t>(m);
        // according to paper, the value of m' should fit in a unsigned integer.
        assert(M_ > 0 && M_ == m);
    }

    HOST_DEVICE_INLINE_QUALIFIER uint32_t div(uint32_t n) const {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
        uint64_t t = __umulhi(M_, n);
        return (t + n) >> l_;
#else
        // Using uint64_t for t, then t + n won't overflow.
        uint64_t t = (static_cast<uint64_t>(M_) * n) >> 32;
        return static_cast<uint32_t>((t + n) >> l_);
#endif
    }

    HOST_DEVICE_INLINE_QUALIFIER uint32_t mod(uint32_t n) const { return n - div(n) * divisor; }

    HOST_DEVICE_INLINE_QUALIFIER void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
        q = div(n);
        r = n - q * divisor;
    }

    HOST_DEVICE_INLINE_QUALIFIER void muladd(uint32_t& n, uint32_t q, uint32_t r) const {
        n = q * divisor + r;
    }

    uint32_t divisor = 1;  // divisor

   private:
    uint32_t M_ = 0;  // m' in the paper.
    uint32_t l_ = 0;  // l_ = ceil(log2(divisor))
};

/**
 * @brief A compile-time version of the fast DivMod technique.
 *        We only use the fast path if (D <= 2^32 - 1). Otherwise,
 *        we fall back to a normal division/modulo.
 */
template <uint64_t D>
class RS_StaticDivMod {
   public:
    // The actual divisor we will store (avoid zero).
    static constexpr uint64_t divisor = (D == 0) ? 1ULL : D;

    template <typename T>
    static constexpr T div(T n) {
        if (IsPowerOfTwo_) {
            return n >> ShiftAmount_;
        } else if constexpr (std::is_same_v<T, uint32_t> && IsFastMethodUsable_) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
            uint64_t t = __umulhi(M_, n);
            return (t + n) >> l_;
#else
            // Using uint64_t for t, then t + n won't overflow.
            uint64_t t = (static_cast<uint64_t>(M_) * n) >> 32;
            return static_cast<uint32_t>((t + n) >> l_);
#endif
        } else if (n <= static_cast<uint64_t>(RS_NumTraits<uint32_t>::highest()) &&
                   IsFastMethodUsable_) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
            uint64_t t = __umulhi(M_, n);
            return (t + n) >> l_;
#else
            uint64_t t = (static_cast<uint64_t>(M_) * n) >> 32;
            return static_cast<uint32_t>((t + n) >> l_);
#endif
        } else {
            return n / divisor;
        }
    }

    template <typename T>
    static constexpr T mod(T n) {
        if (IsPowerOfTwo_) {
            return n & (divisor - 1);
        } else if constexpr (std::is_same_v<T, uint32_t> && IsFastMethodUsable_) {
            return n - div(n) * divisor;
        } else if (n <= static_cast<uint64_t>(RS_NumTraits<uint32_t>::highest()) &&
                   IsFastMethodUsable_) {
            return n - div(n) * divisor;
        } else {
            return n % divisor;
        }
    }

    template <typename T>
    static constexpr void divmod(T n, T& q, T& r) {
        if (IsPowerOfTwo_) {
            q = n >> ShiftAmount_;
            r = n & (divisor - 1);
        } else {
            q = div(n);
            r = n - q * divisor;
        }
    }

    template <typename T>
    static constexpr void muladd(T& n, T q, T r) {
        if (IsPowerOfTwo_) {
            n = (q << ShiftAmount_) + r;
        } else {
            n = q * divisor + r;
        }
    }

   private:
    // Decide at compile time whether to use the "fast" 32-bit logic.
    // The technique from the paper requires the divisor to fit in 32 bits.
    static constexpr bool IsPowerOfTwo_ = isPowerOfTwo(divisor);
    static constexpr uint32_t ShiftAmount_ = IsPowerOfTwo_ ? logTwoOf(divisor) : 0;

    static constexpr bool IsFastMethodUsable_ =
        (divisor <= static_cast<uint64_t>(RS_NumTraits<uint32_t>::highest()));

    // Compute l_ (ceil(log2(divisor))) for the 32-bit fast path.
    static constexpr uint32_t _computeL() {
        uint32_t l = 0;
        while ((1ULL << l) < divisor && l < 32) {
            ++l;
        }
        return l;
    }

    // Compute M_ from the paper for the 32-bit fast path.
    static constexpr uint32_t _computeM() {
        constexpr uint64_t One = 1ULL;
        constexpr uint32_t l_ = _computeL();

        // Guarding the division to assure the compiler it's always valid
        static_assert(divisor != 0, "Divisor must not be zero.");

        // Forcefully cast to unsigned 128-bit if available (GCC/Clang)
        // This prevents overflow and ensures the division is constexpr
        constexpr uint64_t numerator = (One << 32) * ((One << l_) - divisor);
        constexpr uint64_t m = numerator / divisor + One;

        // Ensure the result fits in a 32-bit unsigned integer
        static_assert(
            m <= RS_NumTraits<uint32_t>::highest(), "Fast-div constant M_ overflows 32 bits!"
        );
        return static_cast<uint32_t>(m);
    }

    // We only actually *use* M_ and l_ if we're on the fast path
    static constexpr uint32_t l_ = IsFastMethodUsable_ ? _computeL() : 0;
    static constexpr uint32_t M_ = IsFastMethodUsable_ ? _computeM() : 0;
};

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER void staticAssertAtomicTypeValid() {
    static_assert(std::is_scalar<T>::value, "Device atomic operations only support scalar types.");
    static_assert(
        sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
        "Device atomic operations only support types of 1, 2, 4, or 8 bytes in size."
    );
    static_assert(
        alignof(T) == sizeof(T),
        "Device atomic operations needs alignment of T to match its size (i.e., 1, 2, 4, or 8 "
        "bytes)."
    );
}

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HOST_DEVICE_INLINE_QUALIFIER T deviceAtomicAdd(T* address, T val) {
    staticAssertAtomicTypeValid<T>();
#ifdef __CUDA_ARCH__
    return atomicAdd(address, val);
#else
    std::atomic<T>* atomic_address = reinterpret_cast<std::atomic<T>*>(address);
    T old_val = atomic_address->load();
    T new_val;
    do {
        new_val = old_val + val;
    } while (!atomic_address->compare_exchange_weak(old_val, new_val));
    return old_val;
#endif
}

/**
 * @brief Atomic add for vector 2D.
 * @note This atomic add ensures each component add is atomic, but not the vector as a whole.
 */
template <typename T, typename U>
HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D<T> deviceAtomicAdd(RS_Vec2D<T>* address, RS_Vec2D<U> val) {
    return {
        deviceAtomicAdd(&address->x, static_cast<T>(val.x)),
        deviceAtomicAdd(&address->y, static_cast<T>(val.y))
    };
}

/**
 * @brief Atomic add for vector 3D.
 * @note This atomic add ensures each component add is atomic, but not the vector as a whole.
 */
template <typename T, typename U>
HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> deviceAtomicAdd(RS_Vec3D<T>* address, RS_Vec3D<U> val) {
    return {
        deviceAtomicAdd(&address->x, static_cast<T>(val.x)),
        deviceAtomicAdd(&address->y, static_cast<T>(val.y)),
        deviceAtomicAdd(&address->z, static_cast<T>(val.z))
    };
}

/**
 * @brief Atomic add for vector 4D.
 * @note This atomic add ensures each component add is atomic, but not the vector as a whole.
 */
template <typename T, typename U>
HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D<T> deviceAtomicAdd(RS_Vec4D<T>* address, RS_Vec4D<U> val) {
    return {
        deviceAtomicAdd(&address->x, static_cast<T>(val.x)),
        deviceAtomicAdd(&address->y, static_cast<T>(val.y)),
        deviceAtomicAdd(&address->z, static_cast<T>(val.z)),
        deviceAtomicAdd(&address->w, static_cast<T>(val.w))
    };
}

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HOST_DEVICE_INLINE_QUALIFIER T deviceAtomicSub(T* address, T val) {
    staticAssertAtomicTypeValid<T>();
#ifdef __CUDA_ARCH__
    return atomicSub(address, val);
#else
    std::atomic<T>* atomic_address = reinterpret_cast<std::atomic<T>*>(address);
    T old_val = atomic_address->load();
    T new_val;
    do {
        new_val = old_val - val;
    } while (!atomic_address->compare_exchange_weak(old_val, new_val));
    return old_val;
#endif
}

template <typename T, typename U>
HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D<T> deviceAtomicSub(RS_Vec2D<T>* address, RS_Vec2D<U> val) {
    return {
        deviceAtomicSub(&address->x, static_cast<T>(val.x)),
        deviceAtomicSub(&address->y, static_cast<T>(val.y))
    };
}

template <typename T, typename U>
HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> deviceAtomicSub(RS_Vec3D<T>* address, RS_Vec3D<U> val) {
    return {
        deviceAtomicSub(&address->x, static_cast<T>(val.x)),
        deviceAtomicSub(&address->y, static_cast<T>(val.y)),
        deviceAtomicSub(&address->z, static_cast<T>(val.z))
    };
}

template <typename T, typename U>
HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D<T> deviceAtomicSub(RS_Vec4D<T>* address, RS_Vec4D<U> val) {
    return {
        deviceAtomicSub(&address->x, static_cast<T>(val.x)),
        deviceAtomicSub(&address->y, static_cast<T>(val.y)),
        deviceAtomicSub(&address->z, static_cast<T>(val.z)),
        deviceAtomicSub(&address->w, static_cast<T>(val.w))
    };
}

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER T deviceAtomicOr(T* address, T val) {
    staticAssertAtomicTypeValid<T>();
#ifdef __CUDA_ARCH__
    return atomicOr(address, val);
#else
    std::atomic<T>* atomic_address = reinterpret_cast<std::atomic<T>*>(address);
    T old_val = atomic_address->load();
    T new_val;
    do {
        new_val = old_val | val;
    } while (!atomic_address->compare_exchange_weak(old_val, new_val));
    return old_val;
#endif
}

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER T deviceAtomicAnd(T* address, T val) {
    staticAssertAtomicTypeValid<T>();
#ifdef __CUDA_ARCH__
    return atomicAnd(address, val);
#else
    std::atomic<T>* atomic_address = reinterpret_cast<std::atomic<T>*>(address);
    T old_val = atomic_address->load();
    T new_val;
    do {
        new_val = old_val & val;
    } while (!atomic_address->compare_exchange_weak(old_val, new_val));
    return old_val;
#endif
}

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER T deviceAtomicXor(T* address, T val) {
    staticAssertAtomicTypeValid<T>();
#ifdef __CUDA_ARCH__
    return atomicXor(address, val);
#else
    std::atomic<T>* atomic_address = reinterpret_cast<std::atomic<T>*>(address);
    T old_val = atomic_address->load();
    T new_val;
    do {
        new_val = old_val ^ val;
    } while (!atomic_address->compare_exchange_weak(old_val, new_val));
    return old_val;
#endif
}

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER T deviceAtomicExchange(T* address, T val) {
    staticAssertAtomicTypeValid<T>();
#ifdef __CUDA_ARCH__
    return atomicExch(address, val);
#else
    std::atomic<T>* atomic_address = reinterpret_cast<std::atomic<T>*>(address);
    return atomic_exchange(atomic_address, val);
#endif
}

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER T deviceAtomicCAS(T* address, T compare, T val) {
    staticAssertAtomicTypeValid<T>();
#ifdef __CUDA_ARCH__
    return atomicCAS(address, compare, val);
#else
    std::atomic<T>* atomic_address = reinterpret_cast<std::atomic<T>*>(address);
    atomic_compare_exchange_strong(atomic_address, &compare, val);
    return compare;
#endif
}

/// @brief Returns the index of the lowest, i.e. least significant, on bit in the specified 32 bit
/// word
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
/// @note Adapted from NanoVDB https://developer.nvidia.com/nanovdb
HOST_DEVICE_INLINE_QUALIFIER uint32_t findLowestOnFor32(uint32_t v) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return __ffs(v) - 1;  // one based indexing
#elif defined(_MSC_VER) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    unsigned long index;
    _BitScanForward(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return static_cast<uint32_t>(__builtin_ctzl(v));
#else
    // NANO_WARNING("Using software implementation for findLowestOn(uint32_t v)")
    static const unsigned char DeBruijn[32] = {0,  1,  28, 2,  29, 14, 24, 3,  30, 22, 20,
                                               15, 25, 17, 4,  8,  31, 27, 13, 23, 21, 19,
                                               16, 7,  26, 12, 18, 6,  11, 5,  10, 9};
// disable unary minus on unsigned warning
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return DeBruijn[uint32_t((v & -v) * 0x077CB531U) >> 27];
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(pop)
#endif

#endif
}

/// @brief Returns the index of the lowest, i.e. least significant, on bit in the specified 64 bit
/// word
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint64_t(0)!
/// @note Adapted from NanoVDB https://developer.nvidia.com/nanovdb
HOST_DEVICE_INLINE_QUALIFIER uint32_t findLowestOnFor64(uint64_t v) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return __ffsll(static_cast<unsigned long long int>(v)) - 1;  // one based indexing
#elif defined(_MSC_VER) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    unsigned long index;
    _BitScanForward64(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return static_cast<uint32_t>(__builtin_ctzll(v));
#else
    // NANO_WARNING("Using software implementation for util::findLowestOn(uint64_t)")
    static const unsigned char DeBruijn[64] = {
        0,  1,  2,  53, 3,  7,  54, 27, 4,  38, 41, 8,  34, 55, 48, 28, 62, 5,  39, 46, 44, 42,
        22, 9,  24, 35, 59, 56, 49, 18, 29, 11, 63, 52, 6,  26, 37, 40, 33, 47, 61, 45, 43, 21,
        23, 58, 17, 10, 51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
    };
// disable unary minus on unsigned warning
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return DeBruijn[uint64_t((v & -v) * UINT64_C(0x022FDD63CC95386D)) >> 58];
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(pop)
#endif

#endif
}

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER T findLowestOn(T v) {
    if constexpr (sizeof(T) == 4)
        return findLowestOnFor32(v);
    else
        return findLowestOnFor64(v);
}

/// @brief Returns the index of the highest, i.e. most significant, on bit in the specified 32
/// bit word
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
/// @note Adapted from NanoVDB https://developer.nvidia.com/nanovdb
HOST_DEVICE_INLINE_QUALIFIER uint32_t findHighestOnFor32(uint32_t v) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return sizeof(uint32_t) * 8 - 1 -
           __clz(v);  // Return the number of consecutive high-order zero bits in a 32-bit integer.
#elif defined(_MSC_VER) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    unsigned long index;
    _BitScanReverse(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return sizeof(unsigned long) * 8 - 1 - __builtin_clzl(v);
#else
    // NANO_WARNING("Using software implementation for util::findHighestOn(uint32_t)")
    static const unsigned char DeBruijn[32] = {0,  9,  1,  10, 13, 21, 2,  29, 11, 14, 16,
                                               18, 22, 25, 3,  30, 8,  12, 20, 28, 15, 17,
                                               24, 7,  19, 27, 23, 6,  26, 5,  4,  31};
    v |= v >> 1;  // first round down to one less than a power of 2
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return DeBruijn[uint32_t(v * 0x07C4ACDDU) >> 27];
#endif
}

/// @brief Returns the index of the highest, i.e. most significant, on bit in the specified 64
/// bit word.
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint64_t(0)!
/// @note Adapted from NanoVDB https://developer.nvidia.com/nanovdb
HOST_DEVICE_INLINE_QUALIFIER uint32_t findHighestOnFor64(uint64_t v) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return sizeof(unsigned long) * 8 - 1 - __clzll(static_cast<unsigned long long int>(v));
#elif defined(_MSC_VER) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    unsigned long index;
    _BitScanReverse64(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return sizeof(uint64_t) * 8 - 1 - __builtin_clzll(v);
#else
    const uint32_t* p = reinterpret_cast<const uint32_t*>(&v);
    return p[1] ? 32u + findHighestOnFor32(p[1]) : findHighestOnFor32(p[0]);
#endif
}

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER T findHighestOn(T v) {
    if constexpr (sizeof(T) == 4)
        return findHighestOnFor32(v);
    else
        return findHighestOnFor64(v);
}

/// @brief Returns the number of leading zeros in the specified 32-bit word
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
/// @note Adapted from NanoVDB https://developer.nvidia.com/nanovdb
HOST_DEVICE_INLINE_QUALIFIER uint32_t countLeadingZerosFor32(uint32_t v) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return __clz(v);  // Return the number of consecutive high-order zero bits in a 32-bit integer.
#elif defined(_MSC_VER) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    unsigned long index;
    _BitScanReverse(&index, v);
    return v == 0 ? 32u : static_cast<uint32_t>(sizeof(uint32_t) * 8 - 1 - index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return v == 0 ? 32u : __builtin_clzl(v);
#else
    // NANO_WARNING("Using software implementation for util::countLeadingZeros(uint32_t)")
    static const unsigned char DeBruijn[32] = {0,  9,  1,  10, 13, 21, 2,  29, 11, 14, 16,
                                               18, 22, 25, 3,  30, 8,  12, 20, 28, 15, 17,
                                               24, 7,  19, 27, 23, 6,  26, 5,  4,  31};
    v |= v >> 1;  // first round down to one less than a power of 2
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v == 0 ? 32u : sizeof(uint32_t) * 8 - 1 - DeBruijn[uint32_t(v * 0x07C4ACDDU) >> 27];
#endif
}

/// @brief Returns the number of leading zeros in the specified 64-bit word
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint64_t(0)!
/// @note Adapted from NanoVDB https://developer.nvidia.com/nanovdb
HOST_DEVICE_INLINE_QUALIFIER uint32_t countLeadingZerosFor64(uint64_t v) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return __clzll(static_cast<unsigned long long int>(v));
#elif defined(_MSC_VER) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    unsigned long index;
    _BitScanReverse64(&index, v);
    return v == 0 ? 64u : static_cast<uint32_t>(sizeof(uint64_t) * 8 - 1 - index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return v == 0 ? 64u : __builtin_clzll(v);
#else
    const uint32_t* p = reinterpret_cast<const uint32_t*>(&v);
    return p[0] ? countLeadingZerosFor32(p[0]) : 32u + countLeadingZerosFor32(p[1]);
#endif
}

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER T countLeadingZeros(T v) {
    if constexpr (sizeof(T) == 4)
        return countLeadingZerosFor32(v);
    else
        return countLeadingZerosFor64(v);
}

/// @return Number of bits that are on in the specified 32-bit word
HOST_DEVICE_INLINE_QUALIFIER uint32_t countOnBitsFor32(uint32_t v) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return __popc(v);
#elif defined(_MSC_VER) && defined(_M_X64) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return __popcnt(v);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return __builtin_popcount(v);
#else  // use software implementation
    v = v - ((v >> 1) & uint32_t(0x55555555));
    v = (v & uint32_t(0x33333333)) + ((v >> 2) & uint32_t(0x33333333));
    v = (v + (v >> 4)) & uint32_t(0x0F0F0F0F);
    v *= 0x01010101;
    return v >> 24;
#endif
}

/// @return Number of bits that are on in the specified 64-bit word
/// @note Adapted from NanoVDB https://developer.nvidia.com/nanovdb
HOST_DEVICE_INLINE_QUALIFIER uint32_t countOnBitsFor64(uint64_t v) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return __popcll(v);
    // __popcnt64 intrinsic support was added in VS 2019 16.8
#elif defined(_MSC_VER) && defined(_M_X64) && (_MSC_VER >= 1928) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return uint32_t(__popcnt64(v));
#elif (defined(__GNUC__) || defined(__clang__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
    return __builtin_popcountll(v);
#else  // use software implementation
    v = v - ((v >> 1) & uint64_t(0x5555555555555555));
    v = (v & uint64_t(0x3333333333333333)) + ((v >> 2) & uint64_t(0x3333333333333333));
    return (((v + (v >> 4)) & uint64_t(0xF0F0F0F0F0F0F0F)) * uint64_t(0x101010101010101)) >> 56;
#endif
}

template <typename T>
HOST_DEVICE_INLINE_QUALIFIER uint32_t countOnBits(T v) {
    if constexpr (sizeof(T) == 4)
        return countOnBitsFor32(v);
    else
        return countOnBitsFor64(v);
}

}  // namespace device

}  // namespace rise

#endif  // RS_DEV_UTIL_H
