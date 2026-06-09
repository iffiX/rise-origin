// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_DEF_H
#define RS_DEF_H

// R is a short form for RS
#include <cstdint>
#include <limits>
#include <unordered_map>

#include "rs_conf.h"

#define RS_STRINGIFY(x) #x
#define RS_TOSTRING(x) RS_STRINGIFY(x)
#define RS_EXPAND(x) x

#define RS_GET_MACRO( \
    _1,               \
    _2,               \
    _3,               \
    _4,               \
    _5,               \
    _6,               \
    _7,               \
    _8,               \
    _9,               \
    _10,              \
    _11,              \
    _12,              \
    _13,              \
    _14,              \
    _15,              \
    _16,              \
    _17,              \
    _18,              \
    _19,              \
    _20,              \
    _21,              \
    _22,              \
    _23,              \
    _24,              \
    _25,              \
    _26,              \
    _27,              \
    _28,              \
    _29,              \
    _30,              \
    _31,              \
    _32,              \
    NAME,             \
    ...               \
)                     \
    NAME

// Macro to apply another macro to each argument
#define RS_FOREACH_1(macro, a) macro a
#define RS_FOREACH_2(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_1(macro, __VA_ARGS__))
#define RS_FOREACH_3(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_2(macro, __VA_ARGS__))
#define RS_FOREACH_4(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_3(macro, __VA_ARGS__))
#define RS_FOREACH_5(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_4(macro, __VA_ARGS__))
#define RS_FOREACH_6(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_5(macro, __VA_ARGS__))
#define RS_FOREACH_7(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_6(macro, __VA_ARGS__))
#define RS_FOREACH_8(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_7(macro, __VA_ARGS__))
#define RS_FOREACH_9(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_8(macro, __VA_ARGS__))
#define RS_FOREACH_10(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_9(macro, __VA_ARGS__))
#define RS_FOREACH_11(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_10(macro, __VA_ARGS__))
#define RS_FOREACH_12(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_11(macro, __VA_ARGS__))
#define RS_FOREACH_13(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_12(macro, __VA_ARGS__))
#define RS_FOREACH_14(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_13(macro, __VA_ARGS__))
#define RS_FOREACH_15(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_14(macro, __VA_ARGS__))
#define RS_FOREACH_16(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_15(macro, __VA_ARGS__))
#define RS_FOREACH_17(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_16(macro, __VA_ARGS__))
#define RS_FOREACH_18(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_17(macro, __VA_ARGS__))
#define RS_FOREACH_19(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_18(macro, __VA_ARGS__))
#define RS_FOREACH_20(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_19(macro, __VA_ARGS__))
#define RS_FOREACH_21(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_20(macro, __VA_ARGS__))
#define RS_FOREACH_22(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_21(macro, __VA_ARGS__))
#define RS_FOREACH_23(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_22(macro, __VA_ARGS__))
#define RS_FOREACH_24(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_23(macro, __VA_ARGS__))
#define RS_FOREACH_25(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_24(macro, __VA_ARGS__))
#define RS_FOREACH_26(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_25(macro, __VA_ARGS__))
#define RS_FOREACH_27(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_26(macro, __VA_ARGS__))
#define RS_FOREACH_28(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_27(macro, __VA_ARGS__))
#define RS_FOREACH_29(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_28(macro, __VA_ARGS__))
#define RS_FOREACH_30(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_29(macro, __VA_ARGS__))
#define RS_FOREACH_31(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_30(macro, __VA_ARGS__))
#define RS_FOREACH_32(macro, a, ...) macro a RS_EXPAND(RS_FOREACH_31(macro, __VA_ARGS__))

// Select the correct macro based on the number of arguments
#define RS_MACRO_FOREACH(macro, ...) \
    RS_EXPAND(RS_GET_MACRO(          \
        __VA_ARGS__,                 \
        RS_FOREACH_32, /**/          \
        RS_FOREACH_31,               \
        RS_FOREACH_30,               \
        RS_FOREACH_29,               \
        RS_FOREACH_28,               \
        RS_FOREACH_27,               \
        RS_FOREACH_26,               \
        RS_FOREACH_25,               \
        RS_FOREACH_24, /**/          \
        RS_FOREACH_23,               \
        RS_FOREACH_22,               \
        RS_FOREACH_21,               \
        RS_FOREACH_20,               \
        RS_FOREACH_19,               \
        RS_FOREACH_18,               \
        RS_FOREACH_17,               \
        RS_FOREACH_16, /**/          \
        RS_FOREACH_15,               \
        RS_FOREACH_14,               \
        RS_FOREACH_13,               \
        RS_FOREACH_12,               \
        RS_FOREACH_11,               \
        RS_FOREACH_10,               \
        RS_FOREACH_9,                \
        RS_FOREACH_8, /**/           \
        RS_FOREACH_7,                \
        RS_FOREACH_6,                \
        RS_FOREACH_5,                \
        RS_FOREACH_4,                \
        RS_FOREACH_3,                \
        RS_FOREACH_2,                \
        RS_FOREACH_1                 \
    )(macro, __VA_ARGS__))

#define RS_EXPAND_MULTI_ARGS(...) __VA_ARGS__
#define RS_EXPAND_TUPLE(tuple) RS_EXPAND_MULTI_ARGS tuple

#define RS_CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define RS_FLOOR_DIV(x, y) ((x) / (y))
#define RS_ALIGN(x, y) (RS_CEIL_DIV(x, y) * y)
#define RS_MIN(x, y) ((x) < (y) ? (x) : (y))
#define RS_MAX(x, y) ((x) > (y) ? (x) : (y))
#define RS_BOUND(x, min, max) (RS_MAX(RS_MIN(x, max), min))
#define RS_ABS(x) ((x) < 0 ? -(x) : (x))

#define RS_LOG(logger, level, ...) \
    if (logger) logger->level(__VA_ARGS__)

#define RS_PI 3.14159265358979
#define RS_2PI 6.28318530717958
#define RS_E 2.71828182845905
#define RS_EPS 1e-6
#define RS_EPS2 1e-12

#define RS_ENUM_ENTRY(name, value) name = value,
#define RS_ENUM_CASE_ENTRY(name, value) \
    case name:                          \
        return #name;
#define RS_ENUM_CHECK_ENTRY(name, value) \
    case name:                           \
        return true;

#define RS_ENUM(name, type, ...)                                       \
    enum name : type { RS_MACRO_FOREACH(RS_ENUM_ENTRY, __VA_ARGS__) }; \
                                                                       \
    inline const char *enumToString(name value) {                      \
        switch (value) {                                               \
            RS_MACRO_FOREACH(RS_ENUM_CASE_ENTRY, __VA_ARGS__)          \
            default:                                                   \
                return "Unknown";                                      \
        }                                                              \
    }                                                                  \
    inline bool isValidEnum(name value) {                              \
        switch (value) {                                               \
            RS_MACRO_FOREACH(RS_ENUM_CASE_ENTRY, __VA_ARGS__)          \
            default:                                                   \
                return false;                                          \
        }                                                              \
    }

#define RFLOAT(x) static_cast<RFloat>(x)
#define RINDEX(x) static_cast<RIndex>(x)
#define RSIZE(x) static_cast<RSize>(x)
#define RSIGNED_INDEX(x) static_cast<RSignedIndex>(x)
#define RSIGNED_SIZE(x) static_cast<RSignedSize>(x)
#define RHANDLE(x) static_cast<RHandle>(x)

#define RBIG_FLOAT(x) static_cast<RBigFloat>(x)
#define RBIG_INDEX(x) static_cast<RBigIndex>(x)
#define RBIG_SIZE(x) static_cast<RBigSize>(x)
#define RBIG_SIGNED_INDEX(x) static_cast<RBigSignedIndex>(x)
#define RBIG_SIGNED_SIZE(x) static_cast<RBigSignedSize>(x)
#define RBIG_HANDLE(x) static_cast<RBigHandle>(x)

#if defined(__GNUC__) || defined(__clang__)
// For GCC and Clang
#define RS_CURRENT_FUNCTION_NAME() __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
// For Microsoft Visual C++
#define RS_CURRENT_FUNCTION_NAME() __FUNCSIG__
#elif defined(__INTEL_COMPILER)
// For Intel C++ Compiler
#define RS_CURRENT_FUNCTION_NAME() __PRETTY_FUNCTION__
#else
// Fallback for other compilers
#define RS_CURRENT_FUNCTION_NAME() __func__
#endif

#define RS_AUTO_BLOCK_SIZE 0

namespace rise {
/**
 * Note: Normal data formats are usually used in simulation memory units.
 * They are used for efficiency purposes.
 */
//! Used for general float input/output, not used for computation
using RFloat = float;
//! Used for representing non-negative continuous index, companion of RSize
using RIndex = unsigned int;
//! Used for representing non-negative size, companion of RIndex
using RSize = RIndex;
//! Used for representing a positive/negative index, usually for coordinate
using RSignedIndex = int;
//! Used for representing a positive/negative size, usually for offset
using RSignedSize = RSignedIndex;
//! Used for representing a handle to an object, in array, SoA, etc
using RHandle = unsigned int;

/**
 * Note: big data formats are usually not used in simulation memory units.
 * They are used for identifying objects in the simulation, or records.
 */
//! Used for representing a big float, like time point
using RBigFloat = double;
//! Used for representing a big non-negative continuous index, companion of RBigSize
using RBigIndex = unsigned long long;
//! Used for representing a big non-negative size, companion of RBigIndex
using RBigSize = RBigIndex;
//! Used for representing a big positive/negative index, usually for coordinate
using RBigSignedIndex = long long;
//! Used for representing a big positive/negative size, usually for offset
using RBigSignedSize = RBigSignedIndex;
//! Used for representing a big handle to an object, in array, SoA, etc
using RBigHandle = unsigned long long;

/** 
 * Special data types
 */
//! Used for representing a byte, usually for binary data
using RByte = uint8_t;

//! Used for representing a state, like a link state
using RState = unsigned int;

constexpr RSize RS_MAX_SIZE = std::numeric_limits<RSize>::max();
constexpr RIndex RS_NULL_INDEX = std::numeric_limits<RIndex>::max();
constexpr RHandle RS_NULL_HANDLE = std::numeric_limits<RHandle>::max();
constexpr RSignedIndex RS_NULL_SIGNED_INDEX = std::numeric_limits<RSignedIndex>::max();
constexpr RBigSize RS_MAX_BIG_SIZE = std::numeric_limits<RBigSize>::max();
constexpr RBigIndex RS_NULL_BIG_INDEX = std::numeric_limits<RBigIndex>::max();
constexpr RBigSignedIndex RS_NULL_BIG_SIGNED_INDEX = std::numeric_limits<RBigSignedIndex>::max();
constexpr RBigHandle RS_NULL_BIG_HANDLE = std::numeric_limits<RBigHandle>::max();

constexpr RFloat RS_MAX_FLOAT = std::numeric_limits<RFloat>::max();
constexpr RFloat RS_MIN_FLOAT = std::numeric_limits<RFloat>::lowest();
constexpr RBigFloat RS_MAX_BIG_FLOAT = std::numeric_limits<RBigFloat>::max();
constexpr RBigFloat RS_MIN_BIG_FLOAT = std::numeric_limits<RBigFloat>::lowest();

struct RS_NoneType {};
inline constexpr RS_NoneType RS_NONE{};

#ifdef __CUDACC__
#define GLOBAL_QUALIFIER __global__
#define HOST_DEVICE_QUALIFIER __host__ __device__
#define HOST_QUALIFIER __host__
#define DEVICE_QUALIFIER __device__
#define HOST_DEVICE_INLINE_QUALIFIER __host__ __device__ __forceinline__
#define HOST_INLINE_QUALIFIER __host__ __forceinline__
#define DEVICE_INLINE_QUALIFIER __device__ __forceinline__

#else
#define GLOBAL_QUALIFIER
#define HOST_DEVICE_QUALIFIER
#define HOST_QUALIFIER
#define DEVICE_QUALIFIER
#define HOST_DEVICE_INLINE_QUALIFIER inline
#define HOST_INLINE_QUALIFIER inline
#define DEVICE_INLINE_QUALIFIER inline
#endif

}  // namespace rise

#endif  // RS_DEF_H