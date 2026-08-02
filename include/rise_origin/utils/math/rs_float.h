// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSE_FLOAT_H
#define RS_RSE_FLOAT_H

#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"

namespace rise {

//! Defines the type of a floating point number.
RS_ENUM(RSE_FloatType, unsigned int, (RSE_FLOAT_16, 0), (RSE_FLOAT_32, 1), (RSE_FLOAT_64, 2))

static constexpr RSE_FloatType RS_DEFAULT_FLOAT_TYPE = RSE_FLOAT_32;

template <typename T>
struct RS_FloatIntrinsics;

template <>
struct RS_FloatIntrinsics<float> {
    HOST_DEVICE_QUALIFIER static float rs_isnan(float x) {
#ifdef __CUDA_ARCH__
        return isnan(x);
#else
        return std::isnan(x);
#endif
    }
    HOST_DEVICE_QUALIFIER static float rs_isinf(float x) {
#ifdef __CUDA_ARCH__
        return isinf(x);
#else
        return std::isinf(x);
#endif
    }
    HOST_DEVICE_QUALIFIER static float rs_isfinite(float x) {
#ifdef __CUDA_ARCH__
        return isfinite(x);
#else
        return std::isfinite(x);
#endif
    }
    HOST_DEVICE_QUALIFIER static float rs_fma_rn(float a, float b, float c) {
#ifdef __CUDA_ARCH__
        return __fmaf_rn(a, b, c);
#else
        return a * b + c;
#endif
    }
    HOST_DEVICE_QUALIFIER static int rs_round_down(float x) {
#ifdef __CUDA_ARCH__
        return __float2int_rd(x);
#else
        return static_cast<int>(floor(x));
#endif
    }
    HOST_DEVICE_QUALIFIER static float rs_min(float a, float b) {
#ifdef __CUDA_ARCH__
        return fminf(a, b);
#else
        return a < b ? a : b;
#endif
    }
    HOST_DEVICE_QUALIFIER static float rs_max(float a, float b) {
#ifdef __CUDA_ARCH__
        return fmaxf(a, b);
#else
        return a > b ? a : b;
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_floor(float x) { return floor(x); }
    HOST_DEVICE_QUALIFIER static float rs_ceil(float x) { return ceil(x); }
    HOST_DEVICE_QUALIFIER static float rs_round(float x) { return round(x); }
    HOST_DEVICE_QUALIFIER static float rs_sin(float x) { return sinf(x); }
    HOST_DEVICE_QUALIFIER static float rs_cos(float x) { return cosf(x); }
    HOST_DEVICE_QUALIFIER static float rs_tan(float x) { return tanf(x); }
    HOST_DEVICE_QUALIFIER static float rs_asin(float x) { return asinf(x); }
    HOST_DEVICE_QUALIFIER static float rs_acos(float x) { return acosf(x); }
    HOST_DEVICE_QUALIFIER static float rs_atan(float x) { return atanf(x); }
    HOST_DEVICE_QUALIFIER static float rs_atan2(float y, float x) { return atan2f(y, x); }
    HOST_DEVICE_QUALIFIER static void rs_sincos(float x, float *s, float *c) {
#ifdef _GNU_SOURCE
        sincosf(x, s, c);
#else
        *s = sinf(x);
        *c = cosf(x);
#endif
    }
    HOST_DEVICE_QUALIFIER static float rs_sqrt(float x) { return sqrtf(x); }
    HOST_DEVICE_QUALIFIER static float rs_pow(float x, float y) { return powf(x, y); }
    HOST_DEVICE_QUALIFIER static float rs_log(float x) { return logf(x); }

    HOST_DEVICE_QUALIFIER static float rs_divide_fast(float x, float y) {
#ifdef __CUDA_ARCH__
        return __fdividef(x, y);
#else
        return x / y;
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_sqrt_fast(float x) {
#ifdef __CUDA_ARCH__
        return __fsqrt_rn(x);
#else
        return sqrtf(x);
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_sin_fast(float x) {
#ifdef __CUDA_ARCH__
        return __sinf(x);
#else
        return sinf(x);
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_cos_fast(float x) {
#ifdef __CUDA_ARCH__
        return __cosf(x);
#else
        return cosf(x);
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_tan_fast(float x) {
#ifdef __CUDA_ARCH__
        return __tanf(x);
#else
        return tanf(x);
#endif
    }

    HOST_DEVICE_QUALIFIER static void rs_sincos_fast(float x, float *s, float *c) {
#ifdef __CUDA_ARCH__
        return __sincosf(x, s, c);
#elif defined(_GNU_SOURCE)
        sincosf(x, s, c);
#else
        *s = sinf(x);
        *c = cosf(x);
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_log_fast(float x) {
#ifdef __CUDA_ARCH__
        return __logf(x);
#else
        return logf(x);
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_log2_fast(float x) {
#ifdef __CUDA_ARCH__
        return __log2f(x);
#else
        return log2f(x);
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_log10_fast(float x) {
#ifdef __CUDA_ARCH__
        return __log10f(x);
#else
        return log10f(x);
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_exp_fast(float x) {
#ifdef __CUDA_ARCH__
        return __expf(x);
#else
        return expf(x);
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_exp10_fast(float x) {
#ifdef __CUDA_ARCH__
        return __exp10f(x);
#else
        return powf(10.0f, x);
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_pow_fast(float x, float y) {
#ifdef __CUDA_ARCH__
        return __powf(x, y);
#else
        return powf(x, y);
#endif
    }

    HOST_DEVICE_QUALIFIER static int rs_float_as_int(float f) {
#ifdef __CUDA_ARCH__
        return __float_as_int(f);
#else
        int i;
        std::memcpy(&i, &f, sizeof(i));
        return i;
#endif
    }

    HOST_DEVICE_QUALIFIER static float rs_int_as_float(int i) {
#ifdef __CUDA_ARCH__
        return __int_as_float(i);
#else
        float f;
        std::memcpy(&f, &i, sizeof(f));
        return f;
#endif
    }
};

template <>
struct RS_FloatIntrinsics<double> {
    HOST_DEVICE_QUALIFIER static double rs_isnan(double x) {
#ifdef __CUDA_ARCH__
        return isnan(x);
#else
        return std::isnan(x);
#endif
    }
    HOST_DEVICE_QUALIFIER static double rs_isinf(double x) {
#ifdef __CUDA_ARCH__
        return isinf(x);
#else
        return std::isinf(x);
#endif
    }
    HOST_DEVICE_QUALIFIER static double rs_isfinite(double x) {
#ifdef __CUDA_ARCH__
        return isfinite(x);
#else
        return std::isfinite(x);
#endif
    }
    HOST_DEVICE_QUALIFIER static double rs_fma_rn(double a, double b, double c) {
#ifdef __CUDA_ARCH__
        return __fma_rn(a, b, c);
#else
        return a * b + c;
#endif
    }
    DEVICE_QUALIFIER static int rs_round_down(double x) {
#ifdef __CUDA_ARCH__
        return __double2int_rd(x);
#else
        return static_cast<int>(floor(x));
#endif
    }

    HOST_DEVICE_QUALIFIER static double rs_min(double a, double b) {
#ifdef __CUDA_ARCH__
        return fmin(a, b);
#else
        return a < b ? a : b;
#endif
    }

    HOST_DEVICE_QUALIFIER static double rs_max(double a, double b) {
#ifdef __CUDA_ARCH__
        return fmax(a, b);
#else
        return a > b ? a : b;
#endif
    }

    HOST_DEVICE_QUALIFIER static double rs_floor(double x) { return floor(x); }
    HOST_DEVICE_QUALIFIER static double rs_ceil(double x) { return ceil(x); }
    HOST_DEVICE_QUALIFIER static double rs_round(double x) { return round(x); }
    HOST_DEVICE_QUALIFIER static double rs_sin(double x) { return sin(x); }
    HOST_DEVICE_QUALIFIER static double rs_cos(double x) { return cos(x); }
    HOST_DEVICE_QUALIFIER static double rs_tan(double x) { return tan(x); }
    HOST_DEVICE_QUALIFIER static double rs_asin(double x) { return asin(x); }
    HOST_DEVICE_QUALIFIER static double rs_acos(double x) { return acos(x); }
    HOST_DEVICE_QUALIFIER static double rs_atan(double x) { return atan(x); }
    HOST_DEVICE_QUALIFIER static double rs_atan2(double y, double x) { return atan2(y, x); }
    HOST_DEVICE_QUALIFIER static void rs_sincos(double x, double *s, double *c) {
#ifdef _GNU_SOURCE
        sincos(x, s, c);
#else
        *s = sin(x);
        *c = cos(x);
#endif
    }
    HOST_DEVICE_QUALIFIER static double rs_sqrt(double x) { return sqrt(x); }
    HOST_DEVICE_QUALIFIER static float rs_pow(double x, double y) { return pow(x, y); }
    HOST_DEVICE_QUALIFIER static double rs_log(double x) { return log(x); }

    HOST_DEVICE_QUALIFIER static double rs_divide_fast(double x, double y) { return x / y; }

    HOST_DEVICE_QUALIFIER static double rs_sqrt_fast(double x) { return sqrt(x); }

    HOST_DEVICE_QUALIFIER static double rs_sin_fast(double x) { return sin(x); }

    HOST_DEVICE_QUALIFIER static double rs_cos_fast(double x) { return cos(x); }

    HOST_DEVICE_QUALIFIER static double rs_tan_fast(double x) { return tan(x); }

    HOST_DEVICE_QUALIFIER static void rs_sincos_fast(double x, double *s, double *c) {
        *s = sin(x);
        *c = cos(x);
    }

    HOST_DEVICE_QUALIFIER static double rs_log_fast(double x) { return log(x); }

    HOST_DEVICE_QUALIFIER static double rs_log2_fast(double x) { return log2(x); }

    HOST_DEVICE_QUALIFIER static double rs_log10_fast(double x) { return log10(x); }

    HOST_DEVICE_QUALIFIER static double rs_exp_fast(double x) { return exp(x); }

    HOST_DEVICE_QUALIFIER static double rs_exp10_fast(double x) { return pow(10.0, x); }

    HOST_DEVICE_QUALIFIER static double rs_pow_fast(double x, double y) { return pow(x, y); }
};

template <typename T>
struct is_floating_point : std::is_floating_point<T> {};

template <typename T>
struct is_integral : std::is_integral<T> {};

struct RS_FloatConvert {
    //! Helper functions for bit manipulation
    template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type = 0>
    HOST_DEVICE_QUALIFIER static auto floatingToInteger(T x) -> decltype(auto) {
        union {
            T f;
            std::conditional_t<sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>
                i;
        } u;
        u.f = x;
        return u.i;
    }

    template <typename T, typename std::enable_if<is_integral<T>::value, int>::type = 0>
    HOST_DEVICE_QUALIFIER static auto integerToFloating(T x) -> decltype(auto) {
        union {
            std::conditional_t<sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>
                i;
            std::conditional_t<sizeof(T) == sizeof(unsigned int), float, double> f;
        } u;
        u.i = x;
        return u.f;
    }

    //! Helper functions for half precision storage, rounding to nearest even
    HOST_DEVICE_QUALIFIER static unsigned short floatToHalf(float x) {
#ifdef __CUDACC__
        return __half_as_ushort(__float2half_rn(x));
#else
        unsigned int bits = floatingToInteger(x);
        unsigned int sign = (bits >> 16) & 0x8000u;
        unsigned int mag = bits & 0x7FFFFFFFu;

        if (mag > 0x7F800000u) return static_cast<unsigned short>(sign | 0x7E00u);
        if (mag >= 0x477FF000u) return static_cast<unsigned short>(sign | 0x7C00u);
        if (mag >= 0x38800000u) {
            unsigned int value = mag - 0x38000000u;
            unsigned int round = ((value >> 13) & 1u) + 0x0FFFu;
            return static_cast<unsigned short>(sign | ((value + round) >> 13));
        }
        if (mag < 0x33000000u) return static_cast<unsigned short>(sign);

        int shift = 126 - static_cast<int>(mag >> 23);
        unsigned int mantissa = (mag & 0x007FFFFFu) | 0x00800000u;
        unsigned int value = mantissa >> shift;
        unsigned int remainder = mantissa & ((1u << shift) - 1u);
        unsigned int middle = 1u << (shift - 1);
        if (remainder > middle || (remainder == middle && (value & 1u))) value++;
        return static_cast<unsigned short>(sign | value);
#endif
    }

    HOST_DEVICE_QUALIFIER static float halfToFloat(unsigned short x) {
#ifdef __CUDACC__
        return __half2float(__ushort_as_half(x));
#else
        unsigned int sign = static_cast<unsigned int>(x & 0x8000u) << 16;
        unsigned int exponent = (x >> 10) & 0x1Fu;
        unsigned int mantissa = x & 0x03FFu;

        if (exponent == 0x1Fu)
            return integerToFloating(sign | 0x7F800000u | (mantissa << 13));
        if (exponent == 0) {
            if (mantissa == 0) return integerToFloating(sign);
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FFu;
        }
        return integerToFloating(sign | ((exponent + 112u) << 23) | (mantissa << 13));
#endif
    }
};

template <RSE_FloatType Type>
struct RS_FloatDataType {};

template <>
struct RS_FloatDataType<RSE_FLOAT_16> {
    using DataType = unsigned short;
};

template <>
struct RS_FloatDataType<RSE_FLOAT_32> {
    using DataType = float;
};

template <>
struct RS_FloatDataType<RSE_FLOAT_64> {
    using DataType = double;
};

template <RSE_FloatType Type>
using RS_FloatMem = typename RS_FloatDataType<Type>::DataType;

template <RSE_FloatType InType, RSE_FloatType Min, RSE_FloatType Max = RSE_FLOAT_64>
struct RS_FloatCapAccuracy {
    static constexpr RSE_FloatType Type = (InType > Max) ? Max : ((InType < Min) ? Min : InType);
};

template <typename Function>
void selectComputeAccuracy(RSE_FloatType type, Function &&func) {
    switch (type) {
        case RSE_FLOAT_32:
            func(std::integral_constant<RSE_FloatType, RSE_FLOAT_32>{});
            break;
        case RSE_FLOAT_64:
            func(std::integral_constant<RSE_FloatType, RSE_FLOAT_64>{});
            break;
        default:
            RS_THROW(RS_InvalidArgumentException("Unknown float type"));
    }
}

};  // namespace rise

#endif  // RS_RSE_FLOAT_H
