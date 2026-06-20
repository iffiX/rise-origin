// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RANDOM_H
#define RS_RANDOM_H

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/math/rs_vec.h"

#define RS_PHILOX_M4x32_0 0xD2511F53U
#define RS_PHILOX_M4x32_1 0xCD9E8D57U
#define RS_PHILOX_W32_0 0x9E3779B9U
#define RS_PHILOX_W32_1 0xBB67AE85U

namespace rise {

namespace device {

/**
 * @brief Device side per thread random state, implementing the Philox_4x32_10 engine
 */
class RS_RandomState {
   public:
    RVec2ui key;
    RVec4ui counter;
    RVec4ui result;
    int get_counter;

    HOST_DEVICE_INLINE_QUALIFIER void init(
        const uint64_t seed = 0x123456789ABCDEF0ull,
        const uint64_t subsequence = 0,
        const uint64_t offset = 0
    ) {
        key = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
        counter = RVec4ui::zero();
        result = RVec4ui::zero();
        get_counter = 0;
        restart(subsequence, offset);
    }

    HOST_DEVICE_INLINE_QUALIFIER void
    restart(const uint64_t subsequence = 0, const uint64_t offset = 0) {
        {
            // skip subsequence by adding subsequence to counter
            uint32_t low = static_cast<uint32_t>(subsequence);
            uint32_t high = static_cast<uint32_t>(subsequence >> 32);
            uint32_t old_z = counter.z;
            counter.z += low;
            counter.w += high + (counter.z < old_z ? 1 : 0);  // consider overflow
        }
        {
            // skip offset by adding offset to counter
            uint32_t low = static_cast<uint32_t>(offset);
            uint32_t high = static_cast<uint32_t>(offset >> 32);
            uint32_t old_y = counter.y, old_z = counter.z, old_w = counter.w;
            counter.x += low;
            counter.y += high + (counter.x < old_y ? 1 : 0);
            counter.z += counter.y < old_z ? 1 : 0;
            counter.w += counter.z < old_w ? 1 : 0;  // consider overflow
        }
        _increaseCounter(counter);
        result = _generate(key, counter);
    }

    /**
     * @brief Generate a random unsigned integer in the sequence.
     */
    HOST_DEVICE_INLINE_QUALIFIER uint32_t next() {
        uint32_t res = result[get_counter];
        if (++get_counter == 4) {
            _increaseCounter(counter);
            result = _generate(key, counter);
            get_counter = 0;
        }
        return res;
    }

    /**
     * @brief Generate next 4 random unsigned integers in the sequence.
     */
    HOST_DEVICE_INLINE_QUALIFIER RVec4ui next4() {
        RVec4ui res = result;
        _increaseCounter(counter);
        result = _generate(key, counter);
        return res;
    }

    /**
     * @brief Generate a random integer in range [min, max).
     */
    template <typename T = int>
    HOST_DEVICE_INLINE_QUALIFIER T randint(T min, T max) {
        assert(min <= max);

        using Unsigned = std::conditional_t<sizeof(T) <= 4, uint32_t, uint64_t>;

        Unsigned range;
        if constexpr (std::is_signed_v<T>) {
            // Note: because of modulo effect, this also works for negative ranges.
            range = static_cast<Unsigned>(max) - static_cast<Unsigned>(min);

        } else {
            range = max - min;
        }

        if constexpr (sizeof(T) <= 4) {
            uint32_t r = next();
            uint64_t scaled = range == 0 ? 0 : _mulhi32(r, range);
            return static_cast<T>(scaled + min);
        } else {
            uint64_t r = (static_cast<uint64_t>(next()) << 32) | next();
            uint64_t scaled = (range == 0) ? r : _mulhi64(r, range);
            return static_cast<T>(scaled + min);
        }
    }

    /**
     * @brief Generate a random float in range [0, 1).
     */
    template <typename T = float>
    HOST_DEVICE_INLINE_QUALIFIER T rand() {
        static_assert(std::is_floating_point_v<T>, "rand only supports floating point types");
        using Unsigned = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;

        Unsigned r;
        if constexpr (sizeof(T) == 4) {
            r = next();
        } else {
            r = (static_cast<uint64_t>(next()) << 32) | next();
        }
        // Use half factor to not bias towards 0
        constexpr T factor = T(1.0) / (T(~static_cast<Unsigned>(0)) + T(1.0));
        constexpr T half_factor = T(0.5) * factor;
        return static_cast<T>(r) * factor + half_factor;
    }

    /**
     * @brief Generate a random integer / float in range [min, max)
     */
    template <typename T = float>
    HOST_DEVICE_INLINE_QUALIFIER T uniform(T min, T max) {
        if constexpr (std::is_floating_point_v<T>) {
            return rand<T>() * (max - min) + min;
        } else {
            return randint(min, max);
        }
    }

    /**
     * @brief Generate a random float from standard normal distribution
     * with mean 0 and std 1
     * @note Using the box-muller method.
     */
    template <typename T = float>
    HOST_DEVICE_INLINE_QUALIFIER T randn() {
        static_assert(std::is_floating_point_v<T>, "randn only supports floating point types");
        using RS_FI = RS_FloatIntrinsics<T>;
        T u = rand<T>();
        T v = rand<T>();
        return RS_FI::rs_sqrt(T(-2.0) * RS_FI::rs_log(u)) * RS_FI::rs_cos(T(2.0 * RS_PI) * v);
    }

    /**
     * @brief Generate two random floats from standard normal distribution
     * with mean 0 and std 1. More efficient than call randn() twice.
     */
    template <typename T = float>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D<T> randn2() {
        static_assert(std::is_floating_point_v<T>, "randn2 only supports floating point types");
        using RS_FI = RS_FloatIntrinsics<T>;
        T u = rand<T>();
        T v = rand<T>();
        T r = RS_FI::rs_sqrt(T(-2.0) * RS_FI::rs_log(u));
        T sin_theta, cos_theta;
        RS_FI::rs_sincos(v * T(2.0 * RS_PI), &sin_theta, &cos_theta);
        return {r * cos_theta, r * sin_theta};
    }

    /**
     * @brief Generate a random float from normal distribution with given mean and std.
     */
    template <typename T = float>
    HOST_DEVICE_INLINE_QUALIFIER T normal(T mean, T stddev) {
        return randn<T>() * stddev + mean;
    }

    /**
     * @brief Generate two random floats from normal distribution with given mean and std.
     * More efficient than call randn() twice.
     */
    template <typename T = float>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D<T> normal2(T mean, T stddev) {
        return randn2<T>() * stddev + mean;
    }

   private:
    HOST_DEVICE_INLINE_QUALIFIER void _increaseCounter(RVec4ui& counter) {
        counter.x++;
        uint32_t overflow = counter.x == 0 ? 1 : 0;
        counter.y += overflow;
        overflow = counter.y == 0 ? overflow : 0;
        counter.z += overflow;
        overflow = counter.z == 0 ? overflow : 0;
        counter.w += overflow;
    }

    HOST_DEVICE_INLINE_QUALIFIER RVec4ui _generate(RVec2ui& key, const RVec4ui& counter) {
        RVec4ui result = counter;
        // 10 rounds
        _round(key, result);
        key += {RS_PHILOX_W32_0, RS_PHILOX_W32_1};  // round 0
        _round(key, result);
        key += {RS_PHILOX_W32_0, RS_PHILOX_W32_1};  // round 1
        _round(key, result);
        key += {RS_PHILOX_W32_0, RS_PHILOX_W32_1};  // round 2
        _round(key, result);
        key += {RS_PHILOX_W32_0, RS_PHILOX_W32_1};  // round 3
        _round(key, result);
        key += {RS_PHILOX_W32_0, RS_PHILOX_W32_1};  // round 4
        _round(key, result);
        key += {RS_PHILOX_W32_0, RS_PHILOX_W32_1};  // round 5
        _round(key, result);
        key += {RS_PHILOX_W32_0, RS_PHILOX_W32_1};  // round 6
        _round(key, result);
        key += {RS_PHILOX_W32_0, RS_PHILOX_W32_1};  // round 7
        _round(key, result);
        key += {RS_PHILOX_W32_0, RS_PHILOX_W32_1};  // round 8
        _round(key, result);                        // round 9
        return result;
    }

    HOST_DEVICE_INLINE_QUALIFIER void _round(const RVec2ui& key, RVec4ui& counter) {
        uint32_t hi_0, hi_1;
        uint32_t lo_0 = _mulhilo32(RS_PHILOX_M4x32_0, counter.x, hi_0);
        uint32_t lo_1 = _mulhilo32(RS_PHILOX_M4x32_1, counter.z, hi_1);
        counter.x = hi_1 ^ counter.y ^ key.x;
        counter.y = lo_1;
        counter.z = hi_0 ^ counter.w ^ key.y;
        counter.w = lo_0;
    }

    HOST_DEVICE_INLINE_QUALIFIER uint32_t
    _mulhilo32(const uint32_t x, const uint32_t y, uint32_t& z) {
        uint64_t product = static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
        z = static_cast<uint32_t>(product >> 32);
        return static_cast<uint32_t>(product);
    }

    HOST_DEVICE_INLINE_QUALIFIER uint32_t _mulhi32(const uint32_t x, const uint32_t y) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
        return __umulhi(x, y);
#else
        uint64_t product = static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
        return static_cast<uint32_t>(product >> 32);
#endif
    }

    HOST_DEVICE_INLINE_QUALIFIER
    uint64_t _mulhi64(const uint64_t x, const uint64_t y) {
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
        return __umul64hi(x, y);
#elif defined(_MSC_VER) && defined(RS_DEVICE_ENABLE_INTRINSICS)
        return __umulh(x, y);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(RS_DEVICE_ENABLE_INTRINSICS)
        __uint128_t result = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(y);
        return static_cast<uint64_t>(result >> 64);
#else
        constexpr uint64_t mask = 0xFFFFFFFFULL;
        const uint64_t x_lo = x & mask;
        const uint64_t x_hi = x >> 32;
        const uint64_t y_lo = y & mask;
        const uint64_t y_hi = y >> 32;

        uint64_t x0 = x_lo * y_lo;  // low 64 bits
        uint64_t x1 = x_lo * y_hi;
        uint64_t x2 = x_hi * y_lo;
        uint64_t x3 = x_hi * y_hi;

        uint64_t t = (x0 >> 32) + (x1 & mask) + (x2 & mask);
        uint64_t carry = t >> 32;
        uint64_t hi = x3 + (x1 >> 32) + (x2 >> 32) + carry;
        return hi;
#endif
    }
};

class RSCF_InitRandomState : public RS_ComputeFunction<RSCF_InitRandomState> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_Array<RS_RandomState, 1> random_states, uint64_t seed, RSize offset) {
        RIndex tid = index1D();
        RS_RandomState state;
        state.init(seed, tid + offset);
        random_states[tid + offset] = state;
    }
};

class RSCF_GenerateRandomUniform : public RS_ComputeFunction<RSCF_GenerateRandomUniform> {
   public:
    template <typename ArrayLike, typename T>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_Array<RS_RandomState, 1> random_states, ArrayLike array_like, T min, T max) {
        RIndex tid = index1D();
        RS_RandomState state = random_states[tid];
        array_like[tid] = state.uniform<T>(min, max);
    }
};

class RSCF_GenerateRandomNormal : public RS_ComputeFunction<RSCF_GenerateRandomNormal> {
   public:
    template <typename ArrayLike, typename T>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_Array<RS_RandomState, 1> random_states, ArrayLike array_like, T mean, T stddev) {
        RIndex tid = index1D();
        RS_RandomState state = random_states[tid];
        array_like[tid] = state.normal<T>(mean, stddev);
    }
};

class RS_Random {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    RS_Array<RS_RandomState, 1> random_states;
    uint64_t seed = 0x123456789ABCDEF0ull;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_Random() = default;

    explicit RS_Random(RS_MemoryProvider* provider, uint64_t seed = 0x123456789ABCDEF0ull)
        : random_states(provider), seed(seed) {}

    void reset(uint64_t seed) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);
        castProviderPtrAndCall(provider(), [&](auto p_ptr) {
            p_ptr->template run<RSCF_InitRandomState>(
                {random_states.size()}, random_states, seed, 0
            );
        });
    }

    void free() { random_states.free(); }

    template <typename ArrayLike>
    void fillUniform(
        ArrayLike array_like,
        typename ArrayLike::ObjectType min = 0,
        typename ArrayLike::ObjectType max = 1,
        bool sync = true
    ) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);
        if (array_like.size() != 0) {
            if (array_like.size() > random_states.size()) {
                _adaptStates(array_like.size());
            }
            castProviderPtrAndCall(provider(), [&](auto p_ptr) {
                p_ptr->template run<RSCF_GenerateRandomUniform>(
                    {array_like.size(), RS_AUTO_BLOCK_SIZE, sync},
                    random_states,
                    array_like,
                    min,
                    max
                );
            });
        }
    }

    template <typename ArrayLike>
    void fillNormal(
        ArrayLike array_like,
        typename ArrayLike::ObjectType mean = 0,
        typename ArrayLike::ObjectType stddev = 1,
        bool sync = true
    ) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);
        if (array_like.size() != 0) {
            if (array_like.size() > random_states.size()) {
                _adaptStates(array_like.size());
            }
            castProviderPtrAndCall(provider(), [&](auto p_ptr) {
                p_ptr->template run<RSCF_GenerateRandomNormal>(
                    {array_like.size(), RS_AUTO_BLOCK_SIZE, sync},
                    random_states,
                    array_like,
                    mean,
                    stddev
                );
            });
        }
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return random_states.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return random_states.device(); }

   private:
    void _adaptStates(RSize new_state_size) {
        RSize old_state_size = random_states.size();
        if (new_state_size > old_state_size) {
            random_states.resize(new_state_size);
            castProviderPtrAndCall(provider(), [&](auto p_ptr) {
                p_ptr->template run<RSCF_InitRandomState>(
                    {new_state_size - old_state_size}, random_states, seed, old_state_size
                );
            });
        }
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_RANDOM_H
