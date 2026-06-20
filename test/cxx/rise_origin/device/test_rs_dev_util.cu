#include <gtest/gtest.h>

#include <limits>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/device/rs_dev_util.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_StaticDivMod is a compile time division and modulo helper that
 * uses fast paths for power of two and 32 bit divisors. RS_DivMod is
 * the runtime version of the same helper.
 *
 * - Compile time division and modulo
 *   div, mod, divmod for power of two, 32 bit, and 64 bit divisors
 *
 * - Runtime division and modulo
 *   div, mod, divmod through the RS_DivMod object
 *
 * - Edge cases
 *   Division by one, maximum values, zero dividend
 *
 * These tests run on the host because the helpers are header only and
 * need no device launch.
 *****************************************************************/

class RS_DevUtilTest : public ::testing::Test {
   protected:
    void SetUp() override {
        RS_TEST_SETUP_DEVICE_PROVIDER();
        host_provider.init(0);
    }

    void TearDown() override {
        provider.free();
        host_provider.free();
    }

    RS_TEST_DEVICE_PROVIDER_TYPE provider;
    RS_MemoryCPUProvider host_provider;
};

template <uint64_t Divisor>
static void runStaticDivModTest(uint64_t value, uint64_t expected_div, uint64_t expected_mod) {
    uint64_t q = 0, r = 0;

    auto div_result = RS_StaticDivMod<Divisor>::div(value);
    auto mod_result = RS_StaticDivMod<Divisor>::mod(value);

    EXPECT_EQ(div_result, expected_div);
    EXPECT_EQ(mod_result, expected_mod);

    RS_StaticDivMod<Divisor>::divmod(value, q, r);
    EXPECT_TRUE(q == expected_div && r == expected_mod);
}

template <typename T>
static void runDivModTest(T divisor, T value, T expected_div, T expected_mod) {
    RS_DivMod<T> divmod(divisor);
    T q = 0, r = 0;

    auto div_result = divmod.div(value);
    auto mod_result = divmod.mod(value);

    EXPECT_EQ(div_result, expected_div);
    EXPECT_EQ(mod_result, expected_mod);

    divmod.divmod(value, q, r);
    EXPECT_TRUE(q == expected_div && r == expected_mod);
}

/*****************************************************************
 * Test cases for RS_StaticDivMod
 *
 * Test cases:
 * 1. Compile time assertions on divisor and division by one.
 * 2. Power of two divisors.
 * 3. Small fast path divisors.
 * 4. Larger 32 bit fast path divisors.
 * 5. 64 bit regular path divisors.
 * 6. Edge cases and maximum values.
 *****************************************************************/

TEST_F(RS_DevUtilTest, StaticDivMod) {
    static_assert(RS_StaticDivMod<10>::divisor == 10, "Compile time check failed");
    static_assert(RS_StaticDivMod<1>::div(12345) == 12345, "Division by 1 failed");

    runStaticDivModTest<32>(100, 3, 4);
    runStaticDivModTest<128>(300, 2, 44);

    runStaticDivModTest<2>(10, 5, 0);
    runStaticDivModTest<3>(10, 3, 1);
    runStaticDivModTest<10>(100, 10, 0);
    runStaticDivModTest<10>(101, 10, 1);

    runStaticDivModTest<4294967295>(4294967295, 1, 0);
    runStaticDivModTest<4294967295>(4294967296, 1, 1);

    runStaticDivModTest<1000000000000000000ULL>(1000000000000000000ULL, 1, 0);
    runStaticDivModTest<1000000000000000000ULL>(1000000000000000001ULL, 1, 1);

    runStaticDivModTest<1>(1234567890, 1234567890, 0);
    runStaticDivModTest<100>(0, 0, 0);

    runStaticDivModTest<1>(std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max(), 0);
}

/*****************************************************************
 * Test cases for RS_DivMod
 *
 * Test cases:
 * 1. 32 bit fast path divisors.
 * 2. 64 bit generic path divisors.
 * 3. Edge cases.
 *****************************************************************/

TEST_F(RS_DevUtilTest, RuntimeDivModUInt32) {
    runDivModTest<uint32_t>(10, 100, 10, 0);
    runDivModTest<uint32_t>(10, 101, 10, 1);
    runDivModTest<uint32_t>(3, 10, 3, 1);
    runDivModTest<uint32_t>(7, 49, 7, 0);
    runDivModTest<uint32_t>(7, 50, 7, 1);

    runDivModTest<uint32_t>(32, 100, 3, 4);
    runDivModTest<uint32_t>(128, 300, 2, 44);

    runDivModTest<uint32_t>(1, 12345, 12345, 0);
    runDivModTest<uint32_t>(100, 0, 0, 0);
}

TEST_F(RS_DevUtilTest, RuntimeDivModUInt64) {
    runDivModTest<uint64_t>(10, 100, 10, 0);
    runDivModTest<uint64_t>(10, 101, 10, 1);
    runDivModTest<uint64_t>(3, 10, 3, 1);

    runDivModTest<uint64_t>(1000000000000ULL, 5000000000000ULL, 5, 0);
    runDivModTest<uint64_t>(1000000000000ULL, 5000000000001ULL, 5, 1);

    runDivModTest<uint64_t>(1, 1234567890123456ULL, 1234567890123456ULL, 0);
}
