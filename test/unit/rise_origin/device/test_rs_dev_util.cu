#include <gtest/gtest.h>

#include "rise_origin/device/rs_dev_util.h"

using namespace rise;
using namespace rise::device;

template <uint64_t Divisor>
void runTest(uint64_t value, uint64_t expected_div, uint64_t expected_mod) {
    uint64_t q = 0, r = 0;

    // Test div and mod separately
    auto divResult = RS_StaticDivMod<Divisor>::div(value);
    auto modResult = RS_StaticDivMod<Divisor>::mod(value);

    // Check individual results
    EXPECT_EQ(divResult, expected_div);
    EXPECT_EQ(modResult, expected_mod);

    // Test combined divmod
    RS_StaticDivMod<Divisor>::divmod(value, q, r);
    EXPECT_TRUE(q == expected_div && r == expected_mod);
}

TEST(RS_StaticDivModTest, TestDivide) {
    // Compile-time assertions
    static_assert(RS_StaticDivMod<10>::divisor == 10, "Compile-time check failed");
    static_assert(RS_StaticDivMod<1>::div(12345) == 12345, "Division by 1 failed");

    // Test with power of twos
    runTest<32>(100, 3, 4);
    runTest<128>(300, 2, 44);

    // Test with small divisors (fast path)
    runTest<2>(10, 5, 0);
    runTest<3>(10, 3, 1);
    runTest<10>(100, 10, 0);
    runTest<10>(101, 10, 1);

    // // Test with larger 32-bit divisor (still fast path)
    runTest<4294967295>(4294967295, 1, 0);  // Max 32-bit
    runTest<4294967295>(4294967296, 1, 1);

    // // Test with 64-bit divisor (regular path)
    runTest<1000000000000000000ULL>(1000000000000000000ULL, 1, 0);
    runTest<1000000000000000000ULL>(1000000000000000001ULL, 1, 1);

    // // Test with edge cases
    runTest<1>(1234567890, 1234567890, 0);  // Div by 1
    runTest<100>(0, 0, 0);                  // Div by nonzero, n = 0

    // // Maximum values
    runTest<1>(std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max(), 0);
}