#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/algorithm/rs_reduce.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_Reduce reduces a single input with a reduce operator.
 * RS_ReduceByGroup reduces many groups of variable size by group.
 *
 * - RS_Reduce
 *   Sum, max, and min over int and float. Verified against a host
 *   reference computed with std algorithms.
 *
 * - RS_ReduceByGroup
 *   Returns one reduced value per group as a std::vector. Covers
 *   single group, many small groups, and variable group sizes.
 *
 * - Boundary Conditions
 *   Empty input, single element, two elements, power of two and non
 *   power of two sizes, empty group list, and all zero size groups.
 *
 * - Edge Cases
 *   All zeros, all identical, negative values, and mixed values.
 *****************************************************************/

class RS_ReduceTest : public ::testing::Test {
   protected:
    void SetUp() override {
        RS_TEST_SETUP_DEVICE_PROVIDER();
        host_provider.init(0);
    }

    void TearDown() override {
        provider.free();
        host_provider.free();
    }

    std::vector<int> getRandomData(RSize size, int min_val = 0, int max_val = 1000) {
        static std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(min_val, max_val);
        std::vector<int> data(size);
        for (RSize i = 0; i < size; ++i) {
            data[i] = dist(rng);
        }
        return data;
    }

    RS_TEST_DEVICE_PROVIDER_TYPE provider;
    RS_MemoryCPUProvider host_provider;
};

/*****************************************************************
 * Test cases for RS_Reduce sum
 *
 * Test cases:
 * 1. Sum a large fill.
 * 2. Sum an empty input returns the init value.
 * 3. Sum a single element.
 * 4. Sum two elements.
 * 5. Sum a power of two size.
 * 6. Sum a non power of two size.
 *****************************************************************/

TEST_F(RS_ReduceTest, ReduceSum) {
    const RSize count = 100000;
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.fill(1);

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, static_cast<int>(count));

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceEmpty) {
    const RSize count = 0;
    RS_Array<int, 1> test_array({count}, &provider);

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 42);
    EXPECT_EQ(result, 42);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceSingleElement) {
    const RSize count = 1;
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.fill(100);

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, 100);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceTwoElements) {
    const RSize count = 2;
    std::vector<int> h_data = {10, 20};
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.write(h_data.data());

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, 30);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReducePowerOfTwo) {
    const RSize count = 1024;
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.fill(1);

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, static_cast<int>(count));

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceNonPowerOfTwo) {
    const RSize count = 1000;
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.fill(1);

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, static_cast<int>(count));

    test_array.free();
    reducer.free();
}

/*****************************************************************
 * Test cases for RS_Reduce max and min
 *
 * Test cases:
 * 1. Max over random data.
 * 2. Min over random data.
 *****************************************************************/

TEST_F(RS_ReduceTest, ReduceMax) {
    const RSize count = 10000;
    auto h_data = getRandomData(count, 0, 1000);
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.write(h_data.data());

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceMaxOp>(test_array, std::numeric_limits<int>::min());

    int expected = *std::max_element(h_data.begin(), h_data.end());
    EXPECT_EQ(result, expected);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceMin) {
    const RSize count = 10000;
    auto h_data = getRandomData(count, 0, 1000);
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.write(h_data.data());

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceMinOp>(test_array, std::numeric_limits<int>::max());

    int expected = *std::min_element(h_data.begin(), h_data.end());
    EXPECT_EQ(result, expected);

    test_array.free();
    reducer.free();
}

/*****************************************************************
 * Test cases for RS_Reduce edge cases and types
 *
 * Test cases:
 * 1. All zeros.
 * 2. All identical.
 * 3. Negative values.
 * 4. Mixed positive and negative cancel out.
 * 5. Float sum precision.
 *****************************************************************/

TEST_F(RS_ReduceTest, ReduceAllZeros) {
    const RSize count = 1000;
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.fill(0);

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, 0);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceAllIdentical) {
    const RSize count = 5000;
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.fill(7);

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, static_cast<int>(count) * 7);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceNegativeValues) {
    const RSize count = 1000;
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.fill(-5);

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, static_cast<int>(count) * -5);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceMixedPositiveNegative) {
    const RSize count = 100;
    std::vector<int> h_data(count);
    for (RSize i = 0; i < count; ++i) {
        h_data[i] = (i % 2 == 0) ? 10 : -10;
    }
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.write(h_data.data());

    RS_Reduce reducer(&provider);
    int result = reducer.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, 0);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceFloatPrecision) {
    const RSize count = 1000;
    std::vector<float> h_data(count, 0.1f);
    RS_Array<float, 1> test_array({count}, &provider);
    test_array.write(h_data.data());

    RS_Reduce reducer(&provider);
    float result = reducer.reduce<RS_ReduceSumOp>(test_array, 0.0f);
    EXPECT_NEAR(result, count * 0.1f, 0.1f);

    test_array.free();
    reducer.free();
}

/*****************************************************************
 * Test cases for RS_ReduceByGroup
 *
 * Test cases:
 * 1. Basic group reduction.
 * 2. Variable group sizes.
 * 3. Single group.
 * 4. Many small groups.
 * 5. Single element groups.
 * 6. Max by group.
 * 7. Min by group.
 * 8. Empty group list.
 * 9. All zero size groups return the init value.
 * 10. Negative values by group.
 * 11. Mixed values by group.
 *****************************************************************/

TEST_F(RS_ReduceTest, ReduceByGroupBasic) {
    std::vector<RSize> group_sizes = {100, 200, 300};
    RSize total = std::accumulate(group_sizes.begin(), group_sizes.end(), RSize(0));

    RS_Array<int, 1> test_array({total}, &provider);
    test_array.fill(1);

    RS_ReduceByGroup reducer(&provider);
    auto result = reducer.reduce<RS_ReduceSumOp>(test_array, group_sizes, 0);

    EXPECT_EQ(result.size(), group_sizes.size());
    for (RSize i = 0; i < group_sizes.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(group_sizes[i]));
    }

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupVariableSizes) {
    std::vector<RSize> group_sizes = {1, 1024, 100000, 10000};
    RSize total = std::accumulate(group_sizes.begin(), group_sizes.end(), RSize(0));

    RS_Array<int, 1> test_array({total}, &provider);
    test_array.fill(1);

    RS_ReduceByGroup reducer(&provider);
    auto result = reducer.reduce<RS_ReduceSumOp>(test_array, group_sizes, 0);

    EXPECT_EQ(result.size(), group_sizes.size());
    for (RSize i = 0; i < group_sizes.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(group_sizes[i]));
    }

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupSingleGroup) {
    std::vector<RSize> group_sizes = {5000};
    RSize total = 5000;

    RS_Array<int, 1> test_array({total}, &provider);
    test_array.fill(2);

    RS_ReduceByGroup reducer(&provider);
    auto result = reducer.reduce<RS_ReduceSumOp>(test_array, group_sizes, 0);

    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 10000);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupManySmallGroups) {
    std::vector<RSize> group_sizes(100, 10);
    RSize total = 1000;

    RS_Array<int, 1> test_array({total}, &provider);
    test_array.fill(1);

    RS_ReduceByGroup reducer(&provider);
    auto result = reducer.reduce<RS_ReduceSumOp>(test_array, group_sizes, 0);

    EXPECT_EQ(result.size(), 100u);
    for (auto r : result) {
        EXPECT_EQ(r, 10);
    }

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupSingleElementGroups) {
    std::vector<RSize> group_sizes = {1, 1, 1, 1, 1};
    RSize total = 5;

    std::vector<int> h_data = {10, 20, 30, 40, 50};
    RS_Array<int, 1> test_array({total}, &provider);
    test_array.write(h_data.data());

    RS_ReduceByGroup reducer(&provider);
    auto result = reducer.reduce<RS_ReduceSumOp>(test_array, group_sizes, 0);

    EXPECT_EQ(result.size(), 5u);
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result[1], 20);
    EXPECT_EQ(result[2], 30);
    EXPECT_EQ(result[3], 40);
    EXPECT_EQ(result[4], 50);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupMax) {
    std::vector<RSize> group_sizes = {100, 200};
    RSize total = 300;

    std::vector<int> h_data(total);
    for (RSize i = 0; i < 100; ++i) h_data[i] = static_cast<int>(i);
    for (RSize i = 100; i < 300; ++i) h_data[i] = static_cast<int>(i - 100);

    RS_Array<int, 1> test_array({total}, &provider);
    test_array.write(h_data.data());

    RS_ReduceByGroup reducer(&provider);
    auto result =
        reducer.reduce<RS_ReduceMaxOp>(test_array, group_sizes, std::numeric_limits<int>::min());

    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 99);
    EXPECT_EQ(result[1], 199);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupMin) {
    std::vector<RSize> group_sizes = {100, 200};
    RSize total = 300;

    std::vector<int> h_data(total);
    for (RSize i = 0; i < 100; ++i) h_data[i] = static_cast<int>(i + 50);
    for (RSize i = 100; i < 300; ++i) h_data[i] = static_cast<int>(i);

    RS_Array<int, 1> test_array({total}, &provider);
    test_array.write(h_data.data());

    RS_ReduceByGroup reducer(&provider);
    auto result =
        reducer.reduce<RS_ReduceMinOp>(test_array, group_sizes, std::numeric_limits<int>::max());

    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 50);
    EXPECT_EQ(result[1], 100);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupEmptyGroupList) {
    std::vector<RSize> group_sizes;
    RS_Array<int, 1> test_array({0}, &provider);

    RS_ReduceByGroup reducer(&provider);
    auto result = reducer.reduce<RS_ReduceSumOp>(test_array, group_sizes, 0);

    EXPECT_TRUE(result.empty());

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupAllZeroSizeGroups) {
    std::vector<RSize> group_sizes = {0, 0, 0};
    RS_Array<int, 1> test_array({0}, &provider);

    RS_ReduceByGroup reducer(&provider);
    auto result = reducer.reduce<RS_ReduceSumOp>(test_array, group_sizes, 42);

    EXPECT_EQ(result.size(), 3u);
    for (auto r : result) {
        EXPECT_EQ(r, 42);
    }

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupNegativeValues) {
    std::vector<RSize> group_sizes = {50, 50};
    RSize total = 100;

    RS_Array<int, 1> test_array({total}, &provider);
    test_array.fill(-3);

    RS_ReduceByGroup reducer(&provider);
    auto result = reducer.reduce<RS_ReduceSumOp>(test_array, group_sizes, 0);

    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], -150);
    EXPECT_EQ(result[1], -150);

    test_array.free();
    reducer.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupMixedData) {
    std::vector<RSize> group_sizes = {3, 3};
    RSize total = 6;

    std::vector<int> h_data = {10, -5, 3, -20, 15, 8};
    RS_Array<int, 1> test_array({total}, &provider);
    test_array.write(h_data.data());

    RS_ReduceByGroup reducer(&provider);
    auto result = reducer.reduce<RS_ReduceSumOp>(test_array, group_sizes, 0);

    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 8);
    EXPECT_EQ(result[1], 3);

    test_array.free();
    reducer.free();
}
