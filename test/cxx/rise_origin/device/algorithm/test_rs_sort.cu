#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/algorithm/rs_sort.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_SortPairs (merge sort) and RS_RadixSortPairs (radix sort)
 * sort key value pairs on the selected device.
 *
 * - Correctness Verification
 *   Sorted output is checked with std::is_sorted and element
 *   checks. Large random data sets give wide coverage.
 *
 * - Boundary Conditions
 *   Empty, single element, and two element inputs.
 *
 * - Edge Cases
 *   Already sorted, reverse sorted, and all identical inputs.
 *
 * - Algorithm Specific Features
 *   Merge sort takes a custom compare operator. Radix sort takes
 *   an ascending or descending flag.
 *****************************************************************/

struct CompareOp {
    HOST_DEVICE_QUALIFIER bool operator()(int lhs, int rhs) { return lhs < rhs; }
};

class RS_SortTest : public ::testing::Test {
   protected:
    void SetUp() override {
        RS_TEST_SETUP_DEVICE_PROVIDER();
        host_provider.init(0);
    }

    void TearDown() override {
        provider.free();
        host_provider.free();
    }

    std::vector<int> getRandomData(RSize size) {
        static std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(0, 100000);
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
 * Test cases for RS_SortPairs (merge sort)
 *
 * Test cases:
 * 1. Sort large random data.
 * 2. Sort an empty input.
 * 3. Sort a single element.
 * 4. Sort two elements.
 * 5. Sort already sorted data.
 * 6. Sort reverse sorted data.
 * 7. Sort all identical data.
 *****************************************************************/

TEST_F(RS_SortTest, MergeSortRandom) {
    const RSize count = 100000;
    auto h_keys = getRandomData(count);
    auto h_values = getRandomData(count);

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_SortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, CompareOp());

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, MergeSortEmpty) {
    const RSize count = 0;
    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_SortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, CompareOp());

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, MergeSortSingle) {
    const RSize count = 1;
    std::vector<int> h_keys = {42};
    std::vector<int> h_values = {100};

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_SortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, CompareOp());

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_EQ(h_sorted_keys[0], 42);
    EXPECT_EQ(h_sorted_vals[0], 100);

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, MergeSortTwo) {
    const RSize count = 2;
    std::vector<int> h_keys = {50, 10};
    std::vector<int> h_values = {100, 200};

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_SortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, CompareOp());

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_EQ(h_sorted_keys[0], 10);
    EXPECT_EQ(h_sorted_keys[1], 50);
    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, MergeSortAlreadySorted) {
    const RSize count = 10000;
    std::vector<int> h_keys(count);
    std::vector<int> h_values(count);
    for (RSize i = 0; i < count; ++i) {
        h_keys[i] = static_cast<int>(i);
        h_values[i] = static_cast<int>(i * 2);
    }

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_SortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, CompareOp());

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_sorted_keys[i], static_cast<int>(i));
    }

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, MergeSortReverseSorted) {
    const RSize count = 10000;
    std::vector<int> h_keys(count);
    std::vector<int> h_values(count);
    for (RSize i = 0; i < count; ++i) {
        h_keys[i] = static_cast<int>(count - i - 1);
        h_values[i] = static_cast<int>(i);
    }

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_SortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, CompareOp());

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_sorted_keys[i], static_cast<int>(i));
    }

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, MergeSortAllIdentical) {
    const RSize count = 10000;
    std::vector<int> h_keys(count, 42);
    std::vector<int> h_values = getRandomData(count);

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_SortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, CompareOp());

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_sorted_keys[i], 42);
    }

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

/*****************************************************************
 * Test cases for RS_RadixSortPairs (radix sort)
 *
 * Test cases:
 * 1. Sort large random data ascending.
 * 2. Sort large random data descending.
 * 3. Sort an empty input.
 * 4. Sort a single element.
 * 5. Sort two elements.
 * 6. Sort already sorted data.
 * 7. Sort reverse sorted data.
 * 8. Sort all identical data.
 *****************************************************************/

TEST_F(RS_SortTest, RadixSortRandom) {
    const RSize count = 100000;
    auto h_keys = getRandomData(count);
    auto h_values = getRandomData(count);

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_RadixSortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, true);

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, RadixSortDescending) {
    const RSize count = 50000;
    auto h_keys = getRandomData(count);
    auto h_values = getRandomData(count);

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_RadixSortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, false);

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    for (RSize i = 1; i < count; ++i) {
        EXPECT_GE(h_sorted_keys[i - 1], h_sorted_keys[i]);
    }

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, RadixSortEmpty) {
    const RSize count = 0;
    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_RadixSortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, true);

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, RadixSortSingle) {
    const RSize count = 1;
    std::vector<int> h_keys = {42};
    std::vector<int> h_values = {100};

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_RadixSortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, true);

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_EQ(h_sorted_keys[0], 42);
    EXPECT_EQ(h_sorted_vals[0], 100);

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, RadixSortTwo) {
    const RSize count = 2;
    std::vector<int> h_keys = {50, 10};
    std::vector<int> h_values = {100, 200};

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_RadixSortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, true);

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_EQ(h_sorted_keys[0], 10);
    EXPECT_EQ(h_sorted_keys[1], 50);
    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, RadixSortAlreadySorted) {
    const RSize count = 10000;
    std::vector<int> h_keys(count);
    std::vector<int> h_values(count);
    for (RSize i = 0; i < count; ++i) {
        h_keys[i] = static_cast<int>(i);
        h_values[i] = static_cast<int>(i * 2);
    }

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_RadixSortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, true);

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_sorted_keys[i], static_cast<int>(i));
    }

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, RadixSortReverseSorted) {
    const RSize count = 10000;
    std::vector<int> h_keys(count);
    std::vector<int> h_values(count);
    for (RSize i = 0; i < count; ++i) {
        h_keys[i] = static_cast<int>(count - i - 1);
        h_values[i] = static_cast<int>(i);
    }

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_RadixSortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, true);

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_sorted_keys[i], static_cast<int>(i));
    }

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, RadixSortAllIdentical) {
    const RSize count = 10000;
    std::vector<int> h_keys(count, 42);
    std::vector<int> h_values = getRandomData(count);

    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    RS_RadixSortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, true);

    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_sorted_keys[i], 42);
    }

    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}
