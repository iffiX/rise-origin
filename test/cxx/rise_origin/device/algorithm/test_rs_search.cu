#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/algorithm/rs_search.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * Binary prefix search finds which prefix sum group a query index
 * belongs to. The float variant finds the first cdf bucket strictly
 * greater than the query value.
 *
 * - Integer Prefix Search
 *   RSCF_BinaryPrefixSearchWithArray1DLike returns the group index
 *   and the local index within the group
 *
 * - Float CDF Search
 *   RSCF_BinaryPrefixSearchFloatWithArray1DLike returns the bucket
 *   index for a query value
 *
 * - Edge Cases
 *   Out of bounds index returns RS_NULL_INDEX
 *****************************************************************/

class RS_SearchTest : public ::testing::Test {
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

class RSCF_PrefixSearchProbe : public RS_ComputeFunction<RSCF_PrefixSearchProbe> {
   public:
    template <typename ResultArray, typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(ResultArray result, Args... args) {
        result[index1D()] = RSCF_BinaryPrefixSearchWithArray1DLike::run(args...);
    }
};

class RSCF_PrefixSearchFloatProbe : public RS_ComputeFunction<RSCF_PrefixSearchFloatProbe> {
   public:
    template <typename ResultArray, typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(ResultArray result, Args... args) {
        result[index1D()] = RSCF_BinaryPrefixSearchFloatWithArray1DLike<float>::run(args...);
    }
};

/*****************************************************************
 * Test cases for integer binary prefix search
 *
 * Test cases:
 * 1. Search a valid index inside the prefix sum bounds.
 * 2. Search an index above the last prefix sum value.
 *****************************************************************/

TEST_F(RS_SearchTest, BinaryPrefixSearch) {
    RSize prefix_sums[] = {1, 3, 6, 10, 15};
    RSize num_elements = sizeof(prefix_sums) / sizeof(RSize);
    RIndex search_index = 7;

    auto array = RS_Array<RSize, 1>({num_elements}, &provider);
    auto result_array = RS_Array<RSCF_BinaryPrefixSearch::SearchResult, 1>({1}, &provider);
    array.write(prefix_sums);

    provider.run<RSCF_PrefixSearchProbe>({1}, result_array, search_index, array);

    auto host_result_array = result_array.move(&host_provider);
    EXPECT_EQ(host_result_array[0].index, 3);
    EXPECT_EQ(host_result_array[0].local_index, 1);

    array.free();
    result_array.free();
    host_result_array.free();
}

TEST_F(RS_SearchTest, BinaryPrefixOutOfBoundsSearch) {
    RSize prefix_sums[] = {1, 3, 6, 10, 15};
    RSize num_elements = sizeof(prefix_sums) / sizeof(RSize);
    RIndex search_index = 20;

    auto array = RS_Array<RSize, 1>({num_elements}, &provider);
    auto result_array = RS_Array<RSCF_BinaryPrefixSearch::SearchResult, 1>({1}, &provider);
    array.write(prefix_sums);

    provider.run<RSCF_PrefixSearchProbe>({1}, result_array, search_index, array);

    auto host_result_array = result_array.move(&host_provider);
    EXPECT_EQ(host_result_array[0].index, RS_NULL_INDEX);
    EXPECT_EQ(host_result_array[0].local_index, RS_NULL_INDEX);

    array.free();
    result_array.free();
    host_result_array.free();
}

/*****************************************************************
 * Test cases for float cdf binary search
 *
 * Test cases:
 * 1. Query inside the cdf range maps to the matching bucket.
 * 2. Query below the first cdf entry maps to bucket zero.
 * 3. Query at or above the last cdf entry clamps to the last bucket.
 *****************************************************************/

TEST_F(RS_SearchTest, BinaryPrefixSearchFloat) {
    float cdf[] = {0.5f, 1.5f, 4.0f, 5.0f};
    RSize num_elements = sizeof(cdf) / sizeof(float);

    auto array = RS_Array<float, 1>({num_elements}, &provider);
    auto result_array = RS_Array<RIndex, 1>({1}, &provider);
    array.write(cdf);

    float query = 2.0f;
    provider.run<RSCF_PrefixSearchFloatProbe>({1}, result_array, query, array);

    auto host_result_array = result_array.move(&host_provider);
    EXPECT_EQ(host_result_array[0], 2);

    array.free();
    result_array.free();
    host_result_array.free();
}

TEST_F(RS_SearchTest, BinaryPrefixSearchFloatBelowFirst) {
    float cdf[] = {0.5f, 1.5f, 4.0f, 5.0f};
    RSize num_elements = sizeof(cdf) / sizeof(float);

    auto array = RS_Array<float, 1>({num_elements}, &provider);
    auto result_array = RS_Array<RIndex, 1>({1}, &provider);
    array.write(cdf);

    float query = 0.1f;
    provider.run<RSCF_PrefixSearchFloatProbe>({1}, result_array, query, array);

    auto host_result_array = result_array.move(&host_provider);
    EXPECT_EQ(host_result_array[0], 0);

    array.free();
    result_array.free();
    host_result_array.free();
}

TEST_F(RS_SearchTest, BinaryPrefixSearchFloatClampToLast) {
    float cdf[] = {0.5f, 1.5f, 4.0f, 5.0f};
    RSize num_elements = sizeof(cdf) / sizeof(float);

    auto array = RS_Array<float, 1>({num_elements}, &provider);
    auto result_array = RS_Array<RIndex, 1>({1}, &provider);
    array.write(cdf);

    float query = 100.0f;
    provider.run<RSCF_PrefixSearchFloatProbe>({1}, result_array, query, array);

    auto host_result_array = result_array.move(&host_provider);
    EXPECT_EQ(host_result_array[0], 3);

    array.free();
    result_array.free();
    host_result_array.free();
}
