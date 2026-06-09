#include <gtest/gtest.h>

#include "rise_origin/device/algorithm/rs_search.h"

using namespace rise;
using namespace rise::device;

class RS_SearchTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize CUDA stream and memory provider
        provider.init({RSE_DEVICE_TYPE_CUDA, 0});
        host_provider.init(0);
    }

    void TearDown() override {
        provider.free();
        host_provider.free();
    }

    RS_MemoryCudaProvider provider;
    RS_MemoryCPUProvider host_provider;
};

class BinaryPrefixSearchTestFunction : public RS_ComputeFunction<BinaryPrefixSearchTestFunction> {
   public:
    template <typename ResultArray, typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(ResultArray result, Args... args) {
        result[index1D()] = RSCF_BinaryPrefixSearchWithArray1DLike::run(args...);
    }
};

TEST_F(RS_SearchTest, BinaryPrefixSearch) {
    // Example test for RS_BinaryPrefixSearch
    RSize prefix_sums[] = {1, 3, 6, 10, 15};  // Prefix sums as defined in the function comment
    RSize num_elements = sizeof(prefix_sums) / sizeof(RSize);
    RIndex search_index = 7;  // Example index to search for

    // Allocate device memory for prefix sums
    auto array = RS_Array<RSize, 1>({num_elements}, &provider);
    auto result_array = RS_Array<RSCF_BinaryPrefixSearch::SearchResult, 1>({1}, &provider);
    array.write(prefix_sums);

    // Prepare and run the compute function
    provider.run<BinaryPrefixSearchTestFunction>({1}, result_array, search_index, array);

    auto host_result_array = result_array.move(&host_provider);
    // Check the result
    // Expected result based on the provided searchIndex and prefixSums
    EXPECT_EQ(host_result_array[0].index, 3);  // The prefix sum index the searchIndex belongs to
    EXPECT_EQ(host_result_array[0].local_index, 1);  // The local index within the group

    // Cleanup
    array.free();
    result_array.free();
    host_result_array.free();
}

TEST_F(RS_SearchTest, BinaryPrefixOutOfBoundsSearch) {
    // Example test for RS_BinaryPrefixSearch
    RSize prefix_sums[] = {1, 3, 6, 10, 15};  // Prefix sums as defined in the function comment
    RSize num_elements = sizeof(prefix_sums) / sizeof(RSize);
    RIndex search_index = 20;  // Example index to search for

    // Allocate device memory for prefix sums
    auto array = RS_Array<RSize, 1>({num_elements}, &provider);
    auto result_array = RS_Array<RSCF_BinaryPrefixSearch::SearchResult, 1>({1}, &provider);
    array.write(prefix_sums);

    // Prepare and run the compute function
    provider.run<BinaryPrefixSearchTestFunction>({1}, result_array, search_index, array);

    auto host_result_array = result_array.move(&host_provider);
    // Check the result
    // Expected result based on the provided searchIndex and prefixSums
    EXPECT_EQ(
        host_result_array[0].index, RS_NULL_INDEX
    );  // The prefix sum index the searchIndex belongs to
    EXPECT_EQ(host_result_array[0].local_index, RS_NULL_INDEX);  // The local index within the group

    // Cleanup
    array.free();
    result_array.free();
    host_result_array.free();
}