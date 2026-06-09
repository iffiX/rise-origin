#include <gtest/gtest.h>

#include <algorithm>  // for std::is_sorted
#include <random>
#include <vector>

#include "rise_origin/device/algorithm/rs_sort.h"  // Your merge/radix sort classes
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/utils/rs_def.h"

using namespace rise;
using namespace rise::device;

struct CompareOp {
    HOST_DEVICE_QUALIFIER bool operator()(int lhs, int rhs) { return lhs < rhs; }
};

class RS_SortTest : public ::testing::Test {
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

    // Helper function to generate random integer data in the range [0, 100000].
    std::vector<int> getRandomData(RSize size) {
        static std::mt19937 rng(12345);  // fixed seed for reproducibility
        std::uniform_int_distribution<int> dist(0, 100000);
        std::vector<int> data(size);
        for (RSize i = 0; i < size; ++i) {
            data[i] = dist(rng);
        }
        return data;
    }

    RS_MemoryCudaProvider provider;
    RS_MemoryCPUProvider host_provider;
};

TEST_F(RS_SortTest, MergeSortPairsGPU) {
    RSize count = 100000;
    auto h_keys = getRandomData(count);
    auto h_values = getRandomData(count);

    // Copy data to GPU arrays
    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    // Prepare output arrays
    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    // Sort using RS_SortPairs (merge sort)
    RS_SortPairs sorter(&provider);
    sorter.sort(
        d_keys, d_values, d_out_keys, d_out_values, CompareOp()
        // ascending
    );

    // Copy results back to host and verify
    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());

    // Check that keys are sorted ascending
    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));

    // Clean up
    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, MergeSortPairsCPU) {
    RSize count = 50000;
    auto h_keys = getRandomData(count);
    auto h_values = getRandomData(count);

    // Copy data to CPU arrays
    RS_Array<int, 1> h_keys_arr({count}, &host_provider);
    RS_Array<int, 1> h_values_arr({count}, &host_provider);
    h_keys_arr.write(h_keys.data());
    h_values_arr.write(h_values.data());

    // Prepare output arrays
    RS_Array<int, 1> h_out_keys({count}, &host_provider);
    RS_Array<int, 1> h_out_values({count}, &host_provider);

    // Sort using RS_SortPairs (merge sort) on CPU
    RS_SortPairs sorter(&host_provider);
    sorter.sort(h_keys_arr, h_values_arr, h_out_keys, h_out_values, CompareOp());

    // Copy results back to host and verify
    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    h_out_keys.read(h_sorted_keys.data());
    h_out_values.read(h_sorted_vals.data());
    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));

    // Clean up
    h_keys_arr.free();
    h_values_arr.free();
    h_out_keys.free();
    h_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, RadixSortPairsGPU) {
    RSize count = 100000;
    auto h_keys = getRandomData(count);
    auto h_values = getRandomData(count);

    // Copy data to GPU arrays
    RS_Array<int, 1> d_keys({count}, &provider);
    RS_Array<int, 1> d_values({count}, &provider);
    d_keys.write(h_keys.data());
    d_values.write(h_values.data());

    // Prepare output arrays
    RS_Array<int, 1> d_out_keys({count}, &provider);
    RS_Array<int, 1> d_out_values({count}, &provider);

    // Sort using RS_RadixSortPairs
    RS_RadixSortPairs sorter(&provider);
    sorter.sort(d_keys, d_values, d_out_keys, d_out_values, /*is_ascending=*/true);

    // Copy results back to host and verify
    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    d_out_keys.read(h_sorted_keys.data());
    d_out_values.read(h_sorted_vals.data());
    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));

    // Clean up
    d_keys.free();
    d_values.free();
    d_out_keys.free();
    d_out_values.free();
    sorter.free();
}

TEST_F(RS_SortTest, RadixSortPairsCPU) {
    RSize count = 30000;
    auto h_keys = getRandomData(count);
    auto h_values = getRandomData(count);

    // Copy data to CPU arrays
    RS_Array<int, 1> h_keys_arr({count}, &host_provider);
    RS_Array<int, 1> h_values_arr({count}, &host_provider);
    h_keys_arr.write(h_keys.data());
    h_values_arr.write(h_values.data());

    // Prepare output arrays
    RS_Array<int, 1> h_out_keys({count}, &host_provider);
    RS_Array<int, 1> h_out_values({count}, &host_provider);

    // Sort using RS_RadixSortPairs (CPU fallback)
    RS_RadixSortPairs sorter(&host_provider);
    sorter.sort(
        h_keys_arr,
        h_values_arr,
        h_out_keys,
        h_out_values,
        /*is_ascending=*/true
    );

    // Copy results back to host and verify
    std::vector<int> h_sorted_keys(count);
    std::vector<int> h_sorted_vals(count);
    h_out_keys.read(h_sorted_keys.data());
    h_out_values.read(h_sorted_vals.data());
    EXPECT_TRUE(std::is_sorted(h_sorted_keys.begin(), h_sorted_keys.end()));

    // Clean up
    h_keys_arr.free();
    h_values_arr.free();
    h_out_keys.free();
    h_out_values.free();
    sorter.free();
}
