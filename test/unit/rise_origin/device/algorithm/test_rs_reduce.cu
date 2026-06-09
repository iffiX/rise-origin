#include <gtest/gtest.h>

#include "rise_origin/device/algorithm/rs_reduce.h"

using namespace rise;
using namespace rise::device;

class RS_ReduceTest : public ::testing::Test {
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

TEST_F(RS_ReduceTest, Reduce) {
    RSize count = 100000;
    RS_Array<int, 1> test_array({count}, &provider);
    test_array.fill(1);
    RS_Reduce reduce(&provider);
    int result = reduce.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, count);

    test_array.free();
    reduce.free();
}

TEST_F(RS_ReduceTest, ReduceHost) {
    RSize count = 100000;
    RS_Array<int, 1> test_array({count}, &host_provider);
    test_array.fill(1);
    RS_Reduce reduce(&host_provider);
    int result = reduce.reduce<RS_ReduceSumOp>(test_array, 0);
    EXPECT_EQ(result, count);

    test_array.free();
    reduce.free();
}

TEST_F(RS_ReduceTest, ReduceByGroup) {
    std::vector<RSize> group_sizes1 = {1, 1024, 100000, 10000000};
    std::vector<RSize> group_sizes2 = {200000, 1238989};
    RSize total1 = std::accumulate(group_sizes1.begin(), group_sizes1.end(), 0);
    RSize total2 = std::accumulate(group_sizes2.begin(), group_sizes2.end(), 0);
    RS_Array<int, 1> test_array1({total1}, &provider);
    RS_Array<int, 1> test_array2({total2}, &provider);
    test_array1.fill(1);
    test_array2.fill(1);
    RS_ReduceByGroup reduce(&provider);
    auto result1 = reduce.reduce<RS_ReduceSumOp>(test_array1, group_sizes1);
    for (RSize i = 0; i < group_sizes1.size(); i++) {
        EXPECT_EQ(result1[i], group_sizes1[i]);
    }
    auto result2 = reduce.reduce<RS_ReduceSumOp>(test_array2, group_sizes2);
    for (RSize i = 0; i < group_sizes2.size(); i++) {
        EXPECT_EQ(result2[i], group_sizes2[i]);
    }

    test_array1.free();
    test_array2.free();
    reduce.free();
}

TEST_F(RS_ReduceTest, ReduceByGroupHost) {
    std::vector<RSize> group_sizes1 = {1, 1024, 100000, 10000000};
    std::vector<RSize> group_sizes2 = {200000, 1238989};
    RSize total1 = std::accumulate(group_sizes1.begin(), group_sizes1.end(), 0);
    RSize total2 = std::accumulate(group_sizes2.begin(), group_sizes2.end(), 0);
    RS_Array<int, 1> test_array1({total1}, &host_provider);
    RS_Array<int, 1> test_array2({total2}, &host_provider);
    test_array1.fill(1);
    test_array2.fill(1);
    RS_ReduceByGroup reduce(&host_provider);
    auto result1 = reduce.reduce<RS_ReduceSumOp>(test_array1, group_sizes1);
    for (RSize i = 0; i < group_sizes1.size(); i++) {
        EXPECT_EQ(result1[i], group_sizes1[i]);
    }
    auto result2 = reduce.reduce<RS_ReduceSumOp>(test_array2, group_sizes2);
    for (RSize i = 0; i < group_sizes2.size(); i++) {
        EXPECT_EQ(result2[i], group_sizes2[i]);
    }

    test_array1.free();
    test_array2.free();
    reduce.free();
}