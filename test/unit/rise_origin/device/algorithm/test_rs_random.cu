#include <gtest/gtest.h>

#include <cmath>

#include "rise_origin/device/algorithm/rs_random.h"
#include "rise_origin/device/data/rs_array.h"

using namespace rise;
using namespace rise::device;

class RS_RandomTest : public ::testing::Test {
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

TEST_F(RS_RandomTest, FillUniformFloat) {
    const RSize count = 1000000;
    RS_Random random(&provider);
    RS_Array<float, 1> test_array({count}, &provider);
    random.fillUniform(test_array, 0.0f, 1.0f, true);

    auto host_array = test_array.move(&host_provider);

    float sum = 0.0f;
    for (RSize i = 0; i < count; ++i) {
        ASSERT_GE(host_array[i], 0.0f);
        ASSERT_LT(host_array[i], 1.0f);
        sum += host_array[i];
    }
    float average = sum / count;
    EXPECT_NEAR(average, 0.5f, 0.01f);

    test_array.free();
    host_array.free();
    random.free();
}

TEST_F(RS_RandomTest, FillUniformFloatHost) {
    const RSize count = 1000000;
    RS_Random random(&host_provider);
    RS_Array<float, 1> test_array({count}, &host_provider);
    random.fillUniform(test_array, 0.0f, 1.0f, true);

    float sum = 0.0f;
    for (RSize i = 0; i < count; ++i) {
        ASSERT_GE(test_array[i], 0.0f);
        ASSERT_LT(test_array[i], 1.0f);
        sum += test_array[i];
    }
    float average = sum / count;
    EXPECT_NEAR(average, 0.5f, 0.01f);

    test_array.free();
    random.free();
}

TEST_F(RS_RandomTest, FillUniformInt) {
    const RSize count = 1000000;
    const int min = -5, max = 10;
    RS_Random random(&provider);
    RS_Array<int, 1> test_array({count}, &provider);
    random.fillUniform(test_array, min, max, true);

    auto host_array = test_array.move(&host_provider);

    double sum = 0.0;
    for (RSize i = 0; i < count; ++i) {
        ASSERT_GE(host_array[i], min);
        ASSERT_LT(host_array[i], max);
        sum += host_array[i];
    }
    double average = sum / count;
    double expected_average = (min + (max - 1)) / 2.0;  // max is exclusive
    EXPECT_NEAR(average, expected_average, 0.1);

    test_array.free();
    host_array.free();
    random.free();
}

TEST_F(RS_RandomTest, FillUniformIntHost) {
    const RSize count = 1000000;
    const int min = -5, max = 10;
    RS_Random random(&host_provider);
    RS_Array<int, 1> test_array({count}, &host_provider);
    random.fillUniform(test_array, min, max, true);

    double sum = 0.0;
    for (RSize i = 0; i < count; ++i) {
        ASSERT_GE(test_array[i], min);
        ASSERT_LT(test_array[i], max);
        sum += test_array[i];
    }
    double average = sum / count;
    double expected_average = (min + (max - 1)) / 2.0;  // max is exclusive
    EXPECT_NEAR(average, expected_average, 0.1);

    test_array.free();
    random.free();
}

TEST_F(RS_RandomTest, FillNormal) {
    const RSize count = 1000000;
    const float mean = 5.0f, stddev = 2.0f;
    RS_Random random(&provider);
    RS_Array<float, 1> test_array({count}, &provider);
    random.fillNormal(test_array, mean, stddev, true);

    auto host_array = test_array.move(&host_provider);

    float sum = 0.0f, sum_sq = 0.0f;
    for (RSize i = 0; i < count; ++i) {
        sum += host_array[i];
        sum_sq += host_array[i] * host_array[i];
    }
    float sample_mean = sum / count;
    float sample_var = (sum_sq / count) - (sample_mean * sample_mean);
    float sample_stddev = std::sqrt(sample_var);

    EXPECT_NEAR(sample_mean, mean, 0.1f);
    EXPECT_NEAR(sample_stddev, stddev, 0.1f);

    test_array.free();
    host_array.free();
    random.free();
}

TEST_F(RS_RandomTest, FillNormalHost) {
    const RSize count = 1000000;
    const float mean = 5.0f, stddev = 2.0f;
    RS_Random random(&host_provider);
    RS_Array<float, 1> test_array({count}, &host_provider);
    random.fillNormal(test_array, mean, stddev, true);

    float sum = 0.0f, sum_sq = 0.0f;
    for (RSize i = 0; i < count; ++i) {
        sum += test_array[i];
        sum_sq += test_array[i] * test_array[i];
    }
    float sample_mean = sum / count;
    float sample_var = (sum_sq / count) - (sample_mean * sample_mean);
    float sample_stddev = std::sqrt(sample_var);

    EXPECT_NEAR(sample_mean, mean, 0.1f);
    EXPECT_NEAR(sample_stddev, stddev, 0.1f);

    test_array.free();
    random.free();
}
