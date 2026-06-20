#include <gtest/gtest.h>

#include <cmath>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/algorithm/rs_random.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_Random generates random numbers using the Philox engine.
 *
 * - Uniform Distribution
 *   fillUniform for floats and integers, range bounds and mean
 *
 * - Normal Distribution
 *   fillNormal with a chosen mean and standard deviation
 *
 * - Statistical Validation
 *   Each test checks range bounds and approximate moments of the
 *   generated samples
 *****************************************************************/

class RS_RandomTest : public ::testing::Test {
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

/*****************************************************************
 * Test cases for uniform float distribution
 *
 * Test cases:
 * 1. Generated floats stay in range and average near the midpoint.
 *****************************************************************/

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

/*****************************************************************
 * Test cases for uniform integer distribution
 *
 * Test cases:
 * 1. Generated integers stay in range and average near the midpoint.
 *****************************************************************/

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
    double expected_average = (min + (max - 1)) / 2.0;
    EXPECT_NEAR(average, expected_average, 0.1);

    test_array.free();
    host_array.free();
    random.free();
}

/*****************************************************************
 * Test cases for normal distribution
 *
 * Test cases:
 * 1. Sample mean and standard deviation match the requested values.
 *****************************************************************/

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
