#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/algorithm/rs_generate.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * This test suite covers generateIndex which fills an array with
 * sequential indices starting from zero.
 *****************************************************************/

class RS_GenerateTest : public ::testing::Test {
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
 * Test cases for generateIndex
 *
 * Test cases:
 * 1. Fill a small array with sequential indices.
 * 2. Fill a large array and check sample positions.
 * 3. Empty array is handled without error.
 *****************************************************************/

TEST_F(RS_GenerateTest, GenerateIndex) {
    const RSize count = 1000;
    RS_Array<int, 1> array({count}, &provider);

    generateIndex(array);

    auto host_array = array.move(&host_provider);

    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(host_array[i], static_cast<int>(i));
    }

    array.free();
    host_array.free();
}

TEST_F(RS_GenerateTest, GenerateIndexLarge) {
    const RSize count = 100000;
    RS_Array<int, 1> array({count}, &provider);

    generateIndex(array);

    auto host_array = array.move(&host_provider);

    EXPECT_EQ(host_array[0], 0);
    EXPECT_EQ(host_array[count / 2], static_cast<int>(count / 2));
    EXPECT_EQ(host_array[count - 1], static_cast<int>(count - 1));
    EXPECT_EQ(host_array[100], 100);
    EXPECT_EQ(host_array[50000], 50000);
    EXPECT_EQ(host_array[99999], 99999);

    array.free();
    host_array.free();
}

TEST_F(RS_GenerateTest, GenerateIndexEmpty) {
    const RSize count = 0;
    RS_Array<int, 1> array({count}, &provider);

    EXPECT_NO_THROW(generateIndex(array));
    EXPECT_EQ(array.size(), 0);

    array.free();
}
