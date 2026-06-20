#include <gtest/gtest.h>

#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/algorithm/rs_copy.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_soa.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * This test suite covers copying between different containers:
 * - StructOfArrays and Array1D-like containers
 * - Array1D-like containers to Array1D-like containers
 * - StructOfArrays to StructOfArrays
 * - 2D array to 2D array with copy2D
 *****************************************************************/

struct TestStruct {
    int x;
    float y;
    double z;
};

REFL_AUTO(type(TestStruct), field(x), field(y), field(z))

class RS_CopyTest : public ::testing::Test {
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
 * Test cases for copying from Array to StructOfArrays
 *
 * Test cases:
 * 1. Copy a whole array into a SoA.
 * 2. Copy a slice with source and destination offsets.
 *****************************************************************/

TEST_F(RS_CopyTest, CopyArrayToSoA) {
    const RSize count = 1000;

    RS_Array<TestStruct, 1> src_array({count}, &provider);
    std::vector<TestStruct> host_data(count);
    for (RSize i = 0; i < count; ++i) {
        host_data[i] = {
            static_cast<int>(i), static_cast<float>(i * 1.5f), static_cast<double>(i * 2.5)
        };
    }
    src_array.write(host_data.data());

    RS_StructOfArrays<TestStruct> dst_soa(count, &provider);

    copy(dst_soa, src_array, count);

    std::vector<TestStruct> result(count);
    dst_soa.read(result.data());

    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(result[i].x, host_data[i].x);
        EXPECT_FLOAT_EQ(result[i].y, host_data[i].y);
        EXPECT_DOUBLE_EQ(result[i].z, host_data[i].z);
    }

    src_array.free();
    dst_soa.free();
}

TEST_F(RS_CopyTest, CopyArrayToSoAWithOffset) {
    const RSize count = 1000;
    const RSize offset = 100;
    const RSize copy_count = 500;

    RS_Array<TestStruct, 1> src_array({count}, &provider);
    std::vector<TestStruct> host_data(count);
    for (RSize i = 0; i < count; ++i) {
        host_data[i] = {
            static_cast<int>(i), static_cast<float>(i * 1.5f), static_cast<double>(i * 2.5)
        };
    }
    src_array.write(host_data.data());

    RS_StructOfArrays<TestStruct> dst_soa(count, &provider);

    copy(dst_soa, src_array, copy_count, offset, offset);

    std::vector<TestStruct> result(count);
    dst_soa.read(result.data());

    for (RSize i = 0; i < copy_count; ++i) {
        EXPECT_EQ(result[offset + i].x, host_data[offset + i].x);
        EXPECT_FLOAT_EQ(result[offset + i].y, host_data[offset + i].y);
        EXPECT_DOUBLE_EQ(result[offset + i].z, host_data[offset + i].z);
    }

    src_array.free();
    dst_soa.free();
}

/*****************************************************************
 * Test cases for copying from StructOfArrays to Array
 *
 * Test cases:
 * 1. Copy a whole SoA into an array.
 * 2. Copy a slice with source and destination offsets.
 *****************************************************************/

TEST_F(RS_CopyTest, CopySoAToArray) {
    const RSize count = 1000;

    RS_StructOfArrays<TestStruct> src_soa(count, &provider);
    std::vector<TestStruct> host_data(count);
    for (RSize i = 0; i < count; ++i) {
        host_data[i] = {
            static_cast<int>(i), static_cast<float>(i * 1.5f), static_cast<double>(i * 2.5)
        };
    }
    src_soa.write(host_data.data());

    RS_Array<TestStruct, 1> dst_array({count}, &provider);

    copy(dst_array, src_soa, count);

    std::vector<TestStruct> result(count);
    dst_array.read(result.data());

    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(result[i].x, host_data[i].x);
        EXPECT_FLOAT_EQ(result[i].y, host_data[i].y);
        EXPECT_DOUBLE_EQ(result[i].z, host_data[i].z);
    }

    src_soa.free();
    dst_array.free();
}

TEST_F(RS_CopyTest, CopySoAToArrayWithOffset) {
    const RSize count = 1000;
    const RSize offset = 100;
    const RSize copy_count = 500;

    RS_StructOfArrays<TestStruct> src_soa(count, &provider);
    std::vector<TestStruct> host_data(count);
    for (RSize i = 0; i < count; ++i) {
        host_data[i] = {
            static_cast<int>(i), static_cast<float>(i * 1.5f), static_cast<double>(i * 2.5)
        };
    }
    src_soa.write(host_data.data());

    RS_Array<TestStruct, 1> dst_array({count}, &provider);

    copy(dst_array, src_soa, copy_count, offset, offset);

    std::vector<TestStruct> result(count);
    dst_array.read(result.data());

    for (RSize i = 0; i < copy_count; ++i) {
        EXPECT_EQ(result[offset + i].x, host_data[offset + i].x);
        EXPECT_FLOAT_EQ(result[offset + i].y, host_data[offset + i].y);
        EXPECT_DOUBLE_EQ(result[offset + i].z, host_data[offset + i].z);
    }

    src_soa.free();
    dst_array.free();
}

/*****************************************************************
 * Test cases for copying from Array to Array
 *
 * Test cases:
 * 1. Copy a whole array into another array.
 * 2. Copy a slice with source and destination offsets.
 *****************************************************************/

TEST_F(RS_CopyTest, CopyArrayToArray) {
    const RSize count = 1000;

    RS_Array<TestStruct, 1> src_array({count}, &provider);
    std::vector<TestStruct> host_data(count);
    for (RSize i = 0; i < count; ++i) {
        host_data[i] = {
            static_cast<int>(i), static_cast<float>(i * 1.5f), static_cast<double>(i * 2.5)
        };
    }
    src_array.write(host_data.data());

    RS_Array<TestStruct, 1> dst_array({count}, &provider);

    copy(dst_array, src_array, count);

    std::vector<TestStruct> result(count);
    dst_array.read(result.data());

    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(result[i].x, host_data[i].x);
        EXPECT_FLOAT_EQ(result[i].y, host_data[i].y);
        EXPECT_DOUBLE_EQ(result[i].z, host_data[i].z);
    }

    src_array.free();
    dst_array.free();
}

TEST_F(RS_CopyTest, CopyArrayToArrayWithOffset) {
    const RSize count = 1000;
    const RSize offset = 100;
    const RSize copy_count = 500;

    RS_Array<TestStruct, 1> src_array({count}, &provider);
    std::vector<TestStruct> host_data(count);
    for (RSize i = 0; i < count; ++i) {
        host_data[i] = {
            static_cast<int>(i), static_cast<float>(i * 1.5f), static_cast<double>(i * 2.5)
        };
    }
    src_array.write(host_data.data());

    RS_Array<TestStruct, 1> dst_array({count}, &provider);

    copy(dst_array, src_array, copy_count, offset, offset);

    std::vector<TestStruct> result(count);
    dst_array.read(result.data());

    for (RSize i = 0; i < copy_count; ++i) {
        EXPECT_EQ(result[offset + i].x, host_data[offset + i].x);
        EXPECT_FLOAT_EQ(result[offset + i].y, host_data[offset + i].y);
        EXPECT_DOUBLE_EQ(result[offset + i].z, host_data[offset + i].z);
    }

    src_array.free();
    dst_array.free();
}

/*****************************************************************
 * Test cases for copying from StructOfArrays to StructOfArrays
 *
 * Test cases:
 * 1. Copy a whole SoA into another SoA.
 * 2. Copy a slice with source and destination offsets.
 *****************************************************************/

TEST_F(RS_CopyTest, CopySoAToSoA) {
    const RSize count = 1000;

    RS_StructOfArrays<TestStruct> src_soa(count, &provider);
    std::vector<TestStruct> host_data(count);
    for (RSize i = 0; i < count; ++i) {
        host_data[i] = {
            static_cast<int>(i), static_cast<float>(i * 1.5f), static_cast<double>(i * 2.5)
        };
    }
    src_soa.write(host_data.data());

    RS_StructOfArrays<TestStruct> dst_soa(count, &provider);

    copy(dst_soa, src_soa, count);

    std::vector<TestStruct> result(count);
    dst_soa.read(result.data());

    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(result[i].x, host_data[i].x);
        EXPECT_FLOAT_EQ(result[i].y, host_data[i].y);
        EXPECT_DOUBLE_EQ(result[i].z, host_data[i].z);
    }

    src_soa.free();
    dst_soa.free();
}

TEST_F(RS_CopyTest, CopySoAToSoAWithOffset) {
    const RSize count = 1000;
    const RSize offset = 100;
    const RSize copy_count = 500;

    RS_StructOfArrays<TestStruct> src_soa(count, &provider);
    std::vector<TestStruct> host_data(count);
    for (RSize i = 0; i < count; ++i) {
        host_data[i] = {
            static_cast<int>(i), static_cast<float>(i * 1.5f), static_cast<double>(i * 2.5)
        };
    }
    src_soa.write(host_data.data());

    RS_StructOfArrays<TestStruct> dst_soa(count, &provider);

    copy(dst_soa, src_soa, copy_count, offset, offset);

    std::vector<TestStruct> result(count);
    dst_soa.read(result.data());

    for (RSize i = 0; i < copy_count; ++i) {
        EXPECT_EQ(result[offset + i].x, host_data[offset + i].x);
        EXPECT_FLOAT_EQ(result[offset + i].y, host_data[offset + i].y);
        EXPECT_DOUBLE_EQ(result[offset + i].z, host_data[offset + i].z);
    }

    src_soa.free();
    dst_soa.free();
}

/*****************************************************************
 * Test cases for copy2D
 *
 * Test cases:
 * 1. Copy a 2D array into a larger 2D array, preserving data.
 *****************************************************************/

TEST_F(RS_CopyTest, Copy2DIntoLarger) {
    const RSize src_dim = 4;
    const RSize dst_dim = 8;

    RS_Array<int, 2> src({src_dim, src_dim}, &provider);
    std::vector<int> src_host(src_dim * src_dim);
    for (RSize i = 0; i < src_dim * src_dim; ++i) {
        src_host[i] = static_cast<int>(i);
    }
    src.write(src_host.data());

    RS_Array<int, 2> dst({dst_dim, dst_dim}, &provider);
    dst.fill(-1);

    copy2D(dst, src, src_dim);

    std::vector<int> dst_host(dst_dim * dst_dim);
    dst.read(dst_host.data());

    for (RSize row = 0; row < dst_dim; ++row) {
        for (RSize col = 0; col < dst_dim; ++col) {
            int value = dst_host[row * dst_dim + col];
            if (row < src_dim && col < src_dim) {
                EXPECT_EQ(value, src_host[row * src_dim + col]);
            } else {
                EXPECT_EQ(value, -1);
            }
        }
    }

    src.free();
    dst.free();
}
