#include <gtest/gtest.h>

#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_Vector is a dynamic array container that tracks a used size
 * separately from its allocated capacity, designed for efficient
 * CPU and GPU device memory management.
 *
 * - Construction & Initialization
 *   Constructor with capacity, init(), zero capacity
 *
 * - Capacity Management
 *   capacity(), size(), empty(), resize(), clear(), occupy()
 *
 * - Element Access (Host)
 *   write(), read(), writeSlice(), readSlice()
 *
 * - Element Access (Device)
 *   operator[], get(), set(), front(), back()
 *
 * - Modifiers
 *   fill(), fillSlice(), append(), insert(), erase(), eraseRange()
 *
 * - Operations
 *   copy(), move()
 *****************************************************************/

class RS_VectorTest : public ::testing::Test {
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
 * Test cases for RS_Vector construction and initialization
 *
 * Test cases:
 * 1. Construct with valid capacity.
 * 2. Construct with zero capacity.
 * 3. Init with provider only.
 * 4. Init with capacity.
 *****************************************************************/

TEST_F(RS_VectorTest, CreateWithValidCapacity) {
    RS_Vector<int> vec(10, &provider);
    EXPECT_EQ(vec.capacity(), 10);
    EXPECT_EQ(vec.size(), 0);
    EXPECT_NE(vec.ptr(), nullptr);
    EXPECT_TRUE(vec.empty());
    vec.free();
}

TEST_F(RS_VectorTest, CreateWithZeroCapacity) {
    RS_Vector<int> vec(0, &provider);
    EXPECT_EQ(vec.capacity(), 0);
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.ptr(), nullptr);
    EXPECT_TRUE(vec.empty());
    vec.free();
}

TEST_F(RS_VectorTest, InitWithProvider) {
    RS_Vector<int> vec;
    vec.init(&provider);
    EXPECT_EQ(vec.capacity(), 0);
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.ptr(), nullptr);
    vec.free();
}

TEST_F(RS_VectorTest, InitWithCapacity) {
    RS_Vector<int> vec;
    vec.init(5, &provider);
    EXPECT_EQ(vec.capacity(), 5);
    EXPECT_EQ(vec.size(), 0);
    EXPECT_NE(vec.ptr(), nullptr);
    vec.free();
}

/*****************************************************************
 * Test cases for RS_Vector::resize()
 *
 * Test cases:
 * 1. Resize up, capacity grows and data is preserved.
 * 2. Resize down, capacity shrinks and used size is clamped.
 * 3. Resize to zero.
 *****************************************************************/

TEST_F(RS_VectorTest, ResizeUp) {
    RS_Vector<int> vec(5, &provider);
    int host_data[5] = {1, 2, 3, 4, 5};
    vec.write(host_data);
    EXPECT_EQ(vec.size(), 5);

    vec.resize(10, true);
    EXPECT_EQ(vec.capacity(), 10);
    EXPECT_EQ(vec.size(), 5);

    int output[5] = {0};
    vec.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i], host_data[i]);
    }
    vec.free();
}

TEST_F(RS_VectorTest, ResizeDown) {
    RS_Vector<int> vec(8, &provider);
    int host_data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    vec.write(host_data);
    EXPECT_EQ(vec.size(), 8);

    vec.resize(4, true);
    EXPECT_EQ(vec.capacity(), 4);
    EXPECT_EQ(vec.size(), 4);

    int output[4] = {0};
    vec.read(output);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(output[i], host_data[i]);
    }
    vec.free();
}

TEST_F(RS_VectorTest, ResizeToZero) {
    RS_Vector<int> vec(5, &provider);
    vec.fill(7);
    vec.resize(0, true);
    EXPECT_EQ(vec.capacity(), 0);
    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());
    vec.free();
}

/*****************************************************************
 * Test cases for RS_Vector fill, write, read, and slices
 *
 * Test cases:
 * 1. Fill entire capacity.
 * 2. Fill with zero value.
 * 3. Write and read.
 * 4. Write slice at offset.
 * 5. Read slice at offset.
 * 6. Fill slice at offset.
 *****************************************************************/

TEST_F(RS_VectorTest, FillEntireCapacity) {
    RS_Vector<int> vec(5, &provider);
    vec.fill(123);
    EXPECT_EQ(vec.size(), 5);

    int output[5] = {0};
    vec.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i], 123);
    }
    vec.free();
}

TEST_F(RS_VectorTest, FillWithZero) {
    RS_Vector<int> vec(5, &provider);
    vec.fill(0);
    EXPECT_EQ(vec.size(), 5);

    int output[5] = {1, 1, 1, 1, 1};
    vec.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i], 0);
    }
    vec.free();
}

TEST_F(RS_VectorTest, WriteAndRead) {
    RS_Vector<int> vec(5, &provider);
    int host_data[5] = {10, 20, 30, 40, 50};
    vec.write(host_data);
    EXPECT_EQ(vec.size(), 5);

    int output[5] = {0};
    vec.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i], host_data[i]);
    }
    vec.free();
}

TEST_F(RS_VectorTest, WriteSlice) {
    RS_Vector<int> vec(2, &provider);
    int init_data[2] = {1, 2};
    vec.write(init_data);

    vec.resize(5);

    int extra[3] = {100, 200, 300};
    vec.writeSlice(extra, 3, 2);
    EXPECT_EQ(vec.size(), 5);

    int output[5] = {0};
    vec.read(output);
    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 2);
    EXPECT_EQ(output[2], 100);
    EXPECT_EQ(output[3], 200);
    EXPECT_EQ(output[4], 300);
    vec.free();
}

TEST_F(RS_VectorTest, ReadSlice) {
    RS_Vector<int> vec(5, &provider);
    int host_data[5] = {10, 20, 30, 40, 50};
    vec.write(host_data);

    int output[2] = {0};
    vec.readSlice(output, 2, 2);
    EXPECT_EQ(output[0], 30);
    EXPECT_EQ(output[1], 40);
    vec.free();
}

TEST_F(RS_VectorTest, FillSlice) {
    RS_Vector<int> vec(5, &provider);
    int host_data[5] = {1, 2, 3, 4, 5};
    vec.write(host_data);

    vec.fillSlice(99, 2, 2);

    int output[5] = {0};
    vec.read(output);
    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 2);
    EXPECT_EQ(output[2], 99);
    EXPECT_EQ(output[3], 99);
    EXPECT_EQ(output[4], 5);
    vec.free();
}

/*****************************************************************
 * Test cases for RS_Vector copy and move
 *
 * Test cases:
 * 1. Shallow copy shares the same data pointer.
 * 2. Copy resizes the destination to fit the source.
 * 3. Move to a different provider.
 *****************************************************************/

TEST_F(RS_VectorTest, ShallowCopyConstructor) {
    RS_Vector<int> original(5, &provider);
    int host_data[5] = {10, 20, 30, 40, 50};
    original.write(host_data);

    RS_Vector<int> copy_constructed(original);
    EXPECT_EQ(original.data.ptr(), copy_constructed.data.ptr());
    EXPECT_EQ(original.size(), copy_constructed.size());
    EXPECT_EQ(original.capacity(), copy_constructed.capacity());

    original.free();
}

TEST_F(RS_VectorTest, CopyMethod) {
    RS_Vector<int> src(4, &provider);
    int host_data[4] = {7, 8, 9, 10};
    src.write(host_data);

    RS_Vector<int> dst(2, &provider);
    dst.copy(src);
    EXPECT_EQ(dst.capacity(), 4);
    EXPECT_EQ(dst.size(), 4);

    int output[4] = {0};
    dst.read(output);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(output[i], host_data[i]);
    }
    src.free();
    dst.free();
}

TEST_F(RS_VectorTest, MoveToDifferentProvider) {
    RS_Vector<int> src(4, &provider);
    int host_data[4] = {55, 66, 77, 88};
    src.write(host_data);

    RS_Vector<int> moved = src.move(&host_provider);
    EXPECT_EQ(moved.capacity(), 4);
    EXPECT_EQ(moved.size(), 4);

    int output[4] = {0};
    moved.read(output);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(output[i], host_data[i]);
    }
    src.free();
    moved.free();
}

/*****************************************************************
 * Test cases for RS_Vector insert, append, and erase
 *
 * Test cases:
 * 1. Insert in the middle shifts elements right.
 * 2. Append grows capacity when full.
 * 3. Erase a range shifts the tail down.
 *****************************************************************/

TEST_F(RS_VectorTest, InsertMiddle) {
    RS_Vector<int> vec(3, &provider);
    int init_data[3] = {1, 3, 5};
    vec.write(init_data);

    vec.insert(2, 1);
    EXPECT_GE(vec.capacity(), 4u);
    EXPECT_EQ(vec.size(), 4);

    int output[4] = {0};
    vec.read(output);
    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 2);
    EXPECT_EQ(output[2], 3);
    EXPECT_EQ(output[3], 5);
    vec.free();
}

TEST_F(RS_VectorTest, InsertMiddleLarge) {
    constexpr int n = 10000;
    RS_Vector<int> vec(n, &provider);
    std::vector<int> init_data(n);
    for (int i = 0; i < n; i++) init_data[i] = i;
    vec.write(init_data.data());

    vec.insert(-1, n / 2);
    EXPECT_EQ(vec.size(), n + 1);

    std::vector<int> output(n + 1, 0);
    vec.read(output.data());
    for (int i = 0; i < n / 2; i++) EXPECT_EQ(output[i], i);
    EXPECT_EQ(output[n / 2], -1);
    for (int i = n / 2; i < n; i++) EXPECT_EQ(output[i + 1], i);
    vec.free();
}

TEST_F(RS_VectorTest, Append) {
    RS_Vector<int> vec(2, &provider);
    EXPECT_EQ(vec.size(), 0);

    vec.append(10);
    EXPECT_EQ(vec.size(), 1);
    vec.append(20);
    EXPECT_EQ(vec.size(), 2);

    vec.append(30);
    EXPECT_GE(vec.capacity(), 3u);
    EXPECT_EQ(vec.size(), 3);

    int output[3] = {0};
    vec.read(output);
    EXPECT_EQ(output[0], 10);
    EXPECT_EQ(output[1], 20);
    EXPECT_EQ(output[2], 30);
    vec.free();
}

TEST_F(RS_VectorTest, EraseRange) {
    RS_Vector<int> vec(10, &provider);
    int data[6] = {10, 20, 30, 40, 50, 60};
    vec.writeSlice(data, 6, 0);

    vec.eraseRange(2, 4);
    EXPECT_EQ(vec.size(), 4);

    int output[4] = {0};
    vec.read(output);
    EXPECT_EQ(output[0], 10);
    EXPECT_EQ(output[1], 20);
    EXPECT_EQ(output[2], 50);
    EXPECT_EQ(output[3], 60);

    vec.eraseRange(0, 2);
    EXPECT_EQ(vec.size(), 2);
    vec.read(output);
    EXPECT_EQ(output[0], 50);
    EXPECT_EQ(output[1], 60);
    vec.free();
}

TEST_F(RS_VectorTest, EraseRangeLarge) {
    constexpr int n = 10000;
    RS_Vector<int> vec(n, &provider);
    std::vector<int> init_data(n);
    for (int i = 0; i < n; i++) init_data[i] = i;
    vec.write(init_data.data());

    vec.eraseRange(n / 4, 3 * n / 4);
    EXPECT_EQ(vec.size(), n / 2);

    std::vector<int> output(n / 2, 0);
    vec.read(output.data());
    for (int i = 0; i < n / 4; i++) EXPECT_EQ(output[i], i);
    for (int i = n / 4; i < n / 2; i++) EXPECT_EQ(output[i], i + n / 2);
    vec.free();
}

/*****************************************************************
 * Test cases for RS_Vector device-side access
 *
 * Test cases:
 * 1. operator[], get(), set(), front(), and back() on the device.
 *****************************************************************/

class RSCF_VectorDeviceAccess : public RS_ComputeFunction<RSCF_VectorDeviceAccess> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_Vector<int> vec, RS_Vector<int> results) {
        if (index1D() == 0) {
            results[0] = vec[0];
            results[1] = vec[2];
            results[2] = vec.get(4);
            vec.set(3, 999);
            results[3] = vec[3];
            results[4] = vec.front();
            results[5] = vec.back();
        }
    }
};

TEST_F(RS_VectorTest, DeviceAccess) {
    RS_Vector<int> vec(5, &provider);
    int data[5] = {10, 20, 30, 40, 50};
    vec.write(data);

    RS_Vector<int> results(6, &provider);
    results.fill(0);

    provider.run<RSCF_VectorDeviceAccess>({1}, vec, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    EXPECT_EQ(host_results[0], 10);   // vec[0]
    EXPECT_EQ(host_results[1], 30);   // vec[2]
    EXPECT_EQ(host_results[2], 50);   // vec.get(4)
    EXPECT_EQ(host_results[3], 999);  // vec.set(3, 999)
    EXPECT_EQ(host_results[4], 10);   // vec.front()
    EXPECT_EQ(host_results[5], 50);   // vec.back()

    vec.free();
    results.free();
    host_results.free();
}
