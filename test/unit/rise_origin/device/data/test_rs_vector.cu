#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_vector.h"

using namespace rise;
using namespace rise::device;

/**
 * @brief GTest fixture for RS_Vector tests.
 *
 * Similar to RS_ArrayTest, it sets up memory providers for GPU and CPU.
 */
class RS_VectorTest : public ::testing::Test {
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

/*****************************************************************
 * Basic construction tests
 * 1. Initialize with valid capacity.
 * 2. Initialize with zero capacity.
 *****************************************************************/

TEST_F(RS_VectorTest, CreateWithValidCapacity) {
    // Suppose we want a capacity of 10
    RS_Vector<int> vec(10, &provider);
    EXPECT_EQ(vec.capacity(), 10);
    EXPECT_EQ(vec.size(), 0);
    EXPECT_NE(vec.ptr(), nullptr);
    vec.free();
}

TEST_F(RS_VectorTest, CreateWithZeroCapacity) {
    RS_Vector<int> vec(0, &provider);
    EXPECT_EQ(vec.capacity(), 0);
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.ptr(), nullptr);
    vec.free();
}

/*****************************************************************
 * Test resizing:
 * 1. Resize up -> capacity grows, old data is preserved (if copy=true).
 * 2. Resize down -> capacity shrinks, used_elem_num is clamped.
 *****************************************************************/

TEST_F(RS_VectorTest, ResizeUp) {
    RS_Vector<int> vec(5, &provider);
    // Write some data
    int host_data[5] = {1, 2, 3, 4, 5};
    vec.write(host_data);
    EXPECT_EQ(vec.size(), 5);

    // Resize up to capacity 10
    vec.resize(10, true);
    EXPECT_EQ(vec.capacity(), 10);
    EXPECT_EQ(vec.size(), 5);

    // Read back data and verify
    int output[10] = {0};
    vec.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i], host_data[i]);
    }
    // The remaining slots are uninitialized from the original perspective,
    // but they remain allocated on device.

    vec.free();
}

TEST_F(RS_VectorTest, ResizeDown) {
    RS_Vector<int> vec(8, &provider);
    // Write data: used_elem_num = 8
    int host_data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    vec.write(host_data);
    EXPECT_EQ(vec.size(), 8);

    // Now resize down to 4
    vec.resize(4, true);
    EXPECT_EQ(vec.capacity(), 4);
    // used_elem_num should be clamped to 4
    EXPECT_EQ(vec.size(), 4);

    // Check data after clamping
    int output[4] = {0};
    vec.read(output);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(output[i], host_data[i]);
    }

    vec.free();
}

/*****************************************************************
 * Test fill, write, read, and writeSlice
 *****************************************************************/

TEST_F(RS_VectorTest, FillEntireCapacity) {
    RS_Vector<int> vec(5, &provider);
    vec.fill(123);

    EXPECT_EQ(vec.size(), 5);

    // Let's read back the entire capacity to see if it was indeed filled:
    int output[5] = {0};
    vec.readSlice(output, 5, 0);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i], 123);
    }

    vec.free();
}

TEST_F(RS_VectorTest, WriteAndRead) {
    RS_Vector<int> vec(5, &provider);
    int host_data[5] = {10, 20, 30, 40, 50};
    vec.write(host_data);
    EXPECT_EQ(vec.size(), 5);

    // read back
    int output[5] = {0};
    vec.read(output);
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(output[i], host_data[i]);
    }

    // The rest are unused.
    vec.free();
}

TEST_F(RS_VectorTest, WriteSlice) {
    RS_Vector<int> vec(2, &provider);
    int init_data[2] = {1, 2};
    vec.write(init_data);

    vec.resize(5);

    // write slice of size 3 at offset 2
    int extra[3] = {100, 200, 300};
    vec.writeSlice(extra, 3, 2);

    // used_elem_num should become 5.
    EXPECT_EQ(vec.size(), 5);

    // read back entire used size
    int output[5] = {0};
    vec.read(output);
    // Expected: [1, 2, 100, 200, 300]
    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 2);
    EXPECT_EQ(output[2], 100);
    EXPECT_EQ(output[3], 200);
    EXPECT_EQ(output[4], 300);

    vec.free();
}

/*****************************************************************
 * Test shallow copy and copy(...)
 *****************************************************************/

TEST_F(RS_VectorTest, ShallowCopyConstructor) {
    RS_Vector<int> original(5, &provider);
    int host_data[5] = {10, 20, 30, 40, 50};
    original.write(host_data);

    // Shallow copy
    RS_Vector<int> copy_constructed(original);
    // The data pointers should match (shallow).
    EXPECT_EQ(original.data.data, copy_constructed.data.data);
    EXPECT_EQ(original.size(), copy_constructed.size());
    EXPECT_EQ(original.capacity(), copy_constructed.capacity());

    // But we have to be careful: free on one affects the other.
    // Typically you'd want to test actual usage in real code.
    original.free();
}

TEST_F(RS_VectorTest, CopyMethod) {
    RS_Vector<int> src(4, &provider);
    int host_data[4] = {7, 8, 9, 10};
    src.write(host_data);

    RS_Vector<int> dst(2, &provider);
    // Now we use copy method, which resizes 'dst' to match 'src'
    dst.copy(src);
    // Check
    EXPECT_EQ(dst.capacity(), 4);
    EXPECT_EQ(dst.size(), 4);

    int output[4];
    dst.read(output);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(output[i], host_data[i]);
    }

    // Free to avoid conflicts.
    src.free();
    dst.free();
}

/*****************************************************************
 * Test insert, append, and erase
 *
 * Key checks:
 * - insert in middle -> shift
 * - capacity expansion if needed
 * - erase -> shift down from end
 *****************************************************************/

TEST_F(RS_VectorTest, InsertMiddle) {
    RS_Vector<int> vec(3, &provider);
    int init_data[3] = {1, 3, 5};
    vec.write(init_data);

    // Insert "2" at index=1: we'd have [1,2,3,5] logically,
    // but capacity=3 is fully used, so it must resize first.
    vec.insert(2, 1);
    EXPECT_GE(vec.capacity(), 4u);
    EXPECT_EQ(vec.size(), 4);

    // read back
    int output[10] = {0};
    vec.read(output);
    // Expect [1,2,3,5]
    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 2);
    EXPECT_EQ(output[2], 3);
    EXPECT_EQ(output[3], 5);

    vec.free();
}

TEST_F(RS_VectorTest, Append) {
    RS_Vector<int> vec(2, &provider);
    EXPECT_EQ(vec.size(), 0);

    vec.append(10);
    EXPECT_EQ(vec.size(), 1);
    vec.append(20);
    EXPECT_EQ(vec.size(), 2);

    // at capacity => must resize on next append
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

    // Erase [2,4) => removes elements at indices 2 and 3 (30,40)
    // so the new array becomes [10, 20, 50, 60]
    vec.eraseRange(2, 4);
    EXPECT_EQ(vec.size(), 4);

    int output[10] = {0};
    vec.read(output);
    EXPECT_EQ(output[0], 10);
    EXPECT_EQ(output[1], 20);
    EXPECT_EQ(output[2], 50);
    EXPECT_EQ(output[3], 60);

    // Erase [0,2) => removes elements at indices 0 and 1 (10,20),
    // left with [50, 60]
    vec.eraseRange(0, 2);
    EXPECT_EQ(vec.size(), 2);
    vec.read(output);
    EXPECT_EQ(output[0], 50);
    EXPECT_EQ(output[1], 60);

    vec.free();
}
