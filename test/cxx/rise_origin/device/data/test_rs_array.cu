#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_Array is a multi-dimensional array container with dynamic
 * shape allocation, designed for efficient CPU and GPU device memory
 * management. Supports both row major and column major layouts.
 *
 * - Construction & Initialization
 *   Constructor with shape, free()
 *
 * - Memory Management
 *   resize() with optional data copy, fill()
 *
 * - Data Transfer
 *   write(), writeSlice(), read(), readSlice(), copy()
 *
 * - Indexing & Coordinates
 *   isOutOfBound(), indexToCoords(), coordsToIndex()
 *   Support for 2D, 3D, and 4D arrays
 *
 * - Layout Support
 *   Row major (default) and column major layouts
 *
 * - Static Array
 *   RS_StaticArray isOutOfBound(), indexToCoords(), coordsToIndex()
 *****************************************************************/

class RS_ArrayTest : public ::testing::Test {
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
 * Test cases for the RS_Array constructor
 *
 * Test cases:
 * 1. Initialize with valid shape.
 * 2. Initialize with zero dimensions.
 *****************************************************************/

TEST_F(RS_ArrayTest, InitializeWithValidShape) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    EXPECT_EQ(array.elem_num, 4 * 5 * 6);
    EXPECT_EQ(array.shape[0], 4);
    EXPECT_EQ(array.shape[1], 5);
    EXPECT_EQ(array.shape[2], 6);
    EXPECT_NE(array.data.ptr(), nullptr);
    array.free();
}

TEST_F(RS_ArrayTest, InitializeWithZeroDimensions) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);
    EXPECT_EQ(array.elem_num, 0);
    EXPECT_EQ(array.shape[0], 0);
    EXPECT_EQ(array.shape[1], 0);
    EXPECT_EQ(array.shape[2], 0);
    EXPECT_EQ(array.data.ptr(), nullptr);
    array.free();
}

/*****************************************************************
 * Test cases for the RS_Array::free()
 *
 * Test cases:
 * 1. Free non-empty array
 * 2. Free empty array
 * 3. Multiple frees
 * 4. Free after operations
 *****************************************************************/

TEST_F(RS_ArrayTest, FreeNonEmptyArray) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    EXPECT_NO_THROW(array.free());
    EXPECT_EQ(array.elem_num, 0);
    EXPECT_EQ(array.data.ptr(), nullptr);
}

TEST_F(RS_ArrayTest, FreeEmptyArray) {
    RS_Array<int, 3> array({0, 0, 0}, &provider);
    EXPECT_NO_THROW(array.free());
}

TEST_F(RS_ArrayTest, MultipleFrees) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    array.free();
    EXPECT_NO_THROW(array.free());
    EXPECT_EQ(array.elem_num, 0);
    EXPECT_EQ(array.data.ptr(), nullptr);
}

TEST_F(RS_ArrayTest, FreeAfterOperations) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    array.resize({8, 8, 8});
    array.free();
    EXPECT_EQ(array.elem_num, 0);
    EXPECT_EQ(array.data.ptr(), nullptr);
}

/*****************************************************************
 * Test cases for the RS_Array::resize()
 *
 * Test cases:
 * 1. Resize to larger shape without copy
 * 2. Resize to smaller shape without copy
 * 3. Resize to larger shape with copy
 * 4. Resize to smaller shape with copy
 * 5. Resize to zero shape
 * 6. Resize from zero shape
 * 7. Resize with copy from empty array
 *****************************************************************/

TEST_F(RS_ArrayTest, ResizeToLargerShapeWithoutCopy) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    std::array<RSize, 3> new_shape = {8, 10, 12};
    array.resize(new_shape, false);
    EXPECT_EQ(array.elem_num, 8 * 10 * 12);
    EXPECT_EQ(array.shape[0], 8);
    EXPECT_EQ(array.shape[1], 10);
    EXPECT_EQ(array.shape[2], 12);
    array.free();
}

TEST_F(RS_ArrayTest, ResizeToSmallerShapeWithoutCopy) {
    std::array<RSize, 3> shape = {8, 10, 12};
    RS_Array<int, 3> array(shape, &provider);
    std::array<RSize, 3> new_shape = {4, 5, 6};
    array.resize(new_shape, false);
    EXPECT_EQ(array.elem_num, 4 * 5 * 6);
    EXPECT_EQ(array.shape[0], 4);
    EXPECT_EQ(array.shape[1], 5);
    EXPECT_EQ(array.shape[2], 6);
    array.free();
}

TEST_F(RS_ArrayTest, ResizeToLargerShapeWithCopy) {
    std::array<RSize, 2> shape = {4, 5};
    RS_Array<int, 2> array(shape, &provider);

    int input[4 * 5];
    for (int i = 0; i < 4 * 5; ++i) input[i] = i;
    array.write(input);

    std::array<RSize, 2> new_shape = {8, 10};
    array.resize(new_shape, true);
    EXPECT_EQ(array.elem_num, 8 * 10);
    EXPECT_EQ(array.shape[0], 8);
    EXPECT_EQ(array.shape[1], 10);

    // Check that the old data is preserved
    int output[4 * 5];
    array.readSlice(output, 4 * 5, 0);
    for (int i = 0; i < 4 * 5; ++i) EXPECT_EQ(output[i], i);

    array.free();
}

TEST_F(RS_ArrayTest, ResizeToSmallerShapeWithCopy) {
    std::array<RSize, 2> shape = {8, 10};
    RS_Array<int, 2> array(shape, &provider);

    int input[8 * 10];
    for (int i = 0; i < 8 * 10; ++i) input[i] = i;
    array.write(input);

    std::array<RSize, 2> new_shape = {4, 5};
    array.resize(new_shape, true);
    EXPECT_EQ(array.elem_num, 4 * 5);
    EXPECT_EQ(array.shape[0], 4);
    EXPECT_EQ(array.shape[1], 5);

    // Check that the preserved data is correct
    int output[4 * 5];
    array.read(output);
    for (int i = 0; i < 4 * 5; ++i) EXPECT_EQ(output[i], i);

    array.free();
}

TEST_F(RS_ArrayTest, ResizeToZeroShape) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    std::array<RSize, 3> new_shape = {0, 0, 0};
    array.resize(new_shape);
    EXPECT_EQ(array.elem_num, 0);
    EXPECT_EQ(array.shape[0], 0);
    EXPECT_EQ(array.shape[1], 0);
    EXPECT_EQ(array.shape[2], 0);
    EXPECT_EQ(array.data.ptr(), nullptr);
    array.free();
}

TEST_F(RS_ArrayTest, ResizeFromZeroShape) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);
    std::array<RSize, 3> new_shape = {4, 5, 6};
    array.resize(new_shape);
    EXPECT_EQ(array.elem_num, 4 * 5 * 6);
    EXPECT_EQ(array.shape[0], 4);
    EXPECT_EQ(array.shape[1], 5);
    EXPECT_EQ(array.shape[2], 6);
    EXPECT_NE(array.data.ptr(), nullptr);
    array.free();
}

TEST_F(RS_ArrayTest, ResizeWithCopyFromEmptyArray) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);
    std::array<RSize, 3> new_shape = {4, 5, 6};
    array.resize(new_shape, true);
    EXPECT_EQ(array.elem_num, 4 * 5 * 6);
    EXPECT_EQ(array.shape[0], 4);
    EXPECT_EQ(array.shape[1], 5);
    EXPECT_EQ(array.shape[2], 6);
    EXPECT_NE(array.data.ptr(), nullptr);
    array.free();
}

/*****************************************************************
 * Test cases for the RS_Array::fill()
 *
 * Test cases:
 * 1. Fill non-empty array
 * 2. Fill empty array
 *****************************************************************/

TEST_F(RS_ArrayTest, FillNonEmptyArray) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    int fill_value = 42;
    array.fill(fill_value);

    int output[4 * 5 * 6];
    array.read(output);
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], fill_value);

    array.free();
}

TEST_F(RS_ArrayTest, FillEmptyArray) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);
    int fill_value = 42;
    EXPECT_NO_THROW(array.fill(fill_value));
    array.free();
}

/*****************************************************************
 * Test cases for the RS_Array::write()
 *
 * Test cases:
 * 1. Write valid data
 * 2. Write to empty array
 *****************************************************************/

TEST_F(RS_ArrayTest, WriteValidData) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);

    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;
    array.write(input);

    int output[4 * 5 * 6];
    array.read(output);
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], i);

    array.free();
}

TEST_F(RS_ArrayTest, WriteToEmptyArray) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);

    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;

    EXPECT_NO_THROW(array.write(input));
    EXPECT_EQ(array.data.ptr(), nullptr);

    array.free();
}

/*****************************************************************
 * Test cases for the RS_Array::writeSlice()
 *
 * Test cases:
 * 1. Write valid data
 * 2. Write data exceeds the array size
 * 3. Write zero elements
 * 4. Write data at offset beyond array size
 *****************************************************************/

TEST_F(RS_ArrayTest, WriteSliceValid) {
    RS_Array<int, 1> array({20}, &provider);
    int input[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    array.writeSlice(input, 10, 5);
    int output[20] = {0};
    array.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[5 + i], input[i]);
    }
    array.free();
}

TEST_F(RS_ArrayTest, WriteSliceExceedsSize) {
    RS_Array<int, 1> array({5}, &provider);
    int input[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_THROW(array.writeSlice(input, 10, 0), RS_OutOfBoundsException);
    array.free();
}

TEST_F(RS_ArrayTest, WriteSliceZeroElements) {
    RS_Array<int, 1> array({10}, &provider);
    int input[1];
    EXPECT_NO_THROW(array.writeSlice(input, 0, 5));
    array.free();
}

TEST_F(RS_ArrayTest, WriteSliceOffsetBeyondSize) {
    RS_Array<int, 1> array({10}, &provider);
    int input[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_THROW(array.writeSlice(input, 10, 1000), RS_OutOfBoundsException);
    array.free();
}

/*****************************************************************
 * Test cases for the RS_Array::copy()
 *
 * Test cases:
 * 1. Copy array of same shape
 * 2. Copy array of different shape
 * 3. Copy empty array
 * 4. Copy to empty array
 *****************************************************************/

TEST_F(RS_ArrayTest, CopyArrayOfSameShape) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> source_array(shape, &provider);
    RS_Array<int, 3> dest_array(shape, &provider);

    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;
    source_array.write(input);

    dest_array.copy(source_array);

    int output[4 * 5 * 6];
    dest_array.read(output);
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], i);

    source_array.free();
    dest_array.free();
}

TEST_F(RS_ArrayTest, CopyArrayOfDifferentShape) {
    std::array<RSize, 3> source_shape = {4, 5, 6};
    std::array<RSize, 3> dest_shape = {2, 3, 4};
    RS_Array<int, 3> source_array(source_shape, &provider);
    RS_Array<int, 3> dest_array(dest_shape, &provider);

    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;
    source_array.write(input);

    dest_array.copy(source_array);

    // Verify that the data is copied correctly and dest_array is resized
    EXPECT_EQ(dest_array.elem_num, 4 * 5 * 6);
    EXPECT_EQ(dest_array.shape[0], 4);
    EXPECT_EQ(dest_array.shape[1], 5);
    EXPECT_EQ(dest_array.shape[2], 6);

    int output[4 * 5 * 6];
    dest_array.read(output);
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], i);

    source_array.free();
    dest_array.free();
}

TEST_F(RS_ArrayTest, CopyEmptyArray) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> source_array({0, 0, 0}, &provider);
    RS_Array<int, 3> dest_array(shape, &provider);

    dest_array.copy(source_array);

    // Verify that dest_array becomes empty
    EXPECT_EQ(dest_array.elem_num, 0);
    EXPECT_EQ(dest_array.shape[0], 0);
    EXPECT_EQ(dest_array.shape[1], 0);
    EXPECT_EQ(dest_array.shape[2], 0);
    EXPECT_EQ(dest_array.data.ptr(), nullptr);

    source_array.free();
    dest_array.free();
}

TEST_F(RS_ArrayTest, CopyToEmptyArray) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> source_array(shape, &provider);
    RS_Array<int, 3> dest_array({0, 0, 0}, &provider);

    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;
    source_array.write(input);

    dest_array.copy(source_array);

    EXPECT_EQ(dest_array.elem_num, 4 * 5 * 6);
    EXPECT_EQ(dest_array.shape[0], 4);
    EXPECT_EQ(dest_array.shape[1], 5);
    EXPECT_EQ(dest_array.shape[2], 6);

    int output[4 * 5 * 6];
    dest_array.read(output);
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], i);

    source_array.free();
    dest_array.free();
}

/*****************************************************************
 * Test cases for the RS_Array::read()
 *
 * Test cases:
 * 1. Read non-empty array
 * 2. Read empty array
 *****************************************************************/

TEST_F(RS_ArrayTest, ReadNonEmptyArray) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);

    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;
    array.write(input);

    int output[4 * 5 * 6];
    array.read(output);
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], i);

    array.free();
}

TEST_F(RS_ArrayTest, ReadEmptyArray) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);

    int output[1] = {42};
    array.read(output);

    // Verify that the output remains unchanged
    EXPECT_EQ(output[0], 42);

    array.free();
}

/*****************************************************************
 * Test cases for the RS_Array::readSlice()
 *
 * Test cases:
 * 1. Read valid slice
 * 2. Read slice exceeds the array size
 * 3. Read zero elements
 * 4. Read slice at offset beyond array size
 *****************************************************************/

TEST_F(RS_ArrayTest, ReadSliceValid) {
    RS_Array<int, 1> array({20}, &provider);
    int input[20] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    array.write(input);
    int output[10] = {0};
    array.readSlice(output, 10, 5);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i], input[5 + i]);
    }
    array.free();
}

TEST_F(RS_ArrayTest, ReadSliceExceedsSize) {
    RS_Array<int, 1> array({5}, &provider);
    int output[10] = {0};
    EXPECT_THROW(array.readSlice(output, 10, 0), RS_OutOfBoundsException);
    array.free();
}

TEST_F(RS_ArrayTest, ReadSliceZeroElements) {
    RS_Array<int, 1> array({10}, &provider);
    int output[1];
    EXPECT_NO_THROW(array.readSlice(output, 0, 5));
    array.free();
}

TEST_F(RS_ArrayTest, ReadSliceOffsetBeyondSize) {
    RS_Array<int, 1> array({10}, &provider);
    int output[10] = {0};
    EXPECT_THROW(array.readSlice(output, 10, 1000), RS_OutOfBoundsException);
    array.free();
}

/*****************************************************************
 * Test cases for the RS_Array::indexToCoords() and RS_Array::coordsToIndex()
 * on the host
 *
 * Test cases:
 * 1. Convert linear index to coordinates (indexToCoords)
 * 2. Convert coordinates to linear index (coordsToIndex)
 *****************************************************************/

TEST_F(RS_ArrayTest, IndexToCoordsHostTest) {
    RIndex x, y, z, w;

    RS_Array<int, 2> array2d_row({6, 5}, &host_provider);
    array2d_row.indexToCoords(7, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 2);

    RS_Array<int, 3> array3d_row({6, 5, 4}, &host_provider);
    array3d_row.indexToCoords(31, x, y, z);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 2);
    EXPECT_EQ(z, 3);

    RS_Array<int, 4> array4d_row({6, 5, 4, 3}, &host_provider);
    array4d_row.indexToCoords(95, x, y, z, w);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 2);
    EXPECT_EQ(z, 3);
    EXPECT_EQ(w, 2);

    RS_Array<int, 2, false> array2d_col({5, 6}, &host_provider);
    array2d_col.indexToCoords(7, x, y);
    EXPECT_EQ(x, 2);
    EXPECT_EQ(y, 1);

    RS_Array<int, 3, false> array3d_col({4, 5, 6}, &host_provider);
    array3d_col.indexToCoords(31, x, y, z);
    EXPECT_EQ(x, 3);
    EXPECT_EQ(y, 2);
    EXPECT_EQ(z, 1);

    RS_Array<int, 4, false> array4d_col({3, 4, 5, 6}, &host_provider);
    array4d_col.indexToCoords(95, x, y, z, w);
    EXPECT_EQ(x, 2);
    EXPECT_EQ(y, 3);
    EXPECT_EQ(z, 2);
    EXPECT_EQ(w, 1);

    array2d_row.free();
    array3d_row.free();
    array4d_row.free();
    array2d_col.free();
    array3d_col.free();
    array4d_col.free();
}

TEST_F(RS_ArrayTest, CoordsToIndexHostTest) {
    RIndex index;

    RS_Array<int, 2> array2d_row({6, 5}, &host_provider);
    index = array2d_row.coordsToIndex(1, 2);
    EXPECT_EQ(index, 7);

    RS_Array<int, 3> array3d_row({6, 5, 4}, &host_provider);
    index = array3d_row.coordsToIndex(1, 2, 3);
    EXPECT_EQ(index, 31);

    RS_Array<int, 4> array4d_row({6, 5, 4, 3}, &host_provider);
    index = array4d_row.coordsToIndex(1, 2, 3, 2);
    EXPECT_EQ(index, 95);

    RS_Array<int, 2, false> array2d_col({5, 6}, &host_provider);
    index = array2d_col.coordsToIndex(2, 1);
    EXPECT_EQ(index, 7);

    RS_Array<int, 3, false> array3d_col({4, 5, 6}, &host_provider);
    index = array3d_col.coordsToIndex(3, 2, 1);
    EXPECT_EQ(index, 31);

    RS_Array<int, 4, false> array4d_col({3, 4, 5, 6}, &host_provider);
    index = array4d_col.coordsToIndex(2, 3, 2, 1);
    EXPECT_EQ(index, 95);

    array2d_row.free();
    array3d_row.free();
    array4d_row.free();
    array2d_col.free();
    array3d_col.free();
    array4d_col.free();
}

/*****************************************************************
 * Test cases for the RS_Array device side access
 *
 * Test cases:
 * 1. isOutOfBound() on the device
 * 2. indexToCoords() and coordsToIndex() on the device, both layouts
 *****************************************************************/

class RSCF_ArrayIsOutOfBound : public RS_ComputeFunction<RSCF_ArrayIsOutOfBound> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_Array<int, 3> array, RIndex index, RS_Array<int, 1> result) {
        if (index1D() == 0) {
            result[0] = array.isOutOfBound(index) ? 1 : 0;
        }
    }
};

TEST_F(RS_ArrayTest, IsOutOfBoundDeviceTest) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);

    RS_Array<int, 1> result({1}, &provider);

    RIndex index_within_bounds = 10;
    provider.run<RSCF_ArrayIsOutOfBound>({1}, array, index_within_bounds, result);
    RS_Array<int, 1> host_result_a = result.move(&host_provider);
    EXPECT_EQ(host_result_a[0], 0);

    RIndex index_greater_than_size = array.elem_num + 1;
    provider.run<RSCF_ArrayIsOutOfBound>({1}, array, index_greater_than_size, result);
    RS_Array<int, 1> host_result_b = result.move(&host_provider);
    EXPECT_EQ(host_result_b[0], 1);

    host_result_a.free();
    host_result_b.free();
    result.free();
    array.free();
}

template <typename Array>
class RSCF_ArrayIndexToCoords : public RS_ComputeFunction<RSCF_ArrayIndexToCoords<Array>> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Array array, RIndex index, RS_Array<RIndex, 1> result) {
        if (RS_ComputeFunction<RSCF_ArrayIndexToCoords<Array>>::index1D() == 0) {
            RIndex x, y, z;
            array.indexToCoords(index, x, y, z);
            result[0] = x;
            result[1] = y;
            result[2] = z;
        }
    }
};

template <typename Array>
class RSCF_ArrayCoordsToIndex : public RS_ComputeFunction<RSCF_ArrayCoordsToIndex<Array>> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Array array, RIndex x, RIndex y, RIndex z, RS_Array<RIndex, 1> result) {
        if (RS_ComputeFunction<RSCF_ArrayCoordsToIndex<Array>>::index1D() == 0) {
            result[0] = array.coordsToIndex(x, y, z);
        }
    }
};

TEST_F(RS_ArrayTest, IndexToCoordsRowMajorDeviceTest) {
    std::array<RSize, 3> shape = {6, 5, 4};
    RS_Array<int, 3> array(shape, &provider);

    RS_Array<RIndex, 1> result({3}, &provider);
    provider.run<RSCF_ArrayIndexToCoords<RS_Array<int, 3>>>({1}, array, 31, result);

    RS_Array<RIndex, 1> host_result = result.move(&host_provider);
    EXPECT_EQ(host_result[0], 1);
    EXPECT_EQ(host_result[1], 2);
    EXPECT_EQ(host_result[2], 3);

    host_result.free();
    result.free();
    array.free();
}

TEST_F(RS_ArrayTest, CoordsToIndexRowMajorDeviceTest) {
    std::array<RSize, 3> shape = {6, 5, 4};
    RS_Array<int, 3> array(shape, &provider);

    RS_Array<RIndex, 1> result({1}, &provider);
    provider.run<RSCF_ArrayCoordsToIndex<RS_Array<int, 3>>>({1}, array, 1, 2, 3, result);

    RS_Array<RIndex, 1> host_result = result.move(&host_provider);
    EXPECT_EQ(host_result[0], 31);

    host_result.free();
    result.free();
    array.free();
}

TEST_F(RS_ArrayTest, IndexToCoordsColumnMajorDeviceTest) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3, false> array(shape, &provider);

    RS_Array<RIndex, 1> result({3}, &provider);
    provider.run<RSCF_ArrayIndexToCoords<RS_Array<int, 3, false>>>({1}, array, 31, result);

    RS_Array<RIndex, 1> host_result = result.move(&host_provider);
    EXPECT_EQ(host_result[0], 3);
    EXPECT_EQ(host_result[1], 2);
    EXPECT_EQ(host_result[2], 1);

    host_result.free();
    result.free();
    array.free();
}

TEST_F(RS_ArrayTest, CoordsToIndexColumnMajorDeviceTest) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3, false> array(shape, &provider);

    RS_Array<RIndex, 1> result({1}, &provider);
    provider.run<RSCF_ArrayCoordsToIndex<RS_Array<int, 3, false>>>({1}, array, 3, 2, 1, result);

    RS_Array<RIndex, 1> host_result = result.move(&host_provider);
    EXPECT_EQ(host_result[0], 31);

    host_result.free();
    result.free();
    array.free();
}

/*****************************************************************
 * Test cases for RS_StaticArray on the host
 *
 * Test cases:
 * 1. isOutOfBound()
 * 2. indexToCoords()
 * 3. coordsToIndex()
 *****************************************************************/

TEST_F(RS_ArrayTest, StaticIsOutOfBound) {
    RS_StaticRowMajorArray<int, 4, 5, 6> array{};
    EXPECT_FALSE(array.isOutOfBound({0, 0, 0}));
    EXPECT_TRUE(array.isOutOfBound({10, 10, 10}));
}

TEST_F(RS_ArrayTest, StaticIndexToCoordsHostTest) {
    RIndex x, y, z, w;

    RS_StaticRowMajorArray<int, 6, 5>{}.indexToCoords(7, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 2);

    RS_StaticRowMajorArray<int, 6, 5, 4>{}.indexToCoords(31, x, y, z);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 2);
    EXPECT_EQ(z, 3);

    RS_StaticRowMajorArray<int, 6, 5, 4, 3>{}.indexToCoords(95, x, y, z, w);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 2);
    EXPECT_EQ(z, 3);
    EXPECT_EQ(w, 2);

    RS_StaticColumnMajorArray<int, 5, 6>{}.indexToCoords(7, x, y);
    EXPECT_EQ(x, 2);
    EXPECT_EQ(y, 1);

    RS_StaticColumnMajorArray<int, 4, 5, 6>{}.indexToCoords(31, x, y, z);
    EXPECT_EQ(x, 3);
    EXPECT_EQ(y, 2);
    EXPECT_EQ(z, 1);

    RS_StaticColumnMajorArray<int, 3, 4, 5, 6>{}.indexToCoords(95, x, y, z, w);
    EXPECT_EQ(x, 2);
    EXPECT_EQ(y, 3);
    EXPECT_EQ(z, 2);
    EXPECT_EQ(w, 1);
}

TEST_F(RS_ArrayTest, StaticCoordsToIndexHostTest) {
    RIndex index = RS_StaticRowMajorArray<int, 6, 5>{}.coordsToIndex(1, 2);
    EXPECT_EQ(index, 7);

    index = RS_StaticRowMajorArray<int, 6, 5, 4>{}.coordsToIndex(1, 2, 3);
    EXPECT_EQ(index, 31);

    index = RS_StaticRowMajorArray<int, 6, 5, 4, 3>{}.coordsToIndex(1, 2, 3, 2);
    EXPECT_EQ(index, 95);

    index = RS_StaticColumnMajorArray<int, 5, 6>{}.coordsToIndex(2, 1);
    EXPECT_EQ(index, 7);

    index = RS_StaticColumnMajorArray<int, 4, 5, 6>{}.coordsToIndex(3, 2, 1);
    EXPECT_EQ(index, 31);

    index = RS_StaticColumnMajorArray<int, 3, 4, 5, 6>{}.coordsToIndex(2, 3, 2, 1);
    EXPECT_EQ(index, 95);
}
