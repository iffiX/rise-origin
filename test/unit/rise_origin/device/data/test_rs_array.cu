#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_array.h"

using namespace rise;
using namespace rise::device;

class RS_ArrayTest : public ::testing::Test {
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
    EXPECT_NE(array.data.data, nullptr);
    array.free();
}

TEST_F(RS_ArrayTest, InitializeWithZeroDimensions) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);
    EXPECT_EQ(array.elem_num, 0);
    EXPECT_EQ(array.shape[0], 0);
    EXPECT_EQ(array.shape[1], 0);
    EXPECT_EQ(array.shape[2], 0);
    EXPECT_EQ(array.data.data, nullptr);
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
    EXPECT_EQ(array.data.data, nullptr);
}

TEST_F(RS_ArrayTest, FreeEmptyArray) {
    RS_Array<int, 3> array({0, 0, 0}, &provider);
    EXPECT_NO_THROW(array.free());
}

TEST_F(RS_ArrayTest, MultipleFrees) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    array.free();
    EXPECT_NO_THROW(array.free());  // Should handle multiple frees gracefully
    EXPECT_EQ(array.elem_num, 0);
    EXPECT_EQ(array.data.data, nullptr);
}

TEST_F(RS_ArrayTest, FreeAfterOperations) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    array.resize({8, 8, 8});
    array.free();
    EXPECT_EQ(array.elem_num, 0);
    EXPECT_EQ(array.data.data, nullptr);
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
    // Fill the array with some data
    std::vector<int> input(4 * 5);
    for (int i = 0; i < 4 * 5; ++i) input[i] = i;
    cudaMemcpy(array.data.data, input.data(), 4 * 5 * sizeof(int), cudaMemcpyHostToDevice);

    std::array<RSize, 2> new_shape = {8, 10};
    array.resize(new_shape, true);
    EXPECT_EQ(array.elem_num, 8 * 10);
    EXPECT_EQ(array.shape[0], 8);
    EXPECT_EQ(array.shape[1], 10);

    // Check that the old data is preserved
    std::vector<int> output(4 * 5);
    cudaMemcpy(output.data(), array.data.data, 4 * 5 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4 * 5; ++i) EXPECT_EQ(output[i], i);

    array.free();
}

TEST_F(RS_ArrayTest, ResizeToSmallerShapeWithCopy) {
    std::array<RSize, 2> shape = {8, 10};
    RS_Array<int, 2> array(shape, &provider);
    // Fill the array with some data
    std::vector<int> input(8 * 10);
    for (int i = 0; i < 8 * 10; ++i) input[i] = i;
    cudaMemcpy(array.data.data, input.data(), 8 * 10 * sizeof(int), cudaMemcpyHostToDevice);

    std::array<RSize, 2> new_shape = {4, 5};
    array.resize(new_shape, true);
    EXPECT_EQ(array.elem_num, 4 * 5);
    EXPECT_EQ(array.shape[0], 4);
    EXPECT_EQ(array.shape[1], 5);

    // Check that the preserved data is correct
    std::vector<int> output(4 * 5);
    cudaMemcpy(output.data(), array.data.data, 4 * 5 * sizeof(int), cudaMemcpyDeviceToHost);
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
    EXPECT_EQ(array.data.data, nullptr);
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
    EXPECT_NE(array.data.data, nullptr);
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
    EXPECT_NE(array.data.data, nullptr);
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

    // Verify that all elements are set to the fill_value
    std::vector<int> output(4 * 5 * 6);
    cudaMemcpy(output.data(), array.data.data, 4 * 5 * 6 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], fill_value);

    array.free();
}

TEST_F(RS_ArrayTest, FillEmptyArray) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);
    int fill_value = 42;
    // Expect no errors or crashes when filling an empty array
    EXPECT_NO_THROW(array.fill(fill_value));
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

    // Verify that the data is written correctly to the GPU
    std::vector<int> output(4 * 5 * 6);
    cudaMemcpy(output.data(), array.data.data, 4 * 5 * 6 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], i);

    array.free();
}

TEST_F(RS_ArrayTest, WriteToEmptyArray) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);

    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;

    // Expect no errors or crashes when writing to an empty array
    EXPECT_NO_THROW(array.write(input));

    // Verify that the data is not written to the GPU
    EXPECT_EQ(array.data.data, nullptr);

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
}

TEST_F(RS_ArrayTest, WriteSliceExceedsSize) {
    RS_Array<int, 1> array({5}, &provider);
    int input[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_THROW(array.writeSlice(input, 10, 0), RS_OutOfBoundsException);
}

TEST_F(RS_ArrayTest, WriteSliceZeroElements) {
    RS_Array<int, 1> array({10}, &provider);
    int input[0];
    EXPECT_NO_THROW(array.writeSlice(input, 0, 5));
}

TEST_F(RS_ArrayTest, WriteSliceOffsetBeyondSize) {
    RS_Array<int, 1> array({10}, &provider);
    int input[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_THROW(array.writeSlice(input, 10, 1000), RS_OutOfBoundsException);
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

    // Fill the source array with some data
    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;
    cudaMemcpy(source_array.data.data, input, 4 * 5 * 6 * sizeof(int), cudaMemcpyHostToDevice);

    // Copy data from source_array to dest_array
    dest_array.copy(source_array);

    // Verify that the data is copied correctly
    std::vector<int> output(4 * 5 * 6);
    cudaMemcpy(
        output.data(), dest_array.data.data, 4 * 5 * 6 * sizeof(int), cudaMemcpyDeviceToHost
    );
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], i);

    source_array.free();
    dest_array.free();
}

TEST_F(RS_ArrayTest, CopyArrayOfDifferentShape) {
    std::array<RSize, 3> source_shape = {4, 5, 6};
    std::array<RSize, 3> dest_shape = {2, 3, 4};
    RS_Array<int, 3> source_array(source_shape, &provider);
    RS_Array<int, 3> dest_array(dest_shape, &provider);

    // Fill the source array with some data
    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;
    cudaMemcpy(source_array.data.data, input, 4 * 5 * 6 * sizeof(int), cudaMemcpyHostToDevice);

    // Copy data from source_array to dest_array
    dest_array.copy(source_array);

    // Verify that the data is copied correctly and dest_array is resized
    EXPECT_EQ(dest_array.elem_num, 4 * 5 * 6);
    EXPECT_EQ(dest_array.shape[0], 4);
    EXPECT_EQ(dest_array.shape[1], 5);
    EXPECT_EQ(dest_array.shape[2], 6);

    std::vector<int> output(4 * 5 * 6);
    cudaMemcpy(
        output.data(), dest_array.data.data, 4 * 5 * 6 * sizeof(int), cudaMemcpyDeviceToHost
    );
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], i);

    source_array.free();
    dest_array.free();
}

TEST_F(RS_ArrayTest, CopyEmptyArray) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> source_array({0, 0, 0}, &provider);
    RS_Array<int, 3> dest_array(shape, &provider);

    // Copy data from empty source_array to dest_array
    dest_array.copy(source_array);

    // Verify that dest_array becomes empty
    EXPECT_EQ(dest_array.elem_num, 0);
    EXPECT_EQ(dest_array.shape[0], 0);
    EXPECT_EQ(dest_array.shape[1], 0);
    EXPECT_EQ(dest_array.shape[2], 0);
    EXPECT_EQ(dest_array.data.data, nullptr);

    source_array.free();
    dest_array.free();
}

TEST_F(RS_ArrayTest, CopyToEmptyArray) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> source_array(shape, &provider);
    RS_Array<int, 3> dest_array({0, 0, 0}, &provider);

    // Fill the source array with some data
    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;
    cudaMemcpy(source_array.data.data, input, 4 * 5 * 6 * sizeof(int), cudaMemcpyHostToDevice);

    // Copy data from source_array to dest_array
    dest_array.copy(source_array);

    // Verify that the data is copied correctly and dest_array is resized
    EXPECT_EQ(dest_array.elem_num, 4 * 5 * 6);
    EXPECT_EQ(dest_array.shape[0], 4);
    EXPECT_EQ(dest_array.shape[1], 5);
    EXPECT_EQ(dest_array.shape[2], 6);

    std::vector<int> output(4 * 5 * 6);
    cudaMemcpy(
        output.data(), dest_array.data.data, 4 * 5 * 6 * sizeof(int), cudaMemcpyDeviceToHost
    );
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

    // Fill the array with some data
    int input[4 * 5 * 6];
    for (int i = 0; i < 4 * 5 * 6; ++i) input[i] = i;
    cudaMemcpy(array.data.data, input, 4 * 5 * 6 * sizeof(int), cudaMemcpyHostToDevice);

    // Read data from the GPU to host
    int output[4 * 5 * 6];
    array.read(output);

    // Verify that the data is read correctly from the GPU
    for (int i = 0; i < 4 * 5 * 6; ++i) EXPECT_EQ(output[i], i);

    array.free();
}

TEST_F(RS_ArrayTest, ReadEmptyArray) {
    std::array<RSize, 3> shape = {0, 0, 0};
    RS_Array<int, 3> array(shape, &provider);

    // Prepare output array with initial values
    int output[1] = {42};

    // Read data from the GPU to host (expect no changes)
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
    array.write(input);  // Write data to the array
    int output[10] = {0};
    array.readSlice(output, 10, 5);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i], input[5 + i]);
    }
}

TEST_F(RS_ArrayTest, ReadSliceExceedsSize) {
    RS_Array<int, 1> array({5}, &provider);
    int output[10] = {0};
    EXPECT_THROW(array.readSlice(output, 10, 0), RS_OutOfBoundsException);
}

TEST_F(RS_ArrayTest, ReadSliceZeroElements) {
    RS_Array<int, 1> array({10}, &provider);
    int output[0];
    EXPECT_NO_THROW(array.readSlice(output, 0, 5));
}

TEST_F(RS_ArrayTest, ReadSliceOffsetBeyondSize) {
    RS_Array<int, 1> array({10}, &provider);
    int output[10] = {0};
    EXPECT_THROW(array.readSlice(output, 10, 1000), RS_OutOfBoundsException);
}

/*****************************************************************
 * Test cases for the RS_Array::isOutOfBound()
 *****************************************************************/

__global__ void checkIsOutOfBound(RS_Array<int, 3> array, RIndex index, bool* result) {
    *result = array.isOutOfBound(index);
}

TEST_F(RS_ArrayTest, IsOutOfBoundTest) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3> array(shape, &provider);
    bool host_result;

    // Test case: Index within bounds
    bool* device_result;
    cudaMalloc(&device_result, sizeof(bool));
    RIndex index_within_bounds = 10;  // Example index within bounds (0 <= index < 4*5*6)
    checkIsOutOfBound<<<1, 1>>>(array, index_within_bounds, device_result);
    cudaMemcpy(&host_result, device_result, sizeof(bool), cudaMemcpyDeviceToHost);
    EXPECT_FALSE(host_result);

    // Test case: Index greater or equal than elem_num
    RIndex index_greater_than_size = array.elem_num + 1;  // Index greater than the array's elem_num
    checkIsOutOfBound<<<1, 1>>>(array, index_greater_than_size, device_result);
    cudaMemcpy(&host_result, device_result, sizeof(bool), cudaMemcpyDeviceToHost);
    EXPECT_TRUE(host_result);

    cudaFree(device_result);
    array.free();
}

/*****************************************************************
 * Test cases for the RS_Array::indexToCoords() and RS_Array::coordsToIndex()
 *
 * Test cases:
 * 1. Convert linear index to 3D coordinates (indexToCoords)
 * 2. Convert 3D coordinates to linear index (coordsToIndex)
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

template <typename Array>
__global__ void testIndexToCoords(Array array, RIndex index, RIndex* x, RIndex* y, RIndex* z) {
    array.indexToCoords(index, *x, *y, *z);
}

template <typename Array>
__global__ void testCoordsToIndex(Array array, RIndex x, RIndex y, RIndex z, RIndex* index) {
    *index = array.coordsToIndex(x, y, z);
}

TEST_F(RS_ArrayTest, IndexToCoordsRowMajorTest) {
    std::array<RSize, 3> shape = {6, 5, 4};
    RS_Array<int, 3> array(shape, &provider);

    RIndex index = 31;  // Example linear index
    RIndex host_x, host_y, host_z;
    RIndex *device_x, *device_y, *device_z;
    cudaMalloc(&device_x, sizeof(RIndex));
    cudaMalloc(&device_y, sizeof(RIndex));
    cudaMalloc(&device_z, sizeof(RIndex));

    testIndexToCoords<<<1, 1>>>(array, index, device_x, device_y, device_z);
    cudaMemcpy(&host_x, device_x, sizeof(RIndex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_y, device_y, sizeof(RIndex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_z, device_z, sizeof(RIndex), cudaMemcpyDeviceToHost);

    EXPECT_EQ(host_x, 1);
    EXPECT_EQ(host_y, 2);
    EXPECT_EQ(host_z, 3);

    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
    array.free();
}

TEST_F(RS_ArrayTest, CoordsToIndexRowMajorTest) {
    std::array<RSize, 3> shape = {6, 5, 4};
    RS_Array<int, 3> array(shape, &provider);

    RIndex x = 1, y = 2, z = 3;
    RIndex host_index;
    RIndex* device_index;
    cudaMalloc(&device_index, sizeof(RIndex));

    testCoordsToIndex<<<1, 1>>>(array, x, y, z, device_index);
    cudaMemcpy(&host_index, device_index, sizeof(RIndex), cudaMemcpyDeviceToHost);

    EXPECT_EQ(host_index, 31);

    cudaFree(device_index);
    array.free();
}

TEST_F(RS_ArrayTest, IndexToCoordsColumnMajorTest) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3, false> array(shape, &provider);

    RIndex index = 31;  // Example linear index
    RIndex host_x, host_y, host_z;
    RIndex *device_x, *device_y, *device_z;
    cudaMalloc(&device_x, sizeof(RIndex));
    cudaMalloc(&device_y, sizeof(RIndex));
    cudaMalloc(&device_z, sizeof(RIndex));

    testIndexToCoords<<<1, 1>>>(array, index, device_x, device_y, device_z);
    cudaMemcpy(&host_x, device_x, sizeof(RIndex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_y, device_y, sizeof(RIndex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_z, device_z, sizeof(RIndex), cudaMemcpyDeviceToHost);

    EXPECT_EQ(host_x, 3);
    EXPECT_EQ(host_y, 2);
    EXPECT_EQ(host_z, 1);

    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
    array.free();
}

TEST_F(RS_ArrayTest, CoordsToIndexColumnMajorTest) {
    std::array<RSize, 3> shape = {4, 5, 6};
    RS_Array<int, 3, false> array(shape, &provider);

    RIndex x = 3, y = 2, z = 1;
    RIndex host_index;
    RIndex* device_index;
    cudaMalloc(&device_index, sizeof(RIndex));

    testCoordsToIndex<<<1, 1>>>(array, x, y, z, device_index);
    cudaMemcpy(&host_index, device_index, sizeof(RIndex), cudaMemcpyDeviceToHost);

    EXPECT_EQ(host_index, 31);

    cudaFree(device_index);
    array.free();
}

class RS_StaticArrayTest : public ::testing::Test {
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
 * Test cases for the RS_StaticArray::isOutOfBound()
 *****************************************************************/

TEST_F(RS_StaticArrayTest, IsOutOfBound) {
    RS_StaticRowMajorArray<int, 4, 5, 6> array{};
    EXPECT_FALSE(array.isOutOfBound({0, 0, 0}));
    EXPECT_TRUE(array.isOutOfBound({10, 10, 10}));
}

/*****************************************************************
 * Test cases for the RS_StaticArray::indexToCoords() and RS_StaticArray::coordsToIndex()
 *
 * Test cases:
 * 1. Convert linear index to 3D coordinates (indexToCoords)
 * 2. Convert 3D coordinates to linear index (coordsToIndex)
 *****************************************************************/
TEST_F(RS_StaticArrayTest, StaticIndexToCoordsHostTest) {
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

TEST_F(RS_StaticArrayTest, StaticCoordsToIndexHostTest) {
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

template <typename Array>
__global__ void
testStaticIndexToCoords(Array array, RIndex index, RIndex* x, RIndex* y, RIndex* z) {
    array.indexToCoords(index, *x, *y, *z);
}

template <typename Array>
__global__ void testStaticCoordsToIndex(Array array, RIndex x, RIndex y, RIndex z, RIndex* index) {
    *index = array.coordsToIndex(x, y, z);
}

TEST_F(RS_StaticArrayTest, StaticIndexToCoordsRowMajorTest) {
    RS_StaticRowMajorArray<int, 6, 5, 4> array{};

    RIndex index = 31;  // Example linear index
    RIndex host_x, host_y, host_z;
    RIndex *device_x, *device_y, *device_z;
    cudaMalloc(&device_x, sizeof(RIndex));
    cudaMalloc(&device_y, sizeof(RIndex));
    cudaMalloc(&device_z, sizeof(RIndex));

    testStaticIndexToCoords<<<1, 1>>>(array, index, device_x, device_y, device_z);
    cudaMemcpy(&host_x, device_x, sizeof(RIndex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_y, device_y, sizeof(RIndex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_z, device_z, sizeof(RIndex), cudaMemcpyDeviceToHost);

    EXPECT_EQ(host_x, 1);
    EXPECT_EQ(host_y, 2);
    EXPECT_EQ(host_z, 3);

    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
}

TEST_F(RS_StaticArrayTest, StaticCoordsToIndexRowMajorTest) {
    RS_StaticRowMajorArray<int, 6, 5, 4> array{};

    RIndex x = 1, y = 2, z = 3;
    RIndex host_index;
    RIndex* device_index;
    cudaMalloc(&device_index, sizeof(RIndex));

    testStaticCoordsToIndex<<<1, 1>>>(array, x, y, z, device_index);
    cudaMemcpy(&host_index, device_index, sizeof(RIndex), cudaMemcpyDeviceToHost);

    EXPECT_EQ(host_index, 31);

    cudaFree(device_index);
}

TEST_F(RS_StaticArrayTest, StaticIndexToCoordsColumnMajorTest) {
    RS_StaticColumnMajorArray<int, 4, 5, 6> array{};

    RIndex index = 31;  // Example linear index
    RIndex host_x, host_y, host_z;
    RIndex *device_x, *device_y, *device_z;
    cudaMalloc(&device_x, sizeof(RIndex));
    cudaMalloc(&device_y, sizeof(RIndex));
    cudaMalloc(&device_z, sizeof(RIndex));

    testStaticIndexToCoords<<<1, 1>>>(array, index, device_x, device_y, device_z);
    cudaMemcpy(&host_x, device_x, sizeof(RIndex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_y, device_y, sizeof(RIndex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_z, device_z, sizeof(RIndex), cudaMemcpyDeviceToHost);

    EXPECT_EQ(host_x, 3);
    EXPECT_EQ(host_y, 2);
    EXPECT_EQ(host_z, 1);

    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
}

TEST_F(RS_StaticArrayTest, StaticCoordsToIndexColumnMajorTest) {
    RS_StaticColumnMajorArray<int, 4, 5, 6> array{};

    RIndex x = 3, y = 2, z = 1;
    RIndex host_index;
    RIndex* device_index;
    cudaMalloc(&device_index, sizeof(RIndex));

    testStaticCoordsToIndex<<<1, 1>>>(array, x, y, z, device_index);
    cudaMemcpy(&host_index, device_index, sizeof(RIndex), cudaMemcpyDeviceToHost);

    EXPECT_EQ(host_index, 31);

    cudaFree(device_index);
}
