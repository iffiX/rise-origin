#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_soa.h"

using namespace rise;
using namespace rise::device;

struct Vector {
    int x;
    float y;
    double z;
};

struct A {
    int m;
    Vector v;
};

struct B {
    A struct_a;
    int array[10];
    unsigned char n;
};

REFL_AUTO(type(Vector), field(x), field(y), field(z))

REFL_AUTO(type(A), field(m), field(v))

REFL_AUTO(type(B), field(struct_a), field(array), field(n))

class RS_StructOfArraysTest : public ::testing::Test {
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
 * Test cases for the RS_StructOfArrays::free()
 *
 * Test cases:
 * 1. Free empty instance.
 * 2. Free non-empty instance.
 * 3. Multiple frees:
 *      Test calling the free method multiple times to ensure it
 *      handles repeated calls gracefully.
 * 4. Free after operations:
 *      Test freeing an RS_StructOfArrays after performing various
 *      operations like resize, fill, and write.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, FreeEmptyInstance) {
    RS_StructOfArrays<B> soa(0, &provider);
    EXPECT_NO_THROW(soa.free());
    EXPECT_EQ(soa.size(), 0);
}

TEST_F(RS_StructOfArraysTest, FreeNonEmptyInstance) {
    RS_StructOfArrays<B> soa(10, &provider);
    EXPECT_EQ(soa.size(), 10);
    EXPECT_NO_THROW(soa.free());
    EXPECT_EQ(soa.size(), 0);
}

TEST_F(RS_StructOfArraysTest, MultipleFrees) {
    RS_StructOfArrays<B> soa(10, &provider);
    EXPECT_NO_THROW(soa.free());
    EXPECT_NO_THROW(soa.free());  // Should handle multiple frees gracefully
    EXPECT_EQ(soa.size(), 0);
}

TEST_F(RS_StructOfArraysTest, FreeAfterOperations) {
    RS_StructOfArrays<B> soa(10, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;
    soa.fill(value);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }
    soa.write(input);

    EXPECT_NO_THROW(soa.free());
    EXPECT_EQ(soa.size(), 0);
}

/*****************************************************************
 * Test cases for the RS_StructOfArrays::resize()
 *
 * Test cases:
 * 1. Resize zero to zero.
 * 2. Resize zero to positive:
 *      Test resizing an RS_StructOfArrays from size zero to a
 *      positive size.
 * 3. Resize larger without copy:
 *      Test resizing an RS_StructOfArrays to a larger size
 *      without copying existing data.
 * 4. Resize smaller without copy:
 *      Test resizing an RS_StructOfArrays to a smaller size
 *      without copying existing data.
 * 5. Resize larger with copy:
 *      Test resizing an RS_StructOfArrays to a larger size
 *      with copying existing data.
 * 6. Resize smaller with copy:
 *      Test resizing an RS_StructOfArrays to a smaller size
 *      with copying existing data.
 * 7. Sequential resizes:
 *      Test performing a sequence of resizes (both larger and
 *      smaller) on an RS_StructOfArrays.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, ResizeZeroToZero) {
    RS_StructOfArrays<B> soa(0, &provider);
    soa.resize(0);
    EXPECT_EQ(soa.size(), 0);
}

TEST_F(RS_StructOfArraysTest, ResizeZeroToPositive) {
    RS_StructOfArrays<B> soa(0, &provider);
    soa.resize(5);
    EXPECT_EQ(soa.size(), 5);
}

TEST_F(RS_StructOfArraysTest, ResizeLargerWithoutCopy) {
    RS_StructOfArrays<B> soa(0, &provider);
    soa.resize(10, false);
    EXPECT_EQ(soa.size(), 10);
}

TEST_F(RS_StructOfArraysTest, ResizeSmallerWithoutCopy) {
    RS_StructOfArrays<B> soa(10, &provider);
    soa.resize(5, false);
    EXPECT_EQ(soa.size(), 5);
}

TEST_F(RS_StructOfArraysTest, ResizeLargerWithCopy) {
    RS_StructOfArrays<B> soa(5, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;
    soa.fill(value);

    soa.resize(10, true);
    EXPECT_EQ(soa.size(), 10);

    B output[5];
    soa.readSlice(output, 5, 0);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        EXPECT_EQ(output[i].struct_a.v.x, 1);
        EXPECT_EQ(output[i].struct_a.v.y, 2.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 3.0);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

TEST_F(RS_StructOfArraysTest, ResizeSmallerWithCopy) {
    RS_StructOfArrays<B> soa(10, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;
    soa.fill(value);

    soa.resize(5, true);
    EXPECT_EQ(soa.size(), 5);

    B output[5];
    soa.readSlice(output, 5, 0);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        EXPECT_EQ(output[i].struct_a.v.x, 1);
        EXPECT_EQ(output[i].struct_a.v.y, 2.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 3.0);
        EXPECT_EQ(output[i].array[i], i);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

TEST_F(RS_StructOfArraysTest, ResizeToZeroWithCopy) {
    RS_StructOfArrays<B> soa(10, &provider);
    soa.resize(0, true);
    EXPECT_EQ(soa.size(), 0);
}

TEST_F(RS_StructOfArraysTest, SequentialResizes) {
    RS_StructOfArrays<B> soa(5, &provider);
    EXPECT_EQ(soa.size(), 5);
    soa.resize(10);
    EXPECT_EQ(soa.size(), 10);
    soa.resize(3);
    EXPECT_EQ(soa.size(), 3);
    soa.resize(8);
    EXPECT_EQ(soa.size(), 8);
    soa.resize(0);
    EXPECT_EQ(soa.size(), 0);
}

/*****************************************************************
 * Test cases for the RS_StructOfArrays::fill()
 *
 * Test Cases:
 * 1. Fill empty instance.
 * 2. Fill non-empty instance.
 * 3. Fill after resize.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, FillEmptyInstance) {
    RS_StructOfArrays<B> soa(0, &provider);
    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;

    EXPECT_NO_THROW(soa.fill(value));
    EXPECT_EQ(soa.size(), 0);  // Size should remain zero after fill
}

TEST_F(RS_StructOfArraysTest, FillNonEmptyInstance) {
    RS_StructOfArrays<B> soa(10, &provider);

    B value;
    value.struct_a.m = 2;
    value.struct_a.v = {2, 4.0f, 6.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i * 2;
    }
    value.n = 2;

    soa.fill(value);

    B output[10];
    soa.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 2);
        EXPECT_EQ(output[i].struct_a.v.x, 2);
        EXPECT_EQ(output[i].struct_a.v.y, 4.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 6.0);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j * 2);
        }
        EXPECT_EQ(output[i].n, 2);
    }
}

TEST_F(RS_StructOfArraysTest, FillAfterResize) {
    RS_StructOfArrays<B> soa(5, &provider);

    B value;
    value.struct_a.m = 3;
    value.struct_a.v = {3, 6.0f, 9.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i * 3;
    }
    value.n = 3;

    soa.fill(value);

    soa.resize(10);

    B newValue;
    newValue.struct_a.m = 4;
    newValue.struct_a.v = {4, 8.0f, 12.0};
    for (int i = 0; i < 10; i++) {
        newValue.array[i] = i * 4;
    }
    newValue.n = 4;

    soa.fill(newValue);

    B output[10];
    soa.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 4);
        EXPECT_EQ(output[i].struct_a.v.x, 4);
        EXPECT_EQ(output[i].struct_a.v.y, 8.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 12.0);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j * 4);
        }
        EXPECT_EQ(output[i].n, 4);
    }
}

/*****************************************************************
 * Test cases for the RS_StructOfArrays::write()
 *
 * Test cases:
 * 1. Write to empty instance.
 * 2. Write fewer elements:
 *      Test writing a number of elements less than the current size.
 * 3. Write same elements.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, WriteToEmptyInstance) {
    RS_StructOfArrays<B> soa(0, &provider);

    B input[5];
    EXPECT_THROW(soa.write(input, 5), RS_OutOfBoundsException);
}

TEST_F(RS_StructOfArraysTest, WriteFewerElements) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input, 5);
    EXPECT_EQ(soa.size(), 10);

    B output[10];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        EXPECT_EQ(output[i].struct_a.v.x, i);
        EXPECT_EQ(output[i].struct_a.v.y, static_cast<float>(i));
        EXPECT_EQ(output[i].struct_a.v.z, static_cast<double>(i));
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

TEST_F(RS_StructOfArraysTest, WriteSameElements) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input);
    EXPECT_EQ(soa.size(), 10);

    B output[10];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        EXPECT_EQ(output[i].struct_a.v.x, i);
        EXPECT_EQ(output[i].struct_a.v.y, static_cast<float>(i));
        EXPECT_EQ(output[i].struct_a.v.z, static_cast<double>(i));
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

/*****************************************************************
 * Test cases for the RS_StructOfArrays::writeSlice()
 *
 * Test cases:
 * 1. Valid slice write:
 *      Test writing a slice where offset + num is less than or equal
 *      to the current size.
 * 2. Slice exceeds size:
 *      Test writing a slice where offset + num exceeds the current
 *      size to ensure an exception is thrown.
 * 3. Write at beginning:
 *      Test writing a slice with offset = 0 and num less than or equal
 *      to size.
 * 4. Write zero elements:
 *      Test writing a slice with num = 0.
 * 5. Offset beyond size:
 *      Test writing a slice with offset beyond the current size to
 *      ensure an exception is thrown.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, WriteSliceValid) {
    RS_StructOfArrays<B> soa(10, &provider);

    B initial[10];
    for (int i = 0; i < 10; i++) {
        initial[i].struct_a.m = i;
        initial[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            initial[i].array[j] = i * 10 + j;
        }
        initial[i].n = 1;
    }

    soa.write(initial);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i + 10;
        input[i].struct_a.v = {i + 10, static_cast<float>(i + 10), static_cast<double>(i + 10)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = (i + 10) * 10 + j;
        }
        input[i].n = 2;
    }

    soa.writeSlice(input, 5, 1);

    B output[10];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i + 1].struct_a.m, i + 10);
        EXPECT_EQ(output[i + 1].struct_a.v.x, i + 10);
        EXPECT_EQ(output[i + 1].struct_a.v.y, static_cast<float>(i + 10));
        EXPECT_EQ(output[i + 1].struct_a.v.z, static_cast<double>(i + 10));
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i + 1].array[j], (i + 10) * 10 + j);
        }
        EXPECT_EQ(output[i + 1].n, 2);
    }

    for (int i = 0; i < 10; i++) {
        if (i < 1 || i >= 6) {
            EXPECT_EQ(output[i].struct_a.m, i);
            EXPECT_EQ(output[i].struct_a.v.x, i);
            EXPECT_EQ(output[i].struct_a.v.y, static_cast<float>(i));
            EXPECT_EQ(output[i].struct_a.v.z, static_cast<double>(i));
            for (int j = 0; j < 10; j++) {
                EXPECT_EQ(output[i].array[j], i * 10 + j);
            }
            EXPECT_EQ(output[i].n, 1);
        }
    }
}

TEST_F(RS_StructOfArraysTest, WriteSliceExceedsSize) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i + 10;
        input[i].struct_a.v = {i + 10, static_cast<float>(i + 10), static_cast<double>(i + 10)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 2;
    }

    EXPECT_THROW(soa.writeSlice(input, 5, 6), RS_OutOfBoundsException);
}

TEST_F(RS_StructOfArraysTest, WriteSliceAtBeginning) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i + 10;
        input[i].struct_a.v = {i + 10, static_cast<float>(i + 10), static_cast<double>(i + 10)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 2;
    }

    soa.writeSlice(input, 5, 0);

    B output[10];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i + 10);
        EXPECT_EQ(output[i].struct_a.v.x, i + 10);
        EXPECT_EQ(output[i].struct_a.v.y, static_cast<float>(i + 10));
        EXPECT_EQ(output[i].struct_a.v.z, static_cast<double>(i + 10));
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 2);
    }
}

TEST_F(RS_StructOfArraysTest, WriteSliceZeroElements) {
    RS_StructOfArrays<B> soa(10, &provider);

    B initial[10];
    for (int i = 0; i < 10; i++) {
        initial[i].struct_a.m = i;
        initial[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            initial[i].array[j] = i * 10 + j;
        }
        initial[i].n = 1;
    }

    soa.write(initial);

    B input[0];
    soa.writeSlice(input, 0, 0);

    B output[10];
    soa.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        EXPECT_EQ(output[i].struct_a.v.x, i);
        EXPECT_EQ(output[i].struct_a.v.y, static_cast<float>(i));
        EXPECT_EQ(output[i].struct_a.v.z, static_cast<double>(i));
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

TEST_F(RS_StructOfArraysTest, WriteSliceOffsetBeyondSize) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i + 10;
        input[i].struct_a.v = {i + 10, static_cast<float>(i + 10), static_cast<double>(i + 10)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 2;
    }

    EXPECT_THROW(soa.writeSlice(input, 5, 11), RS_OutOfBoundsException);
}

/*****************************************************************
 * Test cases for the RS_StructOfArrays::copy()
 *
 * Test cases:
 * 1. Copy same size.
 * 2. Copy different size.
 * 3. Copy from empty instance.
 * 4. Overwrite existing data.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, CopySameSize) {
    RS_StructOfArrays<B> source(5, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;
    source.fill(value);

    RS_StructOfArrays<B> destination(5, &provider);
    destination.copy(source);
    EXPECT_EQ(destination.size(), 5);

    B output[5];
    destination.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        EXPECT_EQ(output[i].struct_a.v.x, 1);
        EXPECT_EQ(output[i].struct_a.v.y, 2.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 3.0);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

TEST_F(RS_StructOfArraysTest, CopyDifferentSize) {
    RS_StructOfArrays<B> source(10, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;
    source.fill(value);

    RS_StructOfArrays<B> destination(5, &provider);
    destination.copy(source);
    EXPECT_EQ(destination.size(), 10);

    B output[10];
    destination.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        EXPECT_EQ(output[i].struct_a.v.x, 1);
        EXPECT_EQ(output[i].struct_a.v.y, 2.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 3.0);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

TEST_F(RS_StructOfArraysTest, CopyFromEmptyInstance) {
    RS_StructOfArrays<B> source(0, &provider);

    RS_StructOfArrays<B> destination(10, &provider);
    destination.copy(source);

    EXPECT_EQ(destination.size(), 0);
}

TEST_F(RS_StructOfArraysTest, OverwriteExistingData) {
    RS_StructOfArrays<B> source(5, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;
    source.fill(value);

    RS_StructOfArrays<B> destination(5, &provider);

    // Fill destination with different data
    B newValue;
    newValue.struct_a.m = 2;
    newValue.struct_a.v = {2, 4.0f, 6.0};
    for (int i = 0; i < 10; i++) {
        newValue.array[i] = i * 2;
    }
    newValue.n = 2;
    destination.fill(newValue);

    // Overwrite destination with source data
    destination.copy(source);

    B output[5];
    destination.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        EXPECT_EQ(output[i].struct_a.v.x, 1);
        EXPECT_EQ(output[i].struct_a.v.y, 2.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 3.0);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

/*****************************************************************
 * Test cases for the RS_StructOfArrays::read()
 *
 * Test Cases:
 * 1. Read from empty instance.
 * 2. Read data.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, ReadFromEmptyInstance) {
    RS_StructOfArrays<B> soa(0, &provider);

    B output[1];
    output[0].struct_a.m = -1;  // Initialize to check if it changes

    EXPECT_NO_THROW(soa.read(output));
    EXPECT_EQ(output[0].struct_a.m, -1);  // Should remain unchanged
}

TEST_F(RS_StructOfArraysTest, ReadData) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input);

    B output[5];
    EXPECT_NO_THROW(soa.read(output));
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        EXPECT_EQ(output[i].struct_a.v.x, i);
        EXPECT_EQ(output[i].struct_a.v.y, static_cast<float>(i));
        EXPECT_EQ(output[i].struct_a.v.z, static_cast<double>(i));
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

/*****************************************************************
 * Test cases for the RS_StructOfArrays::readSlice()
 *
 * Test cases:
 * 1. Valid slice read:
 *      Test reading a slice where offset + num is less than or equal
 *      to the size.
 * 2. Slice exceeds size:
 *      Test reading a slice where offset + num exceeds the size to ensure
 *      an exception is thrown.
 * 3. Read from beginning:
 *      Test reading a slice with offset = 0 and num less than or equal
 *      to size.
 * 4. Read zero elements:
 *      Test reading a slice with num = 0 to ensure no data is read
 *      and the output remains unchanged.
 * 5. Offset beyond size:
 *      Test reading a slice with offset beyond the size to ensure
 *      an exception is thrown.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, ReadSliceValid) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input);

    B output[5];
    soa.readSlice(output, 5, 2);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i + 2);
        EXPECT_EQ(output[i].struct_a.v.x, i + 2);
        EXPECT_EQ(output[i].struct_a.v.y, static_cast<float>(i + 2));
        EXPECT_EQ(output[i].struct_a.v.z, static_cast<double>(i + 2));
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], (i + 2) * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

TEST_F(RS_StructOfArraysTest, ReadSliceExceedsSize) {
    RS_StructOfArrays<B> soa(10, &provider);

    B output[5];
    EXPECT_THROW(soa.readSlice(output, 5, 7), RS_OutOfBoundsException);
}

TEST_F(RS_StructOfArraysTest, ReadSliceFromBeginning) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input);

    B output[5];
    soa.readSlice(output, 5, 0);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        EXPECT_EQ(output[i].struct_a.v.x, i);
        EXPECT_EQ(output[i].struct_a.v.y, static_cast<float>(i));
        EXPECT_EQ(output[i].struct_a.v.z, static_cast<double>(i));
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
}

TEST_F(RS_StructOfArraysTest, ReadSliceZeroElements) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input);

    B output[10];
    for (int i = 0; i < 10; i++) {
        output[i].struct_a.m = i + 1;
        output[i].struct_a.v = {i + 1, static_cast<float>(i + 1), static_cast<double>(i + 1)};
        for (int j = 0; j < 10; j++) {
            output[i].array[j] = i * 10 + j + 1;
        }
        output[i].n = 2;
    }
    soa.readSlice(output, 0, 5);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, i + 1);
        EXPECT_EQ(output[i].struct_a.v.x, i + 1);
        EXPECT_EQ(output[i].struct_a.v.y, static_cast<float>(i + 1));
        EXPECT_EQ(output[i].struct_a.v.z, static_cast<double>(i + 1));
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j + 1);
        }
        EXPECT_EQ(output[i].n, 2);
    }
}

TEST_F(RS_StructOfArraysTest, ReadSliceOffsetBeyondSize) {
    RS_StructOfArrays<B> soa(10, &provider);

    B output[5];
    EXPECT_THROW(soa.readSlice(output, 5, 11), RS_OutOfBoundsException);
}

/*****************************************************************
 * Test cases for the RS_StructOfArrays::get()
 *
 * Test cases:
 * 1. Get scalar field:
 *    Test accessing a scalar field with a valid index to ensure it
 *    returns a reference to the correct element.
 *
 * 2. Get and set scalar field:
 *    Test modifying the value returned by get<N>(index) and verify
 *    that the internal data is updated accordingly.
 *
 * 3. Get array field:
 *    Test accessing an array field to ensure it returns an accessor
 *    object that allows element-wise access and modification.
 *
 * 4. Get nested structure (copy):
 *    Test accessing a nested structure field to ensure it returns a
 *    copy, and verify that modifications to the copy do not affect
 *    the internal data.
 *
 * 5. Get nested field reference:
 *    Test accessing nested fields using get<N, N1, Ns...>(index) to
 *    ensure it returns a reference to the nested field.
 *
 * 6. Get and set nested field:
 *    Test modifying the value returned by get<N, N1, Ns...>(index)
 *    and verify that the internal data is updated correctly.
 *
 * 7. Get entire element:
 *    Test getting an entire element using get(index) to confirm it
 *    returns a copy and that modifications do not affect internal data.
 *****************************************************************/

__global__ void testGetScalarField(RS_StructOfArrays<B> soa, int *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        result[index] = soa.get<Pr(B, struct_a), Pr(A, m)>(index);
    }
}

TEST_F(RS_StructOfArraysTest, GetScalarField) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input);

    int *d_result;
    cudaMalloc(&d_result, 10 * sizeof(int));
    testGetScalarField<<<1, 10>>>(soa, d_result);

    int h_result[10];
    cudaMemcpy(h_result, d_result, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(h_result[i], i);
    }
    cudaFree(d_result);
}

__global__ void testGetSetScalarField(RS_StructOfArrays<B> soa, int value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        soa.get<Pr(B, struct_a), Pr(A, m)>(index) = value;
    }
}

TEST_F(RS_StructOfArraysTest, GetSetScalarField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input);

    // Modify the scalar field using the kernel
    testGetSetScalarField<<<1, 5>>>(soa, 42);

    // Verify the changes
    B output[5];
    soa.read(output);
    EXPECT_EQ(output[2].struct_a.m, 42);  // Ensure the field at index 2 is modified
}

__global__ void testGetArrayField(RS_StructOfArrays<B> soa, int *result, int arrayIndex) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        result[index] = soa.get<Pr(B, array)>(index)[arrayIndex];
    }
}

TEST_F(RS_StructOfArraysTest, GetArrayField) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input);

    int *d_result;
    cudaMalloc(&d_result, 10 * sizeof(int));
    testGetArrayField<<<1, 10>>>(soa, d_result, 5);

    int h_result[10];
    cudaMemcpy(h_result, d_result, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(h_result[i], i * 10 + 5);
    }
    cudaFree(d_result);
}

__global__ void testGetNestedStructureCopy(RS_StructOfArrays<B> soa, Vector *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        Vector v_copy = soa.get<Pr(B, struct_a), Pr(A, v)>(index);
        v_copy.x = -1;  // Modify the copy
        result[index] = v_copy;
    }
}

TEST_F(RS_StructOfArraysTest, GetNestedStructureCopy) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }

    soa.write(input);

    Vector *d_result;
    cudaMalloc(&d_result, 5 * sizeof(Vector));
    testGetNestedStructureCopy<<<1, 5>>>(soa, d_result);

    Vector h_result[5];
    cudaMemcpy(h_result, d_result, 5 * sizeof(Vector), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(h_result[i].x, -1);          // Ensure copy is modified
        EXPECT_EQ(output[i].struct_a.v.x, i);  // Ensure original is not modified
    }
}

__global__ void testGetNestedFieldReference(RS_StructOfArrays<B> soa, int *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        result[index] = soa.get<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(index);
    }
}

TEST_F(RS_StructOfArraysTest, GetNestedFieldReference) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }

    soa.write(input);

    int *d_result;
    cudaMalloc(&d_result, 5 * sizeof(int));
    testGetNestedFieldReference<<<1, 5>>>(soa, d_result);

    int h_result[5];
    cudaMemcpy(h_result, d_result, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(h_result[i], i);  // Ensure the correct nested field is accessed
    }
    cudaFree(d_result);
}

__global__ void testGetSetNestedField(RS_StructOfArrays<B> soa, int value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        soa.get<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(index) = value;
    }
}

TEST_F(RS_StructOfArraysTest, GetSetNestedField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }

    soa.write(input);

    testGetSetNestedField<<<1, 5>>>(soa, 42);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.v.x, 42);  // Ensure the nested field is modified correctly
    }
}

__global__ void testGetEntireElement(RS_StructOfArrays<B> soa, B *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        B elem_copy = soa.get(index);
        elem_copy.struct_a.m = -1;  // Modify the copy
        result[index] = elem_copy;
    }
}

TEST_F(RS_StructOfArraysTest, GetEntireElement) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
    }

    soa.write(input);

    B *d_result;
    cudaMalloc(&d_result, 5 * sizeof(B));
    testGetEntireElement<<<1, 5>>>(soa, d_result);

    B h_result[5];
    cudaMemcpy(h_result, d_result, 5 * sizeof(B), cudaMemcpyDeviceToHost);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(h_result[i].struct_a.m, -1);  // Ensure copy is modified
        EXPECT_EQ(output[i].struct_a.m, i);     // Ensure original is not modified
    }
    cudaFree(d_result);
}

/*****************************************************************
 * Test cases for the RS_StructOfArrays::set()
 *
 * Test cases:
 * 1. Set scalar field:
 *    Test setting a scalar field at a valid index to ensure the
 *    internal data updates correctly.
 *
 * 2. Set array field:
 *    Test setting an array field with appropriate data to verify
 *    the entire array field updates correctly.
 *
 * 3. Set nested structure field:
 *    Test setting a nested structure field to ensure that all
 *    nested data is updated.
 *
 * 4. Set nested field:
 *    Test setting nested fields using set<N, N1, Ns...>(index, value)
 *    to confirm that the correct nested field is updated.
 *
 * 5. Set entire element:
 *    Test setting an entire element to verify that all fields,
 *    including nested ones, are updated accordingly.
 *****************************************************************/

__global__ void testSetScalarField(RS_StructOfArrays<B> soa, int value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        soa.set<Pr(B, struct_a), Pr(A, m)>(index, value);
    }
}

TEST_F(RS_StructOfArraysTest, SetScalarField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
    }

    soa.write(input);

    testSetScalarField<<<1, 5>>>(soa, 42);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 42);  // Ensure the field is updated
    }
}

__global__ void testSetArrayField(RS_StructOfArrays<B> soa, int *d_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        soa.set<Pr(B, array)>(index, d_values);
    }
}

TEST_F(RS_StructOfArraysTest, SetArrayField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
    }

    soa.write(input);

    int h_values[10];
    for (int i = 0; i < 10; i++) {
        h_values[i] = 42 + i;
    }

    int *d_values;
    cudaMalloc(&d_values, 10 * sizeof(int));
    cudaMemcpy(d_values, h_values, 10 * sizeof(int), cudaMemcpyHostToDevice);

    testSetArrayField<<<1, 5>>>(soa, d_values);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);  // Ensure the array is updated
        }
    }

    cudaFree(d_values);
}

__global__ void testSetArrayFieldWithKernelArray(RS_StructOfArrays<B> soa) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        int array[10];
        for (int i = 0; i < 10; i++) {
            array[i] = 42 + i;
        }
        soa.set<Pr(B, array)>(index, array);
    }
}

TEST_F(RS_StructOfArraysTest, SetArrayFieldWithKernelArray) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
    }

    soa.write(input);

    testSetArrayFieldWithKernelArray<<<1, 5>>>(soa);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);  // Ensure the array is updated
        }
    }
}

__global__ void testSetNestedStructureField(RS_StructOfArrays<B> soa, Vector value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        soa.set<Pr(B, struct_a), Pr(A, v)>(index, value);
    }
}

TEST_F(RS_StructOfArraysTest, SetNestedStructureField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }

    soa.write(input);

    Vector newValue = {42, 42.0f, 42.0};

    testSetNestedStructureField<<<1, 5>>>(soa, newValue);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.v.x, 42);
        EXPECT_EQ(output[i].struct_a.v.y, 42.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 42.0);  // Ensure the nested structure is updated
    }
}

__global__ void testSetNestedField(RS_StructOfArrays<B> soa, int value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        soa.set<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(index, value);
    }
}

TEST_F(RS_StructOfArraysTest, SetNestedField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }

    soa.write(input);

    testSetNestedField<<<1, 5>>>(soa, 42);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.v.x, 42);  // Ensure the nested field is updated
    }
}

__global__ void testSetEntireElement(RS_StructOfArrays<B> soa, B value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < soa.size()) {
        soa.set(index, value);
    }
}

TEST_F(RS_StructOfArraysTest, SetEntireElement) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    soa.write(input);

    B newValue;
    newValue.struct_a.m = 42;
    newValue.struct_a.v = {42, 42.0f, 42.0};
    for (int i = 0; i < 10; i++) {
        newValue.array[i] = 42 + i;
    }
    newValue.n = 42;

    testSetEntireElement<<<1, 5>>>(soa, newValue);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 42);
        EXPECT_EQ(output[i].struct_a.v.x, 42);
        EXPECT_EQ(output[i].struct_a.v.y, 42.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 42.0);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);
        }
        EXPECT_EQ(output[i].n, 42);  // Ensure the entire element is updated
    }
}