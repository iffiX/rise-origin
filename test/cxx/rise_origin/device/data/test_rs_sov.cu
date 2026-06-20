#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_sov.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_StructOfVectors (SoV) is a structure of vectors container that
 * turns an array of structures layout into a structure of vectors
 * layout for better memory coalescing on the device. Unlike the SoA,
 * the SoV tracks a used count separate from the allocated capacity
 * and supports dynamic insert and erase operations.
 *
 * - Construction & Free
 *   Default constructor, sized constructor, init(), free()
 *
 * - Capacity Management
 *   size(), capacity(), empty(), resize(), clear()
 *
 * - Host Data Operations
 *   fill(), fillSlice(), write(), writeSlice(), read(), readSlice(), copy()
 *
 * - Modifiers
 *   append(), insert(), erase(), eraseRange()
 *
 * - Field Access (Device)
 *   get<>() and set<>() for scalar, array, and nested fields
 *
 * - Reflection
 *   Uses REFL_AUTO to decompose the struct into member arrays
 *****************************************************************/

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
    unsigned int n;
};

REFL_AUTO(type(Vector), field(x), field(y), field(z))

REFL_AUTO(type(A), field(m), field(v))

REFL_AUTO(type(B), field(struct_a), field(array), field(n))

static B makeB(int m) {
    B value;
    value.struct_a.m = m;
    value.struct_a.v = {m, static_cast<float>(m), static_cast<double>(m)};
    for (int j = 0; j < 10; j++) {
        value.array[j] = m * 10 + j;
    }
    value.n = static_cast<unsigned int>(m);
    return value;
}

class RS_StructOfVectorsTest : public ::testing::Test {
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
 * Test cases for RS_StructOfVectors::free()
 *
 * Test cases:
 * 1. Free empty instance.
 * 2. Free non-empty instance.
 * 3. Multiple frees handled gracefully.
 * 4. Free after operations.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, FreeEmptyInstance) {
    RS_StructOfVectors<B> sov(0, &provider);
    EXPECT_NO_THROW(sov.free());
    EXPECT_EQ(sov.size(), 0);
}

TEST_F(RS_StructOfVectorsTest, FreeNonEmptyInstance) {
    RS_StructOfVectors<B> sov(10, &provider);
    EXPECT_EQ(sov.capacity(), 10);
    EXPECT_NO_THROW(sov.free());
    EXPECT_EQ(sov.size(), 0);
}

TEST_F(RS_StructOfVectorsTest, MultipleFrees) {
    RS_StructOfVectors<B> sov(10, &provider);
    EXPECT_NO_THROW(sov.free());
    EXPECT_NO_THROW(sov.free());
    EXPECT_EQ(sov.size(), 0);
}

TEST_F(RS_StructOfVectorsTest, FreeAfterOperations) {
    RS_StructOfVectors<B> sov(10, &provider);

    B value = makeB(1);
    sov.fill(value);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i] = makeB(i);
    }
    sov.write(input);

    EXPECT_NO_THROW(sov.free());
    EXPECT_EQ(sov.size(), 0);
}

/*****************************************************************
 * Test cases for RS_StructOfVectors construction
 *
 * Test cases:
 * 1. Default construction with no provider.
 * 2. Construction with a provider only.
 * 3. Construction with a capacity and a provider.
 * 4. Init with a provider only.
 * 5. Init with a capacity and a provider.
 * 6. Zero capacity construction.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, DefaultConstruction) {
    RS_StructOfVectors<B> sov;
    EXPECT_EQ(sov.capacity(), 0);
    EXPECT_EQ(sov.size(), 0);
    EXPECT_TRUE(sov.empty());
}

TEST_F(RS_StructOfVectorsTest, ConstructionWithProvider) {
    RS_StructOfVectors<B> sov(&provider);
    EXPECT_EQ(sov.capacity(), 0);
    EXPECT_EQ(sov.size(), 0);
    EXPECT_TRUE(sov.empty());
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ConstructionWithSize) {
    RS_StructOfVectors<B> sov(10, &provider);
    EXPECT_EQ(sov.capacity(), 10);
    EXPECT_EQ(sov.size(), 0);
    EXPECT_TRUE(sov.empty());
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, InitWithProvider) {
    RS_StructOfVectors<B> sov;
    sov.init(&provider);
    EXPECT_EQ(sov.capacity(), 0);
    EXPECT_EQ(sov.size(), 0);
    EXPECT_TRUE(sov.empty());
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, InitWithSize) {
    RS_StructOfVectors<B> sov;
    sov.init(5, &provider);
    EXPECT_EQ(sov.capacity(), 5);
    EXPECT_EQ(sov.size(), 0);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ZeroCapacityConstruction) {
    RS_StructOfVectors<B> sov(0, &provider);
    EXPECT_EQ(sov.capacity(), 0);
    EXPECT_EQ(sov.size(), 0);
    EXPECT_TRUE(sov.empty());
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors::resize()
 *
 * Test cases:
 * 1. Resize zero to zero.
 * 2. Resize zero to positive.
 * 3. Resize larger without copy.
 * 4. Resize smaller without copy.
 * 5. Resize larger with copy.
 * 6. Resize smaller with copy.
 * 7. Resize to zero with copy.
 * 8. Sequential resizes.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, ResizeZeroToZero) {
    RS_StructOfVectors<B> sov(0, &provider);
    sov.resize(0);
    EXPECT_EQ(sov.size(), 0);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ResizeZeroToPositive) {
    RS_StructOfVectors<B> sov(0, &provider);
    sov.resize(5);
    EXPECT_EQ(sov.capacity(), 5);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ResizeLargerWithoutCopy) {
    RS_StructOfVectors<B> sov(0, &provider);
    sov.resize(10, false);
    EXPECT_EQ(sov.capacity(), 10);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ResizeSmallerWithoutCopy) {
    RS_StructOfVectors<B> sov(10, &provider);
    sov.resize(5, false);
    EXPECT_EQ(sov.capacity(), 5);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ResizeLargerWithCopy) {
    RS_StructOfVectors<B> sov(5, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 10;
    sov.fill(value);

    sov.resize(10, true);
    EXPECT_EQ(sov.capacity(), 10);

    B output[5];
    sov.readSlice(output, 5, 0);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        EXPECT_EQ(output[i].struct_a.v.x, 1);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 10);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ResizeSmallerWithCopy) {
    RS_StructOfVectors<B> sov(10, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 10;
    sov.fill(value);

    sov.resize(5, true);
    EXPECT_EQ(sov.capacity(), 5);

    B output[5];
    sov.readSlice(output, 5, 0);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        EXPECT_EQ(output[i].struct_a.v.x, 1);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 10);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ResizeToZeroWithCopy) {
    RS_StructOfVectors<B> sov(10, &provider);
    sov.resize(0, true);
    EXPECT_EQ(sov.capacity(), 0);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, SequentialResizes) {
    RS_StructOfVectors<B> sov(5, &provider);
    EXPECT_EQ(sov.capacity(), 5);
    sov.resize(10);
    EXPECT_EQ(sov.capacity(), 10);
    sov.resize(3);
    EXPECT_EQ(sov.capacity(), 3);
    sov.resize(8);
    EXPECT_EQ(sov.capacity(), 8);
    sov.resize(0);
    EXPECT_EQ(sov.capacity(), 0);
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors capacity management
 *
 * Test cases:
 * 1. Resize up grows the capacity and keeps the used data.
 * 2. Resize down keeps the data within the new capacity.
 * 3. Clear keeps the capacity and empties the used range.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, ResizeUpGrowsCapacity) {
    RS_StructOfVectors<B> sov(5, &provider);
    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i] = makeB(i);
    }
    sov.write(input);
    EXPECT_EQ(sov.capacity(), 5);

    sov.resize(10, true);
    EXPECT_EQ(sov.capacity(), 10);
    EXPECT_EQ(sov.size(), 5);

    B output[5];
    sov.readSlice(output, 5, 0);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ResizeDownKeepsData) {
    RS_StructOfVectors<B> sov(5, &provider);
    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i] = makeB(i);
    }
    sov.write(input);

    sov.resize(3, true);
    EXPECT_EQ(sov.capacity(), 3);
    EXPECT_EQ(sov.size(), 3);

    B output[3];
    sov.read(output);
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ClearPreservesCapacity) {
    RS_StructOfVectors<B> sov(5, &provider);
    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i] = makeB(i);
    }
    sov.write(input);
    EXPECT_EQ(sov.size(), 5);

    sov.clear();
    EXPECT_EQ(sov.capacity(), 5);
    EXPECT_EQ(sov.size(), 0);
    EXPECT_TRUE(sov.empty());
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors::fill()
 *
 * Test cases:
 * 1. Fill empty instance.
 * 2. Fill non-empty instance.
 * 3. Fill after resize.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, FillEmptyInstance) {
    RS_StructOfVectors<B> sov(0, &provider);
    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;

    EXPECT_NO_THROW(sov.fill(value));
    EXPECT_EQ(sov.size(), 0);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, FillNonEmptyInstance) {
    RS_StructOfVectors<B> sov(10, &provider);

    B value;
    value.struct_a.m = 2;
    value.struct_a.v = {2, 4.0f, 6.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i * 2;
    }
    value.n = 2;

    sov.fill(value);

    B output[10];
    sov.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 2);
        EXPECT_EQ(output[i].struct_a.v.x, 2);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j * 2);
        }
        EXPECT_EQ(output[i].n, 2);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, FillAfterResize) {
    RS_StructOfVectors<B> sov(5, &provider);

    B value;
    value.struct_a.m = 3;
    value.struct_a.v = {3, 6.0f, 9.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i * 3;
    }
    value.n = 3;

    sov.fill(value);

    sov.resize(10);

    B new_value;
    new_value.struct_a.m = 4;
    new_value.struct_a.v = {4, 8.0f, 12.0};
    for (int i = 0; i < 10; i++) {
        new_value.array[i] = i * 4;
    }
    new_value.n = 4;

    sov.fill(new_value);

    B output[10];
    sov.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 4);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j * 4);
        }
        EXPECT_EQ(output[i].n, 4);
    }
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors::fillSlice()
 *
 * Test cases:
 * 1. Fill slice at the beginning.
 * 2. Fill slice in the middle.
 * 3. Fill a single element slice.
 * 4. Fill zero elements is a no operation.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, FillSliceAtBeginning) {
    RS_StructOfVectors<B> sov(10, &provider);

    B initial_value;
    initial_value.struct_a.m = 0;
    initial_value.struct_a.v = {0, 0.0f, 0.0};
    for (int i = 0; i < 10; i++) {
        initial_value.array[i] = 0;
    }
    initial_value.n = 0;
    sov.fill(initial_value);

    B fill_value;
    fill_value.struct_a.m = 42;
    fill_value.struct_a.v = {42, 42.0f, 42.0};
    for (int i = 0; i < 10; i++) {
        fill_value.array[i] = 42 + i;
    }
    fill_value.n = 42;

    sov.fillSlice(fill_value, 3, 0);

    B output[10];
    sov.read(output);

    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(output[i].struct_a.m, 42);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);
        }
        EXPECT_EQ(output[i].n, 42);
    }

    for (int i = 3; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 0);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, FillSliceInMiddle) {
    RS_StructOfVectors<B> sov(10, &provider);

    B initial_value;
    initial_value.struct_a.m = 0;
    initial_value.struct_a.v = {0, 0.0f, 0.0};
    for (int i = 0; i < 10; i++) {
        initial_value.array[i] = 0;
    }
    initial_value.n = 0;
    sov.fill(initial_value);

    B fill_value;
    fill_value.struct_a.m = 99;
    fill_value.struct_a.v = {99, 99.0f, 99.0};
    for (int i = 0; i < 10; i++) {
        fill_value.array[i] = 99 + i;
    }
    fill_value.n = 99;

    sov.fillSlice(fill_value, 4, 3);

    B output[10];
    sov.read(output);

    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(output[i].struct_a.m, 0);
    }

    for (int i = 3; i < 7; i++) {
        EXPECT_EQ(output[i].struct_a.m, 99);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 99 + j);
        }
        EXPECT_EQ(output[i].n, 99);
    }

    for (int i = 7; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 0);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, FillSliceSingleElement) {
    RS_StructOfVectors<B> sov(10, &provider);

    B initial_value;
    initial_value.struct_a.m = 0;
    initial_value.struct_a.v = {0, 0.0f, 0.0};
    for (int i = 0; i < 10; i++) {
        initial_value.array[i] = 0;
    }
    initial_value.n = 0;
    sov.fill(initial_value);

    B fill_value;
    fill_value.struct_a.m = 123;
    fill_value.struct_a.v = {123, 123.0f, 123.0};
    for (int i = 0; i < 10; i++) {
        fill_value.array[i] = 123 + i;
    }
    fill_value.n = 123;

    sov.fillSlice(fill_value, 1, 5);

    B output[10];
    sov.read(output);

    for (int i = 0; i < 10; i++) {
        if (i == 5) {
            EXPECT_EQ(output[i].struct_a.m, 123);
            for (int j = 0; j < 10; j++) {
                EXPECT_EQ(output[i].array[j], 123 + j);
            }
            EXPECT_EQ(output[i].n, 123);
        } else {
            EXPECT_EQ(output[i].struct_a.m, 0);
        }
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, FillSliceZeroElements) {
    RS_StructOfVectors<B> sov(10, &provider);

    B initial_value;
    initial_value.struct_a.m = 1;
    initial_value.struct_a.v = {1, 1.0f, 1.0};
    for (int i = 0; i < 10; i++) {
        initial_value.array[i] = i;
    }
    initial_value.n = 1;
    sov.fill(initial_value);

    B fill_value;
    fill_value.struct_a.m = 999;
    fill_value.struct_a.v = {999, 999.0f, 999.0};
    for (int i = 0; i < 10; i++) {
        fill_value.array[i] = 999;
    }
    fill_value.n = 89;

    EXPECT_NO_THROW(sov.fillSlice(fill_value, 0, 5));

    B output[10];
    sov.read(output);

    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
    }
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors::write()
 *
 * Test cases:
 * 1. Write to an empty instance.
 * 2. Write data and read it back.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, WriteToEmptyInstance) {
    RS_StructOfVectors<B> sov(0, &provider);
    B input[5];
    EXPECT_NO_THROW(sov.write(input));
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, WriteData) {
    RS_StructOfVectors<B> sov(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    sov.write(input);
    EXPECT_EQ(sov.size(), 10);

    B output[10];
    sov.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors::writeSlice()
 *
 * Test cases:
 * 1. Valid slice write at an offset.
 * 2. Slice exceeds the capacity throws.
 * 3. Write at the beginning.
 * 4. Write zero elements.
 * 5. Offset beyond the capacity throws.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, WriteSliceValid) {
    RS_StructOfVectors<B> sov(10, &provider);

    B initial[10];
    for (int i = 0; i < 10; i++) {
        initial[i].struct_a.m = i;
        initial[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            initial[i].array[j] = i * 10 + j;
        }
        initial[i].n = 1;
    }

    sov.write(initial);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i + 10;
        input[i].struct_a.v = {i + 10, static_cast<float>(i + 10), static_cast<double>(i + 10)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = (i + 10) * 10 + j;
        }
        input[i].n = 2;
    }

    sov.writeSlice(input, 5, 1);

    B output[10];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i + 1].struct_a.m, i + 10);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i + 1].array[j], (i + 10) * 10 + j);
        }
        EXPECT_EQ(output[i + 1].n, 2);
    }

    for (int i = 0; i < 10; i++) {
        if (i < 1 || i >= 6) {
            EXPECT_EQ(output[i].struct_a.m, i);
            EXPECT_EQ(output[i].n, 1);
        }
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, WriteSliceExceedsSize) {
    RS_StructOfVectors<B> sov(10, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i] = makeB(i + 10);
    }

    EXPECT_THROW(sov.writeSlice(input, 5, 6), RS_OutOfBoundsException);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, WriteSliceAtBeginning) {
    RS_StructOfVectors<B> sov(10, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i + 10;
        input[i].struct_a.v = {i + 10, static_cast<float>(i + 10), static_cast<double>(i + 10)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 2;
    }

    sov.writeSlice(input, 5, 0);

    B output[5];
    sov.readSlice(output, 5, 0);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i + 10);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 2);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, WriteSliceZeroElements) {
    RS_StructOfVectors<B> sov(10, &provider);

    B initial[10];
    for (int i = 0; i < 10; i++) {
        initial[i] = makeB(i);
        initial[i].n = 1;
    }

    sov.write(initial);

    sov.writeSlice(initial, 0, 0);

    B output[10];
    sov.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        EXPECT_EQ(output[i].n, 1);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, WriteSliceOffsetBeyondSize) {
    RS_StructOfVectors<B> sov(10, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i] = makeB(i + 10);
    }

    EXPECT_THROW(sov.writeSlice(input, 5, 11), RS_OutOfBoundsException);
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors modifiers
 *
 * Test cases:
 * 1. Append to an empty instance.
 * 2. Append multiple times.
 * 3. Append triggers capacity growth.
 * 4. Insert at the beginning.
 * 5. Insert in the middle.
 * 6. Insert at the end.
 * 7. Erase the first element.
 * 8. Erase the middle element.
 * 9. Erase the last element.
 * 10. Erase a range in the middle.
 * 11. Erase a range at the beginning.
 * 12. Erase a range at the end.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, AppendToEmpty) {
    RS_StructOfVectors<B> sov(0, &provider);
    EXPECT_EQ(sov.size(), 0);

    sov.append(makeB(42));
    EXPECT_EQ(sov.size(), 1);
    EXPECT_GE(sov.capacity(), 1u);

    B output[1];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 42);
    for (int j = 0; j < 10; j++) {
        EXPECT_EQ(output[0].array[j], 42 * 10 + j);
    }
    EXPECT_EQ(output[0].n, 42u);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, AppendMultiple) {
    RS_StructOfVectors<B> sov(0, &provider);
    sov.append(makeB(10));
    sov.append(makeB(20));
    sov.append(makeB(30));
    EXPECT_EQ(sov.size(), 3);

    B output[3];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 10);
    EXPECT_EQ(output[1].struct_a.m, 20);
    EXPECT_EQ(output[2].struct_a.m, 30);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, AppendTriggersCapacityGrowth) {
    RS_StructOfVectors<B> sov(2, &provider);
    B input[2] = {makeB(1), makeB(2)};
    sov.write(input);
    EXPECT_EQ(sov.capacity(), 2);
    EXPECT_EQ(sov.size(), 2);

    sov.append(makeB(3));
    EXPECT_GE(sov.capacity(), 3u);
    EXPECT_EQ(sov.size(), 3);

    B output[3];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 1);
    EXPECT_EQ(output[1].struct_a.m, 2);
    EXPECT_EQ(output[2].struct_a.m, 3);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, InsertAtBeginning) {
    RS_StructOfVectors<B> sov(3, &provider);
    B input[3] = {makeB(2), makeB(3), makeB(4)};
    sov.write(input);

    sov.insert(makeB(1), 0);
    EXPECT_GE(sov.capacity(), 4u);
    EXPECT_EQ(sov.size(), 4);

    B output[4];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 1);
    EXPECT_EQ(output[1].struct_a.m, 2);
    EXPECT_EQ(output[2].struct_a.m, 3);
    EXPECT_EQ(output[3].struct_a.m, 4);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, InsertInMiddle) {
    RS_StructOfVectors<B> sov(4, &provider);
    B input[4] = {makeB(1), makeB(2), makeB(4), makeB(5)};
    sov.write(input);

    sov.insert(makeB(3), 2);
    EXPECT_GE(sov.capacity(), 5u);
    EXPECT_EQ(sov.size(), 5);

    B output[5];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 1);
    EXPECT_EQ(output[1].struct_a.m, 2);
    EXPECT_EQ(output[2].struct_a.m, 3);
    EXPECT_EQ(output[3].struct_a.m, 4);
    EXPECT_EQ(output[4].struct_a.m, 5);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, InsertAtEnd) {
    RS_StructOfVectors<B> sov(3, &provider);
    B input[3] = {makeB(1), makeB(2), makeB(3)};
    sov.write(input);

    sov.insert(makeB(4), 3);
    EXPECT_GE(sov.capacity(), 4u);
    EXPECT_EQ(sov.size(), 4);

    B output[4];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 1);
    EXPECT_EQ(output[1].struct_a.m, 2);
    EXPECT_EQ(output[2].struct_a.m, 3);
    EXPECT_EQ(output[3].struct_a.m, 4);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, EraseFirstElement) {
    RS_StructOfVectors<B> sov(4, &provider);
    B input[4] = {makeB(10), makeB(20), makeB(30), makeB(40)};
    sov.write(input);

    sov.erase(0);
    EXPECT_EQ(sov.size(), 3);

    B output[3];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 20);
    EXPECT_EQ(output[1].struct_a.m, 30);
    EXPECT_EQ(output[2].struct_a.m, 40);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, EraseMiddleElement) {
    RS_StructOfVectors<B> sov(4, &provider);
    B input[4] = {makeB(10), makeB(20), makeB(30), makeB(40)};
    sov.write(input);

    sov.erase(2);
    EXPECT_EQ(sov.size(), 3);

    B output[3];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 10);
    EXPECT_EQ(output[1].struct_a.m, 20);
    EXPECT_EQ(output[2].struct_a.m, 40);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, EraseLastElement) {
    RS_StructOfVectors<B> sov(4, &provider);
    B input[4] = {makeB(10), makeB(20), makeB(30), makeB(40)};
    sov.write(input);

    sov.erase(3);
    EXPECT_EQ(sov.size(), 3);

    B output[3];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 10);
    EXPECT_EQ(output[1].struct_a.m, 20);
    EXPECT_EQ(output[2].struct_a.m, 30);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, EraseRangeInMiddle) {
    RS_StructOfVectors<B> sov(6, &provider);
    B input[6] = {makeB(10), makeB(20), makeB(30), makeB(40), makeB(50), makeB(60)};
    sov.write(input);

    sov.eraseRange(2, 4);
    EXPECT_EQ(sov.size(), 4);

    B output[4];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 10);
    EXPECT_EQ(output[1].struct_a.m, 20);
    EXPECT_EQ(output[2].struct_a.m, 50);
    EXPECT_EQ(output[3].struct_a.m, 60);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, EraseRangeAtBeginning) {
    RS_StructOfVectors<B> sov(5, &provider);
    B input[5] = {makeB(10), makeB(20), makeB(30), makeB(40), makeB(50)};
    sov.write(input);

    sov.eraseRange(0, 2);
    EXPECT_EQ(sov.size(), 3);

    B output[3];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 30);
    EXPECT_EQ(output[1].struct_a.m, 40);
    EXPECT_EQ(output[2].struct_a.m, 50);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, EraseRangeAtEnd) {
    RS_StructOfVectors<B> sov(5, &provider);
    B input[5] = {makeB(10), makeB(20), makeB(30), makeB(40), makeB(50)};
    sov.write(input);

    sov.eraseRange(3, 5);
    EXPECT_EQ(sov.size(), 3);

    B output[3];
    sov.read(output);
    EXPECT_EQ(output[0].struct_a.m, 10);
    EXPECT_EQ(output[1].struct_a.m, 20);
    EXPECT_EQ(output[2].struct_a.m, 30);
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors host out of bounds
 *
 * Test cases:
 * 1. Erase out of bounds throws.
 * 2. Insert out of bounds throws.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, HostEraseOutOfBounds) {
    RS_StructOfVectors<B> sov(5, &host_provider);
    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i] = makeB(i);
    }
    sov.write(input);

    EXPECT_THROW(sov.erase(5), RS_OutOfBoundsException);
    EXPECT_THROW(sov.erase(100), RS_OutOfBoundsException);

    sov.free();
}

TEST_F(RS_StructOfVectorsTest, HostInsertOutOfBounds) {
    RS_StructOfVectors<B> sov(5, &host_provider);
    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i] = makeB(i);
    }
    sov.write(input);

    EXPECT_THROW(sov.insert(makeB(999), 6), RS_OutOfBoundsException);
    EXPECT_THROW(sov.insert(makeB(999), 100), RS_OutOfBoundsException);

    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors::copy()
 *
 * Test cases:
 * 1. Copy same size.
 * 2. Copy different size grows the destination.
 * 3. Copy from an empty instance.
 * 4. Overwrite existing data.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, CopySameSize) {
    RS_StructOfVectors<B> source(5, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;
    source.fill(value);

    RS_StructOfVectors<B> destination(5, &provider);
    destination.copy(source);
    EXPECT_EQ(destination.size(), 5);

    B output[5];
    destination.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    source.free();
    destination.free();
}

TEST_F(RS_StructOfVectorsTest, CopyDifferentSize) {
    RS_StructOfVectors<B> source(10, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;
    source.fill(value);

    RS_StructOfVectors<B> destination(5, &provider);
    destination.copy(source);
    EXPECT_EQ(destination.size(), 10);

    B output[10];
    destination.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    source.free();
    destination.free();
}

TEST_F(RS_StructOfVectorsTest, CopyFromEmptyInstance) {
    RS_StructOfVectors<B> source(0, &provider);

    RS_StructOfVectors<B> destination(10, &provider);
    destination.copy(source);

    EXPECT_EQ(destination.size(), 0);
    source.free();
    destination.free();
}

TEST_F(RS_StructOfVectorsTest, OverwriteExistingData) {
    RS_StructOfVectors<B> source(5, &provider);

    B value;
    value.struct_a.m = 1;
    value.struct_a.v = {1, 2.0f, 3.0};
    for (int i = 0; i < 10; i++) {
        value.array[i] = i;
    }
    value.n = 1;
    source.fill(value);

    RS_StructOfVectors<B> destination(5, &provider);

    B new_value;
    new_value.struct_a.m = 2;
    new_value.struct_a.v = {2, 4.0f, 6.0};
    for (int i = 0; i < 10; i++) {
        new_value.array[i] = i * 2;
    }
    new_value.n = 2;
    destination.fill(new_value);

    destination.copy(source);

    B output[5];
    destination.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    source.free();
    destination.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors::read()
 *
 * Test cases:
 * 1. Read from an empty instance leaves the output unchanged.
 * 2. Read data back.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, ReadFromEmptyInstance) {
    RS_StructOfVectors<B> sov(0, &provider);

    B output[1];
    output[0].struct_a.m = -1;

    EXPECT_NO_THROW(sov.read(output));
    EXPECT_EQ(output[0].struct_a.m, -1);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ReadData) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }

    sov.write(input);

    B output[5];
    EXPECT_NO_THROW(sov.read(output));
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors::readSlice()
 *
 * Test cases:
 * 1. Valid slice read at an offset.
 * 2. Slice exceeds the used range throws.
 * 3. Read from the beginning.
 * 4. Offset beyond the used range throws.
 *****************************************************************/

TEST_F(RS_StructOfVectorsTest, ReadSliceValid) {
    RS_StructOfVectors<B> sov(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 10;
    }

    sov.write(input);

    B output[5];
    sov.readSlice(output, 5, 2);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i + 2);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], (i + 2) * 10 + j);
        }
        EXPECT_EQ(output[i].n, 10u);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ReadSliceExceedsSize) {
    RS_StructOfVectors<B> sov(10, &provider);

    B output[5];
    EXPECT_THROW(sov.readSlice(output, 5, 7), RS_OutOfBoundsException);
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ReadSliceFromBeginning) {
    RS_StructOfVectors<B> sov(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 10;
    }

    sov.write(input);

    B output[5];
    sov.readSlice(output, 5, 0);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 10u);
    }
    sov.free();
}

TEST_F(RS_StructOfVectorsTest, ReadSliceOffsetBeyondSize) {
    RS_StructOfVectors<B> sov(10, &provider);

    B output[5];
    EXPECT_THROW(sov.readSlice(output, 5, 11), RS_OutOfBoundsException);
    sov.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors device get
 *
 * Test cases:
 * 1. Get a scalar field on the device.
 * 2. Get and set a scalar field on the device.
 * 3. Get an array field on the device.
 * 4. Get a nested structure copy on the device.
 * 5. Get a nested field reference on the device.
 * 6. Get and set a nested field on the device.
 * 7. Get an entire element copy on the device.
 *****************************************************************/

class RSCF_SovGetScalarField : public RS_ComputeFunction<RSCF_SovGetScalarField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_StructOfVectors<B> sov, RS_Vector<int> results
    ) {
        RIndex index = index1D();
        if (index < sov.size()) {
            results[index] = sov.get<Pr(B, struct_a), Pr(A, m)>(index);
        }
    }
};

TEST_F(RS_StructOfVectorsTest, GetScalarField) {
    RS_StructOfVectors<B> sov(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }
    sov.write(input);

    RS_Vector<int> results(10, &provider);
    results.fill(0);

    provider.run<RSCF_SovGetScalarField>({10}, sov, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(host_results[i], i);
    }

    sov.free();
    results.free();
    host_results.free();
}

class RSCF_SovGetSetScalarField : public RS_ComputeFunction<RSCF_SovGetSetScalarField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfVectors<B> sov, int value) {
        RIndex index = index1D();
        if (index < sov.size()) {
            sov.get<Pr(B, struct_a), Pr(A, m)>(index) = value;
        }
    }
};

TEST_F(RS_StructOfVectorsTest, GetSetScalarField) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
    }
    sov.write(input);

    provider.run<RSCF_SovGetSetScalarField>({5}, sov, 42);

    B output[5];
    sov.read(output);
    EXPECT_EQ(output[2].struct_a.m, 42);
    sov.free();
}

class RSCF_SovGetArrayField : public RS_ComputeFunction<RSCF_SovGetArrayField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_StructOfVectors<B> sov, RS_Vector<int> results, int array_index
    ) {
        RIndex index = index1D();
        if (index < sov.size()) {
            results[index] = sov.get<Pr(B, array)>(index)[array_index];
        }
    }
};

TEST_F(RS_StructOfVectorsTest, GetArrayField) {
    RS_StructOfVectors<B> sov(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
    }
    sov.write(input);

    RS_Vector<int> results(10, &provider);
    results.fill(0);

    provider.run<RSCF_SovGetArrayField>({10}, sov, results, 5);

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(host_results[i], i * 10 + 5);
    }

    sov.free();
    results.free();
    host_results.free();
}

class RSCF_SovGetNestedStructureCopy : public RS_ComputeFunction<RSCF_SovGetNestedStructureCopy> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_StructOfVectors<B> sov, RS_Vector<Vector> results
    ) {
        RIndex index = index1D();
        if (index < sov.size()) {
            Vector v_copy = sov.get<Pr(B, struct_a), Pr(A, v)>(index);
            v_copy.x = -1;
            results[index] = v_copy;
        }
    }
};

TEST_F(RS_StructOfVectorsTest, GetNestedStructureCopy) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    sov.write(input);

    RS_Vector<Vector> results(5, &provider);
    results.occupy(5);

    provider.run<RSCF_SovGetNestedStructureCopy>({5}, sov, results);

    RS_Vector<Vector> host_results = results.move(&host_provider);
    B output[5];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(host_results[i].x, -1);
        EXPECT_EQ(output[i].struct_a.v.x, i);
    }

    sov.free();
    results.free();
    host_results.free();
}

class RSCF_SovGetNestedFieldReference : public RS_ComputeFunction<RSCF_SovGetNestedFieldReference> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_StructOfVectors<B> sov, RS_Vector<int> results
    ) {
        RIndex index = index1D();
        if (index < sov.size()) {
            results[index] = sov.get<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(index);
        }
    }
};

TEST_F(RS_StructOfVectorsTest, GetNestedFieldReference) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    sov.write(input);

    RS_Vector<int> results(5, &provider);
    results.fill(0);

    provider.run<RSCF_SovGetNestedFieldReference>({5}, sov, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(host_results[i], i);
    }

    sov.free();
    results.free();
    host_results.free();
}

class RSCF_SovGetSetNestedField : public RS_ComputeFunction<RSCF_SovGetSetNestedField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfVectors<B> sov, int value) {
        RIndex index = index1D();
        if (index < sov.size()) {
            sov.get<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(index) = value;
        }
    }
};

TEST_F(RS_StructOfVectorsTest, GetSetNestedField) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    sov.write(input);

    provider.run<RSCF_SovGetSetNestedField>({5}, sov, 42);

    B output[5];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.v.x, 42);
    }
    sov.free();
}

class RSCF_SovGetEntireElement : public RS_ComputeFunction<RSCF_SovGetEntireElement> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfVectors<B> sov, RS_Vector<B> results) {
        RIndex index = index1D();
        if (index < sov.size()) {
            B elem_copy = sov.get(index);
            elem_copy.struct_a.m = -1;
            results[index] = elem_copy;
        }
    }
};

TEST_F(RS_StructOfVectorsTest, GetEntireElement) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
    }
    sov.write(input);

    RS_Vector<B> results(5, &provider);
    results.occupy(5);

    provider.run<RSCF_SovGetEntireElement>({5}, sov, results);

    RS_Vector<B> host_results = results.move(&host_provider);
    B output[5];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(host_results[i].struct_a.m, -1);
        EXPECT_EQ(output[i].struct_a.m, i);
    }

    sov.free();
    results.free();
    host_results.free();
}

/*****************************************************************
 * Test cases for RS_StructOfVectors device set
 *
 * Test cases:
 * 1. Set a scalar field on the device.
 * 2. Set an array field from a device buffer.
 * 3. Set an array field from a local kernel array.
 * 4. Set a nested structure field on the device.
 * 5. Set a nested field on the device.
 * 6. Set an entire element on the device.
 *****************************************************************/

class RSCF_SovSetScalarField : public RS_ComputeFunction<RSCF_SovSetScalarField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfVectors<B> sov, int value) {
        RIndex index = index1D();
        if (index < sov.size()) {
            sov.set<Pr(B, struct_a), Pr(A, m)>(index, value);
        }
    }
};

TEST_F(RS_StructOfVectorsTest, SetScalarField) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
    }
    sov.write(input);

    provider.run<RSCF_SovSetScalarField>({5}, sov, 42);

    B output[5];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 42);
    }
    sov.free();
}

class RSCF_SovSetArrayField : public RS_ComputeFunction<RSCF_SovSetArrayField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfVectors<B> sov, RS_Vector<int> values) {
        RIndex index = index1D();
        if (index < sov.size()) {
            sov.set<Pr(B, array)>(index, values.ptr());
        }
    }
};

TEST_F(RS_StructOfVectorsTest, SetArrayField) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
    }
    sov.write(input);

    int host_values[10];
    for (int i = 0; i < 10; i++) {
        host_values[i] = 42 + i;
    }

    RS_Vector<int> values(10, &provider);
    values.write(host_values);

    provider.run<RSCF_SovSetArrayField>({5}, sov, values);

    B output[5];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);
        }
    }

    sov.free();
    values.free();
}

class RSCF_SovSetArrayFieldWithKernelArray
    : public RS_ComputeFunction<RSCF_SovSetArrayFieldWithKernelArray> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfVectors<B> sov) {
        RIndex index = index1D();
        if (index < sov.size()) {
            int array[10];
            for (int i = 0; i < 10; i++) {
                array[i] = 42 + i;
            }
            sov.set<Pr(B, array)>(index, array);
        }
    }
};

TEST_F(RS_StructOfVectorsTest, SetArrayFieldWithKernelArray) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
    }
    sov.write(input);

    provider.run<RSCF_SovSetArrayFieldWithKernelArray>({5}, sov);

    B output[5];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);
        }
    }
    sov.free();
}

class RSCF_SovSetNestedStructureField
    : public RS_ComputeFunction<RSCF_SovSetNestedStructureField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfVectors<B> sov, Vector value) {
        RIndex index = index1D();
        if (index < sov.size()) {
            sov.set<Pr(B, struct_a), Pr(A, v)>(index, value);
        }
    }
};

TEST_F(RS_StructOfVectorsTest, SetNestedStructureField) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    sov.write(input);

    Vector new_value = {42, 42.0f, 42.0};

    provider.run<RSCF_SovSetNestedStructureField>({5}, sov, new_value);

    B output[5];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.v.x, 42);
        EXPECT_EQ(output[i].struct_a.v.y, 42.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 42.0);
    }
    sov.free();
}

class RSCF_SovSetNestedField : public RS_ComputeFunction<RSCF_SovSetNestedField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfVectors<B> sov, int value) {
        RIndex index = index1D();
        if (index < sov.size()) {
            sov.set<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(index, value);
        }
    }
};

TEST_F(RS_StructOfVectorsTest, SetNestedField) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    sov.write(input);

    provider.run<RSCF_SovSetNestedField>({5}, sov, 42);

    B output[5];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.v.x, 42);
    }
    sov.free();
}

class RSCF_SovSetEntireElement : public RS_ComputeFunction<RSCF_SovSetEntireElement> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfVectors<B> sov, B value) {
        RIndex index = index1D();
        if (index < sov.size()) {
            sov.set(index, value);
        }
    }
};

TEST_F(RS_StructOfVectorsTest, SetEntireElement) {
    RS_StructOfVectors<B> sov(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
        input[i].n = 1;
    }
    sov.write(input);

    B new_value;
    new_value.struct_a.m = 42;
    new_value.struct_a.v = {42, 42.0f, 42.0};
    for (int i = 0; i < 10; i++) {
        new_value.array[i] = 42 + i;
    }
    new_value.n = 42;

    provider.run<RSCF_SovSetEntireElement>({5}, sov, new_value);

    B output[5];
    sov.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 42);
        EXPECT_EQ(output[i].struct_a.v.x, 42);
        EXPECT_EQ(output[i].struct_a.v.y, 42.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 42.0);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);
        }
        EXPECT_EQ(output[i].n, 42u);
    }
    sov.free();
}
