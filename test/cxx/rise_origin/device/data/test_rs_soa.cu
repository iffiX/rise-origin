#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_soa.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_StructOfArrays (SoA) is a container that turns an array of
 * structures layout into a structure of arrays layout for better
 * memory coalescing on the device.
 *
 * - Construction & Free
 *   Constructor with size, free(), repeated free()
 *
 * - Memory Management
 *   resize() with and without data copy
 *
 * - Host Data Operations
 *   fill(), fillSlice(), write(), writeSlice(), read(), readSlice(), copy()
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

class RS_StructOfArraysTest : public ::testing::Test {
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
 * Test cases for RS_StructOfArrays::free()
 *
 * Test cases:
 * 1. Free empty instance.
 * 2. Free non-empty instance.
 * 3. Multiple frees handled gracefully.
 * 4. Free after operations.
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
    EXPECT_NO_THROW(soa.free());
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
 * Test cases for RS_StructOfArrays::resize()
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

TEST_F(RS_StructOfArraysTest, ResizeZeroToZero) {
    RS_StructOfArrays<B> soa(0, &provider);
    soa.resize(0);
    EXPECT_EQ(soa.size(), 0);
    soa.free();
}

TEST_F(RS_StructOfArraysTest, ResizeZeroToPositive) {
    RS_StructOfArrays<B> soa(0, &provider);
    soa.resize(5);
    EXPECT_EQ(soa.size(), 5);
    soa.free();
}

TEST_F(RS_StructOfArraysTest, ResizeLargerWithoutCopy) {
    RS_StructOfArrays<B> soa(0, &provider);
    soa.resize(10, false);
    EXPECT_EQ(soa.size(), 10);
    soa.free();
}

TEST_F(RS_StructOfArraysTest, ResizeSmallerWithoutCopy) {
    RS_StructOfArrays<B> soa(10, &provider);
    soa.resize(5, false);
    EXPECT_EQ(soa.size(), 5);
    soa.free();
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
    soa.free();
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
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    soa.free();
}

TEST_F(RS_StructOfArraysTest, ResizeToZeroWithCopy) {
    RS_StructOfArrays<B> soa(10, &provider);
    soa.resize(0, true);
    EXPECT_EQ(soa.size(), 0);
    soa.free();
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
    soa.free();
}

/*****************************************************************
 * Test cases for RS_StructOfArrays::fill()
 *
 * Test cases:
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
    EXPECT_EQ(soa.size(), 0);
    soa.free();
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
    soa.free();
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

    B new_value;
    new_value.struct_a.m = 4;
    new_value.struct_a.v = {4, 8.0f, 12.0};
    for (int i = 0; i < 10; i++) {
        new_value.array[i] = i * 4;
    }
    new_value.n = 4;

    soa.fill(new_value);

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
    soa.free();
}

/*****************************************************************
 * Test cases for RS_StructOfArrays::fillSlice()
 *
 * Test cases:
 * 1. Fill slice at the beginning.
 * 2. Fill slice in the middle.
 * 3. Fill a single element slice.
 * 4. Fill zero elements is a no operation.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, FillSliceAtBeginning) {
    RS_StructOfArrays<B> soa(10, &provider);

    B initial_value;
    initial_value.struct_a.m = 0;
    initial_value.struct_a.v = {0, 0.0f, 0.0};
    for (int i = 0; i < 10; i++) {
        initial_value.array[i] = 0;
    }
    initial_value.n = 0;
    soa.fill(initial_value);

    B fill_value;
    fill_value.struct_a.m = 42;
    fill_value.struct_a.v = {42, 42.0f, 42.0};
    for (int i = 0; i < 10; i++) {
        fill_value.array[i] = 42 + i;
    }
    fill_value.n = 42;

    soa.fillSlice(fill_value, 3, 0);

    B output[10];
    soa.read(output);

    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(output[i].struct_a.m, 42);
        EXPECT_EQ(output[i].struct_a.v.x, 42);
        EXPECT_EQ(output[i].struct_a.v.y, 42.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 42.0);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);
        }
        EXPECT_EQ(output[i].n, 42);
    }

    for (int i = 3; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 0);
        EXPECT_EQ(output[i].struct_a.v.x, 0);
    }
    soa.free();
}

TEST_F(RS_StructOfArraysTest, FillSliceInMiddle) {
    RS_StructOfArrays<B> soa(10, &provider);

    B initial_value;
    initial_value.struct_a.m = 0;
    initial_value.struct_a.v = {0, 0.0f, 0.0};
    for (int i = 0; i < 10; i++) {
        initial_value.array[i] = 0;
    }
    initial_value.n = 0;
    soa.fill(initial_value);

    B fill_value;
    fill_value.struct_a.m = 99;
    fill_value.struct_a.v = {99, 99.0f, 99.0};
    for (int i = 0; i < 10; i++) {
        fill_value.array[i] = 99 + i;
    }
    fill_value.n = 99;

    soa.fillSlice(fill_value, 4, 3);

    B output[10];
    soa.read(output);

    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(output[i].struct_a.m, 0);
    }

    for (int i = 3; i < 7; i++) {
        EXPECT_EQ(output[i].struct_a.m, 99);
        EXPECT_EQ(output[i].struct_a.v.x, 99);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 99 + j);
        }
        EXPECT_EQ(output[i].n, 99);
    }

    for (int i = 7; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 0);
    }
    soa.free();
}

TEST_F(RS_StructOfArraysTest, FillSliceSingleElement) {
    RS_StructOfArrays<B> soa(10, &provider);

    B initial_value;
    initial_value.struct_a.m = 0;
    initial_value.struct_a.v = {0, 0.0f, 0.0};
    for (int i = 0; i < 10; i++) {
        initial_value.array[i] = 0;
    }
    initial_value.n = 0;
    soa.fill(initial_value);

    B fill_value;
    fill_value.struct_a.m = 123;
    fill_value.struct_a.v = {123, 123.0f, 123.0};
    for (int i = 0; i < 10; i++) {
        fill_value.array[i] = 123 + i;
    }
    fill_value.n = 123;

    soa.fillSlice(fill_value, 1, 5);

    B output[10];
    soa.read(output);

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
    soa.free();
}

TEST_F(RS_StructOfArraysTest, FillSliceZeroElements) {
    RS_StructOfArrays<B> soa(10, &provider);

    B initial_value;
    initial_value.struct_a.m = 1;
    initial_value.struct_a.v = {1, 1.0f, 1.0};
    for (int i = 0; i < 10; i++) {
        initial_value.array[i] = i;
    }
    initial_value.n = 1;
    soa.fill(initial_value);

    B fill_value;
    fill_value.struct_a.m = 999;
    fill_value.struct_a.v = {999, 999.0f, 999.0};
    for (int i = 0; i < 10; i++) {
        fill_value.array[i] = 999;
    }
    fill_value.n = 89;

    EXPECT_NO_THROW(soa.fillSlice(fill_value, 0, 5));

    B output[10];
    soa.read(output);

    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, 1);
        EXPECT_EQ(output[i].struct_a.v.x, 1);
    }
    soa.free();
}

/*****************************************************************
 * Test cases for RS_StructOfArrays::write()
 *
 * Test cases:
 * 1. Write more elements than the size throws.
 * 2. Write fewer elements than the size.
 * 3. Write the same number of elements as the size.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, WriteToEmptyInstance) {
    RS_StructOfArrays<B> soa(0, &provider);

    B input[5];
    EXPECT_THROW(soa.write(input, 5), RS_OutOfBoundsException);
    soa.free();
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
    soa.free();
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
    soa.free();
}

/*****************************************************************
 * Test cases for RS_StructOfArrays::writeSlice()
 *
 * Test cases:
 * 1. Valid slice write at an offset.
 * 2. Slice exceeds the size throws.
 * 3. Write at the beginning.
 * 4. Write zero elements.
 * 5. Offset beyond the size throws.
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
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i + 1].array[j], (i + 10) * 10 + j);
        }
        EXPECT_EQ(output[i + 1].n, 2);
    }

    for (int i = 0; i < 10; i++) {
        if (i < 1 || i >= 6) {
            EXPECT_EQ(output[i].struct_a.m, i);
            EXPECT_EQ(output[i].struct_a.v.x, i);
            for (int j = 0; j < 10; j++) {
                EXPECT_EQ(output[i].array[j], i * 10 + j);
            }
            EXPECT_EQ(output[i].n, 1);
        }
    }
    soa.free();
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
    soa.free();
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
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 2);
    }
    soa.free();
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

    soa.writeSlice(initial, 0, 0);

    B output[10];
    soa.read(output);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, i);
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    soa.free();
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
    soa.free();
}

/*****************************************************************
 * Test cases for RS_StructOfArrays::copy()
 *
 * Test cases:
 * 1. Copy same size.
 * 2. Copy different size resizes the destination.
 * 3. Copy from an empty instance.
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
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    source.free();
    destination.free();
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
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    source.free();
    destination.free();
}

TEST_F(RS_StructOfArraysTest, CopyFromEmptyInstance) {
    RS_StructOfArrays<B> source(0, &provider);

    RS_StructOfArrays<B> destination(10, &provider);
    destination.copy(source);

    EXPECT_EQ(destination.size(), 0);
    source.free();
    destination.free();
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
 * Test cases for RS_StructOfArrays::read()
 *
 * Test cases:
 * 1. Read from an empty instance leaves the output unchanged.
 * 2. Read data back.
 *****************************************************************/

TEST_F(RS_StructOfArraysTest, ReadFromEmptyInstance) {
    RS_StructOfArrays<B> soa(0, &provider);

    B output[1];
    output[0].struct_a.m = -1;

    EXPECT_NO_THROW(soa.read(output));
    EXPECT_EQ(output[0].struct_a.m, -1);
    soa.free();
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
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    soa.free();
}

/*****************************************************************
 * Test cases for RS_StructOfArrays::readSlice()
 *
 * Test cases:
 * 1. Valid slice read at an offset.
 * 2. Slice exceeds the size throws.
 * 3. Read from the beginning.
 * 4. Read zero elements leaves the output unchanged.
 * 5. Offset beyond the size throws.
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
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], (i + 2) * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    soa.free();
}

TEST_F(RS_StructOfArraysTest, ReadSliceExceedsSize) {
    RS_StructOfArrays<B> soa(10, &provider);

    B output[5];
    EXPECT_THROW(soa.readSlice(output, 5, 7), RS_OutOfBoundsException);
    soa.free();
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
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], i * 10 + j);
        }
        EXPECT_EQ(output[i].n, 1);
    }
    soa.free();
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
        output[i].n = 2;
    }
    soa.readSlice(output, 0, 5);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(output[i].struct_a.m, i + 1);
        EXPECT_EQ(output[i].n, 2);
    }
    soa.free();
}

TEST_F(RS_StructOfArraysTest, ReadSliceOffsetBeyondSize) {
    RS_StructOfArrays<B> soa(10, &provider);

    B output[5];
    EXPECT_THROW(soa.readSlice(output, 5, 11), RS_OutOfBoundsException);
    soa.free();
}

/*****************************************************************
 * Test cases for RS_StructOfArrays device get
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

class RSCF_SoaGetScalarField : public RS_ComputeFunction<RSCF_SoaGetScalarField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_StructOfArrays<B> soa, RS_Vector<int> results
    ) {
        RIndex index = index1D();
        if (index < soa.size()) {
            results[index] = soa.get<Pr(B, struct_a), Pr(A, m)>(index);
        }
    }
};

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

    RS_Vector<int> results(10, &provider);
    results.fill(0);

    provider.run<RSCF_SoaGetScalarField>({10}, soa, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(host_results[i], i);
    }

    soa.free();
    results.free();
    host_results.free();
}

class RSCF_SoaGetSetScalarField : public RS_ComputeFunction<RSCF_SoaGetSetScalarField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfArrays<B> soa, int value) {
        RIndex index = index1D();
        if (index < soa.size()) {
            soa.get<Pr(B, struct_a), Pr(A, m)>(index) = value;
        }
    }
};

TEST_F(RS_StructOfArraysTest, GetSetScalarField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
    }
    soa.write(input);

    provider.run<RSCF_SoaGetSetScalarField>({5}, soa, 42);

    B output[5];
    soa.read(output);
    EXPECT_EQ(output[2].struct_a.m, 42);
    soa.free();
}

class RSCF_SoaGetArrayField : public RS_ComputeFunction<RSCF_SoaGetArrayField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_StructOfArrays<B> soa, RS_Vector<int> results, int array_index
    ) {
        RIndex index = index1D();
        if (index < soa.size()) {
            results[index] = soa.get<Pr(B, array)>(index)[array_index];
        }
    }
};

TEST_F(RS_StructOfArraysTest, GetArrayField) {
    RS_StructOfArrays<B> soa(10, &provider);

    B input[10];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
    }
    soa.write(input);

    RS_Vector<int> results(10, &provider);
    results.fill(0);

    provider.run<RSCF_SoaGetArrayField>({10}, soa, results, 5);

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(host_results[i], i * 10 + 5);
    }

    soa.free();
    results.free();
    host_results.free();
}

class RSCF_SoaGetNestedStructureCopy : public RS_ComputeFunction<RSCF_SoaGetNestedStructureCopy> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_StructOfArrays<B> soa, RS_Vector<Vector> results
    ) {
        RIndex index = index1D();
        if (index < soa.size()) {
            Vector v_copy = soa.get<Pr(B, struct_a), Pr(A, v)>(index);
            v_copy.x = -1;
            results[index] = v_copy;
        }
    }
};

TEST_F(RS_StructOfArraysTest, GetNestedStructureCopy) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    soa.write(input);

    RS_Vector<Vector> results(5, &provider);
    results.occupy(5);

    provider.run<RSCF_SoaGetNestedStructureCopy>({5}, soa, results);

    RS_Vector<Vector> host_results = results.move(&host_provider);
    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(host_results[i].x, -1);
        EXPECT_EQ(output[i].struct_a.v.x, i);
    }

    soa.free();
    results.free();
    host_results.free();
}

class RSCF_SoaGetNestedFieldReference : public RS_ComputeFunction<RSCF_SoaGetNestedFieldReference> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_StructOfArrays<B> soa, RS_Vector<int> results
    ) {
        RIndex index = index1D();
        if (index < soa.size()) {
            results[index] = soa.get<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(index);
        }
    }
};

TEST_F(RS_StructOfArraysTest, GetNestedFieldReference) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    soa.write(input);

    RS_Vector<int> results(5, &provider);
    results.fill(0);

    provider.run<RSCF_SoaGetNestedFieldReference>({5}, soa, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(host_results[i], i);
    }

    soa.free();
    results.free();
    host_results.free();
}

class RSCF_SoaGetSetNestedField : public RS_ComputeFunction<RSCF_SoaGetSetNestedField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfArrays<B> soa, int value) {
        RIndex index = index1D();
        if (index < soa.size()) {
            soa.get<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(index) = value;
        }
    }
};

TEST_F(RS_StructOfArraysTest, GetSetNestedField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    soa.write(input);

    provider.run<RSCF_SoaGetSetNestedField>({5}, soa, 42);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.v.x, 42);
    }
    soa.free();
}

class RSCF_SoaGetEntireElement : public RS_ComputeFunction<RSCF_SoaGetEntireElement> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfArrays<B> soa, RS_Vector<B> results) {
        RIndex index = index1D();
        if (index < soa.size()) {
            B elem_copy = soa.get(index);
            elem_copy.struct_a.m = -1;
            results[index] = elem_copy;
        }
    }
};

TEST_F(RS_StructOfArraysTest, GetEntireElement) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
    }
    soa.write(input);

    RS_Vector<B> results(5, &provider);
    results.occupy(5);

    provider.run<RSCF_SoaGetEntireElement>({5}, soa, results);

    RS_Vector<B> host_results = results.move(&host_provider);
    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(host_results[i].struct_a.m, -1);
        EXPECT_EQ(output[i].struct_a.m, i);
    }

    soa.free();
    results.free();
    host_results.free();
}

/*****************************************************************
 * Test cases for RS_StructOfArrays device set
 *
 * Test cases:
 * 1. Set a scalar field on the device.
 * 2. Set an array field from a device buffer.
 * 3. Set an array field from a local kernel array.
 * 4. Set a nested structure field on the device.
 * 5. Set a nested field on the device.
 * 6. Set an entire element on the device.
 *****************************************************************/

class RSCF_SoaSetScalarField : public RS_ComputeFunction<RSCF_SoaSetScalarField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfArrays<B> soa, int value) {
        RIndex index = index1D();
        if (index < soa.size()) {
            soa.set<Pr(B, struct_a), Pr(A, m)>(index, value);
        }
    }
};

TEST_F(RS_StructOfArraysTest, SetScalarField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.m = i;
    }
    soa.write(input);

    provider.run<RSCF_SoaSetScalarField>({5}, soa, 42);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.m, 42);
    }
    soa.free();
}

class RSCF_SoaSetArrayField : public RS_ComputeFunction<RSCF_SoaSetArrayField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfArrays<B> soa, RS_Vector<int> values) {
        RIndex index = index1D();
        if (index < soa.size()) {
            soa.set<Pr(B, array)>(index, values.ptr());
        }
    }
};

TEST_F(RS_StructOfArraysTest, SetArrayField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
    }
    soa.write(input);

    int host_values[10];
    for (int i = 0; i < 10; i++) {
        host_values[i] = 42 + i;
    }

    RS_Vector<int> values(10, &provider);
    values.write(host_values);

    provider.run<RSCF_SoaSetArrayField>({5}, soa, values);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);
        }
    }

    soa.free();
    values.free();
}

class RSCF_SoaSetArrayFieldWithKernelArray
    : public RS_ComputeFunction<RSCF_SoaSetArrayFieldWithKernelArray> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfArrays<B> soa) {
        RIndex index = index1D();
        if (index < soa.size()) {
            int array[10];
            for (int i = 0; i < 10; i++) {
                array[i] = 42 + i;
            }
            soa.set<Pr(B, array)>(index, array);
        }
    }
};

TEST_F(RS_StructOfArraysTest, SetArrayFieldWithKernelArray) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            input[i].array[j] = i * 10 + j;
        }
    }
    soa.write(input);

    provider.run<RSCF_SoaSetArrayFieldWithKernelArray>({5}, soa);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(output[i].array[j], 42 + j);
        }
    }
    soa.free();
}

class RSCF_SoaSetNestedStructureField
    : public RS_ComputeFunction<RSCF_SoaSetNestedStructureField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfArrays<B> soa, Vector value) {
        RIndex index = index1D();
        if (index < soa.size()) {
            soa.set<Pr(B, struct_a), Pr(A, v)>(index, value);
        }
    }
};

TEST_F(RS_StructOfArraysTest, SetNestedStructureField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    soa.write(input);

    Vector new_value = {42, 42.0f, 42.0};

    provider.run<RSCF_SoaSetNestedStructureField>({5}, soa, new_value);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.v.x, 42);
        EXPECT_EQ(output[i].struct_a.v.y, 42.0f);
        EXPECT_EQ(output[i].struct_a.v.z, 42.0);
    }
    soa.free();
}

class RSCF_SoaSetNestedField : public RS_ComputeFunction<RSCF_SoaSetNestedField> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfArrays<B> soa, int value) {
        RIndex index = index1D();
        if (index < soa.size()) {
            soa.set<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(index, value);
        }
    }
};

TEST_F(RS_StructOfArraysTest, SetNestedField) {
    RS_StructOfArrays<B> soa(5, &provider);

    B input[5];
    for (int i = 0; i < 5; i++) {
        input[i].struct_a.v = {i, static_cast<float>(i), static_cast<double>(i)};
    }
    soa.write(input);

    provider.run<RSCF_SoaSetNestedField>({5}, soa, 42);

    B output[5];
    soa.read(output);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(output[i].struct_a.v.x, 42);
    }
    soa.free();
}

class RSCF_SoaSetEntireElement : public RS_ComputeFunction<RSCF_SoaSetEntireElement> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StructOfArrays<B> soa, B value) {
        RIndex index = index1D();
        if (index < soa.size()) {
            soa.set(index, value);
        }
    }
};

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

    B new_value;
    new_value.struct_a.m = 42;
    new_value.struct_a.v = {42, 42.0f, 42.0};
    for (int i = 0; i < 10; i++) {
        new_value.array[i] = 42 + i;
    }
    new_value.n = 42;

    provider.run<RSCF_SoaSetEntireElement>({5}, soa, new_value);

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
        EXPECT_EQ(output[i].n, 42);
    }
    soa.free();
}
