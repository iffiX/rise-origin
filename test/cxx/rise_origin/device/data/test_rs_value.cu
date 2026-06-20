#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_value.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_Value is a template class that wraps a single value stored in
 * device memory (CPU or GPU or ...). It provides a high level interface
 * for managing and manipulating single values across different memory
 * providers.
 *
 * - Constructors
 *   Default, with provider, with value and provider
 *
 * - Initialization
 *   init(provider), init(value, provider)
 *
 * - Memory Management
 *   free()
 *
 * - Copy/Move Operations
 *   copy(), move()
 *
 * - Value Operations
 *   fill(), write(), read()
 *
 * - Device Operations
 *   get(), set(), set<T2>()
 *
 * - Accessors
 *   provider(), device(), baseBytes(), extraBytes()
 *****************************************************************/

class RS_ValueTest : public ::testing::Test {
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

struct Point {
    int x, y;
    HOST_DEVICE_INLINE_QUALIFIER Point() : x(0), y(0) {}
    HOST_DEVICE_INLINE_QUALIFIER Point(int a, int b) : x(a), y(b) {}
};

struct LargeStruct {
    int data[100];
};

struct Color {
    unsigned char r, g, b, a;
    HOST_DEVICE_INLINE_QUALIFIER Color() : r(0), g(0), b(0), a(0) {}
    HOST_DEVICE_INLINE_QUALIFIER Color(
        unsigned char red,
        unsigned char green,
        unsigned char blue,
        unsigned char alpha
    )
        : r(red), g(green), b(blue), a(alpha) {}
};

/*****************************************************************
 * Test cases for RS_Value constructors
 *
 * Test cases:
 * 1. Default constructor
 * 2. Constructor with provider
 * 3. Constructor with value and provider
 *****************************************************************/

TEST_F(RS_ValueTest, DefaultConstructor) {
    RS_Value<int> value;
    // Default constructed value should have null provider
    EXPECT_EQ(value.provider(), nullptr);
}

TEST_F(RS_ValueTest, ConstructorWithProvider) {
    RS_Value<int> value(&host_provider);
    EXPECT_NE(value.provider(), nullptr);
    EXPECT_EQ(value.provider(), &host_provider);
    value.free();
}

TEST_F(RS_ValueTest, ConstructorWithValueAndProvider) {
    RS_Value<int> value(42, &host_provider);
    EXPECT_NE(value.provider(), nullptr);
    EXPECT_EQ(value.provider(), &host_provider);

    int result = 0;
    value.read(&result);
    EXPECT_EQ(result, 42);

    value.free();
}

/*****************************************************************
 * Test cases for RS_Value::init()
 *
 * Test cases:
 * 1. Initialize with provider (default value)
 * 2. Initialize with value and provider
 * 3. Re-initialize already initialized value
 *****************************************************************/

TEST_F(RS_ValueTest, InitWithProvider) {
    RS_Value<int> value;
    value.init(&host_provider);

    EXPECT_NE(value.provider(), nullptr);
    EXPECT_EQ(value.provider(), &host_provider);

    value.free();
}

TEST_F(RS_ValueTest, InitWithValueAndProvider) {
    RS_Value<float> value;
    value.init(3.14f, &host_provider);

    EXPECT_NE(value.provider(), nullptr);

    float result = 0.0f;
    value.read(&result);
    EXPECT_FLOAT_EQ(result, 3.14f);

    value.free();
}

TEST_F(RS_ValueTest, ReinitializeValue) {
    RS_Value<int> value(10, &host_provider);

    // Free and re-initialize
    value.free();
    value.init(20, &host_provider);

    int result = 0;
    value.read(&result);
    EXPECT_EQ(result, 20);

    value.free();
}

/*****************************************************************
 * Test cases for RS_Value::free()
 *
 * Test cases:
 * 1. Free initialized value
 * 2. Free uninitialized value
 * 3. Multiple frees
 *****************************************************************/

TEST_F(RS_ValueTest, FreeInitializedValue) {
    RS_Value<int> value(42, &host_provider);
    EXPECT_NO_THROW(value.free());
    EXPECT_EQ(value.provider(), nullptr);
}

TEST_F(RS_ValueTest, FreeUninitializedValue) {
    RS_Value<int> value;
    EXPECT_NO_THROW(value.free());
}

TEST_F(RS_ValueTest, MultipleFrees) {
    RS_Value<int> value(42, &host_provider);
    value.free();
    EXPECT_NO_THROW(value.free());
}

/*****************************************************************
 * Test cases for RS_Value::copy()
 *
 * Test cases:
 * 1. Copy between same provider type
 * 2. Copy complex type
 *****************************************************************/

TEST_F(RS_ValueTest, CopyBetweenSameProviderType) {
    RS_Value<int> value1(100, &host_provider);
    RS_Value<int> value2(&host_provider);

    value2.copy(value1);

    int result = 0;
    value2.read(&result);
    EXPECT_EQ(result, 100);

    value1.free();
    value2.free();
}

TEST_F(RS_ValueTest, CopyComplexType) {
    Point input(42, 300);
    RS_Value<Point> value1(input, &host_provider);
    RS_Value<Point> value2(&host_provider);

    value2.copy(value1);

    Point result;
    value2.read(&result);
    EXPECT_EQ(result.x, 42);
    EXPECT_EQ(result.y, 300);

    value1.free();
    value2.free();
}

/*****************************************************************
 * Test cases for RS_Value::move()
 *
 * Test cases:
 * 1. Move to different provider
 * 2. Move preserves value
 *****************************************************************/

TEST_F(RS_ValueTest, MoveToDifferentProvider) {
    RS_Value<int> value1(200, &host_provider);
    RS_Value<int> value2 = value1.move(&provider);

    EXPECT_NE(value2.provider(), nullptr);
    EXPECT_EQ(value2.provider(), &provider);

    value1.free();
    value2.free();
}

TEST_F(RS_ValueTest, MovePreservesValue) {
    RS_Value<double> value1(2.71828, &host_provider);
    RS_Value<double> value2 = value1.move(&host_provider);

    double result = 0.0;
    value2.read(&result);
    EXPECT_DOUBLE_EQ(result, 2.71828);

    value1.free();
    value2.free();
}

/*****************************************************************
 * Test cases for RS_Value::fill()
 *
 * Test cases:
 * 1. Fill with different value
 * 2. Fill synchronously
 * 3. Fill with complex type
 *****************************************************************/

TEST_F(RS_ValueTest, FillWithDifferentValue) {
    RS_Value<int> value(10, &host_provider);
    value.fill(99);

    int result = 0;
    value.read(&result);
    EXPECT_EQ(result, 99);

    value.free();
}

TEST_F(RS_ValueTest, FillSynchronously) {
    RS_Value<float> value(&host_provider);
    value.fill(1.5f, true);

    float result = 0.0f;
    value.read(&result);
    EXPECT_FLOAT_EQ(result, 1.5f);

    value.free();
}

TEST_F(RS_ValueTest, FillComplexType) {
    RS_Value<Point> value(&host_provider);
    Point p(10, 20);
    value.fill(p);

    Point result;
    value.read(&result);
    EXPECT_EQ(result.x, 10);
    EXPECT_EQ(result.y, 20);

    value.free();
}

/*****************************************************************
 * Test cases for RS_Value::write()
 *
 * Test cases:
 * 1. Write from pointer
 * 2. Write synchronously
 * 3. Write multiple times
 *****************************************************************/

TEST_F(RS_ValueTest, WriteFromPointer) {
    RS_Value<int> value(&host_provider);
    int input = 777;
    value.write(&input);

    int result = 0;
    value.read(&result);
    EXPECT_EQ(result, 777);

    value.free();
}

TEST_F(RS_ValueTest, WriteSynchronously) {
    RS_Value<double> value(&host_provider);
    double input = 3.14159;
    value.write(&input, true);

    double result = 0.0;
    value.read(&result);
    EXPECT_DOUBLE_EQ(result, 3.14159);

    value.free();
}

TEST_F(RS_ValueTest, WriteMultipleTimes) {
    RS_Value<int> value(&host_provider);

    int input1 = 100;
    value.write(&input1);

    int result1 = 0;
    value.read(&result1);
    EXPECT_EQ(result1, 100);

    int input2 = 200;
    value.write(&input2);

    int result2 = 0;
    value.read(&result2);
    EXPECT_EQ(result2, 200);

    value.free();
}

/*****************************************************************
 * Test cases for RS_Value::read()
 *
 * Test cases:
 * 1. Read to pointer
 * 2. Read synchronously
 * 3. Read after multiple writes
 *****************************************************************/

TEST_F(RS_ValueTest, ReadToPointer) {
    RS_Value<int> value(555, &host_provider);

    int result = 0;
    value.read(&result);
    EXPECT_EQ(result, 555);

    value.free();
}

TEST_F(RS_ValueTest, ReadSynchronously) {
    RS_Value<float> value(2.5f, &host_provider);

    float result = 0.0f;
    value.read(&result, true);
    EXPECT_FLOAT_EQ(result, 2.5f);

    value.free();
}

TEST_F(RS_ValueTest, ReadAfterMultipleWrites) {
    RS_Value<int> value(&host_provider);

    for (int i = 0; i < 10; ++i) {
        int input = i * 10;
        value.write(&input);

        int result = 0;
        value.read(&result);
        EXPECT_EQ(result, i * 10);
    }

    value.free();
}

/*****************************************************************
 * Test cases for RS_Value accessors
 *
 * Test cases:
 * 1. provider()
 * 2. device()
 * 3. baseBytes()
 * 4. extraBytes()
 *****************************************************************/

TEST_F(RS_ValueTest, ProviderAccessor) {
    RS_Value<int> value(&host_provider);
    EXPECT_EQ(value.provider(), &host_provider);
    value.free();
}

TEST_F(RS_ValueTest, DeviceAccessor) {
    RS_Value<int> value(&host_provider);
    RS_DeviceLocation device_loc = value.device();
    EXPECT_EQ(device_loc.type, RSE_DEVICE_TYPE_CPU);
    value.free();
}

TEST_F(RS_ValueTest, BaseBytesAccessor) {
    RS_Value<int> value(&host_provider);
    EXPECT_EQ(value.baseBytes(), sizeof(RS_Value<int>));
    value.free();
}

TEST_F(RS_ValueTest, ExtraBytesAccessor) {
    RS_Value<int> value(&host_provider);
    EXPECT_EQ(value.extraBytes(), sizeof(int));
    value.free();
}

TEST_F(RS_ValueTest, ExtraBytesForDifferentTypes) {
    RS_Value<double> value_double(&host_provider);
    EXPECT_EQ(value_double.extraBytes(), sizeof(double));
    value_double.free();

    RS_Value<char> value_char(&host_provider);
    EXPECT_EQ(value_char.extraBytes(), sizeof(char));
    value_char.free();

    RS_Value<LargeStruct> value_struct(&host_provider);
    EXPECT_EQ(value_struct.extraBytes(), sizeof(LargeStruct));
    value_struct.free();
}

/*****************************************************************
 * Test cases for different data types
 *
 * Test cases:
 * 1. Integer types
 * 2. Floating point types
 * 3. Custom struct types
 *****************************************************************/

TEST_F(RS_ValueTest, IntegerTypes) {
    RS_Value<int> value_int(42, &host_provider);
    int result_int = 0;
    value_int.read(&result_int);
    EXPECT_EQ(result_int, 42);
    value_int.free();

    RS_Value<long> value_long(123456789L, &host_provider);
    long result_long = 0;
    value_long.read(&result_long);
    EXPECT_EQ(result_long, 123456789L);
    value_long.free();

    RS_Value<short> value_short(999, &host_provider);
    short result_short = 0;
    value_short.read(&result_short);
    EXPECT_EQ(result_short, 999);
    value_short.free();
}

TEST_F(RS_ValueTest, FloatingPointTypes) {
    RS_Value<float> value_float(3.14f, &host_provider);
    float result_float = 0.0f;
    value_float.read(&result_float);
    EXPECT_FLOAT_EQ(result_float, 3.14f);
    value_float.free();

    RS_Value<double> value_double(2.71828, &host_provider);
    double result_double = 0.0;
    value_double.read(&result_double);
    EXPECT_DOUBLE_EQ(result_double, 2.71828);
    value_double.free();
}

TEST_F(RS_ValueTest, CustomStructTypes) {
    Color input_color(255, 128, 64, 255);
    RS_Value<Color> value(input_color, &host_provider);

    Color result_color;
    value.read(&result_color);
    EXPECT_EQ(result_color.r, 255);
    EXPECT_EQ(result_color.g, 128);
    EXPECT_EQ(result_color.b, 64);
    EXPECT_EQ(result_color.a, 255);

    value.free();
}

/*****************************************************************
 * Test cases for edge cases
 *
 * Test cases:
 * 1. Large values
 * 2. Zero values
 * 3. Negative values
 *****************************************************************/

TEST_F(RS_ValueTest, LargeValues) {
    RS_Value<long long> value(9223372036854775807LL, &host_provider);
    long long result = 0;
    value.read(&result);
    EXPECT_EQ(result, 9223372036854775807LL);
    value.free();
}

TEST_F(RS_ValueTest, ZeroValues) {
    RS_Value<int> value(0, &host_provider);
    int result = 999;
    value.read(&result);
    EXPECT_EQ(result, 0);
    value.free();
}

TEST_F(RS_ValueTest, NegativeValues) {
    RS_Value<int> value(-12345, &host_provider);
    int result = 0;
    value.read(&result);
    EXPECT_EQ(result, -12345);
    value.free();

    RS_Value<double> value_double(-3.14159, &host_provider);
    double result_double = 0.0;
    value_double.read(&result_double);
    EXPECT_DOUBLE_EQ(result_double, -3.14159);
    value_double.free();
}

/*****************************************************************
 * Test cases for the device provider
 *
 * Test cases:
 * 1. Initialize with device provider
 * 2. Copy between CPU and device providers
 * 3. Move from CPU to device provider
 *****************************************************************/

TEST_F(RS_ValueTest, InitializeWithDeviceProvider) {
    RS_Value<int> value(42, &provider);
    EXPECT_NE(value.provider(), nullptr);
    EXPECT_EQ(value.provider(), &provider);

    int result = 0;
    value.read(&result);
    EXPECT_EQ(result, 42);

    value.free();
}

TEST_F(RS_ValueTest, CopyBetweenCPUAndDeviceProviders) {
    RS_Value<int> cpu_value(100, &host_provider);
    RS_Value<int> device_value(&provider);

    device_value.copy(cpu_value);

    int result = 0;
    device_value.read(&result);
    EXPECT_EQ(result, 100);

    cpu_value.free();
    device_value.free();
}

TEST_F(RS_ValueTest, MoveFromCPUToDeviceProvider) {
    RS_Value<float> cpu_value(3.14f, &host_provider);
    RS_Value<float> device_value = cpu_value.move(&provider);

    EXPECT_EQ(device_value.provider(), &provider);

    float result = 0.0f;
    device_value.read(&result);
    EXPECT_FLOAT_EQ(result, 3.14f);

    cpu_value.free();
    device_value.free();
}

/*****************************************************************
 * Test cases for multiple values
 *
 * Test cases:
 * 1. Multiple independent values
 * 2. Array of values
 *****************************************************************/

TEST_F(RS_ValueTest, MultipleIndependentValues) {
    RS_Value<int> value1(10, &host_provider);
    RS_Value<int> value2(20, &host_provider);
    RS_Value<int> value3(30, &host_provider);

    int result1 = 0, result2 = 0, result3 = 0;
    value1.read(&result1);
    value2.read(&result2);
    value3.read(&result3);

    EXPECT_EQ(result1, 10);
    EXPECT_EQ(result2, 20);
    EXPECT_EQ(result3, 30);

    value1.free();
    value2.free();
    value3.free();
}

TEST_F(RS_ValueTest, ArrayOfValues) {
    const int N = 5;
    RS_Value<int> values[N];

    for (int i = 0; i < N; ++i) {
        values[i].init(i * 100, &host_provider);
    }

    for (int i = 0; i < N; ++i) {
        int result = 0;
        values[i].read(&result);
        EXPECT_EQ(result, i * 100);
    }

    for (int i = 0; i < N; ++i) {
        values[i].free();
    }
}

/*****************************************************************
 * Test cases for RS_Value device side access
 *
 * Test cases:
 * 1. get() and set() on the device
 *****************************************************************/

class RSCF_ValueDeviceAccess : public RS_ComputeFunction<RSCF_ValueDeviceAccess> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_Value<int> value, RS_Value<int> result) {
        if (index1D() == 0) {
            int read_back = value.get();
            value.set(read_back + 1);
            result.set(value.get());
        }
    }
};

TEST_F(RS_ValueTest, DeviceAccess) {
    RS_Value<int> value(41, &provider);
    RS_Value<int> result(0, &provider);

    provider.run<RSCF_ValueDeviceAccess>({1}, value, result);

    RS_Value<int> host_result = result.move(&host_provider);
    int read_back = 0;
    host_result.read(&read_back);
    EXPECT_EQ(read_back, 42);

    value.free();
    result.free();
    host_result.free();
}
