#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "rise_origin/utils/serialization/rs_mp_serialization.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_MessagePackSerializer and RS_MessagePackDeserializer provide MessagePack
 * serialization for std based types through a single serialize() visitor.
 *
 * - Public Interface
 *   serializer.serialize(value), serializer.getData(), serializer.moveData()
 *   deserializer.setData(bytes), deserializer.deserialize(value)
 *
 * - Primitive Fields
 *   int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
 *   bool, float, double
 *
 * - String Fields
 *   std::string
 *
 * - Container Fields
 *   std::vector, std::list, std::array, std::map, std::unordered_map
 *
 * - Enum Fields
 *   Serialized as the underlying integer type
 *
 * - Nested Structures
 *   Structs whose fields are themselves serializable structs
 *
 * - Error Handling
 *   Missing required key, optional missing key, non map root
 *****************************************************************/

/*****************************************************************
 * Helper serializable types
 *****************************************************************/

struct Primitives {
    int8_t i8 = 0;
    int16_t i16 = 0;
    int32_t i32 = 0;
    int64_t i64 = 0;
    uint8_t u8 = 0;
    uint16_t u16 = 0;
    uint32_t u32 = 0;
    uint64_t u64 = 0;
    bool flag = false;
    float f = 0.0f;
    double d = 0.0;

    bool operator==(const Primitives &o) const {
        return i8 == o.i8 && i16 == o.i16 && i32 == o.i32 && i64 == o.i64 && u8 == o.u8 &&
               u16 == o.u16 && u32 == o.u32 && u64 == o.u64 && flag == o.flag && f == o.f &&
               d == o.d;
    }

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("i8", i8);
        s("i16", i16);
        s("i32", i32);
        s("i64", i64);
        s("u8", u8);
        s("u16", u16);
        s("u32", u32);
        s("u64", u64);
        s("flag", flag);
        s("f", f);
        s("d", d);
    }
};

enum class Color : uint8_t { Red = 1, Green = 2, Blue = 200 };

struct Containers {
    std::string name;
    std::vector<int32_t> numbers;
    std::list<int32_t> linked;
    std::array<int32_t, 3> fixed = {0, 0, 0};
    std::map<std::string, int32_t> ordered;
    std::unordered_map<std::string, int32_t> hashed;
    Color color = Color::Red;

    bool operator==(const Containers &o) const {
        return name == o.name && numbers == o.numbers && linked == o.linked && fixed == o.fixed &&
               ordered == o.ordered && hashed == o.hashed && color == o.color;
    }

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("name", name);
        s("numbers", numbers);
        s("linked", linked);
        s("fixed", fixed);
        s("ordered", ordered);
        s("hashed", hashed);
        s("color", color);
    }
};

struct Inner {
    int32_t id = 0;
    std::string label;

    bool operator==(const Inner &o) const { return id == o.id && label == o.label; }

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("id", id);
        s("label", label);
    }
};

struct Outer {
    Inner inner;
    std::vector<Inner> children;

    bool operator==(const Outer &o) const {
        return inner == o.inner && children == o.children;
    }

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("inner", inner);
        s("children", children);
    }
};

/*****************************************************************
 * Roundtrip helper
 *****************************************************************/

template <typename T>
static T roundtrip(const T &input) {
    RS_MessagePackSerializer serializer;
    serializer.serialize(input);
    std::vector<uint8_t> bytes = serializer.moveData();

    RS_MessagePackDeserializer deserializer;
    deserializer.setData(bytes);
    T output{};
    deserializer.deserialize(output);
    return output;
}

/*****************************************************************
 * Test cases for the public interface
 *
 * Test cases:
 * 1. getData returns the serialized bytes.
 * 2. moveData transfers the serialized bytes and clears the serializer.
 *****************************************************************/

TEST(RS_MessagePackSerializationTest, GetDataReturnsBytes) {
    Inner value{7, "seven"};
    RS_MessagePackSerializer serializer;
    serializer.serialize(value);

    const std::vector<uint8_t> &bytes = serializer.getData();
    EXPECT_FALSE(bytes.empty());
}

TEST(RS_MessagePackSerializationTest, MoveDataTransfersBytes) {
    Inner value{7, "seven"};
    RS_MessagePackSerializer serializer;
    serializer.serialize(value);

    std::vector<uint8_t> moved = serializer.moveData();
    EXPECT_FALSE(moved.empty());
    EXPECT_TRUE(serializer.getData().empty());
}

/*****************************************************************
 * Test cases for primitive fields
 *
 * Test cases:
 * 1. Positive and negative integers roundtrip.
 * 2. Unsigned integer limits roundtrip.
 * 3. Bool, float, and double roundtrip.
 *****************************************************************/

TEST(RS_MessagePackSerializationTest, PrimitivesRoundtrip) {
    Primitives value;
    value.i8 = -42;
    value.i16 = -1000;
    value.i32 = -123456;
    value.i64 = -9876543210LL;
    value.u8 = 255;
    value.u16 = 65535;
    value.u32 = 4000000000u;
    value.u64 = 18000000000000000000ull;
    value.flag = true;
    value.f = 3.14159f;
    value.d = 2.718281828459045;

    Primitives result = roundtrip(value);
    EXPECT_EQ(value, result);
}

TEST(RS_MessagePackSerializationTest, PrimitivesZeroRoundtrip) {
    Primitives value;
    Primitives result = roundtrip(value);
    EXPECT_EQ(value, result);
}

/*****************************************************************
 * Test cases for container and enum fields
 *
 * Test cases:
 * 1. Vector, list, array, map, and unordered map roundtrip.
 * 2. Empty containers roundtrip.
 * 3. Enum roundtrips through the underlying integer.
 *****************************************************************/

TEST(RS_MessagePackSerializationTest, ContainersRoundtrip) {
    Containers value;
    value.name = "hello world";
    value.numbers = {1, 2, 3, 4, 5};
    value.linked = {10, 20, 30};
    value.fixed = {7, 8, 9};
    value.ordered = {{"a", 1}, {"b", 2}};
    value.hashed = {{"x", 100}, {"y", 200}};
    value.color = Color::Blue;

    Containers result = roundtrip(value);
    EXPECT_EQ(value, result);
}

TEST(RS_MessagePackSerializationTest, EmptyContainersRoundtrip) {
    Containers value;
    Containers result = roundtrip(value);
    EXPECT_EQ(value, result);
}

/*****************************************************************
 * Test cases for nested structures
 *
 * Test cases:
 * 1. A struct field that is itself serializable roundtrips.
 * 2. A vector of serializable structs roundtrips.
 *****************************************************************/

TEST(RS_MessagePackSerializationTest, NestedRoundtrip) {
    Outer value;
    value.inner = {1, "root"};
    value.children = {{2, "child_a"}, {3, "child_b"}};

    Outer result = roundtrip(value);
    EXPECT_EQ(value, result);
}

/*****************************************************************
 * Test cases for error handling
 *
 * Test cases:
 * 1. A missing required key throws.
 * 2. A missing optional key is ignored.
 * 3. A non map root throws.
 *****************************************************************/

struct OneField {
    int32_t a = 1;
    template <typename Serializer>
    void serialize(Serializer &s) {
        s("a", a);
    }
};

struct RequiresTwoFields {
    int32_t a = 0;
    int32_t b = 0;
    template <typename Serializer>
    void serialize(Serializer &s) {
        s("a", a);
        s("b", b);
    }
};

struct OptionalSecondField {
    int32_t a = 0;
    int32_t b = -1;
    template <typename Serializer>
    void serialize(Serializer &s) {
        s("a", a);
        s("b", b, true);
    }
};

TEST(RS_MessagePackSerializationTest, MissingRequiredKeyThrows) {
    OneField value{5};
    RS_MessagePackSerializer serializer;
    serializer.serialize(value);

    RS_MessagePackDeserializer deserializer;
    deserializer.setData(serializer.moveData());
    RequiresTwoFields output;
    EXPECT_THROW(deserializer.deserialize(output), RS_MessagePackSerializationException);
}

TEST(RS_MessagePackSerializationTest, MissingOptionalKeyIsIgnored) {
    OneField value{5};
    RS_MessagePackSerializer serializer;
    serializer.serialize(value);

    RS_MessagePackDeserializer deserializer;
    deserializer.setData(serializer.moveData());
    OptionalSecondField output;
    EXPECT_NO_THROW(deserializer.deserialize(output));
    EXPECT_EQ(output.a, 5);
    EXPECT_EQ(output.b, -1);
}

TEST(RS_MessagePackSerializationTest, NonMapRootThrows) {
    // A single positive fixint is a valid MessagePack value but not a map.
    std::vector<uint8_t> bytes = {0x01};
    RS_MessagePackDeserializer deserializer;
    deserializer.setData(bytes);
    OneField output;
    EXPECT_THROW(deserializer.deserialize(output), RS_MessagePackSerializationException);
}
