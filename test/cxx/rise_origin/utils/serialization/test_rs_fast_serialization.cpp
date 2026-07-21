#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "rise_origin/utils/serialization/rs_fast_serialization.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_FastSerializer and RS_FastDeserializer provide schema-bound binary
 * serialization for std based types through a single serialize() visitor.
 * The format carries no type tags or field names, so pack and unpack must
 * visit fields in the same order.
 *
 * - Public Interface
 *   serializer.serialize(value), serializer.getData()
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
 * - Zero-copy Blob
 *   RS_FastSerializationBufferView deserializes to a view into the source buffer
 *****************************************************************/

/*****************************************************************
 * Helper serializable types
 *****************************************************************/

enum class Color : uint8_t { Red, Green, Blue };

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

    bool operator==(const Outer &o) const { return inner == o.inner && children == o.children; }

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
    RS_FastSerializer serializer;
    serializer.serialize(input);

    RS_FastDeserializer deserializer;
    deserializer.setData(serializer.getData());
    T output{};
    deserializer.deserialize(output);
    return output;
}

/*****************************************************************
 * Test cases for the public interface
 *****************************************************************/

TEST(RS_FastSerializationTest, GetDataReturnsBytes) {
    Inner value{7, "seven"};
    RS_FastSerializer serializer;
    serializer.serialize(value);

    auto bytes = serializer.getData();
    EXPECT_FALSE(bytes.empty());
}

TEST(RS_FastSerializationTest, PrimitivesRoundtrip) {
    Primitives value{-1, -2, -3, -4, 1, 2, 3, 4, true, 1.5f, 2.5};
    EXPECT_EQ(roundtrip(value), value);
}

TEST(RS_FastSerializationTest, PrimitivesZeroRoundtrip) {
    Primitives value{};
    EXPECT_EQ(roundtrip(value), value);
}

TEST(RS_FastSerializationTest, ContainersRoundtrip) {
    Containers value;
    value.name = "container";
    value.numbers = {1, 2, 3};
    value.linked = {4, 5, 6};
    value.fixed = {7, 8, 9};
    value.ordered = {{"a", 1}, {"b", 2}};
    value.hashed = {{"x", 10}, {"y", 20}};
    value.color = Color::Blue;
    EXPECT_EQ(roundtrip(value), value);
}

TEST(RS_FastSerializationTest, EmptyContainersRoundtrip) {
    Containers value;
    EXPECT_EQ(roundtrip(value), value);
}

TEST(RS_FastSerializationTest, NestedRoundtrip) {
    Outer value;
    value.inner = {1, "root"};
    value.children = {{2, "a"}, {3, "b"}};
    EXPECT_EQ(roundtrip(value), value);
}

TEST(RS_FastSerializationTest, BufferViewZeroCopy) {
    std::vector<RByte> payload = {1, 2, 3, 4, 5};
    RS_FastSerializer serializer;
    RS_FastSerializationBufferView in_view(payload.data(), RSize(payload.size()));
    serializer.serialize(in_view);

    RS_FastDeserializer deserializer;
    auto buffer = serializer.getData();
    deserializer.setData(buffer);
    RS_FastSerializationBufferView out_view;
    deserializer.deserialize(out_view);

    // Content matches the original payload.
    ASSERT_EQ(out_view.size(), payload.size());
    for (RSize i = 0; i < out_view.size(); ++i) {
        EXPECT_EQ(out_view[i], payload[i]);
    }

    // Zero-copy: the deserialized view points inside the source buffer.
    EXPECT_GE(out_view.ptr(), buffer.ptr());
    EXPECT_LE(out_view.ptr() + out_view.size(), buffer.ptr() + buffer.size());
}
