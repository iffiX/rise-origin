// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

// Adapted from https://github.com/mikeloomisgg/cppack

#ifndef RS_MP_SERIALIZATION_H
#define RS_MP_SERIALIZATION_H

#include <fmt/format.h>

#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "rise_origin/utils/rs_exception.h"

namespace rise {

RS_MAKE_EXCEPTION(RS_MessagePackSerializationException, RS_DataException)

enum RS_MessagePackFormatConstants : uint8_t {
    // ext and fixext are not supported since they are not necessary
    // During deserialization, conversion between signed integers,
    // conversion between unsigned integers and conversion
    // between floats are supported.
    // However, the storage can only be casted upwards, (eg: float -> double)
    // Also inter-type conversions are forbidden.

    // positive fixint = 0x00 - 0x7f
    // fixmap = 0x80 - 0x8f
    // fixarray = 0x90 - 0x9a
    // fixstr = 0xa0 - 0xbf
    // negative fixint = 0xe0 - 0xff

    MP_NIL = 0xc0,
    MP_FALSE = 0xc2,
    MP_TRUE = 0xc3,
    MP_BIN8 = 0xc4,
    MP_BIN16 = 0xc5,
    MP_BIN32 = 0xc6,

    //    ext8 = 0xc7,
    //    ext16 = 0xc8,
    //    ext32 = 0xc9,

    MP_FLOAT32 = 0xca,
    MP_FLOAT64 = 0xcb,
    MP_UINT8 = 0xcc,
    MP_UINT16 = 0xcd,
    MP_UINT32 = 0xce,
    MP_UINT64 = 0xcf,
    MP_INT8 = 0xd0,
    MP_INT16 = 0xd1,
    MP_INT32 = 0xd2,
    MP_INT64 = 0xd3,

    //    fixext1 = 0xd4,
    //    fixext2 = 0xd5,
    //    fixext4 = 0xd6,
    //    fixext8 = 0xd7,
    //    fixext16 = 0xd8,

    MP_STR8 = 0xd9,
    MP_STR16 = 0xda,
    MP_STR32 = 0xdb,
    MP_ARRAY16 = 0xdc,
    MP_ARRAY32 = 0xdd,
    MP_MAP16 = 0xde,
    MP_MAP32 = 0xdf
};

namespace mp_helper {
template <typename T>
struct is_container {
    static const bool value = false;
};

template <typename T, typename Alloc>
struct is_container<std::vector<T, Alloc>> {
    static const bool value = true;
};

template <typename T, typename Alloc>
struct is_container<std::list<T, Alloc>> {
    static const bool value = true;
};

template <typename T, typename Alloc>
struct is_container<std::map<T, Alloc>> {
    static const bool value = true;
};

template <typename T, typename Alloc>
struct is_container<std::unordered_map<T, Alloc>> {
    static const bool value = true;
};

template <typename T, typename Alloc>
struct is_container<std::set<T, Alloc>> {
    static const bool value = true;
};

template <typename T>
struct is_stdarray {
    static const bool value = false;
};

template <typename T, std::size_t N>
struct is_stdarray<std::array<T, N>> {
    static const bool value = true;
};

template <typename T>
struct is_map {
    static const bool value = false;
};

template <typename T, typename Alloc>
struct is_map<std::map<T, Alloc>> {
    static const bool value = true;
};

template <typename T, typename Alloc>
struct is_map<std::unordered_map<T, Alloc>> {
    static const bool value = true;
};
}  // namespace mp_helper

class RS_MessagePackSerializer {
   public:
    template <typename T>
    void operator()(const std::string &key, const T &value) {
        _map_size++;
        _pack(key);
        _pack(value);
    }

    [[nodiscard]] std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> full_data;
        if (_map_size < 16) {
            auto size_mask = uint8_t(0b10000000);
            _appendData(full_data, uint8_t(_map_size | size_mask));
        } else if (_map_size < std::numeric_limits<uint16_t>::max()) {
            full_data.emplace_back(MP_MAP16);
            _appendData(full_data, uint16_t(_map_size));
        } else if (_map_size < std::numeric_limits<uint32_t>::max()) {
            full_data.emplace_back(MP_MAP32);
            _appendData(full_data, uint32_t(_map_size));
        }
        full_data.insert(full_data.end(), _data.begin(), _data.end());
        return full_data;
    }

    void clear() { _data.clear(); }

   private:
    std::vector<uint8_t> _data;
    uint32_t _map_size = 0;

    template <typename T>
    static void _appendData(std::vector<uint8_t> &data, T value) {
        for (auto i = sizeof(T); i > 0; --i) {
            data.emplace_back(uint8_t(value >> (8U * (i - 1)) & 0xff));
        }
    }

    template <typename T>
    void _pack(const T &value) {
        constexpr bool is_valid =
            mp_helper::is_map<T>::value || mp_helper::is_container<T>::value ||
            mp_helper::is_stdarray<T>::value || std::is_class_v<T> || std::is_enum_v<T>;
        static_assert(is_valid, "Unknown serialize value type");
        if constexpr (mp_helper::is_map<T>::value) {
            _packMap(value);
        } else if constexpr (mp_helper::is_container<T>::value ||
                             mp_helper::is_stdarray<T>::value) {
            _packArray(value);
        } else if constexpr (std::is_class_v<T>) {
            auto s = RS_MessagePackSerializer{};
            const_cast<T &>(value).serialize(s);
            auto sub_data = s.serialize();
            _data.insert(_data.end(), sub_data.begin(), sub_data.end());
        } else if constexpr (std::is_enum_v<T>) {
            using ValueType = typename std::underlying_type<T>::type;
            _pack(static_cast<ValueType>(value));
        }
    }

    template <typename T>
    inline void _pack(const std::chrono::time_point<T> &value) {
        _pack(value.time_since_epoch().count());
    }

    template <typename T>
    void _packArray(const T &array) {
        if (array.size() < 16) {
            auto size_mask = uint8_t(0b10010000);
            _appendData(_data, uint8_t(array.size() | size_mask));
        } else if (array.size() < std::numeric_limits<uint16_t>::max()) {
            _data.emplace_back(MP_ARRAY16);
            _appendData(_data, uint16_t(array.size()));
        } else if (array.size() < std::numeric_limits<uint32_t>::max()) {
            _data.emplace_back(MP_ARRAY32);
            _appendData(_data, uint32_t(array.size()));
        } else {
            RS_THROW(RS_MessagePackSerializationException(
                "Array size exceeds maximum allowed size (2^32) - 1"
            ));
        }
        for (const auto &elem : array) {
            _pack(elem);
        }
    }

    template <typename T>
    void _packMap(const T &map) {
        if (map.size() < 16) {
            auto size_mask = uint8_t(0b10000000);
            _appendData(_data, uint8_t(map.size() | size_mask));
        } else if (map.size() < std::numeric_limits<uint16_t>::max()) {
            _data.emplace_back(MP_MAP16);
            _appendData(_data, uint16_t(map.size()));
        } else if (map.size() < std::numeric_limits<uint32_t>::max()) {
            _data.emplace_back(MP_MAP32);
            _appendData(_data, uint32_t(map.size()));
        } else {
            RS_THROW(RS_MessagePackSerializationException(
                "Map size exceeds maximum allowed size (2^32) - 1"
            ));
        }
        for (const auto &elem : map) {
            _pack(std::get<0>(elem));
            _pack(std::get<1>(elem));
        }
    }

    static std::bitset<64> _twosComplement(int64_t value) {
        if (value < 0) {
            auto abs_v = llabs(value);
            return ~abs_v + 1;
        } else {
            return {(uint64_t)value};
        }
    }

    static std::bitset<32> _twosComplement(int32_t value) {
        if (value < 0) {
            auto abs_v = abs(value);
            return ~abs_v + 1;
        } else {
            return {(uint32_t)value};
        }
    }

    static std::bitset<16> _twosComplement(int16_t value) {
        if (value < 0) {
            auto abs_v = abs(value);
            return ~abs_v + 1;
        } else {
            return {(uint16_t)value};
        }
    }

    static std::bitset<8> _twosComplement(int8_t value) {
        if (value < 0) {
            auto abs_v = abs(value);
            return ~abs_v + 1;
        } else {
            return {(uint8_t)value};
        }
    }
};

template <>
inline void RS_MessagePackSerializer::_pack(const int8_t &value) {
    if (value > 31 || value < -32) {
        _appendData(_data, MP_INT8);
    }
    _appendData(_data, uint8_t(_twosComplement(value).to_ulong()));
}

template <>
inline void RS_MessagePackSerializer::_pack(const int16_t &value) {
    if (abs(value) < abs(std::numeric_limits<int8_t>::min())) {
        _pack(int8_t(value));
    } else {
        _appendData(_data, MP_INT16);
        _appendData(_data, uint16_t(_twosComplement(value).to_ulong()));
    }
}

template <>
inline void RS_MessagePackSerializer::_pack(const int32_t &value) {
    if (abs(value) < abs(std::numeric_limits<int16_t>::min())) {
        _pack(int16_t(value));
    } else {
        _appendData(_data, MP_INT32);
        _appendData(_data, uint32_t(_twosComplement(value).to_ulong()));
    }
}

template <>
inline void RS_MessagePackSerializer::_pack(const int64_t &value) {
    if (llabs(value) < llabs(std::numeric_limits<int32_t>::min()) &&
        value != std::numeric_limits<int64_t>::min()) {
        _pack(int32_t(value));
    } else {
        _appendData(_data, MP_INT64);
        _appendData(_data, uint64_t(_twosComplement(value).to_ulong()));
    }
}

template <>
inline void RS_MessagePackSerializer::_pack(const uint8_t &value) {
    if (value <= 0x7f) {
        _appendData(_data, value);
    } else {
        _appendData(_data, MP_UINT8);
        _appendData(_data, value);
    }
}

template <>
inline void RS_MessagePackSerializer::_pack(const uint16_t &value) {
    if (value > std::numeric_limits<uint8_t>::max()) {
        _appendData(_data, MP_UINT16);
        _appendData(_data, value);
    } else {
        _pack(uint8_t(value));
    }
}

template <>
inline void RS_MessagePackSerializer::_pack(const uint32_t &value) {
    if (value > std::numeric_limits<uint16_t>::max()) {
        _appendData(_data, MP_UINT32);
        _appendData(_data, value);
    } else {
        _pack(uint16_t(value));
    }
}

template <>
inline void RS_MessagePackSerializer::_pack(const uint64_t &value) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        _appendData(_data, MP_UINT64);
        _appendData(_data, value);
    } else {
        _pack(uint32_t(value));
    }
}

template <>
inline void RS_MessagePackSerializer::_pack(const std::nullptr_t & /*value*/) {
    _appendData(_data, MP_NIL);
}

template <>
inline void RS_MessagePackSerializer::_pack(const bool &value) {
    if (value) {
        _appendData(_data, MP_TRUE);
    } else {
        _appendData(_data, MP_FALSE);
    }
}

template <>
inline void RS_MessagePackSerializer::_pack(const float &value) {
    double integral_part;

    // Currently, decimal floats which uses radix of 10, eg: on IBM machines are not
    // supported
    static_assert(std::numeric_limits<float>::radix == 2);

    uint32_t ieee754_float32;
    if (value == 0) {
        ieee754_float32 = 0;
    } else if (std::isnan(value)) {
        ieee754_float32 = 0x7fc00000;
    } else if (std::isinf(value)) {
        if (value > 0)
            ieee754_float32 = 0x7f800000;
        else
            ieee754_float32 = 0xff800000;
    } else {
        auto exponent = ilogb(value);
        float full_mantissa = value / float(scalbn(1.0, exponent));
        auto sign_mask = std::bitset<32>(uint32_t(std::signbit(full_mantissa)) << 31);
        auto excess_127_exponent_mask = std::bitset<32>(uint32_t(exponent + 127) << 23);
        auto normalized_mantissa_mask = std::bitset<32>();
        float implied_mantissa = fabs(full_mantissa) - 1.0f;
        for (auto i = 23U; i > 0; --i) {
            integral_part = 0;
            implied_mantissa *= 2;
            implied_mantissa = float(modf(implied_mantissa, &integral_part));
            if (uint8_t(integral_part) == 1) {
                normalized_mantissa_mask |= std::bitset<32>(uint32_t(1 << (i - 1)));
            }
        }
        ieee754_float32 =
            (sign_mask | excess_127_exponent_mask | normalized_mantissa_mask).to_ulong();
    }
    _appendData(_data, MP_FLOAT32);
    _appendData(_data, ieee754_float32);
}

template <>
inline void RS_MessagePackSerializer::_pack(const double &value) {
    double integral_part;

    // Currently, decimal floats which uses radix of 10, eg: on IBM machines are not
    // supported
    static_assert(std::numeric_limits<float>::radix == 2);

    uint64_t ieee754_float64;
    if (value == 0) {
        ieee754_float64 = 0;
    } else if (std::isnan(value)) {
        ieee754_float64 = 0x7ff8000000000000;
    } else if (std::isinf(value)) {
        if (value > 0)
            ieee754_float64 = 0x7ff0000000000000;
        else
            ieee754_float64 = 0xfff0000000000000;
    } else {
        auto exponent = ilogb(value);
        double full_mantissa = value / scalbn(1.0, exponent);
        auto sign_mask = std::bitset<64>(uint64_t(std::signbit(full_mantissa)) << 63);
        auto excess_127_exponent_mask = std::bitset<64>(uint64_t(exponent + 1023) << 52);
        auto normalized_mantissa_mask = std::bitset<64>();
        double implied_mantissa = fabs(full_mantissa) - 1.0f;

        for (auto i = 52U; i > 0; --i) {
            integral_part = 0;
            implied_mantissa *= 2;
            implied_mantissa = modf(implied_mantissa, &integral_part);
            if (uint8_t(integral_part) == 1) {
                normalized_mantissa_mask |= std::bitset<64>(uint64_t(1) << (i - 1));
            }
        }
        ieee754_float64 =
            (sign_mask | excess_127_exponent_mask | normalized_mantissa_mask).to_ullong();
    }
    _appendData(_data, MP_FLOAT64);
    _appendData(_data, ieee754_float64);
}

template <>
inline void RS_MessagePackSerializer::_pack(const std::string &value) {
    if (value.size() < 32) {
        _appendData(_data, uint8_t(value.size() | 0b10100000));
    } else if (value.size() < std::numeric_limits<uint8_t>::max()) {
        _appendData(_data, MP_STR8);
        _appendData(_data, uint8_t(value.size()));
    } else if (value.size() < std::numeric_limits<uint16_t>::max()) {
        _appendData(_data, MP_STR16);
        _appendData(_data, uint16_t(value.size()));
    } else if (value.size() < std::numeric_limits<uint32_t>::max()) {
        _appendData(_data, MP_STR32);
        _appendData(_data, uint32_t(value.size()));
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "String size exceeds maximum allowed size (2^32) - 1"
        ));
    }
    for (char i : value) {
        _appendData(_data, static_cast<uint8_t>(i));
    }
}

template <>
inline void RS_MessagePackSerializer::_pack(const std::vector<uint8_t> &value) {
    if (value.size() < std::numeric_limits<uint8_t>::max()) {
        _appendData(_data, MP_BIN8);
        _appendData(_data, uint8_t(value.size()));
    } else if (value.size() < std::numeric_limits<uint16_t>::max()) {
        _appendData(_data, MP_BIN16);
        _appendData(_data, uint16_t(value.size()));
    } else if (value.size() < std::numeric_limits<uint32_t>::max()) {
        _appendData(_data, MP_BIN32);
        _appendData(_data, uint32_t(value.size()));
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Byte array size exceeds maximum allowed size (2^32) - 1"
        ));
    }
    for (const auto &elem : value) {
        _appendData(_data, elem);
    }
}

struct RS_MessagePackMemoryAccessor {
    const uint8_t *data = nullptr;
    const uint8_t *data_end = nullptr;

    RS_MessagePackMemoryAccessor() = default;
    RS_MessagePackMemoryAccessor(const uint8_t *data, const uint8_t *data_end)
        : data(data), data_end(data_end) {}

    uint8_t operator*() const {
        if (data < data_end) return *data;
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected end of data encountered while deserializing"
        ));
        return 0;
    }

    void move(size_t bytes = 1) {
        if (data_end - data >= 0) {
            data += bytes;
        } else {
            RS_THROW(RS_MessagePackSerializationException(
                "Unexpected end of data encountered while deserializing"
            ));
        }
    }

    template <typename T>
    size_t readSize() {
        static_assert(std::is_unsigned_v<T>, "readSize can only read unsigned sizes");
        size_t size = 0;
        for (auto i = sizeof(T); i > 0; --i) {
            size += size_t(this->operator*()) << 8 * (i - 1);
            move();
        }
        return size;
    }
};

struct RS_MessagePackStructureTreeNode {
    bool is_scalar = false;
    bool is_map = false;
    bool is_array = false;
    RS_MessagePackStructureTreeNode *parent = nullptr;
    const uint8_t *data = nullptr;
    size_t size = 0;
    std::vector<RS_MessagePackStructureTreeNode> children;

    RS_MessagePackStructureTreeNode() = default;
    RS_MessagePackStructureTreeNode(const uint8_t *data, size_t size) : data(data), size(size) {}

    void parse(const uint8_t *binary_data, const uint8_t *binary_data_end) {
        data = binary_data;
        RS_MessagePackMemoryAccessor acc(binary_data, binary_data_end);
        uint8_t type = *acc;
        acc.move();
        bool is_fixed_int = (type & 0b10000000) == 0 || (type & 0b11100000) == 0b11100000;
        bool is_fixed_str = (type & 0b11100000) == 0b10100000;
        bool is_fixed_map = (type & 0b11110000) == 0b10000000;
        bool is_fixed_array = (type & 0b11110000) == 0b10010000;
        // size include the type byte
        if (is_fixed_int || type == MP_NIL || type == MP_FALSE || type == MP_TRUE)
            size = 1, is_scalar = true;
        else if (type == MP_INT8 || type == MP_UINT8)
            size = 2, is_scalar = true;
        else if (type == MP_INT16 || type == MP_UINT16)
            size = 3, is_scalar = true;
        else if (type == MP_INT32 || type == MP_UINT32 || type == MP_FLOAT32)
            size = 5, is_scalar = true;
        else if (type == MP_INT64 || type == MP_UINT64 || type == MP_FLOAT64)
            size = 9, is_scalar = true;
        else if (is_fixed_str)
            size = 1 + static_cast<size_t>(type & 0b00011111), is_scalar = true;
        else if (type == MP_STR8 || type == MP_BIN8) {
            size = 2 + acc.readSize<uint8_t>(), is_scalar = true;
        } else if (type == MP_STR16 || type == MP_BIN16) {
            size = 3 + acc.readSize<uint16_t>(), is_scalar = true;
        } else if (type == MP_STR32 || type == MP_BIN32) {
            size = 5 + acc.readSize<uint32_t>(), is_scalar = true;
        } else if (type == MP_ARRAY16 || type == MP_ARRAY32) {
            is_array = true;
            size = 1 + (type == MP_ARRAY16 ? 2 : 4);
            size_t num = type == MP_ARRAY16 ? acc.readSize<uint16_t>() : acc.readSize<uint32_t>();
            for (size_t i = 0; i < num; i++) {
                children.emplace_back();
                auto &new_child = children.back();
                new_child.parent = this;
                new_child.parse(acc.data, binary_data_end);
                size += new_child.size;
                acc.move(new_child.size);
            }
        } else if (is_fixed_array) {
            is_array = true;
            size = 1;
            size_t num = type & 0b00001111;
            for (size_t i = 0; i < num; i++) {
                children.emplace_back();
                auto &new_child = children.back();
                new_child.parent = this;
                new_child.parse(acc.data, binary_data_end);
                size += new_child.size;
                acc.move(new_child.size);
            }
        } else if (type == MP_MAP16 || type == MP_MAP32) {
            is_map = true;
            size = 1 + (type == MP_MAP16 ? 2 : 4);
            size_t num = type == MP_MAP16 ? acc.readSize<uint16_t>() : acc.readSize<uint32_t>();
            for (size_t i = 0; i < num; i++) {
                children.emplace_back();
                auto &new_key = children.back();
                new_key.parent = this;
                new_key.parse(acc.data, binary_data_end);
                size += new_key.size;
                acc.move(new_key.size);

                children.emplace_back();
                auto &new_value = children.back();
                new_value.parent = this;
                new_value.parse(acc.data, binary_data_end);
                size += new_value.size;
                acc.move(new_value.size);
            }
        } else if (is_fixed_map) {
            is_map = true;
            size = 1;
            size_t num = type & 0b00001111;
            for (size_t i = 0; i < num; i++) {
                children.emplace_back();
                auto &new_key = children.back();
                new_key.parent = this;
                new_key.parse(acc.data, binary_data_end);
                size += new_key.size;
                acc.move(new_key.size);

                children.emplace_back();
                auto &new_value = children.back();
                new_value.parent = this;
                new_value.parse(acc.data, binary_data_end);
                size += new_value.size;
                acc.move(new_value.size);
            }
        } else {
            RS_THROW(RS_MessagePackSerializationException(
                "Unexpected data type {:d} encountered while parsing", type
            ));
        }
    }
};

class RS_MessagePackDeserializer {
   public:
    template <typename T>
    void operator()(const std::string &key, T &value, bool optional = false) {
        if (_map.find(key) != _map.end()) {
            _unpack(*_map.at(key), value);
        } else if (!optional) {
            RS_THROW(RS_MessagePackSerializationException("Unable to find key {}", key));
        }
    }

    void deserialize(const uint8_t *data, std::size_t size) {
        _root.parse(data, data + size);
        deserialize(_root);
    }

    // Use template Node, since the _unpack call on key requires instantiation of the function
    // but it's defined later. Also we cannot separate the definition of this function since
    // it's defined in the header file and will cause multiple definitions if separated.
    template <typename Node>
    void deserialize(Node &root) {
        _root = root;
        if (!_root.is_map) {
            RS_THROW(
                RS_MessagePackSerializationException("The root of deserialized data is not a map")
            );
        }
        for (size_t i = 0, total = _root.children.size() / 2; i < total; i++) {
            std::string key;
            auto &key_node = _root.children[i * 2];
            auto &value_node = _root.children[i * 2 + 1];
            _unpack(key_node, key);
            _map[key] = &value_node;
        }
    }

   private:
    RS_MessagePackStructureTreeNode _root;
    std::unordered_map<std::string, RS_MessagePackStructureTreeNode *> _map;

    template <typename T>
    void _unpack(RS_MessagePackStructureTreeNode &node, T &value) {
        constexpr bool is_valid =
            mp_helper::is_map<T>::value || mp_helper::is_container<T>::value ||
            mp_helper::is_stdarray<T>::value || std::is_class_v<T> || std::is_enum_v<T>;
        static_assert(is_valid, "Unknown deserialize value type");
        if constexpr (mp_helper::is_map<T>::value) {
            _unpackMap(node, value);
        } else if constexpr (mp_helper::is_container<T>::value) {
            _unpackArray(node, value);
        } else if constexpr (mp_helper::is_stdarray<T>::value) {
            _unpackStdArray(node, value);
        } else if constexpr (std::is_class_v<T>) {
            auto d = RS_MessagePackDeserializer();
            d.deserialize(node);
            value.serialize(d);
        } else if constexpr (std::is_enum_v<T>) {
            using ValueType = typename std::underlying_type<T>::type;
            ValueType val{};
            _unpack(node, val);
            value = static_cast<T>(val);
        }
    }

    template <class Clock, class Duration>
    inline void _unpack(
        RS_MessagePackStructureTreeNode &node,
        std::chrono::time_point<Clock, Duration> &value
    ) {
        using RepType = typename std::chrono::time_point<Clock, Duration>::rep;
        using DurationType = Duration;
        using TimepointType = typename std::chrono::time_point<Clock, Duration>;
        auto placeholder = RepType{};
        _unpack(node, placeholder);
        value = TimepointType(DurationType(placeholder));
    }

    template <typename T>
    void _unpackArray(RS_MessagePackStructureTreeNode &node, T &array) {
        if (!node.is_array)
            RS_THROW(RS_MessagePackSerializationException(
                "Unpacking node(is_scalar:{}, is_array:{}, is_map:{}) to array type is forbidden",
                node.is_scalar,
                node.is_array,
                node.is_map
            ));
        using ValueType = typename T::value_type;
        for (auto &child : node.children) {
            ValueType val{};
            _unpack(child, val);
            array.emplace_back(val);
        }
    }

    template <typename T>
    void _unpackStdArray(RS_MessagePackStructureTreeNode &node, T &array) {
        using ValueType = typename T::value_type;
        auto vec = std::vector<ValueType>{};
        _unpackArray(node, vec);
        std::copy(vec.begin(), vec.end(), array.begin());
    }

    template <typename T>
    void _unpackMap(RS_MessagePackStructureTreeNode &node, T &map) {
        if (!node.is_map)
            RS_THROW(RS_MessagePackSerializationException(
                "Unpacking node(is_scalar:{}, is_array:{}, is_map:{}) to map type is forbidden",
                node.is_scalar,
                node.is_array,
                node.is_map
            ));
        using KeyType = typename T::key_type;
        using MappedType = typename T::mapped_type;

        for (size_t i = 0, total = node.children.size() / 2; i < total; i++) {
            KeyType key{};
            MappedType value{};
            _unpack(node.children[i * 2], key);
            _unpack(node.children[i * 2 + 1], value);
            map.insert_or_assign(key, value);
        }
    }
};

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, int8_t &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    if (*acc == MP_INT8) {
        acc.move();
        value = static_cast<int8_t>(*acc);
        acc.move();
    } else if ((*acc & 0b10000000) == 0 || (*acc & 0b11100000) == 0b11100000) {
        value = static_cast<int8_t>(*acc);
        acc.move();
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing int8", *acc
        ));
    }
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, int16_t &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    if (*acc == MP_INT16) {
        acc.move();
        std::bitset<16> bits;
        for (auto i = sizeof(uint16_t); i > 0; --i) {
            bits |= uint16_t(*acc) << 8 * (i - 1);
            acc.move();
        }
        if (bits[15]) {
            value = int16_t(-1) * static_cast<int16_t>(uint16_t((~bits).to_ulong()) + 1);
        } else {
            value = static_cast<int16_t>(bits.to_ulong());
        }
    } else if (*acc == MP_INT8) {
        int8_t val;
        _unpack(node, val);
        value = static_cast<int16_t>(val);
    } else if ((*acc & 0b10000000) == 0 || (*acc & 0b11100000) == 0b11100000) {
        value = static_cast<int16_t>(*acc);
        acc.move();
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing int16", *acc
        ));
    }
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, int32_t &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    if (*acc == MP_INT32) {
        acc.move();
        std::bitset<32> bits;
        for (auto i = sizeof(uint32_t); i > 0; --i) {
            bits |= uint32_t(*acc) << 8 * (i - 1);
            acc.move();
        }
        if (bits[31]) {
            value = int32_t(-1) * static_cast<int32_t>(uint32_t((~bits).to_ulong()) + 1);
        } else {
            value = static_cast<int32_t>(bits.to_ulong());
        }
    } else if (*acc == MP_INT16) {
        int16_t val;
        _unpack(node, val);
        value = val;
    } else if (*acc == MP_INT8) {
        int8_t val;
        _unpack(node, val);
        value = static_cast<int32_t>(val);
    } else if ((*acc & 0b10000000) == 0 || (*acc & 0b11100000) == 0b11100000) {
        value = static_cast<int32_t>(*acc);
        acc.move();
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing int32", *acc
        ));
    }
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, int64_t &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    if (*acc == MP_INT64) {
        acc.move();
        std::bitset<64> bits;
        for (auto i = sizeof(value); i > 0; --i) {
            bits |= std::bitset<8>(*acc).to_ullong() << 8 * (i - 1);
            acc.move();
        }
        if (bits[63]) {
            value = int64_t(-1) * static_cast<int64_t>(uint64_t((~bits).to_ullong()) + 1);
        } else {
            value = static_cast<int64_t>(bits.to_ullong());
        }
    } else if (*acc == MP_INT32) {
        int32_t val;
        _unpack(node, val);
        value = val;
    } else if (*acc == MP_INT16) {
        int16_t val;
        _unpack(node, val);
        value = val;
    } else if (*acc == MP_INT8) {
        int8_t val;
        _unpack(node, val);
        value = static_cast<int64_t>(val);
    } else if ((*acc & 0b10000000) == 0 || (*acc & 0b11100000) == 0b11100000) {
        value = static_cast<int64_t>(*acc);
        acc.move();
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing int64", *acc
        ));
    }
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, uint8_t &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    if (*acc == MP_UINT8) {
        acc.move();
        value = *acc;
        acc.move();
    } else if ((*acc & 0b10000000) == 0) {
        value = *acc;
        acc.move();
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing uint8", *acc
        ));
    }
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, uint16_t &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    value = 0;  // accumulated below; must not start from the member's prior value
    if (*acc == MP_UINT16) {
        acc.move();
        for (auto i = sizeof(uint16_t); i > 0; --i) {
            value += *acc << 8 * (i - 1);
            acc.move();
        }
    } else if (*acc == MP_UINT8) {
        acc.move();
        value = *acc;
        acc.move();
    } else if ((*acc & 0b10000000) == 0) {
        value = *acc;
        acc.move();
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing uint16", *acc
        ));
    }
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, uint32_t &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    value = 0;  // accumulated below; must not start from the member's prior value
    if (*acc == MP_UINT32) {
        acc.move();
        for (auto i = sizeof(uint32_t); i > 0; --i) {
            value += *acc << 8 * (i - 1);
            acc.move();
        }
    } else if (*acc == MP_UINT16) {
        acc.move();
        for (auto i = sizeof(uint16_t); i > 0; --i) {
            value += *acc << 8 * (i - 1);
            acc.move();
        }
    } else if (*acc == MP_UINT8) {
        acc.move();
        value = *acc;
        acc.move();
    } else if ((*acc & 0b10000000) == 0) {
        value = *acc;
        acc.move();
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing uint32", *acc
        ));
    }
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, uint64_t &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    value = 0;  // accumulated below; must not start from the member's prior value
    if (*acc == MP_UINT64) {
        acc.move();
        for (auto i = sizeof(uint64_t); i > 0; --i) {
            value += uint64_t(*acc) << 8 * (i - 1);
            acc.move();
        }
    } else if (*acc == MP_UINT32) {
        acc.move();
        for (auto i = sizeof(uint32_t); i > 0; --i) {
            value += uint64_t(*acc) << 8 * (i - 1);
            acc.move();
        }
    } else if (*acc == MP_UINT16) {
        acc.move();
        for (auto i = sizeof(uint16_t); i > 0; --i) {
            value += uint64_t(*acc) << 8 * (i - 1);
            acc.move();
        }
    } else if ((*acc & 0b10000000) == 0) {
        value = *acc;
        acc.move();
    } else if (*acc == MP_UINT8) {
        acc.move();
        value = *acc;
        acc.move();
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing uint64", *acc
        ));
    }
}

template <>
inline void RS_MessagePackDeserializer::
    _unpack(RS_MessagePackStructureTreeNode &node, std::nullptr_t & /*value*/) {
    // No need to do anything
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, bool &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    if (*acc != MP_FALSE && *acc != MP_TRUE)
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected byte {:d} encountered while deserializing bool", *acc
        ));
    value = *acc == MP_TRUE;
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, float &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    if (*acc == MP_FLOAT32) {
        acc.move();
        uint32_t data = 0;
        for (auto i = sizeof(uint32_t); i > 0; --i) {
            data += *acc << 8 * (i - 1);
            acc.move();
        }
        if (data == 0) {
            value = 0;
        } else if (data == 0x7fc00000) {
            value = std::numeric_limits<float>::quiet_NaN();
        } else if (data == 0x7f800000) {
            value = std::numeric_limits<float>::infinity();
        } else if (data == 0xff800000) {
            value = -std::numeric_limits<float>::infinity();
        } else {
            auto bits = std::bitset<32>(data);
            auto mantissa = 1.0f;
            for (auto i = 23U; i > 0; --i) {
                if (bits[i - 1]) {
                    mantissa += 1.0f / static_cast<float>(1 << (24 - i));
                }
            }
            if (bits[31]) {
                mantissa *= -1;
            }
            int exponent = 0;
            for (auto i = 0U; i < 8; ++i) {
                exponent += bits[i + 23] << i;
            }
            exponent -= 127;
            value = ldexp(mantissa, exponent);
        }
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing float", *acc
        ));
    }
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, double &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    if (*acc == MP_FLOAT64) {
        acc.move();
        uint64_t data = 0;
        for (auto i = sizeof(uint64_t); i > 0; --i) {
            data += uint64_t(*acc) << 8 * (i - 1);
            acc.move();
        }
        if (data == 0) {
            value = 0;
        } else if (data == 0x7ff8000000000000) {
            value = std::numeric_limits<double>::quiet_NaN();
        } else if (data == 0x7ff0000000000000) {
            value = std::numeric_limits<double>::infinity();
        } else if (data == 0xfff0000000000000) {
            value = -std::numeric_limits<double>::infinity();
        } else {
            auto bits = std::bitset<64>(data);
            auto mantissa = 1.0;
            for (auto i = 52U; i > 0; --i) {
                if (bits[i - 1]) {
                    mantissa += 1.0 / static_cast<double>(uint64_t(1) << (53 - i));
                }
            }
            if (bits[63]) {
                mantissa *= -1;
            }
            int exponent = 0;
            for (auto i = 0U; i < 11; ++i) {
                exponent += bits[i + 52] << i;
            }
            exponent -= 1023;
            value = ldexp(mantissa, exponent);
        }
    } else if (*acc == MP_FLOAT32) {
        float val;
        _unpack(node, val);
        value = val;
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing double", *acc
        ));
    }
}

template <>
inline void
RS_MessagePackDeserializer::_unpack(RS_MessagePackStructureTreeNode &node, std::string &value) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    std::size_t str_size = 0;
    if (*acc == MP_STR32) {
        acc.move();
        for (auto i = sizeof(uint32_t); i > 0; --i) {
            str_size += uint32_t(*acc) << 8 * (i - 1);
            acc.move();
        }
    } else if (*acc == MP_STR16) {
        acc.move();
        for (auto i = sizeof(uint16_t); i > 0; --i) {
            str_size += uint16_t(*acc) << 8 * (i - 1);
            acc.move();
        }
    } else if (*acc == MP_STR8) {
        acc.move();
        for (auto i = sizeof(uint8_t); i > 0; --i) {
            str_size += uint8_t(*acc) << 8 * (i - 1);
            acc.move();
        }
    } else if ((*acc & 0b11100000) == 0b10100000) {
        str_size = *acc & 0b00011111;
        acc.move();
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing string", *acc
        ));
    }
    value = std::string{acc.data, acc.data + str_size};
}

template <>
inline void RS_MessagePackDeserializer::_unpack(
    RS_MessagePackStructureTreeNode &node,
    std::vector<uint8_t> &value
) {
    RS_MessagePackMemoryAccessor acc(node.data, node.data + node.size);
    std::size_t bin_size = 0;
    if (*acc == MP_BIN32) {
        acc.move();
        for (auto i = sizeof(uint32_t); i > 0; --i) {
            bin_size += uint32_t(*acc) << 8 * (i - 1);
            acc.move();
        }
    } else if (*acc == MP_BIN16) {
        acc.move();
        for (auto i = sizeof(uint16_t); i > 0; --i) {
            bin_size += uint16_t(*acc) << 8 * (i - 1);
            acc.move();
        }
    } else if (*acc == MP_BIN8) {
        acc.move();
        for (auto i = sizeof(uint8_t); i > 0; --i) {
            bin_size += uint8_t(*acc) << 8 * (i - 1);
            acc.move();
        }
    } else {
        RS_THROW(RS_MessagePackSerializationException(
            "Unexpected data type {:d} encountered while deserializing byte array", *acc
        ));
    }
    value = std::vector<uint8_t>{acc.data, acc.data + bin_size};
}

}  // namespace rise

#endif  // RS_MP_SERIALIZATION_H