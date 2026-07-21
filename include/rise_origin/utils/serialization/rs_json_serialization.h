// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_JSON_SERIALIZATION_H
#define RS_JSON_SERIALIZATION_H

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "rise_origin/utils/data/rs_bytes_view.h"
#include "rise_origin/utils/data/rs_data_traits.h"
#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/string/rs_string_traits.h"
#include "rs_serialization_traits.h"

namespace rise {

/*****************************************************************
 * Exceptions
 *****************************************************************/
RS_MAKE_EXCEPTION(RS_JSONSerializationException, RS_DataException)

/*****************************************************************
 * Class Definitions
 *****************************************************************/
/**
 * @brief CRTP base for JSON serializers.
 * @tparam FinalSerializer The final-derived serializer type (CRTP parameter).
 * @details Derived classes may shadow `_pack` to intercept specific types.
 */
template <typename FinalSerializer>
class RS_JSONSerializerBase {
   public:
    static constexpr bool isSerialize() { return true; }

    /**
     * @brief Serialize a key-value pair.
     * @param key The key to serialize.
     * @param value The value to serialize.
     */
    template <typename T>
    void operator()(const std::string_view &key, const T &value) {
        static_assert(RS_IsSerializable<T>::value, "Value type is not serializable");
        _is_data_valid = false;
        if (!_is_serializing) {
            RS_THROW(RS_JSONSerializationException(
                "Serializer is not initialized, use .serialize(value) to initialize it."
            ));
        }
        _json[std::string(key)] = _self()._pack(value);
    }

    /**
     * @brief Serialize the root value.
     * @param value The root value to serialize.
     */
    template <typename T>
    void serialize(const T &value) {
        static_assert(RS_IsSerializable<T>::value, "Value type is not serializable");
        _is_data_valid = false;
        _is_serializing = true;
        _json = _self()._pack(value);
    }

    /**
     * @brief Get the serialized data.
     * @return The serialized data.
     */
    [[nodiscard]] RS_BytesView getData() {
        if (!_is_data_valid) {
            _data = _json.dump();
            _is_data_valid = true;
        }
        return {_data.data(), _data.size()};
    }

    /**
     * @brief Get the serialized JSON.
     * @return The serialized JSON.
     */
    [[nodiscard]] const RS_JSON &getJSON() const { return _json; }

    /**
     * @brief Move the serialized JSON out of the serializer.
     * @return The serialized JSON.
     */
    RS_JSON &&moveJSON() {
        _is_data_valid = false;
        return std::move(_json);
    }

    /**
     * @brief Clear the serialized data.
     */
    void clear() {
        _is_data_valid = false;
        _json = RS_JSON{};
        _data.clear();
    }

   protected:
    bool _is_data_valid = false;
    bool _is_serializing = false;
    RS_JSON _json{};
    std::string _data{};

    FinalSerializer &_self() { return static_cast<FinalSerializer &>(*this); }
    const FinalSerializer &_self() const { return static_cast<const FinalSerializer &>(*this); }

    // Template specializations for primitive types
    RS_JSON _pack(const int8_t &value) { return static_cast<int>(value); }

    RS_JSON _pack(const int16_t &value) { return static_cast<int>(value); }

    RS_JSON _pack(const int32_t &value) { return value; }

    RS_JSON _pack(const int64_t &value) { return value; }

    RS_JSON _pack(const uint8_t &value) { return static_cast<unsigned int>(value); }

    RS_JSON _pack(const uint16_t &value) { return static_cast<unsigned int>(value); }

    RS_JSON _pack(const uint32_t &value) { return value; }

    RS_JSON _pack(const uint64_t &value) { return value; }

    RS_JSON _pack(const std::nullptr_t & /*value*/) { return nullptr; }

    RS_JSON _pack(const bool &value) { return value; }

    RS_JSON _pack(const float &value) { return value; }

    RS_JSON _pack(const double &value) { return value; }

    /**
     * @brief Override hook for complex types — FinalSerializer may shadow this.
     * Default implementation delegates to _packDefault.
     */
    template <
        typename T,
        typename = std::enable_if_t<RS_IsSerializable<T>::is_complex_serializable>>
    RS_JSON _pack(const T &value) {
        return _self()._packDefault(value);
    }

    /**
     * @brief Full dispatch logic.  Not intended to be overridden.
     * All recursive calls go through _self()._pack() so that any FinalSerializer
     * override is honored at every nesting level.
     */
    template <
        typename T,
        typename = std::enable_if_t<RS_IsSerializable<T>::is_complex_serializable>>
    RS_JSON _packDefault(const T &value) {
        if constexpr (std::is_enum_v<T>) {
            using ValueType = typename std::underlying_type<T>::type;
            return static_cast<ValueType>(value);
        } else if constexpr (std::is_same_v<T, RS_NoneType>) {
            return nullptr;
        } else if constexpr (RS_IsStringLike<T>::value) {
            return std::string(toStringView(value));
        } else if constexpr (RS_IsMapLike<T>::value) {
            return _packMapLike(value);
        } else if constexpr (RS_IsSetLike<T>::value) {
            return _packSetLike(value);
        } else if constexpr (RS_IsLinearLike<T>::value) {
            if constexpr (RS_IsSerializableBinary<T>::value) {
                return _base64Encode(value);
            } else {
                return _packLinearLike(value);
            }
        } else if constexpr (RS_IsSerializableClass<T>::value) {
            FinalSerializer sub_serializer;
            sub_serializer._is_serializing = true;
            const_cast<T &>(value).serialize(sub_serializer);
            return std::move(sub_serializer._json);
        } else if constexpr (RS_IsExternallySerializableClass<T>::value) {
            FinalSerializer sub_serializer;
            sub_serializer._is_serializing = true;
            RS_Serialize<T>{}(value, sub_serializer);
            return std::move(sub_serializer._json);
        } else {
            // Should not happen
            RS_THROW(RS_JSONSerializationException("Unknown complex serialize value type"));
        }
    }

    template <typename T>
    RS_JSON _packMapLike(const T &map_like) {
        // Static assert to check if key type is string like
        static_assert(
            RS_IsStringLike<typename T::key_type>::value, "Map key type must be string like."
        );

        RS_JSON json_object = RS_JSON::object();
        for (const auto &elem : map_like) {
            json_object[std::string(toStringView(elem.first))] = _self()._pack(elem.second);
        }
        return json_object;
    }

    template <typename T>
    RS_JSON _packSetLike(const T &set_like) {
        RS_JSON json_array = RS_JSON::array();
        for (const auto &elem : set_like) {
            json_array.push_back(_self()._pack(elem));
        }
        return json_array;
    }

    template <typename T>
    RS_JSON _packLinearLike(const T &linear_like) {
        RS_JSON json_array = RS_JSON::array();
        for (const auto &elem : linear_like) {
            json_array.push_back(_self()._pack(elem));
        }
        return json_array;
    }

    // Base64 encoding helper
    template <typename BinaryType>
    std::string _base64Encode(const BinaryType &data) {
        std::string_view chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string result;
        result.reserve(((data.size() + 2) / 3) * 4);

        for (size_t i = 0; i < data.size(); i += 3) {
            uint32_t octet_a = i < data.size() ? data[i] : 0;
            uint32_t octet_b = i + 1 < data.size() ? data[i + 1] : 0;
            uint32_t octet_c = i + 2 < data.size() ? data[i + 2] : 0;
            uint32_t triple = (octet_a << 16) + (octet_b << 8) + octet_c;

            result += chars[(triple >> 18) & 0x3F];
            result += chars[(triple >> 12) & 0x3F];
            result += (i + 1 < data.size()) ? chars[(triple >> 6) & 0x3F] : '=';
            result += (i + 2 < data.size()) ? chars[triple & 0x3F] : '=';
        }
        return result;
    }
};

/**
 * @brief Concrete JSON serializer. Instantiation of RS_JSONSerializerBase.
 */
class RS_JSONSerializer : public RS_JSONSerializerBase<RS_JSONSerializer> {
    friend class RS_JSONSerializerBase<RS_JSONSerializer>;

   public:
    RS_JSONSerializer() = default;
};

/**
 * @brief CRTP base for JSON deserializers.
 * @tparam FinalDeserializer The most-derived deserializer type (CRTP parameter).
 * @details Derived classes may shadow `_unpack` to intercept specific types.
 */
template <typename FinalDeserializer>
class RS_JSONDeserializerBase {
   public:
    static constexpr bool isSerialize() { return false; }

    /**
     * @brief Deserialize a value using the key.
     * @param key The key to deserialize.
     * @param value The value to deserialize.
     * @param optional Whether the key is optional. if true, the key is not required to be present.
     * @throws RS_JSONSerializationException if the key is not found and optional is false.
     */
    template <typename T>
    void operator()(const std::string_view &key, T &value, bool optional = false) {
        static_assert(RS_IsDeserializable<T>::value, "Value type is not deserializable");
        auto it = _current->find(std::string(key));
        if (it != _current->end()) {
            _self()._unpack(*it, value);
        } else if (!optional) {
            RS_THROW(RS_JSONSerializationException("Unable to find key '{}'", key));
        }
    }

    /**
     * @brief Deserialize the root value using the deserializer.
     * @param value The root value to deserialize.
     * @throws RS_JSONSerializationException if the data is not a valid JSON.
     */
    template <typename T>
    void deserialize(T &value) {
        static_assert(RS_IsDeserializable<T>::value, "Value type is not deserializable");
        _self()._unpack(_json, value);
    }

    /**
     * @brief Set the data to the deserializer.
     * @param data The data to load.
     * @throws RS_JSONSerializationException if the data is not a valid JSON.
     */
    void setData(RS_BytesView data) {
        std::string_view json_str(reinterpret_cast<const char *>(data.ptr()), data.size());
        try {
            _json = RS_JSON::parse(json_str);
        } catch (const std::exception &e) {
            RS_THROW(RS_JSONSerializationException("Failed to parse JSON: {}", e.what()));
        }
        _current = &_json;
    }

    /**
     * @brief Set the JSON to the deserializer.
     * @param json The JSON to load.
     */
    void setJSON(const RS_JSON &json) {
        _json = json;
        _current = &_json;
    }

    /**
     * @brief Move the JSON into the deserializer.
     * @param json The JSON to load.
     */
    void moveJSON(RS_JSON &&json) {
        _json = std::move(json);
        _current = &_json;
    }

   protected:
    RS_JSON _json = {};
    const RS_JSON *_current = nullptr;

    FinalDeserializer &_self() { return static_cast<FinalDeserializer &>(*this); }
    const FinalDeserializer &_self() const { return static_cast<const FinalDeserializer &>(*this); }

    void _unpack(const RS_JSON &json_node, int8_t &value) {
        if (!json_node.is_number_integer()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not an integer"));
        }
        value = static_cast<int8_t>(json_node.get<int>());
    }

    void _unpack(const RS_JSON &json_node, int16_t &value) {
        if (!json_node.is_number_integer()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not an integer"));
        }
        value = static_cast<int16_t>(json_node.get<int>());
    }

    void _unpack(const RS_JSON &json_node, int32_t &value) {
        if (!json_node.is_number_integer()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not an integer"));
        }
        value = json_node.get<int32_t>();
    }

    void _unpack(const RS_JSON &json_node, int64_t &value) {
        if (!json_node.is_number_integer()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not an integer"));
        }
        value = json_node.get<int64_t>();
    }

    void _unpack(const RS_JSON &json_node, uint8_t &value) {
        if (!json_node.is_number_unsigned()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not an unsigned integer"));
        }
        value = static_cast<uint8_t>(json_node.get<unsigned int>());
    }

    void _unpack(const RS_JSON &json_node, uint16_t &value) {
        if (!json_node.is_number_unsigned()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not an unsigned integer"));
        }
        value = static_cast<uint16_t>(json_node.get<unsigned int>());
    }

    void _unpack(const RS_JSON &json_node, uint32_t &value) {
        if (!json_node.is_number_unsigned()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not an unsigned integer"));
        }
        value = json_node.get<uint32_t>();
    }

    void _unpack(const RS_JSON &json_node, uint64_t &value) {
        if (!json_node.is_number_unsigned()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not an unsigned integer"));
        }
        value = json_node.get<uint64_t>();
    }

    void _unpack(const RS_JSON &json_node, std::nullptr_t & /*value*/) {
        if (!json_node.is_null()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not null"));
        }
        // No need to assign anything
    }

    void _unpack(const RS_JSON &json_node, bool &value) {
        if (!json_node.is_boolean()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not a boolean"));
        }
        value = json_node.get<bool>();
    }

    void _unpack(const RS_JSON &json_node, float &value) {
        if (!json_node.is_number()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not a number"));
        }
        value = json_node.get<float>();
    }

    void _unpack(const RS_JSON &json_node, double &value) {
        if (!json_node.is_number()) {
            RS_THROW(RS_JSONSerializationException("JSON node is not a number"));
        }
        value = json_node.get<double>();
    }

    /**
     * @brief Override hook for complex types — FinalDeserializer may shadow this.
     * Default implementation delegates to _unpackDefault.
     */
    template <
        typename T,
        typename = std::enable_if_t<RS_IsDeserializable<T>::is_complex_deserializable>>
    void _unpack(const RS_JSON &json_node, T &value) {
        _self()._unpackDefault(json_node, value);
    }

    /**
     * @brief Full dispatch logic.  Not intended to be overridden.
     * All recursive calls go through _self()._unpack() so that any FinalDeserializer
     * override is honoured at every nesting level.
     */
    template <
        typename T,
        typename = std::enable_if_t<RS_IsDeserializable<T>::is_complex_deserializable>>
    void _unpackDefault(const RS_JSON &json_node, T &value) {
        if constexpr (std::is_enum_v<T>) {
            using ValueType = typename std::underlying_type<T>::type;
            ValueType val{};
            _self()._unpack(json_node, val);
            value = static_cast<T>(val);
        } else if constexpr (std::is_same_v<T, RS_NoneType>) {
            if (!json_node.is_null()) {
                RS_THROW(RS_JSONSerializationException("JSON node is not null"));
            }
        } else if constexpr (RS_IsStringLike<T>::value) {
            value = json_node.get<T>();
        } else if constexpr (RS_IsMapLike<T>::value) {
            return _unpackMapLike(json_node, value);
        } else if constexpr (RS_IsSetLike<T>::value) {
            return _unpackSetLike(json_node, value);
        } else if constexpr (RS_IsLinearLike<T>::value) {
            if constexpr (RS_IsSerializableBinary<T>::value) {
                if (!json_node.is_string()) {
                    RS_THROW(
                        RS_JSONSerializationException("JSON node is not a string (expected Base64)")
                    );
                }
                _base64Decode(json_node.get<std::string_view>(), value);
            } else if constexpr (RS_IsStdArray<T>::value) {
                return _unpackArray(json_node, value);
            } else {
                return _unpackLinearLike(json_node, value);
            }
        } else if constexpr (RS_IsSerializableClass<T>::value) {
            const RS_JSON *saved = _current;
            _current = &json_node;
            value.serialize(_self());
            _current = saved;
        } else if constexpr (RS_IsExternallySerializableClass<T>::value) {
            const RS_JSON *saved = _current;
            _current = &json_node;
            RS_Serialize<T>{}(value, _self());
            _current = saved;
        } else {
            // Should not happen
            RS_THROW(RS_JSONSerializationException("Unknown complex deserialize value type"));
        }
    }

    template <typename T>
    void _unpackMapLike(const RS_JSON &json_node, T &map_like) {
        // Static assert to check if key type is string like
        static_assert(
            RS_IsStringLike<typename T::key_type>::value, "Map key type must be string like."
        );

        if (!json_node.is_object()) {
            RS_THROW(RS_JSONSerializationException(
                "JSON node is not an object, cannot unpack to map type"
            ));
        }
        using KeyType = typename T::key_type;
        using MappedType = typename T::mapped_type;

        for (auto it = json_node.begin(); it != json_node.end(); ++it) {
            KeyType key(it.key());
            MappedType mapped_value;
            _self()._unpack(*it, mapped_value);
            map_like.insert_or_assign(std::move(key), std::move(mapped_value));
        }
    }

    template <typename T>
    void _unpackSetLike(const RS_JSON &json_node, T &set_like) {
        if (!json_node.is_array()) {
            RS_THROW(RS_JSONSerializationException(
                "JSON node is not an array, cannot unpack to set type"
            ));
        }

        using ValueType = typename T::value_type;
        for (const auto &child : json_node) {
            ValueType val;
            _self()._unpack(child, val);
            set_like.insert(std::move(val));
        }
    }

    template <typename T>
    void _unpackArray(const RS_JSON &json_node, T &array) {
        if (!json_node.is_array()) {
            RS_THROW(RS_JSONSerializationException(
                "JSON node is not an array, cannot unpack to array type"
            ));
        }
        if (json_node.size() != array.size()) {
            RS_THROW(RS_JSONSerializationException(
                "Array size mismatch, expected {} elements, but found {}",
                array.size(),
                json_node.size()
            ));
        }
        for (std::size_t i = 0; i < array.size(); ++i) {
            _self()._unpack(json_node[i], array[i]);
        }
    }

    template <typename T>
    void _unpackLinearLike(const RS_JSON &json_node, T &linear_like) {
        if (!json_node.is_array()) {
            RS_THROW(RS_JSONSerializationException(
                "JSON node is not an array, cannot unpack to linear or set like type"
            ));
        }
        using ValueType = typename T::value_type;
        for (const auto &child : json_node) {
            ValueType val;
            _self()._unpack(child, val);
            linear_like.emplace_back(std::move(val));
        }
    }

    // Base64 decoding helper
    template <typename T>
    static void _base64Decode(const std::string_view &encoded, T &value) {
        std::string_view chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        value.clear();

        if (encoded.empty()) return;

        // Create lookup table
        std::array<int, 256> lookup{};
        std::fill(lookup.begin(), lookup.end(), -1);
        for (size_t i = 0; i < chars.size(); ++i) {
            lookup[static_cast<unsigned char>(chars[i])] = static_cast<int>(i);
        }

        value.reserve((encoded.size() * 3) / 4);

        for (size_t i = 0; i < encoded.size(); i += 4) {
            uint32_t triple = 0;
            int padding = 0;

            for (int j = 0; j < 4; ++j) {
                if (i + j < encoded.size()) {
                    char c = encoded[i + j];
                    if (c == '=') {
                        padding++;
                    } else {
                        int val = lookup[static_cast<unsigned char>(c)];
                        if (val == -1) {
                            RS_THROW(RS_JSONSerializationException("Invalid Base64 character"));
                        }
                        triple = (triple << 6) | val;
                    }
                } else {
                    triple <<= 6;
                }
            }

            if (padding < 3) value.push_back((triple >> 16) & 0xFF);
            if (padding < 2) value.push_back((triple >> 8) & 0xFF);
            if (padding < 1) value.push_back(triple & 0xFF);
        }
    }
};

/**
 * @brief Concrete JSON deserializer. Instantiation of RS_JSONDeserializerBase.
 */
class RS_JSONDeserializer : public RS_JSONDeserializerBase<RS_JSONDeserializer> {
    friend class RS_JSONDeserializerBase<RS_JSONDeserializer>;

   public:
    RS_JSONDeserializer() = default;
};

}  // namespace rise

#endif  // RS_JSON_SERIALIZATION_H
