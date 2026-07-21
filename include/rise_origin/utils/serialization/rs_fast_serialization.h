// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_FAST_SERIALIZATION_H
#define RS_FAST_SERIALIZATION_H

#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "rise_origin/utils/data/rs_any.h"
#include "rise_origin/utils/data/rs_bytes.h"
#include "rise_origin/utils/data/rs_bytes_view.h"
#include "rise_origin/utils/data/rs_vector_view.h"
#include "rise_origin/utils/memory/rs_nd_memory.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_system.h"
#include "rs_serialization_helper.h"
#include "rs_serialization_traits.h"

namespace rise {

/*****************************************************************
 * Exceptions
 *****************************************************************/
RS_MAKE_EXCEPTION(RS_FastSerializationException, RS_DataException)

/*****************************************************************
 * Class Definitions
 *****************************************************************/
/**
 * @brief RS_FastSerializationBufferView is a non-owning view over
 * a byte range that receives special zero-copy treatment by
 * RS_FastSerializer and RS_FastDeserializer.
 *
 * @details
 *  - Serialization  : uint64_t size (little-endian) + raw bytes
 *  - Deserialization: the view's data/elem_num are set to point
 *    directly inside the source buffer — no allocation.
 */
class RS_FastSerializationBufferView : public RS_BytesView {
   public:
    RS_FastSerializationBufferView() = default;

    RS_FastSerializationBufferView(const RByte* data_ptr, RBigSize size) noexcept
        : RS_BytesView(data_ptr, size) {}

    explicit RS_FastSerializationBufferView(RS_BytesView view) noexcept : RS_BytesView(view) {}
};

/**
 * @brief Type-erased byte-buffer abstraction with a unified append / getWrittenView /
 * size / clear interface.
 *
 * @details Backed by one of:
 *   - RS_Bytes             (default, dynamic CPU buffer)
 *   - std::vector<uint8_t>  (STL interop)
 *   - RS_NdMemory<RByte>    (pre-allocated / device-aware; assumes CPU-accessible memory)
 *
 * Each backing type has two constructor forms:
 *   - Owning  (rvalue / by-value): the backing is moved into internal storage.
 *   - Non-owning (lvalue reference): the backing is referenced via std::reference_wrapper;
 *     the caller retains ownership and the container writes directly into the caller's buffer.
 */
class RS_FastSerializationContainer {
   public:
    RS_FastSerializationContainer() { _storage = RS_Bytes{}; }

    RS_FastSerializationContainer(RS_Bytes& backing) {
        _storage = std::reference_wrapper<RS_Bytes>(backing);
    }
    RS_FastSerializationContainer(std::vector<uint8_t>& backing) {
        _storage = std::reference_wrapper<std::vector<uint8_t>>(backing);
    }
    RS_FastSerializationContainer(std::vector<char>& backing) {
        _storage = std::reference_wrapper<std::vector<char>>(backing);
    }
    RS_FastSerializationContainer(RS_NdMemory<RByte>& backing) {
        _storage = std::reference_wrapper<RS_NdMemory<RByte>>(backing);
    }
    RS_FastSerializationContainer(RS_NdMemory<char>& backing) {
        _storage = std::reference_wrapper<RS_NdMemory<char>>(backing);
    }

    /**
     * @brief Append raw bytes to the container.
     * @param ptr  Pointer to the first byte to append.
     * @param size Number of bytes to append.
     */
    void append(const RByte* ptr, RBigSize size) {
        if (auto* bytes = _getRef<RS_Bytes>()) {
            bytes->append(ptr, size);
        } else if (auto* vec = _getRef<std::vector<uint8_t>>()) {
            vec->insert(
                vec->end(),
                reinterpret_cast<const uint8_t*>(ptr),
                reinterpret_cast<const uint8_t*>(ptr) + size
            );
        } else if (auto* vec = _getRef<std::vector<char>>()) {
            vec->insert(
                vec->end(),
                reinterpret_cast<const char*>(ptr),
                reinterpret_cast<const char*>(ptr) + size
            );
        } else if (auto* nd = _getRef<RS_NdMemory<RByte>>()) {
            RBigSize offset = nd->size();
            nd->resize(offset + size);
            nd->writeSlice(ptr, size, offset);
        } else if (auto* nd = _getRef<RS_NdMemory<char>>()) {
            RBigSize offset = nd->size();
            nd->resize(offset + size);
            nd->writeSlice(reinterpret_cast<const char*>(ptr), size, offset);
        } else {
            RS_THROW(RS_FastSerializationException(
                "RS_FastSerializationContainer: backing storage not initialised"
            ));
        }
    }

    /// Alias for append — identical semantics.
    void write(const RByte* ptr, RBigSize size) { append(ptr, size); }

    /// Returns a view over all bytes written so far.
    [[nodiscard]] RS_BytesView getWrittenView() const {
        if (const auto* bytes = _getRef<RS_Bytes>()) return bytes->toView();
        if (const auto* vec = _getRef<std::vector<uint8_t>>())
            return {reinterpret_cast<const RByte*>(vec->data()), vec->size()};
        if (const auto* vec = _getRef<std::vector<char>>())
            return {reinterpret_cast<const RByte*>(vec->data()), vec->size()};
        if (const auto* nd = _getRef<RS_NdMemory<RByte>>())
            return {reinterpret_cast<const RByte*>(nd->toMemoryPtr().as<RByte>()), nd->size()};
        if (const auto* nd = _getRef<RS_NdMemory<char>>())
            return {reinterpret_cast<const RByte*>(nd->toMemoryPtr().as<char>()), nd->size()};
        return {};
    }

    /// Returns the number of bytes currently stored.
    [[nodiscard]] RBigSize size() const {
        if (const auto* bytes = _getRef<RS_Bytes>()) return bytes->size();
        if (const auto* vec = _getRef<std::vector<uint8_t>>()) return vec->size();
        if (const auto* vec = _getRef<std::vector<char>>()) return vec->size();
        if (const auto* nd = _getRef<RS_NdMemory<RByte>>()) return nd->size();
        if (const auto* nd = _getRef<RS_NdMemory<char>>()) return nd->size();
        return 0;
    }

    void clear() {
        if (auto* bytes = _getRef<RS_Bytes>())
            bytes->clear();
        else if (auto* vec = _getRef<std::vector<uint8_t>>())
            vec->clear();
        else if (auto* vec = _getRef<std::vector<char>>())
            vec->clear();
        else if (auto* nd = _getRef<RS_NdMemory<RByte>>())
            nd->resize(0);
        else if (auto* nd = _getRef<RS_NdMemory<char>>())
            nd->resize(0);
    }

    /// Returns true whether T is the active backing type (owned or referenced).
    template <typename T>
    [[nodiscard]] bool is() const {
        return _storage.is<T>() || _storage.is<std::reference_wrapper<T>>();
    }

    template <typename T>
    T& as() & {
        if (auto* p = _getRef<T>()) return *p;
        return _storage.as<T>();
    }

    template <typename T>
    const T& as() const& {
        if (const auto* p = _getRef<T>()) return *p;
        return _storage.as<T>();
    }

    template <typename T>
    T&& as() && {
        if (_storage.is<T>()) return std::move(_storage).as<T>();
        if (_storage.is<std::reference_wrapper<T>>())
            return std::move(_storage.as<std::reference_wrapper<T>>().get());
        return std::move(_storage).as<T>();
    }

   private:
    RS_Any _storage;

    template <typename T>
    T* _getRef() {
        if (_storage.is<T>()) return &_storage.as<T>();
        if (_storage.is<std::reference_wrapper<T>>())
            return &_storage.as<std::reference_wrapper<T>>().get();
        return nullptr;
    }

    template <typename T>
    const T* _getRef() const {
        if (_storage.is<T>()) return &_storage.as<T>();
        if (_storage.is<std::reference_wrapper<T>>())
            return &_storage.as<std::reference_wrapper<T>>().get();
        return nullptr;
    }
};

/**
 * @brief CRTP base for schema-bound binary serializers with little-endian byte order and
 * zero per-field overhead (no type tags, no field-name storage).
 * @tparam FinalSerializer The final-derived serializer type (CRTP parameter).
 * @details Derived classes may shadow `_pack` to intercept specific types.
 * @note When a class calls s("key", field) inside its serialize() method,
 * the key is silently discarded and only the value is written.  Fields are
 * stored in serialization-declaration order.
 */
template <typename FinalSerializer>
class RS_FastSerializerBase {
   public:
    explicit RS_FastSerializerBase(
        RS_FastSerializationContainer container = RS_FastSerializationContainer{},
        bool is_big_endian_compatible = false
    )
        : _container(std::move(container)), _is_big_endian_compatible(is_big_endian_compatible) {
        if (RS_System::getSystemEndianness() == RSE_SYSTEM_ENDIANNESS_BIG &&
            !is_big_endian_compatible) {
            RS_THROW(RS_FastSerializationException(
                "RS_FastSerializer: big-endian system requires big-endian compatible mode"
            ));
        }
    }

    static constexpr bool isSerialize() { return true; }

    /**
     * @brief Serialize a key-value pair.  The key is ignored.
     */
    template <typename T>
    void operator()(const std::string_view& /*key*/, const T& value) {
        _self()._pack(value);
    }

    /**
     * @brief Serialize a root value.
     */
    template <typename T>
    void serialize(const T& value) {
        _self()._pack(value);
    }

    /// Returns a view over the serialized bytes.
    [[nodiscard]] RS_BytesView getData() const { return _container.getWrittenView(); }

    void clear() { _container.clear(); }

   protected:
    RS_FastSerializationContainer _container;
    bool _is_big_endian_compatible = false;

    FinalSerializer& _self() { return static_cast<FinalSerializer&>(*this); }
    const FinalSerializer& _self() const { return static_cast<const FinalSerializer&>(*this); }

    /// Append raw bytes to the container.
    void _append(const RByte* ptr, RBigSize size) { _container.append(ptr, size); }

    /// Write a scalar value in little-endian byte order.
    template <typename T>
    void _write(T value) {
        using namespace serialization_helper;
        static_assert(
            sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
            "Fast serializer _write: type must be 1, 2, 4 or 8 bytes"
        );
        if constexpr (RS_System::getSystemEndianness() == RSE_SYSTEM_ENDIANNESS_LITTLE) {
            // Little-endian host: wire format is already correct.
            _append(reinterpret_cast<const RByte*>(&value), sizeof(T));
        } else {
            // Big-endian host: swap to little-endian before writing.
            if constexpr (sizeof(T) == 1) {
                uint8_t b = bitcast<uint8_t>(value);
                _append(reinterpret_cast<const RByte*>(&b), 1);
            } else if constexpr (sizeof(T) == 2) {
                uint16_t b = bswap16(bitcast<uint16_t>(value));
                _append(reinterpret_cast<const RByte*>(&b), 2);
            } else if constexpr (sizeof(T) == 4) {
                uint32_t b = bswap32(bitcast<uint32_t>(value));
                _append(reinterpret_cast<const RByte*>(&b), 4);
            } else {
                uint64_t b = bswap64(bitcast<uint64_t>(value));
                _append(reinterpret_cast<const RByte*>(&b), 8);
            }
        }
    }

    void _writeRaw(const RByte* ptr, RBigSize size) { _append(ptr, size); }

    /**
     * @brief Override hook — FinalSerializer may shadow this to intercept specific types.
     * The default implementation delegates to _packDefault.
     */
    template <typename T>
    void _pack(const T& value) {
        _packDefault(value);
    }

    /**
     * @brief Full dispatch logic.  Not intended to be overridden.
     * All recursive calls go through _self()._pack() so that any FinalSerializer
     * override is honored at every nesting level.
     */
    template <typename T>
    void _packDefault(const T& value) {
        // 0. RS_FastSerializationBufferView — special zero-copy path
        if constexpr (std::is_same_v<T, RS_FastSerializationBufferView>) {
            _write(static_cast<uint64_t>(value.elem_num));
            _writeRaw(value.data, value.elem_num);

            // 1. nullptr_t — 0 bytes
        } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
            // nothing

            // 2. bool — 1 byte
        } else if constexpr (std::is_same_v<T, bool>) {
            _write(static_cast<uint8_t>(value ? 1u : 0u));

            // 3. 1-byte integers
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            _write(value);

            // 4. 2-byte integers
        } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>) {
            _write(value);

            // 5. 4-byte scalars
        } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                             std::is_same_v<T, float>) {
            _write(value);

            // 6. 8-byte scalars
        } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
                             std::is_same_v<T, double>) {
            _write(value);

            // 7. enum — cast to underlying type
        } else if constexpr (std::is_enum_v<T>) {
            _self()._pack(static_cast<std::underlying_type_t<T>>(value));

            // 8. RS_NoneType — 0 bytes
        } else if constexpr (std::is_same_v<T, RS_NoneType>) {
            // nothing

            // 10. Binary blobs (before string-like to catch RS_VectorView<uint8_t> etc.)
        } else if constexpr (RS_IsSerializableBinary<T>::value) {
            _packBinaryLike(value);

            // 11. String-like
        } else if constexpr (RS_IsStringLike<T>::value) {
            _packStringLike(value);

            // 13. Map-like (before linear/set to avoid ambiguity)
        } else if constexpr (RS_IsMapLike<T>::value) {
            _packMapLike(value);

            // 14. Set-like
        } else if constexpr (RS_IsSetLike<T>::value) {
            _packSetLike(value);

            // 15. Linear containers (vectors, lists, static/dynamic arrays)
        } else if constexpr (RS_IsLinearLike<T>::value) {
            _packLinearLike(value);

            // 16. Class with serialize() method
        } else if constexpr (RS_IsSerializableClass<T>::value) {
            const_cast<T&>(value).serialize(_self());

            // 17. Externally serializable (RS_Serialize<T> specialisation)
        } else if constexpr (RS_IsExternallySerializableClass<T>::value) {
            RS_Serialize<T>{}(value, _self());

        } else {
            RS_THROW(RS_FastSerializationException(
                "RS_FastSerializer: value type {} is not serializable", typeid(T).name()
            ));
        }
    }

    template <typename T>
    void _packStringLike(const T& str) {
        auto sv = toStringView(str);
        _write(static_cast<uint64_t>(sv.size()));
        _writeRaw(reinterpret_cast<const RByte*>(sv.data()), sv.size());
    }

    template <typename T>
    void _packBinaryLike(const T& bin) {
        _write(static_cast<uint64_t>(bin.size()));
        if constexpr (RS_IsVectorView<T>::value) {
            _writeRaw(reinterpret_cast<const RByte*>(bin.ptr()), bin.size());
        } else {
            _writeRaw(reinterpret_cast<const RByte*>(bin.data()), bin.size());
        }
    }

    template <typename T>
    void _packLinearLike(const T& container) {
        if constexpr (RS_IsStdArray<T>::value && std::is_trivially_copyable_v<T> &&
                      RS_System::getSystemEndianness() == RSE_SYSTEM_ENDIANNESS_LITTLE) {
            // Fixed-size trivially copyable — bulk copy, no count
            if (!_is_big_endian_compatible) {
                _writeRaw(reinterpret_cast<const RByte*>(&container), sizeof(T));
                return;
            }
        }
        if constexpr (RS_IsStdArray<T>::value) {
            // Fixed-size — element by element, no count
            for (const auto& elem : container) {
                _self()._pack(elem);
            }
        } else if constexpr (RS_IsStdVector<T>::value &&
                             std::is_trivially_copyable_v<typename T::value_type> &&
                             RS_System::getSystemEndianness() == RSE_SYSTEM_ENDIANNESS_LITTLE) {
            // Dynamic container trivially copyable — write count and bulk copy
            if (!_is_big_endian_compatible) {
                using ElemT = typename T::value_type;
                _write(static_cast<uint64_t>(container.size()));
                _writeRaw(
                    reinterpret_cast<const RByte*>(container.data()),
                    container.size() * sizeof(ElemT)
                );
                return;
            }
        }
        if constexpr (!RS_IsStdArray<T>::value) {
            // Generic dynamic — write count then elements
            _write(static_cast<uint64_t>(container.size()));
            for (const auto& elem : container) {
                _self()._pack(elem);
            }
        }
    }

    template <typename T>
    void _packMapLike(const T& map) {
        _write(static_cast<uint64_t>(map.size()));
        for (const auto& elem : map) {
            _self()._pack(elem.first);
            _self()._pack(elem.second);
        }
    }

    template <typename T>
    void _packSetLike(const T& set) {
        _write(static_cast<uint64_t>(set.size()));
        for (const auto& elem : set) {
            _self()._pack(elem);
        }
    }
};

/**
 * @brief Schema-bound binary serializer with little-endian byte order and zero
 * per-field overhead (no type tags, no field-name storage). It is a concrete
 * instantiation of RS_FastSerializerBase.
 */
class RS_FastSerializer : public RS_FastSerializerBase<RS_FastSerializer> {
    friend class RS_FastSerializerBase<RS_FastSerializer>;

   public:
    explicit RS_FastSerializer(
        RS_FastSerializationContainer container = RS_FastSerializationContainer{},
        bool is_big_endian_compatible = false
    )
        : RS_FastSerializerBase<RS_FastSerializer>(std::move(container), is_big_endian_compatible) {
    }
};

/**
 * @brief CRTP base for sequential cursor-based binary deserializers matching the format
 * produced by RS_FastSerializerBase.
 * @tparam FinalDeserializer The most-derived deserializer type (CRTP parameter).
 * @details Derived classes may shadow `_unpack` to intercept specific types.
 * @note When the deserialized type is RS_FastSerializationBufferView, the view's
 * data pointer is set to point directly inside the source buffer.  The caller
 * must keep the buffer alive as long as the view is used.
 */
template <typename FinalDeserializer>
class RS_FastDeserializerBase {
   public:
    explicit RS_FastDeserializerBase(bool is_big_endian_compatible = false)
        : _is_big_endian_compatible(is_big_endian_compatible) {
        if (RS_System::getSystemEndianness() == RSE_SYSTEM_ENDIANNESS_BIG &&
            !is_big_endian_compatible) {
            RS_THROW(RS_FastSerializationException(
                "RS_FastDeserializer: big-endian system requires big-endian compatible mode"
            ));
        }
    }

    static constexpr bool isSerialize() { return false; }

    /**
     * @brief Deserialize a key-value pair.  The key is ignored.
     */
    template <typename T>
    void operator()(const std::string_view& /*key*/, T& value, bool /*optional*/ = false) {
        _self()._unpack(value);
    }

    /**
     * @brief Deserialize a root value.
     */
    template <typename T>
    void deserialize(T& value) {
        _self()._unpack(value);
    }

    /**
     * @brief Set the source buffer for deserialization.
     * @note The buffer must remain valid for the lifetime of any
     *       RS_FastSerializationBufferView instances produced by this deserializer.
     */
    void setData(RS_BytesView data) {
        _ptr = data.ptr();
        _end = data.ptr() + data.size();
    }

    /// Returns the number of bytes not yet consumed.
    [[nodiscard]] RBigSize remaining() const {
        return (_ptr <= _end) ? static_cast<RBigSize>(_end - _ptr) : 0;
    }

   protected:
    bool _is_big_endian_compatible = false;
    const RByte* _ptr = nullptr;
    const RByte* _end = nullptr;

    FinalDeserializer& _self() { return static_cast<FinalDeserializer&>(*this); }
    const FinalDeserializer& _self() const { return static_cast<const FinalDeserializer&>(*this); }

    void _checkAvailable(RBigSize needed) const {
        if (remaining() < needed) {
            RS_THROW(RS_FastSerializationException(
                "RS_FastDeserializer: unexpected end of data — need {} bytes, {} available",
                needed,
                remaining()
            ));
        }
    }

    template <typename T>
    T _read() {
        using namespace serialization_helper;
        static_assert(
            sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
            "Fast deserializer _read: type must be 1, 2, 4 or 8 bytes"
        );
        _checkAvailable(sizeof(T));
        if constexpr (RS_System::getSystemEndianness() == RSE_SYSTEM_ENDIANNESS_LITTLE) {
            // Little-endian host: wire bytes are already in native order.
            T value;
            std::memcpy(&value, _ptr, sizeof(T));
            _ptr += sizeof(T);
            return value;
        } else {
            // Big-endian host: swap from little-endian wire format.
            if constexpr (sizeof(T) == 1) {
                return bitcast<T>(*_ptr++);
            } else if constexpr (sizeof(T) == 2) {
                uint16_t b;
                std::memcpy(&b, _ptr, 2);
                _ptr += 2;
                return bitcast<T>(bswap16(b));
            } else if constexpr (sizeof(T) == 4) {
                uint32_t b;
                std::memcpy(&b, _ptr, 4);
                _ptr += 4;
                return bitcast<T>(bswap32(b));
            } else {
                uint64_t b;
                std::memcpy(&b, _ptr, 8);
                _ptr += 8;
                return bitcast<T>(bswap64(b));
            }
        }
    }

    /**
     * @brief Override hook — FinalDeserializer may shadow this to intercept specific types.
     * The default implementation delegates to _unpackDefault.
     */
    template <typename T>
    void _unpack(T& value) {
        _unpackDefault(value);
    }

    /**
     * @brief Full dispatch logic.  Not intended to be overridden.
     * All recursive calls go through _self()._unpack() so that any FinalDeserializer
     * override is honoured at every nesting level.
     */
    template <typename T>
    void _unpackDefault(T& value) {
        // 0. RS_FastSerializationBufferView — zero-copy
        if constexpr (std::is_same_v<T, RS_FastSerializationBufferView>) {
            uint64_t sz = _read<uint64_t>();
            _checkAvailable(static_cast<RBigSize>(sz));
            value = RS_FastSerializationBufferView(_ptr, static_cast<RBigSize>(sz));
            _ptr += sz;

            // 1. nullptr_t — 0 bytes
        } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
            // nothing

            // 2. bool
        } else if constexpr (std::is_same_v<T, bool>) {
            value = (_read<uint8_t>() != 0u);

            // 3. 1-byte integers
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            value = _read<T>();

            // 4. 2-byte integers
        } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>) {
            value = _read<T>();

            // 5. 4-byte scalars
        } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                             std::is_same_v<T, float>) {
            value = _read<T>();

            // 6. 8-byte scalars
        } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
                             std::is_same_v<T, double>) {
            value = _read<T>();

            // 7. enum
        } else if constexpr (std::is_enum_v<T>) {
            using U = std::underlying_type_t<T>;
            U u;
            _self()._unpack(u);
            value = static_cast<T>(u);

            // 8. RS_NoneType — 0 bytes
        } else if constexpr (std::is_same_v<T, RS_NoneType>) {
            // nothing

            // 10. Binary blobs
        } else if constexpr (RS_IsSerializableBinary<T>::value) {
            _unpackBinaryLike(value);

            // 11. String-like (owning only — views excluded by RS_IsDeserializable)
        } else if constexpr (RS_IsStringLike<T>::value) {
            _unpackStringLike(value);

            // 13. Map-like
        } else if constexpr (RS_IsMapLike<T>::value) {
            _unpackMapLike(value);

            // 14. Set-like
        } else if constexpr (RS_IsSetLike<T>::value) {
            _unpackSetLike(value);

            // 15. Linear containers
        } else if constexpr (RS_IsLinearLike<T>::value) {
            _unpackLinearLike(value);

            // 16. Class with serialize() method
        } else if constexpr (RS_IsSerializableClass<T>::value) {
            value.serialize(_self());

            // 17. Externally serializable (RS_Serialize<T> specialisation)
        } else if constexpr (RS_IsExternallySerializableClass<T>::value) {
            RS_Serialize<T>{}(value, _self());

        } else {
            RS_THROW(RS_FastSerializationException(
                "RS_FastDeserializer: value type {} is not deserializable", typeid(T).name()
            ));
        }
    }

    template <typename T>
    void _unpackStringLike(T& str) {
        uint64_t len = _read<uint64_t>();
        _checkAvailable(static_cast<RBigSize>(len));
        str = T{reinterpret_cast<const char*>(_ptr), static_cast<size_t>(len)};
        _ptr += len;
    }

    template <typename T>
    void _unpackBinaryLike(T& bin) {
        uint64_t sz = _read<uint64_t>();
        _checkAvailable(static_cast<RBigSize>(sz));
        // RS_VectorView specialisations are never deserialised (non-owning), so
        // only owning types (std::vector<uint8_t/char>) reach here.
        bin.resize(static_cast<std::size_t>(sz));
        std::memcpy(bin.data(), _ptr, static_cast<std::size_t>(sz));
        _ptr += sz;
    }

    /// Fixed-size array deserialization — no count in stream.
    template <typename T>
    void _unpackArray(T& arr) {
        for (auto& elem : arr) {
            _self()._unpack(elem);
        }
    }

    /// Linear container deserialization — handles fixed-size and dynamic containers.
    template <typename T>
    void _unpackLinearLike(T& container) {
        if constexpr (RS_IsStdArray<T>::value && std::is_trivially_copyable_v<T> &&
                      RS_System::getSystemEndianness() == RSE_SYSTEM_ENDIANNESS_LITTLE) {
            // Fixed-size trivially copyable — bulk copy, no count
            if (!_is_big_endian_compatible) {
                _checkAvailable(sizeof(T));
                std::memcpy(&container, _ptr, sizeof(T));
                _ptr += sizeof(T);
                return;
            }
        }
        if constexpr (RS_IsStdArray<T>::value) {
            // Fixed-size — element by element, no count
            _unpackArray(container);
        } else if constexpr (RS_IsStdVector<T>::value &&
                             std::is_trivially_copyable_v<typename T::value_type> &&
                             RS_System::getSystemEndianness() == RSE_SYSTEM_ENDIANNESS_LITTLE) {
            // Dynamic container trivially copyable — read count and bulk copy
            if (!_is_big_endian_compatible) {
                using ElemT = typename T::value_type;
                uint64_t count = _read<uint64_t>();
                _checkAvailable(static_cast<RBigSize>(count) * sizeof(ElemT));
                container.resize(static_cast<std::size_t>(count));
                std::memcpy(container.data(), _ptr, count * sizeof(ElemT));
                _ptr += count * sizeof(ElemT);
                return;
            }
            // fall through to generic path below
        }
        if constexpr (!RS_IsStdArray<T>::value) {
            // Generic dynamic — read count then elements
            uint64_t count = _read<uint64_t>();
            if constexpr (RS_IsStdVector<T>::value) {
                container.resize(static_cast<std::size_t>(count));
                for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i) {
                    _self()._unpack(container[i]);
                }
            } else {
                // std::list, etc.
                container.clear();
                for (uint64_t i = 0; i < count; ++i) {
                    typename T::value_type v{};
                    _self()._unpack(v);
                    container.push_back(std::move(v));
                }
            }
        }
    }

    template <typename T>
    void _unpackMapLike(T& map) {
        uint64_t count = _read<uint64_t>();
        map.clear();
        for (uint64_t i = 0; i < count; ++i) {
            typename T::key_type k{};
            typename T::mapped_type v{};
            _self()._unpack(k);
            _self()._unpack(v);
            map.emplace(std::move(k), std::move(v));
        }
    }

    template <typename T>
    void _unpackSetLike(T& set) {
        uint64_t count = _read<uint64_t>();
        set.clear();
        for (uint64_t i = 0; i < count; ++i) {
            typename T::value_type v{};
            _self()._unpack(v);
            set.insert(std::move(v));
        }
    }
};

/**
 * @brief Sequential cursor-based binary deserializer matching the format
 * produced by RS_FastSerializer. It is a concrete instantiation of
 * RS_FastDeserializerBase.
 */
class RS_FastDeserializer : public RS_FastDeserializerBase<RS_FastDeserializer> {
    friend class RS_FastDeserializerBase<RS_FastDeserializer>;

   public:
    explicit RS_FastDeserializer(bool is_big_endian_compatible = false)
        : RS_FastDeserializerBase<RS_FastDeserializer>(is_big_endian_compatible) {}
};

}  // namespace rise

#endif  // RS_FAST_SERIALIZATION_H
