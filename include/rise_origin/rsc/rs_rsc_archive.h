// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_ARCHIVE_H
#define RS_RSC_ARCHIVE_H

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "rise_origin/utils/data/rs_bytes.h"
#include "rise_origin/utils/data/rs_bytes_view.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/serialization/rs_fast_serialization.h"
#include "rise_origin/utils/serialization/rs_json_serialization.h"
#include "rs_rsc_registry.h"
#include "rsc_base.h"

namespace rise {

namespace rsc {

/*****************************************************************
 * Exceptions
 *****************************************************************/
RS_MAKE_EXCEPTION(RS_RSCArchiveException, RS_DataException)

/*****************************************************************
 * Class Definitions
 *****************************************************************/
/**
 * @brief In-engine serialization of an RSC node graph to bytes or JSON, preserving sharing.
 * @details The archive is the same-version, positional counterpart of RS_RSCFile. It carries no
 * version gate and no missing-property tolerance, so it stays lean for the in-process pickle and
 * MCP paths. A node and every node it shares are flattened into one pool, so a body shared by many
 * structures is stored once and its shared_ptr identity is restored on read. Bytes go through the
 * fast serializer (pickle); JSON goes through the JSON serializer with field-name keys (MCP).
 * Each RSC drives the pool through _serialize/_deserialize, calling serializeShared on its shared
 * members so the archive alone owns the reference model. A missing key or index on read throws
 * RS_RSCArchiveException rather than a cryptic container error, so an incompatible file is clear.
 */
class RS_RSCArchive {
   public:
    /**
     * @brief Read a node of type T from its byte form.
     * @param bytes The serialized bytes.
     */
    template <typename T>
    static T fromBytes(RS_BytesView bytes) {
        RSC_Base::RSCRef<RSC_Base> node = fromBytesAny(bytes);
        T* typed = dynamic_cast<T*>(node.get());
        if (typed == nullptr) {
            RS_THROW(RS_RSCArchiveException("RSC archive holds a different type than T"));
        }
        return *typed;
    }

    /**
     * @brief Read a node from its byte form.
     * @param bytes The serialized bytes.
     */
    static RSC_Base::RSCRef<RSC_Base> fromBytesAny(RS_BytesView bytes);

    /**
     * @brief Write a node to its byte form.
     * @param node The node to serialize.
     */
    template <typename T>
    static RS_Bytes toBytes(const T& node) {
        return toBytesAny(node);
    }

    /**
     * @brief Write a node to its byte form.
     * @param node The node to serialize.
     */
    static RS_Bytes toBytesAny(const RSC_Base& node);

    /**
     * @brief Read a node of type T from its JSON form.
     * @param json The serialized JSON.
     */
    template <typename T>
    static T fromJSON(const RS_JSON& json) {
        RSC_Base::RSCRef<RSC_Base> node = fromJSONAny(json);
        T* typed = dynamic_cast<T*>(node.get());
        if (typed == nullptr) {
            RS_THROW(RS_RSCArchiveException("RSC archive holds a different type than T"));
        }
        return *typed;
    }

    /**
     * @brief Read a node from its JSON form.
     * @param json The serialized JSON.
     */
    static RSC_Base::RSCRef<RSC_Base> fromJSONAny(const RS_JSON& json);

    /**
     * @brief Write a node to its JSON form.
     * @param node The node to serialize.
     */
    template <typename T>
    static RS_JSON toJSON(const T& node) {
        return toJSONAny(node);
    }

    /**
     * @brief Write a node to its JSON form.
     * @param node The node to serialize.
     */
    static RS_JSON toJSONAny(const RSC_Base& node);

    /**
     * @brief Whether this archive is writing, false when reading.
     */
    [[nodiscard]] bool isSerialize() const;

    /**
     * @brief Read a plain field, or a nested RSC value member under its own key.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param value The field to fill.
     */
    template <typename T>
    void deserialize(std::string_view key, T& value) {
        if constexpr (std::is_base_of_v<RSC_Base, T>) {
            if (_mode == Mode::ReadBytes) {
                value.deserialize(*this);
            } else {
                RS_JSON* saved = _json_node;
                _json_node = &_getChild(*saved, key);
                value.deserialize(*this);
                _json_node = saved;
            }
        } else if (_mode == Mode::ReadBytes) {
            _fast_deserializer(key, value);
        } else {
            RS_JSONDeserializer field_deserializer;
            field_deserializer.setJSON(_getChild(*_json_node, key));
            field_deserializer.deserialize(value);
        }
    }

    /**
     * @brief Read a shared reference as a pool index.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param ref The reference to fill from the pool.
     */
    template <typename T>
    void deserializeShared(std::string_view key, RSC_Base::RSCRef<T>& ref) {
        RIndex index = _deserializeIndex(key);
        if (index == RS_NULL_INDEX) {
            ref = nullptr;
            return;
        }
        ref = std::static_pointer_cast<T>(getSharedAt(index));
    }

    /**
     * @brief Read a vector of shared references as pool indices.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param vec The vector to fill from the pool.
     */
    template <typename T>
    void deserializeShared(std::string_view key, RSC_Base::RSCVec<T>& vec) {
        vec.clear();
        if (_mode == Mode::ReadBytes) {
            uint64_t count = 0;
            _fast_deserializer(key, count);
            for (uint64_t i = 0; i < count; i++) {
                uint64_t index = 0;
                _fast_deserializer(key, index);
                vec.push_back(
                    RSC_Base::RSCRef<T>(
                        std::static_pointer_cast<T>(getSharedAt(static_cast<RIndex>(index)))
                    )
                );
            }
        } else {
            for (const RS_JSON& element : _getChild(*_json_node, key)) {
                vec.push_back(
                    RSC_Base::RSCRef<T>(
                        std::static_pointer_cast<T>(getSharedAt(element.get<RIndex>()))
                    )
                );
            }
        }
    }

    /**
     * @brief Read a single variant, stored inline as a type name and its data.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param variant The variant to fill.
     */
    template <typename... Ts>
    void deserializeVariant(std::string_view key, RS_Variant<Ts...>& variant) {
        RIndex index = RS_NULL_INDEX;
        RS_JSON* saved = nullptr;
        if (_mode == Mode::ReadBytes) {
            uint64_t raw = 0;
            _fast_deserializer("index", raw);
            index = static_cast<RIndex>(raw);
        } else {
            RS_JSON& entry = _getChild(*_json_node, key);
            index = _getChild(entry, RSC_Base::VAR_INDEX_KEY).get<RIndex>();
            saved = _json_node;
            _json_node = &_getChild(entry, RSC_Base::VAR_DATA_KEY);
        }
        switchOver<0, sizeof...(Ts)>(index, [this, &variant](auto i) {
            auto& held = variant.template emplace<i.value>();
            held.deserialize(*this);
        });
        if (saved != nullptr) {
            _json_node = saved;
        }
    }

    /**
     * @brief Read a vector of variants, each element inline as a type name and its data.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param vec The vector to fill.
     */
    template <typename... Ts>
    void deserializeVariantVec(std::string_view key, RSC_Base::RSCVec<RS_Variant<Ts...>>& vec) {
        vec.clear();
        if (_mode == Mode::ReadBytes) {
            uint64_t count = 0;
            _fast_deserializer(key, count);
            for (uint64_t i = 0; i < count; i++) {
                std::string type;
                _fast_deserializer(key, type);
                RSC_Base::RSCRef<RSC_Base> node = RS_RSCRegistry::getExample(type);
                node->deserialize(*this);
                RS_Variant<Ts...> variant;
                if (!RS_RSCRegistry::tryAssign(variant, node)) {
                    RS_THROW(RS_RSCArchiveException("Unexpected RSC type {}", type));
                }
                vec.push_back(variant);
            }
        } else {
            RS_JSON* saved = _json_node;
            for (RS_JSON& element : _getChild(*saved, key)) {
                std::string type = _getChild(element, RSC_Base::VAR_TYPE_KEY).get<std::string>();
                RSC_Base::RSCRef<RSC_Base> node = RS_RSCRegistry::getExample(type);
                _json_node = &_getChild(element, RSC_Base::VAR_DATA_KEY);
                node->deserialize(*this);
                RS_Variant<Ts...> variant;
                if (!RS_RSCRegistry::tryAssign(variant, node)) {
                    RS_THROW(RS_RSCArchiveException("Unexpected RSC type {}", type));
                }
                vec.push_back(variant);
            }
            _json_node = saved;
        }
    }

    /**
     * @brief Write a plain field, or a nested RSC value member under its own key.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param value The field to write.
     */
    template <typename T>
    void serialize(std::string_view key, const T& value) {
        if constexpr (std::is_base_of_v<RSC_Base, T>) {
            if (_is_collecting) {
                value.serialize(*this);
            } else if (_mode == Mode::WriteBytes) {
                value.serialize(*this);
            } else {
                RS_JSON* saved = _json_node;
                RS_JSON data = RS_JSON::object();
                _json_node = &data;
                value.serialize(*this);
                _json_node = saved;
                (*_json_node)[std::string(key)] = std::move(data);
            }
        } else if (_is_collecting) {
            return;
        } else if (_mode == Mode::WriteBytes) {
            _fast_serializer(key, value);
        } else {
            RS_JSONSerializer field_serializer;
            field_serializer.serialize(value);
            (*_json_node)[std::string(key)] = field_serializer.moveJSON();
        }
    }

    /**
     * @brief Write a shared reference as a pool index, pooling the referenced node once.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param ref The reference to write.
     */
    template <typename T>
    void serializeShared(std::string_view key, const RSC_Base::RSCRef<T>& ref) {
        if (_is_collecting) {
            _collectShared(ref.shared());
            return;
        }
        _serializeIndex(key, ref ? _shared_to_index_map.at(ref.get()) : RS_NULL_INDEX);
    }

    /**
     * @brief Write a vector of shared references as pool indices, pooling each node once.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param vec The vector to write.
     */
    template <typename T>
    void serializeShared(std::string_view key, const RSC_Base::RSCVec<T>& vec) {
        if (_is_collecting) {
            for (std::size_t i = 0; i < vec.size(); i++) {
                _collectShared(vec.ref(i).shared());
            }
            return;
        }
        if (_mode == Mode::WriteBytes) {
            _fast_serializer(key, static_cast<uint64_t>(vec.size()));
            for (std::size_t i = 0; i < vec.size(); i++) {
                _fast_serializer(
                    key, static_cast<uint64_t>(_shared_to_index_map.at(vec.ref(i).get()))
                );
            }
        } else {
            RS_JSON indices = RS_JSON::array();
            for (std::size_t i = 0; i < vec.size(); i++) {
                indices.push_back(_shared_to_index_map.at(vec.ref(i).get()));
            }
            (*_json_node)[std::string(key)] = std::move(indices);
        }
    }

    /**
     * @brief Write a single variant inline as a type name and its data.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param variant The variant to write.
     */
    template <typename... Ts>
    void serializeVariant(std::string_view key, const RS_Variant<Ts...>& variant) {
        if (_is_collecting) {
            variant.visit([this](const auto& held) { held.serialize(*this); });
            return;
        }
        RIndex index = variant.index();
        if (_mode == Mode::WriteBytes) {
            _fast_serializer("index", static_cast<uint64_t>(index));
            variant.visit([this](const auto& held) { held.serialize(*this); });
        } else {
            RS_JSON* saved = _json_node;
            RS_JSON entry = RS_JSON::object();
            RS_JSON data = RS_JSON::object();
            variant.visit([this, &data](const auto& held) {
                _json_node = &data;
                held.serialize(*this);
            });
            entry[RSC_Base::VAR_INDEX_KEY] = index;
            entry[RSC_Base::VAR_DATA_KEY] = std::move(data);
            _json_node = saved;
            (*_json_node)[std::string(key)] = std::move(entry);
        }
    }

    /**
     * @brief Write a vector of variants, each element inline as a type name and its data.
     * @param key The field name, used by the JSON backend and ignored by the byte backend.
     * @param vec The vector to write.
     */
    template <typename... Ts>
    void serializeVariantVec(std::string_view key, const RSC_Base::RSCVec<RS_Variant<Ts...>>& vec) {
        if (_is_collecting) {
            for (std::size_t i = 0; i < vec.size(); i++) {
                vec[i].visit([this](const auto& held) { held.serialize(*this); });
            }
            return;
        }
        if (_mode == Mode::WriteBytes) {
            _fast_serializer(key, static_cast<uint64_t>(vec.size()));
            for (std::size_t i = 0; i < vec.size(); i++) {
                vec[i].visit([this](const auto& held) {
                    _fast_serializer("type", RS_RSCRegistry::getName(typeid(held)));
                    held.serialize(*this);
                });
            }
        } else {
            RS_JSON array = RS_JSON::array();
            RS_JSON* saved = _json_node;
            for (std::size_t i = 0; i < vec.size(); i++) {
                vec[i].visit([this, &array](const auto& held) {
                    RS_JSON entry = RS_JSON::object();
                    RS_JSON data = RS_JSON::object();
                    _json_node = &data;
                    held.serialize(*this);
                    entry[RSC_Base::VAR_TYPE_KEY] = RS_RSCRegistry::getName(typeid(held));
                    entry[RSC_Base::VAR_DATA_KEY] = std::move(data);
                    array.push_back(std::move(entry));
                });
            }
            _json_node = saved;
            (*_json_node)[std::string(key)] = std::move(array);
        }
    }

   private:
    /// The direction and wire format this archive runs.
    enum class Mode { ReadBytes, WriteBytes, ReadJSON, WriteJSON };

    /// The direction and wire format this archive runs.
    Mode _mode;

    /// Whether the write is in its first pass, gathering shared nodes into the pool.
    bool _is_collecting = false;

    /// The JSON document, the pool on write and the source on read.
    RS_JSON _json = {};

    /// The data object of the shared entry currently being read or written, for the JSON backend.
    RS_JSON* _json_node = nullptr;

    /// The shared node pool, the base node at index 0 and every shared node after it.
    std::vector<std::shared_ptr<RSC_Base>> _shared;

    /// Maps a pool index to its node, filled on read before any data is read.
    std::unordered_map<RIndex, std::shared_ptr<RSC_Base>> _index_to_shared_map;

    /// Maps a node to its pool index, filled on write while gathering the pool.
    std::unordered_map<const RSC_Base*, RIndex> _shared_to_index_map;

    /// The byte backend, reading direction.
    RS_FastDeserializer _fast_deserializer;

    /// The byte backend, writing direction.
    RS_FastSerializer _fast_serializer;

    explicit RS_RSCArchive(Mode mode) : _mode(mode) {}

    /**
     * @brief Build the pool of shells, then fill each, returning the base node.
     */
    RSC_Base::RSCRef<RSC_Base> _deserializeGraph();

    /**
     * @brief Gather the shared graph rooted at base into the pool, then write the pool.
     */
    void _serializeGraph(const std::shared_ptr<RSC_Base>& base);

    /**
     * @brief Register a node in the pool before recursing into it, so cycles resolve.
     */
    void _collectShared(const std::shared_ptr<RSC_Base>& node);

    /**
     * @brief Read a pool index under key from the active backend.
     */
    RIndex _deserializeIndex(std::string_view key);

    /**
     * @brief Write a pool index under key to the active backend.
     */
    void _serializeIndex(std::string_view key, RIndex index);

    /**
     * @brief Get the pool node at index, throwing when the file names an unknown index.
     */
    std::shared_ptr<RSC_Base> getSharedAt(RIndex index) const;

    /**
     * @brief Get the child of node under key, throwing when the file is missing that key.
     */
    static RS_JSON& _getChild(RS_JSON& node, std::string_view key);
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_ARCHIVE_H
