// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_FILE_TREE_READER_H
#define RS_RSC_FILE_TREE_READER_H

#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/string/rs_string.h"
#include "rs_rsc_file_tree_context.h"

namespace rise {

namespace rsc {

/**
 * @brief Read cursor over a read context, positioned at one node of the document.
 */
class RS_RSCFileTreeReader {
   public:
    /**
     * @brief Init at the root of a document's context.
     */
    explicit RS_RSCFileTreeReader(RS_RSCFileTreeReadContext &context);

    /**
     * @brief Get the schema version this document was written as.
     */
    [[nodiscard]] RS_Version version() const;

    /**
     * @brief Get a value of type T under sub_node_path, throwing when it is absent.
     */
    template <typename T>
    [[nodiscard]] T get(const std::string &sub_node_path) const {
        if (const RS_JSON *node = _resolve(sub_node_path)) {
            return node->get<T>();
        }
        RS_THROW(RS_InvalidCharacterizationException(
            "Failed to find sub_node_path {} under path {}", sub_node_path, join(".", _path)
        ));
    }

    /**
     * @brief Get a value of type T under sub_node_path, or default_value when it is absent.
     */
    template <typename T>
    [[nodiscard]] T get(const std::string &sub_node_path, T default_value) const {
        if (const RS_JSON *node = _resolve(sub_node_path)) {
            return node->get<T>();
        }
        return default_value;
    }

    /**
     * @brief Whether a node exists under key.
     */
    [[nodiscard]] bool hasChild(const std::string &key) const;

    /**
     * @brief Get a sub tree under sub_node_path, throwing when it is absent.
     */
    [[nodiscard]] RS_RSCFileTreeReader getChild(const std::string &sub_node_path) const;

    /**
     * @brief Get all object members of the current node as key and sub tree pairs.
     */
    [[nodiscard]] std::vector<std::pair<std::string, RS_RSCFileTreeReader>> getChildren() const;

    /**
     * @brief Get every element of the array under sub_node_path, empty when absent.
     */
    [[nodiscard]] std::vector<RS_RSCFileTreeReader> getArray(
        const std::string &sub_node_path
    ) const;

    /**
     * @brief Load a bulk array into dst, which the caller has sized to the shape. The binary format
     * reads a compressed field, the text format an inline JSON array.
     */
    template <typename T>
    void getBulk(const std::string &key, T *dst, const std::vector<RBigSize> &shape) const {
        const RS_JSON *node = _resolve(key);
        if (node && node->is_object() && node->contains("$bulk")) {
            std::string field = std::to_string(node->at("$bulk").get<RIndex>());
            RS_MemoryPtr raw = _context->unpacker->getField(field);
            RS_FastPackingFieldType type = _context->unpacker->getFieldType(field);
            _fromBulkMemory(dst, raw, type);
        } else if (node) {
            RBigSize offset = 0;
            _fromBulkJson(*node, dst, offset);
        }
    }

    /**
     * @brief Get the shape of a bulk array under key, from the binary marker or the nested text
     * array, so a caller can size a destination it cannot derive from other fields. Empty when the
     * key is absent.
     */
    [[nodiscard]] std::vector<RBigSize> getBulkShape(const std::string &key) const;

    /**
     * @brief Whether a shared index is already built.
     */
    [[nodiscard]] bool hasShared(RIndex index) const;

    /**
     * @brief Get the object at a shared index, constructing and reading it on the first reference.
     * A later reference to the same index reuses the built object, so cycles and diamonds are safe.
     * Registered before its body is read so a back reference resolves. Throws when the index was
     * built as another type, or its entry is missing from the shared region.
     */
    template <typename T>
    [[nodiscard]] std::shared_ptr<T> getShared(RIndex index) const {
        auto it = _context->shared_map.find(index);
        if (it != _context->shared_map.end()) {
            std::shared_ptr<T> typed = std::dynamic_pointer_cast<T>(it->second);
            if (!typed) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Shared index {} was built as a different type", index
                ));
            }
            return typed;
        }
        std::shared_ptr<T> rsc = std::make_shared<T>();
        _context->shared_map[index] = rsc;
        rsc->read(_getSharedEntry(index, typeid(T)));
        return rsc;
    }

    /**
     * @brief Get the object at a shared index, or nullptr when it has not been built.
     */
    [[nodiscard]] std::shared_ptr<RSC_Base> findShared(RIndex index) const;

   private:
    /// Whole-document state, shared by every cursor.
    RS_RSCFileTreeReadContext *_context;

    /// Current node, a pointer into the context root.
    const RS_JSON *_node;

    /// Path to the current node, kept for error messages.
    std::vector<std::string> _path;

    RS_RSCFileTreeReader(
        RS_RSCFileTreeReadContext *context,
        const RS_JSON &node,
        std::vector<std::string> path
    );

    /**
     * @brief Walk the dotted sub_node_path from the current node, null when a step is missing.
     */
    [[nodiscard]] const RS_JSON *_resolve(const std::string &sub_node_path) const;

    /**
     * @brief Get a cursor over the data of the shared entry at index, after checking its stored
     * type name matches the expected type. Throws when the entry is missing or its type does not
     * match.
     */
    [[nodiscard]] RS_RSCFileTreeReader
    _getSharedEntry(RIndex index, const std::type_info &expected) const;

    /**
     * @brief Copy field bytes into dst, mapping an enum element from its underlying integer, with a
     * memcpy on matching width and an element convert otherwise.
     */
    template <typename T>
    static void _fromBulkMemory(T *dst, const RS_MemoryPtr &raw, RS_FastPackingFieldType type) {
        using U = typename RS_RSCBulkScalar<T>::type;
        if (type.data_size == 0) return;
        RBigSize count = raw.size() / type.data_size;
        if (sizeof(U) == type.data_size) {
            std::memcpy(dst, raw.as<RByte>(), std::size_t(count) * sizeof(T));
        } else {
            const RByte *base = raw.as<RByte>();
            for (RBigSize i = 0; i < count; ++i)
                dst[i] = static_cast<T>(_readElement<U>(base + i * type.data_size, type));
        }
    }

    /**
     * @brief Flatten a nested JSON array into dst.
     */
    template <typename T>
    static void _fromBulkJson(const RS_JSON &node, T *dst, RBigSize &offset) {
        if (node.is_array()) {
            for (const RS_JSON &child : node) _fromBulkJson(child, dst, offset);
        } else {
            dst[offset++] = node.get<T>();
        }
    }

    /**
     * @brief Read one element at the field's stored dtype and width, cast to U.
     */
    template <typename U>
    static U _readElement(const RByte *pointer, RS_FastPackingFieldType type) {
        switch (type.data_type) {
            case 'i':
                if (type.data_size == 1) return U(_loadScalar<int8_t>(pointer));
                if (type.data_size == 2) return U(_loadScalar<int16_t>(pointer));
                if (type.data_size == 4) return U(_loadScalar<int32_t>(pointer));
                if (type.data_size == 8) return U(_loadScalar<int64_t>(pointer));
                break;
            case 'u':
                if (type.data_size == 1) return U(_loadScalar<uint8_t>(pointer));
                if (type.data_size == 2) return U(_loadScalar<uint16_t>(pointer));
                if (type.data_size == 4) return U(_loadScalar<uint32_t>(pointer));
                if (type.data_size == 8) return U(_loadScalar<uint64_t>(pointer));
                break;
            case 'f':
                if (type.data_size == 4) return U(_loadScalar<float>(pointer));
                if (type.data_size == 8) return U(_loadScalar<double>(pointer));
                break;
        }
        RS_THROW(RS_InvalidCharacterizationException("Unsupported packed field element type"));
    }

    template <typename V>
    static V _loadScalar(const RByte *pointer) {
        V value;
        std::memcpy(&value, pointer, sizeof(V));
        return value;
    }
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_FILE_TREE_READER_H
