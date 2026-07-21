// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_FILE_TREE_WRITER_H
#define RS_RSC_FILE_TREE_WRITER_H

#include <string>
#include <type_traits>
#include <vector>

#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/string/rs_string.h"
#include "rs_rsc_file_tree_context.h"

namespace rise {

namespace rsc {

/**
 * @brief Write cursor over a write context, positioned at one node of the document.
 */
class RS_RSCFileTreeWriter {
   public:
    /**
     * @brief Init at the root of a document's context.
     */
    explicit RS_RSCFileTreeWriter(RS_RSCFileTreeWriteContext &context);

    /**
     * @brief Get the schema version this document is written as.
     */
    [[nodiscard]] RS_Version version() const;

    /**
     * @brief Add a value of type T under sub_node_path, creating intermediate objects.
     */
    template <typename T>
    void add(const std::string &sub_node_path, const T &value) {
        _at(sub_node_path) = value;
    }

    /**
     * @brief Add a child object under sub_node_path and return its writer.
     */
    RS_RSCFileTreeWriter addChild(const std::string &sub_node_path);

    /**
     * @brief Add an array under sub_node_path and return its writer.
     */
    RS_RSCFileTreeWriter addArray(const std::string &sub_node_path);

    /**
     * @brief Append an object element to the current array and return its writer.
     */
    RS_RSCFileTreeWriter add();

    /**
     * @brief Add a bulk array from a scalar pointer and shape. The binary format extracts it to a
     * compressed field and leaves a {"$bulk": n, "shape": [...]} marker; the text format inlines it
     * as a nested JSON array. The shape is recoverable on read from the marker or the nested array.
     */
    template <typename T>
    void addBulk(const std::string &key, const T *data, const std::vector<RBigSize> &shape) {
        if (_context->packer) {
            RIndex ordinal = _context->next_bulk_index++;
            RSE_FastPackingCompressionMethod method =
                _context->is_compressing ? RSE_FAST_PACKING_COMPRESSION_METHOD_ZSTD
                                         : RSE_FAST_PACKING_COMPRESSION_METHOD_NONE;
            RSE_FastPackingCompressionFilter filter =
                _context->is_compressing ? RSE_FAST_PACKING_COMPRESSION_FILTER_SHUFFLE
                                         : RSE_FAST_PACKING_COMPRESSION_FILTER_NONE;
            _context->packer->addField(
                std::to_string(ordinal), _toBulkMemory(data, shape), method, filter
            );
            RS_JSON marker = RS_JSON::object();
            marker["$bulk"] = ordinal;
            marker["shape"] = shape;
            add(key, marker);
        } else {
            RBigSize offset = 0;
            add(key, _toBulkJson(data, shape, 0, offset));
        }
    }

    /**
     * @brief Whether an object is already pooled in the shared region.
     */
    [[nodiscard]] bool hasShared(const RSC_Base *rsc) const;

    /**
     * @brief Pool an object into the shared region, returning a writer for its data entry. Call
     * once, after hasShared is false. Throws when the object is already pooled.
     */
    RS_RSCFileTreeWriter addShared(const RSC_Base *rsc);

    /**
     * @brief Get the shared index of an already pooled object, or RS_NULL_INDEX when it is not.
     */
    [[nodiscard]] RIndex findShared(const RSC_Base *rsc) const;

   private:
    /// Whole-document state, shared by every cursor.
    RS_RSCFileTreeWriteContext *_context;

    /// Current node, a pointer into the context root.
    RS_JSON *_node;

    RS_RSCFileTreeWriter(RS_RSCFileTreeWriteContext *context, RS_JSON &node);

    /**
     * @brief Build a sub writer at node, sharing the context.
     */
    RS_RSCFileTreeWriter _sub(RS_JSON &node);

    /**
     * @brief Descend the dotted sub_node_path from the current node, creating objects, return leaf.
     */
    RS_JSON &_at(const std::string &sub_node_path);

    /**
     * @brief Wrap the array as a non-owning nd memory, mapping an enum element to its underlying
     * integer so the packer records a supported dtype.
     */
    template <typename T>
    static auto _toBulkMemory(const T *data, const std::vector<RBigSize> &shape) {
        using U = typename RS_RSCBulkScalar<T>::type;
        RBigSize count = 1;
        for (RBigSize dim : shape) count *= dim;
        RS_MemoryPtr weak(const_cast<void *>(static_cast<const void *>(data)), count * sizeof(T));
        return RS_NdMemory<U>(weak, shape, false);
    }

    /**
     * @brief Build a nested JSON array of the given shape, flat for rank one, from flat data.
     */
    template <typename T>
    static RS_JSON _toBulkJson(
        const T *data,
        const std::vector<RBigSize> &shape,
        RBigSize dimension,
        RBigSize &offset
    ) {
        RS_JSON array = RS_JSON::array();
        if (dimension + 1 == shape.size()) {
            for (RBigSize i = 0; i < shape[dimension]; ++i) array.push_back(data[offset++]);
        } else {
            for (RBigSize i = 0; i < shape[dimension]; ++i)
                array.push_back(_toBulkJson(data, shape, dimension + 1, offset));
        }
        return array;
    }
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_FILE_TREE_WRITER_H
