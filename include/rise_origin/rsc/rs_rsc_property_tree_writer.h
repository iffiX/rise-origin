// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_PROPERTY_TREE_WRITER_H
#define RS_RSC_PROPERTY_TREE_WRITER_H

#include <memory>
#include <string>
#include <utility>

#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/string/rs_string.h"
#include "rise_origin/utils/version/rs_version.h"

namespace rise {

namespace rsc {

/**
 * @brief Write side of the durable RSC tree, building an RS_JSON document.
 */
class RS_RSCPropertyTreeWriter {
   public:
    /// Shared document root being built.
    std::shared_ptr<RS_JSON> root;

    /// Current node, a pointer into the shared root.
    RS_JSON *current_node;

    /**
     * @brief Init as the root node, carrying the schema version being written.
     */
    explicit RS_RSCPropertyTreeWriter(RS_Version version = {})
        : root(std::make_shared<RS_JSON>(RS_JSON::object())),
          current_node(root.get()),
          _version(version) {}

    /**
     * @brief Init as a sub node of the document.
     */
    RS_RSCPropertyTreeWriter(
        std::shared_ptr<RS_JSON> root,
        RS_JSON &current_node,
        RS_Version version
    )
        : root(std::move(root)), current_node(&current_node), _version(version) {}

    /**
     * @brief Get the schema version this tree is written as.
     */
    [[nodiscard]] RS_Version version() const { return _version; }

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
    RS_RSCPropertyTreeWriter addChild(const std::string &sub_node_path) {
        RS_JSON &node = _at(sub_node_path);
        node = RS_JSON::object();
        return RS_RSCPropertyTreeWriter(root, node, _version);
    }

    /**
     * @brief Add an array under sub_node_path and return its writer.
     */
    RS_RSCPropertyTreeWriter addArray(const std::string &sub_node_path) {
        RS_JSON &node = _at(sub_node_path);
        node = RS_JSON::array();
        return RS_RSCPropertyTreeWriter(root, node, _version);
    }

    /**
     * @brief Append an object element to the current array and return its writer.
     */
    RS_RSCPropertyTreeWriter add() {
        current_node->push_back(RS_JSON::object());
        return RS_RSCPropertyTreeWriter(root, current_node->back(), _version);
    }

   private:
    /// Schema version this tree is written as.
    RS_Version _version;

    /**
     * @brief Descend the dotted sub_node_path from the current node, creating objects, return leaf.
     */
    RS_JSON &_at(const std::string &sub_node_path) {
        RS_JSON *node = current_node;
        for (auto &&key : splitToView(sub_node_path, ".")) {
            if (!node->is_object()) *node = RS_JSON::object();
            node = &(*node)[std::string(key)];
        }
        return *node;
    }
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_PROPERTY_TREE_WRITER_H
