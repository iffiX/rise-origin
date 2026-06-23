// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_PROPERTY_TREE_READER_H
#define RS_RSC_PROPERTY_TREE_READER_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/string/rs_string.h"
#include "rise_origin/utils/version/rs_version.h"

namespace rise {

namespace rsc {

/**
 * @brief Read side of the durable RSC tree, backed by an RS_JSON document.
 */
class RS_RSCPropertyTreeReader {
   public:
    /// Path to the current node, kept for error messages.
    std::vector<std::string> path;

    /// Shared document root, so sub readers avoid copying it.
    std::shared_ptr<const RS_JSON> root;

    /// Current node, a pointer into the shared root.
    const RS_JSON *current_node;

    /**
     * @brief Init as the root node, carrying the file's schema version.
     */
    explicit RS_RSCPropertyTreeReader(const RS_JSON &root, RS_Version version = {})
        : root(std::make_shared<const RS_JSON>(root)),
          current_node(this->root.get()),
          _version(version) {}

    /**
     * @brief Init as a sub node of the document.
     */
    RS_RSCPropertyTreeReader(
        std::vector<std::string> path,
        std::shared_ptr<const RS_JSON> root,
        const RS_JSON &current_node,
        RS_Version version
    )
        : path(std::move(path)),
          root(std::move(root)),
          current_node(&current_node),
          _version(version) {}

    /**
     * @brief Get the file schema version this tree was read as.
     */
    [[nodiscard]] RS_Version version() const { return _version; }

    /**
     * @brief Get a value of type T under sub_node_path, throwing when it is absent.
     */
    template <typename T>
    [[nodiscard]] T get(const std::string &sub_node_path) const {
        if (const RS_JSON *node = _resolve(sub_node_path)) {
            return node->get<T>();
        }
        RS_THROW(RS_InvalidCharacterizationException(
            "Failed to find sub_node_path {} under path {}", sub_node_path, join(".", path)
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
     * @brief Get a sub tree under sub_node_path, throwing when it is absent.
     */
    [[nodiscard]] RS_RSCPropertyTreeReader getChild(const std::string &sub_node_path) const {
        const RS_JSON *node = _resolve(sub_node_path);
        auto new_path = path;
        for (auto &&key : splitToView(sub_node_path, ".")) new_path.emplace_back(key);
        if (node) {
            return RS_RSCPropertyTreeReader(std::move(new_path), root, *node, _version);
        }
        RS_THROW(RS_InvalidCharacterizationException(
            "Failed to find sub tree {} under path {}", sub_node_path, join(".", path)
        ));
    }

    /**
     * @brief Get all object members of the current node as key and sub tree pairs.
     */
    [[nodiscard]] std::vector<std::pair<std::string, RS_RSCPropertyTreeReader>>
    getChildren() const {
        std::vector<std::pair<std::string, RS_RSCPropertyTreeReader>> children;
        if (!current_node->is_object()) return children;
        for (auto &item : current_node->items()) {
            auto new_path = path;
            new_path.push_back(item.key());
            children.emplace_back(
                item.key(), RS_RSCPropertyTreeReader(new_path, root, item.value(), _version)
            );
        }
        return children;
    }

    /**
     * @brief Get every element of the array under sub_node_path, empty when absent.
     */
    [[nodiscard]] std::vector<RS_RSCPropertyTreeReader> getArray(
        const std::string &sub_node_path
    ) const {
        std::vector<RS_RSCPropertyTreeReader> elements;
        const RS_JSON *node = _resolve(sub_node_path);
        if (!node || !node->is_array()) return elements;
        RSize index = 0;
        for (auto &element : *node) {
            auto new_path = path;
            new_path.push_back(sub_node_path + "[" + std::to_string(index++) + "]");
            elements.emplace_back(std::move(new_path), root, element, _version);
        }
        return elements;
    }

    /**
     * @brief Whether a node exists under key.
     */
    [[nodiscard]] bool isChildExist(const std::string &key) const {
        return _resolve(key) != nullptr;
    }

   private:
    /// File schema version this tree was read as.
    RS_Version _version;

    /**
     * @brief Walk the dotted sub_node_path from the current node, null when a step is missing.
     */
    [[nodiscard]] const RS_JSON *_resolve(const std::string &sub_node_path) const {
        const RS_JSON *node = current_node;
        for (auto &&key : splitToView(sub_node_path, ".")) {
            if (!node->is_object()) return nullptr;
            auto it = node->find(std::string(key));
            if (it == node->end()) return nullptr;
            node = &(*it);
        }
        return node;
    }
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_PROPERTY_TREE_READER_H
