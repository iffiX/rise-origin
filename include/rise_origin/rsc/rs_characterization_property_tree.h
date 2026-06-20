// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CHARACTERIZATION_PROPERTY_TREE_H
#define RS_CHARACTERIZATION_PROPERTY_TREE_H

#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <unordered_set>
#include <vector>

#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/string/rs_format.h"
#include "rise_origin/utils/rs_logging.h"
#include "rise_origin/utils/string/rs_string.h"

namespace rise {

namespace rsc {

class RS_CharacterizationPropertyTree {
   public:
    /// Path to current node in the tree, each string is a key
    std::vector<std::string> path;

    /// Root of the tree (shared to avoid deep copies)
    std::shared_ptr<const boost::property_tree::ptree> root;

    /// Current node (pointer into the shared root tree)
    const boost::property_tree::ptree *current_node;

    /// Logger
    std::shared_ptr<logging::logger> logger;

    /// @brief Init as root node of the tree.
    RS_CharacterizationPropertyTree(const boost::property_tree::ptree &root)
        : root(std::make_shared<const boost::property_tree::ptree>(root)),
          current_node(this->root.get()) {
        if (logging::get("root")) logger = logging::get("root")->clone("rsc");
    }

    /// @brief Init as a sub node of the tree.
    RS_CharacterizationPropertyTree(
        std::vector<std::string> path,
        std::shared_ptr<const boost::property_tree::ptree> root,
        const boost::property_tree::ptree &current_node
    )
        : path(std::move(path)), root(std::move(root)), current_node(&current_node) {
        if (logging::get("root")) logger = logging::get("root")->clone("rsc");
    }

    /// @brief Reset the tree and move to root.
    void reset() {
        path.clear();
        current_node = root.get();
    }

    /// @brief Get a value as type T under specified sub_node_path from the tree.
    template <typename T>
    [[nodiscard]] T get(const std::string &sub_node_path) const {
        if (current_node->get_child_optional(sub_node_path))
            return current_node->get<T>(sub_node_path);
        else {
            std::string full_path = join(".", path);
            RS_THROW(RS_InvalidCharacterizationException(
                "Failed to find sub_node_path {} under path {}", sub_node_path, full_path
            ));
        }
    }

    /// @brief Get a value as type T under specified sub_node_path from the tree,
    /// if not found, use default value.
    template <typename T>
    [[nodiscard]] T get(const std::string &sub_node_path, T default_value) const {
        if (current_node->get_child_optional(sub_node_path))
            return current_node->get<T>(sub_node_path);
        else {
            std::string full_path = join(".", path);
            if (logger) {
                logger->trace(
                    "Failed to find sub_node_path {} under path {}, "
                    "using default value",
                    sub_node_path,
                    full_path,
                    default_value
                );
            }
            return default_value;
        }
    }

    /// @brief Get a sub tree under specified sub_node_path from the tree.
    [[nodiscard]] auto getChild(const std::string &sub_node_path) const {
        std::string full_path = join(".", path);
        auto new_path = path;
        new_path.push_back(sub_node_path);
        if (current_node->get_child_optional(sub_node_path))
            return RS_CharacterizationPropertyTree(
                new_path, root, current_node->get_child(sub_node_path)
            );
        else {
            RS_THROW(RS_InvalidCharacterizationException(
                "Failed to find sub tree {} under path {}", sub_node_path, full_path
            ));
        }
    }

    /// @brief Get all sub trees from the current tree node.
    [[nodiscard]] auto getChildren() const {
        std::vector<std::pair<std::string, RS_CharacterizationPropertyTree>> children;
        for (auto &c : current_node->get_child("")) {
            if (c.first[0] == '<') continue;
            auto new_path = path;
            new_path.push_back(c.first);
            children.push_back({c.first, {new_path, root, c.second}});
        }
        return children;
    }

    /// @brief Get all child keys from the current tree node.
    [[nodiscard]] auto getChildrenKeys() const {
        std::vector<std::string> keys;
        for (auto &c : current_node->get_child("")) {
            if (c.first[0] == '<') continue;
            keys.push_back(c.first);
        }
        return keys;
    }

    /// @brief Get all child values from the current tree node.
    [[nodiscard]] auto getChildrenValues() const {
        std::vector<RS_CharacterizationPropertyTree> values;
        for (auto &c : current_node->get_child("")) {
            auto new_path = path;
            if (c.first[0] == '<') continue;
            new_path.push_back(c.first);
            values.emplace_back(new_path, root, c.second);
        }
        return values;
    }

    /// @brief Check if the tree has a child under specified key.
    [[nodiscard]] bool isChildExist(const std::string &key) const {
        return current_node->get_child_optional(key).operator bool();
    }

    /// @brief Check if the tree has the exact same child keys as specified.
    [[nodiscard]] bool isChildrenSameAsRequired(const std::vector<std::string> &keys) const {
        std::unordered_set required_keys(keys.begin(), keys.end());
        std::unordered_set<std::string> found_keys;
        for (auto &c : current_node->get_child("")) {
            if (required_keys.find(c.first) == required_keys.end()) return false;
            found_keys.insert(c.first);
        }
        return required_keys == found_keys;
    }

    /// @brief Check if the tree has the required child keys as specified, extra keys are ignored.
    [[nodiscard]] bool isChildrenFound(const std::vector<std::string> &keys) const {
        std::unordered_set required_keys(keys.begin(), keys.end());
        std::unordered_set<std::string> found_keys;
        for (auto &c : current_node->get_child("")) {
            if (required_keys.find(c.first) == required_keys.end()) continue;
            found_keys.insert(c.first);
        };
        return required_keys == found_keys;
    }
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_CHARACTERIZATION_PROPERTY_TREE_H
