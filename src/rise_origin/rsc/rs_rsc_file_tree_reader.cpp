// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_rsc_file_tree_reader.h"

#include <string>
#include <utility>
#include <vector>

#include "rise_origin/rsc/rs_rsc_registry.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/string/rs_string.h"

namespace rise {

namespace rsc {

RS_RSCFileTreeReader::RS_RSCFileTreeReader(RS_RSCFileTreeReadContext &context)
    : _context(&context), _node(&context.root) {}

RS_Version RS_RSCFileTreeReader::version() const { return _context->version; }

bool RS_RSCFileTreeReader::hasChild(const std::string &key) const {
    return _resolve(key) != nullptr;
}

RS_RSCFileTreeReader RS_RSCFileTreeReader::getChild(const std::string &sub_node_path) const {
    const RS_JSON *node = _resolve(sub_node_path);
    auto new_path = _path;
    for (auto &&key : splitToView(sub_node_path, ".")) new_path.emplace_back(key);
    if (node) {
        return RS_RSCFileTreeReader(_context, *node, std::move(new_path));
    }
    RS_THROW(RS_InvalidCharacterizationException(
        "Failed to find sub tree {} under path {}", sub_node_path, join(".", _path)
    ));
}

std::vector<std::pair<std::string, RS_RSCFileTreeReader>>
RS_RSCFileTreeReader::getChildren() const {
    std::vector<std::pair<std::string, RS_RSCFileTreeReader>> children;
    if (!_node->is_object()) return children;
    for (auto &item : _node->items()) {
        auto new_path = _path;
        new_path.push_back(item.key());
        children.emplace_back(
            item.key(), RS_RSCFileTreeReader(_context, item.value(), std::move(new_path))
        );
    }
    return children;
}

std::vector<RS_RSCFileTreeReader> RS_RSCFileTreeReader::getArray(
    const std::string &sub_node_path
) const {
    std::vector<RS_RSCFileTreeReader> elements;
    const RS_JSON *node = _resolve(sub_node_path);
    if (!node || !node->is_array()) return elements;
    RSize index = 0;
    for (auto &element : *node) {
        auto new_path = _path;
        new_path.push_back(sub_node_path + "[" + std::to_string(index++) + "]");
        elements.push_back(RS_RSCFileTreeReader(_context, element, std::move(new_path)));
    }
    return elements;
}

std::vector<RBigSize> RS_RSCFileTreeReader::getBulkShape(const std::string &key) const {
    std::vector<RBigSize> shape;
    const RS_JSON *node = _resolve(key);
    if (!node) return shape;
    if (node->is_object() && node->contains("$bulk")) {
        return node->at("shape").get<std::vector<RBigSize>>();
    }
    for (const RS_JSON *cursor = node; cursor->is_array(); cursor = &cursor->front()) {
        shape.push_back(cursor->size());
        if (cursor->empty()) break;
    }
    return shape;
}

bool RS_RSCFileTreeReader::hasShared(RIndex index) const {
    return _context->shared_map.count(index) != 0;
}

std::shared_ptr<RSC_Base> RS_RSCFileTreeReader::findShared(RIndex index) const {
    auto it = _context->shared_map.find(index);
    return it == _context->shared_map.end() ? nullptr : it->second;
}

RS_RSCFileTreeReader::RS_RSCFileTreeReader(
    RS_RSCFileTreeReadContext *context,
    const RS_JSON &node,
    std::vector<std::string> path
)
    : _context(context), _node(&node), _path(std::move(path)) {}

const RS_JSON *RS_RSCFileTreeReader::_resolve(const std::string &sub_node_path) const {
    const RS_JSON *node = _node;
    for (auto &&key : splitToView(sub_node_path, ".")) {
        if (!node->is_object()) return nullptr;
        auto it = node->find(std::string(key));
        if (it == node->end()) return nullptr;
        node = &(*it);
    }
    return node;
}

RS_RSCFileTreeReader
RS_RSCFileTreeReader::_getSharedEntry(RIndex index, const std::type_info &expected) const {
    auto shared_it = _context->root.find("shared");
    if (shared_it != _context->root.end()) {
        auto entry_it = shared_it->find(std::to_string(index));
        if (entry_it != shared_it->end() && entry_it->contains(RSC_Base::VAR_TYPE_KEY) &&
            entry_it->contains(RSC_Base::VAR_DATA_KEY)) {
            const std::string &expected_name = RS_RSCRegistry::getName(expected);
            std::string stored = entry_it->at(RSC_Base::VAR_TYPE_KEY).get<std::string>();
            if (stored != expected_name) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Shared entry {} has type {} but {} was expected", index, stored, expected_name
                ));
            }
            std::vector<std::string> new_path = {
                "shared", std::to_string(index), RSC_Base::VAR_DATA_KEY
            };
            return RS_RSCFileTreeReader(
                _context, entry_it->at(RSC_Base::VAR_DATA_KEY), std::move(new_path)
            );
        }
    }
    RS_THROW(RS_InvalidCharacterizationException(
        "Missing shared entry {} referenced by a $shared marker", index
    ));
}

}  // namespace rsc

}  // namespace rise
