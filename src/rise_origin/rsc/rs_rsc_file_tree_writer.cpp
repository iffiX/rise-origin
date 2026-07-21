// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_rsc_file_tree_writer.h"

#include <string>

#include "rise_origin/rsc/rs_rsc_registry.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/string/rs_string.h"

namespace rise {

namespace rsc {

RS_RSCFileTreeWriter::RS_RSCFileTreeWriter(RS_RSCFileTreeWriteContext &context)
    : _context(&context), _node(&context.root) {}

RS_Version RS_RSCFileTreeWriter::version() const { return _context->version; }

RS_RSCFileTreeWriter RS_RSCFileTreeWriter::addChild(const std::string &sub_node_path) {
    RS_JSON &node = _at(sub_node_path);
    node = RS_JSON::object();
    return _sub(node);
}

RS_RSCFileTreeWriter RS_RSCFileTreeWriter::addArray(const std::string &sub_node_path) {
    RS_JSON &node = _at(sub_node_path);
    node = RS_JSON::array();
    return _sub(node);
}

RS_RSCFileTreeWriter RS_RSCFileTreeWriter::add() {
    _node->push_back(RS_JSON::object());
    return _sub(_node->back());
}

bool RS_RSCFileTreeWriter::hasShared(const RSC_Base *rsc) const {
    return _context->shared_map.count(rsc) != 0;
}

RS_RSCFileTreeWriter RS_RSCFileTreeWriter::addShared(const RSC_Base *rsc) {
    if (_context->shared_map.count(rsc) != 0) {
        RS_THROW(RS_InvalidCharacterizationException("Shared object is already pooled"));
    }
    RIndex index = _context->next_shared_index++;
    _context->shared_map.emplace(rsc, index);
    RS_JSON &shared = _context->root["shared"];
    if (!shared.is_object()) shared = RS_JSON::object();
    RS_JSON &entry = shared[std::to_string(index)];
    entry = RS_JSON::object();
    entry[RSC_Base::VAR_TYPE_KEY] = RS_RSCRegistry::getName(typeid(*rsc));
    RS_JSON &data = entry[RSC_Base::VAR_DATA_KEY];
    data = RS_JSON::object();
    return _sub(data);
}

RIndex RS_RSCFileTreeWriter::findShared(const RSC_Base *rsc) const {
    auto it = _context->shared_map.find(rsc);
    return it == _context->shared_map.end() ? RS_NULL_INDEX : it->second;
}

RS_RSCFileTreeWriter::RS_RSCFileTreeWriter(RS_RSCFileTreeWriteContext *context, RS_JSON &node)
    : _context(context), _node(&node) {}

RS_RSCFileTreeWriter RS_RSCFileTreeWriter::_sub(RS_JSON &node) {
    return RS_RSCFileTreeWriter(_context, node);
}

RS_JSON &RS_RSCFileTreeWriter::_at(const std::string &sub_node_path) {
    RS_JSON *node = _node;
    for (auto &&key : splitToView(sub_node_path, ".")) {
        if (!node->is_object()) *node = RS_JSON::object();
        node = &(*node)[std::string(key)];
    }
    return *node;
}

}  // namespace rsc

}  // namespace rise
