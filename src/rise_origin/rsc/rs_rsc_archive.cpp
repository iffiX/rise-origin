// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_rsc_archive.h"

namespace rise {

namespace rsc {

RSC_Base::RSCRef<RSC_Base> RS_RSCArchive::fromBytesAny(RS_BytesView bytes) {
    RS_RSCArchive archive(Mode::ReadBytes);
    archive._fast_deserializer.setData(bytes);
    return archive._deserializeGraph();
}

RS_Bytes RS_RSCArchive::toBytesAny(const RSC_Base &node) {
    RS_RSCArchive archive(Mode::WriteBytes);
    std::shared_ptr<RSC_Base> base(std::shared_ptr<void>(), const_cast<RSC_Base *>(&node));
    archive._serializeGraph(base);
    RS_BytesView view = archive._fast_serializer.getData();
    return RS_Bytes(view.ptr(), view.size());
}

RSC_Base::RSCRef<RSC_Base> RS_RSCArchive::fromJSONAny(const RS_JSON &json) {
    RS_RSCArchive archive(Mode::ReadJSON);
    archive._json = json;
    return archive._deserializeGraph();
}

RS_JSON RS_RSCArchive::toJSONAny(const RSC_Base &node) {
    RS_RSCArchive archive(Mode::WriteJSON);
    std::shared_ptr<RSC_Base> base(std::shared_ptr<void>(), const_cast<RSC_Base *>(&node));
    archive._serializeGraph(base);
    return std::move(archive._json);
}

bool RS_RSCArchive::isSerialize() const {
    return _mode == Mode::WriteBytes || _mode == Mode::WriteJSON;
}

RSC_Base::RSCRef<RSC_Base> RS_RSCArchive::_deserializeGraph() {
    if (_mode == Mode::ReadBytes) {
        uint64_t count = 0;
        _fast_deserializer("count", count);
        std::vector<std::string> names(static_cast<std::size_t>(count));
        for (uint64_t i = 0; i < count; i++) _fast_deserializer("type", names[i]);
        for (uint64_t i = 0; i < count; i++) {
            std::shared_ptr<RSC_Base> shell = RS_RSCRegistry::getExample(names[i]).shared();
            _shared.push_back(shell);
            _index_to_shared_map[static_cast<RIndex>(i)] = shell;
        }
        for (uint64_t i = 0; i < count; i++) _shared[i]->deserialize(*this);
        return _shared.empty() ? RSC_Base::RSCRef<RSC_Base>() : _shared[0];
    }

    RS_JSON &shared_array = _getChild(_json, "shared");
    for (std::size_t i = 0; i < shared_array.size(); i++) {
        std::string type = _getChild(shared_array[i], RSC_Base::VAR_TYPE_KEY).get<std::string>();
        std::shared_ptr<RSC_Base> shell = RS_RSCRegistry::getExample(type).shared();
        _shared.push_back(shell);
        _index_to_shared_map[static_cast<RIndex>(i)] = shell;
    }
    for (std::size_t i = 0; i < shared_array.size(); i++) {
        _json_node = &_getChild(shared_array[i], RSC_Base::VAR_DATA_KEY);
        _shared[i]->deserialize(*this);
    }
    return getSharedAt(_getChild(_json, "base").get<RIndex>());
}

void RS_RSCArchive::_serializeGraph(const std::shared_ptr<RSC_Base> &base) {
    _is_collecting = true;
    _collectShared(base);
    _is_collecting = false;

    if (_mode == Mode::WriteBytes) {
        _fast_serializer("count", static_cast<uint64_t>(_shared.size()));
        for (const auto &node : _shared)
            _fast_serializer("type", RS_RSCRegistry::getName(typeid(*node)));
        for (const auto &node : _shared) node->serialize(*this);
        return;
    }

    RS_JSON shared_array = RS_JSON::array();
    for (const auto &node : _shared) {
        RS_JSON entry = RS_JSON::object();
        RS_JSON data = RS_JSON::object();
        _json_node = &data;
        node->serialize(*this);
        entry[RSC_Base::VAR_TYPE_KEY] = RS_RSCRegistry::getName(typeid(*node));
        entry[RSC_Base::VAR_DATA_KEY] = std::move(data);
        shared_array.push_back(std::move(entry));
    }
    _json = RS_JSON::object();
    _json["shared"] = std::move(shared_array);
    _json["base"] = 0;
}

void RS_RSCArchive::_collectShared(const std::shared_ptr<RSC_Base> &node) {
    if (node == nullptr || _shared_to_index_map.count(node.get()) > 0) return;
    _shared_to_index_map[node.get()] = static_cast<RIndex>(_shared.size());
    _shared.push_back(node);
    node->serialize(*this);
}

RIndex RS_RSCArchive::_deserializeIndex(std::string_view key) {
    if (_mode == Mode::ReadBytes) {
        uint64_t index = 0;
        _fast_deserializer("index", index);
        return static_cast<RIndex>(index);
    }
    return _getChild(*_json_node, key).get<RIndex>();
}

void RS_RSCArchive::_serializeIndex(std::string_view key, RIndex index) {
    if (_mode == Mode::WriteBytes) {
        _fast_serializer("index", static_cast<uint64_t>(index));
    } else {
        (*_json_node)[std::string(key)] = index;
    }
}

std::shared_ptr<RSC_Base> RS_RSCArchive::getSharedAt(RIndex index) const {
    auto found = _index_to_shared_map.find(index);
    if (found == _index_to_shared_map.end()) {
        RS_THROW(RS_RSCArchiveException("RSC archive references unknown shared index {}", index));
    }
    return found->second;
}

RS_JSON &RS_RSCArchive::_getChild(RS_JSON &node, std::string_view key) {
    auto found = node.find(std::string(key));
    if (found == node.end()) {
        RS_THROW(RS_RSCArchiveException("RSC archive is missing key {}", std::string(key)));
    }
    return *found;
}

}  // namespace rsc

}  // namespace rise
