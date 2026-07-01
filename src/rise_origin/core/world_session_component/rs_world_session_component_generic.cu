// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <utility>

#include "rise_origin/core/rs_world_session_component.h"

namespace rise {
RS_WorldSessionComponentGeneric::RS_WorldSessionComponentGeneric(Callback callback)
    : _callback(std::move(callback)) {
}

std::shared_ptr<RS_WorldSessionHandle> RS_WorldSessionComponentGeneric::getHandle() const {
    return _access ? _access->handle() : nullptr;
}

void RS_WorldSessionComponentGeneric::setCallback(Callback callback) {
    _callback = std::move(callback);
}

void RS_WorldSessionComponentGeneric::onAttach(RS_WorldSessionAccess access) {
    _access = std::move(access);
}

void RS_WorldSessionComponentGeneric::onCallback(RBigFloat time, RS_ThreadPool& pool) {
    if (!_callback) return;
    pool.detachTask([this, time] { _callback(time, _access->controllers()); });
}

void RS_WorldSessionComponentGeneric::onDetach() {
}
} // namespace rise