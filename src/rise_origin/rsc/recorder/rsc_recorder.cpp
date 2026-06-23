// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/recorder/rsc_recorder.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_RecorderHDF5::operator==(const RSC_RecorderHDF5 &other) const {
    return RSC_RecorderBase::operator==(other) && path == other.path;
}

void RSC_RecorderHDF5::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_RecorderBase::_read(tree);
    path = tree.get<string>("path");
}

void RSC_RecorderHDF5::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_RecorderBase::_write(tree);
    tree.add("path", path);
}

}  // namespace rsc

}  // namespace rise
