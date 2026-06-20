// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/recorder/rsc_recorder.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_RecorderHDF5::operator==(const RSC_RecorderHDF5 &other) const {
    return RSC_RecorderBase::operator==(other) && path == other.path;
}

void RSC_RecorderHDF5::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_RecorderBase::_read(rsc_tree);
    path = rsc_tree.get<string>("Path");
}

}  // namespace rsc

}  // namespace rise
