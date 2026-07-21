// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/recorder/rsc_recorder.h"

#include "rise_origin/rsc/rs_rsc_archive.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_RecorderHDF5::operator==(const RSC_RecorderHDF5 &other) const {
    return RSC_RecorderBase::operator==(other) && path == other.path;
}

void RSC_RecorderHDF5::_read(const RS_RSCFileTreeReader &tree) {
    RSC_RecorderBase::_read(tree);
    path = tree.get<string>("path");
}

void RSC_RecorderHDF5::_write(RS_RSCFileTreeWriter &tree) const {
    RSC_RecorderBase::_write(tree);
    tree.add("path", path);
}

void RSC_RecorderHDF5::_deserialize(RS_RSCArchive &archive) {
    RSC_RecorderBase::_deserialize(archive);
    archive.deserialize("path", path);
}

void RSC_RecorderHDF5::_serialize(RS_RSCArchive &archive) const {
    RSC_RecorderBase::_serialize(archive);
    archive.serialize("path", path);
}

}  // namespace rsc

}  // namespace rise
