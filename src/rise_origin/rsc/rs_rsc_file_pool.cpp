// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_rsc_file_pool.h"

#include "rise_origin/rsc/rs_rsc_file.h"

namespace rise {

namespace rsc {

RS_RSCFilePool &RS_RSCFilePool::getInstance() {
    static RS_RSCFilePool instance;
    return instance;
}

RS_FastPacker RS_RSCFilePool::getPacker() { return getInstance()._getPacker(); }

void RS_RSCFilePool::pack(const RS_FastPacker &packer, const std::string &path) {
    packer.pack(path, RS_RSCB_MAGIC);
}

RS_FastUnpacker RS_RSCFilePool::getUnpacker(const std::string &path) {
    return getInstance()._getUnpacker(path);
}

RS_FastPacker RS_RSCFilePool::_getPacker() {
    return RS_FastPacker(&RS_MemoryDefaultProvider::getInstance(), &_pool);
}

RS_FastUnpacker RS_RSCFilePool::_getUnpacker(const std::string &path) {
    RS_FastUnpacker unpacker(&RS_MemoryDefaultProvider::getInstance(), &_pool);
    unpacker.unpack(path, RS_RSCB_MAGIC);
    return unpacker;
}

}  // namespace rsc

}  // namespace rise
