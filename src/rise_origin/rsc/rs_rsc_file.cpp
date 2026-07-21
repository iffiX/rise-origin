// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_rsc_file.h"

#include <fstream>

#include "rise_origin/rsc/rs_rsc_file_pool.h"
#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/string/rs_string.h"

namespace rise {

namespace rsc {

void RS_RSCFile::saveAny(const RSC_Base &node, const std::string &path, bool is_compressing) {
    RSE_RSCFileFormat format = _formatOfPath(path);
    RS_RSCFileTreeWriteContext context;
    context.version = currentVersion();
    context.is_compressing = is_compressing;
    if (format == RSE_RSC_FILE_FORMAT_BINARY) {
        context.packer = RS_RSCFilePool::getPacker();
    }

    RS_RSCFileTreeWriter writer(context);
    RS_RSCFileHeader header;
    header.magic = "RSC";
    header.edition = RS_RSC_EDITION;
    header.rsc_version = currentVersion();
    header.engine_version = engineVersion();
    header.type = RS_RSCRegistry::getName(typeid(node));
    header.write(writer.addChild("header"));
    node.write(writer.addChild("body"));

    _commit(context, path, format);
}

RSC_Base::RSCRef<RSC_Base> RS_RSCFile::loadAny(const std::string &path) {
    RS_RSCFileTreeReadContext context;
    RS_RSCFileHeader header = _open(context, path);
    RS_RSCFileTreeReader reader(context);
    return RS_RSCRegistry::make(header.type, reader.getChild("body"));
}

RS_Version RS_RSCFile::currentVersion() {
    RS_Version version;
    version.fromString(RS_RSC_VERSION);
    return version;
}

RS_Version RS_RSCFile::engineVersion() {
    RS_Version version;
    version.fromString(RS_ENGINE_VERSION);
    return version;
}

RSE_RSCFileFormat RS_RSCFile::_formatOfPath(const std::string &path) {
    if (endsWith(path, ".rsct")) return RSE_RSC_FILE_FORMAT_TEXT;
    if (endsWith(path, ".rscb")) return RSE_RSC_FILE_FORMAT_BINARY;
    RS_THROW(RS_InvalidArgumentException("RSC file path {} must end with .rsct or .rscb", path));
}

RS_RSCFileHeader RS_RSCFile::_open(RS_RSCFileTreeReadContext &context, const std::string &path) {
    RSE_RSCFileFormat format = _formatOfPath(path);
    if (format == RSE_RSC_FILE_FORMAT_TEXT) {
        context.root = _readTextFile(path);
    } else {
        context.root = _readBinaryFile(path, context);
    }

    RS_RSCFileTreeReader reader(context);
    RS_RSCFileHeader header;
    header.read(reader.getChild("header"));
    _gate(header);
    context.version = header.rsc_version;
    if (header.rsc_version < currentVersion()) {
        _migrate(context.root.at("body"), header.rsc_version);
    }
    return header;
}

void RS_RSCFile::_gate(const RS_RSCFileHeader &header) {
    if (header.magic != "RSC") {
        RS_THROW(RS_InvalidCharacterizationException("Not an RSC file"));
    }
    if (header.edition != RS_RSC_EDITION) {
        RS_THROW(RS_InvalidCharacterizationException(
            "RSC edition mismatch: file is {}, engine is {}", header.edition, RS_RSC_EDITION
        ));
    }
}

void RS_RSCFile::_migrate(RS_JSON &body, const RS_Version &from) {
    // TODO: when a new schema version arrives, migrate here.
    (void)body;
    (void)from;
}

void RS_RSCFile::_commit(
    RS_RSCFileTreeWriteContext &context,
    const std::string &path,
    RSE_RSCFileFormat format
) {
    if (format == RSE_RSC_FILE_FORMAT_TEXT) {
        _writeTextFile(path, context.root);
    } else {
        _writeBinaryFile(path, context);
    }
}

RS_JSON RS_RSCFile::_readTextFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        RS_THROW(RS_DataException("Cannot open RSC file {}", path));
    }
    return RS_JSON::parse(file);
}

void RS_RSCFile::_writeTextFile(const std::string &path, const RS_JSON &root) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        RS_THROW(RS_DataException("Cannot write RSC file {}", path));
    }
    file << root.dump(2);
}

RS_JSON RS_RSCFile::_readBinaryFile(const std::string &path, RS_RSCFileTreeReadContext &context) {
    context.unpacker = RS_RSCFilePool::getUnpacker(path);
    return context.unpacker->getMetaData();
}

void RS_RSCFile::_writeBinaryFile(const std::string &path, RS_RSCFileTreeWriteContext &context) {
    RSE_FastPackingCompressionMethod meta_method = context.is_compressing
                                                       ? RSE_FAST_PACKING_COMPRESSION_METHOD_ZSTD
                                                       : RSE_FAST_PACKING_COMPRESSION_METHOD_NONE;
    context.packer->setMetaData(context.root, meta_method);
    RS_RSCFilePool::pack(*context.packer, path);
}

}  // namespace rsc

}  // namespace rise
