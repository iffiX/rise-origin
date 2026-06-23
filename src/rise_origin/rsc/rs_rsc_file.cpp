// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_rsc_file.h"

#include <fstream>
#include <iterator>

#include "rise_origin/utils/string/rs_string.h"

namespace rise {

namespace rsc {

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

RSC_Base::RSCRef<RSC_Base> RS_RSCFile::loadAny(const std::string &path) {
    RS_JSON root = _decode(_readFile(path), _formatOfPath(path));
    RS_RSCPropertyTreeReader root_reader(root);
    RS_RSCFileHeader header;
    header.read(root_reader.getChild("header"));
    _gate(header);

    RS_JSON &body = root.at("body");
    if (header.rsc_version < currentVersion()) {
        _migrate(body, header.rsc_version);
    }

    RS_RSCPropertyTreeReader reader(body, header.rsc_version);
    return RS_RSCRegistry::getInstance().make(header.type, reader);
}

RSE_RSCFileFormat RS_RSCFile::_formatOfPath(const std::string &path) {
    if (endsWith(path, ".rsct")) return RSE_RSC_FILE_FORMAT_TEXT;
    if (endsWith(path, ".rscb")) return RSE_RSC_FILE_FORMAT_BINARY;
    RS_THROW(RS_InvalidArgumentException("RSC file path {} must end with .rsct or .rscb", path));
}

std::vector<uint8_t> RS_RSCFile::_encode(const RS_JSON &root, RSE_RSCFileFormat format) {
    if (format == RSE_RSC_FILE_FORMAT_TEXT) {
        std::string text = root.dump(2);
        return std::vector<uint8_t>(text.begin(), text.end());
    }
    return RS_JSON::to_msgpack(root);
}

RS_JSON RS_RSCFile::_decode(const std::vector<uint8_t> &bytes, RSE_RSCFileFormat format) {
    if (format == RSE_RSC_FILE_FORMAT_TEXT) {
        return RS_JSON::parse(bytes.begin(), bytes.end());
    }
    return RS_JSON::from_msgpack(bytes);
}

std::vector<uint8_t> RS_RSCFile::_readFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        RS_THROW(RS_DataException("Cannot open RSC file {}", path));
    }
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()
    );
}

void RS_RSCFile::_writeFile(const std::string &path, const std::vector<uint8_t> &bytes) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        RS_THROW(RS_DataException("Cannot write RSC file {}", path));
    }
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<RBigSize>(bytes.size()));
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
    RS_Version current = currentVersion();
    if (header.rsc_version.major != current.major || current < header.rsc_version) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Unsupported RSC version {}, engine writes {}",
            header.rsc_version.toString(),
            current.toString()
        ));
    }
}

void RS_RSCFile::_migrate(RS_JSON &body, const RS_Version &from) {
    // TODO: when a new schema version arrives, migrate here.
    (void)body;
    (void)from;
}

}  // namespace rsc

}  // namespace rise
