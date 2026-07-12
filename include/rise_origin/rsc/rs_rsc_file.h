// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_FILE_H
#define RS_RSC_FILE_H

#include <string>
#include <typeinfo>
#include <vector>

#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/version/rs_version.h"
#include "rs_rsc_property_tree_reader.h"
#include "rs_rsc_property_tree_writer.h"
#include "rs_rsc_registry.h"
#include "rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief On disk format of an RSC file, chosen by file extension.
 */
RS_ENUM(RSE_RSCFileFormat, uint8_t, (RSE_RSC_FILE_FORMAT_TEXT, 0), (RSE_RSC_FILE_FORMAT_BINARY, 1))

/**
 * @brief Identity and version fields at the head of every RSC file.
 */
struct RS_RSCFileHeader {
    /// Magic guard, always "RSC".
    std::string magic;

    /// RSC edition that wrote the file.
    std::string edition;

    /// On disk schema version, drives migration.
    RS_Version rsc_version;

    /// Engine build that wrote the file.
    RS_Version engine_version;

    /// Registered name of the root node type.
    std::string type;

    /**
     * @brief Read the header fields from a reader.
     */
    void read(const RS_RSCPropertyTreeReader &tree) {
        magic = tree.get<std::string>("magic");
        edition = tree.get<std::string>("edition");
        rsc_version.fromString(tree.get<std::string>("rsc_version"));
        engine_version.fromString(tree.get<std::string>("engine_version"));
        type = tree.get<std::string>("type");
    }

    /**
     * @brief Write the header fields to a writer.
     */
    void write(RS_RSCPropertyTreeWriter tree) const {
        tree.add("magic", magic);
        tree.add("edition", edition);
        tree.add("rsc_version", rsc_version.toString());
        tree.add("engine_version", engine_version.toString());
        tree.add("type", type);
    }
};

/**
 * @brief Durable save and load of any RSC node, in the text or binary format.
 * @details Holds a node as { header, body }, writes identity and version, and checks and
 * migrates on load. The format is picked from the path extension, .rsct for text and .rscb
 * for binary.
 */
class RS_RSCFile {
   public:
    /**
     * @brief Save a node to path, format chosen by extension.
     */
    template <typename T>
    static void save(const T &node, const std::string &path) {
        _writeFile(path, _toBytes(node, _formatOfPath(path)));
    }

    /**
     * @brief Load a node of type T from path, verifying the file holds that type.
     */
    template <typename T>
    static T load(const std::string &path) {
        return _fromBytes<T>(_readFile(path), _formatOfPath(path));
    }

    /**
     * @brief Save a node to path.
     */
    static void saveAny(const RSC_Base &node, const std::string &path);

    /**
     * @brief Load a node from path.
     */
    static RSC_Base::RSCRef<RSC_Base> loadAny(const std::string &path);

    /**
     * @brief Get the schema version this build writes.
     */
    static RS_Version currentVersion();

    /**
     * @brief Get the engine build version.
     */
    static RS_Version engineVersion();

   private:
    /**
     * @brief Choose the format from a path extension, throwing on an unknown one.
     */
    static RSE_RSCFileFormat _formatOfPath(const std::string &path);

    /**
     * @brief Encode a document to bytes in the given format.
     */
    static std::vector<uint8_t> _encode(const RS_JSON &root, RSE_RSCFileFormat format);

    /**
     * @brief Decode a document from bytes in the given format.
     */
    static RS_JSON _decode(const std::vector<uint8_t> &bytes, RSE_RSCFileFormat format);

    /**
     * @brief Read a whole file as bytes.
     */
    static std::vector<uint8_t> _readFile(const std::string &path);

    /**
     * @brief Write bytes to a whole file.
     */
    static void _writeFile(const std::string &path, const std::vector<uint8_t> &bytes);

    /**
     * @brief Check magic, edition, and that the version is readable, throwing otherwise.
     */
    static void _gate(const RS_RSCFileHeader &header);

    /**
     * @brief Upgrade an older body to the current schema in place.
     */
    static void _migrate(RS_JSON &body, const RS_Version &from);

    /**
     * @brief Build the document bytes for a node in the given format.
     */
    template <typename T>
    static std::vector<uint8_t> _toBytes(const T &node, RSE_RSCFileFormat format) {
        RS_RSCPropertyTreeWriter writer(currentVersion());
        RS_RSCFileHeader header;
        header.magic = "RSC";
        header.edition = RS_RSC_EDITION;
        header.rsc_version = currentVersion();
        header.engine_version = engineVersion();
        header.type = RS_RSCRegistry::getInstance().getName(typeid(node));
        header.write(writer.addChild("header"));
        node.write(writer.addChild("body"));
        return _encode(*writer.root, format);
    }

    /**
     * @brief Decode, gate, migrate, verify the type, and read a node of type T.
     */
    template <typename T>
    static T _fromBytes(const std::vector<uint8_t> &bytes, RSE_RSCFileFormat format) {
        RS_JSON root = _decode(bytes, format);
        RS_RSCPropertyTreeReader root_reader(root);
        RS_RSCFileHeader header;
        header.read(root_reader.getChild("header"));
        _gate(header);

        RS_JSON &body = root.at("body");
        if (header.rsc_version < currentVersion()) {
            _migrate(body, header.rsc_version);
        }

        std::string expected = RS_RSCRegistry::getInstance().getName(typeid(T));
        if (header.type != expected) {
            RS_THROW(RS_InvalidCharacterizationException(
                "RSC file holds {} but {} was requested", header.type, expected
            ));
        }

        T node;
        RS_RSCPropertyTreeReader reader(body, header.rsc_version);
        node.read(reader);
        return node;
    }
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_FILE_H
