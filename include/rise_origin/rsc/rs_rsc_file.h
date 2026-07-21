// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_FILE_H
#define RS_RSC_FILE_H

#include <cstdint>
#include <string>
#include <typeinfo>
#include <vector>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/version/rs_version.h"
#include "rs_rsc_file_tree_context.h"
#include "rs_rsc_file_tree_reader.h"
#include "rs_rsc_file_tree_writer.h"
#include "rs_rsc_registry.h"
#include "rsc_base.h"

namespace rise {

namespace rsc {

/// Magic guard at the head of every binary RSC file, the four bytes "RSCB".
constexpr uint32_t RS_RSCB_MAGIC = 0x52534342u;

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
    void read(const RS_RSCFileTreeReader &tree) {
        magic = tree.get<std::string>("magic");
        edition = tree.get<std::string>("edition");
        rsc_version.fromString(tree.get<std::string>("rsc_version"));
        engine_version.fromString(tree.get<std::string>("engine_version"));
        type = tree.get<std::string>("type");
    }

    /**
     * @brief Write the header fields to a writer.
     */
    void write(RS_RSCFileTreeWriter tree) const {
        tree.add("magic", magic);
        tree.add("edition", edition);
        tree.add("rsc_version", rsc_version.toString());
        tree.add("engine_version", engine_version.toString());
        tree.add("type", type);
    }
};

/**
 * @brief Durable save and load of any RSC node, in the text or binary format.
 * @details Holds a node as { header, shared, body }, writes identity and version, and gates and
 * migrates on load. The format is picked from the path extension, .rsct for text and .rscb for
 * binary. The per operation state is a stack local context, which for .rscb also owns the packer
 * or unpacker moved in from the file pool.
 */
class RS_RSCFile {
   public:
    /**
     * @brief Save a node to path, format chosen by extension.
     * @param node The node to save.
     * @param path The destination path, .rsct for text or .rscb for binary.
     * @param is_compressing Whether the binary format compresses its fields, ignored for text.
     */
    template <typename T>
    static void save(const T &node, const std::string &path, bool is_compressing = true) {
        saveAny(node, path, is_compressing);
    }

    /**
     * @brief Load a node of type T from path, verifying the file holds that type.
     */
    template <typename T>
    static T load(const std::string &path) {
        RS_RSCFileTreeReadContext context;
        RS_RSCFileHeader header = _open(context, path);
        std::string expected = RS_RSCRegistry::getName(typeid(T));
        if (header.type != expected) {
            RS_THROW(RS_InvalidCharacterizationException(
                "RSC file holds {} but {} was requested", header.type, expected
            ));
        }
        RS_RSCFileTreeReader reader(context);
        T node;
        node.read(reader.getChild("body"));
        return node;
    }

    /**
     * @brief Save a node to path.
     * @param node The node to save.
     * @param path The destination path, .rsct for text or .rscb for binary.
     * @param is_compressing Whether the binary format compresses its fields, ignored for text.
     */
    static void saveAny(const RSC_Base &node, const std::string &path, bool is_compressing = true);

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
     * @brief Begin a load, filling the context root, gating, and migrating, returning the header.
     */
    static RS_RSCFileHeader _open(RS_RSCFileTreeReadContext &context, const std::string &path);

    /**
     * @brief Check magic and edition, throwing otherwise. The version never locks a read.
     */
    static void _gate(const RS_RSCFileHeader &header);

    /**
     * @brief Upgrade an older body to the current schema in place.
     */
    static void _migrate(RS_JSON &body, const RS_Version &from);

    /**
     * @brief Finish a save, writing the text file or packing the binary file.
     */
    static void
    _commit(RS_RSCFileTreeWriteContext &context, const std::string &path, RSE_RSCFileFormat format);

    /**
     * @brief Parse the root document from a text file.
     */
    static RS_JSON _readTextFile(const std::string &path);

    /**
     * @brief Write the root document to a text file as pretty JSON.
     */
    static void _writeTextFile(const std::string &path, const RS_JSON &root);

    /**
     * @brief Unpack a binary file into the context, returning its root document from the meta.
     */
    static RS_JSON _readBinaryFile(const std::string &path, RS_RSCFileTreeReadContext &context);

    /**
     * @brief Pack the context to a binary file, extracting the root as meta and bulk as fields.
     */
    static void _writeBinaryFile(const std::string &path, RS_RSCFileTreeWriteContext &context);
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_FILE_H
