// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_FILE_TREE_CONTEXT_H
#define RS_RSC_FILE_TREE_CONTEXT_H

#include <memory>
#include <type_traits>
#include <unordered_map>

#include "rise_origin/utils/data/rs_optional.h"
#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/serialization/rs_fast_pack.h"
#include "rise_origin/utils/version/rs_version.h"

namespace rise {

namespace rsc {

/// Forward declared to key the shared table by pointee identity.
class RSC_Base;

/// Underlying integer of an enum, else the type itself, the scalar dtype a bulk array is stored as.
template <typename T, bool = std::is_enum_v<T>>
struct RS_RSCBulkScalar {
    using type = T;
};
template <typename T>
struct RS_RSCBulkScalar<T, true> {
    using type = std::underlying_type_t<T>;
};

/**
 * @brief The whole state of one file write, shared by every writer cursor of the document.
 */
struct RS_RSCFileTreeWriteContext {
    /// Document being built, holding header, shared, and body.
    RS_JSON root;

    /// Schema version this document is written as.
    RS_Version version;

    /// Receives extracted bulk fields, disengaged in the text format.
    RS_Optional<RS_FastPacker> packer;

    /// Whether the binary format compresses its bulk fields and meta, ignored by the text format.
    bool is_compressing = true;

    /// Next bulk field ordinal, used when the packer is engaged.
    RIndex next_bulk_index = 0;

    /// Next index into root["shared"].
    RIndex next_shared_index = 0;

    /// Maps a pooled rsc to its index in root["shared"].
    std::unordered_map<const RSC_Base *, RIndex> shared_map;
};

/**
 * @brief The whole state of one file read, shared by every reader cursor of the document.
 */
struct RS_RSCFileTreeReadContext {
    /// Document being read, holding header, shared, and body.
    RS_JSON root;

    /// Schema version this document was written as.
    RS_Version version;

    /// Loads bulk fields on demand, disengaged in the text format.
    RS_Optional<RS_FastUnpacker> unpacker;

    /// Maps a shared index to its object, filled two-phase before the body is read.
    std::unordered_map<RIndex, std::shared_ptr<RSC_Base>> shared_map;
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_FILE_TREE_CONTEXT_H
