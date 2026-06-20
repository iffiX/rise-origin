// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_RECORDER_H
#define RSC_RECORDER_H

#include <string>

#include "rise_origin/rsc/rsc_base.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rsc_recorder_base.h"

namespace rise {

namespace rsc {

/**
 * @brief HDF5 recorder — persists records to an HDF5 file.
 */
class RSC_RecorderHDF5 : public RSC_RecorderBase {
   public:
    /// Output file path.
    std::string path;

    bool operator==(const RSC_RecorderHDF5& other) const;

    template <typename Serializer>
    void serialize(Serializer& s) {
        RSC_RecorderBase::serialize(s);
        s("path", path);
    }

   protected:
    void _read(const RS_CharacterizationPropertyTree& rsc_tree) override;
};

using RSC_RecorderVariant = RS_Variant<RSC_RecorderHDF5>;

}  // namespace rsc

}  // namespace rise

#endif  // RSC_RECORDER_H
