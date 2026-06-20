// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/signal/rsc_mechanical.h"

#include "rise_origin/rsc/rs_rsc_util.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_Force::operator==(const RSC_Force &other) const {
    return RSC_SignalBase::operator==(other) && structure_name == other.structure_name &&
           voxel_range == other.voxel_range && body_sid == other.body_sid &&
           segment_bid == other.segment_bid;
}

void RSC_Force::_validate() const {
    RSC_SignalBase::_validate();
    RSC_EXPECT_NOT_EMPTY(RSC_Force, structure_name);
    if (!voxel_range.isValid()) {
        RS_THROW(
            RS_InvalidCharacterizationException("Invalid RSC_Force, voxel_range must be valid")
        );
    }
    RSC_EXPECT_NOT_NULL(RSC_Force, body_sid, RS_NULL_INDEX);
    RSC_EXPECT_NOT_NULL(RSC_Force, segment_bid, RS_NULL_INDEX);
}

void RSC_Force::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_SignalBase::_read(rsc_tree);
    structure_name = rsc_tree.get<string>("StructureName");
    voxel_range = readBound3ri(rsc_tree.get<string>("VoxelRange"));
    body_sid = rsc_tree.get("BodySid", body_sid);
    segment_bid = rsc_tree.get("SegmentBid", segment_bid);
}

bool RSC_Torque::operator==(const RSC_Torque &other) const {
    return RSC_SignalBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid;
}

void RSC_Torque::_validate() const {
    RSC_SignalBase::_validate();
    RSC_EXPECT_NOT_EMPTY(RSC_Torque, structure_name);
    RSC_EXPECT_NOT_NULL(RSC_Torque, body_sid, RS_NULL_INDEX);
    RSC_EXPECT_NOT_NULL(RSC_Torque, segment_bid, RS_NULL_INDEX);
}

void RSC_Torque::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    RSC_SignalBase::_read(rsc_tree);
    structure_name = rsc_tree.get<string>("StructureName");
    body_sid = rsc_tree.get("BodySid", body_sid);
    segment_bid = rsc_tree.get("SegmentBid", segment_bid);
}

}  // namespace rsc

}  // namespace rise
