// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/signal/rsc_mechanical.h"

#include "rise_origin/rsc/rs_rsc_util.h"
#include "rise_origin/rsc/rs_rsc_validator.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_Force::operator==(const RSC_Force &other) const {
    return RSC_SignalBase::operator==(other) && structure_name == other.structure_name &&
           voxel_range == other.voxel_range && body_sid == other.body_sid &&
           segment_bid == other.segment_bid;
}

void RSC_Force::_validate(RS_RSCValidator &validator) const {
    RSC_SignalBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_Force::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_Force, structure_name);
        RSC_EXPECT_NOT_NULL(RSC_Force, body_sid, RS_NULL_INDEX);
        RSC_EXPECT_NOT_NULL(RSC_Force, segment_bid, RS_NULL_INDEX);
    });
}

void RSC_Force::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_SignalBase::_read(tree);
    structure_name = tree.get<string>("structure_name");
    voxel_range = readBound3ri(tree, "voxel_range");
    body_sid = tree.get("body_sid", body_sid);
    segment_bid = tree.get("segment_bid", segment_bid);
}

void RSC_Force::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_SignalBase::_write(tree);
    tree.add("structure_name", structure_name);
    writeBound3ri(tree, "voxel_range", voxel_range);
    tree.add("body_sid", body_sid);
    tree.add("segment_bid", segment_bid);
}

bool RSC_Torque::operator==(const RSC_Torque &other) const {
    return RSC_SignalBase::operator==(other) && structure_name == other.structure_name &&
           body_sid == other.body_sid && segment_bid == other.segment_bid;
}

void RSC_Torque::_validate(RS_RSCValidator &validator) const {
    RSC_SignalBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_Torque::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_Torque, structure_name);
        RSC_EXPECT_NOT_NULL(RSC_Torque, body_sid, RS_NULL_INDEX);
        RSC_EXPECT_NOT_NULL(RSC_Torque, segment_bid, RS_NULL_INDEX);
    });
}

void RSC_Torque::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_SignalBase::_read(tree);
    structure_name = tree.get<string>("structure_name");
    body_sid = tree.get("body_sid", body_sid);
    segment_bid = tree.get("segment_bid", segment_bid);
}

void RSC_Torque::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_SignalBase::_write(tree);
    tree.add("structure_name", structure_name);
    tree.add("body_sid", body_sid);
    tree.add("segment_bid", segment_bid);
}

bool RSC_Transform::operator==(const RSC_Transform &other) const {
    return RSC_SignalBase::operator==(other) && structure_name == other.structure_name;
}

void RSC_Transform::_validate(RS_RSCValidator &validator) const {
    RSC_SignalBase::_validate(validator);
    validator.submitValidatorIfNotExists("RSC_Transform::_validate", this, [this] {
        RSC_EXPECT_NOT_EMPTY(RSC_Transform, structure_name);
    });
}

void RSC_Transform::_read(const RS_RSCPropertyTreeReader &tree) {
    RSC_SignalBase::_read(tree);
    structure_name = tree.get<string>("structure_name");
}

void RSC_Transform::_write(RS_RSCPropertyTreeWriter &tree) const {
    RSC_SignalBase::_write(tree);
    tree.add("structure_name", structure_name);
}

}  // namespace rsc

}  // namespace rise
