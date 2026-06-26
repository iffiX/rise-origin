// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rsc_simulation.h"

#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rs_rsc_util.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_SimulationEnvironment::operator==(const RSC_SimulationEnvironment &other) const {
    return RSC_Base::operator==(other) && gravity_acceleration == other.gravity_acceleration;
}

void RSC_SimulationEnvironment::_validate(RS_RSCValidator &) const {}

void RSC_SimulationEnvironment::_read(const RS_RSCPropertyTreeReader &tree) {
    gravity_acceleration = tree.get("gravity_acceleration", gravity_acceleration);
}

void RSC_SimulationEnvironment::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("gravity_acceleration", gravity_acceleration);
}

bool RSC_SimulationTime::operator==(const RSC_SimulationTime &other) const {
    return RSC_Base::operator==(other) && dt == other.dt && dt_ratio == other.dt_ratio &&
           dt_update_interval == other.dt_update_interval &&
           dt_softbody_and_rigidbody_update == other.dt_softbody_and_rigidbody_update &&
           dt_rigidbody_contact_and_constraint_update ==
               other.dt_rigidbody_contact_and_constraint_update &&
           dt_contact_update == other.dt_contact_update &&
           dt_mechanical_signal_update == other.dt_mechanical_signal_update &&
           dt_sentinel_update == other.dt_sentinel_update;
}

void RSC_SimulationTime::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_SimulationTime::_validate", this, [this] {
        if (dt) RSC_EXPECT_GT(RSC_SimulationTime, dt.get(), 0);
        RSC_EXPECT_GT(RSC_SimulationTime, dt_ratio, 0);
        if (dt_softbody_and_rigidbody_update)
            RSC_EXPECT_GT(RSC_SimulationTime, dt_softbody_and_rigidbody_update.get(), 0);
        if (dt_rigidbody_contact_and_constraint_update)
            RSC_EXPECT_GT(RSC_SimulationTime, dt_rigidbody_contact_and_constraint_update.get(), 0);
        if (dt_contact_update) RSC_EXPECT_GT(RSC_SimulationTime, dt_contact_update.get(), 0);
        if (dt_mechanical_signal_update)
            RSC_EXPECT_GT(RSC_SimulationTime, dt_mechanical_signal_update.get(), 0);
        if (dt_sentinel_update) RSC_EXPECT_GT(RSC_SimulationTime, dt_sentinel_update.get(), 0);
    });
}

void RSC_SimulationTime::_read(const RS_RSCPropertyTreeReader &tree) {
    if (tree.isChildExist("dt")) dt = tree.get<RFloat>("dt");
    dt_ratio = tree.get("dt_ratio", dt_ratio);
    dt_update_interval = tree.get("dt_update_interval", dt_update_interval);
    if (tree.isChildExist("dt_softbody_and_rigidbody_update"))
        dt_softbody_and_rigidbody_update = tree.get<RFloat>("dt_softbody_and_rigidbody_update");
    if (tree.isChildExist("dt_rigidbody_contact_and_constraint_update"))
        dt_rigidbody_contact_and_constraint_update =
            tree.get<RFloat>("dt_rigidbody_contact_and_constraint_update");
    if (tree.isChildExist("dt_contact_update"))
        dt_contact_update = tree.get<RFloat>("dt_contact_update");
    if (tree.isChildExist("dt_mechanical_signal_update"))
        dt_mechanical_signal_update = tree.get<RFloat>("dt_mechanical_signal_update");
    if (tree.isChildExist("dt_sentinel_update"))
        dt_sentinel_update = tree.get<RFloat>("dt_sentinel_update");
}

void RSC_SimulationTime::_write(RS_RSCPropertyTreeWriter &tree) const {
    if (dt) tree.add("dt", dt.get());
    tree.add("dt_ratio", dt_ratio);
    tree.add("dt_update_interval", dt_update_interval);
    if (dt_softbody_and_rigidbody_update)
        tree.add("dt_softbody_and_rigidbody_update", dt_softbody_and_rigidbody_update.get());
    if (dt_rigidbody_contact_and_constraint_update)
        tree.add(
            "dt_rigidbody_contact_and_constraint_update",
            dt_rigidbody_contact_and_constraint_update.get()
        );
    if (dt_contact_update) tree.add("dt_contact_update", dt_contact_update.get());
    if (dt_mechanical_signal_update)
        tree.add("dt_mechanical_signal_update", dt_mechanical_signal_update.get());
    if (dt_sentinel_update) tree.add("dt_sentinel_update", dt_sentinel_update.get());
}

bool RSC_SimulationSoftBody::operator==(const RSC_SimulationSoftBody &other) const {
    return RSC_Base::operator==(other) && internal_damping_z == other.internal_damping_z &&
           contact_damping_z == other.contact_damping_z &&
           global_damping_z == other.global_damping_z &&
           integration_frequency_multiplier == other.integration_frequency_multiplier &&
           link_max_compression == other.link_max_compression;
}

void RSC_SimulationSoftBody::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_SimulationSoftBody::_validate", this, [this] {
        RSC_EXPECT_GE(RSC_SimulationSoftBody, internal_damping_z, 0);
        RSC_EXPECT_LE(RSC_SimulationSoftBody, internal_damping_z, 1);
        RSC_EXPECT_GE(RSC_SimulationSoftBody, contact_damping_z, 0);
        RSC_EXPECT_LE(RSC_SimulationSoftBody, contact_damping_z, 1);
        RSC_EXPECT_GE(RSC_SimulationSoftBody, global_damping_z, 0);
        RSC_EXPECT_LE(RSC_SimulationSoftBody, global_damping_z, 1);
        RSC_EXPECT_GT(RSC_SimulationSoftBody, integration_frequency_multiplier, 0);
        RSC_EXPECT_GE(RSC_SimulationSoftBody, link_max_compression, 0);
        RSC_EXPECT_LT(RSC_SimulationSoftBody, link_max_compression, 1);
    });
}

void RSC_SimulationSoftBody::_read(const RS_RSCPropertyTreeReader &tree) {
    internal_damping_z = tree.get("internal_damping_z", internal_damping_z);
    contact_damping_z = tree.get("contact_damping_z", contact_damping_z);
    global_damping_z = tree.get("global_damping_z", global_damping_z);
    integration_frequency_multiplier =
        tree.get("integration_frequency_multiplier", integration_frequency_multiplier);
    link_max_compression = tree.get("link_max_compression", link_max_compression);
}

void RSC_SimulationSoftBody::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("internal_damping_z", internal_damping_z);
    tree.add("contact_damping_z", contact_damping_z);
    tree.add("global_damping_z", global_damping_z);
    tree.add("integration_frequency_multiplier", integration_frequency_multiplier);
    tree.add("link_max_compression", link_max_compression);
}

bool RSC_SimulationRigidBody::operator==(const RSC_SimulationRigidBody &other) const {
    return RSC_Base::operator==(other) && constraint_iterations == other.constraint_iterations &&
           constraint_position_correction_ratio == other.constraint_position_correction_ratio &&
           constraint_warm_start_damping_z == other.constraint_warm_start_damping_z &&
           constraint_stiffness_initial == other.constraint_stiffness_initial &&
           constraint_stiffness_min_threshold == other.constraint_stiffness_min_threshold &&
           constraint_stiffness_max_threshold == other.constraint_stiffness_max_threshold &&
           constraint_stiffness_auto_adjust_watt_threshold ==
               other.constraint_stiffness_auto_adjust_watt_threshold &&
           constraint_stiffness_auto_adjust_step == other.constraint_stiffness_auto_adjust_step &&
           contact_iterations == other.contact_iterations &&
           contact_position_correction_iterations == other.contact_position_correction_iterations &&
           contact_position_correction_ratio == other.contact_position_correction_ratio &&
           contact_warm_start_damping_z == other.contact_warm_start_damping_z &&
           contact_slop == other.contact_slop && linear_damping_z == other.linear_damping_z &&
           angular_damping_z == other.angular_damping_z;
}

void RSC_SimulationRigidBody::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_SimulationRigidBody::_validate", this, [this] {
        RSC_EXPECT_GE(RSC_SimulationRigidBody, constraint_iterations, 0);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, constraint_position_correction_ratio, 0);
        RSC_EXPECT_LE(RSC_SimulationRigidBody, constraint_position_correction_ratio, 1);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, constraint_warm_start_damping_z, 0);
        RSC_EXPECT_LE(RSC_SimulationRigidBody, constraint_warm_start_damping_z, 1);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, constraint_stiffness_min_threshold, 0);
        RSC_EXPECT_GE(
            RSC_SimulationRigidBody,
            constraint_stiffness_max_threshold,
            constraint_stiffness_min_threshold
        );
        RSC_EXPECT_GE(RSC_SimulationRigidBody, constraint_stiffness_auto_adjust_step, 0);
        RSC_EXPECT_LE(RSC_SimulationRigidBody, constraint_stiffness_auto_adjust_step, 1);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, contact_iterations, 0);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, contact_position_correction_iterations, 0);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, contact_position_correction_ratio, 0);
        RSC_EXPECT_LE(RSC_SimulationRigidBody, contact_position_correction_ratio, 1);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, contact_warm_start_damping_z, 0);
        RSC_EXPECT_LE(RSC_SimulationRigidBody, contact_warm_start_damping_z, 1);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, contact_slop, 0);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, linear_damping_z, 0);
        RSC_EXPECT_LE(RSC_SimulationRigidBody, linear_damping_z, 1);
        RSC_EXPECT_GE(RSC_SimulationRigidBody, angular_damping_z, 0);
        RSC_EXPECT_LE(RSC_SimulationRigidBody, angular_damping_z, 1);
    });
}

void RSC_SimulationRigidBody::_read(const RS_RSCPropertyTreeReader &tree) {
    constraint_iterations = tree.get("constraint_iterations", constraint_iterations);
    constraint_position_correction_ratio =
        tree.get("constraint_position_correction_ratio", constraint_position_correction_ratio);
    constraint_warm_start_damping_z =
        tree.get("constraint_warm_start_damping_z", constraint_warm_start_damping_z);
    constraint_stiffness_initial =
        tree.get("constraint_stiffness_initial", constraint_stiffness_initial);
    constraint_stiffness_min_threshold =
        tree.get("constraint_stiffness_min_threshold", constraint_stiffness_min_threshold);
    constraint_stiffness_max_threshold =
        tree.get("constraint_stiffness_max_threshold", constraint_stiffness_max_threshold);
    constraint_stiffness_auto_adjust_watt_threshold = tree.get(
        "constraint_stiffness_auto_adjust_watt_threshold",
        constraint_stiffness_auto_adjust_watt_threshold
    );
    constraint_stiffness_auto_adjust_step =
        tree.get("constraint_stiffness_auto_adjust_step", constraint_stiffness_auto_adjust_step);
    contact_iterations = tree.get("contact_iterations", contact_iterations);
    contact_position_correction_iterations =
        tree.get("contact_position_correction_iterations", contact_position_correction_iterations);
    contact_position_correction_ratio =
        tree.get("contact_position_correction_ratio", contact_position_correction_ratio);
    contact_warm_start_damping_z =
        tree.get("contact_warm_start_damping_z", contact_warm_start_damping_z);
    contact_slop = tree.get("contact_slop", contact_slop);
    linear_damping_z = tree.get("linear_damping_z", linear_damping_z);
    angular_damping_z = tree.get("angular_damping_z", angular_damping_z);
}

void RSC_SimulationRigidBody::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("constraint_iterations", constraint_iterations);
    tree.add("constraint_position_correction_ratio", constraint_position_correction_ratio);
    tree.add("constraint_warm_start_damping_z", constraint_warm_start_damping_z);
    tree.add("constraint_stiffness_initial", constraint_stiffness_initial);
    tree.add("constraint_stiffness_min_threshold", constraint_stiffness_min_threshold);
    tree.add("constraint_stiffness_max_threshold", constraint_stiffness_max_threshold);
    tree.add(
        "constraint_stiffness_auto_adjust_watt_threshold",
        constraint_stiffness_auto_adjust_watt_threshold
    );
    tree.add("constraint_stiffness_auto_adjust_step", constraint_stiffness_auto_adjust_step);
    tree.add("contact_iterations", contact_iterations);
    tree.add("contact_position_correction_iterations", contact_position_correction_iterations);
    tree.add("contact_position_correction_ratio", contact_position_correction_ratio);
    tree.add("contact_warm_start_damping_z", contact_warm_start_damping_z);
    tree.add("contact_slop", contact_slop);
    tree.add("linear_damping_z", linear_damping_z);
    tree.add("angular_damping_z", angular_damping_z);
}

bool RSC_SimulationContact::operator==(const RSC_SimulationContact &other) const {
    return RSC_Base::operator==(other) &&
           bvh_rebuild_sub_interval == other.bvh_rebuild_sub_interval &&
           bvh_refit_sub_interval == other.bvh_refit_sub_interval;
}

void RSC_SimulationContact::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_SimulationContact::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_SimulationContact, bvh_rebuild_sub_interval, 0);
        RSC_EXPECT_GT(RSC_SimulationContact, bvh_refit_sub_interval, 0);
        if (bvh_rebuild_sub_interval % bvh_refit_sub_interval != 0) {
            RS_THROW(RS_InvalidCharacterizationException(
                "RSC_SimulationContact: bvh_rebuild_sub_interval ({}) must be a multiple of "
                "bvh_refit_sub_interval ({})",
                bvh_rebuild_sub_interval,
                bvh_refit_sub_interval
            ));
        }
    });
}

void RSC_SimulationContact::_read(const RS_RSCPropertyTreeReader &tree) {
    bvh_rebuild_sub_interval = tree.get("bvh_rebuild_sub_interval", bvh_rebuild_sub_interval);
    bvh_refit_sub_interval = tree.get("bvh_refit_sub_interval", bvh_refit_sub_interval);
}

void RSC_SimulationContact::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("bvh_rebuild_sub_interval", bvh_rebuild_sub_interval);
    tree.add("bvh_refit_sub_interval", bvh_refit_sub_interval);
}

bool RSC_SimulationVisionTrace::operator==(const RSC_SimulationVisionTrace &other) const {
    return RSC_Base::operator==(other) && firefly_cap == other.firefly_cap &&
           russian_roulette_start_depth == other.russian_roulette_start_depth;
}

void RSC_SimulationVisionTrace::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_SimulationVisionTrace::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_SimulationVisionTrace, firefly_cap, 0);
    });
}

void RSC_SimulationVisionTrace::_read(const RS_RSCPropertyTreeReader &tree) {
    firefly_cap = tree.get<RFloat>("firefly_cap", firefly_cap);
    russian_roulette_start_depth =
        tree.get<RSize>("russian_roulette_start_depth", russian_roulette_start_depth);
}

void RSC_SimulationVisionTrace::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("firefly_cap", firefly_cap);
    tree.add("russian_roulette_start_depth", russian_roulette_start_depth);
}

bool RSC_SimulationVisionReSTIR::operator==(const RSC_SimulationVisionReSTIR &other) const {
    return RSC_Base::operator==(other) && is_enabled == other.is_enabled &&
           disocclusion_position_threshold == other.disocclusion_position_threshold &&
           disocclusion_normal_threshold == other.disocclusion_normal_threshold &&
           max_specular_chain == other.max_specular_chain && di_m_cap == other.di_m_cap &&
           di_spatial_neighbors == other.di_spatial_neighbors &&
           di_spatial_radius == other.di_spatial_radius &&
           is_gi_bounce_nee_enabled == other.is_gi_bounce_nee_enabled &&
           gi_m_cap == other.gi_m_cap && gi_spatial_neighbors == other.gi_spatial_neighbors &&
           gi_spatial_radius == other.gi_spatial_radius &&
           gi_validation_period == other.gi_validation_period;
}

void RSC_SimulationVisionReSTIR::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_SimulationVisionReSTIR::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_SimulationVisionReSTIR, disocclusion_position_threshold, 0);
        RSC_EXPECT_GT(RSC_SimulationVisionReSTIR, disocclusion_normal_threshold, 0);
        RSC_EXPECT_LE(RSC_SimulationVisionReSTIR, disocclusion_normal_threshold, 1);
        RSC_EXPECT_GE(RSC_SimulationVisionReSTIR, max_specular_chain, 1);
        RSC_EXPECT_GE(RSC_SimulationVisionReSTIR, di_m_cap, 1);
        RSC_EXPECT_GT(RSC_SimulationVisionReSTIR, di_spatial_radius, 0);
        RSC_EXPECT_GE(RSC_SimulationVisionReSTIR, gi_m_cap, 1);
        RSC_EXPECT_GT(RSC_SimulationVisionReSTIR, gi_spatial_radius, 0);
    });
}

void RSC_SimulationVisionReSTIR::_read(const RS_RSCPropertyTreeReader &tree) {
    is_enabled = tree.get<bool>("is_enabled", is_enabled);
    disocclusion_position_threshold =
        tree.get<RFloat>("disocclusion_position_threshold", disocclusion_position_threshold);
    disocclusion_normal_threshold =
        tree.get<RFloat>("disocclusion_normal_threshold", disocclusion_normal_threshold);
    max_specular_chain = tree.get<RSize>("max_specular_chain", max_specular_chain);
    di_m_cap = tree.get<RSize>("di_m_cap", di_m_cap);
    di_spatial_neighbors = tree.get<RSize>("di_spatial_neighbors", di_spatial_neighbors);
    di_spatial_radius = tree.get<RFloat>("di_spatial_radius", di_spatial_radius);
    is_gi_bounce_nee_enabled = tree.get<bool>("is_gi_bounce_nee_enabled", is_gi_bounce_nee_enabled);
    gi_m_cap = tree.get<RSize>("gi_m_cap", gi_m_cap);
    gi_spatial_neighbors = tree.get<RSize>("gi_spatial_neighbors", gi_spatial_neighbors);
    gi_spatial_radius = tree.get<RFloat>("gi_spatial_radius", gi_spatial_radius);
    gi_validation_period = tree.get<RSize>("gi_validation_period", gi_validation_period);
}

void RSC_SimulationVisionReSTIR::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("is_enabled", is_enabled);
    tree.add("disocclusion_position_threshold", disocclusion_position_threshold);
    tree.add("disocclusion_normal_threshold", disocclusion_normal_threshold);
    tree.add("max_specular_chain", max_specular_chain);
    tree.add("di_m_cap", di_m_cap);
    tree.add("di_spatial_neighbors", di_spatial_neighbors);
    tree.add("di_spatial_radius", di_spatial_radius);
    tree.add("is_gi_bounce_nee_enabled", is_gi_bounce_nee_enabled);
    tree.add("gi_m_cap", gi_m_cap);
    tree.add("gi_spatial_neighbors", gi_spatial_neighbors);
    tree.add("gi_spatial_radius", gi_spatial_radius);
    tree.add("gi_validation_period", gi_validation_period);
}

bool RSC_SimulationVisionDenoiseNone::operator==(
    const RSC_SimulationVisionDenoiseNone &other
) const {
    return RSC_Base::operator==(other);
}

void RSC_SimulationVisionDenoiseNone::_validate(RS_RSCValidator &) const {}

void RSC_SimulationVisionDenoiseNone::_read(const RS_RSCPropertyTreeReader &) {}

void RSC_SimulationVisionDenoiseNone::_write(RS_RSCPropertyTreeWriter &) const {}

bool RSC_SimulationVisionDenoiseAtrous::operator==(
    const RSC_SimulationVisionDenoiseAtrous &other
) const {
    return RSC_Base::operator==(other) && atrous_iterations == other.atrous_iterations &&
           atrous_min_passes == other.atrous_min_passes && sigma_color == other.sigma_color &&
           sigma_normal == other.sigma_normal && sigma_depth == other.sigma_depth;
}

void RSC_SimulationVisionDenoiseAtrous::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists(
        "RSC_SimulationVisionDenoiseAtrous::_validate", this, [this] {
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseAtrous, atrous_iterations, 0);
            RSC_EXPECT_GE(RSC_SimulationVisionDenoiseAtrous, atrous_min_passes, 0);
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseAtrous, sigma_color, 0);
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseAtrous, sigma_normal, 0);
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseAtrous, sigma_depth, 0);
        }
    );
}

void RSC_SimulationVisionDenoiseAtrous::_read(const RS_RSCPropertyTreeReader &tree) {
    atrous_iterations = tree.get<RSize>("atrous_iterations", atrous_iterations);
    atrous_min_passes = tree.get<RSize>("atrous_min_passes", atrous_min_passes);
    sigma_color = tree.get<RFloat>("sigma_color", sigma_color);
    sigma_normal = tree.get<RFloat>("sigma_normal", sigma_normal);
    sigma_depth = tree.get<RFloat>("sigma_depth", sigma_depth);
}

void RSC_SimulationVisionDenoiseAtrous::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("atrous_iterations", atrous_iterations);
    tree.add("atrous_min_passes", atrous_min_passes);
    tree.add("sigma_color", sigma_color);
    tree.add("sigma_normal", sigma_normal);
    tree.add("sigma_depth", sigma_depth);
}

bool RSC_SimulationVisionDenoiseSVGF::operator==(
    const RSC_SimulationVisionDenoiseSVGF &other
) const {
    return RSC_Base::operator==(other) && atrous_iterations == other.atrous_iterations &&
           atrous_min_passes == other.atrous_min_passes && sigma_color == other.sigma_color &&
           sigma_normal == other.sigma_normal && sigma_depth == other.sigma_depth &&
           temporal_alpha == other.temporal_alpha &&
           temporal_alpha_moments == other.temporal_alpha_moments &&
           temporal_min_history == other.temporal_min_history;
}

void RSC_SimulationVisionDenoiseSVGF::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists(
        "RSC_SimulationVisionDenoiseSVGF::_validate", this, [this] {
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, atrous_iterations, 0);
            RSC_EXPECT_GE(RSC_SimulationVisionDenoiseSVGF, atrous_min_passes, 0);
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, sigma_color, 0);
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, sigma_normal, 0);
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, sigma_depth, 0);
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, temporal_alpha, 0);
            RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, temporal_alpha_moments, 0);
            RSC_EXPECT_GE(RSC_SimulationVisionDenoiseSVGF, temporal_min_history, 1);
        }
    );
}

void RSC_SimulationVisionDenoiseSVGF::_read(const RS_RSCPropertyTreeReader &tree) {
    atrous_iterations = tree.get<RSize>("atrous_iterations", atrous_iterations);
    atrous_min_passes = tree.get<RSize>("atrous_min_passes", atrous_min_passes);
    sigma_color = tree.get<RFloat>("sigma_color", sigma_color);
    sigma_normal = tree.get<RFloat>("sigma_normal", sigma_normal);
    sigma_depth = tree.get<RFloat>("sigma_depth", sigma_depth);
    temporal_alpha = tree.get<RFloat>("temporal_alpha", temporal_alpha);
    temporal_alpha_moments = tree.get<RFloat>("temporal_alpha_moments", temporal_alpha_moments);
    temporal_min_history = tree.get<RFloat>("temporal_min_history", temporal_min_history);
}

void RSC_SimulationVisionDenoiseSVGF::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("atrous_iterations", atrous_iterations);
    tree.add("atrous_min_passes", atrous_min_passes);
    tree.add("sigma_color", sigma_color);
    tree.add("sigma_normal", sigma_normal);
    tree.add("sigma_depth", sigma_depth);
    tree.add("temporal_alpha", temporal_alpha);
    tree.add("temporal_alpha_moments", temporal_alpha_moments);
    tree.add("temporal_min_history", temporal_min_history);
}

bool RSC_SimulationVision::operator==(const RSC_SimulationVision &other) const {
    return RSC_Base::operator==(other) && trace == other.trace && restir == other.restir &&
           denoise == other.denoise;
}

void RSC_SimulationVision::_validate(RS_RSCValidator &validator) const {
    trace.validate(validator);
    restir.validate(validator);
    denoise.visit([&validator](const auto &d) { d.validate(validator); });
}

void RSC_SimulationVision::_read(const RS_RSCPropertyTreeReader &tree) {
    if (tree.isChildExist("trace")) {
        trace.read(tree.getChild("trace"));
    }
    if (tree.isChildExist("restir")) {
        restir.read(tree.getChild("restir"));
    }
    if (tree.isChildExist("denoise")) {
        auto denoise_tree = tree.getChild("denoise");
        auto children = denoise_tree.getChildren();
        if (children.empty()) {
            denoise = RSC_SimulationVisionDenoiseNone{};
        } else if (children.size() != 1) {
            RS_THROW(RS_InvalidCharacterizationException(
                "denoise must have exactly one alternative child"
            ));
        } else {
            auto &[tag, body] = children.front();
            if (tag == "none") {
                denoise = RSC_SimulationVisionDenoiseNone{};
            } else if (tag == "atrous") {
                RSC_SimulationVisionDenoiseAtrous d;
                d.read(body);
                denoise = d;
            } else if (tag == "svgf") {
                RSC_SimulationVisionDenoiseSVGF d;
                d.read(body);
                denoise = d;
            } else {
                RS_THROW(RS_InvalidCharacterizationException("Unknown denoise tag: {}", tag));
            }
        }
    }
}

void RSC_SimulationVision::_write(RS_RSCPropertyTreeWriter &tree) const {
    trace.write(tree.addChild("trace"));
    restir.write(tree.addChild("restir"));
    auto denoise_tree = tree.addChild("denoise");
    denoise.visit([&denoise_tree](const auto &d) {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, RSC_SimulationVisionDenoiseNone>) {
            denoise_tree.addChild("none");
        } else if constexpr (std::is_same_v<T, RSC_SimulationVisionDenoiseAtrous>) {
            d.write(denoise_tree.addChild("atrous"));
        } else if constexpr (std::is_same_v<T, RSC_SimulationVisionDenoiseSVGF>) {
            d.write(denoise_tree.addChild("svgf"));
        }
    });
}

bool RSC_SimulationControl::operator==(const RSC_SimulationControl &other) const {
    return RSC_Base::operator==(other) && frequency == other.frequency;
}

void RSC_SimulationControl::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_SimulationControl::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_SimulationControl, frequency, 0);
    });
}

void RSC_SimulationControl::_read(const RS_RSCPropertyTreeReader &tree) {
    frequency = tree.get("frequency", frequency);
}

void RSC_SimulationControl::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("frequency", frequency);
}

bool RSC_SimulationRecord::operator==(const RSC_SimulationRecord &other) const {
    return RSC_Base::operator==(other) && frequency == other.frequency;
}

void RSC_SimulationRecord::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_SimulationRecord::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_SimulationRecord, frequency, 0);
    });
}

void RSC_SimulationRecord::_read(const RS_RSCPropertyTreeReader &tree) {
    frequency = tree.get("frequency", frequency);
}

void RSC_SimulationRecord::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("frequency", frequency);
}

bool RSC_SimulationSentinel::operator==(const RSC_SimulationSentinel &other) const {
    return RSC_Base::operator==(other) && is_enabled == other.is_enabled &&
           degenerate_min_dt == other.degenerate_min_dt;
}

void RSC_SimulationSentinel::_validate(RS_RSCValidator &) const {}

void RSC_SimulationSentinel::_read(const RS_RSCPropertyTreeReader &tree) {
    is_enabled = tree.get("is_enabled", is_enabled);
    degenerate_min_dt = tree.get<RFloat>("degenerate_min_dt", degenerate_min_dt);
}

void RSC_SimulationSentinel::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("is_enabled", is_enabled);
    tree.add("degenerate_min_dt", degenerate_min_dt);
}

bool RSC_SimulationRuntime::operator==(const RSC_SimulationRuntime &other) const {
    return RSC_Base::operator==(other) && max_time == other.max_time &&
           record_buffer_size == other.record_buffer_size &&
           is_save_record == other.is_save_record && compute_accuracy == other.compute_accuracy &&
           memory_accuracy == other.memory_accuracy && log_level == other.log_level;
}

void RSC_SimulationRuntime::_validate(RS_RSCValidator &validator) const {
    validator.submitValidatorIfNotExists("RSC_SimulationRuntime::_validate", this, [this] {
        RSC_EXPECT_GT(RSC_SimulationRuntime, max_time, 0);
        RSC_EXPECT_GT(RSC_SimulationRuntime, record_buffer_size, 0);
    });
}

void RSC_SimulationRuntime::_read(const RS_RSCPropertyTreeReader &tree) {
    max_time = tree.get("max_time", max_time);
    record_buffer_size = tree.get("record_buffer_size", record_buffer_size);
    is_save_record = tree.get("is_save_record", is_save_record);
    compute_accuracy = readEnum<RSE_FloatType>(tree, "compute_accuracy", compute_accuracy);
    memory_accuracy = readEnum<RSE_FloatType>(tree, "memory_accuracy", memory_accuracy);
    log_level = tree.get("log_level", log_level);
}

void RSC_SimulationRuntime::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("max_time", max_time);
    tree.add("record_buffer_size", record_buffer_size);
    tree.add("is_save_record", is_save_record);
    writeEnum(tree, "compute_accuracy", compute_accuracy);
    writeEnum(tree, "memory_accuracy", memory_accuracy);
    tree.add("log_level", log_level);
}

bool RSC_Simulation::operator==(const RSC_Simulation &other) const {
    return RSC_Base::operator==(other) && seed == other.seed &&
           stop_condition.expr == other.stop_condition.expr && environment == other.environment &&
           time == other.time && soft_body == other.soft_body && rigid_body == other.rigid_body &&
           contact == other.contact && sentinel == other.sentinel && vision == other.vision &&
           control == other.control && record == other.record && runtime == other.runtime;
}

void RSC_Simulation::_validate(RS_RSCValidator &validator) const {
    environment.validate(validator);
    time.validate(validator);
    soft_body.validate(validator);
    rigid_body.validate(validator);
    contact.validate(validator);
    sentinel.validate(validator);
    vision.validate(validator);
    control.validate(validator);
    record.validate(validator);
    runtime.validate(validator);
}

void RSC_Simulation::_read(const RS_RSCPropertyTreeReader &tree) {
    seed = tree.get("seed", seed);

    if (tree.isChildExist("stop_condition")) {
        auto stop_condition_str = tree.get<string>("stop_condition", "");
        try {
            stop_condition = stop_condition_str;
        } catch (const RS_InvalidExpressionException &e) {
            RS_THROW(RS_InvalidCharacterizationException("Invalid stop condition: {}", e.what()));
        }
    }

    if (tree.isChildExist("environment")) {
        environment.read(tree.getChild("environment"));
    }
    if (tree.isChildExist("time")) {
        time.read(tree.getChild("time"));
    }
    if (tree.isChildExist("soft_body")) {
        soft_body.read(tree.getChild("soft_body"));
    }
    if (tree.isChildExist("rigid_body")) {
        rigid_body.read(tree.getChild("rigid_body"));
    }
    if (tree.isChildExist("contact")) {
        contact.read(tree.getChild("contact"));
    }
    if (tree.isChildExist("sentinel")) {
        sentinel.read(tree.getChild("sentinel"));
    }
    if (tree.isChildExist("vision")) {
        vision.read(tree.getChild("vision"));
    }
    if (tree.isChildExist("control")) {
        control.read(tree.getChild("control"));
    }
    if (tree.isChildExist("record")) {
        record.read(tree.getChild("record"));
    }
    if (tree.isChildExist("runtime")) {
        runtime.read(tree.getChild("runtime"));
    }
}

void RSC_Simulation::_write(RS_RSCPropertyTreeWriter &tree) const {
    tree.add("seed", seed);
    environment.write(tree.addChild("environment"));
    time.write(tree.addChild("time"));
    soft_body.write(tree.addChild("soft_body"));
    rigid_body.write(tree.addChild("rigid_body"));
    contact.write(tree.addChild("contact"));
    sentinel.write(tree.addChild("sentinel"));
    vision.write(tree.addChild("vision"));
    control.write(tree.addChild("control"));
    record.write(tree.addChild("record"));
    runtime.write(tree.addChild("runtime"));
}

}  // namespace rsc

}  // namespace rise
