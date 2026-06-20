// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rsc_simulation.h"

#include "rs_rsc_util.h"

using namespace std;

namespace rise {

namespace rsc {

bool RSC_SimulationEnvironment::operator==(const RSC_SimulationEnvironment &other) const {
    return RSC_Base::operator==(other) && gravity_acceleration == other.gravity_acceleration;
}

void RSC_SimulationEnvironment::_validate() const {}

void RSC_SimulationEnvironment::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    gravity_acceleration = rsc_tree.get("GravityAcceleration", gravity_acceleration);
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

void RSC_SimulationTime::_validate() const {
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
}

void RSC_SimulationTime::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    if (rsc_tree.isChildExist("Dt")) dt = rsc_tree.get<RFloat>("Dt");
    dt_ratio = rsc_tree.get("DtRatio", dt_ratio);
    dt_update_interval = rsc_tree.get("DtUpdateInterval", dt_update_interval);
    if (rsc_tree.isChildExist("DtSoftBodyAndRigidBodyUpdate"))
        dt_softbody_and_rigidbody_update = rsc_tree.get<RFloat>("DtSoftBodyAndRigidBodyUpdate");
    if (rsc_tree.isChildExist("DtRigidBodyContactAndConstraintUpdate"))
        dt_rigidbody_contact_and_constraint_update =
            rsc_tree.get<RFloat>("DtRigidBodyContactAndConstraintUpdate");
    if (rsc_tree.isChildExist("DtContactUpdate"))
        dt_contact_update = rsc_tree.get<RFloat>("DtContactUpdate");
    if (rsc_tree.isChildExist("DtMechanicalSignalUpdate"))
        dt_mechanical_signal_update = rsc_tree.get<RFloat>("DtMechanicalSignalUpdate");
    if (rsc_tree.isChildExist("DtSentinelUpdate"))
        dt_sentinel_update = rsc_tree.get<RFloat>("DtSentinelUpdate");
}

bool RSC_SimulationSoftBody::operator==(const RSC_SimulationSoftBody &other) const {
    return RSC_Base::operator==(other) &&
           internal_damping_z == other.internal_damping_z &&
           contact_damping_z == other.contact_damping_z &&
           global_damping_z == other.global_damping_z &&
           integration_frequency_multiplier == other.integration_frequency_multiplier &&
           link_max_compression == other.link_max_compression;
}

void RSC_SimulationSoftBody::_validate() const {
    RSC_EXPECT_GE(RSC_SimulationSoftBody, internal_damping_z, 0);
    RSC_EXPECT_LE(RSC_SimulationSoftBody, internal_damping_z, 1);
    RSC_EXPECT_GE(RSC_SimulationSoftBody, contact_damping_z, 0);
    RSC_EXPECT_LE(RSC_SimulationSoftBody, contact_damping_z, 1);
    RSC_EXPECT_GE(RSC_SimulationSoftBody, global_damping_z, 0);
    RSC_EXPECT_LE(RSC_SimulationSoftBody, global_damping_z, 1);
    RSC_EXPECT_GT(RSC_SimulationSoftBody, integration_frequency_multiplier, 0);
    RSC_EXPECT_GE(RSC_SimulationSoftBody, link_max_compression, 0);
    RSC_EXPECT_LT(RSC_SimulationSoftBody, link_max_compression, 1);
}

void RSC_SimulationSoftBody::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    internal_damping_z = rsc_tree.get("InternalDampingZ", internal_damping_z);
    contact_damping_z = rsc_tree.get("ContactDampingZ", contact_damping_z);
    global_damping_z = rsc_tree.get("GlobalDampingZ", global_damping_z);
    integration_frequency_multiplier =
        rsc_tree.get("IntegrationFrequencyMultiplier", integration_frequency_multiplier);
    link_max_compression = rsc_tree.get("LinkMaxCompression", link_max_compression);
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

void RSC_SimulationRigidBody::_validate() const {
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
}

void RSC_SimulationRigidBody::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    constraint_iterations = rsc_tree.get("ConstraintIterations", constraint_iterations);
    constraint_position_correction_ratio =
        rsc_tree.get("ConstraintPositionCorrectionRatio", constraint_position_correction_ratio);
    constraint_warm_start_damping_z =
        rsc_tree.get("ConstraintWarmStartDampingZ", constraint_warm_start_damping_z);
    constraint_stiffness_initial =
        rsc_tree.get("ConstraintStiffnessInitial", constraint_stiffness_initial);
    constraint_stiffness_min_threshold =
        rsc_tree.get("ConstraintStiffnessMinThreshold", constraint_stiffness_min_threshold);
    constraint_stiffness_max_threshold =
        rsc_tree.get("ConstraintStiffnessMaxThreshold", constraint_stiffness_max_threshold);
    constraint_stiffness_auto_adjust_watt_threshold = rsc_tree.get(
        "ConstraintStiffnessAutoAdjustWattThreshold",
        constraint_stiffness_auto_adjust_watt_threshold
    );
    constraint_stiffness_auto_adjust_step = rsc_tree.get(
        "ConstraintStiffnessAutoAdjustStep", constraint_stiffness_auto_adjust_step
    );
    contact_iterations = rsc_tree.get("ContactIterations", contact_iterations);
    contact_position_correction_iterations = rsc_tree.get(
        "ContactPositionCorrectionIterations", contact_position_correction_iterations
    );
    contact_position_correction_ratio =
        rsc_tree.get("ContactPositionCorrectionRatio", contact_position_correction_ratio);
    contact_warm_start_damping_z =
        rsc_tree.get("ContactWarmStartDampingZ", contact_warm_start_damping_z);
    contact_slop = rsc_tree.get("ContactSlop", contact_slop);
    linear_damping_z = rsc_tree.get("LinearDampingZ", linear_damping_z);
    angular_damping_z = rsc_tree.get("AngularDampingZ", angular_damping_z);
}

bool RSC_SimulationContact::operator==(const RSC_SimulationContact &other) const {
    return RSC_Base::operator==(other) &&
           bvh_rebuild_sub_interval == other.bvh_rebuild_sub_interval &&
           bvh_refit_sub_interval == other.bvh_refit_sub_interval;
}

void RSC_SimulationContact::_validate() const {
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
}

void RSC_SimulationContact::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    bvh_rebuild_sub_interval = rsc_tree.get("BvhRebuildSubInterval", bvh_rebuild_sub_interval);
    bvh_refit_sub_interval = rsc_tree.get("BvhRefitSubInterval", bvh_refit_sub_interval);
}

bool RSC_SimulationVisionTrace::operator==(const RSC_SimulationVisionTrace &other) const {
    return RSC_Base::operator==(other) && firefly_cap == other.firefly_cap &&
           russian_roulette_start_depth == other.russian_roulette_start_depth;
}

void RSC_SimulationVisionTrace::_validate() const {
    RSC_EXPECT_GT(RSC_SimulationVisionTrace, firefly_cap, 0);
}

void RSC_SimulationVisionTrace::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    firefly_cap = rsc_tree.get<RFloat>("FireflyCap", firefly_cap);
    russian_roulette_start_depth =
        rsc_tree.get<RSize>("RussianRouletteStartDepth", russian_roulette_start_depth);
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

void RSC_SimulationVisionReSTIR::_validate() const {
    RSC_EXPECT_GT(RSC_SimulationVisionReSTIR, disocclusion_position_threshold, 0);
    RSC_EXPECT_GT(RSC_SimulationVisionReSTIR, disocclusion_normal_threshold, 0);
    RSC_EXPECT_LE(RSC_SimulationVisionReSTIR, disocclusion_normal_threshold, 1);
    RSC_EXPECT_GE(RSC_SimulationVisionReSTIR, max_specular_chain, 1);
    RSC_EXPECT_GE(RSC_SimulationVisionReSTIR, di_m_cap, 1);
    RSC_EXPECT_GT(RSC_SimulationVisionReSTIR, di_spatial_radius, 0);
    RSC_EXPECT_GE(RSC_SimulationVisionReSTIR, gi_m_cap, 1);
    RSC_EXPECT_GT(RSC_SimulationVisionReSTIR, gi_spatial_radius, 0);
}

void RSC_SimulationVisionReSTIR::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    is_enabled = rsc_tree.get<bool>("IsEnabled", is_enabled);
    disocclusion_position_threshold =
        rsc_tree.get<RFloat>("DisocclusionPositionThreshold", disocclusion_position_threshold);
    disocclusion_normal_threshold =
        rsc_tree.get<RFloat>("DisocclusionNormalThreshold", disocclusion_normal_threshold);
    max_specular_chain = rsc_tree.get<RSize>("MaxSpecularChain", max_specular_chain);
    di_m_cap = rsc_tree.get<RSize>("DiMCap", di_m_cap);
    di_spatial_neighbors = rsc_tree.get<RSize>("DiSpatialNeighbors", di_spatial_neighbors);
    di_spatial_radius = rsc_tree.get<RFloat>("DiSpatialRadius", di_spatial_radius);
    is_gi_bounce_nee_enabled =
        rsc_tree.get<bool>("IsGiBounceNeeEnabled", is_gi_bounce_nee_enabled);
    gi_m_cap = rsc_tree.get<RSize>("GiMCap", gi_m_cap);
    gi_spatial_neighbors = rsc_tree.get<RSize>("GiSpatialNeighbors", gi_spatial_neighbors);
    gi_spatial_radius = rsc_tree.get<RFloat>("GiSpatialRadius", gi_spatial_radius);
    gi_validation_period = rsc_tree.get<RSize>("GiValidationPeriod", gi_validation_period);
}

bool RSC_SimulationVisionDenoiseNone::operator==(
    const RSC_SimulationVisionDenoiseNone &other
) const {
    return RSC_Base::operator==(other);
}

void RSC_SimulationVisionDenoiseNone::_validate() const {}

void RSC_SimulationVisionDenoiseNone::_read(const RS_CharacterizationPropertyTree &) {}

bool RSC_SimulationVisionDenoiseAtrous::operator==(
    const RSC_SimulationVisionDenoiseAtrous &other
) const {
    return RSC_Base::operator==(other) && atrous_iterations == other.atrous_iterations &&
           atrous_min_passes == other.atrous_min_passes && sigma_color == other.sigma_color &&
           sigma_normal == other.sigma_normal && sigma_depth == other.sigma_depth;
}

void RSC_SimulationVisionDenoiseAtrous::_validate() const {
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseAtrous, atrous_iterations, 0);
    RSC_EXPECT_GE(RSC_SimulationVisionDenoiseAtrous, atrous_min_passes, 0);
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseAtrous, sigma_color, 0);
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseAtrous, sigma_normal, 0);
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseAtrous, sigma_depth, 0);
}

void RSC_SimulationVisionDenoiseAtrous::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    atrous_iterations = rsc_tree.get<RSize>("AtrousIterations", atrous_iterations);
    atrous_min_passes = rsc_tree.get<RSize>("AtrousMinPasses", atrous_min_passes);
    sigma_color = rsc_tree.get<RFloat>("SigmaColor", sigma_color);
    sigma_normal = rsc_tree.get<RFloat>("SigmaNormal", sigma_normal);
    sigma_depth = rsc_tree.get<RFloat>("SigmaDepth", sigma_depth);
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

void RSC_SimulationVisionDenoiseSVGF::_validate() const {
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, atrous_iterations, 0);
    RSC_EXPECT_GE(RSC_SimulationVisionDenoiseSVGF, atrous_min_passes, 0);
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, sigma_color, 0);
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, sigma_normal, 0);
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, sigma_depth, 0);
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, temporal_alpha, 0);
    RSC_EXPECT_GT(RSC_SimulationVisionDenoiseSVGF, temporal_alpha_moments, 0);
    RSC_EXPECT_GE(RSC_SimulationVisionDenoiseSVGF, temporal_min_history, 1);
}

void RSC_SimulationVisionDenoiseSVGF::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    atrous_iterations = rsc_tree.get<RSize>("AtrousIterations", atrous_iterations);
    atrous_min_passes = rsc_tree.get<RSize>("AtrousMinPasses", atrous_min_passes);
    sigma_color = rsc_tree.get<RFloat>("SigmaColor", sigma_color);
    sigma_normal = rsc_tree.get<RFloat>("SigmaNormal", sigma_normal);
    sigma_depth = rsc_tree.get<RFloat>("SigmaDepth", sigma_depth);
    temporal_alpha = rsc_tree.get<RFloat>("TemporalAlpha", temporal_alpha);
    temporal_alpha_moments =
        rsc_tree.get<RFloat>("TemporalAlphaMoments", temporal_alpha_moments);
    temporal_min_history = rsc_tree.get<RFloat>("TemporalMinHistory", temporal_min_history);
}

bool RSC_SimulationVision::operator==(const RSC_SimulationVision &other) const {
    return RSC_Base::operator==(other) && trace == other.trace && restir == other.restir &&
           denoise == other.denoise;
}

void RSC_SimulationVision::_validate() const {
    trace.validate();
    restir.validate();
    denoise.visit([](const auto &d) { d.validate(); });
}

void RSC_SimulationVision::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    if (rsc_tree.isChildExist("Trace")) {
        trace.read(rsc_tree.getChild("Trace"), false);
    }
    if (rsc_tree.isChildExist("ReSTIR")) {
        restir.read(rsc_tree.getChild("ReSTIR"), false);
    }
    if (rsc_tree.isChildExist("Denoise")) {
        auto denoise_tree = rsc_tree.getChild("Denoise");
        auto children = denoise_tree.getChildren();
        if (children.empty()) {
            denoise = RSC_SimulationVisionDenoiseNone{};
        } else if (children.size() != 1) {
            RS_THROW(
                RS_InvalidCharacterizationException("Denoise must have exactly one alternative child")
            );
        } else {
            auto &[tag, body] = children.front();
            if (tag == "None") {
                denoise = RSC_SimulationVisionDenoiseNone{};
            } else if (tag == "Atrous") {
                RSC_SimulationVisionDenoiseAtrous d;
                d.read(body, false);
                denoise = d;
            } else if (tag == "SVGF") {
                RSC_SimulationVisionDenoiseSVGF d;
                d.read(body, false);
                denoise = d;
            } else {
                RS_THROW(RS_InvalidCharacterizationException("Unknown denoise tag: {}", tag));
            }
        }
    }
}

bool RSC_SimulationControl::operator==(const RSC_SimulationControl &other) const {
    return RSC_Base::operator==(other) && frequency == other.frequency;
}

void RSC_SimulationControl::_validate() const {
    RSC_EXPECT_GT(RSC_SimulationControl, frequency, 0);
}

void RSC_SimulationControl::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    frequency = rsc_tree.get("Frequency", frequency);
}

bool RSC_SimulationRecord::operator==(const RSC_SimulationRecord &other) const {
    return RSC_Base::operator==(other) && frequency == other.frequency;
}

void RSC_SimulationRecord::_validate() const { RSC_EXPECT_GT(RSC_SimulationRecord, frequency, 0); }

void RSC_SimulationRecord::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    frequency = rsc_tree.get("Frequency", frequency);
}

bool RSC_SimulationSentinel::operator==(const RSC_SimulationSentinel &other) const {
    return RSC_Base::operator==(other) && is_enabled == other.is_enabled;
}

void RSC_SimulationSentinel::_validate() const {}

void RSC_SimulationSentinel::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    is_enabled = rsc_tree.get("IsEnabled", is_enabled);
}

bool RSC_SimulationRuntime::operator==(const RSC_SimulationRuntime &other) const {
    return RSC_Base::operator==(other) && max_time == other.max_time &&
           record_buffer_size == other.record_buffer_size &&
           is_save_record == other.is_save_record && compute_accuracy == other.compute_accuracy &&
           memory_accuracy == other.memory_accuracy && log_level == other.log_level;
}

void RSC_SimulationRuntime::_validate() const {
    RSC_EXPECT_GT(RSC_SimulationRuntime, max_time, 0);
    RSC_EXPECT_GT(RSC_SimulationRuntime, record_buffer_size, 0);
}

void RSC_SimulationRuntime::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    max_time = rsc_tree.get("MaxTime", max_time);
    record_buffer_size = rsc_tree.get("RecordBufferSize", record_buffer_size);
    is_save_record = rsc_tree.get("IsSaveRecord", is_save_record);
    if (rsc_tree.isChildExist("ComputeAccuracy")) {
        readEnum(
            rsc_tree.get<std::string>("ComputeAccuracy"), compute_accuracy, "ComputeAccuracy"
        );
    }
    if (rsc_tree.isChildExist("MemoryAccuracy")) {
        readEnum(rsc_tree.get<std::string>("MemoryAccuracy"), memory_accuracy, "MemoryAccuracy");
    }
    log_level = rsc_tree.get("LogLevel", log_level);
}

bool RSC_Simulation::operator==(const RSC_Simulation &other) const {
    return RSC_Base::operator==(other) && seed == other.seed &&
           stop_condition.expr == other.stop_condition.expr && environment == other.environment &&
           time == other.time && soft_body == other.soft_body && rigid_body == other.rigid_body &&
           contact == other.contact && sentinel == other.sentinel && vision == other.vision &&
           control == other.control && record == other.record && runtime == other.runtime;
}

void RSC_Simulation::_validate() const {
    environment.validate();
    time.validate();
    soft_body.validate();
    rigid_body.validate();
    contact.validate();
    sentinel.validate();
    vision.validate();
    control.validate();
    record.validate();
    runtime.validate();
}

void RSC_Simulation::_read(const RS_CharacterizationPropertyTree &rsc_tree) {
    seed = rsc_tree.get("Seed", seed);

    if (rsc_tree.isChildExist("StopCondition")) {
        auto stop_condition_str = rsc_tree.get<string>("StopCondition", "");
        try {
            stop_condition = stop_condition_str;
        } catch (const RS_InvalidExpressionException &e) {
            RS_THROW(RS_InvalidCharacterizationException("Invalid stop condition: {}", e.what()));
        }
    }
    if (stop_condition.empty() && rsc_tree.logger) {
        rsc_tree.logger->warn("Stop condition not set");
    }

    if (rsc_tree.isChildExist("Environment")) {
        environment.read(rsc_tree.getChild("Environment"), false);
    }
    if (rsc_tree.isChildExist("Time")) {
        time.read(rsc_tree.getChild("Time"), false);
    }
    if (rsc_tree.isChildExist("SoftBody")) {
        soft_body.read(rsc_tree.getChild("SoftBody"), false);
    }
    if (rsc_tree.isChildExist("RigidBody")) {
        rigid_body.read(rsc_tree.getChild("RigidBody"), false);
    }
    if (rsc_tree.isChildExist("Contact")) {
        contact.read(rsc_tree.getChild("Contact"), false);
    }
    if (rsc_tree.isChildExist("Sentinel")) {
        sentinel.read(rsc_tree.getChild("Sentinel"), false);
    }
    if (rsc_tree.isChildExist("Vision")) {
        vision.read(rsc_tree.getChild("Vision"), false);
    }
    if (rsc_tree.isChildExist("Control")) {
        control.read(rsc_tree.getChild("Control"), false);
    }
    if (rsc_tree.isChildExist("Record")) {
        record.read(rsc_tree.getChild("Record"), false);
    }
    if (rsc_tree.isChildExist("Runtime")) {
        runtime.read(rsc_tree.getChild("Runtime"), false);
    }
}

}  // namespace rsc

}  // namespace rise
