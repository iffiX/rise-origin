// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/rsc/rs_simulation_config.h"

#include "rs_rsc_util.h"

using namespace std;

namespace rise {

void RS_SimulationConfig::_validate() const {
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, runtime_dt_update_interval, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, runtime_rigidbody_update_interval, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, runtime_contact_update_interval, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, runtime_contact_bvh_rebuild_sub_interval, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, runtime_contact_bvh_refit_sub_interval, 0);
    if (runtime_contact_bvh_rebuild_sub_interval % runtime_contact_bvh_refit_sub_interval != 0) {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid RS_SimulationConfig: runtime_contact_bvh_rebuild_sub_interval ({}) "
            "must be a multiple of runtime_contact_bvh_refit_sub_interval ({})",
            runtime_contact_bvh_rebuild_sub_interval,
            runtime_contact_bvh_refit_sub_interval
        ));
    }
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, runtime_mechanical_signal_update_interval, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, runtime_divergence_check_interval, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, runtime_record_buffer_size, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, runtime_max_time, 0);
    if (runtime_log_level != "err" && runtime_log_level != "warn" &&
        runtime_log_level != "info" && runtime_log_level != "debug" &&
        runtime_log_level != "trace") {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid RS_SimulationConfig: runtime_log_level must be one of "
            "'err', 'warn', 'info', 'debug', 'trace', got '{}'",
            runtime_log_level
        ));
    }
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, integration_time_step, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, integration_dt_frac, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, integration_dt_frac, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, softbody_internal_damping_z, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, softbody_internal_damping_z, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, softbody_contact_damping_z, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, softbody_contact_damping_z, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, softbody_global_damping_z, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, softbody_global_damping_z, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_constraint_iterations, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_constraint_position_correction_ratio, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_constraint_position_correction_ratio, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_constraint_warm_start_damping_z, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_constraint_warm_start_damping_z, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_constraint_compliance_initial, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_constraint_compliance_initial, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_constraint_compliance_min_threshold, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_constraint_compliance_min_threshold, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_constraint_compliance_max_threshold, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_constraint_compliance_max_threshold, 1);
    RS_CONFIG_EXPECT_GE(
        RS_SimulationConfig, rigidbody_constraint_compliance_auto_adjust_watt_threshold, 0
    );
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_constraint_compliance_auto_adjust_step, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_constraint_compliance_auto_adjust_step, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_contact_iterations, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_contact_position_correction_iterations, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_contact_position_correction_ratio, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_contact_position_correction_ratio, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_contact_warm_start_damping_z, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_contact_warm_start_damping_z, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_contact_restitution, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_contact_restitution, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_contact_slop, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_linear_damping_z, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_linear_damping_z, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, rigidbody_angular_damping_z, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, rigidbody_angular_damping_z, 1);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, vision_trace_firefly_cap, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_trace_russian_roulette_start_depth, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, vision_restir_disocclusion_position_threshold, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, vision_restir_disocclusion_normal_threshold, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, vision_restir_disocclusion_normal_threshold, 1);
    // ReSTIR knobs accept 0 as a debug/disable setting: candidates=0 leaves the
    // reservoir empty, m_cap=0 disables temporal merge, neighbors/radius/chain=0
    // skip the corresponding loop. All branches are guarded against div-by-zero.
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_restir_max_specular_chain, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_restir_di_m_cap, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_restir_di_spatial_neighbors, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_restir_di_spatial_radius, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_restir_gi_m_cap, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_restir_gi_spatial_neighbors, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_restir_gi_spatial_radius, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_restir_gi_validation_period, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_denoise_atrous_iterations, 0);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_denoise_atrous_min_passes, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, vision_denoise_sigma_color, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, vision_denoise_sigma_normal, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, vision_denoise_sigma_depth, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, vision_denoise_temporal_alpha, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, vision_denoise_temporal_alpha, 1);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, vision_denoise_temporal_alpha_moments, 0);
    RS_CONFIG_EXPECT_LE(RS_SimulationConfig, vision_denoise_temporal_alpha_moments, 1);
    RS_CONFIG_EXPECT_GE(RS_SimulationConfig, vision_denoise_temporal_min_history, 1);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, record_frequency, 0);
    RS_CONFIG_EXPECT_GT(RS_SimulationConfig, control_frequency, 0);
}

void RS_SimulationConfig::_read(const RS_ConfigPropertyTree& config_tree) {
    // root of tree: RSC.Simulation
    seed = config_tree.get("Seed", seed);

    runtime_dt_update_interval =
        config_tree.get("RuntimeDtUpdateInterval", runtime_dt_update_interval);
    runtime_rigidbody_update_interval =
        config_tree.get("RuntimeRigidbodyUpdateInterval", runtime_rigidbody_update_interval);
    runtime_contact_update_interval =
        config_tree.get("RuntimeContactUpdateInterval", runtime_contact_update_interval);
    runtime_contact_bvh_rebuild_sub_interval = config_tree.get(
        "RuntimeContactBvhRebuildSubInterval", runtime_contact_bvh_rebuild_sub_interval
    );
    runtime_contact_bvh_refit_sub_interval = config_tree.get(
        "RuntimeContactBvhRefitSubInterval", runtime_contact_bvh_refit_sub_interval
    );
    runtime_mechanical_signal_update_interval = config_tree.get(
        "RuntimeMechanicalSignalUpdateInterval", runtime_mechanical_signal_update_interval
    );
    runtime_divergence_check_interval =
        config_tree.get("RuntimeDivergenceCheckInterval", runtime_divergence_check_interval);
    runtime_record_buffer_size =
        config_tree.get("RuntimeRecordBufferSize", runtime_record_buffer_size);
    runtime_max_time = config_tree.get("RuntimeMaxTime", runtime_max_time);
    runtime_save_record = config_tree.get("RuntimeSaveRecord", runtime_save_record);
    if (config_tree.isChildExist("RuntimeSimulationComputeAccuracy")) {
        runtime_simulation_compute_accuracy = readFloatType(
            config_tree.get<string>("RuntimeSimulationComputeAccuracy"),
            "runtime_simulation_compute_accuracy"
        );
    }
    if (config_tree.isChildExist("RuntimeSimulationMemoryAccuracy")) {
        runtime_simulation_memory_accuracy = readFloatType(
            config_tree.get<string>("RuntimeSimulationMemoryAccuracy"),
            "runtime_simulation_memory_accuracy"
        );
    }
    runtime_log_level = config_tree.get<string>("RuntimeLogLevel", runtime_log_level);

    integration_time_step = config_tree.get("IntegrationTimeStep", integration_time_step);
    integration_dt_frac = config_tree.get("IntegrationDtFrac", integration_dt_frac);

    softbody_internal_damping_z =
        config_tree.get("SoftbodyInternalDampingZ", softbody_internal_damping_z);
    softbody_contact_damping_z =
        config_tree.get("SoftbodyContactDampingZ", softbody_contact_damping_z);
    softbody_global_damping_z =
        config_tree.get("SoftbodyGlobalDampingZ", softbody_global_damping_z);

    rigidbody_constraint_iterations =
        config_tree.get("RigidbodyConstraintIterations", rigidbody_constraint_iterations);
    rigidbody_constraint_position_correction_ratio = config_tree.get(
        "RigidbodyConstraintPositionCorrectionRatio", rigidbody_constraint_position_correction_ratio
    );
    rigidbody_constraint_warm_start_damping_z = config_tree.get(
        "RigidbodyConstraintWarmStartDampingZ", rigidbody_constraint_warm_start_damping_z
    );
    rigidbody_constraint_compliance_initial = config_tree.get(
        "RigidbodyConstraintComplianceInitial", rigidbody_constraint_compliance_initial
    );
    rigidbody_constraint_compliance_min_threshold = config_tree.get(
        "RigidbodyConstraintComplianceMinThreshold",
        rigidbody_constraint_compliance_min_threshold
    );
    rigidbody_constraint_compliance_max_threshold = config_tree.get(
        "RigidbodyConstraintComplianceMaxThreshold",
        rigidbody_constraint_compliance_max_threshold
    );
    rigidbody_constraint_compliance_auto_adjust_watt_threshold = config_tree.get(
        "RigidbodyConstraintComplianceAutoAdjustWattThreshold",
        rigidbody_constraint_compliance_auto_adjust_watt_threshold
    );
    rigidbody_constraint_compliance_auto_adjust_step = config_tree.get(
        "RigidbodyConstraintComplianceAutoAdjustStep",
        rigidbody_constraint_compliance_auto_adjust_step
    );
    rigidbody_contact_iterations =
        config_tree.get("RigidbodyContactIterations", rigidbody_contact_iterations);
    rigidbody_contact_position_correction_iterations = config_tree.get(
        "RigidbodyContactPositionCorrectionIterations",
        rigidbody_contact_position_correction_iterations
    );
    rigidbody_contact_position_correction_ratio = config_tree.get(
        "RigidbodyContactPositionCorrectionRatio", rigidbody_contact_position_correction_ratio
    );
    rigidbody_contact_warm_start_damping_z = config_tree.get(
        "RigidbodyContactWarmStartDampingZ", rigidbody_contact_warm_start_damping_z
    );
    rigidbody_contact_restitution =
        config_tree.get("RigidbodyContactRestitution", rigidbody_contact_restitution);
    rigidbody_contact_slop = config_tree.get("RigidbodyContactSlop", rigidbody_contact_slop);
    rigidbody_linear_damping_z =
        config_tree.get("RigidbodyLinearDampingZ", rigidbody_linear_damping_z);
    rigidbody_angular_damping_z =
        config_tree.get("RigidbodyAngularDampingZ", rigidbody_angular_damping_z);

    vision_trace_firefly_cap =
        config_tree.get("VisionTraceFireflyCap", vision_trace_firefly_cap);
    vision_trace_russian_roulette_start_depth = config_tree.get(
        "VisionTraceRussianRouletteStartDepth", vision_trace_russian_roulette_start_depth
    );
    vision_is_restir_enabled =
        config_tree.get("VisionIsRestirEnabled", vision_is_restir_enabled);
    vision_restir_disocclusion_position_threshold = config_tree.get(
        "VisionRestirDisocclusionPositionThreshold",
        vision_restir_disocclusion_position_threshold
    );
    vision_restir_disocclusion_normal_threshold = config_tree.get(
        "VisionRestirDisocclusionNormalThreshold",
        vision_restir_disocclusion_normal_threshold
    );
    vision_restir_max_specular_chain = config_tree.get(
        "VisionRestirMaxSpecularChain", vision_restir_max_specular_chain
    );
    vision_restir_di_m_cap = config_tree.get("VisionRestirDiMCap", vision_restir_di_m_cap);
    vision_restir_di_spatial_neighbors = config_tree.get(
        "VisionRestirDiSpatialNeighbors", vision_restir_di_spatial_neighbors
    );
    vision_restir_di_spatial_radius = config_tree.get(
        "VisionRestirDiSpatialRadius", vision_restir_di_spatial_radius
    );
    vision_restir_gi_is_bounce_nee_enabled = config_tree.get(
        "VisionRestirGiIsBounceNeeEnabled", vision_restir_gi_is_bounce_nee_enabled
    );
    vision_restir_gi_m_cap = config_tree.get("VisionRestirGiMCap", vision_restir_gi_m_cap);
    vision_restir_gi_spatial_neighbors = config_tree.get(
        "VisionRestirGiSpatialNeighbors", vision_restir_gi_spatial_neighbors
    );
    vision_restir_gi_spatial_radius = config_tree.get(
        "VisionRestirGiSpatialRadius", vision_restir_gi_spatial_radius
    );
    vision_restir_gi_validation_period = config_tree.get(
        "VisionRestirGiValidationPeriod", vision_restir_gi_validation_period
    );
    auto denoise_type_str =
        config_tree.get<string>("VisionDenoiseType", "None");
    if (denoise_type_str == "None") {
        vision_denoise_type = RSE_VISION_DENOISE_NONE;
    } else if (denoise_type_str == "Atrous") {
        vision_denoise_type = RSE_VISION_DENOISE_ATROUS;
    } else if (denoise_type_str == "Svgf") {
        vision_denoise_type = RSE_VISION_DENOISE_SVGF;
    } else {
        RS_THROW(RS_InvalidConfigurationException(
            "Invalid vision denoise type: {}, available types are: None, Atrous, Svgf",
            denoise_type_str
        ));
    }
    vision_denoise_atrous_iterations =
        config_tree.get("VisionDenoiseAtrousIterations", vision_denoise_atrous_iterations);
    vision_denoise_atrous_min_passes =
        config_tree.get("VisionDenoiseAtrousMinPasses", vision_denoise_atrous_min_passes);
    vision_denoise_sigma_color =
        config_tree.get("VisionDenoiseSigmaColor", vision_denoise_sigma_color);
    vision_denoise_sigma_normal =
        config_tree.get("VisionDenoiseSigmaNormal", vision_denoise_sigma_normal);
    vision_denoise_sigma_depth =
        config_tree.get("VisionDenoiseSigmaDepth", vision_denoise_sigma_depth);
    vision_denoise_temporal_alpha = config_tree.get(
        "VisionDenoiseTemporalAlpha", vision_denoise_temporal_alpha
    );
    vision_denoise_temporal_alpha_moments = config_tree.get(
        "VisionDenoiseTemporalAlphaMoments", vision_denoise_temporal_alpha_moments
    );
    vision_denoise_temporal_min_history = config_tree.get(
        "VisionDenoiseTemporalMinHistory", vision_denoise_temporal_min_history
    );

    control_frequency = config_tree.get("ControlFrequency", control_frequency);
    record_frequency = config_tree.get("RecordFrequency", record_frequency);

    if (stop_condition.empty()) {
        auto stop_condition_str = config_tree.get<string>("StopCondition", "");
        try {
            stop_condition = stop_condition_str;
        } catch (const RS_InvalidExpressionException& e) {
            RS_THROW(RS_InvalidConfigurationException("Invalid stop condition: {}", e.what()));
        }
        if (stop_condition.empty())
            if (config_tree.logger) config_tree.logger->warn("Stop condition not set");
    }
}

}  // namespace rise