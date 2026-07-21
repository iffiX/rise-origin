// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/rspy_init_simulation.h"

void initRSPYSimulation(py::module &m) {
    /***************************************************************
     * Simulation
     ***************************************************************/
    rspyRSC<RSC_SimulationEnvironment, RSC_Base>(m, "RSC_SimulationEnvironment")
        .def(py::init<>())
        .def_readwrite("gravity_acceleration", &RSC_SimulationEnvironment::gravity_acceleration);

    rspyRSC<RSC_SimulationTime, RSC_Base>(m, "RSC_SimulationTime")
        .def(py::init<>())
        .def_property(
            "dt",
            [](RSC_SimulationTime &self) { return rspyOptionalGet(self.dt); },
            [](RSC_SimulationTime &self, py::object arg) { rspyOptionalSet(self.dt, arg); }
        )
        .def_readwrite("dt_ratio", &RSC_SimulationTime::dt_ratio)
        .def_readwrite("dt_update_interval", &RSC_SimulationTime::dt_update_interval)
        .def_property(
            "dt_softbody_and_rigidbody_update",
            [](RSC_SimulationTime &self) {
                return rspyOptionalGet(self.dt_softbody_and_rigidbody_update);
            },
            [](RSC_SimulationTime &self, py::object arg) {
                rspyOptionalSet(self.dt_softbody_and_rigidbody_update, arg);
            }
        )
        .def_property(
            "dt_rigidbody_contact_and_constraint_update",
            [](RSC_SimulationTime &self) {
                return rspyOptionalGet(self.dt_rigidbody_contact_and_constraint_update);
            },
            [](RSC_SimulationTime &self, py::object arg) {
                rspyOptionalSet(self.dt_rigidbody_contact_and_constraint_update, arg);
            }
        )
        .def_property(
            "dt_contact_update",
            [](RSC_SimulationTime &self) { return rspyOptionalGet(self.dt_contact_update); },
            [](RSC_SimulationTime &self, py::object arg) {
                rspyOptionalSet(self.dt_contact_update, arg);
            }
        )
        .def_property(
            "dt_mechanical_signal_update",
            [](RSC_SimulationTime &self) {
                return rspyOptionalGet(self.dt_mechanical_signal_update);
            },
            [](RSC_SimulationTime &self, py::object arg) {
                rspyOptionalSet(self.dt_mechanical_signal_update, arg);
            }
        )
        .def_property(
            "dt_sentinel_update",
            [](RSC_SimulationTime &self) { return rspyOptionalGet(self.dt_sentinel_update); },
            [](RSC_SimulationTime &self, py::object arg) {
                rspyOptionalSet(self.dt_sentinel_update, arg);
            }
        )
        .def_readwrite("is_aligned_to_wallclock", &RSC_SimulationTime::is_aligned_to_wallclock);

    rspyRSC<RSC_SimulationSoftBody, RSC_Base>(m, "RSC_SimulationSoftBody")
        .def(py::init<>())
        .def_readwrite("internal_damping_z", &RSC_SimulationSoftBody::internal_damping_z)
        .def_readwrite("contact_damping_z", &RSC_SimulationSoftBody::contact_damping_z)
        .def_readwrite("global_damping_z", &RSC_SimulationSoftBody::global_damping_z)
        .def_readwrite(
            "integration_frequency_multiplier",
            &RSC_SimulationSoftBody::integration_frequency_multiplier
        )
        .def_readwrite("link_max_compression", &RSC_SimulationSoftBody::link_max_compression);

    rspyRSC<RSC_SimulationRigidBody, RSC_Base>(m, "RSC_SimulationRigidBody")
        .def(py::init<>())
        .def_readwrite("constraint_iterations", &RSC_SimulationRigidBody::constraint_iterations)
        .def_readwrite(
            "constraint_position_correction_ratio",
            &RSC_SimulationRigidBody::constraint_position_correction_ratio
        )
        .def_readwrite(
            "constraint_warm_start_damping_z",
            &RSC_SimulationRigidBody::constraint_warm_start_damping_z
        )
        .def_readwrite(
            "constraint_stiffness_initial", &RSC_SimulationRigidBody::constraint_stiffness_initial
        )
        .def_readwrite(
            "constraint_stiffness_min_threshold",
            &RSC_SimulationRigidBody::constraint_stiffness_min_threshold
        )
        .def_readwrite(
            "constraint_stiffness_max_threshold",
            &RSC_SimulationRigidBody::constraint_stiffness_max_threshold
        )
        .def_readwrite(
            "constraint_stiffness_auto_adjust_watt_threshold",
            &RSC_SimulationRigidBody::constraint_stiffness_auto_adjust_watt_threshold
        )
        .def_readwrite(
            "constraint_stiffness_auto_adjust_step",
            &RSC_SimulationRigidBody::constraint_stiffness_auto_adjust_step
        )
        .def_readwrite("contact_iterations", &RSC_SimulationRigidBody::contact_iterations)
        .def_readwrite(
            "contact_position_correction_iterations",
            &RSC_SimulationRigidBody::contact_position_correction_iterations
        )
        .def_readwrite(
            "contact_position_correction_ratio",
            &RSC_SimulationRigidBody::contact_position_correction_ratio
        )
        .def_readwrite(
            "contact_warm_start_damping_z", &RSC_SimulationRigidBody::contact_warm_start_damping_z
        )
        .def_readwrite("contact_slop", &RSC_SimulationRigidBody::contact_slop)
        .def_readwrite("linear_damping_z", &RSC_SimulationRigidBody::linear_damping_z)
        .def_readwrite("angular_damping_z", &RSC_SimulationRigidBody::angular_damping_z);

    rspyRSC<RSC_SimulationContact, RSC_Base>(m, "RSC_SimulationContact")
        .def(py::init<>())
        .def_readwrite("bvh_rebuild_sub_interval", &RSC_SimulationContact::bvh_rebuild_sub_interval)
        .def_readwrite("bvh_refit_sub_interval", &RSC_SimulationContact::bvh_refit_sub_interval);

    rspyRSC<RSC_SimulationVisionTrace, RSC_Base>(m, "RSC_SimulationVisionTrace")
        .def(py::init<>())
        .def_readwrite("firefly_cap", &RSC_SimulationVisionTrace::firefly_cap)
        .def_readwrite(
            "russian_roulette_start_depth", &RSC_SimulationVisionTrace::russian_roulette_start_depth
        );

    rspyRSC<RSC_SimulationVisionReSTIR, RSC_Base>(m, "RSC_SimulationVisionReSTIR")
        .def(py::init<>())
        .def_readwrite("is_enabled", &RSC_SimulationVisionReSTIR::is_enabled)
        .def_readwrite(
            "disocclusion_position_threshold",
            &RSC_SimulationVisionReSTIR::disocclusion_position_threshold
        )
        .def_readwrite(
            "disocclusion_normal_threshold",
            &RSC_SimulationVisionReSTIR::disocclusion_normal_threshold
        )
        .def_readwrite("max_specular_chain", &RSC_SimulationVisionReSTIR::max_specular_chain)
        .def_readwrite("di_m_cap", &RSC_SimulationVisionReSTIR::di_m_cap)
        .def_readwrite("di_spatial_neighbors", &RSC_SimulationVisionReSTIR::di_spatial_neighbors)
        .def_readwrite("di_spatial_radius", &RSC_SimulationVisionReSTIR::di_spatial_radius)
        .def_readwrite(
            "is_gi_bounce_nee_enabled", &RSC_SimulationVisionReSTIR::is_gi_bounce_nee_enabled
        )
        .def_readwrite("gi_m_cap", &RSC_SimulationVisionReSTIR::gi_m_cap)
        .def_readwrite("gi_spatial_neighbors", &RSC_SimulationVisionReSTIR::gi_spatial_neighbors)
        .def_readwrite("gi_spatial_radius", &RSC_SimulationVisionReSTIR::gi_spatial_radius)
        .def_readwrite("gi_validation_period", &RSC_SimulationVisionReSTIR::gi_validation_period);

    rspyRSC<RSC_SimulationVisionDenoiseNone, RSC_Base>(m, "RSC_SimulationVisionDenoiseNone")
        .def(py::init<>());

    rspyRSC<RSC_SimulationVisionDenoiseAtrous, RSC_Base>(m, "RSC_SimulationVisionDenoiseAtrous")
        .def(py::init<>())
        .def_readwrite("atrous_iterations", &RSC_SimulationVisionDenoiseAtrous::atrous_iterations)
        .def_readwrite("atrous_min_passes", &RSC_SimulationVisionDenoiseAtrous::atrous_min_passes)
        .def_readwrite("sigma_color", &RSC_SimulationVisionDenoiseAtrous::sigma_color)
        .def_readwrite("sigma_normal", &RSC_SimulationVisionDenoiseAtrous::sigma_normal)
        .def_readwrite("sigma_depth", &RSC_SimulationVisionDenoiseAtrous::sigma_depth);

    rspyRSC<RSC_SimulationVisionDenoiseSVGF, RSC_Base>(m, "RSC_SimulationVisionDenoiseSVGF")
        .def(py::init<>())
        .def_readwrite("atrous_iterations", &RSC_SimulationVisionDenoiseSVGF::atrous_iterations)
        .def_readwrite("atrous_min_passes", &RSC_SimulationVisionDenoiseSVGF::atrous_min_passes)
        .def_readwrite("sigma_color", &RSC_SimulationVisionDenoiseSVGF::sigma_color)
        .def_readwrite("sigma_normal", &RSC_SimulationVisionDenoiseSVGF::sigma_normal)
        .def_readwrite("sigma_depth", &RSC_SimulationVisionDenoiseSVGF::sigma_depth)
        .def_readwrite("temporal_alpha", &RSC_SimulationVisionDenoiseSVGF::temporal_alpha)
        .def_readwrite(
            "temporal_alpha_moments", &RSC_SimulationVisionDenoiseSVGF::temporal_alpha_moments
        )
        .def_readwrite(
            "temporal_min_history", &RSC_SimulationVisionDenoiseSVGF::temporal_min_history
        );

    rspyRSC<RSC_SimulationVision, RSC_Base>(m, "RSC_SimulationVision")
        .def(py::init<>())
        .def_readwrite("trace", &RSC_SimulationVision::trace)
        .def_readwrite("restir", &RSC_SimulationVision::restir)
        .def_property(
            "denoise",
            [](RSC_SimulationVision &self) { return rspyVariantGet(self.denoise); },
            [](RSC_SimulationVision &self, py::object arg) { rspyVariantSet(self.denoise, arg); }
        );

    rspyRSC<RSC_SimulationSentinel, RSC_Base>(m, "RSC_SimulationSentinel")
        .def(py::init<>())
        .def_readwrite("is_enabled", &RSC_SimulationSentinel::is_enabled)
        .def_readwrite("degenerate_min_dt", &RSC_SimulationSentinel::degenerate_min_dt);

    rspyRSC<RSC_SimulationControl, RSC_Base>(m, "RSC_SimulationControl")
        .def(py::init<>())
        .def_readwrite("frequency", &RSC_SimulationControl::frequency);

    rspyRSC<RSC_SimulationRecord, RSC_Base>(m, "RSC_SimulationRecord")
        .def(py::init<>())
        .def_readwrite("frequency", &RSC_SimulationRecord::frequency);

    rspyRSC<RSC_SimulationRuntime, RSC_Base>(m, "RSC_SimulationRuntime")
        .def(py::init<>())
        .def_readwrite("is_paused_on_start", &RSC_SimulationRuntime::is_paused_on_start)
        .def_readwrite("max_time", &RSC_SimulationRuntime::max_time)
        .def_readwrite("record_buffer_size", &RSC_SimulationRuntime::record_buffer_size)
        .def_readwrite("is_save_record", &RSC_SimulationRuntime::is_save_record)
        .def_readwrite("compute_accuracy", &RSC_SimulationRuntime::compute_accuracy)
        .def_readwrite("memory_accuracy", &RSC_SimulationRuntime::memory_accuracy)
        .def_readwrite("log_level", &RSC_SimulationRuntime::log_level);

    rspyRSC<RSC_Simulation, RSC_Base>(m, "RSC_Simulation")
        .def(py::init<>())
        .def_readwrite("seed", &RSC_Simulation::seed)
        .def_readwrite("stop_condition", &RSC_Simulation::stop_condition)
        .def_readwrite("environment", &RSC_Simulation::environment)
        .def_readwrite("time", &RSC_Simulation::time)
        .def_readwrite("soft_body", &RSC_Simulation::soft_body)
        .def_readwrite("rigid_body", &RSC_Simulation::rigid_body)
        .def_readwrite("contact", &RSC_Simulation::contact)
        .def_readwrite("vision", &RSC_Simulation::vision)
        .def_readwrite("sentinel", &RSC_Simulation::sentinel)
        .def_readwrite("control", &RSC_Simulation::control)
        .def_readwrite("record", &RSC_Simulation::record)
        .def_readwrite("runtime", &RSC_Simulation::runtime);
}
