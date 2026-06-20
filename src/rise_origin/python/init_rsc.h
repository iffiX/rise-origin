// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef INIT_RSC_H
#define INIT_RSC_H

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "rise_origin/rise_origin.h"
#include "rise_origin/utils/serialization/rs_mp_serialization.h"

namespace py = pybind11;
using namespace rise;
using namespace rise::rsc;

PYBIND11_MAKE_OPAQUE(std::vector<RSC_StructureVoxelBody>);
PYBIND11_MAKE_OPAQUE(std::vector<RSC_StructureVoxelConstraint>);

PYBIND11_MAKE_OPAQUE(std::vector<RSC_EntityVariant>);
PYBIND11_MAKE_OPAQUE(std::vector<RSC_EssenceVariant>);
PYBIND11_MAKE_OPAQUE(std::vector<RSC_ResourceVariant>);
PYBIND11_MAKE_OPAQUE(std::vector<RSC_RecordVariant>);
PYBIND11_MAKE_OPAQUE(std::vector<RSC_SignalVariant>);
PYBIND11_MAKE_OPAQUE(std::vector<RSC_ControllerVariant>);
PYBIND11_MAKE_OPAQUE(std::vector<RSC_RecorderVariant>);

// Serialization helper functions
template <typename T>
py::bytes rspySerializeRSC(const T &rsc) {
    RS_MessagePackSerializer s;
    s.serialize(rsc);
    auto data = s.moveData();
    return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
}

template <typename T>
T rspyDeserializeRSC(const py::bytes &pybuf) {
    char *buffer;
    Py_ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(pybuf.ptr(), &buffer, &length)) {
        throw std::runtime_error("Failed to extract bytes contents");
    }

    T rsc;
    RS_MessagePackDeserializer d;
    d.setData(std::vector<uint8_t>(
        reinterpret_cast<const uint8_t *>(buffer),
        reinterpret_cast<const uint8_t *>(buffer) + length
    ));
    d.deserialize(rsc);
    return rsc;
}

// Variant-property helpers: expose an RS_Variant<Alternatives...> field as a
// Python attribute whose get returns the currently held alternative by reference
// and whose set dispatches on the concrete Python type.
template <typename... Alternatives>
py::object rspyVariantGet(RS_Variant<Alternatives...> &v) {
    if (v.index() == RS_NULL_INDEX) {
        return py::none();
    }
    py::object result = py::none();
    v.visit([&](auto &held) { result = py::cast(&held, py::return_value_policy::reference); });
    return result;
}

template <typename Alt, typename... Alternatives>
bool rspyVariantTryAssign(RS_Variant<Alternatives...> &v, py::object arg) {
    if (py::isinstance<Alt>(arg)) {
        v = arg.cast<Alt>();
        return true;
    }
    return false;
}

template <typename... Alternatives>
void rspyVariantSet(RS_Variant<Alternatives...> &v, py::object arg) {
    bool ok = false;
    // Fold-expression: try each alternative type in order.
    ((ok = ok || rspyVariantTryAssign<Alternatives, Alternatives...>(v, arg)), ...);
    if (!ok) {
        throw std::runtime_error("Unsupported type assigned to variant field");
    }
}

// Optional-property helpers: expose an RS_Optional<T> field as a Python attribute
// that reads None when empty and accepts None to clear it.
template <typename T>
py::object rspyOptionalGet(const RS_Optional<T> &o) {
    return o.empty() ? py::none() : py::cast(o.get());
}

template <typename T>
void rspyOptionalSet(RS_Optional<T> &o, py::object arg) {
    if (arg.is_none()) {
        o.reset();
    } else {
        o = arg.cast<T>();
    }
}

inline void initRSPYRSC(py::module &m) {
    /***************************************************************
     * Bases
     ***************************************************************/
    // Note: all RSC_Base classes are consolidated here. They must be bound
    // before any derived class references them.
    py::class_<RSC_Base>(m, "RSC_Base");

    py::class_<RSC_ResourceBase, RSC_Base>(m, "RSC_ResourceBase")
        .def_readwrite("name", &RSC_ResourceBase::name);

    py::class_<RSC_EssenceBase, RSC_Base>(m, "RSC_EssenceBase")
        .def_readwrite("name", &RSC_EssenceBase::name);

    py::class_<RSC_EntityBase, RSC_Base>(m, "RSC_EntityBase")
        .def_readwrite("name", &RSC_EntityBase::name);

    py::class_<RSC_RecordBase, RSC_Base>(m, "RSC_RecordBase")
        .def_readwrite("name", &RSC_RecordBase::name)
        .def_readwrite("target_controller_reference", &RSC_RecordBase::target_controller_reference)
        .def_readwrite("target_recorder_reference", &RSC_RecordBase::target_recorder_reference);

    py::class_<RSC_SignalBase, RSC_Base>(m, "RSC_SignalBase")
        .def_readwrite("name", &RSC_SignalBase::name)
        .def_readwrite("source_controller_name", &RSC_SignalBase::source_controller_name);

    py::class_<RSC_RecorderBase, RSC_Base>(m, "RSC_RecorderBase")
        .def_readwrite("name", &RSC_RecorderBase::name);

    py::class_<RSC_ControllerBase, RSC_Base>(m, "RSC_ControllerBase")
        .def_readwrite("name", &RSC_ControllerBase::name);

    /***************************************************************
     * Simulation
     ***************************************************************/
    py::class_<RSC_SimulationEnvironment, RSC_Base>(m, "RSC_SimulationEnvironment")
        .def(py::init<>())
        .def_readwrite("gravity_acceleration", &RSC_SimulationEnvironment::gravity_acceleration)
        .def(
            py::pickle(
                [](const RSC_SimulationEnvironment &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_SimulationEnvironment>(buffer);
                }
            )
        );

    py::class_<RSC_SimulationTime, RSC_Base>(m, "RSC_SimulationTime")
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
        .def(
            py::pickle(
                [](const RSC_SimulationTime &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SimulationTime>(buffer); }
            )
        );

    py::class_<RSC_SimulationSoftBody, RSC_Base>(m, "RSC_SimulationSoftBody")
        .def(py::init<>())
        .def_readwrite("internal_damping_z", &RSC_SimulationSoftBody::internal_damping_z)
        .def_readwrite("contact_damping_z", &RSC_SimulationSoftBody::contact_damping_z)
        .def_readwrite("global_damping_z", &RSC_SimulationSoftBody::global_damping_z)
        .def_readwrite(
            "integration_frequency_multiplier",
            &RSC_SimulationSoftBody::integration_frequency_multiplier
        )
        .def_readwrite("link_max_compression", &RSC_SimulationSoftBody::link_max_compression)
        .def(
            py::pickle(
                [](const RSC_SimulationSoftBody &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SimulationSoftBody>(buffer); }
            )
        );

    py::class_<RSC_SimulationRigidBody, RSC_Base>(m, "RSC_SimulationRigidBody")
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
        .def_readwrite("angular_damping_z", &RSC_SimulationRigidBody::angular_damping_z)
        .def(
            py::pickle(
                [](const RSC_SimulationRigidBody &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SimulationRigidBody>(buffer); }
            )
        );

    py::class_<RSC_SimulationContact, RSC_Base>(m, "RSC_SimulationContact")
        .def(py::init<>())
        .def_readwrite("bvh_rebuild_sub_interval", &RSC_SimulationContact::bvh_rebuild_sub_interval)
        .def_readwrite("bvh_refit_sub_interval", &RSC_SimulationContact::bvh_refit_sub_interval)
        .def(
            py::pickle(
                [](const RSC_SimulationContact &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SimulationContact>(buffer); }
            )
        );

    py::class_<RSC_SimulationVisionTrace, RSC_Base>(m, "RSC_SimulationVisionTrace")
        .def(py::init<>())
        .def_readwrite("firefly_cap", &RSC_SimulationVisionTrace::firefly_cap)
        .def_readwrite(
            "russian_roulette_start_depth", &RSC_SimulationVisionTrace::russian_roulette_start_depth
        )
        .def(
            py::pickle(
                [](const RSC_SimulationVisionTrace &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_SimulationVisionTrace>(buffer);
                }
            )
        );

    py::class_<RSC_SimulationVisionReSTIR, RSC_Base>(m, "RSC_SimulationVisionReSTIR")
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
        .def_readwrite("gi_validation_period", &RSC_SimulationVisionReSTIR::gi_validation_period)
        .def(
            py::pickle(
                [](const RSC_SimulationVisionReSTIR &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_SimulationVisionReSTIR>(buffer);
                }
            )
        );

    py::class_<RSC_SimulationVisionDenoiseNone, RSC_Base>(m, "RSC_SimulationVisionDenoiseNone")
        .def(py::init<>())
        .def(
            py::pickle(
                [](const RSC_SimulationVisionDenoiseNone &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_SimulationVisionDenoiseNone>(buffer);
                }
            )
        );

    py::class_<RSC_SimulationVisionDenoiseAtrous, RSC_Base>(m, "RSC_SimulationVisionDenoiseAtrous")
        .def(py::init<>())
        .def_readwrite("atrous_iterations", &RSC_SimulationVisionDenoiseAtrous::atrous_iterations)
        .def_readwrite("atrous_min_passes", &RSC_SimulationVisionDenoiseAtrous::atrous_min_passes)
        .def_readwrite("sigma_color", &RSC_SimulationVisionDenoiseAtrous::sigma_color)
        .def_readwrite("sigma_normal", &RSC_SimulationVisionDenoiseAtrous::sigma_normal)
        .def_readwrite("sigma_depth", &RSC_SimulationVisionDenoiseAtrous::sigma_depth)
        .def(
            py::pickle(
                [](const RSC_SimulationVisionDenoiseAtrous &self) {
                    return rspySerializeRSC(self);
                },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_SimulationVisionDenoiseAtrous>(buffer);
                }
            )
        );

    py::class_<RSC_SimulationVisionDenoiseSVGF, RSC_Base>(m, "RSC_SimulationVisionDenoiseSVGF")
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
        )
        .def(
            py::pickle(
                [](const RSC_SimulationVisionDenoiseSVGF &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_SimulationVisionDenoiseSVGF>(buffer);
                }
            )
        );

    py::class_<RSC_SimulationVision, RSC_Base>(m, "RSC_SimulationVision")
        .def(py::init<>())
        .def_readwrite("trace", &RSC_SimulationVision::trace)
        .def_readwrite("restir", &RSC_SimulationVision::restir)
        .def_property(
            "denoise",
            [](RSC_SimulationVision &self) { return rspyVariantGet(self.denoise); },
            [](RSC_SimulationVision &self, py::object arg) { rspyVariantSet(self.denoise, arg); }
        )
        .def(
            py::pickle(
                [](const RSC_SimulationVision &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SimulationVision>(buffer); }
            )
        );

    py::class_<RSC_SimulationSentinel, RSC_Base>(m, "RSC_SimulationSentinel")
        .def(py::init<>())
        .def_readwrite("is_enabled", &RSC_SimulationSentinel::is_enabled)
        .def(
            py::pickle(
                [](const RSC_SimulationSentinel &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SimulationSentinel>(buffer); }
            )
        );

    py::class_<RSC_SimulationControl, RSC_Base>(m, "RSC_SimulationControl")
        .def(py::init<>())
        .def_readwrite("frequency", &RSC_SimulationControl::frequency)
        .def(
            py::pickle(
                [](const RSC_SimulationControl &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SimulationControl>(buffer); }
            )
        );

    py::class_<RSC_SimulationRecord, RSC_Base>(m, "RSC_SimulationRecord")
        .def(py::init<>())
        .def_readwrite("frequency", &RSC_SimulationRecord::frequency)
        .def(
            py::pickle(
                [](const RSC_SimulationRecord &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SimulationRecord>(buffer); }
            )
        );

    py::class_<RSC_SimulationRuntime, RSC_Base>(m, "RSC_SimulationRuntime")
        .def(py::init<>())
        .def_readwrite("max_time", &RSC_SimulationRuntime::max_time)
        .def_readwrite("record_buffer_size", &RSC_SimulationRuntime::record_buffer_size)
        .def_readwrite("is_save_record", &RSC_SimulationRuntime::is_save_record)
        .def_readwrite("compute_accuracy", &RSC_SimulationRuntime::compute_accuracy)
        .def_readwrite("memory_accuracy", &RSC_SimulationRuntime::memory_accuracy)
        .def_readwrite("log_level", &RSC_SimulationRuntime::log_level)
        .def(
            py::pickle(
                [](const RSC_SimulationRuntime &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SimulationRuntime>(buffer); }
            )
        );

    py::class_<RSC_Simulation, RSC_Base>(m, "RSC_Simulation")
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
        .def_readwrite("runtime", &RSC_Simulation::runtime)
        .def(
            py::pickle(
                [](const RSC_Simulation &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_Simulation>(buffer); }
            )
        );

    /***************************************************************
     * Resources
     ***************************************************************/
    py::class_<RSC_TextureBase, RSC_ResourceBase>(m, "RSC_TextureBase");

    py::class_<RSC_TextureConstant, RSC_TextureBase>(m, "RSC_TextureConstant")
        .def(py::init<>())
        .def_readwrite("value", &RSC_TextureConstant::value)
        .def(
            py::pickle(
                [](const RSC_TextureConstant &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_TextureConstant>(buffer); }
            )
        );

    py::class_<RSC_TextureGrid, RSC_TextureBase>(m, "RSC_TextureGrid")
        .def(py::init<>())
        .def_readwrite("fill_color", &RSC_TextureGrid::fill_color)
        .def_readwrite("line_color", &RSC_TextureGrid::line_color)
        .def_readwrite("is_meters", &RSC_TextureGrid::is_meters)
        .def_readwrite("x_spacing", &RSC_TextureGrid::x_spacing)
        .def_readwrite("y_spacing", &RSC_TextureGrid::y_spacing)
        .def_readwrite("line_width", &RSC_TextureGrid::line_width)
        .def(
            py::pickle(
                [](const RSC_TextureGrid &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_TextureGrid>(buffer); }
            )
        );

    /***************************************************************
     * Essences
     ***************************************************************/
    py::class_<RSC_Material, RSC_EssenceBase>(m, "RSC_Material")
        .def(py::init<>())
        .def(
            py::pickle(
                [](const RSC_Material &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_Material>(buffer); }
            )
        );

    py::class_<RSC_MaterialUnaryProperty, RSC_EssenceBase>(m, "RSC_MaterialUnaryProperty")
        .def_readwrite("material_reference", &RSC_MaterialUnaryProperty::material_reference);

    py::class_<RSC_MaterialPhysicalPropertyBase, RSC_MaterialUnaryProperty>(
        m, "RSC_MaterialPhysicalPropertyBase"
    );

    py::class_<RSC_MaterialPhysicalPropertyFixed, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertyFixed"
    )
        .def(py::init<>())
        .def(
            py::pickle(
                [](const RSC_MaterialPhysicalPropertyFixed &self) {
                    return rspySerializeRSC(self);
                },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_MaterialPhysicalPropertyFixed>(buffer);
                }
            )
        );

    py::class_<RSC_MaterialPhysicalPropertySoft, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertySoft"
    )
        .def(py::init<>())
        .def_readwrite("elastic_modulus", &RSC_MaterialPhysicalPropertySoft::elastic_modulus)
        .def_readwrite("fail_stress", &RSC_MaterialPhysicalPropertySoft::fail_stress)
        .def_readwrite("density", &RSC_MaterialPhysicalPropertySoft::density)
        .def_readwrite("poissons_ratio", &RSC_MaterialPhysicalPropertySoft::poissons_ratio)
        .def_readwrite("min_expansion", &RSC_MaterialPhysicalPropertySoft::min_expansion)
        .def_readwrite("max_expansion", &RSC_MaterialPhysicalPropertySoft::max_expansion)
        .def(
            py::pickle(
                [](const RSC_MaterialPhysicalPropertySoft &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_MaterialPhysicalPropertySoft>(buffer);
                }
            )
        );

    py::class_<RSC_MaterialPhysicalPropertyRigid, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertyRigid"
    )
        .def(py::init<>())
        .def_readwrite("density", &RSC_MaterialPhysicalPropertyRigid::density)
        .def(
            py::pickle(
                [](const RSC_MaterialPhysicalPropertyRigid &self) {
                    return rspySerializeRSC(self);
                },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_MaterialPhysicalPropertyRigid>(buffer);
                }
            )
        );

    py::class_<RSC_MaterialOpticalPropertyBase, RSC_MaterialUnaryProperty>(
        m, "RSC_MaterialOpticalPropertyBase"
    );

    py::class_<RSC_MaterialOpticalPropertySimpleMicrofacet, RSC_MaterialOpticalPropertyBase>(
        m, "RSC_MaterialOpticalPropertySimpleMicrofacet"
    )
        .def(py::init<>())
        .def_readwrite(
            "albedo_texture_reference",
            &RSC_MaterialOpticalPropertySimpleMicrofacet::albedo_texture_reference
        )
        .def_readwrite(
            "roughness_metalness_texture_reference",
            &RSC_MaterialOpticalPropertySimpleMicrofacet::roughness_metalness_texture_reference
        )
        .def_readwrite(
            "is_transparent", &RSC_MaterialOpticalPropertySimpleMicrofacet::is_transparent
        )
        .def_readwrite("ior", &RSC_MaterialOpticalPropertySimpleMicrofacet::ior)
        .def_readwrite(
            "emission_texture_reference",
            &RSC_MaterialOpticalPropertySimpleMicrofacet::emission_texture_reference
        )
        .def_readwrite(
            "emission_intensity", &RSC_MaterialOpticalPropertySimpleMicrofacet::emission_intensity
        )
        .def(
            py::pickle(
                [](const RSC_MaterialOpticalPropertySimpleMicrofacet &self) {
                    return rspySerializeRSC(self);
                },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_MaterialOpticalPropertySimpleMicrofacet>(buffer);
                }
            )
        );

    py::class_<RSC_MaterialPairPropertyBase, RSC_EssenceBase>(m, "RSC_MaterialPairPropertyBase")
        .def_readwrite(
            "first_material_reference", &RSC_MaterialPairPropertyBase::first_material_reference
        )
        .def_readwrite(
            "second_material_reference", &RSC_MaterialPairPropertyBase::second_material_reference
        );

    py::class_<RSC_MaterialPairPropertyCoulombFriction, RSC_MaterialPairPropertyBase>(
        m, "RSC_MaterialPairPropertyCoulombFriction"
    )
        .def(py::init<>())
        .def_readwrite("u_static", &RSC_MaterialPairPropertyCoulombFriction::u_static)
        .def_readwrite("u_dynamic", &RSC_MaterialPairPropertyCoulombFriction::u_dynamic)
        .def(
            py::pickle(
                [](const RSC_MaterialPairPropertyCoulombFriction &self) {
                    return rspySerializeRSC(self);
                },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_MaterialPairPropertyCoulombFriction>(buffer);
                }
            )
        );

    py::class_<RSC_MaterialPairPropertyRestitution, RSC_MaterialPairPropertyBase>(
        m, "RSC_MaterialPairPropertyRestitution"
    )
        .def(py::init<>())
        .def_readwrite("restitution", &RSC_MaterialPairPropertyRestitution::restitution)
        .def(
            py::pickle(
                [](const RSC_MaterialPairPropertyRestitution &self) {
                    return rspySerializeRSC(self);
                },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_MaterialPairPropertyRestitution>(buffer);
                }
            )
        );

    /***************************************************************
     * Entities
     ***************************************************************/

    // --- Cosmic background ---
    py::class_<RSC_CosmicBackgroundBase, RSC_EntityBase>(m, "RSC_CosmicBackgroundBase");

    py::class_<RSC_CosmicBackgroundUniform, RSC_CosmicBackgroundBase>(
        m, "RSC_CosmicBackgroundUniform"
    )
        .def(py::init<>())
        .def_readwrite("texture_reference", &RSC_CosmicBackgroundUniform::texture_reference)
        .def_readwrite("intensity", &RSC_CosmicBackgroundUniform::intensity)
        .def_readwrite("rotation_rad", &RSC_CosmicBackgroundUniform::rotation_rad)
        .def(
            py::pickle(
                [](const RSC_CosmicBackgroundUniform &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_CosmicBackgroundUniform>(buffer);
                }
            )
        );

    py::class_<RSC_CosmicBackgroundGradient, RSC_CosmicBackgroundBase>(
        m, "RSC_CosmicBackgroundGradient"
    )
        .def(py::init<>())
        .def_readwrite("zenith_color", &RSC_CosmicBackgroundGradient::zenith_color)
        .def_readwrite("horizon_color", &RSC_CosmicBackgroundGradient::horizon_color)
        .def_readwrite("sun_direction", &RSC_CosmicBackgroundGradient::sun_direction)
        .def_readwrite("sun_color", &RSC_CosmicBackgroundGradient::sun_color)
        .def_readwrite("sun_angular_radius", &RSC_CosmicBackgroundGradient::sun_angular_radius)
        .def(
            py::pickle(
                [](const RSC_CosmicBackgroundGradient &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_CosmicBackgroundGradient>(buffer);
                }
            )
        );

    py::class_<RSC_CosmicBackgroundHosekWilkie, RSC_CosmicBackgroundBase>(
        m, "RSC_CosmicBackgroundHosekWilkie"
    )
        .def(py::init<>())
        .def_readwrite("sun_direction", &RSC_CosmicBackgroundHosekWilkie::sun_direction)
        .def_readwrite("sun_color", &RSC_CosmicBackgroundHosekWilkie::sun_color)
        .def_readwrite("sun_angular_radius", &RSC_CosmicBackgroundHosekWilkie::sun_angular_radius)
        .def_readwrite("turbidity", &RSC_CosmicBackgroundHosekWilkie::turbidity)
        .def_readwrite("ground_color", &RSC_CosmicBackgroundHosekWilkie::ground_color)
        .def_readwrite("intensity", &RSC_CosmicBackgroundHosekWilkie::intensity)
        .def(
            py::pickle(
                [](const RSC_CosmicBackgroundHosekWilkie &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_CosmicBackgroundHosekWilkie>(buffer);
                }
            )
        );

    // --- Floor ---
    py::class_<RSC_FloorBase, RSC_EntityBase>(m, "RSC_FloorBase")
        .def_readwrite("material_reference", &RSC_FloorBase::material_reference);

    py::class_<RSC_FloorFlat, RSC_FloorBase>(m, "RSC_FloorFlat")
        .def(py::init<>())
        .def_readwrite("height", &RSC_FloorFlat::height)
        .def_readwrite("uv_scale", &RSC_FloorFlat::uv_scale)
        .def(
            py::pickle(
                [](const RSC_FloorFlat &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_FloorFlat>(buffer); }
            )
        );

    py::class_<RSC_FloorArray, RSC_FloorBase>(m, "RSC_FloorArray")
        .def(py::init<>())
        .def_readwrite("x_size", &RSC_FloorArray::x_size)
        .def_readwrite("y_size", &RSC_FloorArray::y_size)
        .def_readwrite("x_values", &RSC_FloorArray::x_values)
        .def_readwrite("y_values", &RSC_FloorArray::y_values)
        .def_readwrite("is_x_repeat", &RSC_FloorArray::is_x_repeat)
        .def_readwrite("is_y_repeat", &RSC_FloorArray::is_y_repeat)
        .def_readwrite("height", &RSC_FloorArray::height)
        .def(
            py::pickle(
                [](const RSC_FloorArray &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_FloorArray>(buffer); }
            )
        );

    // --- Structure (voxel) ---
    py::enum_<RSE_StructureExpansionControlMode>(m, "RSE_StructureExpansionControlMode")
        .value(
            "RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION",
            RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION
        )
        .value(
            "RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE", RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE
        )
        .export_values();

    py::enum_<RSE_StructureVoxelConstraintType>(m, "RSE_StructureVoxelConstraintType")
        .value(
            "RSE_STRUCTURE_VOXEL_BALL_AND_SOCKET_JOINT", RSE_STRUCTURE_VOXEL_BALL_AND_SOCKET_JOINT
        )
        .value("RSE_STRUCTURE_VOXEL_FIXED_JOINT", RSE_STRUCTURE_VOXEL_FIXED_JOINT)
        .value("RSE_STRUCTURE_VOXEL_HINGE_JOINT", RSE_STRUCTURE_VOXEL_HINGE_JOINT)
        .value("RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT", RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT)
        .value("RSE_STRUCTURE_VOXEL_NULL_JOINT", RSE_STRUCTURE_VOXEL_NULL_JOINT)
        .export_values();

    py::enum_<RSE_StructureVoxelHingeControlMode>(m, "RSE_StructureVoxelHingeControlMode")
        .value(
            "RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION",
            RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION
        )
        .value(
            "RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION",
            RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION
        )
        .value("RSE_STRUCTURE_VOXEL_HINGE_CONTROL_TORQUE", RSE_STRUCTURE_VOXEL_HINGE_CONTROL_TORQUE)
        .value(
            "RSE_STRUCTURE_VOXEL_HINGE_CONTROL_VELOCITY", RSE_STRUCTURE_VOXEL_HINGE_CONTROL_VELOCITY
        )
        .export_values();

    py::enum_<RSE_StructureVoxelPrismaticControlMode>(m, "RSE_StructureVoxelPrismaticControlMode")
        .value(
            "RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION",
            RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION
        )
        .value(
            "RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION",
            RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION
        )
        .value(
            "RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_FORCE",
            RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_FORCE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_VELOCITY",
            RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_VELOCITY
        )
        .export_values();

    py::enum_<RSE_StructureVoxelRecordVoxelRange>(m, "RSE_StructureVoxelRecordVoxelRange")
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_NONE",
            RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_NONE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_ALL", RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_ALL
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_RANDOM",
            RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_RANDOM
        )
        .export_values();

    py::enum_<RSE_StructureVoxelRecordLinkRange>(m, "RSE_StructureVoxelRecordLinkRange")
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_NONE", RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_NONE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_ALL", RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_ALL
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_RANDOM",
            RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_RANDOM
        )
        .export_values();

    py::enum_<RSE_StructureVoxelRecordRigidBodyRange>(m, "RSE_StructureVoxelRecordRigidBodyRange")
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_NONE",
            RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_NONE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_ALL",
            RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_ALL
        )
        .export_values();

    py::enum_<RSE_StructureVoxelRecordJointRange>(m, "RSE_StructureVoxelRecordJointRange")
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_NONE",
            RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_NONE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_ALL", RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_ALL
        )
        .export_values();

    py::class_<RSC_StructureVoxelBody, RSC_Base>(m, "RSC_StructureVoxelBody")
        .def(py::init<>())
        .def_readwrite(
            "relative_origin_position", &RSC_StructureVoxelBody::relative_origin_position
        )
        .def_readwrite("relative_orientation", &RSC_StructureVoxelBody::relative_orientation)
        .def_readwrite("x_voxels", &RSC_StructureVoxelBody::x_voxels)
        .def_readwrite("y_voxels", &RSC_StructureVoxelBody::y_voxels)
        .def_readwrite("z_voxels", &RSC_StructureVoxelBody::z_voxels)
        .def_readwrite("material_reference_sid", &RSC_StructureVoxelBody::material_reference_sid)
        .def_readwrite("segment_bid", &RSC_StructureVoxelBody::segment_bid)
        .def_readwrite("expansion_signal_sid", &RSC_StructureVoxelBody::expansion_signal_sid)
        .def_readwrite("expansion_control_mode", &RSC_StructureVoxelBody::expansion_control_mode)
        .def_readwrite("link", &RSC_StructureVoxelBody::link)
        .def_readwrite("voxel_size", &RSC_StructureVoxelBody::voxel_size)
        .def(
            py::pickle(
                [](const RSC_StructureVoxelBody &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_StructureVoxelBody>(buffer); }
            )
        );

    py::bind_vector<std::vector<RSC_StructureVoxelBody>>(m, "VecStructureVoxelBody");

    py::class_<RSC_StructureVoxelConstraint, RSC_Base>(m, "RSC_StructureVoxelConstraint")
        .def(py::init<>())
        .def_readwrite("type", &RSC_StructureVoxelConstraint::type)
        .def_readwrite("is_collision_enabled", &RSC_StructureVoxelConstraint::is_collision_enabled)
        .def_readwrite("a_body_sid", &RSC_StructureVoxelConstraint::a_body_sid)
        .def_readwrite("a_segment_bid", &RSC_StructureVoxelConstraint::a_segment_bid)
        .def_readwrite("b_body_sid", &RSC_StructureVoxelConstraint::b_body_sid)
        .def_readwrite("b_segment_bid", &RSC_StructureVoxelConstraint::b_segment_bid)
        .def_readwrite("relative_anchor", &RSC_StructureVoxelConstraint::relative_anchor)
        .def_readwrite("hinge_control_mode", &RSC_StructureVoxelConstraint::hinge_control_mode)
        .def_readwrite("hinge_relative_axis", &RSC_StructureVoxelConstraint::hinge_relative_axis)
        .def_readwrite("hinge_min", &RSC_StructureVoxelConstraint::hinge_min)
        .def_readwrite("hinge_max", &RSC_StructureVoxelConstraint::hinge_max)
        .def_readwrite(
            "hinge_position_control_kp", &RSC_StructureVoxelConstraint::hinge_position_control_kp
        )
        .def_readwrite(
            "hinge_position_control_kd", &RSC_StructureVoxelConstraint::hinge_position_control_kd
        )
        .def_readwrite("hinge_max_speed", &RSC_StructureVoxelConstraint::hinge_max_speed)
        .def_readwrite("hinge_max_torque", &RSC_StructureVoxelConstraint::hinge_max_torque)
        .def_readwrite("hinge_signal_sid", &RSC_StructureVoxelConstraint::hinge_signal_sid)
        .def_readwrite(
            "prismatic_control_mode", &RSC_StructureVoxelConstraint::prismatic_control_mode
        )
        .def_readwrite(
            "prismatic_relative_axis", &RSC_StructureVoxelConstraint::prismatic_relative_axis
        )
        .def_readwrite("prismatic_min", &RSC_StructureVoxelConstraint::prismatic_min)
        .def_readwrite("prismatic_max", &RSC_StructureVoxelConstraint::prismatic_max)
        .def_readwrite(
            "prismatic_position_control_kp",
            &RSC_StructureVoxelConstraint::prismatic_position_control_kp
        )
        .def_readwrite(
            "prismatic_position_control_kd",
            &RSC_StructureVoxelConstraint::prismatic_position_control_kd
        )
        .def_readwrite("prismatic_max_speed", &RSC_StructureVoxelConstraint::prismatic_max_speed)
        .def_readwrite("prismatic_max_force", &RSC_StructureVoxelConstraint::prismatic_max_force)
        .def_readwrite("prismatic_signal_sid", &RSC_StructureVoxelConstraint::prismatic_signal_sid)
        .def(
            py::pickle(
                [](const RSC_StructureVoxelConstraint &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_StructureVoxelConstraint>(buffer);
                }
            )
        );

    py::bind_vector<std::vector<RSC_StructureVoxelConstraint>>(m, "VecStructureVoxelConstraint");

    py::class_<RSC_StructureVoxelRecord, RSC_RecordBase>(m, "RSC_StructureVoxelRecord")
        .def(py::init<>())
        .def_readwrite("voxel_record_range", &RSC_StructureVoxelRecord::voxel_record_range)
        .def_readwrite("link_record_range", &RSC_StructureVoxelRecord::link_record_range)
        .def_readwrite(
            "rigid_body_record_range", &RSC_StructureVoxelRecord::rigid_body_record_range
        )
        .def_readwrite("joint_record_range", &RSC_StructureVoxelRecord::joint_record_range)
        .def_readwrite("voxel_sample_num", &RSC_StructureVoxelRecord::voxel_sample_num)
        .def_readwrite("link_sample_num", &RSC_StructureVoxelRecord::link_sample_num)
        .def(
            py::pickle(
                [](const RSC_StructureVoxelRecord &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_StructureVoxelRecord>(buffer);
                }
            )
        );

    py::class_<RSC_StructureVoxelSignal, RSC_SignalBase>(m, "RSC_StructureVoxelSignal")
        .def(py::init<>())
        .def_readwrite("expansion_signal_num", &RSC_StructureVoxelSignal::expansion_signal_num)
        .def_readwrite("hinge_signal_num", &RSC_StructureVoxelSignal::hinge_signal_num)
        .def_readwrite("prismatic_signal_num", &RSC_StructureVoxelSignal::prismatic_signal_num)
        .def(
            py::pickle(
                [](const RSC_StructureVoxelSignal &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_StructureVoxelSignal>(buffer);
                }
            )
        );

    py::class_<RSC_StructureBase, RSC_EntityBase>(m, "RSC_StructureBase")
        .def_readwrite("origin_position", &RSC_StructureBase::origin_position)
        .def_readwrite("orientation", &RSC_StructureBase::orientation)
        .def_readwrite("material_references", &RSC_StructureBase::material_references);

    py::class_<RSC_StructureVoxel, RSC_StructureBase>(m, "RSC_StructureVoxel")
        .def(py::init<>())
        .def_readwrite("is_static", &RSC_StructureVoxel::is_fixed)
        .def_readwrite("bodies", &RSC_StructureVoxel::bodies)
        .def_readwrite("constraints", &RSC_StructureVoxel::constraints)
        .def_readwrite("record", &RSC_StructureVoxel::record)
        .def_readwrite("signal", &RSC_StructureVoxel::signal)
        .def(
            py::pickle(
                [](const RSC_StructureVoxel &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_StructureVoxel>(buffer); }
            )
        );

    // --- Camera ---
    py::class_<RSC_CameraPerspectiveRecord, RSC_RecordBase>(m, "RSC_CameraPerspectiveRecord")
        .def(py::init<>())
        .def_readwrite("has_rgb", &RSC_CameraPerspectiveRecord::has_rgb)
        .def_readwrite("has_depth", &RSC_CameraPerspectiveRecord::has_depth)
        .def_readwrite("has_entity_info", &RSC_CameraPerspectiveRecord::has_entity_info)
        .def(
            py::pickle(
                [](const RSC_CameraPerspectiveRecord &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_CameraPerspectiveRecord>(buffer);
                }
            )
        );

    py::class_<RSC_CameraPerspectiveSignal, RSC_SignalBase>(m, "RSC_CameraPerspectiveSignal")
        .def(py::init<>())
        .def_readwrite("position_signal_num", &RSC_CameraPerspectiveSignal::position_signal_num)
        .def_readwrite(
            "orientation_signal_num", &RSC_CameraPerspectiveSignal::orientation_signal_num
        )
        .def_readwrite("optics_signal_num", &RSC_CameraPerspectiveSignal::optics_signal_num)
        .def(
            py::pickle(
                [](const RSC_CameraPerspectiveSignal &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_CameraPerspectiveSignal>(buffer);
                }
            )
        );

    py::class_<RSC_CameraBase, RSC_EntityBase>(m, "RSC_CameraBase");

    py::class_<RSC_CameraPerspectiveBase, RSC_CameraBase>(m, "RSC_CameraPerspectiveBase")
        .def_readwrite("image_width", &RSC_CameraPerspectiveBase::image_width)
        .def_readwrite("image_height", &RSC_CameraPerspectiveBase::image_height)
        .def_readwrite("samples_per_pixel", &RSC_CameraPerspectiveBase::samples_per_pixel)
        .def_readwrite("max_depth", &RSC_CameraPerspectiveBase::max_depth)
        .def_readwrite("field_of_view", &RSC_CameraPerspectiveBase::field_of_view)
        .def_readwrite("defocus_angle", &RSC_CameraPerspectiveBase::defocus_angle)
        .def_readwrite("focus_distance", &RSC_CameraPerspectiveBase::focus_distance)
        .def_readwrite("record", &RSC_CameraPerspectiveBase::record)
        .def_readwrite("signal", &RSC_CameraPerspectiveBase::signal);

    py::class_<RSC_CameraPerspectiveFree, RSC_CameraPerspectiveBase>(m, "RSC_CameraPerspectiveFree")
        .def(py::init<>())
        .def_readwrite("origin_position", &RSC_CameraPerspectiveFree::origin_position)
        .def_readwrite("orientation", &RSC_CameraPerspectiveFree::orientation)
        .def(
            py::pickle(
                [](const RSC_CameraPerspectiveFree &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_CameraPerspectiveFree>(buffer);
                }
            )
        );

    py::class_<RSC_CameraPerspectiveRigidBodyLinked, RSC_CameraPerspectiveBase>(
        m, "RSC_CameraPerspectiveRigidBodyLinked"
    )
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_CameraPerspectiveRigidBodyLinked::structure_name)
        .def_readwrite("body_sid", &RSC_CameraPerspectiveRigidBodyLinked::body_sid)
        .def_readwrite("segment_bid", &RSC_CameraPerspectiveRigidBodyLinked::segment_bid)
        .def_readwrite("relative_anchor", &RSC_CameraPerspectiveRigidBodyLinked::relative_anchor)
        .def_readwrite(
            "relative_orientation", &RSC_CameraPerspectiveRigidBodyLinked::relative_orientation
        )
        .def(
            py::pickle(
                [](const RSC_CameraPerspectiveRigidBodyLinked &self) {
                    return rspySerializeRSC(self);
                },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_CameraPerspectiveRigidBodyLinked>(buffer);
                }
            )
        );

    /***************************************************************
     * Record
     ***************************************************************/
    py::class_<RSC_IntersectRegionCuboid, RSC_RecordBase>(m, "RSC_IntersectRegionCuboid")
        .def(py::init<>())
        .def_readwrite("position", &RSC_IntersectRegionCuboid::position)
        .def_readwrite("orientation", &RSC_IntersectRegionCuboid::orientation)
        .def_readwrite("size", &RSC_IntersectRegionCuboid::size)
        .def(
            py::pickle(
                [](const RSC_IntersectRegionCuboid &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_IntersectRegionCuboid>(buffer);
                }
            )
        );

    py::class_<RSC_IntersectRegionCuboidRigidBodyLinked, RSC_RecordBase>(
        m, "RSC_IntersectRegionCuboidRigidBodyLinked"
    )
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_IntersectRegionCuboidRigidBodyLinked::structure_name)
        .def_readwrite("body_sid", &RSC_IntersectRegionCuboidRigidBodyLinked::body_sid)
        .def_readwrite("segment_bid", &RSC_IntersectRegionCuboidRigidBodyLinked::segment_bid)
        .def_readwrite(
            "relative_anchor", &RSC_IntersectRegionCuboidRigidBodyLinked::relative_anchor
        )
        .def_readwrite(
            "relative_orientation", &RSC_IntersectRegionCuboidRigidBodyLinked::relative_orientation
        )
        .def_readwrite("size", &RSC_IntersectRegionCuboidRigidBodyLinked::size)
        .def(
            py::pickle(
                [](const RSC_IntersectRegionCuboidRigidBodyLinked &self) {
                    return rspySerializeRSC(self);
                },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_IntersectRegionCuboidRigidBodyLinked>(buffer);
                }
            )
        );

    py::class_<RSC_IntersectRegionSphere, RSC_RecordBase>(m, "RSC_IntersectRegionSphere")
        .def(py::init<>())
        .def_readwrite("position", &RSC_IntersectRegionSphere::position)
        .def_readwrite("radius", &RSC_IntersectRegionSphere::radius)
        .def(
            py::pickle(
                [](const RSC_IntersectRegionSphere &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) {
                    return rspyDeserializeRSC<RSC_IntersectRegionSphere>(buffer);
                }
            )
        );

    /***************************************************************
     * Signal
     ***************************************************************/
    py::class_<RSC_Force, RSC_SignalBase>(m, "RSC_Force")
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_Force::structure_name)
        .def_readwrite("voxel_range", &RSC_Force::voxel_range)
        .def_readwrite("body_sid", &RSC_Force::body_sid)
        .def_readwrite("segment_bid", &RSC_Force::segment_bid)
        .def(
            py::pickle(
                [](const RSC_Force &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_Force>(buffer); }
            )
        );

    py::class_<RSC_Torque, RSC_SignalBase>(m, "RSC_Torque")
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_Torque::structure_name)
        .def_readwrite("body_sid", &RSC_Torque::body_sid)
        .def_readwrite("segment_bid", &RSC_Torque::segment_bid)
        .def(
            py::pickle(
                [](const RSC_Torque &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_Torque>(buffer); }
            )
        );

    /***************************************************************
     * Recorder
     ***************************************************************/
    py::class_<RSC_RecorderHDF5, RSC_RecorderBase>(m, "RSC_RecorderHDF5")
        .def(py::init<>())
        .def_readwrite("path", &RSC_RecorderHDF5::path)
        .def(
            py::pickle(
                [](const RSC_RecorderHDF5 &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_RecorderHDF5>(buffer); }
            )
        );

    /***************************************************************
     * Controller
     ***************************************************************/
    py::class_<RSC_ControllerCallback, RSC_ControllerBase>(m, "RSC_ControllerCallback")
        .def(py::init<>())
        .def(
            py::pickle(
                [](const RSC_ControllerCallback &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_ControllerCallback>(buffer); }
            )
        );

    /***************************************************************
     * World (top level)
     ***************************************************************/
    py::bind_vector<std::vector<RSC_ResourceVariant>>(m, "VecResourceVariant");
    py::bind_vector<std::vector<RSC_EssenceVariant>>(m, "VecEssenceVariant");
    py::bind_vector<std::vector<RSC_EntityVariant>>(m, "VecEntityVariant");
    py::bind_vector<std::vector<RSC_RecordVariant>>(m, "VecRecordVariant");
    py::bind_vector<std::vector<RSC_SignalVariant>>(m, "VecSignalVariant");
    py::bind_vector<std::vector<RSC_RecorderVariant>>(m, "VecRecorderVariant");
    py::bind_vector<std::vector<RSC_ControllerVariant>>(m, "VecControllerVariant");

    // Variant aliases exposed as Python types so that users can construct them
    // from any of their alternatives and inspect the held value via `value`.
    py::class_<RSC_ResourceVariant>(m, "RSC_ResourceVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_ResourceVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_ResourceVariant &self) { return rspyVariantGet(self); },
            [](RSC_ResourceVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        )
        .def(
            py::pickle(
                [](const RSC_ResourceVariant &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_ResourceVariant>(buffer); }
            )
        );

    py::class_<RSC_EssenceVariant>(m, "RSC_EssenceVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_EssenceVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_EssenceVariant &self) { return rspyVariantGet(self); },
            [](RSC_EssenceVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        )
        .def(
            py::pickle(
                [](const RSC_EssenceVariant &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_EssenceVariant>(buffer); }
            )
        );

    py::class_<RSC_EntityVariant>(m, "RSC_EntityVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_EntityVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_EntityVariant &self) { return rspyVariantGet(self); },
            [](RSC_EntityVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        )
        .def(
            py::pickle(
                [](const RSC_EntityVariant &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_EntityVariant>(buffer); }
            )
        );

    py::class_<RSC_RecordVariant>(m, "RSC_RecordVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_RecordVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_RecordVariant &self) { return rspyVariantGet(self); },
            [](RSC_RecordVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        )
        .def(
            py::pickle(
                [](const RSC_RecordVariant &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_RecordVariant>(buffer); }
            )
        );

    py::class_<RSC_SignalVariant>(m, "RSC_SignalVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_SignalVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_SignalVariant &self) { return rspyVariantGet(self); },
            [](RSC_SignalVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        )
        .def(
            py::pickle(
                [](const RSC_SignalVariant &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_SignalVariant>(buffer); }
            )
        );

    py::class_<RSC_RecorderVariant>(m, "RSC_RecorderVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_RecorderVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_RecorderVariant &self) { return rspyVariantGet(self); },
            [](RSC_RecorderVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        )
        .def(
            py::pickle(
                [](const RSC_RecorderVariant &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_RecorderVariant>(buffer); }
            )
        );

    py::class_<RSC_ControllerVariant>(m, "RSC_ControllerVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_ControllerVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_ControllerVariant &self) { return rspyVariantGet(self); },
            [](RSC_ControllerVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        )
        .def(
            py::pickle(
                [](const RSC_ControllerVariant &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_ControllerVariant>(buffer); }
            )
        );

    py::class_<RSC_World, RSC_Base>(m, "RSC_World")
        .def(py::init<>())
        .def(
            py::init([](py::iterable rsc_strings, bool throw_error = true) {
                std::vector<std::string> cpp_rsc_strings;
                for (auto item : rsc_strings) {
                    cpp_rsc_strings.push_back(py::cast<std::string>(item));
                }
                return RSC_World(cpp_rsc_strings, throw_error);
            }),
            py::arg("rsc_strings"),
            py::arg("throw_error") = true,
            R"doc(
         Constructs an RSC_World from a Python iterable of XML characterization strings.

         Args:
             rsc_strings: A list or other iterable of XML characterization strings to parse.
             throw_error: If True, throw an exception when validation fails. If False,
                         store validity in is_valid and the reason in invalid_reason.
        )doc"
        )
        .def_readwrite("name", &RSC_World::name)
        .def_readwrite("is_valid", &RSC_World::is_valid)
        .def_readwrite("invalid_reason", &RSC_World::invalid_reason)
        .def_readwrite("simulation", &RSC_World::simulation)
        .def_readwrite("resources", &RSC_World::resources)
        .def_readwrite("essences", &RSC_World::essences)
        .def_readwrite("entities", &RSC_World::entities)
        .def_readwrite("records", &RSC_World::records)
        .def_readwrite("signals", &RSC_World::signals)
        .def_readwrite("recorders", &RSC_World::recorders)
        .def_readwrite("controllers", &RSC_World::controllers)
        .def(
            "open",
            [](RSC_World &self, py::iterable paths, bool throw_error = true) {
                std::vector<std::string> cpp_paths;
                for (auto item : paths) {
                    cpp_paths.push_back(py::cast<std::string>(item));
                }
                self.open(cpp_paths, throw_error);
            },
            py::arg("paths"),
            py::arg("throw_error") = true,
            R"doc(
            Opens and parses XML characterization files.

            Args:
                paths: A list or other iterable of RSC file paths to parse.
                throw_error: If True, throw an exception when validation fails. If False,
                            store validity in is_valid and the reason in invalid_reason.
        )doc"
        )
        .def(
            "validate",
            &RSC_World::validate,
            py::arg("throw_error") = true,
            R"doc(
            Validates the world characterization.

            Args:
                throw_error: If True, throw an exception when validation fails. If False,
                            store validity in is_valid and the reason in invalid_reason.
        )doc"
        )
        .def(
            py::pickle(
                [](const RSC_World &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<RSC_World>(buffer); }
            )
        );
}

#endif  // INIT_RSC_H
