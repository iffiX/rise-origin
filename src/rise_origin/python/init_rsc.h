// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef INIT_RSC_H
#define INIT_RSC_H

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "rise_origin/rise_origin.h"
#include "rise_origin/utils/rs_mp_serialization.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<RSE_StructureExpansionControlMode>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_StructureBodyConfig>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_StructureTendonConfig>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_StructureConstraintConfig>);

PYBIND11_MAKE_OPAQUE(std::vector<RS_MaterialConfig>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_StructureConfig>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_RecordConfig>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_SignalConfig>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_RecorderConfig>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_CameraConfig>);

// Serialization helper functions
template <typename T>
py::bytes serializeConfig(const T &config) {
    RS_MessagePackSerializer s;
    s("config_root", config);
    auto data = s.serialize();
    return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
}

template <typename T>
T deserializeConfig(const py::bytes &pybuf) {
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(pybuf.ptr(), &buffer, &length)) {
        throw std::runtime_error("Failed to extract bytes contents");
    }

    T config;
    RS_MessagePackDeserializer d;
    d.deserialize(reinterpret_cast<const uint8_t *>(buffer), static_cast<size_t>(length));
    d("config_root", config);
    return config;
}

inline void initRSPYRSC(py::module &m) {
    /***************************************************************
     * Floor configs
     ***************************************************************/
    py::enum_<RSE_FloorDataType>(m, "RSE_FloorDataType")
        .value("RSE_FLOOR_DATA_NONE", RSE_FLOOR_DATA_NONE)
        .value("RSE_FLOOR_DATA_FLAT", RSE_FLOOR_DATA_FLAT)
        .value("RSE_FLOOR_DATA_ARRAY", RSE_FLOOR_DATA_ARRAY)
        .export_values();

    py::enum_<RSE_FloorRenderType>(m, "RSE_FloorRenderType")
        .value("RSE_FLOOR_RENDER_OPAQUE", RSE_FLOOR_RENDER_OPAQUE)
        .value("RSE_FLOOR_RENDER_OPAQUE_GRID", RSE_FLOOR_RENDER_OPAQUE_GRID)
        .value("RSE_FLOOR_RENDER_TRANSPARENT", RSE_FLOOR_RENDER_TRANSPARENT)
        .export_values();

    py::class_<RS_ArrayFloorDataConfig>(m, "RS_ArrayFloorDataConfig")
        .def(py::init<>())
        .def_readwrite("x_size", &RS_ArrayFloorDataConfig::x_size)
        .def_readwrite("y_size", &RS_ArrayFloorDataConfig::y_size)
        .def_readwrite("x_values", &RS_ArrayFloorDataConfig::x_values)
        .def_readwrite("y_values", &RS_ArrayFloorDataConfig::y_values)
        .def_readwrite("is_x_repeat", &RS_ArrayFloorDataConfig::is_x_repeat)
        .def_readwrite("is_y_repeat", &RS_ArrayFloorDataConfig::is_y_repeat)
        .def_readwrite("height", &RS_ArrayFloorDataConfig::height)
        .def(
            py::pickle(
                [](const RS_ArrayFloorDataConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_ArrayFloorDataConfig>(buffer); }
            )
        );

    py::class_<RS_OpaqueFloorRenderConfig>(m, "RS_OpaqueFloorRenderConfig")
        .def(py::init<>())
        .def_readwrite("color", &RS_OpaqueFloorRenderConfig::color)
        .def_readwrite("roughness", &RS_OpaqueFloorRenderConfig::roughness)
        .def_readwrite("metalness", &RS_OpaqueFloorRenderConfig::metalness)
        .def(
            py::pickle(
                [](const RS_OpaqueFloorRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_OpaqueFloorRenderConfig>(buffer);
                }
            )
        );

    py::class_<RS_OpaqueGridFloorRenderConfig>(m, "RS_OpaqueGridFloorRenderConfig")
        .def(py::init<>())
        .def_readwrite("x_spacing", &RS_OpaqueGridFloorRenderConfig::x_spacing)
        .def_readwrite("y_spacing", &RS_OpaqueGridFloorRenderConfig::y_spacing)
        .def_readwrite("line_width", &RS_OpaqueGridFloorRenderConfig::line_width)
        .def_readwrite("fill_color", &RS_OpaqueGridFloorRenderConfig::fill_color)
        .def_readwrite("line_color", &RS_OpaqueGridFloorRenderConfig::line_color)
        .def_readwrite("roughness", &RS_OpaqueGridFloorRenderConfig::roughness)
        .def_readwrite("metalness", &RS_OpaqueGridFloorRenderConfig::metalness)
        .def(
            py::pickle(
                [](const RS_OpaqueGridFloorRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_OpaqueGridFloorRenderConfig>(buffer);
                }
            )
        );

    py::class_<RS_TransparentFloorRenderConfig>(m, "RS_TransparentFloorRenderConfig")
        .def(py::init<>())
        .def_readwrite("color", &RS_TransparentFloorRenderConfig::color)
        .def_readwrite("ior", &RS_TransparentFloorRenderConfig::ior)
        .def_readwrite("roughness", &RS_TransparentFloorRenderConfig::roughness)
        .def(
            py::pickle(
                [](const RS_TransparentFloorRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_TransparentFloorRenderConfig>(buffer);
                }
            )
        );

    py::class_<RS_FloorDataConfig>(m, "RS_FloorDataConfig")
        .def(py::init<>())
        .def_readwrite("type", &RS_FloorDataConfig::type)
        .def_property(
            "config",
            // Getter
            [](RS_FloorDataConfig &self) -> py::object {
                if (self.type == RSE_FLOOR_DATA_ARRAY) {
                    // Return reference to the stored RS_ArrayFloorDataConfig
                    auto &cfg = self.config.as<RS_ArrayFloorDataConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                }
                return py::none();
            },
            // Setter
            [](RS_FloorDataConfig &self, py::object arg) {
                // Check the Python runtime type
                if (py::isinstance<RS_ArrayFloorDataConfig>(arg)) {
                    self.type = RSE_FLOOR_DATA_ARRAY;
                    // Copy from Python object into std::any
                    self.config = arg.cast<RS_ArrayFloorDataConfig>();
                } else {
                    throw std::runtime_error(
                        "Unsupported type for config; "
                        "Supported types are: RS_ArrayFloorDataConfig."
                    );
                }
            }
        )
        .def(
            py::pickle(
                [](const RS_FloorDataConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_FloorDataConfig>(buffer); }
            )
        );

    py::class_<RS_FloorRenderConfig>(m, "RS_FloorRenderConfig")
        .def(py::init<>())
        .def_readwrite("type", &RS_FloorRenderConfig::type)
        .def_property(
            "config",
            // Getter
            [](RS_FloorRenderConfig &self) -> py::object {
                if (self.type == RSE_FLOOR_RENDER_OPAQUE) {
                    auto &cfg = self.config.as<RS_OpaqueFloorRenderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                } else if (self.type == RSE_FLOOR_RENDER_OPAQUE_GRID) {
                    auto &cfg = self.config.as<RS_OpaqueGridFloorRenderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                } else if (self.type == RSE_FLOOR_RENDER_TRANSPARENT) {
                    auto &cfg = self.config.as<RS_TransparentFloorRenderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                }
                return py::none();
            },
            // Setter
            [](RS_FloorRenderConfig &self, py::object arg) {
                if (py::isinstance<RS_OpaqueFloorRenderConfig>(arg)) {
                    self.type = RSE_FLOOR_RENDER_OPAQUE;
                    self.config = arg.cast<RS_OpaqueFloorRenderConfig>();
                } else if (py::isinstance<RS_OpaqueGridFloorRenderConfig>(arg)) {
                    self.type = RSE_FLOOR_RENDER_OPAQUE_GRID;
                    self.config = arg.cast<RS_OpaqueGridFloorRenderConfig>();
                } else if (py::isinstance<RS_TransparentFloorRenderConfig>(arg)) {
                    self.type = RSE_FLOOR_RENDER_TRANSPARENT;
                    self.config = arg.cast<RS_TransparentFloorRenderConfig>();
                } else {
                    throw std::runtime_error(
                        "Unsupported type for config; "
                        "Supported types are: RS_OpaqueFloorRenderConfig, "
                        "RS_OpaqueGridFloorRenderConfig, RS_TransparentFloorRenderConfig."
                    );
                }
            }
        )
        .def(
            py::pickle(
                [](const RS_FloorRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_FloorRenderConfig>(buffer); }
            )
        );

    py::class_<RS_FloorConfig>(m, "RS_FloorConfig")
        .def(py::init<>())
        .def_readwrite("data_config", &RS_FloorConfig::data_config)
        .def_readwrite("render_config", &RS_FloorConfig::render_config)
        .def(
            py::pickle(
                [](const RS_FloorConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_FloorConfig>(buffer); }
            )
        );

    /***************************************************************
     * World configs
     ***************************************************************/
    py::enum_<RSE_WorldRenderType>(m, "RSE_WorldRenderType")
        .value("RSE_WORLD_RENDER_CONSTANT_COLOR", RSE_WORLD_RENDER_CONSTANT_COLOR)
        .value("RSE_WORLD_RENDER_GRADIENT", RSE_WORLD_RENDER_GRADIENT)
        .value("RSE_WORLD_RENDER_HOSEK_WILKIE", RSE_WORLD_RENDER_HOSEK_WILKIE)
        .export_values();

    py::class_<RS_ConstantColorWorldRenderConfig>(m, "RS_ConstantColorWorldRenderConfig")
        .def(py::init<>())
        .def_readwrite("color", &RS_ConstantColorWorldRenderConfig::color)
        .def(
            py::pickle(
                [](const RS_ConstantColorWorldRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_ConstantColorWorldRenderConfig>(buffer);
                }
            )
        );

    py::class_<RS_GradientSkyWorldRenderConfig>(m, "RS_GradientSkyWorldRenderConfig")
        .def(py::init<>())
        .def_readwrite("zenith_color", &RS_GradientSkyWorldRenderConfig::zenith_color)
        .def_readwrite("horizon_color", &RS_GradientSkyWorldRenderConfig::horizon_color)
        .def_readwrite("sun_direction", &RS_GradientSkyWorldRenderConfig::sun_direction)
        .def_readwrite("sun_color", &RS_GradientSkyWorldRenderConfig::sun_color)
        .def_readwrite(
            "sun_angular_radius", &RS_GradientSkyWorldRenderConfig::sun_angular_radius
        )
        .def(
            py::pickle(
                [](const RS_GradientSkyWorldRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_GradientSkyWorldRenderConfig>(buffer);
                }
            )
        );

    py::class_<RS_HosekWilkieSkyWorldRenderConfig>(m, "RS_HosekWilkieSkyWorldRenderConfig")
        .def(py::init<>())
        .def_readwrite("sun_direction", &RS_HosekWilkieSkyWorldRenderConfig::sun_direction)
        .def_readwrite("sun_color", &RS_HosekWilkieSkyWorldRenderConfig::sun_color)
        .def_readwrite(
            "sun_angular_radius", &RS_HosekWilkieSkyWorldRenderConfig::sun_angular_radius
        )
        .def_readwrite("turbidity", &RS_HosekWilkieSkyWorldRenderConfig::turbidity)
        .def_readwrite("ground_color", &RS_HosekWilkieSkyWorldRenderConfig::ground_color)
        .def_readwrite("intensity", &RS_HosekWilkieSkyWorldRenderConfig::intensity)
        .def(
            py::pickle(
                [](const RS_HosekWilkieSkyWorldRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_HosekWilkieSkyWorldRenderConfig>(buffer);
                }
            )
        );

    py::class_<RS_WorldRenderConfig>(m, "RS_WorldRenderConfig")
        .def(py::init<>())
        .def_readwrite("type", &RS_WorldRenderConfig::type)
        .def_property(
            "config",
            // Getter
            [](RS_WorldRenderConfig &self) -> py::object {
                if (self.type == RSE_WORLD_RENDER_CONSTANT_COLOR) {
                    auto &cfg = self.config.as<RS_ConstantColorWorldRenderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                } else if (self.type == RSE_WORLD_RENDER_GRADIENT) {
                    auto &cfg = self.config.as<RS_GradientSkyWorldRenderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                } else if (self.type == RSE_WORLD_RENDER_HOSEK_WILKIE) {
                    auto &cfg = self.config.as<RS_HosekWilkieSkyWorldRenderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                }
                return py::none();
            },
            // Setter
            [](RS_WorldRenderConfig &self, py::object arg) {
                if (py::isinstance<RS_ConstantColorWorldRenderConfig>(arg)) {
                    self.type = RSE_WORLD_RENDER_CONSTANT_COLOR;
                    self.config = arg.cast<RS_ConstantColorWorldRenderConfig>();
                } else if (py::isinstance<RS_GradientSkyWorldRenderConfig>(arg)) {
                    self.type = RSE_WORLD_RENDER_GRADIENT;
                    self.config = arg.cast<RS_GradientSkyWorldRenderConfig>();
                } else if (py::isinstance<RS_HosekWilkieSkyWorldRenderConfig>(arg)) {
                    self.type = RSE_WORLD_RENDER_HOSEK_WILKIE;
                    self.config = arg.cast<RS_HosekWilkieSkyWorldRenderConfig>();
                } else {
                    throw std::runtime_error(
                        "Unsupported type for config; "
                        "Supported types are: RS_ConstantColorWorldRenderConfig, "
                        "RS_GradientSkyWorldRenderConfig, RS_HosekWilkieSkyWorldRenderConfig."
                    );
                }
            }
        )
        .def(
            py::pickle(
                [](const RS_WorldRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_WorldRenderConfig>(buffer); }
            )
        );

    py::class_<RS_WorldConfig>(m, "RS_WorldConfig")
        .def(py::init<>())
        .def_readwrite("grav_acc", &RS_WorldConfig::grav_acc)
        .def_readwrite("render_config", &RS_WorldConfig::render_config)
        .def(
            py::pickle(
                [](const RS_WorldConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_WorldConfig>(buffer); }
            )
        );

    /***************************************************************
     * Material configs
     ***************************************************************/
    py::enum_<RSE_MaterialRenderType>(m, "RSE_MaterialRenderType")
        .value("RSE_MATERIAL_RENDER_OPAQUE", RSE_MATERIAL_RENDER_OPAQUE)
        .value("RSE_MATERIAL_RENDER_TRANSPARENT", RSE_MATERIAL_RENDER_TRANSPARENT)
        .value("RSE_MATERIAL_RENDER_EMISSIVE", RSE_MATERIAL_RENDER_EMISSIVE)
        .export_values();

    py::class_<RS_OpaqueMaterialRenderConfig>(m, "RS_OpaqueMaterialRenderConfig")
        .def(py::init<>())
        .def_readwrite("color", &RS_OpaqueMaterialRenderConfig::color)
        .def_readwrite("roughness", &RS_OpaqueMaterialRenderConfig::roughness)
        .def_readwrite("metalness", &RS_OpaqueMaterialRenderConfig::metalness)
        .def(
            py::pickle(
                [](const RS_OpaqueMaterialRenderConfig &self) {
                    return serializeConfig(self);
                },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_OpaqueMaterialRenderConfig>(buffer);
                }
            )
        );

    py::class_<RS_TransparentMaterialRenderConfig>(m, "RS_TransparentMaterialRenderConfig")
        .def(py::init<>())
        .def_readwrite("color", &RS_TransparentMaterialRenderConfig::color)
        .def_readwrite("ior", &RS_TransparentMaterialRenderConfig::ior)
        .def_readwrite("roughness", &RS_TransparentMaterialRenderConfig::roughness)
        .def(
            py::pickle(
                [](const RS_TransparentMaterialRenderConfig &self) {
                    return serializeConfig(self);
                },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_TransparentMaterialRenderConfig>(buffer);
                }
            )
        );

    py::class_<RS_EmissiveMaterialRenderConfig>(m, "RS_EmissiveMaterialRenderConfig")
        .def(py::init<>())
        .def_readwrite("color", &RS_EmissiveMaterialRenderConfig::color)
        .def_readwrite("intensity", &RS_EmissiveMaterialRenderConfig::intensity)
        .def(
            py::pickle(
                [](const RS_EmissiveMaterialRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_EmissiveMaterialRenderConfig>(buffer);
                }
            )
        );

    py::class_<RS_MaterialRenderConfig>(m, "RS_MaterialRenderConfig")
        .def(py::init<>())
        .def_readwrite("type", &RS_MaterialRenderConfig::type)
        .def_property(
            "config",
            // Getter
            [](RS_MaterialRenderConfig &self) -> py::object {
                if (self.type == RSE_MATERIAL_RENDER_OPAQUE) {
                    auto &cfg = self.config.as<RS_OpaqueMaterialRenderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                } else if (self.type == RSE_MATERIAL_RENDER_TRANSPARENT) {
                    auto &cfg = self.config.as<RS_TransparentMaterialRenderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                } else if (self.type == RSE_MATERIAL_RENDER_EMISSIVE) {
                    auto &cfg = self.config.as<RS_EmissiveMaterialRenderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                }
                return py::none();
            },
            // Setter
            [](RS_MaterialRenderConfig &self, py::object arg) {
                if (py::isinstance<RS_OpaqueMaterialRenderConfig>(arg)) {
                    self.type = RSE_MATERIAL_RENDER_OPAQUE;
                    self.config = arg.cast<RS_OpaqueMaterialRenderConfig>();
                } else if (py::isinstance<RS_TransparentMaterialRenderConfig>(arg)) {
                    self.type = RSE_MATERIAL_RENDER_TRANSPARENT;
                    self.config = arg.cast<RS_TransparentMaterialRenderConfig>();
                } else if (py::isinstance<RS_EmissiveMaterialRenderConfig>(arg)) {
                    self.type = RSE_MATERIAL_RENDER_EMISSIVE;
                    self.config = arg.cast<RS_EmissiveMaterialRenderConfig>();
                } else {
                    throw std::runtime_error(
                        "Unsupported type for config; "
                        "Supported types are: RS_OpaqueMaterialRenderConfig, "
                        "RS_TransparentMaterialRenderConfig, RS_EmissiveMaterialRenderConfig."
                    );
                }
            }
        )
        .def(
            py::pickle(
                [](const RS_MaterialRenderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_MaterialRenderConfig>(buffer); }
            )
        );

    py::class_<RS_MechanicalMaterialConfig>(m, "RS_MechanicalMaterialConfig")
        .def(py::init<>())
        .def_readwrite("is_fixed", &RS_MechanicalMaterialConfig::is_fixed)
        .def_readwrite("is_rigid", &RS_MechanicalMaterialConfig::is_rigid)
        .def_readwrite("min_expansion", &RS_MechanicalMaterialConfig::min_expansion)
        .def_readwrite("max_expansion", &RS_MechanicalMaterialConfig::max_expansion)
        .def_readwrite("elastic_modulus", &RS_MechanicalMaterialConfig::elastic_modulus)
        .def_readwrite("fail_stress", &RS_MechanicalMaterialConfig::fail_stress)
        .def_readwrite("density", &RS_MechanicalMaterialConfig::density)
        .def_readwrite("poissons_ratio", &RS_MechanicalMaterialConfig::poissons_ratio)
        .def_readwrite("u_static", &RS_MechanicalMaterialConfig::u_static)
        .def_readwrite("u_dynamic", &RS_MechanicalMaterialConfig::u_dynamic)
        .def(
            py::pickle(
                [](const RS_MechanicalMaterialConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_MechanicalMaterialConfig>(buffer);
                }
            )
        );

    py::class_<RS_MaterialConfig>(m, "RS_MaterialConfig")
        .def(py::init<>())
        .def_readwrite("name", &RS_MaterialConfig::name)
        .def_readwrite("render_config", &RS_MaterialConfig::render_config)
        .def_readwrite("mechanical_config", &RS_MaterialConfig::mechanical_config)
        .def(
            py::pickle(
                [](const RS_MaterialConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_MaterialConfig>(buffer); }
            )
        );

    /***************************************************************
     * Simulation configs
     ***************************************************************/
    py::enum_<RSE_VisionDenoiseType>(m, "RSE_VisionDenoiseType")
        .value("RSE_VISION_DENOISE_NONE", RSE_VISION_DENOISE_NONE)
        .value("RSE_VISION_DENOISE_ATROUS", RSE_VISION_DENOISE_ATROUS)
        .value("RSE_VISION_DENOISE_SVGF", RSE_VISION_DENOISE_SVGF)
        .export_values();

    py::class_<RS_SimulationConfig>(m, "RS_SimulationConfig")
        .def(py::init<>())
        .def_readwrite("seed", &RS_SimulationConfig::seed)
        .def_readwrite("integration_time_step", &RS_SimulationConfig::integration_time_step)
        .def_readwrite("integration_dt_frac", &RS_SimulationConfig::integration_dt_frac)
        .def_readwrite(
            "softbody_internal_damping_z", &RS_SimulationConfig::softbody_internal_damping_z
        )
        .def_readwrite(
            "softbody_contact_damping_z", &RS_SimulationConfig::softbody_contact_damping_z
        )
        .def_readwrite("softbody_global_damping_z", &RS_SimulationConfig::softbody_global_damping_z)
        .def_readwrite(
            "rigidbody_constraint_iterations", &RS_SimulationConfig::rigidbody_constraint_iterations
        )
        .def_readwrite(
            "rigidbody_constraint_position_correction_ratio",
            &RS_SimulationConfig::rigidbody_constraint_position_correction_ratio
        )
        .def_readwrite(
            "rigidbody_constraint_warm_start_damping_z",
            &RS_SimulationConfig::rigidbody_constraint_warm_start_damping_z
        )
        .def_readwrite(
            "rigidbody_constraint_compliance_initial",
            &RS_SimulationConfig::rigidbody_constraint_compliance_initial
        )
        .def_readwrite(
            "rigidbody_constraint_compliance_min_threshold",
            &RS_SimulationConfig::rigidbody_constraint_compliance_min_threshold
        )
        .def_readwrite(
            "rigidbody_constraint_compliance_max_threshold",
            &RS_SimulationConfig::rigidbody_constraint_compliance_max_threshold
        )
        .def_readwrite(
            "rigidbody_constraint_compliance_auto_adjust_watt_threshold",
            &RS_SimulationConfig::rigidbody_constraint_compliance_auto_adjust_watt_threshold
        )
        .def_readwrite(
            "rigidbody_constraint_compliance_auto_adjust_step",
            &RS_SimulationConfig::rigidbody_constraint_compliance_auto_adjust_step
        )
        .def_readwrite(
            "rigidbody_contact_iterations", &RS_SimulationConfig::rigidbody_contact_iterations
        )
        .def_readwrite(
            "rigidbody_contact_position_correction_iterations",
            &RS_SimulationConfig::rigidbody_contact_position_correction_iterations
        )
        .def_readwrite(
            "rigidbody_contact_position_correction_ratio",
            &RS_SimulationConfig::rigidbody_contact_position_correction_ratio
        )
        .def_readwrite(
            "rigidbody_contact_warm_start_damping_z",
            &RS_SimulationConfig::rigidbody_contact_warm_start_damping_z
        )
        .def_readwrite(
            "rigidbody_contact_restitution", &RS_SimulationConfig::rigidbody_contact_restitution
        )
        .def_readwrite("rigidbody_contact_slop", &RS_SimulationConfig::rigidbody_contact_slop)
        .def_readwrite("rigidbody_linear_damping_z", &RS_SimulationConfig::rigidbody_linear_damping_z)
        .def_readwrite(
            "rigidbody_angular_damping_z", &RS_SimulationConfig::rigidbody_angular_damping_z
        )
        .def_readwrite(
            "vision_trace_firefly_cap", &RS_SimulationConfig::vision_trace_firefly_cap
        )
        .def_readwrite(
            "vision_trace_russian_roulette_start_depth",
            &RS_SimulationConfig::vision_trace_russian_roulette_start_depth
        )
        .def_readwrite(
            "vision_is_restir_enabled",
            &RS_SimulationConfig::vision_is_restir_enabled
        )
        .def_readwrite(
            "vision_restir_disocclusion_position_threshold",
            &RS_SimulationConfig::vision_restir_disocclusion_position_threshold
        )
        .def_readwrite(
            "vision_restir_disocclusion_normal_threshold",
            &RS_SimulationConfig::vision_restir_disocclusion_normal_threshold
        )
        .def_readwrite(
            "vision_restir_max_specular_chain",
            &RS_SimulationConfig::vision_restir_max_specular_chain
        )
        .def_readwrite(
            "vision_restir_di_m_cap", &RS_SimulationConfig::vision_restir_di_m_cap
        )
        .def_readwrite(
            "vision_restir_di_spatial_neighbors",
            &RS_SimulationConfig::vision_restir_di_spatial_neighbors
        )
        .def_readwrite(
            "vision_restir_di_spatial_radius",
            &RS_SimulationConfig::vision_restir_di_spatial_radius
        )
        .def_readwrite(
            "vision_restir_gi_is_bounce_nee_enabled",
            &RS_SimulationConfig::vision_restir_gi_is_bounce_nee_enabled
        )
        .def_readwrite(
            "vision_restir_gi_m_cap", &RS_SimulationConfig::vision_restir_gi_m_cap
        )
        .def_readwrite(
            "vision_restir_gi_spatial_neighbors",
            &RS_SimulationConfig::vision_restir_gi_spatial_neighbors
        )
        .def_readwrite(
            "vision_restir_gi_spatial_radius",
            &RS_SimulationConfig::vision_restir_gi_spatial_radius
        )
        .def_readwrite(
            "vision_restir_gi_validation_period",
            &RS_SimulationConfig::vision_restir_gi_validation_period
        )
        .def_readwrite("vision_denoise_type", &RS_SimulationConfig::vision_denoise_type)
        .def_readwrite(
            "vision_denoise_atrous_iterations",
            &RS_SimulationConfig::vision_denoise_atrous_iterations
        )
        .def_readwrite(
            "vision_denoise_atrous_min_passes",
            &RS_SimulationConfig::vision_denoise_atrous_min_passes
        )
        .def_readwrite(
            "vision_denoise_sigma_color", &RS_SimulationConfig::vision_denoise_sigma_color
        )
        .def_readwrite(
            "vision_denoise_sigma_normal", &RS_SimulationConfig::vision_denoise_sigma_normal
        )
        .def_readwrite(
            "vision_denoise_sigma_depth", &RS_SimulationConfig::vision_denoise_sigma_depth
        )
        .def_readwrite(
            "vision_denoise_temporal_alpha",
            &RS_SimulationConfig::vision_denoise_temporal_alpha
        )
        .def_readwrite(
            "vision_denoise_temporal_alpha_moments",
            &RS_SimulationConfig::vision_denoise_temporal_alpha_moments
        )
        .def_readwrite(
            "vision_denoise_temporal_min_history",
            &RS_SimulationConfig::vision_denoise_temporal_min_history
        )
        .def_readwrite("control_frequency", &RS_SimulationConfig::control_frequency)
        .def_readwrite("record_frequency", &RS_SimulationConfig::record_frequency)
        .def_readwrite("stop_condition", &RS_SimulationConfig::stop_condition)
        .def_readwrite(
            "runtime_dt_update_interval", &RS_SimulationConfig::runtime_dt_update_interval
        )
        .def_readwrite(
            "runtime_rigidbody_update_interval",
            &RS_SimulationConfig::runtime_rigidbody_update_interval
        )
        .def_readwrite(
            "runtime_contact_update_interval",
            &RS_SimulationConfig::runtime_contact_update_interval
        )
        .def_readwrite(
            "runtime_contact_bvh_rebuild_sub_interval",
            &RS_SimulationConfig::runtime_contact_bvh_rebuild_sub_interval
        )
        .def_readwrite(
            "runtime_contact_bvh_refit_sub_interval",
            &RS_SimulationConfig::runtime_contact_bvh_refit_sub_interval
        )
        .def_readwrite(
            "runtime_mechanical_signal_update_interval",
            &RS_SimulationConfig::runtime_mechanical_signal_update_interval
        )
        .def_readwrite(
            "runtime_divergence_check_interval",
            &RS_SimulationConfig::runtime_divergence_check_interval
        )
        .def_readwrite(
            "runtime_record_buffer_size", &RS_SimulationConfig::runtime_record_buffer_size
        )
        .def_readwrite("runtime_max_time", &RS_SimulationConfig::runtime_max_time)
        .def_readwrite("runtime_save_record", &RS_SimulationConfig::runtime_save_record)
        .def_readwrite(
            "runtime_simulation_compute_accuracy",
            &RS_SimulationConfig::runtime_simulation_compute_accuracy
        )
        .def_readwrite(
            "runtime_simulation_memory_accuracy",
            &RS_SimulationConfig::runtime_simulation_memory_accuracy
        )
        .def_readwrite("runtime_log_level", &RS_SimulationConfig::runtime_log_level)
        .def(
            py::pickle(
                [](const RS_SimulationConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_SimulationConfig>(buffer); }
            )
        );

    /***************************************************************
     * Camera configs
     ***************************************************************/
    py::enum_<RSE_CameraType>(m, "RSE_CameraType")
        .value("RSE_CAMERA_NONE", RSE_CAMERA_NONE)
        .value("RSE_CAMERA_FREE_PERSPECTIVE", RSE_CAMERA_FREE_PERSPECTIVE)
        .value("RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE", RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE)
        .export_values();

    py::class_<perspective_camera_config::RS_PerspectiveCameraConfig>(
        m, "RS_PerspectiveCameraConfig"
    )
        .def(py::init<>())
        .def_readwrite(
            "image_width", &perspective_camera_config::RS_PerspectiveCameraConfig::image_width
        )
        .def_readwrite(
            "image_height", &perspective_camera_config::RS_PerspectiveCameraConfig::image_height
        )
        .def_readwrite(
            "samples_per_pixel",
            &perspective_camera_config::RS_PerspectiveCameraConfig::samples_per_pixel
        )
        .def_readwrite(
            "max_depth", &perspective_camera_config::RS_PerspectiveCameraConfig::max_depth
        )
        .def_readwrite(
            "field_of_view", &perspective_camera_config::RS_PerspectiveCameraConfig::field_of_view
        )
        .def_readwrite(
            "defocus_angle", &perspective_camera_config::RS_PerspectiveCameraConfig::defocus_angle
        )
        .def_readwrite(
            "focus_distance", &perspective_camera_config::RS_PerspectiveCameraConfig::focus_distance
        )
        .def(
            py::pickle(
                [](const perspective_camera_config::RS_PerspectiveCameraConfig &self) {
                    return serializeConfig(self);
                },
                [](py::bytes buffer) {
                    return deserializeConfig<perspective_camera_config::RS_PerspectiveCameraConfig>(
                        buffer
                    );
                }
            )
        );

    py::class_<
        RS_FreePerspectiveCameraConfig,
        perspective_camera_config::RS_PerspectiveCameraConfig>(m, "RS_FreePerspectiveCameraConfig")
        .def(py::init<>())
        .def_readwrite("origin_position", &RS_FreePerspectiveCameraConfig::origin_position)
        .def_readwrite("orientation", &RS_FreePerspectiveCameraConfig::orientation)
        .def(
            py::pickle(
                [](const RS_FreePerspectiveCameraConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_FreePerspectiveCameraConfig>(buffer);
                }
            )
        );

    py::class_<
        RS_RigidBodyLinkedPerspectiveCameraConfig,
        perspective_camera_config::RS_PerspectiveCameraConfig>(
        m, "RS_RigidBodyLinkedPerspectiveCameraConfig"
    )
        .def(py::init<>())
        .def_readwrite("structure_name", &RS_RigidBodyLinkedPerspectiveCameraConfig::structure_name)
        .def_readwrite("body_sid", &RS_RigidBodyLinkedPerspectiveCameraConfig::body_sid)
        .def_readwrite("segment_bid", &RS_RigidBodyLinkedPerspectiveCameraConfig::segment_bid)
        .def_readwrite(
            "relative_anchor", &RS_RigidBodyLinkedPerspectiveCameraConfig::relative_anchor
        )
        .def_readwrite(
            "relative_orientation", &RS_RigidBodyLinkedPerspectiveCameraConfig::relative_orientation
        )
        .def(
            py::pickle(
                [](const RS_RigidBodyLinkedPerspectiveCameraConfig &self) {
                    return serializeConfig(self);
                },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_RigidBodyLinkedPerspectiveCameraConfig>(buffer);
                }
            )
        );

    py::class_<RS_CameraConfig>(m, "RS_CameraConfig")
        .def(py::init<>())
        .def_readwrite("name", &RS_CameraConfig::name)
        .def_readwrite("type", &RS_CameraConfig::type)
        .def_property(
            "config",
            // Getter
            [](RS_CameraConfig &self) -> py::object {
                if (self.type == RSE_CAMERA_FREE_PERSPECTIVE) {
                    auto &cfg = self.config.as<RS_FreePerspectiveCameraConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                } else if (self.type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE) {
                    auto &cfg = self.config.as<RS_RigidBodyLinkedPerspectiveCameraConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                }
                return py::none();
            },
            // Setter
            [](RS_CameraConfig &self, py::object arg) {
                if (py::isinstance<RS_FreePerspectiveCameraConfig>(arg)) {
                    self.type = RSE_CAMERA_FREE_PERSPECTIVE;
                    self.config = arg.cast<RS_FreePerspectiveCameraConfig>();
                } else if (py::isinstance<RS_RigidBodyLinkedPerspectiveCameraConfig>(arg)) {
                    self.type = RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE;
                    self.config = arg.cast<RS_RigidBodyLinkedPerspectiveCameraConfig>();
                } else {
                    throw std::runtime_error(
                        "Unsupported type for config; "
                        "Supported types are: RS_FreePerspectiveCameraConfig, "
                        "RS_RigidBodyLinkedPerspectiveCameraConfig."
                    );
                }
            }
        )
        .def(
            py::pickle(
                [](const RS_CameraConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_CameraConfig>(buffer); }
            )
        );

    /***************************************************************
     * Structure configs
     ***************************************************************/
    py::enum_<RSE_StructureExpansionControlMode>(m, "RSE_StructureExpansionControlMode")
        .value(
            "RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION",
            RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION
        )
        .value(
            "RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE",
            RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE
        )
        .export_values();

    py::bind_vector<std::vector<RSE_StructureExpansionControlMode>>(
        m, "VecStructureExpansionControlMode"
    );

    py::class_<RS_StructureBodyConfig>(m, "RS_StructureBodyConfig")
        .def(py::init<>())
        .def_readwrite("body_sid", &RS_StructureBodyConfig::body_sid)
        .def_readwrite(
            "relative_origin_position", &RS_StructureBodyConfig::relative_origin_position
        )
        .def_readwrite("relative_orientation", &RS_StructureBodyConfig::relative_orientation)
        .def_readwrite("x_voxels", &RS_StructureBodyConfig::x_voxels)
        .def_readwrite("y_voxels", &RS_StructureBodyConfig::y_voxels)
        .def_readwrite("z_voxels", &RS_StructureBodyConfig::z_voxels)
        .def_readwrite("material_reference_sid", &RS_StructureBodyConfig::material_reference_sid)
        .def_readwrite("segment_bid", &RS_StructureBodyConfig::segment_bid)
        .def_readwrite(
            "expansion_signal_sid", &RS_StructureBodyConfig::expansion_signal_sid
        )
        .def_readwrite(
            "expansion_control_mode", &RS_StructureBodyConfig::expansion_control_mode
        )
        .def_readwrite("link", &RS_StructureBodyConfig::link)
        .def(
            py::pickle(
                [](const RS_StructureBodyConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_StructureBodyConfig>(buffer); }
            )
        );

    py::class_<RS_StructureTendonConfig>(m, "RS_StructureTendonConfig")
        .def(py::init<>())
        .def_readwrite("material_reference_sid", &RS_StructureTendonConfig::material_reference_sid)
        .def_readwrite("a_body_sid", &RS_StructureTendonConfig::a_body_sid)
        .def_readwrite("a_voxel_x", &RS_StructureTendonConfig::a_voxel_x)
        .def_readwrite("a_voxel_y", &RS_StructureTendonConfig::a_voxel_y)
        .def_readwrite("a_voxel_z", &RS_StructureTendonConfig::a_voxel_z)
        .def_readwrite("a_voxel_face", &RS_StructureTendonConfig::a_voxel_face)
        .def_readwrite("b_body_sid", &RS_StructureTendonConfig::b_body_sid)
        .def_readwrite("b_voxel_x", &RS_StructureTendonConfig::b_voxel_x)
        .def_readwrite("b_voxel_y", &RS_StructureTendonConfig::b_voxel_y)
        .def_readwrite("b_voxel_z", &RS_StructureTendonConfig::b_voxel_z)
        .def_readwrite("b_voxel_face", &RS_StructureTendonConfig::b_voxel_face)
        .def(
            py::pickle(
                [](const RS_StructureTendonConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_StructureTendonConfig>(buffer); }
            )
        );

    py::enum_<RSE_StructureConstraintType>(m, "RSE_StructureConstraintType")
        .value("RSE_BALL_AND_SOCKET_JOINT", RSE_BALL_AND_SOCKET_JOINT)
        .value("RSE_FIXED_JOINT", RSE_FIXED_JOINT)
        .value("RSE_HINGE_JOINT", RSE_HINGE_JOINT)
        .value("RSE_PRISMATIC_JOINT", RSE_PRISMATIC_JOINT)
        .value("RSE_NULL_JOINT", RSE_NULL_JOINT)
        .export_values();

    py::enum_<RSE_StructureHingeJointControlMode>(m, "RSE_StructureHingeJointControlMode")
        .value(
            "RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_BASIC_POSITION",
            RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_BASIC_POSITION
        )
        .value(
            "RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_PD_POSITION",
            RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_PD_POSITION
        )
        .value(
            "RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_TORQUE",
            RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_TORQUE
        )
        .value(
            "RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_VELOCITY",
            RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_VELOCITY
        )
        .export_values();

    py::enum_<RSE_StructurePrismaticJointControlMode>(m, "RSE_StructurePrismaticJointControlMode")
        .value(
            "RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_BASIC_POSITION",
            RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_BASIC_POSITION
        )
        .value(
            "RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_PD_POSITION",
            RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_PD_POSITION
        )
        .value(
            "RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_FORCE",
            RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_FORCE
        )
        .value(
            "RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_VELOCITY",
            RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_VELOCITY
        )
        .export_values();

    py::class_<RS_StructureConstraintConfig>(m, "RS_StructureConstraintConfig")
        .def(py::init<>())
        .def_readwrite("type", &RS_StructureConstraintConfig::type)
        .def_readwrite("a_body_sid", &RS_StructureConstraintConfig::a_body_sid)
        .def_readwrite("a_segment_bid", &RS_StructureConstraintConfig::a_segment_bid)
        .def_readwrite("b_body_sid", &RS_StructureConstraintConfig::b_body_sid)
        .def_readwrite("b_segment_bid", &RS_StructureConstraintConfig::b_segment_bid)
        .def_readwrite("relative_anchor", &RS_StructureConstraintConfig::relative_anchor)
        .def_readwrite("hinge_control_mode", &RS_StructureConstraintConfig::hinge_control_mode)
        .def_readwrite("hinge_relative_axis", &RS_StructureConstraintConfig::hinge_relative_axis)
        .def_readwrite("hinge_min", &RS_StructureConstraintConfig::hinge_min)
        .def_readwrite("hinge_max", &RS_StructureConstraintConfig::hinge_max)
        .def_readwrite(
            "hinge_position_control_kp", &RS_StructureConstraintConfig::hinge_position_control_kp
        )
        .def_readwrite(
            "hinge_position_control_kd", &RS_StructureConstraintConfig::hinge_position_control_kd
        )
        .def_readwrite("hinge_max_speed", &RS_StructureConstraintConfig::hinge_max_speed)
        .def_readwrite("hinge_max_torque", &RS_StructureConstraintConfig::hinge_max_torque)
        .def_readwrite(
            "hinge_signal_sid",
            &RS_StructureConstraintConfig::hinge_signal_sid
        )
        .def_readwrite(
            "prismatic_control_mode", &RS_StructureConstraintConfig::prismatic_control_mode
        )
        .def_readwrite(
            "prismatic_relative_axis", &RS_StructureConstraintConfig::prismatic_relative_axis
        )
        .def_readwrite("prismatic_min", &RS_StructureConstraintConfig::prismatic_min)
        .def_readwrite("prismatic_max", &RS_StructureConstraintConfig::prismatic_max)
        .def_readwrite(
            "prismatic_position_control_kp",
            &RS_StructureConstraintConfig::prismatic_position_control_kp
        )
        .def_readwrite(
            "prismatic_position_control_kd",
            &RS_StructureConstraintConfig::prismatic_position_control_kd
        )
        .def_readwrite("prismatic_max_speed", &RS_StructureConstraintConfig::prismatic_max_speed)
        .def_readwrite("prismatic_max_force", &RS_StructureConstraintConfig::prismatic_max_force)
        .def_readwrite(
            "prismatic_signal_sid",
            &RS_StructureConstraintConfig::prismatic_signal_sid
        )
        .def(
            py::pickle(
                [](const RS_StructureConstraintConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) {
                    return deserializeConfig<RS_StructureConstraintConfig>(buffer);
                }
            )
        );

    py::bind_vector<std::vector<RS_StructureBodyConfig>>(m, "VecStructureBodyConfig");
    py::bind_vector<std::vector<RS_StructureTendonConfig>>(m, "VecStructureTendonConfig");
    py::bind_vector<std::vector<RS_StructureConstraintConfig>>(m, "VecStructureConstraintConfig");
    py::class_<RS_StructureConfig>(m, "RS_StructureConfig")
        .def(py::init<>())
        .def_readwrite("name", &RS_StructureConfig::name)
        .def_readwrite("is_fixed", &RS_StructureConfig::is_fixed)
        .def_readwrite("voxel_size", &RS_StructureConfig::voxel_size)
        .def_readwrite("expansion_signal_num", &RS_StructureConfig::expansion_signal_num)
        .def_readwrite("hinge_signal_num", &RS_StructureConfig::hinge_signal_num)
        .def_readwrite(
            "prismatic_signal_num", &RS_StructureConfig::prismatic_signal_num
        )
        .def_readwrite("origin_position", &RS_StructureConfig::origin_position)
        .def_readwrite("orientation", &RS_StructureConfig::orientation)
        .def_readwrite("material_references", &RS_StructureConfig::material_references)
        .def_readwrite("bodies", &RS_StructureConfig::bodies)
        .def_readwrite("tendons", &RS_StructureConfig::tendons)
        .def_readwrite("constraints", &RS_StructureConfig::constraints)
        .def(
            py::pickle(
                [](const RS_StructureConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_StructureConfig>(buffer); }
            )
        );

    /***************************************************************
     * Record configs
     ***************************************************************/
    py::enum_<RSE_StructureVoxelRecordRange>(m, "RSE_StructureVoxelRecordRange")
        .value("RSE_VOXEL_RECORD_RANGE_NONE", RSE_VOXEL_RECORD_RANGE_NONE)
        .value("RSE_VOXEL_RECORD_RANGE_ALL", RSE_VOXEL_RECORD_RANGE_ALL)
        .value("RSE_VOXEL_RECORD_RANGE_RANDOM", RSE_VOXEL_RECORD_RANGE_RANDOM)
        .export_values();

    py::enum_<RSE_StructureLinkRecordRange>(m, "RSE_StructureLinkRecordRange")
        .value("RSE_LINK_RECORD_RANGE_NONE", RSE_LINK_RECORD_RANGE_NONE)
        .value("RSE_LINK_RECORD_RANGE_ALL", RSE_LINK_RECORD_RANGE_ALL)
        .value("RSE_LINK_RECORD_RANGE_RANDOM", RSE_LINK_RECORD_RANGE_RANDOM)
        .export_values();

    py::enum_<RSE_StructureRigidBodyRecordRange>(m, "RSE_StructureRigidBodyRecordRange")
        .value("RSE_RIGID_BODY_RECORD_RANGE_NONE", RSE_RIGID_BODY_RECORD_RANGE_NONE)
        .value("RSE_RIGID_BODY_RECORD_RANGE_ALL", RSE_RIGID_BODY_RECORD_RANGE_ALL)
        .export_values();

    py::enum_<RSE_StructureJointRecordRange>(m, "RSE_StructureJointRecordRange")
        .value("RSE_JOINT_RECORD_RANGE_NONE", RSE_JOINT_RECORD_RANGE_NONE)
        .value("RSE_JOINT_RECORD_RANGE_ALL", RSE_JOINT_RECORD_RANGE_ALL)
        .export_values();

    py::class_<RS_StructureRecordConfig>(m, "RS_StructureRecordConfig")
        .def(py::init<>())
        .def_readwrite("structure_name", &RS_StructureRecordConfig::structure_name)
        .def_readwrite("voxel_record_range", &RS_StructureRecordConfig::voxel_record_range)
        .def_readwrite("link_record_range", &RS_StructureRecordConfig::link_record_range)
        .def_readwrite(
            "rigid_body_record_range", &RS_StructureRecordConfig::rigid_body_record_range
        )
        .def_readwrite("joint_record_range", &RS_StructureRecordConfig::joint_record_range)
        .def_readwrite("voxel_sample_num", &RS_StructureRecordConfig::voxel_sample_num)
        .def_readwrite("link_sample_num", &RS_StructureRecordConfig::link_sample_num)
        .def(
            py::pickle(
                [](const RS_StructureRecordConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_StructureRecordConfig>(buffer); }
            )
        );

    py::enum_<RSE_RecordType>(m, "RSE_RecordType")
        .value("RSE_RECORD_STRUCTURE", RSE_RECORD_STRUCTURE)
        .export_values();

    py::class_<RS_RecordConfig>(m, "RS_RecordConfig")
        .def(py::init<>())
        .def_readwrite("name", &RS_RecordConfig::name)
        .def_readwrite("type", &RS_RecordConfig::type)
        .def_property(
            "config",
            // Getter
            [](RS_RecordConfig &self) -> py::object {
                if (self.type == RSE_RECORD_STRUCTURE) {
                    auto &cfg = self.config.as<RS_StructureRecordConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                }
                return py::none();
            },
            // Setter
            [](RS_RecordConfig &self, py::object arg) {
                // If user passes an RS_StructureRecordConfig, store it
                if (py::isinstance<RS_StructureRecordConfig>(arg)) {
                    self.type = RSE_RECORD_STRUCTURE;
                    self.config = arg.cast<RS_StructureRecordConfig>();
                } else {
                    throw std::runtime_error(
                        "Unsupported type for config; "
                        "Supported types are: RS_StructureRecordConfig."
                    );
                }
            }
        )
        .def(
            py::pickle(
                [](const RS_RecordConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_RecordConfig>(buffer); }
            )
        );

    /***************************************************************
     * Signal configs
     ***************************************************************/
    py::enum_<RSE_SignalType>(m, "RSE_SignalType")
        .value("RSE_SIGNAL_FORCE", RSE_SIGNAL_FORCE)
        .value("RSE_SIGNAL_TORQUE", RSE_SIGNAL_TORQUE)
        .export_values();

    py::class_<RS_ForceSignalConfig>(m, "RS_ForceSignalConfig")
        .def(py::init<>())
        .def_readwrite("structure_name", &RS_ForceSignalConfig::structure_name)
        .def_readwrite("voxel_range", &RS_ForceSignalConfig::voxel_range)
        .def_readwrite("body_sid", &RS_ForceSignalConfig::body_sid)
        .def_readwrite("segment_bid", &RS_ForceSignalConfig::segment_bid)
        .def(
            py::pickle(
                [](const RS_ForceSignalConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_ForceSignalConfig>(buffer); }
            )
        );

    py::class_<RS_TorqueSignalConfig>(m, "RS_TorqueSignalConfig")
        .def(py::init<>())
        .def_readwrite("structure_name", &RS_TorqueSignalConfig::structure_name)
        .def_readwrite("body_sid", &RS_TorqueSignalConfig::body_sid)
        .def_readwrite("segment_bid", &RS_TorqueSignalConfig::segment_bid)
        .def(
            py::pickle(
                [](const RS_TorqueSignalConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_TorqueSignalConfig>(buffer); }
            )
        );

    py::class_<RS_SignalConfig>(m, "RS_SignalConfig")
        .def(py::init<>())
        .def_readwrite("name", &RS_SignalConfig::name)
        .def_readwrite("type", &RS_SignalConfig::type)
        .def_property(
            "config",
            // Getter
            [](RS_SignalConfig &self) -> py::object {
                if (self.type == RSE_SIGNAL_FORCE) {
                    auto &cfg = self.config.as<RS_ForceSignalConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                } else if (self.type == RSE_SIGNAL_TORQUE) {
                    auto &cfg = self.config.as<RS_TorqueSignalConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                }
                return py::none();
            },
            // Setter
            [](RS_SignalConfig &self, py::object arg) {
                if (py::isinstance<RS_ForceSignalConfig>(arg)) {
                    self.type = RSE_SIGNAL_FORCE;
                    self.config = arg.cast<RS_ForceSignalConfig>();
                } else if (py::isinstance<RS_TorqueSignalConfig>(arg)) {
                    self.type = RSE_SIGNAL_TORQUE;
                    self.config = arg.cast<RS_TorqueSignalConfig>();
                } else {
                    throw std::runtime_error(
                        "Unsupported type for config; "
                        "Supported types are: RS_ForceSignalConfig, RS_TorqueSignalConfig."
                    );
                }
            }
        )
        .def(
            py::pickle(
                [](const RS_SignalConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_SignalConfig>(buffer); }
            )
        );

    /***************************************************************
     * Recorder configs
     ***************************************************************/
    py::enum_<RSE_RecorderType>(m, "RSE_RecorderType")
        .value("RSE_RECORDER_HDF5", RSE_RECORDER_HDF5)
        .export_values();

    py::class_<RS_HDF5RecorderConfig>(m, "RS_HDF5RecorderConfig")
        .def(py::init<>())
        .def_readwrite("path", &RS_HDF5RecorderConfig::path)
        .def(
            py::pickle(
                [](const RS_HDF5RecorderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_HDF5RecorderConfig>(buffer); }
            )
        );

    py::class_<RS_RecorderConfig>(m, "RS_RecorderConfig")
        .def(py::init<>())
        .def_readwrite("type", &RS_RecorderConfig::type)
        .def_property(
            "config",
            // Getter
            [](RS_RecorderConfig &self) -> py::object {
                if (self.type == RSE_RECORDER_HDF5) {
                    auto &cfg = self.config.as<RS_HDF5RecorderConfig>();
                    return py::cast(&cfg, py::return_value_policy::reference);
                }
                return py::none();
            },
            // Setter
            [](RS_RecorderConfig &self, py::object arg) {
                // If user passes an RS_HDF5RecorderConfig, store it
                if (py::isinstance<RS_HDF5RecorderConfig>(arg)) {
                    self.type = RSE_RECORDER_HDF5;
                    self.config = arg.cast<RS_HDF5RecorderConfig>();
                } else {
                    throw std::runtime_error(
                        "Unsupported type for config; "
                        "Supported types are: RS_HDF5RecorderConfig."
                    );
                }
            }
        )
        .def(
            py::pickle(
                [](const RS_RecorderConfig &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_RecorderConfig>(buffer); }
            )
        );

    /***************************************************************
     * RS_Config (top level)
     ***************************************************************/
    py::bind_vector<std::vector<RS_MaterialConfig>>(m, "VecMaterialConfig");
    py::bind_vector<std::vector<RS_StructureConfig>>(m, "VecStructureConfig");
    py::bind_vector<std::vector<RS_RecordConfig>>(m, "VecRecordConfig");
    py::bind_vector<std::vector<RS_SignalConfig>>(m, "VecSignalConfig");
    py::bind_vector<std::vector<RS_RecorderConfig>>(m, "VecRecorderConfig");
    py::bind_vector<std::vector<RS_CameraConfig>>(m, "VecCameraConfig");

    py::class_<RS_Config>(m, "RS_Config")
        .def(py::init<>())
        .def(
            py::init([](py::iterable paths, bool throw_error = true) {
                std::vector<std::string> cpp_paths;
                for (auto item : paths) {
                    cpp_paths.push_back(py::cast<std::string>(item));
                }
                return RS_Config(cpp_paths, throw_error);
            }),
            py::arg("paths"),
            py::arg("throw_error") = true,
            R"doc(
         Constructs an RS_Config from a Python iterable of strings (e.g. a list).

         Args:
             paths: A list or other iterable of configuration file paths to parse
             throw_error: If True, throw an exception when validation fails. If False,
                         store validity in is_valid and the reason in invalid_reason.
        )doc"
        )
        .def_readwrite("is_valid", &RS_Config::is_valid)
        .def_readwrite("invalid_reason", &RS_Config::invalid_reason)
        .def_readwrite("simulation_config", &RS_Config::simulation_config)
        .def_readwrite("floor_config", &RS_Config::floor_config)
        .def_readwrite("world_config", &RS_Config::world_config)
        .def_readwrite("material_configs", &RS_Config::material_configs)
        .def_readwrite("structure_configs", &RS_Config::structure_configs)
        .def_readwrite("record_configs", &RS_Config::record_configs)
        .def_readwrite("signal_configs", &RS_Config::signal_configs)
        .def_readwrite("recorder_configs", &RS_Config::recorder_configs)
        .def_readwrite("camera_configs", &RS_Config::camera_configs)
        .def(
            "open",
            [](RS_Config &self, py::iterable paths, bool throw_error = true) {
                std::vector<std::string> cpp_paths;
                for (auto item : paths) {
                    cpp_paths.push_back(py::cast<std::string>(item));
                }
                self.open(cpp_paths, throw_error);
            },
            py::arg("paths"),
            py::arg("throw_error") = true,
            R"doc(
            Opens and parses configuration files.

            Args:
                paths: A list or other iterable of configuration file paths to parse
                throw_error: If True, throw an exception when validation fails. If False,
                            store validity in is_valid and the reason in invalid_reason.
        )doc"
        )
        .def(
            "validate",
            &RS_Config::validate,
            py::arg("throw_error") = true,
            R"doc(
            Validates the configuration.

            Args:
                throw_error: If True, throw an exception when validation fails. If False,
                            store validity in is_valid and the reason in invalid_reason.
        )doc"
        )
        .def(
            "saveAsBinary",
            [](const RS_Config &self) {
                std::vector<uint8_t> data = self.saveAsBinary();
                // Convert vector<uint8_t> to py::bytes
                return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
            },
            R"doc(
            Serializes the config to MsgPack bytes, returning a Python bytes object.
         )doc"
        )
        .def(
            "loadFromBinary",
            [](RS_Config &self, py::bytes pybuf) {
                // Copy from py::bytes into std::vector<uint8_t>
                std::string temp = pybuf;  // Convert bytes -> std::string
                std::vector<uint8_t> data(temp.begin(), temp.end());
                self.loadFromBinary(data);
            },
            R"doc(
            Loads the config from a Python bytes object containing MsgPack data.
         )doc"
        )
        .def(
            py::pickle(
                [](const RS_Config &self) { return serializeConfig(self); },
                [](py::bytes buffer) { return deserializeConfig<RS_Config>(buffer); }
            )
        );
}

#endif  // INIT_RSC_H
