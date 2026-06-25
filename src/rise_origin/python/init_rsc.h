// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef INIT_RSC_H
#define INIT_RSC_H

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstring>

#include "rise_origin/rise_origin.h"
#include "rise_origin/rsc/rs_rsc_file.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/utils/serialization/rs_mp_serialization.h"

namespace py = pybind11;
using namespace rise;
using namespace rise::rsc;

// True when T is an RS_Variant specialization, used to route variant element conversion.
template <typename T>
struct rspy_is_variant : std::false_type {};
template <typename... A>
struct rspy_is_variant<RS_Variant<A...>> : std::true_type {};

// The integral element type a per-voxel array exposes to numpy: enums expose their underlying type.
template <typename E, bool = std::is_enum_v<E>>
struct rspy_numpy_element {
    using type = E;
};
template <typename E>
struct rspy_numpy_element<E, true> {
    using type = std::underlying_type_t<E>;
};

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
    d.setData(
        std::vector<uint8_t>(
            reinterpret_cast<const uint8_t *>(buffer),
            reinterpret_cast<const uint8_t *>(buffer) + length
        )
    );
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

// Expose a flat std::vector member as a zero-copy writable numpy view (get) and a memcpy replace
// (set). The view pins the owning RSC Python object so the buffer stays valid.
template <typename PyClass, typename C, typename E>
void rspyBindArray(PyClass &cls, const char *name, std::vector<E> C::*member) {
    using U = typename rspy_numpy_element<E>::type;
    cls.def_property(
        name,
        [member](py::object self) {
            std::vector<E> &v = self.cast<C &>().*member;
            return py::array_t<U>(
                {static_cast<py::ssize_t>(v.size())},
                {static_cast<py::ssize_t>(sizeof(E))},
                reinterpret_cast<U *>(v.data()),
                self
            );
        },
        [member](C &c, py::array_t<U, py::array::c_style | py::array::forcecast> arr) {
            std::vector<E> &v = c.*member;
            v.resize(static_cast<std::size_t>(arr.size()));
            std::memcpy(v.data(), arr.data(), v.size() * sizeof(E));
        }
    );
}

// Register an RSCVec specialization as an opaque, list like Python type. Reads unwrap a variant
// element to its held alternative or return a live reference into the vector so mutation writes
// through; writes accept either a built variant or any of its alternatives. A def_readwrite member
// of this type then gives append in place through its getter and whole list assignment through the
// iterable conversion registered here.
template <typename T>
py::class_<rise::rsc::RSC_Base::RSCVec<T>> rspyBindRSCVec(py::handle scope, const char *name) {
    using Vec = rise::rsc::RSC_Base::RSCVec<T>;
    auto make = [](const py::object &value) -> T {
        if constexpr (rspy_is_variant<T>::value) {
            if (py::isinstance<T>(value)) {
                return value.cast<T>();
            }
            T held;
            rspyVariantSet(held, value);
            return held;
        } else {
            return value.cast<T>();
        }
    };
    auto read = [](py::handle parent, Vec &v, std::size_t i) -> py::object {
        if constexpr (rspy_is_variant<T>::value) {
            return rspyVariantGet(v[i]);
        } else {
            return py::cast(&v[i], py::return_value_policy::reference_internal, parent);
        }
    };
    auto norm = [](py::ssize_t i, std::size_t n) -> std::size_t {
        if (i < 0) {
            i += static_cast<py::ssize_t>(n);
        }
        if (i < 0 || static_cast<std::size_t>(i) >= n) {
            throw py::index_error();
        }
        return static_cast<std::size_t>(i);
    };

    py::class_<Vec> cls(scope, name);
    cls.def(py::init<>());
    cls.def(py::init([make](const py::iterable &items) {
        Vec v;
        for (py::handle item : items) {
            v.push_back(make(py::reinterpret_borrow<py::object>(item)));
        }
        return v;
    }));
    cls.def("__len__", [](const Vec &v) { return v.size(); });
    cls.def("__getitem__", [read, norm](py::object self, py::ssize_t i) {
        Vec &v = self.cast<Vec &>();
        return read(self, v, norm(i, v.size()));
    });
    cls.def("__setitem__", [make, norm](Vec &v, py::ssize_t i, const py::object &value) {
        v[norm(i, v.size())] = make(value);
    });
    cls.def("__delitem__", [norm](Vec &v, py::ssize_t i) { v.erase(norm(i, v.size())); });
    cls.def("__iter__", [read](py::object self) {
        Vec &v = self.cast<Vec &>();
        py::list out;
        for (std::size_t i = 0; i < v.size(); i++) {
            out.append(read(self, v, i));
        }
        return py::iter(out);
    });
    cls.def("append", [make](Vec &v, const py::object &value) { v.push_back(make(value)); });
    cls.def("extend", [make](Vec &v, const py::iterable &items) {
        for (py::handle item : items) {
            v.push_back(make(py::reinterpret_borrow<py::object>(item)));
        }
    });
    cls.def("insert", [make](Vec &v, py::ssize_t i, const py::object &value) {
        v.insert(i < 0 ? 0 : std::min<std::size_t>(i, v.size()), make(value));
    });
    cls.def("clear", [](Vec &v) { v.clear(); });
    py::implicitly_convertible<py::iterable, Vec>();
    return cls;
}

// Create a py::class_ for an RSC type and attach pickle then copy support. For abstract bases and
// non RSC types the extra methods are skipped, so it is safe to use for every binding.
template <typename T, typename... Options>
py::class_<T, Options...> rspyRSC(py::module &m, const char *name) {
    py::class_<T, Options...> cls(m, name);
    if constexpr (!std::is_abstract_v<T>) {
        // Add pickle methods.
        cls.def(
            py::pickle(
                [](const T &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<T>(buffer); }
            )
        );
        if constexpr (std::is_base_of_v<RSC_Base, T>) {
            // Copy methods.
            cls.def(
                "shallow_copy",
                [](py::object self, const py::kwargs &overrides) {
                    py::object copy = py::cast(T(self.cast<T &>()));
                    // Apply overrides.
                    for (auto item : overrides) {
                        std::string key = item.first.cast<std::string>();
                        if (!py::hasattr(copy, key.c_str())) {
                            throw py::type_error("unknown field '" + key + "'");
                        }
                        py::setattr(
                            copy, key.c_str(), py::reinterpret_borrow<py::object>(item.second)
                        );
                    }
                    return copy;
                },
                R"doc(
            Copy this characterization while sharing its sub RSC children.

            The copy and the original point at the same child nodes, so mutating a shared child
            shows in both. Pass field=value keyword arguments to override fields on the copy.
        )doc"
            );
            cls.def(
                "deep_copy",
                [](py::object self, const py::kwargs &overrides) {
                    auto deep = self.cast<T &>().deepCopy();
                    py::object copy = py::cast(T(*std::static_pointer_cast<T>(deep.shared())));
                    // Apply overrides.
                    for (auto item : overrides) {
                        std::string key = item.first.cast<std::string>();
                        if (!py::hasattr(copy, key.c_str())) {
                            throw py::type_error("unknown field '" + key + "'");
                        }
                        py::setattr(
                            copy, key.c_str(), py::reinterpret_borrow<py::object>(item.second)
                        );
                    }
                    return copy;
                },
                R"doc(
            Copy this characterization and all its sub RSC children independently.

            A node shared by several parents is copied once, so the sharing structure is preserved.
            Pass field=value keyword arguments to override fields on the copy.
        )doc"
            );
            cls.def(
                "save",
                [](const T &self, const std::string &path) { RS_RSCFile::save(self, path); },
                py::arg("path"),
                R"doc(
            Save to a durable file, the format chosen by the path extension.

            A .rsct path writes readable JSON text; a .rscb path writes compact binary. The two
            formats are interchangeable and carry the same data.
        )doc"
            );
            cls.def_static(
                "load",
                [](const std::string &path) { return RS_RSCFile::load<T>(path); },
                py::arg("path"),
                R"doc(
            Load from a durable file, the format chosen by the path extension.

            Reads a .rsct (JSON text) or .rscb (binary) file and verifies it holds this type.
        )doc"
            );
        }
    }
    return cls;
}

namespace pybind11 {

namespace detail {

// Bind an embedded sub RSC held by RSCRef as the referenced object, or None when null. Reading
// returns a reference into the owner; writing copies the assigned value into a fresh shared node.
template <typename T>
struct type_caster<rise::rsc::RSC_Base::RSCRef<T>> {
    using RefType = rise::rsc::RSC_Base::RSCRef<T>;
    PYBIND11_TYPE_CASTER(RefType, const_name("Optional[") + make_caster<T>::name + const_name("]"));

    bool load(handle src, bool convert) {
        if (src.is_none()) {
            value = RefType();
            return true;
        }
        make_caster<T> caster;
        if (!caster.load(src, convert)) {
            return false;
        }
        value = RefType(cast_op<T &>(caster));
        return true;
    }

    static handle cast(const RefType &src, return_value_policy, handle parent) {
        if (src.isNull()) {
            return none().release();
        }
        return pybind11::cast(src.get(), return_value_policy::reference_internal, parent).release();
    }
};

}  // namespace detail

}  // namespace pybind11

inline void initRSPYRSC(py::module &m) {
    /***************************************************************
     * Exceptions
     ***************************************************************/
    auto data_exception = py::register_exception<RS_DataException>(m, "RS_DataException");
    py::register_exception<RS_InvalidCharacterizationException>(
        m, "RS_InvalidCharacterizationException", data_exception.ptr()
    );

    /***************************************************************
     * Bases
     ***************************************************************/
    // Note: all RSC_Base classes are consolidated here. They must be bound
    // before any derived class references them.
    rspyRSC<RSC_Base>(m, "RSC_Base");

    // Opaque list like containers for every RSCVec member, registered before any member uses them.
    rspyBindRSCVec<RSC_StructureVoxelBody>(m, "RSCVecStructureVoxelBody");
    rspyBindRSCVec<RSC_StructureVoxelConstraint>(m, "RSCVecStructureVoxelConstraint");
    rspyBindRSCVec<RSC_EntityVariant>(m, "RSCVecEntity");
    rspyBindRSCVec<RSC_EssenceVariant>(m, "RSCVecEssence");
    rspyBindRSCVec<RSC_ResourceVariant>(m, "RSCVecResource");
    rspyBindRSCVec<RSC_RecordVariant>(m, "RSCVecRecord");
    rspyBindRSCVec<RSC_SignalVariant>(m, "RSCVecSignal");
    rspyBindRSCVec<RSC_ControllerVariant>(m, "RSCVecController");
    rspyBindRSCVec<RSC_RecorderVariant>(m, "RSCVecRecorder");

    rspyRSC<RSC_ResourceBase, RSC_Base>(m, "RSC_ResourceBase")
        .def_readwrite("name", &RSC_ResourceBase::name);

    rspyRSC<RSC_EssenceBase, RSC_Base>(m, "RSC_EssenceBase")
        .def_readwrite("name", &RSC_EssenceBase::name);

    rspyRSC<RSC_EntityBase, RSC_Base>(m, "RSC_EntityBase")
        .def_readwrite("name", &RSC_EntityBase::name);

    rspyRSC<RSC_RecordBase, RSC_Base>(m, "RSC_RecordBase")
        .def_readwrite("name", &RSC_RecordBase::name)
        .def_readwrite("target_controller_reference", &RSC_RecordBase::target_controller_reference)
        .def_readwrite("target_recorder_reference", &RSC_RecordBase::target_recorder_reference);

    rspyRSC<RSC_SignalBase, RSC_Base>(m, "RSC_SignalBase")
        .def_readwrite("name", &RSC_SignalBase::name)
        .def_readwrite("source_controller_name", &RSC_SignalBase::source_controller_name);

    rspyRSC<RSC_RecorderBase, RSC_Base>(m, "RSC_RecorderBase")
        .def_readwrite("name", &RSC_RecorderBase::name);

    rspyRSC<RSC_ControllerBase, RSC_Base>(m, "RSC_ControllerBase")
        .def_readwrite("name", &RSC_ControllerBase::name);

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
        );

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
        .def_readwrite("is_enabled", &RSC_SimulationSentinel::is_enabled);

    rspyRSC<RSC_SimulationControl, RSC_Base>(m, "RSC_SimulationControl")
        .def(py::init<>())
        .def_readwrite("frequency", &RSC_SimulationControl::frequency);

    rspyRSC<RSC_SimulationRecord, RSC_Base>(m, "RSC_SimulationRecord")
        .def(py::init<>())
        .def_readwrite("frequency", &RSC_SimulationRecord::frequency);

    rspyRSC<RSC_SimulationRuntime, RSC_Base>(m, "RSC_SimulationRuntime")
        .def(py::init<>())
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

    /***************************************************************
     * Resources
     ***************************************************************/
    rspyRSC<RSC_TextureBase, RSC_ResourceBase>(m, "RSC_TextureBase");

    rspyRSC<RSC_TextureConstant, RSC_TextureBase>(m, "RSC_TextureConstant")
        .def(py::init<>())
        .def_readwrite("value", &RSC_TextureConstant::value);

    rspyRSC<RSC_TextureGrid, RSC_TextureBase>(m, "RSC_TextureGrid")
        .def(py::init<>())
        .def_readwrite("fill_color", &RSC_TextureGrid::fill_color)
        .def_readwrite("line_color", &RSC_TextureGrid::line_color)
        .def_readwrite("is_meters", &RSC_TextureGrid::is_meters)
        .def_readwrite("x_spacing", &RSC_TextureGrid::x_spacing)
        .def_readwrite("y_spacing", &RSC_TextureGrid::y_spacing)
        .def_readwrite("line_width", &RSC_TextureGrid::line_width);

    /***************************************************************
     * Essences
     ***************************************************************/
    rspyRSC<RSC_Material, RSC_EssenceBase>(m, "RSC_Material").def(py::init<>());

    rspyRSC<RSC_MaterialUnaryProperty, RSC_EssenceBase>(m, "RSC_MaterialUnaryProperty")
        .def_readwrite("material_reference", &RSC_MaterialUnaryProperty::material_reference);

    rspyRSC<RSC_MaterialPhysicalPropertyBase, RSC_MaterialUnaryProperty>(
        m, "RSC_MaterialPhysicalPropertyBase"
    );

    rspyRSC<RSC_MaterialPhysicalPropertyFixed, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertyFixed"
    )
        .def(py::init<>());

    rspyRSC<RSC_MaterialPhysicalPropertySoft, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertySoft"
    )
        .def(py::init<>())
        .def_readwrite("elastic_modulus", &RSC_MaterialPhysicalPropertySoft::elastic_modulus)
        .def_readwrite("fail_stress", &RSC_MaterialPhysicalPropertySoft::fail_stress)
        .def_readwrite("density", &RSC_MaterialPhysicalPropertySoft::density)
        .def_readwrite("poissons_ratio", &RSC_MaterialPhysicalPropertySoft::poissons_ratio)
        .def_readwrite("min_expansion", &RSC_MaterialPhysicalPropertySoft::min_expansion)
        .def_readwrite("max_expansion", &RSC_MaterialPhysicalPropertySoft::max_expansion);

    rspyRSC<RSC_MaterialPhysicalPropertyRigid, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertyRigid"
    )
        .def(py::init<>())
        .def_readwrite("density", &RSC_MaterialPhysicalPropertyRigid::density);

    rspyRSC<RSC_MaterialPhysicalPropertyAtlas, RSC_MaterialPhysicalPropertyBase>(
        m, "RSC_MaterialPhysicalPropertyAtlas"
    )
        .def(py::init<>())
        .def_readwrite("mass", &RSC_MaterialPhysicalPropertyAtlas::mass)
        .def_readwrite("com_position", &RSC_MaterialPhysicalPropertyAtlas::com_position)
        .def_readwrite("inertia", &RSC_MaterialPhysicalPropertyAtlas::inertia);

    rspyRSC<RSC_MaterialOpticalPropertyBase, RSC_MaterialUnaryProperty>(
        m, "RSC_MaterialOpticalPropertyBase"
    );

    rspyRSC<RSC_MaterialOpticalPropertySimpleMicrofacet, RSC_MaterialOpticalPropertyBase>(
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
        );

    rspyRSC<RSC_MaterialPairPropertyBase, RSC_EssenceBase>(m, "RSC_MaterialPairPropertyBase")
        .def_readwrite(
            "first_material_reference", &RSC_MaterialPairPropertyBase::first_material_reference
        )
        .def_readwrite(
            "second_material_reference", &RSC_MaterialPairPropertyBase::second_material_reference
        );

    rspyRSC<RSC_MaterialPairPropertyCoulombFriction, RSC_MaterialPairPropertyBase>(
        m, "RSC_MaterialPairPropertyCoulombFriction"
    )
        .def(py::init<>())
        .def_readwrite("u_static", &RSC_MaterialPairPropertyCoulombFriction::u_static)
        .def_readwrite("u_dynamic", &RSC_MaterialPairPropertyCoulombFriction::u_dynamic);

    rspyRSC<RSC_MaterialPairPropertyRestitution, RSC_MaterialPairPropertyBase>(
        m, "RSC_MaterialPairPropertyRestitution"
    )
        .def(py::init<>())
        .def_readwrite("restitution", &RSC_MaterialPairPropertyRestitution::restitution);

    /***************************************************************
     * Entities
     ***************************************************************/

    // --- Cosmic background ---
    rspyRSC<RSC_CosmicBackgroundBase, RSC_EntityBase>(m, "RSC_CosmicBackgroundBase");

    rspyRSC<RSC_CosmicBackgroundUniform, RSC_CosmicBackgroundBase>(m, "RSC_CosmicBackgroundUniform")
        .def(py::init<>())
        .def_readwrite("texture_reference", &RSC_CosmicBackgroundUniform::texture_reference)
        .def_readwrite("intensity", &RSC_CosmicBackgroundUniform::intensity)
        .def_readwrite("rotation_rad", &RSC_CosmicBackgroundUniform::rotation_rad);

    rspyRSC<RSC_CosmicBackgroundGradient, RSC_CosmicBackgroundBase>(
        m, "RSC_CosmicBackgroundGradient"
    )
        .def(py::init<>())
        .def_readwrite("zenith_color", &RSC_CosmicBackgroundGradient::zenith_color)
        .def_readwrite("horizon_color", &RSC_CosmicBackgroundGradient::horizon_color)
        .def_readwrite("sun_direction", &RSC_CosmicBackgroundGradient::sun_direction)
        .def_readwrite("sun_color", &RSC_CosmicBackgroundGradient::sun_color)
        .def_readwrite("sun_angular_radius", &RSC_CosmicBackgroundGradient::sun_angular_radius);

    rspyRSC<RSC_CosmicBackgroundHosekWilkie, RSC_CosmicBackgroundBase>(
        m, "RSC_CosmicBackgroundHosekWilkie"
    )
        .def(py::init<>())
        .def_readwrite("sun_direction", &RSC_CosmicBackgroundHosekWilkie::sun_direction)
        .def_readwrite("sun_color", &RSC_CosmicBackgroundHosekWilkie::sun_color)
        .def_readwrite("sun_angular_radius", &RSC_CosmicBackgroundHosekWilkie::sun_angular_radius)
        .def_readwrite("turbidity", &RSC_CosmicBackgroundHosekWilkie::turbidity)
        .def_readwrite("ground_color", &RSC_CosmicBackgroundHosekWilkie::ground_color)
        .def_readwrite("intensity", &RSC_CosmicBackgroundHosekWilkie::intensity);

    // --- Floor ---
    rspyRSC<RSC_FloorBase, RSC_EntityBase>(m, "RSC_FloorBase")
        .def_readwrite("material_reference", &RSC_FloorBase::material_reference);

    rspyRSC<RSC_FloorFlat, RSC_FloorBase>(m, "RSC_FloorFlat")
        .def(py::init<>())
        .def_readwrite("height", &RSC_FloorFlat::height)
        .def_readwrite("uv_scale", &RSC_FloorFlat::uv_scale);

    auto cls_floor_array = rspyRSC<RSC_FloorArray, RSC_FloorBase>(m, "RSC_FloorArray");
    cls_floor_array.def(py::init<>())
        .def_readwrite("x_size", &RSC_FloorArray::x_size)
        .def_readwrite("y_size", &RSC_FloorArray::y_size)
        .def_readwrite("x_values", &RSC_FloorArray::x_values)
        .def_readwrite("y_values", &RSC_FloorArray::y_values)
        .def_readwrite("is_x_repeat", &RSC_FloorArray::is_x_repeat)
        .def_readwrite("is_y_repeat", &RSC_FloorArray::is_y_repeat);
    rspyBindArray(cls_floor_array, "height", &RSC_FloorArray::height);

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

    auto cls_body = rspyRSC<RSC_StructureVoxelBody, RSC_Base>(m, "RSC_StructureVoxelBody");
    cls_body.def(py::init<>())
        .def_readwrite(
            "relative_origin_position", &RSC_StructureVoxelBody::relative_origin_position
        )
        .def_readwrite("relative_orientation", &RSC_StructureVoxelBody::relative_orientation)
        .def_readwrite("x_voxels", &RSC_StructureVoxelBody::x_voxels)
        .def_readwrite("y_voxels", &RSC_StructureVoxelBody::y_voxels)
        .def_readwrite("z_voxels", &RSC_StructureVoxelBody::z_voxels)
        .def_readwrite("voxel_size", &RSC_StructureVoxelBody::voxel_size);
    rspyBindArray(
        cls_body, "material_reference_sid", &RSC_StructureVoxelBody::material_reference_sid
    );
    rspyBindArray(cls_body, "segment_bid", &RSC_StructureVoxelBody::segment_bid);
    rspyBindArray(cls_body, "expansion_signal_sid", &RSC_StructureVoxelBody::expansion_signal_sid);
    rspyBindArray(
        cls_body, "expansion_control_mode", &RSC_StructureVoxelBody::expansion_control_mode
    );
    rspyBindArray(cls_body, "link", &RSC_StructureVoxelBody::link);

    rspyRSC<RSC_StructureVoxelConstraint, RSC_Base>(m, "RSC_StructureVoxelConstraint")
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
        .def_readwrite("prismatic_signal_sid", &RSC_StructureVoxelConstraint::prismatic_signal_sid);

    rspyRSC<RSC_StructureVoxelRecord, RSC_RecordBase>(m, "RSC_StructureVoxelRecord")
        .def(py::init<>())
        .def_readwrite("voxel_record_range", &RSC_StructureVoxelRecord::voxel_record_range)
        .def_readwrite("link_record_range", &RSC_StructureVoxelRecord::link_record_range)
        .def_readwrite(
            "rigid_body_record_range", &RSC_StructureVoxelRecord::rigid_body_record_range
        )
        .def_readwrite("joint_record_range", &RSC_StructureVoxelRecord::joint_record_range)
        .def_readwrite("voxel_sample_num", &RSC_StructureVoxelRecord::voxel_sample_num)
        .def_readwrite("link_sample_num", &RSC_StructureVoxelRecord::link_sample_num);

    rspyRSC<RSC_StructureVoxelSignal, RSC_SignalBase>(m, "RSC_StructureVoxelSignal")
        .def(py::init<>())
        .def_readwrite("expansion_signal_num", &RSC_StructureVoxelSignal::expansion_signal_num)
        .def_readwrite("hinge_signal_num", &RSC_StructureVoxelSignal::hinge_signal_num)
        .def_readwrite("prismatic_signal_num", &RSC_StructureVoxelSignal::prismatic_signal_num);

    rspyRSC<RSC_StructureBase, RSC_EntityBase>(m, "RSC_StructureBase")
        .def_readwrite("origin_position", &RSC_StructureBase::origin_position)
        .def_readwrite("orientation", &RSC_StructureBase::orientation)
        .def_readwrite("material_references", &RSC_StructureBase::material_references);

    rspyRSC<RSC_StructureVoxel, RSC_StructureBase>(m, "RSC_StructureVoxel")
        .def(py::init<>())
        .def_readwrite("is_static", &RSC_StructureVoxel::is_fixed)
        .def_readwrite("bodies", &RSC_StructureVoxel::bodies)
        .def_readwrite("constraints", &RSC_StructureVoxel::constraints)
        .def_readwrite("record", &RSC_StructureVoxel::record)
        .def_readwrite("signal", &RSC_StructureVoxel::signal);

    // --- Camera ---
    rspyRSC<RSC_CameraPerspectiveRecord, RSC_RecordBase>(m, "RSC_CameraPerspectiveRecord")
        .def(py::init<>())
        .def_readwrite("has_rgb", &RSC_CameraPerspectiveRecord::has_rgb)
        .def_readwrite("has_depth", &RSC_CameraPerspectiveRecord::has_depth)
        .def_readwrite("has_entity_info", &RSC_CameraPerspectiveRecord::has_entity_info);

    rspyRSC<RSC_CameraPerspectiveSignal, RSC_SignalBase>(m, "RSC_CameraPerspectiveSignal")
        .def(py::init<>())
        .def_readwrite("position_signal_num", &RSC_CameraPerspectiveSignal::position_signal_num)
        .def_readwrite(
            "orientation_signal_num", &RSC_CameraPerspectiveSignal::orientation_signal_num
        )
        .def_readwrite("optics_signal_num", &RSC_CameraPerspectiveSignal::optics_signal_num);

    rspyRSC<RSC_CameraBase, RSC_EntityBase>(m, "RSC_CameraBase");

    rspyRSC<RSC_CameraPerspectiveBase, RSC_CameraBase>(m, "RSC_CameraPerspectiveBase")
        .def_readwrite("image_width", &RSC_CameraPerspectiveBase::image_width)
        .def_readwrite("image_height", &RSC_CameraPerspectiveBase::image_height)
        .def_readwrite("samples_per_pixel", &RSC_CameraPerspectiveBase::samples_per_pixel)
        .def_readwrite("max_depth", &RSC_CameraPerspectiveBase::max_depth)
        .def_readwrite("field_of_view", &RSC_CameraPerspectiveBase::field_of_view)
        .def_readwrite("defocus_angle", &RSC_CameraPerspectiveBase::defocus_angle)
        .def_readwrite("focus_distance", &RSC_CameraPerspectiveBase::focus_distance)
        .def_readwrite("record", &RSC_CameraPerspectiveBase::record)
        .def_readwrite("signal", &RSC_CameraPerspectiveBase::signal);

    rspyRSC<RSC_CameraPerspectiveFree, RSC_CameraPerspectiveBase>(m, "RSC_CameraPerspectiveFree")
        .def(py::init<>())
        .def_readwrite("origin_position", &RSC_CameraPerspectiveFree::origin_position)
        .def_readwrite("orientation", &RSC_CameraPerspectiveFree::orientation);

    rspyRSC<RSC_CameraPerspectiveRigidBodyLinked, RSC_CameraPerspectiveBase>(
        m, "RSC_CameraPerspectiveRigidBodyLinked"
    )
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_CameraPerspectiveRigidBodyLinked::structure_name)
        .def_readwrite("body_sid", &RSC_CameraPerspectiveRigidBodyLinked::body_sid)
        .def_readwrite("segment_bid", &RSC_CameraPerspectiveRigidBodyLinked::segment_bid)
        .def_readwrite("relative_anchor", &RSC_CameraPerspectiveRigidBodyLinked::relative_anchor)
        .def_readwrite(
            "relative_orientation", &RSC_CameraPerspectiveRigidBodyLinked::relative_orientation
        );

    /***************************************************************
     * Record
     ***************************************************************/
    rspyRSC<RSC_IntersectRegionCuboid, RSC_RecordBase>(m, "RSC_IntersectRegionCuboid")
        .def(py::init<>())
        .def_readwrite("position", &RSC_IntersectRegionCuboid::position)
        .def_readwrite("orientation", &RSC_IntersectRegionCuboid::orientation)
        .def_readwrite("size", &RSC_IntersectRegionCuboid::size);

    rspyRSC<RSC_IntersectRegionCuboidRigidBodyLinked, RSC_RecordBase>(
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
        .def_readwrite("size", &RSC_IntersectRegionCuboidRigidBodyLinked::size);

    rspyRSC<RSC_IntersectRegionSphere, RSC_RecordBase>(m, "RSC_IntersectRegionSphere")
        .def(py::init<>())
        .def_readwrite("position", &RSC_IntersectRegionSphere::position)
        .def_readwrite("radius", &RSC_IntersectRegionSphere::radius);

    /***************************************************************
     * Signal
     ***************************************************************/
    rspyRSC<RSC_Force, RSC_SignalBase>(m, "RSC_Force")
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_Force::structure_name)
        .def_readwrite("voxel_range", &RSC_Force::voxel_range)
        .def_readwrite("body_sid", &RSC_Force::body_sid)
        .def_readwrite("segment_bid", &RSC_Force::segment_bid);

    rspyRSC<RSC_Torque, RSC_SignalBase>(m, "RSC_Torque")
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_Torque::structure_name)
        .def_readwrite("body_sid", &RSC_Torque::body_sid)
        .def_readwrite("segment_bid", &RSC_Torque::segment_bid);

    rspyRSC<RSC_Transform, RSC_SignalBase>(m, "RSC_Transform")
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_Transform::structure_name);

    /***************************************************************
     * Recorder
     ***************************************************************/
    rspyRSC<RSC_RecorderHDF5, RSC_RecorderBase>(m, "RSC_RecorderHDF5")
        .def(py::init<>())
        .def_readwrite("path", &RSC_RecorderHDF5::path);

    /***************************************************************
     * Controller
     ***************************************************************/
    rspyRSC<RSC_ControllerCallback, RSC_ControllerBase>(m, "RSC_ControllerCallback")
        .def(py::init<>());

    /***************************************************************
     * World (top level)
     ***************************************************************/
    // Variant aliases exposed as Python types so that users can construct them
    // from any of their alternatives and inspect the held value via `value`.
    rspyRSC<RSC_ResourceVariant>(m, "RSC_ResourceVariant")
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
        );

    rspyRSC<RSC_EssenceVariant>(m, "RSC_EssenceVariant")
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
        );

    rspyRSC<RSC_EntityVariant>(m, "RSC_EntityVariant")
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
        );

    rspyRSC<RSC_RecordVariant>(m, "RSC_RecordVariant")
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
        );

    rspyRSC<RSC_SignalVariant>(m, "RSC_SignalVariant")
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
        );

    rspyRSC<RSC_RecorderVariant>(m, "RSC_RecorderVariant")
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
        );

    rspyRSC<RSC_ControllerVariant>(m, "RSC_ControllerVariant")
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
        );

    rspyRSC<RSC_World, RSC_Base>(m, "RSC_World")
        .def(py::init<>())
        .def_readwrite("name", &RSC_World::name)
        .def_readwrite("simulation", &RSC_World::simulation)
        .def_readwrite("resources", &RSC_World::resources)
        .def_readwrite("essences", &RSC_World::essences)
        .def_readwrite("entities", &RSC_World::entities)
        .def_readwrite("records", &RSC_World::records)
        .def_readwrite("signals", &RSC_World::signals)
        .def_readwrite("recorders", &RSC_World::recorders)
        .def_readwrite("controllers", &RSC_World::controllers)
        .def(
            "assemble_from",
            [](RSC_World &self, py::iterable paths) {
                std::vector<std::string> cpp_paths;
                for (auto item : paths) {
                    cpp_paths.push_back(py::cast<std::string>(item));
                }
                self.assembleFrom(cpp_paths);
            },
            py::arg("paths"),
            R"doc(
            Assemble this world from saved RSC files (.rsct or .rscb), read in order.

            Each file holds either a single component (resource, essence, entity, record, signal,
            recorder, controller), which appends to its section, or a world, which appends its
            components and replaces the simulation.

            Args:
                paths: A list or other iterable of RSC file paths to load.

            Raises:
                RS_DataException: When a path cannot be opened or parsed.
        )doc"
        );

    py::class_<RS_RSCValidator>(m, "RS_RSCValidator")
        .def(py::init<>())
        .def(
            "validate",
            &RS_RSCValidator::validate,
            py::arg("rsc"),
            R"doc(
            Validates a characterization and raises on the first failed check.

            Args:
                rsc: Any RSC_Base subclass, such as an RSC_World, to validate.

            Raises:
                RS_InvalidCharacterizationException: When a check fails.
        )doc"
        );
}

#endif  // INIT_RSC_H
