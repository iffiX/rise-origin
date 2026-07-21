// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_VARIANT_H
#define RSPY_VARIANT_H

#include <stdexcept>

#include "common/rspy_pybind.h"

/**
 * @brief Read an RS_Variant field as its held alternative by reference, or None when empty.
 * @param v The variant field.
 */
template <typename... Alternatives>
py::object rspyVariantGet(RS_Variant<Alternatives...> &v) {
    if (v.index() == RS_NULL_INDEX) {
        return py::none();
    }
    py::object result = py::none();
    v.visit([&](auto &held) { result = py::cast(&held, py::return_value_policy::reference); });
    return result;
}

/**
 * @brief Assign arg to the variant when it matches alternative Alt.
 * @param v The variant field.
 * @param arg The Python value to assign.
 * @return True when arg matched Alt and was assigned.
 */
template <typename Alt, typename... Alternatives>
bool rspyVariantTryAssign(RS_Variant<Alternatives...> &v, py::object arg) {
    if (py::isinstance<Alt>(arg)) {
        v = arg.cast<Alt>();
        return true;
    }
    return false;
}

/**
 * @brief Assign a Python value to a variant field, dispatching on the concrete alternative type.
 * @param v The variant field.
 * @param arg The Python value to assign.
 */
template <typename... Alternatives>
void rspyVariantSet(RS_Variant<Alternatives...> &v, py::object arg) {
    bool ok = false;
    ((ok = ok || rspyVariantTryAssign<Alternatives, Alternatives...>(v, arg)), ...);
    if (!ok) {
        throw std::runtime_error("Unsupported type assigned to variant field");
    }
}

/**
 * @brief Register each alternative as implicitly convertible to its variant, so a builder can be
 * passed wherever the variant is expected, such as handle.add_entity(StructureVoxel(...)).
 */
template <typename... Alternatives>
void rspyRegisterVariantConversions(const RS_Variant<Alternatives...> *) {
    (py::implicitly_convertible<Alternatives, RS_Variant<Alternatives...>>(), ...);
}

#endif  // RSPY_VARIANT_H
