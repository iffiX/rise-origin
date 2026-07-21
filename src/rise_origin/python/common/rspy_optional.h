// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_OPTIONAL_H
#define RSPY_OPTIONAL_H

#include "common/rspy_pybind.h"

/**
 * @brief Read an RS_Optional field as its value, or None when empty.
 * @param o The optional field.
 */
template <typename T>
py::object rspyOptionalGet(const RS_Optional<T> &o) {
    return o.empty() ? py::none() : py::cast(o.get());
}

/**
 * @brief Assign a Python value to an RS_Optional field, clearing it when None.
 * @param o The optional field.
 * @param arg The Python value to assign, or None.
 */
template <typename T>
void rspyOptionalSet(RS_Optional<T> &o, py::object arg) {
    if (arg.is_none()) {
        o.reset();
    } else {
        o = arg.cast<T>();
    }
}

#endif  // RSPY_OPTIONAL_H
