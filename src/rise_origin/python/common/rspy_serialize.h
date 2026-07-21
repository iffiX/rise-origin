// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_SERIALIZE_H
#define RSPY_SERIALIZE_H

#include <stdexcept>

#include "common/rspy_pybind.h"
#include "common/rspy_traits.h"

/**
 * @brief Serialize an RSC node or variant to bytes through the in-engine archive.
 * @param rsc The node or variant to serialize.
 */
template <typename T>
py::bytes rspySerializeRSC(const T &rsc) {
    RS_Bytes bytes;
    if constexpr (RSPY_IsVariant<T>::value) {
        rsc.visit([&bytes](const auto &held) { bytes = RS_RSCArchive::toBytesAny(held); });
    } else {
        bytes = RS_RSCArchive::toBytes(rsc);
    }
    RS_BytesView view = bytes.toView();
    return py::bytes(reinterpret_cast<const char *>(view.ptr()), view.size());
}

/**
 * @brief Deserialize bytes into an RSC node or variant through the in-engine archive.
 * @param pybuf The pickled bytes.
 */
template <typename T>
T rspyDeserializeRSC(const py::bytes &pybuf) {
    char *buffer;
    Py_ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(pybuf.ptr(), &buffer, &length)) {
        throw std::runtime_error("Failed to extract bytes contents");
    }
    RS_BytesView view(reinterpret_cast<const RByte *>(buffer), static_cast<RBigSize>(length));
    if constexpr (RSPY_IsVariant<T>::value) {
        auto node = RS_RSCArchive::fromBytesAny(view);
        T variant;
        if (!RS_RSCRegistry::tryAssign(variant, node)) {
            throw std::runtime_error("Pickled RSC type is not valid for this variant");
        }
        return variant;
    } else {
        return RS_RSCArchive::fromBytes<T>(view);
    }
}

#endif  // RSPY_SERIALIZE_H
