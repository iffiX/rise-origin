// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_RSC_SERIALIZE_H
#define RSPY_RSC_SERIALIZE_H

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "common/rspy_pybind.h"
#include "common/rspy_traits.h"

/**
 * @brief Owns an archive byte blob and exposes it through the buffer protocol, so pickle can view it
 * zero copy through a PickleBuffer rather than copying it into a py::bytes.
 */
struct RSPY_ArchiveBytes {
    RS_Bytes bytes;
};

/**
 * @brief Register the archive byte owner. Called once during module init.
 * @param m The module to register into.
 */
inline void rspyRegisterArchiveBytes(py::module &m) {
    py::class_<RSPY_ArchiveBytes>(m, "RS_ArchiveBytes", py::buffer_protocol())
        .def_buffer([](RSPY_ArchiveBytes &self) -> py::buffer_info {
            RS_BytesView view = self.bytes.toView();
            return py::buffer_info(
                const_cast<RByte *>(view.ptr()),
                static_cast<py::ssize_t>(sizeof(RByte)),
                py::format_descriptor<std::uint8_t>::format(),
                1,
                {static_cast<py::ssize_t>(view.size())},
                {static_cast<py::ssize_t>(sizeof(RByte))},
                true
            );
        });
}

/**
 * @brief Serialize an RSC node or variant to archive bytes.
 * @param rsc The node or variant to serialize.
 */
template <typename T>
RS_Bytes rspyArchiveRSC(const T &rsc) {
    RS_Bytes bytes;
    if constexpr (RSPY_IsVariant<T>::value) {
        rsc.visit([&bytes](const auto &held) { bytes = RS_RSCArchive::toBytesAny(held); });
    } else {
        bytes = RS_RSCArchive::toBytes(rsc);
    }
    return bytes;
}

/**
 * @brief The pickle __getstate__ state, a plain bytes that every protocol can carry.
 * @param rsc The node or variant to serialize.
 */
template <typename T>
py::bytes rspySerializeRSC(const T &rsc) {
    RS_Bytes bytes = rspyArchiveRSC(rsc);
    RS_BytesView view = bytes.toView();
    return py::bytes(reinterpret_cast<const char *>(view.ptr()), view.size());
}

/**
 * @brief The protocol 5 state, a PickleBuffer that views the archive bytes with no extra copy.
 * @param rsc The node or variant to serialize.
 */
template <typename T>
py::object rspySerializeRSCZeroCopy(const T &rsc) {
    RSPY_ArchiveBytes owner{rspyArchiveRSC(rsc)};
    py::object pickle_buffer = py::module_::import("pickle").attr("PickleBuffer");
    return pickle_buffer(py::cast(std::move(owner)));
}

/**
 * @brief Deserialize an RSC node or variant from any buffer produced by the pickle state, a bytes
 * or a PickleBuffer.
 * @param state The pickled state, any object supporting the buffer protocol.
 */
template <typename T>
T rspyDeserializeRSC(const py::object &state) {
    py::buffer_info info = py::reinterpret_borrow<py::buffer>(state).request();
    RBigSize size = static_cast<RBigSize>(info.size) * static_cast<RBigSize>(info.itemsize);
    RS_BytesView view(reinterpret_cast<const RByte *>(info.ptr), size);
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

#endif  // RSPY_RSC_SERIALIZE_H
