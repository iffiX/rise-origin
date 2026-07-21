// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_TRAITS_H
#define RSPY_TRAITS_H

#include <type_traits>
#include <vector>

#include "common/rspy_pybind.h"

/**
 * @brief True when T is an RS_Variant specialization, used to route variant element conversion.
 */
template <typename T>
struct RSPY_IsVariant : std::false_type {};
template <typename... A>
struct RSPY_IsVariant<RS_Variant<A...>> : std::true_type {};

/**
 * @brief The integral element a per-voxel array exposes to numpy, unwrapping an enum to its base.
 */
template <typename E, bool = std::is_enum_v<E>>
struct RSPY_NumpyElement {
    using type = E;
};
template <typename E>
struct RSPY_NumpyElement<E, true> {
    using type = std::underlying_type_t<E>;
};

/**
 * @brief True when E is a std::vector, used to route a nested vector member to a recursive subview.
 */
template <typename E>
struct RSPY_IsVector : std::false_type {};
template <typename U, typename A>
struct RSPY_IsVector<std::vector<U, A>> : std::true_type {};

#endif  // RSPY_TRAITS_H
