// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_STATIC_CONSTRUCT_H
#define RS_STATIC_CONSTRUCT_H

#include "platform/rs_platform.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace device {
/**
 * @brief Applies a function to each index in a compile time range.
 *
 * The function is called with each index wrapped in rs_device_std::integral_constant.
 *
 * @tparam Start The start index.
 * @tparam Stop The stop index.
 * @tparam Step The step size.
 */
template <RSize Start, RSize Stop, RSize Step = 1, typename Func>
HOST_DEVICE_INLINE_QUALIFIER constexpr void loop(Func&& func) {
    static_assert(Step > 0, "Step must be greater than 0");
    if constexpr (Start < Stop) {
        func(rs_device_std::integral_constant<RIndex, Start>{});
        loop<Start + Step, Stop, Step>(rs_device_std::forward<Func>(func));
    }
}

/**
 * @brief Applies a function to each index in a compile-time sequence.
 *
 * The indices are provided as non-type template parameters, and the function
 * is called for each index.
 *
 * @tparam Indices Compile-time constant indices.
 * @tparam Func Type of the function to apply.
 * @param func The function to apply.
 */
template <RIndex... Indices, typename Func>
HOST_DEVICE_INLINE_QUALIFIER constexpr void forEach(Func&& func) {
    (func(Indices), ...);
}

/**
 * @brief Represents a 2D coordinate with compile-time X and Y values.
 *
 * @tparam XVal Compile-time constant for the X coordinate.
 * @tparam YVal Compile-time constant for the Y coordinate.
 */
template <int XVal, int YVal>
struct RS_ForEach2D {
    static constexpr int X = XVal;
    static constexpr int Y = YVal;
};

/**
 * @brief Container for a sequence of 2D coordinates.
 *
 * @tparam Coords A parameter pack of 2D coordinate types.
 */
template <typename... Coords>
struct RS_ForEach2DSequence {};

/**
 * @brief Applies a function to each 2D coordinate provided as template parameters.
 *
 * The function is called with each coordinate's X and Y values, wrapped in
 * rs_device_std::integral_constant.
 *
 * @tparam Coords A pack of 2D coordinate types.
 * @tparam Func Type of the function to apply.
 * @param func The function to call.
 */
template <typename... Coords, typename Func>
HOST_DEVICE_INLINE_QUALIFIER constexpr void forEach2D(Func&& func) {
    (func(
         rs_device_std::integral_constant<int, Coords::X>{},
         rs_device_std::integral_constant<int, Coords::Y>{}
     ),
     ...);
}

/**
 * @brief Represents a 3D coordinate with compile-time X, Y, and Z values.
 *
 * @tparam XVal Compile-time constant for the X coordinate.
 * @tparam YVal Compile-time constant for the Y coordinate.
 * @tparam ZVal Compile-time constant for the Z coordinate.
 */
template <int XVal, int YVal, int ZVal>
struct RS_ForEach3D {
    static constexpr int X = XVal;
    static constexpr int Y = YVal;
    static constexpr int Z = ZVal;
};

/**
 * @brief Container for a sequence of 3D coordinates.
 *
 * @tparam Coords A parameter pack of 3D coordinate types.
 */
template <typename... Coords>
struct RS_ForEach3DSequence {};

/**
 * @brief Applies a function to each 3D coordinate provided as template parameters.
 *
 * The function is called with each coordinate's X, Y, and Z values, wrapped in
 * rs_device_std::integral_constant.
 *
 * @tparam Coords A pack of 3D coordinate types.
 * @tparam Func Type of the function to apply.
 * @param func The function to call.
 */
template <typename... Coords, typename Func>
HOST_DEVICE_INLINE_QUALIFIER constexpr void forEach3D(Func&& func) {
    (func(
         rs_device_std::integral_constant<int, Coords::X>{},
         rs_device_std::integral_constant<int, Coords::Y>{},
         rs_device_std::integral_constant<int, Coords::Z>{}
     ),
     ...);
}

/**
 * @brief Switches over a compile-time index to call a function corresponding to the matching
 * compile-time range.
 *
 * For each compile-time index, if it matches the runtime index, the function is called
 * with that index wrapped in rs_device_std::integral_constant.
 *
 * @tparam Start The start index.
 * @tparam Stop The stop index.
 * @tparam Step The step size.
 * @param index The runtime index.
 * @param func The function to call if a match is found.
 */
template <RSize Start, RSize Stop, RSize Step = 1, typename Func>
HOST_DEVICE_INLINE_QUALIFIER constexpr void switchOver(RSize index, Func&& func) {
    static_assert(Step > 0, "Step must be greater than 0");
    if constexpr (Start < Stop) {
        if (index == Start)
            func(rs_device_std::integral_constant<RIndex, Start>{});
        else
            switchOver<Start + Step, Stop, Step>(index, rs_device_std::forward<Func>(func));
    }
}

/**
 * @brief Switches over a runtime index to call a function corresponding to the matching
 * compile-time index.
 *
 * For each compile-time index, if it matches the runtime index, the function is called
 * with that index wrapped in rs_device_std::integral_constant.
 *
 * @tparam Indices A pack of compile-time constant indices.
 * @tparam Func Type of the function to call.
 * @param index The runtime index.
 * @param func The function to call if a match is found.
 */
template <RIndex... Indices, typename Func>
HOST_DEVICE_INLINE_QUALIFIER constexpr void switchEach(RIndex index, Func&& func) {
    bool matched =
        ((Indices == index ? (func(rs_device_std::integral_constant<RIndex, Indices>{}), true)
                           : false) ||
         ...);
    (void)matched;
}

}  // namespace device

}  // namespace rise

#endif  // RS_STATIC_CONSTRUCT_H
