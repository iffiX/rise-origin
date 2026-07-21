// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_DATA_TRAITS_H
#define RS_DATA_TRAITS_H

#include <array>
#include <list>
#include <map>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rise_origin/utils/rs_def.h"

namespace rise {

/*****************************************************************
 * Forward Declarations
 *****************************************************************/
template <typename T>
class RS_VectorView;

class RS_Bytes;

class RS_BytesView;

/*****************************************************************
 * Standard Container Traits
 *****************************************************************/

template <typename T>
struct RS_IsStdVector {
    static constexpr bool value = false;
};

template <typename T, typename Alloc>
struct RS_IsStdVector<std::vector<T, Alloc>> {
    static constexpr bool value = true;
};

template <typename T>
struct RS_IsStdArray {
    static constexpr bool value = false;
};

template <typename T, std::size_t N>
struct RS_IsStdArray<std::array<T, N>> {
    static constexpr bool value = true;
};

template <typename T>
struct RS_IsStdList {
    static constexpr bool value = false;
};

template <typename T, typename Alloc>
struct RS_IsStdList<std::list<T, Alloc>> {
    static constexpr bool value = true;
};

template <typename T>
struct RS_IsStdMap {
    static constexpr bool value = false;
};

template <typename Key, typename T, typename Compare, typename Alloc>
struct RS_IsStdMap<std::map<Key, T, Compare, Alloc>> {
    static constexpr bool value = true;
};

template <typename T>
struct RS_IsStdUnorderedMap {
    static constexpr bool value = false;
};

template <typename Key, typename T, typename Hash, typename KeyEqual, typename Alloc>
struct RS_IsStdUnorderedMap<std::unordered_map<Key, T, Hash, KeyEqual, Alloc>> {
    static constexpr bool value = true;
};

template <typename T>
struct RS_IsStdSet {
    static constexpr bool value = false;
};

template <typename T, typename Compare, typename Alloc>
struct RS_IsStdSet<std::set<T, Compare, Alloc>> {
    static constexpr bool value = true;
};

template <typename T>
struct RS_IsStdUnorderedSet {
    static constexpr bool value = false;
};

template <typename T, typename Hash, typename KeyEqual, typename Alloc>
struct RS_IsStdUnorderedSet<std::unordered_set<T, Hash, KeyEqual, Alloc>> {
    static constexpr bool value = true;
};

/*****************************************************************
 * RISE Container Traits
 *****************************************************************/

template <typename T>
struct RS_IsVectorView {
    static constexpr bool value = false;
};

template <typename T>
struct RS_IsVectorView<RS_VectorView<T>> {
    static constexpr bool value = true;
};

template <typename T>
struct RS_IsBytes {
    static constexpr bool value = false;
};

template <>
struct RS_IsBytes<RS_Bytes> {
    static constexpr bool value = true;
};

template <typename T>
struct RS_IsBytesView {
    static constexpr bool value = false;
};

template <>
struct RS_IsBytesView<RS_BytesView> {
    static constexpr bool value = true;
};

/*****************************************************************
 * Abstract Container Traits
 *****************************************************************/
/*
 * @brief Checks if a type is a linear like.
 *
 * @tparam T The type to check
 * @return True if the type is a linear like, false otherwise
 */
template <typename T>
struct RS_IsLinearLike {
    static constexpr bool value = RS_IsStdVector<T>::value || RS_IsStdList<T>::value ||
                                  RS_IsStdArray<T>::value || RS_IsVectorView<T>::value ||
                                  RS_IsBytes<T>::value || RS_IsBytesView<T>::value;
};

/*
 * @brief Checks if a type is a map like.
 *
 * @tparam T The type to check
 * @return True if the type is a map like, false otherwise
 */
template <typename T>
struct RS_IsMapLike {
    static constexpr bool value = RS_IsStdMap<T>::value || RS_IsStdUnorderedMap<T>::value;
};

/*
 * @brief Checks if a type is a set like.
 *
 * @tparam T The type to check
 * @return True if the type is a set like, false otherwise
 */
template <typename T>
struct RS_IsSetLike {
    static constexpr bool value = RS_IsStdSet<T>::value || RS_IsStdUnorderedSet<T>::value;
};

/*
 * @brief Checks if a type is a container like.
 *
 * @tparam T The type to check
 * @return True if the type is a container like, false otherwise
 */
template <typename T>
struct RS_IsContainerLike {
    static constexpr bool value = RS_IsLinearLike<T>::value || RS_IsMapLike<T>::value ||
                                  RS_IsSetLike<T>::value;
};

/*
 * @brief Checks if a type is a container view like.
 *
 * @tparam T The type to check
 * @return True if the type is a container view like, false otherwise
 */
template <typename T>
struct RS_IsContainerViewLike {
    static constexpr bool value = RS_IsVectorView<T>::value || RS_IsBytesView<T>::value;
};

}  // namespace rise

#endif  // RS_DATA_TRAITS_H
