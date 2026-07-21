// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VECTOR_VIEW_H
#define RS_VECTOR_VIEW_H

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <vector>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"

namespace rise {

/*****************************************************************
 * Class Definitions
 *****************************************************************/
/**
 * @brief A non-owning view over a contiguous sequence of elements
 *
 * RS_VectorView provides a read-only view into a vector or array without
 * owning the data.
 *
 * @tparam T The element type
 */
template <typename T>
class RS_VectorView {
   public:
    using value_type = T;
    using size_type = RSize;
    using difference_type = RSignedSize;
    using reference = const T&;
    using const_reference = const T&;
    using pointer = const T*;
    using const_pointer = const T*;
    using iterator = const T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /*****************************************************************
     * Data
     *****************************************************************/
    const T* data = nullptr;  ///< Pointer to the data
    RSize elem_num = 0;       ///< Number of elements in the view

    /*****************************************************************
     * Constructors
     *****************************************************************/

    /**
     * @brief Default constructor - creates an empty view
     */
    constexpr RS_VectorView() noexcept = default;

    /**
     * @brief Copy constructor
     */
    constexpr RS_VectorView(const RS_VectorView& other) noexcept = default;

    /**
     * @brief Assignment operator
     */
    constexpr RS_VectorView& operator=(const RS_VectorView& other) noexcept = default;

    /**
     * @brief Construct from pointer and size
     * @param data Pointer to the first element
     * @param size Number of elements
     */
    constexpr RS_VectorView(const T* data, RSize size) noexcept : data(data), elem_num(size) {}

    /**
     * @brief Construct from C-array
     */
    template <size_t N>
    constexpr RS_VectorView(const T (&arr)[N]) noexcept : data(arr), elem_num(N) {}

    /**
     * @brief Construct from iterator range
     */
    template <typename Iterator>
    constexpr RS_VectorView(Iterator first, Iterator last) noexcept
        : data(&(*first)), elem_num(std::distance(first, last)) {}

    /**
     * @brief Construct from std::vector
     */
    template <typename Allocator>
    constexpr RS_VectorView(const std::vector<T, Allocator>& vec) noexcept
        : data(vec.data()), elem_num(vec.size()) {}

    /*****************************************************************
     * Element Access
     *****************************************************************/

    /**
     * @brief Access element at specified position with bounds checking
     * @param index The position of the element to access
     * @return Reference to the element at the specified position
     * @throw RS_OutOfBoundsException if index >= size()
     */
    constexpr const T& at(RIndex index) const {
        if (index >= elem_num) {
            RS_THROW(RS_OutOfBoundsException("Index {} out of bounds (size={})", index, elem_num));
        }
        return data[index];
    }

    /**
     * @brief Access element at specified position without bounds checking
     * @param index The position of the element to access
     * @return Reference to the element at the specified position
     */
    constexpr const T& operator[](RIndex index) const { return data[index]; }

    /**
     * @brief Access the first element
     * @return Reference to the first element
     * @throw RS_OutOfBoundsException if the view is empty
     */
    constexpr const T& front() const {
        if (elem_num == 0) {
            RS_THROW(RS_OutOfBoundsException("Cannot access front of empty view"));
        }
        return data[0];
    }

    /**
     * @brief Access the last element
     * @return Reference to the last element
     * @throw RS_OutOfBoundsException if the view is empty
     */
    constexpr const T& back() const {
        if (elem_num == 0) {
            RS_THROW(RS_OutOfBoundsException("Cannot access back of empty view"));
        }
        return data[elem_num - 1];
    }

    /**
     * @brief Get pointer to the underlying array
     */
    constexpr const T* ptr() const noexcept { return data; }

    /*****************************************************************
     * Iterators
     *****************************************************************/

    /**
     * @brief Get iterator to the beginning
     */
    constexpr const_iterator begin() const noexcept { return data; }

    /**
     * @brief Get iterator to the beginning
     */
    constexpr const_iterator cbegin() const noexcept { return data; }

    /**
     * @brief Get iterator to the end
     */
    constexpr const_iterator end() const noexcept { return data + elem_num; }

    /**
     * @brief Get iterator to the end
     */
    constexpr const_iterator cend() const noexcept { return data + elem_num; }

    /**
     * @brief Get reverse iterator to the beginning
     */
    constexpr const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(end());
    }

    /**
     * @brief Get reverse iterator to the beginning
     */
    constexpr const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator(end());
    }

    /**
     * @brief Get reverse iterator to the end
     */
    constexpr const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(begin());
    }

    /**
     * @brief Get reverse iterator to the end
     */
    constexpr const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator(begin());
    }

    /*****************************************************************
     * Capacity
     *****************************************************************/

    /**
     * @brief Check if the view is empty
     */
    [[nodiscard]] constexpr bool empty() const noexcept { return elem_num == 0; }

    /**
     * @brief Get the current number of elements in the view
     */
    [[nodiscard]] constexpr size_type size() const noexcept { return elem_num; }

    /*****************************************************************
     * Operations
     *****************************************************************/

    /**
     * @brief Swap contents with another view
     */
    constexpr void swap(RS_VectorView& other) noexcept {
        std::swap(data, other.data);
        std::swap(elem_num, other.elem_num);
    }

    /**
     * @brief Get a sub-view from [begin, end)
     * @param begin Starting index
     * @param end Ending index (RS_NULL_INDEX means to the end)
     * @return A RS_VectorView of the sliced portion
     * @throw RS_OutOfBoundsException if begin > size()
     */
    [[nodiscard]] constexpr RS_VectorView
    sliceToView(RIndex begin, RIndex end = RS_NULL_INDEX) const {
        if (begin > elem_num) {
            RS_THROW(
                RS_OutOfBoundsException("Slice begin {} out of bounds (size={})", begin, elem_num)
            );
        }
        if (end == RS_NULL_INDEX) {
            end = elem_num;
        }
        if (end > elem_num) {
            end = elem_num;
        }
        if (begin > end) {
            RS_THROW(RS_OutOfBoundsException("Slice begin {} > end {}", begin, end));
        }
        return RS_VectorView(data + begin, end - begin);
    }

    /*****************************************************************
     * Comparison Operators (member functions)
     *****************************************************************/

    /**
     * @brief Equality comparison
     */
    constexpr bool operator==(const RS_VectorView& other) const noexcept {
        if (elem_num != other.elem_num) {
            return false;
        }
        return std::equal(begin(), end(), other.begin());
    }

    /**
     * @brief Inequality comparison
     */
    constexpr bool operator!=(const RS_VectorView& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Less-than comparison
     */
    constexpr bool operator<(const RS_VectorView& other) const noexcept {
        return std::lexicographical_compare(begin(), end(), other.begin(), other.end());
    }

    /**
     * @brief Greater-than comparison
     */
    constexpr bool operator>(const RS_VectorView& other) const noexcept { return other < *this; }

    /**
     * @brief Less-than-or-equal comparison
     */
    constexpr bool operator<=(const RS_VectorView& other) const noexcept {
        return !(other < *this);
    }

    /**
     * @brief Greater-than-or-equal comparison
     */
    constexpr bool operator>=(const RS_VectorView& other) const noexcept {
        return !(*this < other);
    }

    /// No generic serialization, handled by serializers
};

}  // namespace rise

#endif  // RS_VECTOR_VIEW_H
