// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BYTES_VIEW_H
#define RS_BYTES_VIEW_H

#include <algorithm>
#include <iterator>
#include <string_view>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"

namespace rise {

/*****************************************************************
 * Class Definitions
 *****************************************************************/
/**
 * @brief A non-owning view over a contiguous sequence of bytes
 *
 * RS_BytesView provides a read-only view into a byte buffer or array without
 * owning the data. It is the byte counterpart of RS_VectorView, but uses RBigSize
 * (64-bit) for size and index, so it is guaranteed to address more than 4 GB, whereas
 * RSize is probably limited to 32-bit, as it can be reconfigured.
 *
 * @see RS_Bytes for the owning byte buffer
 * @see RS_VectorView for the generic element view
 */
class RS_BytesView {
   public:
    using value_type = RByte;
    using size_type = RBigSize;
    using difference_type = RBigSignedSize;
    using reference = const RByte&;
    using const_reference = const RByte&;
    using pointer = const RByte*;
    using const_pointer = const RByte*;
    using iterator = const RByte*;
    using const_iterator = const RByte*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /*****************************************************************
     * Data
     *****************************************************************/
    const RByte* data = nullptr;  ///< Pointer to the data
    RBigSize elem_num = 0;        ///< Number of bytes in the view

    /*****************************************************************
     * Constructors
     *****************************************************************/

    /**
     * @brief Default constructor - creates an empty view
     */
    constexpr RS_BytesView() noexcept = default;

    /**
     * @brief Copy constructor
     */
    constexpr RS_BytesView(const RS_BytesView& other) noexcept = default;

    /**
     * @brief Assignment operator
     */
    constexpr RS_BytesView& operator=(const RS_BytesView& other) noexcept = default;

    /**
     * @brief Construct from pointer and size
     * @param data Pointer to the first byte
     * @param size Number of bytes
     */
    constexpr RS_BytesView(const RByte* data, RBigSize size) noexcept
        : data(data), elem_num(size) {}

    /**
     * @brief Construct from a character pointer and size (text bytes)
     * @param data Pointer to the first character
     * @param size Number of bytes
     */
    RS_BytesView(const char* data, RBigSize size) noexcept
        : data(reinterpret_cast<const RByte*>(data)), elem_num(size) {}

    /**
     * @brief Construct from C-array
     */
    template <size_t N>
    constexpr RS_BytesView(const RByte (&arr)[N]) noexcept : data(arr), elem_num(N) {}

    /**
     * @brief Construct from iterator range
     */
    template <typename Iterator>
    constexpr RS_BytesView(Iterator first, Iterator last) noexcept
        : data(&(*first)), elem_num(std::distance(first, last)) {}

    /*****************************************************************
     * Element Access
     *****************************************************************/

    /**
     * @brief Access byte at specified position with bounds checking
     * @param index The position of the byte to access
     * @return Reference to the byte at the specified position
     * @throw RS_OutOfBoundsException if index >= size()
     */
    constexpr const RByte& at(RBigIndex index) const {
        if (index >= elem_num) {
            RS_THROW(RS_OutOfBoundsException("Index {} out of bounds (size={})", index, elem_num));
        }
        return data[index];
    }

    /**
     * @brief Access byte at specified position without bounds checking
     * @param index The position of the byte to access
     * @return Reference to the byte at the specified position
     */
    constexpr const RByte& operator[](RBigIndex index) const { return data[index]; }

    /**
     * @brief Access the first byte
     * @return Reference to the first byte
     * @throw RS_OutOfBoundsException if the view is empty
     */
    constexpr const RByte& front() const {
        if (elem_num == 0) {
            RS_THROW(RS_OutOfBoundsException("Cannot access front of empty view"));
        }
        return data[0];
    }

    /**
     * @brief Access the last byte
     * @return Reference to the last byte
     * @throw RS_OutOfBoundsException if the view is empty
     */
    constexpr const RByte& back() const {
        if (elem_num == 0) {
            RS_THROW(RS_OutOfBoundsException("Cannot access back of empty view"));
        }
        return data[elem_num - 1];
    }

    /**
     * @brief Get pointer to the underlying array
     */
    constexpr const RByte* ptr() const noexcept { return data; }

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
     * @brief Get the current number of bytes in the view
     */
    [[nodiscard]] constexpr size_type size() const noexcept { return elem_num; }

    /*****************************************************************
     * Operations
     *****************************************************************/

    /**
     * @brief Swap contents with another view
     */
    void swap(RS_BytesView& other) noexcept {
        std::swap(data, other.data);
        std::swap(elem_num, other.elem_num);
    }

    /**
     * @brief Get a sub-view from [begin, end)
     * @param begin Starting index
     * @param end Ending index (RS_NULL_BIG_INDEX means to the end)
     * @return A RS_BytesView of the sliced portion
     * @throw RS_OutOfBoundsException if begin > size()
     */
    [[nodiscard]] constexpr RS_BytesView
    sliceToView(RBigIndex begin, RBigIndex end = RS_NULL_BIG_INDEX) const {
        if (begin > elem_num) {
            RS_THROW(
                RS_OutOfBoundsException("Slice begin {} out of bounds (size={})", begin, elem_num)
            );
        }
        if (end == RS_NULL_BIG_INDEX) {
            end = elem_num;
        }
        if (end > elem_num) {
            end = elem_num;
        }
        if (begin > end) {
            RS_THROW(RS_OutOfBoundsException("Slice begin {} > end {}", begin, end));
        }
        return RS_BytesView(data + begin, end - begin);
    }

    /*****************************************************************
     * Comparison Operators (member functions)
     *****************************************************************/

    /**
     * @brief Equality comparison
     */
    bool operator==(const RS_BytesView& other) const noexcept {
        if (elem_num != other.elem_num) {
            return false;
        }
        return std::equal(begin(), end(), other.begin());
    }

    /**
     * @brief Inequality comparison
     */
    bool operator!=(const RS_BytesView& other) const noexcept { return !(*this == other); }

    /**
     * @brief Less-than comparison
     */
    bool operator<(const RS_BytesView& other) const noexcept {
        return std::lexicographical_compare(begin(), end(), other.begin(), other.end());
    }

    /**
     * @brief Greater-than comparison
     */
    bool operator>(const RS_BytesView& other) const noexcept { return other < *this; }

    /**
     * @brief Less-than-or-equal comparison
     */
    bool operator<=(const RS_BytesView& other) const noexcept { return !(other < *this); }

    /**
     * @brief Greater-than-or-equal comparison
     */
    bool operator>=(const RS_BytesView& other) const noexcept { return !(*this < other); }

    /*****************************************************************
     * Special Methods
     *****************************************************************/
    /// No generic serialization, handled by serializers
};

}  // namespace rise

#endif  // RS_BYTES_VIEW_H
