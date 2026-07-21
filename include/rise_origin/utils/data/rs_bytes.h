// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BYTES_H
#define RS_BYTES_H

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <string_view>
#include <type_traits>
#include <utility>

#include "rise_origin/utils/data/rs_bytes_view.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"

namespace rise {

/*****************************************************************
 * Class Definitions
 *****************************************************************/
/**
 * @brief An owning, growable byte buffer with custom memory management
 *
 * RS_Bytes is the byte counterpart of RS_Vector. It stores a contiguous, mutable
 * sequence of RByte and, unlike RS_Vector, uses RBigSize (64-bit) for size, capacity,
 * and index, so it is guaranteed to hold more than 4 GB, whereas RSize is probably
 * limited to 32-bit, as it can be reconfigured. Because RByte is trivially copyable, all
 * grow/shift/copy operations use std::memcpy / std::memmove instead of per-element
 * construction.
 *
 * Key features:
 * - Custom memory allocation with RS_MemoryProvider integration
 * - 64-bit size, capacity and index (RBigSize / RBigIndex)
 * - Performance-optimized operations (resize without initialization, in-place writes)
 * - Comprehensive byte access with bounds checking
 * - Flexible slicing operations with view support
 * - Seamless conversion to RS_BytesView
 * - Concatenation operators for intuitive buffer composition
 * - Content hashing, so RS_Bytes is usable directly as a hash map or set key
 *
 * ## Usage Examples
 * ```cpp
 * // Basic construction
 * RS_Bytes bytes;                              // Empty buffer
 * RS_Bytes sized(10);                          // 10 uninitialized bytes
 * RS_Bytes filled(5, 0xFF);                    // 5 bytes, each 0xFF
 * RS_Bytes init{0x01, 0x02, 0x03};             // Initializer list construction
 *
 * // Construction with custom memory provider
 * auto* custom_provider = &MyCustomMemoryProvider::getInstance();
 * RS_Bytes custom(custom_provider);
 *
 * // Construction from a raw range
 * RS_Bytes from_ptr(source_ptr, source_size);
 *
 * // Construction from RS_BytesView
 * RS_BytesView view = other.toView();
 * RS_Bytes from_view(view);
 *
 * // Byte access and modification
 * bytes.pushBack(0x10);                        // Add byte to end
 * bytes[0] = 0x0F;                             // Direct access (no bounds check)
 * RByte safe = bytes.at(0);                    // Bounds-checked access
 *
 * // Appending raw bytes (primary serialization primitive)
 * bytes.append(view);                          // Append another buffer's bytes
 * bytes.append(source_ptr, source_size);       // Append a raw range
 *
 * // In-place random-access writes
 * bytes.writeSlice(src_ptr, num, offset);      // Overwrite [offset, offset + num)
 * bytes.readSlice(dst_ptr, num, offset);       // Read [offset, offset + num)
 *
 * // Performance-optimized resize (bytes not initialized)
 * bytes.resize(100);                           // Resize to 100, contents uninitialized
 * bytes.resizeAndFill(50, 0x00);               // Resize to 50, new bytes filled with 0x00
 *
 * // Slicing operations
 * RS_Bytes slice = bytes.slice(2, 5);          // New buffer with bytes [2, 5)
 * RS_BytesView sub = bytes.sliceToView(1, 4);  // Lightweight view of bytes [1, 4)
 *
 * // Type conversions
 * RS_BytesView view = bytes.toView();          // Get lightweight view
 * RS_BytesView implicit_view = bytes;          // Implicit conversion to view
 * ```
 *
 * ## Performance Considerations
 * - Use `resize()` instead of `resizeAndFill()` when you overwrite the bytes yourself
 * - Reserve capacity with `reserve()` when the final size is known to avoid reallocations
 * - Use `sliceToView()` instead of `slice()` when you don't need to own the data
 *
 * @see RS_BytesView for non-owning byte views
 * @see RS_Vector for the generic element container
 * @see RS_MemoryProvider for custom memory allocation strategies
 */
class RS_Bytes {
   private:
    /*****************************************************************
     * Iterator Classes
     *****************************************************************/

    class Iterator {
       private:
        RByte* data = nullptr;

       public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = RByte;
        using difference_type = RBigSignedSize;
        using reference = RByte&;
        using pointer = RByte*;

        Iterator() = default;
        explicit Iterator(RByte* p) : data(p) {}

        reference operator*() const { return *data; }
        pointer operator->() const { return data; }

        Iterator& operator++() {
            ++data;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++data;
            return tmp;
        }

        Iterator& operator--() {
            --data;
            return *this;
        }

        Iterator operator--(int) {
            Iterator tmp = *this;
            --data;
            return tmp;
        }

        Iterator& operator+=(difference_type n) {
            data += n;
            return *this;
        }

        Iterator& operator-=(difference_type n) {
            data -= n;
            return *this;
        }

        Iterator operator+(difference_type n) const { return Iterator(data + n); }

        Iterator operator-(difference_type n) const { return Iterator(data - n); }

        difference_type operator-(const Iterator& other) const { return data - other.data; }

        reference operator[](difference_type n) const { return data[n]; }

        friend bool operator==(const Iterator& lhs, const Iterator& rhs) {
            return lhs.data == rhs.data;
        }

        friend bool operator!=(const Iterator& lhs, const Iterator& rhs) {
            return lhs.data != rhs.data;
        }

        friend bool operator<(const Iterator& lhs, const Iterator& rhs) {
            return lhs.data < rhs.data;
        }

        friend bool operator<=(const Iterator& lhs, const Iterator& rhs) {
            return lhs.data <= rhs.data;
        }

        friend bool operator>(const Iterator& lhs, const Iterator& rhs) {
            return lhs.data > rhs.data;
        }

        friend bool operator>=(const Iterator& lhs, const Iterator& rhs) {
            return lhs.data >= rhs.data;
        }

        friend Iterator operator+(difference_type n, const Iterator& it) {
            return Iterator(it.data + n);
        }

        // Internal access for RS_Bytes
        RByte* ptr() const { return data; }

        friend class RS_Bytes;
        friend class ConstIterator;
    };

    class ConstIterator {
       private:
        const RByte* data = nullptr;

       public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = RByte;
        using difference_type = RBigSignedSize;
        using reference = const RByte&;
        using pointer = const RByte*;

        ConstIterator() = default;
        explicit ConstIterator(const RByte* p) : data(p) {}
        ConstIterator(const Iterator& other) : data(other.data) {}

        reference operator*() const { return *data; }
        pointer operator->() const { return data; }

        ConstIterator& operator++() {
            ++data;
            return *this;
        }

        ConstIterator operator++(int) {
            ConstIterator tmp = *this;
            ++data;
            return tmp;
        }

        ConstIterator& operator--() {
            --data;
            return *this;
        }

        ConstIterator operator--(int) {
            ConstIterator tmp = *this;
            --data;
            return tmp;
        }

        ConstIterator& operator+=(difference_type n) {
            data += n;
            return *this;
        }

        ConstIterator& operator-=(difference_type n) {
            data -= n;
            return *this;
        }

        ConstIterator operator+(difference_type n) const { return ConstIterator(data + n); }

        ConstIterator operator-(difference_type n) const { return ConstIterator(data - n); }

        difference_type operator-(const ConstIterator& other) const { return data - other.data; }

        reference operator[](difference_type n) const { return data[n]; }

        friend bool operator==(const ConstIterator& lhs, const ConstIterator& rhs) {
            return lhs.data == rhs.data;
        }

        friend bool operator!=(const ConstIterator& lhs, const ConstIterator& rhs) {
            return lhs.data != rhs.data;
        }

        friend bool operator<(const ConstIterator& lhs, const ConstIterator& rhs) {
            return lhs.data < rhs.data;
        }

        friend bool operator<=(const ConstIterator& lhs, const ConstIterator& rhs) {
            return lhs.data <= rhs.data;
        }

        friend bool operator>(const ConstIterator& lhs, const ConstIterator& rhs) {
            return lhs.data > rhs.data;
        }

        friend bool operator>=(const ConstIterator& lhs, const ConstIterator& rhs) {
            return lhs.data >= rhs.data;
        }

        friend ConstIterator operator+(difference_type n, const ConstIterator& it) {
            return ConstIterator(it.data + n);
        }

        // Internal access for RS_Bytes
        const RByte* ptr() const { return data; }

        friend class RS_Bytes;
    };

   public:
    /*****************************************************************
     * Type aliases
     *****************************************************************/
    using value_type = RByte;
    using allocator_type = RS_MemoryAllocator<RByte>;
    using size_type = RBigSize;
    using difference_type = RBigSignedSize;
    using reference = RByte&;
    using const_reference = const RByte&;
    using pointer = RByte*;
    using const_pointer = const RByte*;
    using iterator = Iterator;
    using const_iterator = ConstIterator;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /*****************************************************************
     * Data
     *****************************************************************/
    RByte* data = nullptr;      ///< Pointer to the raw data array
    RBigSize max_elem_num = 0;  ///< Total number of allocated bytes
    RBigSize elem_num = 0;      ///< Number of bytes actually in use
    allocator_type allocator;   ///< Allocator instance

    /*****************************************************************
     * Constructors And Destructor
     *****************************************************************/

    /**
     * @brief Default constructor
     */
    RS_Bytes(RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance())
        : allocator(provider) {}

    /**
     * @brief Construct with count uninitialized bytes
     */
    RS_Bytes(RBigSize count, RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance())
        : allocator(provider) {
        _allocate(count);
        elem_num = count;
    }

    /**
     * @brief Construct with count copies of value
     */
    RS_Bytes(
        RBigSize count,
        RByte value,
        RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance()
    )
        : allocator(provider) {
        _allocate(count);
        if (count > 0) {
            std::memset(data, value, count);
        }
        elem_num = count;
    }

    /**
     * @brief Construct from a raw byte range
     */
    RS_Bytes(
        const RByte* source,
        RBigSize count,
        RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance()
    )
        : allocator(provider) {
        _allocateAndCopyFrom(source, count);
        elem_num = count;
    }

    /**
     * @brief Construct from a raw character range (text bytes)
     */
    RS_Bytes(
        const char* source,
        RBigSize count,
        RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance()
    )
        : RS_Bytes(reinterpret_cast<const RByte*>(source), count, provider) {}

    /**
     * @brief Construct from iterator range
     */
    template <
        typename InputIt,
        typename = std::enable_if_t<std::is_base_of_v<
            std::input_iterator_tag,
            typename std::iterator_traits<InputIt>::iterator_category>>>
    RS_Bytes(
        InputIt first,
        InputIt last,
        RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance()
    )
        : allocator(provider) {
        _constructFromRange(first, last);
    }

    /**
     * @brief Construct from initializer list
     */
    RS_Bytes(
        std::initializer_list<RByte> init,
        RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance()
    )
        : RS_Bytes(init.begin(), init.end(), provider) {}

    /**
     * @brief Construct from RS_BytesView
     */
    RS_Bytes(
        const RS_BytesView& view,
        RS_MemoryProvider* provider = &RS_MemoryDefaultProvider::getInstance()
    )
        : allocator(provider) {
        if (view.size() > 0) {
            _allocateAndCopyFrom(view.ptr(), view.size());
            elem_num = view.size();
        }
    }

    /**
     * @brief Copy constructor
     */
    RS_Bytes(const RS_Bytes& other) : allocator(other.allocator) {
        if (other.elem_num > 0) {
            _allocateAndCopyFrom(other.data, other.elem_num);
            elem_num = other.elem_num;
        }
    }

    /**
     * @brief Move constructor
     */
    RS_Bytes(RS_Bytes&& other) noexcept : allocator(std::move(other.allocator)) {
        data = other.data;
        max_elem_num = other.max_elem_num;
        elem_num = other.elem_num;

        other.data = nullptr;
        other.max_elem_num = 0;
        other.elem_num = 0;
    }

    /**
     * @brief Destructor
     */
    ~RS_Bytes() { _cleanup(); }

    /*****************************************************************
     * Assignment operators
     *****************************************************************/

    /**
     * @brief Copy assignment operator
     */
    RS_Bytes& operator=(const RS_Bytes& other) {
        if (this != &other) {
            _assignCopy(other.data, other.elem_num);
        }
        return *this;
    }

    /**
     * @brief Move assignment operator
     */
    RS_Bytes& operator=(RS_Bytes&& other) noexcept {
        if (this != &other) {
            _cleanup();
            allocator = std::move(other.allocator);
            data = other.data;
            max_elem_num = other.max_elem_num;
            elem_num = other.elem_num;

            other.data = nullptr;
            other.max_elem_num = 0;
            other.elem_num = 0;
        }
        return *this;
    }

    /*****************************************************************
     * Element access
     *****************************************************************/

    /**
     * @brief Access byte with bounds checking
     */
    reference at(RBigIndex index) {
        if (index >= elem_num) {
            RS_THROW(RS_OutOfBoundsException("Index {} out of bounds (size={})", index, elem_num));
        }
        return data[index];
    }

    /**
     * @brief Const access byte with bounds checking
     */
    const_reference at(RBigIndex index) const {
        if (index >= elem_num) {
            RS_THROW(RS_OutOfBoundsException("Index {} out of bounds (size={})", index, elem_num));
        }
        return data[index];
    }

    /**
     * @brief Access byte without bounds checking
     */
    reference operator[](RBigIndex index) { return data[index]; }

    /**
     * @brief Const access byte without bounds checking
     */
    const_reference operator[](RBigIndex index) const { return data[index]; }

    /**
     * @brief Access byte at specified index
     */
    reference get(RBigIndex index) { return at(index); }

    /**
     * @brief Access byte at specified index
     */
    const_reference get(RBigIndex index) const { return at(index); }

    /**
     * @brief Set byte at specified index
     * @note This method check the index bounds.
     */
    void set(RBigIndex index, RByte value) { at(index) = value; }

    /**
     * @brief Access first byte
     */
    reference front() {
        if (elem_num == 0) {
            RS_THROW(RS_OutOfBoundsException("Cannot access front of empty bytes"));
        }
        return data[0];
    }

    const_reference front() const {
        if (elem_num == 0) {
            RS_THROW(RS_OutOfBoundsException("Cannot access front of empty bytes"));
        }
        return data[0];
    }

    /**
     * @brief Access last byte
     */
    reference back() {
        if (elem_num == 0) {
            RS_THROW(RS_OutOfBoundsException("Cannot access back of empty bytes"));
        }
        return data[elem_num - 1];
    }

    const_reference back() const {
        if (elem_num == 0) {
            RS_THROW(RS_OutOfBoundsException("Cannot access back of empty bytes"));
        }
        return data[elem_num - 1];
    }

    /**
     * @brief Direct access to underlying array
     */
    RByte* ptr() noexcept { return data; }

    const RByte* ptr() const noexcept { return data; }

    /*****************************************************************
     * Iterators
     *****************************************************************/

    iterator begin() noexcept { return iterator(data); }
    const_iterator begin() const noexcept { return const_iterator(data); }
    const_iterator cbegin() const noexcept { return const_iterator(data); }

    iterator end() noexcept { return iterator(data + elem_num); }
    const_iterator end() const noexcept { return const_iterator(data + elem_num); }
    const_iterator cend() const noexcept { return const_iterator(data + elem_num); }

    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

    /*****************************************************************
     * Capacity
     *****************************************************************/

    /**
     * @brief Check if buffer is empty
     */
    [[nodiscard]] bool empty() const noexcept { return elem_num == 0; }

    /**
     * @brief Get number of bytes
     */
    [[nodiscard]] RBigSize size() const noexcept { return elem_num; }

    /**
     * @brief Get maximum possible number of bytes
     */
    [[nodiscard]] RBigSize maxSize() const noexcept {
        return std::allocator_traits<allocator_type>::max_size(allocator);
    }

    /**
     * @brief Get current capacity
     */
    [[nodiscard]] RBigSize capacity() const noexcept { return max_elem_num; }

    /*****************************************************************
     * Modifiers
     *****************************************************************/

    /**
     * @brief Clear all bytes (keeps capacity)
     */
    void clear() noexcept { elem_num = 0; }

    /**
     * @brief Reserve space for at least count bytes
     * @param count New capacity
     */
    void reserve(RBigSize count) {
        if (count > max_elem_num) {
            _reallocate(count);
        }
    }

    /**
     * @brief Resize to contain count bytes without initializing new bytes
     *
     * This is a performance-optimized resize that:
     * - If count > capacity: reallocates and optionally copies existing bytes
     * - If count > size: adjusts size without initializing new bytes (user must fill them)
     * - If count < size: shrinks the size
     *
     * @warning This method does not initialize new bytes, so user must fill them manually.
     * @param count New size
     * @param is_copied If true, existing bytes are copied to new allocation when reallocating.
     *                  If false, old bytes are dropped and new allocation is uninitialized.
     */
    void resize(RBigSize count, bool is_copied = true) {
        if (count > max_elem_num) {
            if (is_copied) {
                _reallocate(count);
            } else {
                _cleanup();
                _allocate(count);
            }
        }
        elem_num = count;
    }

    /**
     * @brief Resize to contain count bytes, filling any new bytes with value
     *
     * - If count > size, new bytes are filled with value
     * - If count < size, the buffer shrinks
     * - If count > capacity, reallocates and optionally preserves existing bytes
     *
     * @param count New size
     * @param value Value to fill new bytes with (default 0)
     * @param is_copied If true, existing bytes are copied to new allocation when reallocating.
     *                  If false, old bytes are dropped before filling all bytes.
     */
    void resizeAndFill(RBigSize count, RByte value = 0, bool is_copied = true) {
        if (count > elem_num) {
            if (count > max_elem_num) {
                if (is_copied) {
                    _reallocate(count);
                } else {
                    _cleanup();
                    _allocate(count);
                }
            }
            // Fill new bytes (from elem_num to count, or from 0 if !is_copied)
            RBigSize start = is_copied ? elem_num : 0;
            std::memset(data + start, value, count - start);
        }
        elem_num = count;
    }

    /**
     * @brief Reduce capacity to match the current size
     */
    void shrinkToFit() {
        if (max_elem_num > elem_num) {
            if (elem_num == 0) {
                _cleanup();
                data = nullptr;
                max_elem_num = 0;
            } else {
                _reallocate(elem_num);
            }
        }
    }

    /**
     * @brief Swap contents with another buffer
     */
    void swap(RS_Bytes& other) noexcept {
        std::swap(data, other.data);
        std::swap(max_elem_num, other.max_elem_num);
        std::swap(elem_num, other.elem_num);
    }

    /**
     * @brief Insert a byte at a given index
     */
    void insert(RBigIndex index, RByte value) {
        if (index > elem_num) {
            RS_THROW(RS_OutOfBoundsException("Index {} out of bounds (size={})", index, elem_num));
        }

        RBigSize new_size = elem_num + 1;
        if (new_size > max_elem_num) {
            RBigSize new_capacity = _calculateGrowth(new_size);
            RByte* new_data = _rawAllocate(new_capacity);

            std::memcpy(new_data, data, index);
            new_data[index] = value;
            std::memcpy(new_data + index + 1, data + index, elem_num - index);

            _cleanup();
            data = new_data;
            max_elem_num = new_capacity;
        } else {
            // Shift bytes right
            std::memmove(data + index + 1, data + index, elem_num - index);
            data[index] = value;
        }
        elem_num = new_size;
    }

    /**
     * @brief Insert a byte at a given const iterator position
     */
    void insert(const_iterator pos, RByte value) { insert(pos.ptr() - data, value); }

    /**
     * @brief Insert a byte range at a given index
     */
    void insertRange(RBigIndex index, const RS_BytesView& other) {
        if (index > elem_num) {
            RS_THROW(RS_OutOfBoundsException("Index {} out of bounds (size={})", index, elem_num));
        }

        RBigSize count = other.size();
        if (count == 0) {
            return;
        }

        RBigSize new_size = elem_num + count;
        if (new_size > max_elem_num) {
            RBigSize new_capacity = _calculateGrowth(new_size);
            RByte* new_data = _rawAllocate(new_capacity);

            std::memcpy(new_data, data, index);
            std::memcpy(new_data + index, other.ptr(), count);
            std::memcpy(new_data + index + count, data + index, elem_num - index);

            _cleanup();
            data = new_data;
            max_elem_num = new_capacity;
        } else {
            // Shift bytes right, then copy in the new bytes
            std::memmove(data + index + count, data + index, elem_num - index);
            std::memcpy(data + index, other.ptr(), count);
        }
        elem_num = new_size;
    }

    /**
     * @brief Erase byte at index
     */
    void erase(RBigIndex index) { eraseRange(index, index + 1); }

    /**
     * @brief Erase byte at a given iterator position
     */
    void erase(const_iterator pos) { eraseRange(pos.ptr() - data, pos.ptr() - data + 1); }

    /**
     * @brief Erase bytes in range [begin, end)
     */
    void eraseRange(RBigIndex begin, RBigIndex end) {
        if (end > elem_num) {
            RS_THROW(RS_OutOfBoundsException("End {} exceeds size {}", end, elem_num));
        }
        if (begin >= end) {
            return;
        }

        RBigSize count = end - begin;
        if (end < elem_num) {
            std::memmove(data + begin, data + end, elem_num - end);
        }
        elem_num -= count;
    }

    /**
     * @brief Erase bytes in range [begin, end)
     */
    void eraseRange(const_iterator begin, const_iterator end) {
        eraseRange(begin.ptr() - data, end.ptr() - data);
    }

    /**
     * @brief Add byte to the end
     */
    void pushBack(RByte value) {
        if (elem_num == max_elem_num) {
            _reallocate(_calculateGrowth(elem_num + 1));
        }
        data[elem_num] = value;
        ++elem_num;
    }

    /**
     * @brief Remove last byte
     */
    void popBack() {
        if (elem_num == 0) {
            RS_THROW(RS_OutOfBoundsException("Cannot pop from empty bytes"));
        }
        --elem_num;
    }

    /**
     * @brief Append a raw byte range to the end of this buffer
     * @param source Pointer to the first byte to append
     * @param count Number of bytes to append
     * @return Reference to this buffer
     */
    RS_Bytes& append(const RByte* source, RBigSize count) {
        if (count == 0) {
            return *this;
        }

        RBigSize new_size = elem_num + count;
        if (new_size > max_elem_num) {
            _reallocate(_calculateGrowth(new_size));
        }
        std::memcpy(data + elem_num, source, count);
        elem_num = new_size;
        return *this;
    }

    /**
     * @brief Append a byte buffer to the end of this buffer
     * @param other The bytes to append
     * @return Reference to this buffer
     */
    RS_Bytes& append(const RS_BytesView& other) { return append(other.ptr(), other.size()); }

    /**
     * @brief Append a single byte to the end of this buffer
     * @param value The byte to append
     * @return Reference to this buffer
     */
    RS_Bytes& append(RByte value) {
        pushBack(value);
        return *this;
    }

    /**
     * @brief Overwrite bytes in place at a given offset
     * @param input Pointer to the first byte to write
     * @param num Number of bytes to write
     * @param offset Position at which writing starts
     * @throw RS_OutOfBoundsException if offset + num > size()
     */
    void writeSlice(const RByte* input, RBigSize num, RBigSize offset) {
        if (offset + num > elem_num) {
            RS_THROW(RS_OutOfBoundsException(
                "writeSlice [{}, {}) exceeds size {}", offset, offset + num, elem_num
            ));
        }
        if (num > 0) {
            std::memcpy(data + offset, input, num);
        }
    }

    /**
     * @brief Read bytes in place from a given offset
     * @param output Pointer to the destination buffer
     * @param num Number of bytes to read
     * @param offset Position at which reading starts
     * @throw RS_OutOfBoundsException if offset + num > size()
     */
    void readSlice(RByte* output, RBigSize num, RBigSize offset) const {
        if (offset + num > elem_num) {
            RS_THROW(RS_OutOfBoundsException(
                "readSlice [{}, {}) exceeds size {}", offset, offset + num, elem_num
            ));
        }
        if (num > 0) {
            std::memcpy(output, data + offset, num);
        }
    }

    /**
     * @brief Fill all bytes with a value
     */
    void fill(RByte value) {
        if (elem_num > 0) {
            std::memset(data, value, elem_num);
        }
    }

    /**
     * @brief Concatenate a single byte to this buffer (operator+=)
     */
    RS_Bytes& operator+=(RByte value) { return append(value); }

    /**
     * @brief Concatenate a byte buffer to this buffer (operator+=)
     */
    RS_Bytes& operator+=(const RS_BytesView& other) { return append(other); }

    /**
     * @brief Concatenate two buffers (creates a new buffer)
     */
    RS_Bytes operator+(const RS_BytesView& rhs) const {
        RS_Bytes result = *this;
        result += rhs;
        return result;
    }

    /**
     * @brief Concatenate a buffer and a single byte (creates a new buffer)
     */
    RS_Bytes operator+(RByte rhs) const {
        RS_Bytes result = *this;
        result += rhs;
        return result;
    }

    /*****************************************************************
     * Slicing operations
     *****************************************************************/

    /**
     * @brief Slice buffer from [begin, end) and create a new buffer
     * @param begin Starting position
     * @param end Ending position (RS_NULL_BIG_INDEX means to the end)
     * @return A new RS_Bytes containing the sliced bytes
     * @throw RS_OutOfBoundsException if begin > size()
     */
    [[nodiscard]] RS_Bytes slice(RBigIndex begin, RBigIndex end = RS_NULL_BIG_INDEX) const {
        if (begin > elem_num) {
            RS_THROW(RS_OutOfBoundsException("Slice begin out of bounds"));
        }
        if (end == RS_NULL_BIG_INDEX) {
            end = elem_num;
        }
        if (end > elem_num) {
            end = elem_num;
        }

        RS_Bytes result(allocator.provider());
        RBigSize slice_size = (end > begin) ? (end - begin) : 0;
        if (slice_size > 0) {
            result._allocateAndCopyFrom(data + begin, slice_size);
            result.elem_num = slice_size;
        }
        return result;
    }

    /**
     * @brief Slice buffer from [begin, end) and get a view
     * @param begin Starting position
     * @param end Ending position (RS_NULL_BIG_INDEX means to the end)
     * @return A RS_BytesView of the sliced portion
     * @throw RS_OutOfBoundsException if begin > size()
     */
    [[nodiscard]] RS_BytesView
    sliceToView(RBigIndex begin, RBigIndex end = RS_NULL_BIG_INDEX) const {
        if (begin > elem_num) {
            RS_THROW(RS_OutOfBoundsException("Slice begin out of bounds"));
        }
        if (end == RS_NULL_BIG_INDEX) {
            end = elem_num;
        }
        if (end > elem_num) {
            end = elem_num;
        }
        return RS_BytesView(data, elem_num).sliceToView(begin, end);
    }

    /*****************************************************************
     * Conversion
     *****************************************************************/
    /**
     * @brief Get a view of this buffer
     * @return RS_BytesView pointing to this buffer's data
     */
    RS_BytesView toView() const { return RS_BytesView(data, elem_num); }

    /**
     * @brief Implicit conversion to RS_BytesView
     */
    operator RS_BytesView() const { return toView(); }

    /*****************************************************************
     * Comparison Operators
     *****************************************************************/

    /**
     * @brief Equality comparison - checks if two buffers have the same size and bytes
     */
    bool operator==(const RS_Bytes& rhs) const {
        return elem_num == rhs.elem_num && std::equal(begin(), end(), rhs.begin());
    }

    /**
     * @brief Inequality comparison - checks if two buffers differ
     */
    bool operator!=(const RS_Bytes& rhs) const { return !(*this == rhs); }

    /**
     * @brief Less-than comparison - lexicographical ordering
     */
    bool operator<(const RS_Bytes& rhs) const {
        return std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
    }

    /**
     * @brief Greater-than comparison - lexicographical ordering
     */
    bool operator>(const RS_Bytes& rhs) const { return rhs < *this; }

    /**
     * @brief Less-than-or-equal comparison - lexicographical ordering
     */
    bool operator<=(const RS_Bytes& rhs) const { return !(rhs < *this); }

    /**
     * @brief Greater-than-or-equal comparison - lexicographical ordering
     */
    bool operator>=(const RS_Bytes& rhs) const { return !(*this < rhs); }

    /*****************************************************************
     * Provider
     *****************************************************************/

    /**
     * @brief Get memory provider
     */
    [[nodiscard]] RS_MemoryProvider* provider() const noexcept { return allocator.provider(); }

    /*****************************************************************
     * Special Methods
     *****************************************************************/
    /// No generic serialization, handled by serializers

   private:
    /*****************************************************************
     * Private helper methods
     *****************************************************************/

    /**
     * @brief Calculate growth capacity
     */
    [[nodiscard]] RBigSize _calculateGrowth(RBigSize min_capacity) const {
        const RBigSize geometric_growth = max_elem_num * 2;
        if (geometric_growth < min_capacity) {
            return min_capacity;
        }
        return geometric_growth;
    }

    /**
     * @brief Allocate raw memory without adjusting size
     */
    RByte* _rawAllocate(RBigSize count) {
        return std::allocator_traits<allocator_type>::allocate(allocator, count);
    }

    /**
     * @brief Allocate raw memory for count bytes
     */
    void _allocate(RBigSize count) {
        if (count == 0) {
            data = nullptr;
            max_elem_num = 0;
            return;
        }

        data = _rawAllocate(count);
        max_elem_num = count;
    }

    /**
     * @brief Deallocate raw memory
     */
    void _deallocate() {
        if (data != nullptr) {
            std::allocator_traits<allocator_type>::deallocate(allocator, data, max_elem_num);
            data = nullptr;
            max_elem_num = 0;
        }
    }

    /**
     * @brief Deallocate and reset size
     */
    void _cleanup() {
        if (data != nullptr) {
            _deallocate();
            elem_num = 0;
        }
    }

    /**
     * @brief Reallocate with new capacity (preserves existing bytes)
     */
    void _reallocate(RBigSize new_capacity) {
        RByte* new_data = _rawAllocate(new_capacity);
        if (elem_num > 0) {
            std::memcpy(new_data, data, elem_num);
        }
        _deallocate();
        data = new_data;
        max_elem_num = new_capacity;
    }

    /**
     * @brief Allocate and copy from source
     */
    void _allocateAndCopyFrom(const RByte* source, RBigSize count) {
        _allocate(count);
        if (count > 0) {
            std::memcpy(data, source, count);
        }
    }

    /**
     * @brief Assign by copying from source
     */
    void _assignCopy(const RByte* source, RBigSize count) {
        if (count > max_elem_num) {
            _cleanup();
            _allocateAndCopyFrom(source, count);
        } else if (count > 0) {
            std::memcpy(data, source, count);
        }
        elem_num = count;
    }

    /**
     * @brief Construct from iterator range (used in constructors)
     */
    template <typename InputIt>
    void _constructFromRange(InputIt first, InputIt last) {
        // For random access iterators, we can reserve upfront
        if constexpr (std::is_base_of_v<
                          std::random_access_iterator_tag,
                          typename std::iterator_traits<InputIt>::iterator_category>) {
            RBigSize count = std::distance(first, last);
            _allocate(count);
            RBigSize i = 0;
            for (auto it = first; it != last; ++it, ++i) {
                data[i] = *it;
            }
            elem_num = count;
        } else {
            // For input iterators, we need to grow dynamically
            for (auto it = first; it != last; ++it) {
                pushBack(*it);
            }
        }
    }
};

}  // namespace rise

#endif  // RS_BYTES_H
