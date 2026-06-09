// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BITMASK_H
#define RS_BITMASK_H

#include <cassert>

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/device/rs_dev_util.h"
#include "rs_data_util.h"

namespace rise {

namespace device {

class RSCF_BitmaskFill : public RS_ComputeFunction<RSCF_BitmaskFill> {
   public:
    template <typename Bitmask, typename WordType>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Bitmask mask, RSize word_num, WordType full_word, WordType last_word) {
        RIndex idx = index1D();
        auto word_ptr = static_cast<WordType*>(mask.data.ptr());
        if (idx < word_num - 1) word_ptr[idx] = full_word;
        if (idx == word_num - 1) word_ptr[idx] = last_word;
    }
};

template <typename WordType = uint32_t>
class RS_BitMask {
   public:
    static_assert(
        std::is_same_v<WordType, uint32_t> || std::is_same_v<WordType, uint64_t>,
        "WordType must be uint32_t or uint64_t"
    );
    static constexpr RSize WordBits = std::is_same_v<WordType, uint32_t> ? 32 : 64;
    /// The number of bits to shift for the bit index when getting the target word
    static constexpr RSize WordShift = std::is_same_v<WordType, uint32_t> ? 5 : 6;
    /// The bit mask for the target bit
    static constexpr RSize BitMask = WordBits - 1;
    static constexpr WordType FullWordMask =
        std::is_same_v<WordType, uint32_t> ? 0xFFFFFFFF : 0xFFFFFFFFFFFFFFFF;
    /*****************************************************************
     * Data
     *****************************************************************/
    RS_MemoryPtr data;
    RSize elem_num = 0;
    RSize word_num = 0;  // Number of words

    /*****************************************************************
     * Methods
     *****************************************************************/
    /// @brief Default constructor.
    RS_BitMask() = default;

    /// @brief Constructs an empty bitmask with the given memory provider.
    explicit RS_BitMask(RS_MemoryProvider* provider) { init(provider); }

    /// @brief Constructs a bitmask with the given number of bits and memory provider.
    RS_BitMask(RSize num_bits, RS_MemoryProvider* provider) { init(num_bits, provider); }

    /// @brief Copy constructor.
    RS_BitMask(const RS_BitMask& other) = default;

    /// @brief Assignment operator.
    RS_BitMask& operator=(const RS_BitMask& other) = default;

    /// @brief Initialized an empty bitmask with the given memory provider.
    void init(RS_MemoryProvider* provider) {
        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        word_num = 0;
    }

    /// @brief Initializes the bitmask with the given number of bits and memory provider.
    void init(RSize num_bits, RS_MemoryProvider* provider) {
        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        word_num = 0;
        resize(num_bits, false);
        fill(false);
    }

    /// @brief Frees the data associated with the bitmask.
    void free() {
        data.free();
        elem_num = 0;
        word_num = 0;
    }

    /// @brief Resizes the bitmask to the new number of bits.
    /// @param new_num_bits The new number of bits.
    /// @param copy If true, copy data from the old bitmask to the new one.
    /// @param new_is_on If true, set all new bits to on, if not, set all new bits to off.
    void resize(RSize new_num_bits, bool copy = true, bool new_is_on = false) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        auto old_data = data;
        RSize old_word_num = word_num;

        elem_num = new_num_bits;
        word_num = (new_num_bits + WordBits - 1) / WordBits;

        // Check if the new size is the same as the old size,
        // we can safely ignore remaining bits at tail
        if (word_num == old_word_num) return;

        if (word_num == 0) {
            data = RS_MemoryPtr(nullptr, 0, 0, old_data.device(), old_data.provider());
        } else {
            auto new_data = old_data.provider()->allocate(sizeof(WordType) * word_num, true);

            new_data.set(new_is_on ? 0xFF : 0, sizeof(WordType) * word_num);
            if (copy && old_word_num != 0) {
                RSize min_word_num = std::min(word_num, old_word_num);
                new_data.copyFrom(old_data, sizeof(WordType) * min_word_num);
            }
            data = new_data;
            old_data.free();
        }
    }

    /// @brief Sets all bits to the given value.
    void fill(bool value, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        castProviderPtrAndCall(data.provider(), [&](auto p_ptr) {
            WordType last_word = 0;

            if (value) {
                RSize last_bits = elem_num & BitMask;
                last_word = last_bits == 0 ? FullWordMask : (WordType(1) << last_bits) - 1;
            }
            p_ptr->template run<RSCF_BitmaskFill>(
                {word_num, RS_AUTO_BLOCK_SIZE, sync},
                *this,
                word_num,
                value ? FullWordMask : WordType(0),
                last_word
            );
        });
    }

    /// @brief Writes data from the host to the bitmask.
    void write(const WordType* input, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_POINTER_NOT_NULL(input);

        if (word_num > 0) {
            data.copyFrom(
                RS_ConstMemoryPtr(const_cast<WordType*>(input), sizeof(WordType) * word_num),
                sizeof(WordType) * word_num,
                sync
            );
        }
    }

    /// @brief Copies data from another RS_BitMask.
    void copy(const RS_BitMask& other) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(other.data.provider() != nullptr);

        if (other.data == data) return;
        if (other.elem_num != elem_num) {
            resize(other.elem_num, false);
        }
        if (word_num > 0) {
            data.copyFrom(other.data, sizeof(WordType) * word_num);
        }
    }

    /// @brief Reads data from the bitmask to the host.
    void read(WordType* output, bool sync = true) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_POINTER_NOT_NULL(output);

        if (word_num > 0) {
            data.copyTo({output, sizeof(WordType) * word_num}, sizeof(WordType) * word_num, sync);
        }
    }

    /// @brief Moves data to specified provider and returns a new RS_BitMask.
    RS_BitMask move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        if (elem_num == 0) return RS_BitMask(provider);

        RS_BitMask new_bitmask(elem_num, provider);
        new_bitmask.copy(*this);
        return new_bitmask;
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return data.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return data.device(); }

    /// @brief Returns the number of bits in the bitmask.
    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize size() const { return elem_num; }

    /// @brief Returns true if the bitmask is empty.
    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return elem_num == 0; }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const { return data.size(); }

    /// @brief Return the total number of set bits in this mask.
    HOST_DEVICE_INLINE_QUALIFIER RSize count() const {
        RSize sum = 0;
        if (word_num > 0) {
            auto word_ptr = static_cast<const WordType*>(data.ptr());
            RSize last_bits = elem_num & BitMask;
            WordType last_word_mask = last_bits == 0 ? FullWordMask
                                                     : (WordType(1) << last_bits) - 1;
            for (RSize i = 0; i < word_num - 1; ++i) {
                sum += countOnBits(word_ptr[i]);
            }
            sum += countOnBits(word_ptr[word_num - 1] & last_word_mask);
        }
        return sum;
    }

    /// @brief Checks if the bit at index is set.
    HOST_DEVICE_INLINE_QUALIFIER bool isOn(RIndex index) const {
        assert(index < elem_num);
        auto word_ptr = static_cast<const WordType*>(data.ptr());
        return 0 != (word_ptr[index >> WordShift] & (WordType(1) << (index & BitMask)));
    }

    /// @brief Checks if the bit at index is not set.
    HOST_DEVICE_INLINE_QUALIFIER bool isOff(RIndex index) const { return !isOn(index); }

    /// @brief Checks if all bits are set.
    HOST_DEVICE_INLINE_QUALIFIER bool isAllOn() const {
        if (word_num == 0) return true;
        auto word_ptr = static_cast<const WordType*>(data.ptr());
        RSize last_bits = elem_num & BitMask;
        WordType last_word_mask = last_bits == 0 ? FullWordMask : (WordType(1) << last_bits) - 1;
        for (RSize i = 0; i < word_num - 1; ++i) {
            if (word_ptr[i] != FullWordMask) return false;
        }
        return (word_ptr[word_num - 1] & last_word_mask) == last_word_mask;
    }

    /// @brief Checks if all bits are off.
    HOST_DEVICE_INLINE_QUALIFIER bool isAllOff() const {
        if (word_num == 0) return true;
        auto word_ptr = static_cast<const WordType*>(data.ptr());
        RSize last_bits = elem_num & BitMask;
        WordType last_word_mask = last_bits == 0 ? FullWordMask : (WordType(1) << last_bits) - 1;
        for (RSize i = 0; i < word_num - 1; ++i) {
            if (word_ptr[i] != 0) return false;
        }
        return (word_ptr[word_num - 1] & last_word_mask) == 0;
    }

    HOST_DEVICE_INLINE_QUALIFIER bool operator[](RIndex index) const { return isOn(index); }

    HOST_DEVICE_INLINE_QUALIFIER bool get(RIndex index) const { return isOn(index); }

    /// @brief Sets the bit at index.
    HOST_DEVICE_INLINE_QUALIFIER void setOn(RIndex index) {
        assert(index < elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        word_ptr[index >> WordShift] |= (WordType(1) << (index & BitMask));
    }

    /// @brief Clears the bit at index.
    HOST_DEVICE_INLINE_QUALIFIER void setOff(RIndex index) {
        assert(index < elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        word_ptr[index >> WordShift] &= ~(WordType(1) << (index & BitMask));
    }

    /// @brief Sets the bit at index to the given value.
    HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, bool on) {
        assert(index < elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        auto& word = word_ptr[index >> WordShift];
        index &= BitMask;
        word &= ~(WordType(1) << index);
        word |= static_cast<WordType>(on) << index;
    }

    /// @brief Sets the bit at index atomically.
    HOST_DEVICE_INLINE_QUALIFIER void setOnAtomic(RIndex index) {
        assert(index < elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        // Device code
        deviceAtomicOr(word_ptr + (index >> WordShift), (WordType(1) << (index & BitMask)));
    }

    /// @brief Clears the bit at index atomically.
    HOST_DEVICE_INLINE_QUALIFIER void setOffAtomic(RIndex index) {
        assert(index < elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        deviceAtomicAnd(word_ptr + (index >> WordShift), ~(WordType(1) << (index & BitMask)));
    }

    /// @brief Sets the bit at index to the given value atomically.
    HOST_DEVICE_INLINE_QUALIFIER void setAtomic(RIndex index, bool on) {
        on ? setOnAtomic(index) : setOffAtomic(index);
    }

    /// @brief Locks the bit (set it to on) at index atomically using CAS.
    HOST_DEVICE_INLINE_QUALIFIER bool lockAtomic(RIndex index) {
        assert(index < elem_num);
        auto target_word_ptr = static_cast<WordType*>(data.ptr()) + (index >> WordShift);
        WordType mask = (WordType(1) << (index & BitMask));
        WordType old;
        do {
            old = *target_word_ptr;
            /// If the bit is already occupied, return false
            if (old & mask) return false;
        } while (deviceAtomicCAS(target_word_ptr, old, old | mask) != old);
        return true;
    }

    /// @brief Sets all bits to on.
    HOST_DEVICE_INLINE_QUALIFIER void setAllOn() {
        if (word_num > 0) {
            auto word_ptr = static_cast<WordType*>(data.ptr());
            for (RSize i = 0; i < word_num - 1; ++i) {
                word_ptr[i] = FullWordMask;
            }
            // Handle last word separately
            RSize last_bits = elem_num & BitMask;
            WordType last_word_mask = last_bits == 0 ? FullWordMask
                                                     : (WordType(1) << last_bits) - 1;
            word_ptr[word_num - 1] = last_word_mask;
        }
    }

    /// @brief Sets all bits to off.
    HOST_DEVICE_INLINE_QUALIFIER void setAllOff() {
        if (word_num > 0) {
            auto word_ptr = static_cast<WordType*>(data.ptr());
            for (RSize i = 0; i < word_num; ++i) {
                word_ptr[i] = 0;
            }
        }
    }

    HOST_DEVICE_INLINE_QUALIFIER void setAll(bool on) { on ? setAllOn() : setAllOff(); }

    /// @brief Toggles the bit at index.
    HOST_DEVICE_INLINE_QUALIFIER void toggle(RIndex index) {
        assert(index < elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        word_ptr[index >> WordShift] ^= (WordType(1) << (index & BitMask));
    }

    /// @brief Bitwise intersection
    HOST_DEVICE_INLINE_QUALIFIER RS_BitMask& operator&=(const RS_BitMask& other) {
        assert(elem_num == other.elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        auto other_word_ptr = static_cast<const WordType*>(other.data.ptr());
        for (RSize i = 0; i < word_num; ++i) {
            word_ptr[i] &= other_word_ptr[i];
        }
        return *this;
    }

    /// @brief Bitwise union
    HOST_DEVICE_INLINE_QUALIFIER RS_BitMask& operator|=(const RS_BitMask& other) {
        assert(elem_num == other.elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        auto other_word_ptr = static_cast<const WordType*>(other.data.ptr());
        for (RSize i = 0; i < word_num; ++i) {
            word_ptr[i] |= other_word_ptr[i];
        }
        return *this;
    }

    /// @brief Bitwise difference
    HOST_DEVICE_INLINE_QUALIFIER RS_BitMask& operator-=(const RS_BitMask& other) {
        assert(elem_num == other.elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        auto other_word_ptr = static_cast<const WordType*>(other.data.ptr());
        for (RSize i = 0; i < word_num; ++i) {
            word_ptr[i] &= ~other_word_ptr[i];
        }
        return *this;
    }

    /// @brief Bitwise XOR
    HOST_DEVICE_INLINE_QUALIFIER RS_BitMask& operator^=(const RS_BitMask& other) {
        assert(elem_num == other.elem_num);
        auto word_ptr = static_cast<WordType*>(data.ptr());
        auto other_word_ptr = static_cast<const WordType*>(other.data.ptr());
        for (RSize i = 0; i < word_num; ++i) {
            word_ptr[i] ^= other_word_ptr[i];
        }
        return *this;
    }

    /// @brief Find first on/off bit.
    /// @note: for max efficiency, each thread should find a few words, then reduce.
    /// @param start_word start word index
    /// @tparam ON true to find the first set bit, false to find the first unset bit.
    /// @tparam FindWordNum number of words to search.
    template <bool ON = true, RSize FindWordNum = RS_MAX_SIZE>
    HOST_DEVICE_INLINE_QUALIFIER RIndex findFirst(RIndex start_word = 0) const {
        const RSize max_word_num = (FindWordNum == RS_MAX_SIZE) ? word_num : FindWordNum;
        auto word_ptr = static_cast<const WordType*>(data.ptr());

        for (RSize index = start_word, max_index = RS_MIN(word_num, start_word + max_word_num);
             index < max_index;
             ++index) {
            if (WordType word = ON ? word_ptr[index] : ~word_ptr[index]) {
                RIndex bit_index = (index << WordShift) + findLowestOn(word);
                if (bit_index < elem_num) {
                    return bit_index;
                }
                return RS_NULL_INDEX;
            }
        }
        return RS_NULL_INDEX;
    }

    /// @brief find next on/off bit
    /// @note This should be an operation with spatial locality and search will be quick.
    /// @param start_bit start bit index
    /// @tparam ON true: find next on bit, false: find next off bit
    template <bool ON = true>
    HOST_DEVICE_INLINE_QUALIFIER RIndex findNext(RIndex start_bit) const {
        if (start_bit >= elem_num) return RS_NULL_INDEX;
        auto word_ptr = static_cast<const WordType*>(data.ptr());
        RIndex index = start_bit >> WordShift;

        // Start bit index in the word it is in
        RSize start_local_bit = start_bit & BitMask;
        WordType word = ON ? word_ptr[index] : ~word_ptr[index];

        // start bit is the first found one, return itself
        if (word & (WordType(1) << start_local_bit)) return start_bit;

        // Remove previous bits in current word
        word &= ~WordType(0) << start_local_bit;
        while (!word && ++index < word_num) word = ON ? word_ptr[index] : ~word_ptr[index];
        RIndex result = word ? (index << WordShift) + findLowestOn(word) : RS_NULL_INDEX;
        return result >= elem_num ? RS_NULL_INDEX : result;
    }

    /// @brief find previous on/off bit
    /// @note This should be an operation with spatial locality and search will be quick.
    /// @param start_bit start bit index
    /// @tparam ON true: find previous on bit, false: find previous off bit
    template <bool ON = true>
    HOST_DEVICE_INLINE_QUALIFIER RIndex findPrev(RIndex start_bit) const {
        if (start_bit >= elem_num) return RS_NULL_INDEX;
        auto word_ptr = static_cast<const WordType*>(data.ptr());
        RIndex index = start_bit >> WordShift;

        // Start bit index in the word it is in
        RSize start_local_bit = start_bit & BitMask;
        WordType word = ON ? word_ptr[index] : ~word_ptr[index];

        // start bit is the first found one, return itself
        if (word & (WordType(1) << start_local_bit)) return start_bit;

        word &= (WordType(1) << start_local_bit) - 1u;
        while (!word && index--) word = ON ? word_ptr[index] : ~word_ptr[index];
        RIndex result = word ? (index << WordShift) + findHighestOn(word) : RS_NULL_INDEX;
        return result >= elem_num ? RS_NULL_INDEX : result;
    }
};

/**
 * Static fixed size bit mask class, use 32/64bit words to represent a bit mask.
 * Adapted from NanoVDB https://developer.nvidia.com/nanovdb
 * Optimized lookup efficiency for find functions.
 * @note Unused bits will be filled with 0.
 * @tparam Bits The number of bits needed, real storage size will be aligned
 * to the next n * 2^5 (32 bit words) or n * 2^6 (64 bit words) bits (n >= 0).
 */
template <RSize Bits, typename WordType = uint32_t>
class RS_StaticBitMask {
   public:
    static_assert(
        std::is_same_v<WordType, uint32_t> || std::is_same_v<WordType, uint64_t>,
        "WordType must be uint32_t or uint64_t"
    );
    static constexpr RSize WordBits = std::is_same_v<WordType, uint32_t> ? 32 : 64;
    static constexpr RSize UsedBits = Bits;
    static constexpr RSize WordNum = RS_CEIL_DIV(Bits, WordBits);

    /// The number of bits to shift for the bit index when getting the target word
    static constexpr RSize WordShift = std::is_same_v<WordType, uint32_t> ? 5 : 6;
    /// The bit mask for the target bit
    static constexpr RSize BitMask = WordBits - 1;

    /// Use trinary expression in last word mask to prevent overflow when last word is fully used
    static constexpr WordType FullWordMask =
        std::is_same_v<WordType, uint32_t> ? 0xFFFFFFFF : 0xFFFFFFFFFFFFFFFF;
    static constexpr WordType LastWordMask =
        (UsedBits & BitMask) == 0 ? FullWordMask : (WordType(1) << (Bits & BitMask)) - WordType(1);

    /*****************************************************************
     * Data
     *****************************************************************/
    WordType data[WordNum];

    /*****************************************************************
     * Methods
     *****************************************************************/
    /// @brief Initialize all bits to zero.
    RS_StaticBitMask() { setAllOff(); }

    /// @brief Initialize all bits to specified value.
    explicit RS_StaticBitMask(const bool isAllOn) { isAllOn ? setAllOn() : setAllOff(); }

    /// @brief Copy constructor.
    RS_StaticBitMask(const RS_StaticBitMask& other) {
        std::copy(std::begin(other.data), std::end(other.data), std::begin(data));
    }

    /// @brief Assignment operator.
    RS_StaticBitMask& operator=(const RS_StaticBitMask&) = default;

    /// @brief Equality operator.
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_StaticBitMask& other) const {
        for (RSize i = 0; i < WordNum; ++i)
            if (data[i] != other.data[i]) return false;
        return true;
    }

    /// @brief Inequality operator.
    bool operator!=(const RS_StaticBitMask& other) const { return !(*this == other); }

    /// @brief Return the total number of used bits in this mask.
    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize size() const { return Bits; }

    /// @brief Return true if the mask is empty.
    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return Bits == 0; }

    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const { return 0; }

    /// @brief Return the total number of set bits in this mask.
    HOST_DEVICE_INLINE_QUALIFIER RSize count() const {
        /// Count the number of set bits in the last used word
        auto last_word = data + (WordNum - 1);
        RSize sum = countOnBits(*last_word & LastWordMask);
        for (auto w = data; w < last_word; ++w) {
            sum += countOnBits(*w);
        }
        return sum;
    }

    /// @brief Return true if the given bit is set.
    HOST_DEVICE_INLINE_QUALIFIER bool isOn(RIndex index) const {
        assert(index < Bits);
        return 0 != (data[index >> WordShift] & (WordType(1) << (index & BitMask)));
    }

    /// @brief Return true if the given bit is not set.
    HOST_DEVICE_INLINE_QUALIFIER bool isOff(RIndex index) const { return !isOn(index); }

    /// @brief Return true if all used bits are set.
    HOST_DEVICE_INLINE_QUALIFIER bool isAllOn() const {
        for (RIndex i = 0; i < WordNum - 1; ++i) {
            if (data[i] != FullWordMask) return false;
        }
        return (data[WordNum - 1] & LastWordMask) == LastWordMask;
    }

    /// @brief Return true if all used bits are not set.
    HOST_DEVICE_INLINE_QUALIFIER bool isAllOff() const {
        for (uint32_t i = 0; i < WordNum - 1; ++i) {
            if (data[i] != 0) return false;
        }
        return (data[WordNum - 1] & LastWordMask) == 0;
    }

    HOST_DEVICE_INLINE_QUALIFIER bool operator[](RIndex index) const { return isOn(index); }

    HOST_DEVICE_INLINE_QUALIFIER bool get(RIndex index) const { return isOn(index); }

    /// @brief Set the given bit to on.
    HOST_DEVICE_INLINE_QUALIFIER void setOn(RIndex index) {
        assert(index < Bits);
        data[index >> WordShift] |= (WordType(1) << (index & BitMask));
    }

    /// @brief Set the given bit to off.
    HOST_DEVICE_INLINE_QUALIFIER void setOff(RIndex index) {
        assert(index < Bits);
        data[index >> WordShift] &= ~(WordType(1) << (index & BitMask));
    }

    /// @brief Set the given bit to on or off.
    HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, bool on) {
        assert(index < Bits);
        auto& word = data[index >> WordShift];
        index &= BitMask;
        word &= ~(WordType(1) << index);
        word |= static_cast<WordType>(on) << index;
    }

    /// @brief Set the given bit to on with atomic operation.
    HOST_DEVICE_INLINE_QUALIFIER void setOnAtomic(RIndex index) {
        assert(index < Bits);
        deviceAtomicOr(data + (index >> WordShift), (WordType(1) << (index & BitMask)));
    }

    /// @brief Set the given bit to off with atomic operation.
    HOST_DEVICE_INLINE_QUALIFIER void setOffAtomic(RIndex index) {
        assert(index < Bits);
        deviceAtomicAnd(data + (index >> WordShift), ~(WordType(1) << (index & BitMask)));
    }

    /// @brief Set the given bit to on or off with atomic operation.
    HOST_DEVICE_INLINE_QUALIFIER void setAtomic(RIndex index, bool on) {
        on ? setOnAtomic(index) : setOffAtomic(index);
    }

    /// @brief Locks the bit (sets it to on) at index atomically using CAS.
    HOST_DEVICE_INLINE_QUALIFIER bool lockAtomic(RIndex index) {
        assert(index < Bits);
        auto word_ptr = data + (index >> WordShift);
        WordType mask = (WordType(1) << (index & BitMask));
        WordType old;

        do {
            old = *word_ptr;
            /// If the bit is already occupied, return false;
            if (old & mask) return false;
        } while (deviceAtomicCAS(word_ptr, old, old | mask) != old);
        return true;
    }

    /// @brief Set all used bits to on. Unused bits will be filled with 0.
    HOST_DEVICE_INLINE_QUALIFIER void setAllOn() {
        /// Fill used bits with 1 and fill unused bits with 0
        for (RIndex i = 0; i < WordNum - 1; ++i) {
            data[i] = FullWordMask;
        }
        data[WordNum - 1] = LastWordMask;
    }

    /// @brief Set all used bits to off. Unused bits will be filled with 0.
    HOST_DEVICE_INLINE_QUALIFIER void setAllOff() {
        /// For all off case, fill unused bits with 0
        for (RIndex i = 0; i < WordNum; ++i) data[i] = 0;
    }

    /// @brief Set all used bits to on or off. Unused bits will be filled with 0.
    HOST_DEVICE_INLINE_QUALIFIER void setAll(bool on) { on ? setAllOn() : setAllOff(); }

    /// @brief Toggle the given bit.
    HOST_DEVICE_INLINE_QUALIFIER void toggle(RIndex index) {
        assert(index < Bits);
        data[index >> WordShift] ^= (WordType(1) << (index & BitMask));
    }

    /// @brief Bitwise intersection
    HOST_DEVICE_INLINE_QUALIFIER RS_StaticBitMask& operator&=(const RS_StaticBitMask& other) {
        for (RIndex index = 0; index < WordNum; ++index) data[index] &= other.data[index];
        return *this;
    }

    /// @brief Bitwise union
    HOST_DEVICE_INLINE_QUALIFIER RS_StaticBitMask& operator|=(const RS_StaticBitMask& other) {
        for (RIndex index = 0; index < WordNum; ++index) data[index] |= other.data[index];
        return *this;
    }

    /// @brief Bitwise difference
    HOST_DEVICE_INLINE_QUALIFIER RS_StaticBitMask& operator-=(const RS_StaticBitMask& other) {
        for (RIndex index = 0; index < WordNum; ++index) data[index] &= ~other.data[index];
        return *this;
    }

    /// @brief Bitwise XOR
    HOST_DEVICE_INLINE_QUALIFIER RS_StaticBitMask& operator^=(const RS_StaticBitMask& other) {
        for (RIndex index = 0; index < WordNum; ++index) data[index] ^= other.data[index];
        return *this;
    }

    /// @brief find first on/off bit.
    /// @note: for max efficiency, each thread should find a few words, then reduce.
    /// @param start_word start word index
    /// @tparam ON true: find first on bit, false: find first off bit
    /// @tparam FindWordNum number of words to find
    template <bool ON = true, RSize FindWordNum = WordNum>
    HOST_DEVICE_INLINE_QUALIFIER RIndex findFirst(RIndex start_word = 0) const {
        for (RSize offset = 0; offset < FindWordNum; ++offset) {
            RIndex index = start_word + offset;
            if (index < WordNum) {
                if (WordType word = ON ? data[index] : ~data[index])
                    return (index << WordShift) + findLowestOn(word);
            }
        }
        return RS_NULL_INDEX;
    }

    /// @brief find next on/off bit
    /// @note This should be an operation with spatial locality and search will be quick.
    /// @param start_bit start bit index
    /// @tparam ON true: find next on bit, false: find next off bit
    template <bool ON = true>
    HOST_DEVICE_INLINE_QUALIFIER RIndex findNext(RIndex start_bit) const {
        if (start_bit >= UsedBits) return RS_NULL_INDEX;
        RIndex index = start_bit >> WordShift;

        // Start bit index in the word it is in
        RSize start_local_bit = start_bit & BitMask;
        WordType word = ON ? data[index] : ~data[index];

        // start bit is the first found one, return itself
        if (word & (WordType(1) << start_local_bit)) return start_bit;

        // Remove previous bits in current word
        word &= ~WordType(0) << start_local_bit;
        while (!word && ++index < WordNum) word = ON ? data[index] : ~data[index];
        RIndex result = word ? (index << WordShift) + findLowestOn(word) : RS_NULL_INDEX;
        return result >= UsedBits ? RS_NULL_INDEX : result;
    }

    /// @brief find previous on/off bit
    /// @note This should be an operation with spatial locality and search will be quick.
    /// @param start_bit start bit index
    /// @tparam ON true: find previous on bit, false: find previous off bit
    template <bool ON = true>
    HOST_DEVICE_INLINE_QUALIFIER RIndex findPrev(RIndex start_bit) const {
        if (start_bit >= UsedBits) return RS_NULL_INDEX;
        RIndex index = start_bit >> WordShift;

        // Start bit index in the word it is in
        RSize start_local_bit = start_bit & BitMask;
        WordType word = ON ? data[index] : ~data[index];

        // start bit is the first found one, return itself
        if (word & (WordType(1) << start_local_bit)) return start_bit;

        word &= (WordType(1) << start_local_bit) - 1u;
        while (!word && index--) word = ON ? data[index] : ~data[index];
        RIndex result = word ? (index << WordShift) + findHighestOn(word) : RS_NULL_INDEX;
        return result >= UsedBits ? RS_NULL_INDEX : result;
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_BITMASK_H
