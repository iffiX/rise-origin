#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_bitmask.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_BitMask is a dynamic bitmask container for efficient bit
 * handling on both CPU and GPU device memory. Words may be 32 or
 * 64 bits wide. Most query and set methods run on the host as well
 * as the device, while the atomic methods are exercised on the
 * device through a compute function.
 *
 * - Query Operations
 *   count(), isOn(), isOff(), isAllOn(), isAllOff()
 *
 * - Bit Modification
 *   setOn(), setOff(), set(), toggle()
 *
 * - Bulk Operations
 *   setAllOn(), setAllOff(), setAll()
 *
 * - Bitwise Operators
 *   &= (and), |= (or), -= (difference), ^= (xor)
 *
 * - Search Operations
 *   findFirst(), findNext(), findPrev()
 *
 * - Atomic Operations on the device
 *   setOnAtomic(), setOffAtomic(), setAtomic()
 *
 * - Memory Management
 *   Constructor with size and memory provider, free()
 *****************************************************************/

class RS_BitMaskTest : public ::testing::Test {
   protected:
    void SetUp() override {
        RS_TEST_SETUP_DEVICE_PROVIDER();
        host_provider.init(0);
    }

    void TearDown() override {
        provider.free();
        host_provider.free();
    }

    RS_TEST_DEVICE_PROVIDER_TYPE provider;
    RS_MemoryCPUProvider host_provider;
};

/*****************************************************************
 * Test cases for RS_BitMask::count()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Count an all off mask.
 * 2. Count an all on mask.
 * 3. Count a mask with a few bits set.
 *****************************************************************/

TEST_F(RS_BitMaskTest, CountAllOffMask) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOff();
    EXPECT_EQ(bitmask.count(), 0);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, CountAllOnMask) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    EXPECT_EQ(bitmask.count(), 64);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, CountRandomBitsSet) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(1);
    bitmask.setOn(3);
    bitmask.setOn(5);
    EXPECT_EQ(bitmask.count(), 3);
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::isOn()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Check a bit set to on.
 * 2. Check a bit set to off.
 *****************************************************************/

TEST_F(RS_BitMaskTest, IsOnCheckBitSetToOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    EXPECT_TRUE(bitmask.isOn(5));
    bitmask.free();
}

TEST_F(RS_BitMaskTest, IsOnCheckBitSetToOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOff(5);
    EXPECT_FALSE(bitmask.isOn(5));
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::isOff()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Check a bit set to off.
 * 2. Check a bit set to on.
 *****************************************************************/

TEST_F(RS_BitMaskTest, IsOffCheckBitSetToOff) {
    RS_BitMask bitmask(64, &host_provider);
    EXPECT_TRUE(bitmask.isOff(5));
    bitmask.free();
}

TEST_F(RS_BitMaskTest, IsOffCheckBitSetToOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    EXPECT_FALSE(bitmask.isOff(5));
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::isAllOn()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. All bits on.
 * 2. Not all bits on.
 * 3. Unused bits handling.
 *****************************************************************/

TEST_F(RS_BitMaskTest, IsAllOnAllBitsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
    bitmask.free();
}

TEST_F(RS_BitMaskTest, IsAllOnNotAllBitsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    bitmask.setOff(5);
    EXPECT_FALSE(bitmask.isAllOn());
    bitmask.free();
}

TEST_F(RS_BitMaskTest, IsAllOnUnusedBitsHandling) {
    RS_BitMask bitmask(70, &host_provider);
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::isAllOff()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. All bits off.
 * 2. Not all bits off.
 * 3. Unused bits handling.
 *****************************************************************/

TEST_F(RS_BitMaskTest, IsAllOffAllBitsOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
    bitmask.free();
}

TEST_F(RS_BitMaskTest, IsAllOffNotAllBitsOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOff();
    bitmask.setOn(5);
    EXPECT_FALSE(bitmask.isAllOff());
    bitmask.free();
}

TEST_F(RS_BitMaskTest, IsAllOffUnusedBitsHandling) {
    RS_BitMask bitmask(70, &host_provider);
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::setOn()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set a bit on.
 * 2. Set the same bit on again.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetOnSetBitOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    EXPECT_TRUE(bitmask.isOn(5));
    bitmask.free();
}

TEST_F(RS_BitMaskTest, SetOnRepeatedSetOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(5);
    EXPECT_TRUE(bitmask.isOn(5));
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::setOff()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set a bit off.
 * 2. Set the same bit off again.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetOffSetBitOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOff(5);
    EXPECT_TRUE(bitmask.isOff(5));
    bitmask.free();
}

TEST_F(RS_BitMaskTest, SetOffRepeatedSetOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOff(5);
    bitmask.setOff(5);
    EXPECT_TRUE(bitmask.isOff(5));
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::set()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set a bit on.
 * 2. Set a bit off.
 * 3. Switch a bit on and off.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetSetBitOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.set(5, true);
    EXPECT_TRUE(bitmask.isOn(5));
    bitmask.free();
}

TEST_F(RS_BitMaskTest, SetSetBitOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.set(5, true);
    bitmask.set(5, false);
    EXPECT_TRUE(bitmask.isOff(5));
    bitmask.free();
}

TEST_F(RS_BitMaskTest, SetToggleBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.set(5, true);
    EXPECT_TRUE(bitmask.isOn(5));
    bitmask.set(5, false);
    EXPECT_TRUE(bitmask.isOff(5));
    bitmask.set(5, true);
    EXPECT_TRUE(bitmask.isOn(5));
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::setAllOn()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set all bits on.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetAllOnSetAllBitsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::setAllOff()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set all bits off.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetAllOffSetAllBitsOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::setAll()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set all on with true.
 * 2. Set all off with false.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetAllSetAllOnWithTrue) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAll(true);
    EXPECT_TRUE(bitmask.isAllOn());
    bitmask.free();
}

TEST_F(RS_BitMaskTest, SetAllSetAllOffWithFalse) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAll(false);
    EXPECT_TRUE(bitmask.isAllOff());
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::toggle()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Toggle a bit from off to on.
 * 2. Toggle a bit from on to off.
 * 3. Toggle a bit several times.
 *****************************************************************/

TEST_F(RS_BitMaskTest, ToggleBitFromOffToOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOff(5);
    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));
    bitmask.free();
}

TEST_F(RS_BitMaskTest, ToggleBitFromOnToOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOff(5));
    bitmask.free();
}

TEST_F(RS_BitMaskTest, MultipleToggles) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOff(5);

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOff(5));

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));

    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::operator&=()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Bitwise and with different masks.
 *****************************************************************/

TEST_F(RS_BitMaskTest, BitwiseAndWithDifferentMasks) {
    RS_BitMask bitmask1(64, &host_provider);
    RS_BitMask bitmask2(64, &host_provider);

    bitmask1.setOn(1);
    bitmask1.setOn(3);
    bitmask1.setOn(5);

    bitmask2.setOn(3);
    bitmask2.setOn(5);
    bitmask2.setOn(7);

    bitmask1 &= bitmask2;

    EXPECT_FALSE(bitmask1.isOn(1));
    EXPECT_TRUE(bitmask1.isOn(3));
    EXPECT_TRUE(bitmask1.isOn(5));
    EXPECT_FALSE(bitmask1.isOn(7));

    bitmask1.free();
    bitmask2.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::operator|=()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Bitwise or with different masks.
 *****************************************************************/

TEST_F(RS_BitMaskTest, BitwiseOrWithDifferentMasks) {
    RS_BitMask bitmask1(64, &host_provider);
    RS_BitMask bitmask2(64, &host_provider);

    bitmask1.setOn(1);
    bitmask1.setOn(3);
    bitmask1.setOn(5);

    bitmask2.setOn(3);
    bitmask2.setOn(5);
    bitmask2.setOn(7);

    bitmask1 |= bitmask2;

    EXPECT_TRUE(bitmask1.isOn(1));
    EXPECT_TRUE(bitmask1.isOn(3));
    EXPECT_TRUE(bitmask1.isOn(5));
    EXPECT_TRUE(bitmask1.isOn(7));

    bitmask1.free();
    bitmask2.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::operator-=()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Bitwise difference with partial overlap.
 *****************************************************************/

TEST_F(RS_BitMaskTest, BitwiseDifferenceWithPartialOverlap) {
    RS_BitMask bitmask1(64, &host_provider);
    RS_BitMask bitmask2(64, &host_provider);

    bitmask1.setOn(1);
    bitmask1.setOn(3);
    bitmask1.setOn(5);
    bitmask1.setOn(7);

    bitmask2.setOn(3);
    bitmask2.setOn(5);

    bitmask1 -= bitmask2;

    EXPECT_TRUE(bitmask1.isOn(1));
    EXPECT_FALSE(bitmask1.isOn(3));
    EXPECT_FALSE(bitmask1.isOn(5));
    EXPECT_TRUE(bitmask1.isOn(7));

    bitmask1.free();
    bitmask2.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::operator^=()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Bitwise xor with different masks.
 *****************************************************************/

TEST_F(RS_BitMaskTest, BitwiseXorWithDifferentMasks) {
    RS_BitMask bitmask1(64, &host_provider);
    RS_BitMask bitmask2(64, &host_provider);

    bitmask1.setOn(1);
    bitmask1.setOn(3);
    bitmask1.setOn(5);

    bitmask2.setOn(3);
    bitmask2.setOn(5);
    bitmask2.setOn(7);

    bitmask1 ^= bitmask2;

    EXPECT_TRUE(bitmask1.isOn(1));
    EXPECT_FALSE(bitmask1.isOn(3));
    EXPECT_FALSE(bitmask1.isOn(5));
    EXPECT_TRUE(bitmask1.isOn(7));

    bitmask1.free();
    bitmask2.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::findFirst()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Find the first on bit.
 * 2. Find the first off bit when all bits are on.
 * 3. No on bits.
 * 4. Start at a word offset.
 * 5. Limit the number of words searched.
 *****************************************************************/

TEST_F(RS_BitMaskTest, FindFirstOnBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex first_on = bitmask.findFirst<true>();
    EXPECT_EQ(first_on, 5);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindFirstOffBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    RIndex first_off = bitmask.findFirst<false>();
    EXPECT_EQ(first_off, RS_NULL_INDEX);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindFirstNoOnBits) {
    RS_BitMask bitmask(64, &host_provider);
    RIndex first_on = bitmask.findFirst<true>();
    EXPECT_EQ(first_on, RS_NULL_INDEX);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindFirstStartWordOffset) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(42);
    RIndex first_on = bitmask.findFirst<true>(1);
    EXPECT_EQ(first_on, 42);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindFirstFindWordNumLimitation) {
    RS_BitMask bitmask(119, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(70);
    RIndex first_on = bitmask.findFirst<true, 1>(1);
    EXPECT_EQ(first_on, RS_NULL_INDEX);
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::findNext()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Find the next on bit after the start bit.
 * 2. Find the next off bit after the start bit.
 * 3. The start bit is on.
 * 4. No next on bit.
 * 5. Start bit out of range.
 *****************************************************************/

TEST_F(RS_BitMaskTest, FindNextOnBitAfterStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex next_on = bitmask.findNext<true>(6);
    EXPECT_EQ(next_on, 10);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindNextOffBitAfterStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    bitmask.setOff(10);
    RIndex next_off = bitmask.findNext<false>(5);
    EXPECT_EQ(next_off, 10);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindNextStartBitIsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    RIndex next_on = bitmask.findNext<true>(5);
    EXPECT_EQ(next_on, 5);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindNextNoNextOnBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    RIndex next_on = bitmask.findNext<true>(6);
    EXPECT_EQ(next_on, RS_NULL_INDEX);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindNextOutOfBoundsStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    RIndex next_on = bitmask.findNext<true>(64);
    EXPECT_EQ(next_on, RS_NULL_INDEX);
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask::findPrev()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Find the previous on bit before the start bit.
 * 2. Find the previous off bit before the start bit.
 * 3. The start bit is on.
 * 4. No previous on bit.
 * 5. Start bit at zero.
 * 6. Start bit out of range.
 *****************************************************************/

TEST_F(RS_BitMaskTest, FindPrevOnBitBeforeStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex prev_on = bitmask.findPrev<true>(9);
    EXPECT_EQ(prev_on, 5);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindPrevOffBitBeforeStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    bitmask.setOff(10);
    RIndex prev_off = bitmask.findPrev<false>(10);
    EXPECT_EQ(prev_off, 10);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindPrevStartBitIsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    RIndex prev_on = bitmask.findPrev<true>(5);
    EXPECT_EQ(prev_on, 5);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindPrevNoPreviousOnBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    RIndex prev_on = bitmask.findPrev<true>(4);
    EXPECT_EQ(prev_on, RS_NULL_INDEX);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, FindPrevStartBitAtZero) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(0);
    RIndex prev_on = bitmask.findPrev<true>(0);
    EXPECT_EQ(prev_on, 0);
    bitmask.free();
}

TEST_F(RS_BitMaskTest, OutOfBoundsStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    RIndex prev_on = bitmask.findPrev<true>(64);
    EXPECT_EQ(prev_on, RS_NULL_INDEX);
    bitmask.free();
}

/*****************************************************************
 * Test cases for RS_BitMask device side atomic methods
 * Using 32 bit word size.
 *
 * Each compute function runs one thread per bit and uses the
 * atomic method on its own bit. The mask lives on the device
 * provider and is moved to the host to check the result.
 *
 * Test cases:
 * 1. setOnAtomic() sets every bit on.
 * 2. setOffAtomic() clears every bit.
 * 3. setAtomic() sets every bit on.
 * 4. setAtomic() clears every bit.
 * 5. setAtomic() sets even bits on and odd bits off.
 *****************************************************************/

class RSCF_BitmaskSetOnAtomic : public RS_ComputeFunction<RSCF_BitmaskSetOnAtomic> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_BitMask<> mask, RSize size) {
        RIndex index = index1D();
        if (index < size) {
            mask.setOnAtomic(index);
        }
    }
};

class RSCF_BitmaskSetOffAtomic : public RS_ComputeFunction<RSCF_BitmaskSetOffAtomic> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_BitMask<> mask, RSize size) {
        RIndex index = index1D();
        if (index < size) {
            mask.setOffAtomic(index);
        }
    }
};

class RSCF_BitmaskSetAtomicOn : public RS_ComputeFunction<RSCF_BitmaskSetAtomicOn> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_BitMask<> mask, RSize size) {
        RIndex index = index1D();
        if (index < size) {
            mask.setAtomic(index, true);
        }
    }
};

class RSCF_BitmaskSetAtomicOff : public RS_ComputeFunction<RSCF_BitmaskSetAtomicOff> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_BitMask<> mask, RSize size) {
        RIndex index = index1D();
        if (index < size) {
            mask.setAtomic(index, false);
        }
    }
};

class RSCF_BitmaskSetAtomicEvenOddOnOff
    : public RS_ComputeFunction<RSCF_BitmaskSetAtomicEvenOddOnOff> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_BitMask<> mask, RSize size) {
        RIndex index = index1D();
        if (index < size) {
            mask.setAtomic(index, index % 2 == 0);
        }
    }
};

TEST_F(RS_BitMaskTest, SetOnAtomicSetBitOnAtomically) {
    RS_BitMask device_bitmask(64, &provider);

    provider.run<RSCF_BitmaskSetOnAtomic>({64}, device_bitmask, RSize(64));

    RS_BitMask host_bitmask = device_bitmask.move(&host_provider);
    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(host_bitmask.isOn(i));
    }

    device_bitmask.free();
    host_bitmask.free();
}

TEST_F(RS_BitMaskTest, SetOffAtomicSetBitOffAtomically) {
    RS_BitMask host_seed(64, &host_provider);
    host_seed.setAllOn();

    RS_BitMask device_bitmask(64, &provider);
    device_bitmask.copy(host_seed);

    provider.run<RSCF_BitmaskSetOffAtomic>({64}, device_bitmask, RSize(64));

    RS_BitMask host_bitmask = device_bitmask.move(&host_provider);
    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(host_bitmask.isOff(i));
    }

    host_seed.free();
    device_bitmask.free();
    host_bitmask.free();
}

TEST_F(RS_BitMaskTest, SetAtomicSetBitOnAtomically) {
    RS_BitMask device_bitmask(64, &provider);

    provider.run<RSCF_BitmaskSetAtomicOn>({64}, device_bitmask, RSize(64));

    RS_BitMask host_bitmask = device_bitmask.move(&host_provider);
    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(host_bitmask.isOn(i));
    }

    device_bitmask.free();
    host_bitmask.free();
}

TEST_F(RS_BitMaskTest, SetAtomicSetBitOffAtomically) {
    RS_BitMask host_seed(64, &host_provider);
    host_seed.setAllOn();

    RS_BitMask device_bitmask(64, &provider);
    device_bitmask.copy(host_seed);

    provider.run<RSCF_BitmaskSetAtomicOff>({64}, device_bitmask, RSize(64));

    RS_BitMask host_bitmask = device_bitmask.move(&host_provider);
    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(host_bitmask.isOff(i));
    }

    host_seed.free();
    device_bitmask.free();
    host_bitmask.free();
}

TEST_F(RS_BitMaskTest, SetAtomicConcurrentSetting) {
    RS_BitMask device_bitmask(64, &provider);

    provider.run<RSCF_BitmaskSetAtomicEvenOddOnOff>({64}, device_bitmask, RSize(64));

    RS_BitMask host_bitmask = device_bitmask.move(&host_provider);
    for (int i = 0; i < 64; ++i) {
        if (i % 2 == 0) {
            EXPECT_TRUE(host_bitmask.isOn(i));
        } else {
            EXPECT_TRUE(host_bitmask.isOff(i));
        }
    }

    device_bitmask.free();
    host_bitmask.free();
}
