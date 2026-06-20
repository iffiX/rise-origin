#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_bitmask.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_StaticBitMask is a fixed size bitmask whose storage lives on
 * the stack. Words may be 32 or 64 bits wide. All query and set
 * methods run on the host as well as the device, while the atomic
 * methods are exercised on the device through a compute function
 * that takes a device side pointer to the mask.
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
 *****************************************************************/

class RS_StaticBitMaskTest : public ::testing::Test {
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
 * Test cases for RS_StaticBitMask::count()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Count an all off mask.
 * 2. Count an all on mask.
 * 3. Count a mask with a few bits set.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, CountAllOffMask) {
    RS_StaticBitMask<64> bitmask;
    EXPECT_EQ(bitmask.count(), 0);
}

TEST_F(RS_StaticBitMaskTest, CountAllOnMask) {
    RS_StaticBitMask<64> bitmask(true);
    EXPECT_EQ(bitmask.count(), 64);
}

TEST_F(RS_StaticBitMaskTest, CountRandomBitsSet) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(1);
    bitmask.setOn(3);
    bitmask.setOn(5);
    EXPECT_EQ(bitmask.count(), 3);
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::isOn()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Check a bit set to on.
 * 2. Check a bit set to off.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, IsOnCheckBitSetToOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_StaticBitMaskTest, IsOnCheckBitSetToOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOff(5);
    EXPECT_FALSE(bitmask.isOn(5));
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::isOff()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Check a bit set to off.
 * 2. Check a bit set to on.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, IsOffCheckBitSetToOff) {
    RS_StaticBitMask<64> bitmask;
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_StaticBitMaskTest, IsOffCheckBitSetToOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    EXPECT_FALSE(bitmask.isOff(5));
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::isAllOn()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. All bits on.
 * 2. Not all bits on.
 * 3. Unused bits handling.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, IsAllOnAllBitsOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
}

TEST_F(RS_StaticBitMaskTest, IsAllOnNotAllBitsOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    bitmask.setOff(5);
    EXPECT_FALSE(bitmask.isAllOn());
}

TEST_F(RS_StaticBitMaskTest, IsAllOnUnusedBitsHandling) {
    RS_StaticBitMask<70> bitmask;
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::isAllOff()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. All bits off.
 * 2. Not all bits off.
 * 3. Unused bits handling.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, IsAllOffAllBitsOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
}

TEST_F(RS_StaticBitMaskTest, IsAllOffNotAllBitsOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOff();
    bitmask.setOn(5);
    EXPECT_FALSE(bitmask.isAllOff());
}

TEST_F(RS_StaticBitMaskTest, IsAllOffUnusedBitsHandling) {
    RS_StaticBitMask<70> bitmask(true);
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::setOn()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set a bit on.
 * 2. Set the same bit on again.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, SetOnSetBitOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_StaticBitMaskTest, SetOnRepeatedSetOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::setOff()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set a bit off.
 * 2. Set the same bit off again.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, SetOffSetBitOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOff(5);
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_StaticBitMaskTest, SetOffRepeatedSetOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOff(5);
    bitmask.setOff(5);
    EXPECT_TRUE(bitmask.isOff(5));
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::set()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set a bit on.
 * 2. Set a bit off.
 * 3. Switch a bit on and off.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, SetSetBitOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.set(5, true);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_StaticBitMaskTest, SetSetBitOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.set(5, true);
    bitmask.set(5, false);
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_StaticBitMaskTest, SetToggleBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.set(5, true);
    EXPECT_TRUE(bitmask.isOn(5));
    bitmask.set(5, false);
    EXPECT_TRUE(bitmask.isOff(5));
    bitmask.set(5, true);
    EXPECT_TRUE(bitmask.isOn(5));
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::setAllOn()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set all bits on.
 * 2. Set all bits on and check the used bits with an odd size.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, SetAllOnSetAllBitsOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
}

TEST_F(RS_StaticBitMaskTest, SetAllOnCheckUnusedBits) {
    RS_StaticBitMask<70> bitmask;
    bitmask.setAllOn();

    for (int i = 0; i < 70; ++i) {
        EXPECT_TRUE(bitmask.isOn(i));
    }
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::setAllOff()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set all bits off.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, SetAllOffSetAllBitsOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::setAll()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Set all on with true.
 * 2. Set all off with false.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, SetAllSetAllOnWithTrue) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAll(true);
    EXPECT_TRUE(bitmask.isAllOn());
}

TEST_F(RS_StaticBitMaskTest, SetAllSetAllOffWithFalse) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAll(false);
    EXPECT_TRUE(bitmask.isAllOff());
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::toggle()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Toggle a bit from off to on.
 * 2. Toggle a bit from on to off.
 * 3. Toggle a bit several times.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, ToggleBitFromOffToOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOff(5);
    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_StaticBitMaskTest, ToggleBitFromOnToOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_StaticBitMaskTest, MultipleToggles) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOff(5);

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOff(5));

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::operator&=()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Bitwise and with different masks.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, BitwiseAndWithDifferentMasks) {
    RS_StaticBitMask<64> bitmask1;
    RS_StaticBitMask<64> bitmask2;

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
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::operator|=()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Bitwise or with different masks.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, BitwiseOrWithDifferentMasks) {
    RS_StaticBitMask<64> bitmask1;
    RS_StaticBitMask<64> bitmask2;

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
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::operator-=()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Bitwise difference with partial overlap.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, BitwiseDifferenceWithPartialOverlap) {
    RS_StaticBitMask<64> bitmask1;
    RS_StaticBitMask<64> bitmask2;

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
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::operator^=()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Bitwise xor with different masks.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, BitwiseXorWithDifferentMasks) {
    RS_StaticBitMask<64> bitmask1;
    RS_StaticBitMask<64> bitmask2;

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
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::findFirst()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Find the first on bit.
 * 2. Find the first off bit when all bits are on.
 * 3. No on bits.
 * 4. Start at a word offset.
 * 5. Limit the number of words searched.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, FindFirstOnBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex first_on = bitmask.findFirst<true>();
    EXPECT_EQ(first_on, 5);
}

TEST_F(RS_StaticBitMaskTest, FindFirstOffBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    RIndex first_off = bitmask.findFirst<false>();
    EXPECT_EQ(first_off, RS_NULL_INDEX);
}

TEST_F(RS_StaticBitMaskTest, FindFirstNoOnBits) {
    RS_StaticBitMask<64> bitmask;
    RIndex first_on = bitmask.findFirst<true>();
    EXPECT_EQ(first_on, RS_NULL_INDEX);
}

TEST_F(RS_StaticBitMaskTest, FindFirstStartWordOffset) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(42);
    RIndex first_on = bitmask.findFirst<true>(1);
    EXPECT_EQ(first_on, 42);
}

TEST_F(RS_StaticBitMaskTest, FindFirstFindWordNumLimitation) {
    RS_StaticBitMask<119> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(70);
    RIndex first_on = bitmask.findFirst<true, 1>(1);
    EXPECT_EQ(first_on, RS_NULL_INDEX);
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::findNext()
 * Using 32 bit word size.
 *
 * Test cases:
 * 1. Find the next on bit after the start bit.
 * 2. Find the next off bit after the start bit.
 * 3. The start bit is on.
 * 4. No next on bit.
 * 5. Start bit out of range.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, FindNextOnBitAfterStartBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex next_on = bitmask.findNext<true>(6);
    EXPECT_EQ(next_on, 10);
}

TEST_F(RS_StaticBitMaskTest, FindNextOffBitAfterStartBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    bitmask.setOff(10);
    RIndex next_off = bitmask.findNext<false>(5);
    EXPECT_EQ(next_off, 10);
}

TEST_F(RS_StaticBitMaskTest, FindNextStartBitIsOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    RIndex next_on = bitmask.findNext<true>(5);
    EXPECT_EQ(next_on, 5);
}

TEST_F(RS_StaticBitMaskTest, FindNextNoNextOnBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    RIndex next_on = bitmask.findNext<true>(6);
    EXPECT_EQ(next_on, RS_NULL_INDEX);
}

TEST_F(RS_StaticBitMaskTest, FindNextOutOfBoundsStartBit) {
    RS_StaticBitMask<64> bitmask;
    RIndex next_on = bitmask.findNext<true>(64);
    EXPECT_EQ(next_on, RS_NULL_INDEX);
}

/*****************************************************************
 * Test cases for RS_StaticBitMask::findPrev()
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

TEST_F(RS_StaticBitMaskTest, FindPrevOnBitBeforeStartBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex prev_on = bitmask.findPrev<true>(9);
    EXPECT_EQ(prev_on, 5);
}

TEST_F(RS_StaticBitMaskTest, FindPrevOffBitBeforeStartBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    bitmask.setOff(10);
    RIndex prev_off = bitmask.findPrev<false>(10);
    EXPECT_EQ(prev_off, 10);
}

TEST_F(RS_StaticBitMaskTest, FindPrevStartBitIsOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    RIndex prev_on = bitmask.findPrev<true>(5);
    EXPECT_EQ(prev_on, 5);
}

TEST_F(RS_StaticBitMaskTest, FindPrevNoPreviousOnBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    RIndex prev_on = bitmask.findPrev<true>(4);
    EXPECT_EQ(prev_on, RS_NULL_INDEX);
}

TEST_F(RS_StaticBitMaskTest, FindPrevStartBitAtZero) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(0);
    RIndex prev_on = bitmask.findPrev<true>(0);
    EXPECT_EQ(prev_on, 0);
}

TEST_F(RS_StaticBitMaskTest, OutOfBoundsStartBit) {
    RS_StaticBitMask<64> bitmask;
    RIndex prev_on = bitmask.findPrev<true>(64);
    EXPECT_EQ(prev_on, RS_NULL_INDEX);
}

/*****************************************************************
 * Test cases for RS_StaticBitMask device side atomic methods
 * Using 32 bit word size.
 *
 * The mask is copied into device memory, a compute function runs
 * one thread per bit and uses the atomic method on its own bit
 * through a device side pointer, and the mask is copied back to
 * the host to check the result.
 *
 * Test cases:
 * 1. setOnAtomic() sets every bit on.
 * 2. setOffAtomic() clears every bit.
 * 3. setAtomic() sets every bit on.
 * 4. setAtomic() clears every bit.
 * 5. setAtomic() sets even bits on and odd bits off.
 *****************************************************************/

class RSCF_StaticBitmaskSetOnAtomic : public RS_ComputeFunction<RSCF_StaticBitmaskSetOnAtomic> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StaticBitMask<64>* mask) {
        RIndex index = index1D();
        if (index < 64) {
            mask->setOnAtomic(index);
        }
    }
};

class RSCF_StaticBitmaskSetOffAtomic : public RS_ComputeFunction<RSCF_StaticBitmaskSetOffAtomic> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StaticBitMask<64>* mask) {
        RIndex index = index1D();
        if (index < 64) {
            mask->setOffAtomic(index);
        }
    }
};

class RSCF_StaticBitmaskSetAtomicOn : public RS_ComputeFunction<RSCF_StaticBitmaskSetAtomicOn> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StaticBitMask<64>* mask) {
        RIndex index = index1D();
        if (index < 64) {
            mask->setAtomic(index, true);
        }
    }
};

class RSCF_StaticBitmaskSetAtomicOff : public RS_ComputeFunction<RSCF_StaticBitmaskSetAtomicOff> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StaticBitMask<64>* mask) {
        RIndex index = index1D();
        if (index < 64) {
            mask->setAtomic(index, false);
        }
    }
};

class RSCF_StaticBitmaskSetAtomicEvenOddOnOff
    : public RS_ComputeFunction<RSCF_StaticBitmaskSetAtomicEvenOddOnOff> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_StaticBitMask<64>* mask) {
        RIndex index = index1D();
        if (index < 64) {
            mask->setAtomic(index, index % 2 == 0);
        }
    }
};

TEST_F(RS_StaticBitMaskTest, SetOnAtomicSetBitOnAtomically) {
    RS_StaticBitMask<64> bitmask;

    RS_MemoryPtr device_mask = provider.allocate(sizeof(RS_StaticBitMask<64>));
    RS_MemoryPtr host_mask = provider.allocateHost(sizeof(RS_StaticBitMask<64>));

    *host_mask.as<RS_StaticBitMask<64>>() = bitmask;
    device_mask.copyFrom(host_mask, sizeof(RS_StaticBitMask<64>));

    provider.run<RSCF_StaticBitmaskSetOnAtomic>({64}, device_mask.as<RS_StaticBitMask<64>>());

    device_mask.copyTo(host_mask, sizeof(RS_StaticBitMask<64>));
    bitmask = *host_mask.as<RS_StaticBitMask<64>>();

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOn(i));
    }

    device_mask.free();
    host_mask.free();
}

TEST_F(RS_StaticBitMaskTest, SetOffAtomicSetBitOffAtomically) {
    RS_StaticBitMask<64> bitmask(true);

    RS_MemoryPtr device_mask = provider.allocate(sizeof(RS_StaticBitMask<64>));
    RS_MemoryPtr host_mask = provider.allocateHost(sizeof(RS_StaticBitMask<64>));

    *host_mask.as<RS_StaticBitMask<64>>() = bitmask;
    device_mask.copyFrom(host_mask, sizeof(RS_StaticBitMask<64>));

    provider.run<RSCF_StaticBitmaskSetOffAtomic>({64}, device_mask.as<RS_StaticBitMask<64>>());

    device_mask.copyTo(host_mask, sizeof(RS_StaticBitMask<64>));
    bitmask = *host_mask.as<RS_StaticBitMask<64>>();

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOff(i));
    }

    device_mask.free();
    host_mask.free();
}

TEST_F(RS_StaticBitMaskTest, SetAtomicSetBitOnAtomically) {
    RS_StaticBitMask<64> bitmask;

    RS_MemoryPtr device_mask = provider.allocate(sizeof(RS_StaticBitMask<64>));
    RS_MemoryPtr host_mask = provider.allocateHost(sizeof(RS_StaticBitMask<64>));

    *host_mask.as<RS_StaticBitMask<64>>() = bitmask;
    device_mask.copyFrom(host_mask, sizeof(RS_StaticBitMask<64>));

    provider.run<RSCF_StaticBitmaskSetAtomicOn>({64}, device_mask.as<RS_StaticBitMask<64>>());

    device_mask.copyTo(host_mask, sizeof(RS_StaticBitMask<64>));
    bitmask = *host_mask.as<RS_StaticBitMask<64>>();

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOn(i));
    }

    device_mask.free();
    host_mask.free();
}

TEST_F(RS_StaticBitMaskTest, SetAtomicSetBitOffAtomically) {
    RS_StaticBitMask<64> bitmask(true);

    RS_MemoryPtr device_mask = provider.allocate(sizeof(RS_StaticBitMask<64>));
    RS_MemoryPtr host_mask = provider.allocateHost(sizeof(RS_StaticBitMask<64>));

    *host_mask.as<RS_StaticBitMask<64>>() = bitmask;
    device_mask.copyFrom(host_mask, sizeof(RS_StaticBitMask<64>));

    provider.run<RSCF_StaticBitmaskSetAtomicOff>({64}, device_mask.as<RS_StaticBitMask<64>>());

    device_mask.copyTo(host_mask, sizeof(RS_StaticBitMask<64>));
    bitmask = *host_mask.as<RS_StaticBitMask<64>>();

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOff(i));
    }

    device_mask.free();
    host_mask.free();
}

TEST_F(RS_StaticBitMaskTest, SetAtomicConcurrentSetting) {
    RS_StaticBitMask<64> bitmask;

    RS_MemoryPtr device_mask = provider.allocate(sizeof(RS_StaticBitMask<64>));
    RS_MemoryPtr host_mask = provider.allocateHost(sizeof(RS_StaticBitMask<64>));

    *host_mask.as<RS_StaticBitMask<64>>() = bitmask;
    device_mask.copyFrom(host_mask, sizeof(RS_StaticBitMask<64>));

    provider.run<RSCF_StaticBitmaskSetAtomicEvenOddOnOff>(
        {64}, device_mask.as<RS_StaticBitMask<64>>()
    );

    device_mask.copyTo(host_mask, sizeof(RS_StaticBitMask<64>));
    bitmask = *host_mask.as<RS_StaticBitMask<64>>();

    for (int i = 0; i < 64; ++i) {
        if (i % 2 == 0) {
            EXPECT_TRUE(bitmask.isOn(i));
        } else {
            EXPECT_TRUE(bitmask.isOff(i));
        }
    }

    device_mask.free();
    host_mask.free();
}
