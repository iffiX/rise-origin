#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_bitmask.h"

using namespace rise;
using namespace rise::device;

class RS_BitMaskTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize CUDA stream and memory provider
        provider.init({RSE_DEVICE_TYPE_CUDA, 0});
        host_provider.init(0);
    }

    void TearDown() override {
        provider.free();
        host_provider.free();
    }

    RS_MemoryCudaProvider provider;
    RS_MemoryCPUProvider host_provider;
};

/*****************************************************************
 * Test cases for the RS_BitMask::count()
 * Using 32-bit word size.

 * Test cases:
 * 1. Count All Off Mask.
 * 2. Count All On Mask.
 * 3. Count Random Bits Set.
 *****************************************************************/

TEST_F(RS_BitMaskTest, CountAllOffMask) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOff();
    EXPECT_EQ(bitmask.count(), 0);
}

TEST_F(RS_BitMaskTest, CountAllOnMask) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    EXPECT_EQ(bitmask.count(), 64);
}

TEST_F(RS_BitMaskTest, CountRandomBitsSet) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(1);
    bitmask.setOn(3);
    bitmask.setOn(5);
    EXPECT_EQ(bitmask.count(), 3);
}

/*****************************************************************
 * Test cases for the RS_BitMask::isOn()
 * Using 32-bit word size.

 * Test cases:
 * 1. Check Bit Set to On.
 * 2. Check Bit Set to Off.
 * 3. Out-of-Bounds Index.
 *****************************************************************/

TEST_F(RS_BitMaskTest, IsOnCheckBitSetToOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_BitMaskTest, IsOnCheckBitSetToOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOff(5);
    EXPECT_FALSE(bitmask.isOn(5));
}

TEST_F(RS_BitMaskTest, IsOnOutOfBoundsIndex) {
    RS_BitMask bitmask(64, &host_provider);
    EXPECT_DEATH(bitmask.isOn(64), ".*");
}

/*****************************************************************
 * Test cases for the RS_BitMask::isOff()
 * Using 32-bit word size.

 * Test cases:
 * 1. Check Bit Set to Off.
 * 2. Check Bit Set to On.
 * 3. Out-of-Bounds Index.
 *****************************************************************/

TEST_F(RS_BitMaskTest, IsOffCheckBitSetToOff) {
    RS_BitMask bitmask(64, &host_provider);
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_BitMaskTest, IsOffCheckBitSetToOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    EXPECT_FALSE(bitmask.isOff(5));
}

TEST_F(RS_BitMaskTest, IsOffOutOfBoundsIndex) {
    RS_BitMask bitmask(64, &host_provider);
    EXPECT_DEATH(bitmask.isOff(64), ".*");
}

/*****************************************************************
 * Test cases for the RS_BitMask::isAllOn()
 * Using 32-bit word size.

 * Test cases:
 * 1. All Bits On.
 * 2. Not All Bits On.
 * 3. Unused Bits Handling.
 *****************************************************************/

TEST_F(RS_BitMaskTest, IsAllOnAllBitsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
}

TEST_F(RS_BitMaskTest, IsAllOnNotAllBitsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    bitmask.setOff(5);
    EXPECT_FALSE(bitmask.isAllOn());
}

TEST_F(RS_BitMaskTest, IsAllOnUnusedBitsHandling) {
    RS_BitMask bitmask(70, &host_provider);
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
}

/*****************************************************************
 * Test cases for the RS_BitMask::isAllOff()
 * Using 32-bit word size.

 * Test cases:
 * 1. All Bits Off.
 * 2. Not All Bits Off.
 * 3. Unused Bits Handling.
 *****************************************************************/

TEST_F(RS_BitMaskTest, IsAllOffAllBitsOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
}

TEST_F(RS_BitMaskTest, IsAllOffNotAllBitsOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOff();
    bitmask.setOn(5);
    EXPECT_FALSE(bitmask.isAllOff());
}

TEST_F(RS_BitMaskTest, IsAllOffUnusedBitsHandling) {
    RS_BitMask bitmask(70, &host_provider);
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
}

/*****************************************************************
 * Test cases for the RS_BitMask::setOn()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit On.
 * 2. Repeated Set On.
 * 3. Out-of-Bounds Index.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetOnSetBitOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_BitMaskTest, SetOnRepeatedSetOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_BitMaskTest, SetOnOutOfBoundsIndex) {
    RS_BitMask bitmask(64, &host_provider);
    EXPECT_DEATH(bitmask.setOn(64), ".*");
}

/*****************************************************************
 * Test cases for the RS_BitMask::setOff()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit Off.
 * 2. Repeated Set Off.
 * 3. Out-of-Bounds Index.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetOffSetBitOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOff(5);
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_BitMaskTest, SetOffRepeatedSetOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOff(5);
    bitmask.setOff(5);
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_BitMaskTest, SetOffOutOfBoundsIndex) {
    RS_BitMask bitmask(64, &host_provider);
    EXPECT_DEATH(bitmask.setOff(64), ".*");
}

/*****************************************************************
 * Test cases for the RS_BitMask::set()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit On.
 * 2. Set Bit Off.
 * 3. Toggle Bit.
 * 4. Out-of-Bounds Index.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetSetBitOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.set(5, true);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_BitMaskTest, SetSetBitOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.set(5, true);
    bitmask.set(5, false);
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_BitMaskTest, SetToggleBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.set(5, true);
    EXPECT_TRUE(bitmask.isOn(5));
    bitmask.set(5, false);
    EXPECT_TRUE(bitmask.isOff(5));
    bitmask.set(5, true);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_BitMaskTest, SetOutOfBoundsIndex) {
    RS_BitMask bitmask(64, &host_provider);
    EXPECT_DEATH(bitmask.set(64, true), ".*");
}

/*****************************************************************
 * Test cases for the RS_BitMask::setOnAtomic()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit On Atomically.
 * 2. (TODO) Out-of-Bounds Index.
 *      (Cannot test now since it needs device assertion
 *      which corrupts the test environment.)
 *****************************************************************/

__global__ void testSetOnAtomicSetBitOnAtomically(RS_BitMask<> bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        bitmask.setOnAtomic(index);
    }
}

TEST_F(RS_BitMaskTest, SetOnAtomicSetBitOnAtomically) {
    RS_BitMask d_bitmask(64, &provider);
    RS_BitMask bitmask(64, &host_provider);
    testSetOnAtomicSetBitOnAtomically<<<1, 64>>>(d_bitmask);
    bitmask.copy(d_bitmask);

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOn(i));
    }

    bitmask.free();
    d_bitmask.free();
}

/*****************************************************************
 * Test cases for the RS_BitMask::setOffAtomic()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit Off Atomically.
 * 2. (TODO) Out-of-Bounds Index.
 *      (Cannot test now since it needs device assertion
 *      which corrupts the test environment.)
 *****************************************************************/

__global__ void testSetOffAtomicSetBitOffAtomically(RS_BitMask<> bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        bitmask.setOffAtomic(index);
    }
}

TEST_F(RS_BitMaskTest, SetOffAtomicSetBitOffAtomically) {
    RS_BitMask d_bitmask(64, &provider);
    RS_BitMask bitmask(64, &host_provider);

    // Set all bits to on initially
    for (int i = 0; i < 64; ++i) {
        bitmask.setOn(i);
    }

    testSetOffAtomicSetBitOffAtomically<<<1, 64>>>(d_bitmask);
    bitmask.copy(d_bitmask);

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOff(i));
    }

    bitmask.free();
    d_bitmask.free();
}

/*****************************************************************
 * Test cases for the RS_BitMask::setAtomic()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit On Atomically.
 * 2. Set Bit Off Atomically.
 * 3. Concurrent Setting.
 *****************************************************************/

__global__ void testSetAtomicSetBitOnAtomically(RS_BitMask<> bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        bitmask.setAtomic(index, true);
    }
}

__global__ void testSetAtomicSetBitOffAtomically(RS_BitMask<> bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        bitmask.setAtomic(index, false);
    }
}

__global__ void testSetAtomicConcurrentSetting(RS_BitMask<> bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        if (index % 2 == 0) {
            bitmask.setAtomic(index, true);
        } else {
            bitmask.setAtomic(index, false);
        }
    }
}

TEST_F(RS_BitMaskTest, SetAtomicSetBitOnAtomically) {
    RS_BitMask d_bitmask(64, &provider);
    RS_BitMask bitmask(64, &host_provider);

    testSetAtomicSetBitOnAtomically<<<1, 64>>>(d_bitmask);
    bitmask.copy(d_bitmask);

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOn(i));
    }

    bitmask.free();
    d_bitmask.free();
}

TEST_F(RS_BitMaskTest, SetAtomicSetBitOffAtomically) {
    RS_BitMask d_bitmask(64, &provider);
    RS_BitMask bitmask(64, &host_provider);

    // Set all bits to on initially
    for (int i = 0; i < 64; ++i) {
        bitmask.setOn(i);
    }

    testSetAtomicSetBitOffAtomically<<<1, 64>>>(d_bitmask);
    bitmask.copy(d_bitmask);

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOff(i));
    }

    bitmask.free();
    d_bitmask.free();
}

TEST_F(RS_BitMaskTest, SetAtomicConcurrentSetting) {
    RS_BitMask d_bitmask(64, &provider);
    RS_BitMask bitmask(64, &host_provider);

    testSetAtomicConcurrentSetting<<<1, 64>>>(d_bitmask);
    bitmask.copy(d_bitmask);

    for (int i = 0; i < 64; ++i) {
        if (i % 2 == 0) {
            EXPECT_TRUE(bitmask.isOn(i));
        } else {
            EXPECT_TRUE(bitmask.isOff(i));
        }
    }

    bitmask.free();
    d_bitmask.free();
}

/*****************************************************************
 * Test cases for the RS_BitMask::setAllOn()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set All Bits On.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetAllOnSetAllBitsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
}

/*****************************************************************
 * Test cases for the RS_BitMask::setAllOff()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set All Bits Off.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetAllOffSetAllBitsOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
}

/*****************************************************************
 * Test cases for the RS_BitMask::setAll()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set All On with true.
 * 2. Set All Off with false.
 *****************************************************************/

TEST_F(RS_BitMaskTest, SetAllSetAllOnWithTrue) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAll(true);
    EXPECT_TRUE(bitmask.isAllOn());
}

TEST_F(RS_BitMaskTest, SetAllSetAllOffWithFalse) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAll(false);
    EXPECT_TRUE(bitmask.isAllOff());
}

/*****************************************************************
 * Test cases for the RS_BitMask::toggle()
 * Using 32-bit word size.

 * Test cases:
 * 1. Toggle Bit From Off to On.
 * 2. Toggle Bit From On to Off.
 * 3. Multiple Toggles.
 * 4. Out-of-Bounds Index.
 *****************************************************************/

TEST_F(RS_BitMaskTest, ToggleBitFromOffToOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOff(5);  // Ensure bit is off
    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_BitMaskTest, ToggleBitFromOnToOff) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);  // Ensure bit is on
    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_BitMaskTest, MultipleToggles) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOff(5);  // Ensure bit is off

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOff(5));

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_BitMaskTest, ToggleOutOfBoundsIndex) {
    RS_BitMask bitmask(64, &host_provider);
    EXPECT_DEATH(bitmask.toggle(64), ".*");  // Index 64 is out-of-bounds
}

/*****************************************************************
 * Test cases for the RS_BitMask::operator&=()
 * Using 32-bit word size.

 * Test cases:
 * 1. Bitwise AND with Different Masks.
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
}

/*****************************************************************
 * Test cases for the RS_BitMask::operator|=()
 * Using 32-bit word size.

 * Test cases:
 * 1. Bitwise OR with Different Masks.
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
}

/*****************************************************************
 * Test cases for the RS_BitMask::operator-=()
 * Using 32-bit word size.

 * Test cases:
 * 1. Bitwise Difference with Partial Overlap.
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
}

/*****************************************************************
 * Test cases for the RS_BitMask::operator^=()
 * Using 32-bit word size.

 * Test cases:
 * 1. Bitwise XOR with Different Masks.
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

    EXPECT_TRUE(bitmask1.isOn(1));   // 1 ^ 0 = 1
    EXPECT_FALSE(bitmask1.isOn(3));  // 1 ^ 1 = 0
    EXPECT_FALSE(bitmask1.isOn(5));  // 1 ^ 1 = 0
    EXPECT_TRUE(bitmask1.isOn(7));   // 0 ^ 1 = 1
}

/*****************************************************************
 * Test cases for the RS_BitMask::findFirst()
 * Using 32-bit word size.

 * Test cases:
 * 1. Find First On Bit.
 * 2. Find First Off Bit.
 * 3. No On Bits.
 * 4. Start Word Offset.
 * 5. FindWordNum Limitation.
 *****************************************************************/

TEST_F(RS_BitMaskTest, FindFirstOnBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex firstOn = bitmask.findFirst<true>();
    EXPECT_EQ(firstOn, 5);
}

TEST_F(RS_BitMaskTest, FindFirstOffBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    RIndex firstOff = bitmask.findFirst<false>();
    EXPECT_EQ(firstOff, RS_NULL_INDEX);
}

TEST_F(RS_BitMaskTest, FindFirstNoOnBits) {
    RS_BitMask bitmask(64, &host_provider);
    RIndex firstOn = bitmask.findFirst<true>();
    EXPECT_EQ(firstOn, RS_NULL_INDEX);
}

TEST_F(RS_BitMaskTest, FindFirstStartWordOffset) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(42);                            // In the second word
    RIndex firstOn = bitmask.findFirst<true>(1);  // Start from the second word
    EXPECT_EQ(firstOn, 42);
}

TEST_F(RS_BitMaskTest, FindFirstFindWordNumLimitation) {
    RS_BitMask bitmask(119, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(70);                               // In the third word
    RIndex firstOn = bitmask.findFirst<true, 1>(1);  // Limit search to the second word
    EXPECT_EQ(firstOn, RS_NULL_INDEX);
}

/*****************************************************************
 * Test cases for the RS_BitMask::findNext()
 * Using 32-bit word size.

 * Test cases:
 * 1. Find Next On Bit After Start Bit.
 * 2. Find Next Off Bit After Start Bit.
 * 3. Start Bit Is On.
 * 4. No Next On Bit.
 * 5. Out-of-Bounds Start Bit.
 *****************************************************************/

TEST_F(RS_BitMaskTest, FindNextOnBitAfterStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex nextOn = bitmask.findNext<true>(6);
    EXPECT_EQ(nextOn, 10);
}

TEST_F(RS_BitMaskTest, FindNextOffBitAfterStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    bitmask.setOff(10);
    RIndex nextOff = bitmask.findNext<false>(5);
    EXPECT_EQ(nextOff, 10);
}

TEST_F(RS_BitMaskTest, FindNextStartBitIsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    RIndex nextOn = bitmask.findNext<true>(5);
    EXPECT_EQ(nextOn, 5);
}

TEST_F(RS_BitMaskTest, FindNextNoNextOnBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    RIndex nextOn = bitmask.findNext<true>(6);
    EXPECT_EQ(nextOn, RS_NULL_INDEX);
}

TEST_F(RS_BitMaskTest, FindNextOutOfBoundsStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    RIndex nextOn = bitmask.findNext<true>(64);  // Out-of-bounds
    EXPECT_EQ(nextOn, RS_NULL_INDEX);
}

/*****************************************************************
 * Test cases for the RS_BitMask::findPrev()
 * Using 32-bit word size.

 * Test cases:
 * 1. Find Previous On Bit Before Start Bit.
 * 2. Find Previous Off Bit Before Start Bit.
 * 3. Start Bit Is On.
 * 4. No Previous On Bit.
 * 5. Start Bit at Zero.
 * 6. Out-of-Bounds Start Bit.
 *****************************************************************/

TEST_F(RS_BitMaskTest, FindPrevOnBitBeforeStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex prevOn = bitmask.findPrev<true>(9);
    EXPECT_EQ(prevOn, 5);
}

TEST_F(RS_BitMaskTest, FindPrevOffBitBeforeStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setAllOn();
    bitmask.setOff(10);
    RIndex prevOff = bitmask.findPrev<false>(10);
    EXPECT_EQ(prevOff, 10);
}

TEST_F(RS_BitMaskTest, FindPrevStartBitIsOn) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    RIndex prevOn = bitmask.findPrev<true>(5);
    EXPECT_EQ(prevOn, 5);
}

TEST_F(RS_BitMaskTest, FindPrevNoPreviousOnBit) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(5);
    RIndex prevOn = bitmask.findPrev<true>(4);
    EXPECT_EQ(prevOn, RS_NULL_INDEX);
}

TEST_F(RS_BitMaskTest, FindPrevStartBitAtZero) {
    RS_BitMask bitmask(64, &host_provider);
    bitmask.setOn(0);
    RIndex prevOn = bitmask.findPrev<true>(0);
    EXPECT_EQ(prevOn, 0);
}

TEST_F(RS_BitMaskTest, OutOfBoundsStartBit) {
    RS_BitMask bitmask(64, &host_provider);
    RIndex prevOn = bitmask.findPrev<true>(64);  // Out-of-bounds
    EXPECT_EQ(prevOn, RS_NULL_INDEX);
}