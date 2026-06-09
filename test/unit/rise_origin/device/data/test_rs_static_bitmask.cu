#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_bitmask.h"

using namespace rise;
using namespace rise::device;

class RS_StaticBitMaskTest : public ::testing::Test {
   protected:
    void SetUp() override {
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        cudaStreamCreate(&mockStream);
    }

    void TearDown() override {
        cudaStreamDestroy(mockStream);
        GTEST_FLAG_SET(death_test_style, "fast");
    }

    cudaStream_t mockStream;
};

/*****************************************************************
 * Test cases for the RS_StaticBitMask::count()
 * Using 32-bit word size.

 * Test cases:
 * 1. Count All Off Mask.
 * 2. Count All On Mask.
 * 3. Count Random Bits Set.
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
 * Test cases for the RS_StaticBitMask::isOn()
 * Using 32-bit word size.

 * Test cases:
 * 1. Check Bit Set to On.
 * 2. Check Bit Set to Off.
 * 3. Out-of-Bounds Index.
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

TEST_F(RS_StaticBitMaskTest, IsOnOutOfBoundsIndex) {
    RS_StaticBitMask<64> bitmask;
    EXPECT_DEATH(bitmask.isOn(64), ".*");
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::isOff()
 * Using 32-bit word size.

 * Test cases:
 * 1. Check Bit Set to Off.
 * 2. Check Bit Set to On.
 * 3. Out-of-Bounds Index.
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

TEST_F(RS_StaticBitMaskTest, IsOffOutOfBoundsIndex) {
    RS_StaticBitMask<64> bitmask;
    EXPECT_DEATH(bitmask.isOff(64), ".*");
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::isAllOn()
 * Using 32-bit word size.

 * Test cases:
 * 1. All Bits On.
 * 2. Not All Bits On.
 * 3. Unused Bits Handling.
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
 * Test cases for the RS_StaticBitMask::isAllOff()
 * Using 32-bit word size.

 * Test cases:
 * 1. All Bits Off.
 * 2. Not All Bits Off.
 * 3. Unused Bits Handling.
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
 * Test cases for the RS_StaticBitMask::setOn()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit On.
 * 2. Repeated Set On.
 * 3. Out-of-Bounds Index.
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

TEST_F(RS_StaticBitMaskTest, SetOnOutOfBoundsIndex) {
    RS_StaticBitMask<64> bitmask;
    EXPECT_DEATH(bitmask.setOn(64), ".*");
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::setOff()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit Off.
 * 2. Repeated Set Off.
 * 3. Out-of-Bounds Index.
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

TEST_F(RS_StaticBitMaskTest, SetOffOutOfBoundsIndex) {
    RS_StaticBitMask<64> bitmask;
    EXPECT_DEATH(bitmask.setOff(64), ".*");
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::set()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit On.
 * 2. Set Bit Off.
 * 3. Toggle Bit.
 * 4. Out-of-Bounds Index.
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

TEST_F(RS_StaticBitMaskTest, SetOutOfBoundsIndex) {
    RS_StaticBitMask<64> bitmask;
    EXPECT_DEATH(bitmask.set(64, true), ".*");
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::setOnAtomic()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit On Atomically.
 * 2. (TODO) Out-of-Bounds Index.
 *      (Cannot test now since it needs device assertion
 *      which corrupts the test environment.)
 *****************************************************************/

__global__ void testSetOnAtomicSetBitOnAtomically(RS_StaticBitMask<64> *bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        bitmask->setOnAtomic(index);
    }
}

TEST_F(RS_StaticBitMaskTest, SetOnAtomicSetBitOnAtomically) {
    RS_StaticBitMask<64> bitmask;
    RS_StaticBitMask<64> *d_bitmask;

    cudaMalloc(&d_bitmask, sizeof(RS_StaticBitMask<64>));
    cudaMemcpy(d_bitmask, &bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyHostToDevice);

    testSetOnAtomicSetBitOnAtomically<<<1, 64>>>(d_bitmask);
    cudaMemcpy(&bitmask, d_bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOn(i));
    }

    cudaFree(d_bitmask);
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::setOffAtomic()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit Off Atomically.
 * 2. (TODO) Out-of-Bounds Index.
 *      (Cannot test now since it needs device assertion
 *      which corrupts the test environment.)
 *****************************************************************/

__global__ void testSetOffAtomicSetBitOffAtomically(RS_StaticBitMask<64> *bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        bitmask->setOffAtomic(index);
    }
}

TEST_F(RS_StaticBitMaskTest, SetOffAtomicSetBitOffAtomically) {
    RS_StaticBitMask<64> bitmask;
    RS_StaticBitMask<64> *d_bitmask;

    cudaMalloc(&d_bitmask, sizeof(RS_StaticBitMask<64>));
    cudaMemcpy(d_bitmask, &bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyHostToDevice);

    // Set all bits to on initially
    for (int i = 0; i < 64; ++i) {
        bitmask.setOn(i);
    }
    cudaMemcpy(d_bitmask, &bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyHostToDevice);

    testSetOffAtomicSetBitOffAtomically<<<1, 64>>>(d_bitmask);
    cudaMemcpy(&bitmask, d_bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOff(i));
    }

    cudaFree(d_bitmask);
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::setAtomic()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set Bit On Atomically.
 * 2. Set Bit Off Atomically.
 * 3. Concurrent Setting.
 *****************************************************************/

__global__ void testSetAtomicSetBitOnAtomically(RS_StaticBitMask<64> *bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        bitmask->setAtomic(index, true);
    }
}

__global__ void testSetAtomicSetBitOffAtomically(RS_StaticBitMask<64> *bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        bitmask->setAtomic(index, false);
    }
}

__global__ void testSetAtomicConcurrentSetting(RS_StaticBitMask<64> *bitmask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 64) {
        if (index % 2 == 0) {
            bitmask->setAtomic(index, true);
        } else {
            bitmask->setAtomic(index, false);
        }
    }
}

TEST_F(RS_StaticBitMaskTest, SetAtomicSetBitOnAtomically) {
    RS_StaticBitMask<64> bitmask;
    RS_StaticBitMask<64> *d_bitmask;

    cudaMalloc(&d_bitmask, sizeof(RS_StaticBitMask<64>));
    cudaMemcpy(d_bitmask, &bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyHostToDevice);

    testSetAtomicSetBitOnAtomically<<<1, 64>>>(d_bitmask);
    cudaMemcpy(&bitmask, d_bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOn(i));
    }

    cudaFree(d_bitmask);
}

TEST_F(RS_StaticBitMaskTest, SetAtomicSetBitOffAtomically) {
    RS_StaticBitMask<64> bitmask;
    RS_StaticBitMask<64> *d_bitmask;

    cudaMalloc(&d_bitmask, sizeof(RS_StaticBitMask<64>));
    cudaMemcpy(d_bitmask, &bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyHostToDevice);

    // Set all bits to on initially
    for (int i = 0; i < 64; ++i) {
        bitmask.setOn(i);
    }
    cudaMemcpy(d_bitmask, &bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyHostToDevice);

    testSetAtomicSetBitOffAtomically<<<1, 64>>>(d_bitmask);
    cudaMemcpy(&bitmask, d_bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(bitmask.isOff(i));
    }

    cudaFree(d_bitmask);
}

TEST_F(RS_StaticBitMaskTest, SetAtomicConcurrentSetting) {
    RS_StaticBitMask<64> bitmask;
    RS_StaticBitMask<64> *d_bitmask;

    cudaMalloc(&d_bitmask, sizeof(RS_StaticBitMask<64>));
    cudaMemcpy(d_bitmask, &bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyHostToDevice);

    testSetAtomicConcurrentSetting<<<1, 64>>>(d_bitmask);
    cudaMemcpy(&bitmask, d_bitmask, sizeof(RS_StaticBitMask<64>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; ++i) {
        if (i % 2 == 0) {
            EXPECT_TRUE(bitmask.isOn(i));
        } else {
            EXPECT_TRUE(bitmask.isOff(i));
        }
    }

    cudaFree(d_bitmask);
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::setAllOn()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set All Bits On.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, SetAllOnSetAllBitsOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    EXPECT_TRUE(bitmask.isAllOn());
}

TEST_F(RS_StaticBitMaskTest, SetAllOnCheckUnusedBits) {
    RS_StaticBitMask<70> bitmask;
    bitmask.setAllOn();

    // Check that all used bits are on
    for (int i = 0; i < 70; ++i) {
        EXPECT_TRUE(bitmask.isOn(i));
    }
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::setAllOff()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set All Bits Off.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, SetAllOffSetAllBitsOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOff();
    EXPECT_TRUE(bitmask.isAllOff());
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::setAll()
 * Using 32-bit word size.

 * Test cases:
 * 1. Set All On with true.
 * 2. Set All Off with false.
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
 * Test cases for the RS_StaticBitMask::toggle()
 * Using 32-bit word size.

 * Test cases:
 * 1. Toggle Bit From Off to On.
 * 2. Toggle Bit From On to Off.
 * 3. Multiple Toggles.
 * 4. Out-of-Bounds Index.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, ToggleBitFromOffToOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOff(5);  // Ensure bit is off
    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_StaticBitMaskTest, ToggleBitFromOnToOff) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);  // Ensure bit is on
    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOff(5));
}

TEST_F(RS_StaticBitMaskTest, MultipleToggles) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOff(5);  // Ensure bit is off

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOff(5));

    bitmask.toggle(5);
    EXPECT_TRUE(bitmask.isOn(5));
}

TEST_F(RS_StaticBitMaskTest, ToggleOutOfBoundsIndex) {
    RS_StaticBitMask<64> bitmask;
    EXPECT_DEATH(bitmask.toggle(64), ".*");
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::operator&=()
 * Using 32-bit word size.

 * Test cases:
 * 1. Bitwise AND with Different Masks.
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
 * Test cases for the RS_StaticBitMask::operator|=()
 * Using 32-bit word size.

 * Test cases:
 * 1. Bitwise OR with Different Masks.
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
 * Test cases for the RS_StaticBitMask::operator-=()
 * Using 32-bit word size.

 * Test cases:
 * 1. Bitwise Difference with Partial Overlap.
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
 * Test cases for the RS_StaticBitMask::operator^=()
 * Using 32-bit word size.

 * Test cases:
 * 1. Bitwise XOR with Different Masks.
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

    EXPECT_TRUE(bitmask1.isOn(1));   // 1 ^ 0 = 1
    EXPECT_FALSE(bitmask1.isOn(3));  // 1 ^ 1 = 0
    EXPECT_FALSE(bitmask1.isOn(5));  // 1 ^ 1 = 0
    EXPECT_TRUE(bitmask1.isOn(7));   // 0 ^ 1 = 1
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::findFirst()
 * Using 32-bit word size.

 * Test cases:
 * 1. Find First On Bit.
 * 2. Find First Off Bit.
 * 3. No On Bits.
 * 4. Start Word Offset.
 * 5. FindWordNum Limitation.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, FindFirstOnBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex firstOn = bitmask.findFirst<true>();
    EXPECT_EQ(firstOn, 5);
}

TEST_F(RS_StaticBitMaskTest, FindFirstOffBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    RIndex firstOff = bitmask.findFirst<false>();
    EXPECT_EQ(firstOff, RS_NULL_INDEX);
}

TEST_F(RS_StaticBitMaskTest, FindFirstNoOnBits) {
    RS_StaticBitMask<64> bitmask;
    RIndex firstOn = bitmask.findFirst<true>();
    EXPECT_EQ(firstOn, RS_NULL_INDEX);
}

TEST_F(RS_StaticBitMaskTest, FindFirstStartWordOffset) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(42);                            // In the second word
    RIndex firstOn = bitmask.findFirst<true>(1);  // Start from the second word
    EXPECT_EQ(firstOn, 42);
}

TEST_F(RS_StaticBitMaskTest, FindFirstFindWordNumLimitation) {
    RS_StaticBitMask<119> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(70);                               // In the third word
    RIndex firstOn = bitmask.findFirst<true, 1>(1);  // Limit search to the second word
    EXPECT_EQ(firstOn, RS_NULL_INDEX);
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::findNext()
 * Using 32-bit word size.

 * Test cases:
 * 1. Find Next On Bit After Start Bit.
 * 2. Find Next Off Bit After Start Bit.
 * 3. Start Bit Is On.
 * 4. No Next On Bit.
 * 5. Out-of-Bounds Start Bit.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, FindNextOnBitAfterStartBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex nextOn = bitmask.findNext<true>(6);
    EXPECT_EQ(nextOn, 10);
}

TEST_F(RS_StaticBitMaskTest, FindNextOffBitAfterStartBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    bitmask.setOff(10);
    RIndex nextOff = bitmask.findNext<false>(5);
    EXPECT_EQ(nextOff, 10);
}

TEST_F(RS_StaticBitMaskTest, FindNextStartBitIsOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    RIndex nextOn = bitmask.findNext<true>(5);
    EXPECT_EQ(nextOn, 5);
}

TEST_F(RS_StaticBitMaskTest, FindNextNoNextOnBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    RIndex nextOn = bitmask.findNext<true>(6);
    EXPECT_EQ(nextOn, RS_NULL_INDEX);
}

TEST_F(RS_StaticBitMaskTest, FindNextOutOfBoundsStartBit) {
    RS_StaticBitMask<64> bitmask;
    RIndex nextOn = bitmask.findNext<true>(64);  // Out-of-bounds
    EXPECT_EQ(nextOn, RS_NULL_INDEX);
}

/*****************************************************************
 * Test cases for the RS_StaticBitMask::findPrev()
 * Using 32-bit word size.

 * Test cases:
 * 1. Find Previous On Bit Before Start Bit.
 * 2. Find Previous Off Bit Before Start Bit.
 * 3. Start Bit Is On.
 * 4. No Previous On Bit.
 * 5. Start Bit at Zero.
 * 6. Out-of-Bounds Start Bit.
 *****************************************************************/

TEST_F(RS_StaticBitMaskTest, FindPrevOnBitBeforeStartBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    bitmask.setOn(10);
    RIndex prevOn = bitmask.findPrev<true>(9);
    EXPECT_EQ(prevOn, 5);
}

TEST_F(RS_StaticBitMaskTest, FindPrevOffBitBeforeStartBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setAllOn();
    bitmask.setOff(10);
    RIndex prevOff = bitmask.findPrev<false>(10);
    EXPECT_EQ(prevOff, 10);
}

TEST_F(RS_StaticBitMaskTest, FindPrevStartBitIsOn) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    RIndex prevOn = bitmask.findPrev<true>(5);
    EXPECT_EQ(prevOn, 5);
}

TEST_F(RS_StaticBitMaskTest, FindPrevNoPreviousOnBit) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(5);
    RIndex prevOn = bitmask.findPrev<true>(4);
    EXPECT_EQ(prevOn, RS_NULL_INDEX);
}

TEST_F(RS_StaticBitMaskTest, FindPrevStartBitAtZero) {
    RS_StaticBitMask<64> bitmask;
    bitmask.setOn(0);
    RIndex prevOn = bitmask.findPrev<true>(0);
    EXPECT_EQ(prevOn, 0);
}

TEST_F(RS_StaticBitMaskTest, OutOfBoundsStartBit) {
    RS_StaticBitMask<64> bitmask;
    RIndex prevOn = bitmask.findPrev<true>(64);  // Out-of-bounds
    EXPECT_EQ(prevOn, RS_NULL_INDEX);
}