#include <gtest/gtest.h>

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/utils/rs_memory.h"

using namespace rise;
using namespace rise::device;

class RS_HostBlockTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize CUDA stream and memory provider
        host_provider = RS_MemoryCPUProvider(0);
    }

    void TearDown() override { host_provider.free(); }

    RS_MemoryCPUProvider host_provider;
};

class HostBlockTestFunction : public RS_ComputeFunction<HostBlockTestFunction> {
   public:
    using IsSyncBlockRequired = std::true_type;

    struct SharedMemLayout {
        int x;
    };

    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_Array<int, 1> input, RS_Array<int, 1> result) {
        int idx = index1D();
        sharedMemAt1D(&SharedMemLayout::x) = input[idx];
        syncThreadsInBlock();

        if (threadIdxX() < (blockDimX() / 2))
            sharedMemAt1D(&SharedMemLayout::x) =
                sharedMemAt1D(&SharedMemLayout::x) +
                sharedMemAt(&SharedMemLayout::x, blockDimX(), threadIdxX() + (blockDimX() / 2));

        syncThreadsInBlock();
        result[idx] = sharedMemAt1D(&SharedMemLayout::x);
    }
};

TEST_F(RS_HostBlockTest, HostBlockTest) {
    RS_Array<int, 1> input_array({32}, &host_provider);
    RS_Array<int, 1> result_array({32}, &host_provider);
    input_array.fill(1);
    result_array.fill(0);

    host_provider.run<HostBlockTestFunction>({32, 32}, input_array, result_array);
    for (int i = 0; i < 16; i++) {
        ASSERT_EQ(result_array[i], 2);
    }
    for (int i = 16; i < 32; i++) {
        ASSERT_EQ(result_array[i], 1);
    }
    input_array.free();
    result_array.free();
}

TEST_F(RS_HostBlockTest, DefaultProviderHostBlockTest) {
    RS_Array<int, 1> input_array({32}, &RS_MemoryDefaultProvider::getInstance());
    RS_Array<int, 1> result_array({32}, &RS_MemoryDefaultProvider::getInstance());
    input_array.fill(1);
    result_array.fill(0);

    host_provider.run<HostBlockTestFunction>({32, 32}, input_array, result_array);
    for (int i = 0; i < 16; i++) {
        ASSERT_EQ(result_array[i], 2);
    }
    for (int i = 16; i < 32; i++) {
        ASSERT_EQ(result_array[i], 1);
    }
    input_array.free();
    result_array.free();
}