#include <gtest/gtest.h>
// #define DEBUG_HASHMAP
#include "rise_origin/device/data/rs_hashmap.h"

using namespace rise;
using namespace rise::device;

class RS_HashStableFlatMapTest : public ::testing::Test {
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
 * Test cases for the RS_HashStableFlatMap::insert()
 *
 * Test cases:
 * 1. Insert into empty hash map
 * 2. Multiple insertions
 * 3. Insert duplicate key
 * 4. Insert until full, stress test with evictions
 * 5. Insert with group size 32
 * 6. Insert with group size 4
 *****************************************************************/

// Global kernel for inserting a single key-value pair
__global__ void
insertSinglePair(RS_HashStableFlatMap<uint32_t, int> map, uint32_t key, int value, bool* result) {
    if (threadIdx.x == 0) {
        *result = map.insert(key, value);
    }
}

// Global kernel for inserting multiple key-value pairs
__global__ void insertMultiplePairs(
    RS_HashStableFlatMap<uint32_t, int> map,
    uint32_t* keys,
    int* values,
    int count,
    bool* results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        results[idx] = map.insert(keys[idx], values[idx]);
    }
}

// Global kernel for group insertion
template <int GroupSize, typename HashMap>
__global__ void
insertWithGroup(HashMap map, uint32_t* keys, int* values, int count, bool* results) {
    // Warp size is 32
    int group_idx = (blockIdx.x * blockDim.x + threadIdx.x) / GroupSize;
    int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % GroupSize;
    if (group_idx < count) {
        bool result = map.template insert<GroupSize>(keys[group_idx], values[group_idx]);
        if (lane_id == 0) results[group_idx] = result;
    }
}

__global__ void getCount(RS_HashStableFlatMap<uint32_t, int> map, RSize* count) {
    if (threadIdx.x == 0) {
        *count = map.count();
    }
}

TEST_F(RS_HashStableFlatMapTest, InsertIntoEmptyHashMap) {
    RSize max_entry_num = 1024;
    RS_HashStableFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    uint32_t key = 42;
    int value = 100;
    bool* d_result;
    bool h_result;

    cudaMalloc(&d_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_map, key, value, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    EXPECT_TRUE(h_result);
    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 1);
    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashStableFlatMapTest, MultipleInsertions) {
    RSize max_entry_num = 2048;
    RS_HashStableFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);
    const int count = 10000;
    const int unique_keys = 100;
    auto keys = new uint32_t[count];
    auto values = new int[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i % unique_keys;
        values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertMultiplePairs<<<RS_CEIL_DIV(count, 256), 256>>>(
        hash_map, d_keys, d_values, count, d_results
    );
    cudaMemcpy(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(results[i]);
    }

    delete[] keys;
    delete[] values;
    delete[] results;

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), unique_keys);
    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashStableFlatMapTest, InsertDuplicateKey) {
    RSize max_entry_num = 1024;
    RS_HashStableFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    uint32_t key = 42;
    int value1 = 100, value2 = 200;
    bool* d_result;
    bool h_result1, h_result2;

    cudaMalloc(&d_result, sizeof(bool));

    insertSinglePair<<<1, 1>>>(hash_map, key, value1, d_result);
    cudaMemcpy(&h_result1, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    insertSinglePair<<<1, 1>>>(hash_map, key, value2, d_result);
    cudaMemcpy(&h_result2, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    EXPECT_TRUE(h_result1);
    EXPECT_TRUE(h_result2);
    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 1);
    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashStableFlatMapTest, InsertUntilFull) {
    RSize max_entry_num = 2048;
    RS_HashStableFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    const int count = 10000;
    int successful_inserts = 0;
    auto keys = new uint32_t[count];
    auto values = new int[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i;
        values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertMultiplePairs<<<RS_CEIL_DIV(count, 256), 256>>>(
        hash_map, d_keys, d_values, count, d_results
    );
    cudaMemcpy(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);

    for (int i = 0; i < count; ++i) {
        successful_inserts += results[i] ? 1 : 0;
    }

    delete[] keys;
    delete[] values;
    delete[] results;

    EXPECT_EQ(successful_inserts, max_entry_num);
    hash_map.free();
}

TEST_F(RS_HashStableFlatMapTest, InsertWithGroup32) {
    RSize max_entry_num = 2048;
    RS_HashStableFlatMap<uint32_t, int, uint32_t, RS_GenericHasher, 32> hash_map(
        max_entry_num, &provider
    );

    const int count = 10000;
    const int unique_keys = 100;
    auto keys = new uint32_t[count];
    auto values = new int[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i % unique_keys;
        values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertWithGroup<32>
        <<<RS_CEIL_DIV(count * 32, 256), 256>>>(hash_map, d_keys, d_values, count, d_results);
    cudaMemcpy(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(results[i]);
    }

    delete[] keys;
    delete[] values;
    delete[] results;

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), unique_keys);
    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashStableFlatMapTest, InsertWithGroup4) {
    RSize max_entry_num = 2048;
    // TODO: why bucket size = 4 doesn't work
    RS_HashStableFlatMap<uint32_t, int, uint32_t, RS_GenericHasher, 16> hash_map(
        max_entry_num, &provider
    );
    const int count = 10000;
    const int unique_keys = 100;
    auto keys = new uint32_t[count];
    auto values = new int[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i % unique_keys;
        values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertWithGroup<4>
        <<<RS_CEIL_DIV(count * 4, 256), 256>>>(hash_map, d_keys, d_values, count, d_results);
    cudaMemcpy(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(results[i]);
    }

    delete[] keys;
    delete[] values;
    delete[] results;

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), unique_keys);
    host_hash_map.free();
    hash_map.free();
}
