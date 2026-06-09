#include <gtest/gtest.h>
// #define DEBUG_HASHMAP
#include "rise_origin/device/data/rs_hashmap.h"

using namespace rise;
using namespace rise::device;

class RS_HashFlatMapTest : public ::testing::Test {
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
 * Test cases for the RS_HashFlatMap constructor
 *
 * Test cases:
 * 1. Construct with valid parameters.
 * 2. Construct with zero capacity.
 * 3. Construct with large capacity.
 * 4. Default constructor.
 *****************************************************************/

TEST_F(RS_HashFlatMapTest, ConstructWithValidParameters) {
    // Initialize with a valid maximum entry number
    RSize max_entry_num = 1024;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Calculate expected bucket number
    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(
        hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size
    );  // 2 tables, BucketSize = 32
}

TEST_F(RS_HashFlatMapTest, ConstructWithZeroCapacity) {
    // Initialize with zero maximum entry number
    RSize max_entry_num = 0;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Even with zero capacity, bucket_num_per_table should be at least BucketSize
    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
}

TEST_F(RS_HashFlatMapTest, ConstructWithLargeCapacity) {
    // Initialize with a large maximum entry number
    RSize max_entry_num = 1 << 20;  // 1 million entries
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
}

TEST_F(RS_HashFlatMapTest, DefaultConstructor) {
    // Use the default constructor
    RS_HashFlatMap<uint32_t, int> hash_map;

    // Since no initialization, bucket_num_per_table should be zero
    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), 0);
}

/*****************************************************************
 * Test cases for the RS_HashFlatMap::free()
 *
 * Test cases:
 * 1. Free after initialization.
 * 2. Multiple frees.
 *****************************************************************/
TEST_F(RS_HashFlatMapTest, FreeAfterInitialization) {
    // Initialize with a valid maximum entry number
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Free the hash map
    hash_map.free();

    // Check that resources are deallocated
    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
}

TEST_F(RS_HashFlatMapTest, MultipleFrees) {
    // Initialize with a valid maximum entry number
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Free the hash map multiple times
    hash_map.free();
    hash_map.free();
    hash_map.free();

    // Check that resources are deallocated and no errors occur
    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
}

/*****************************************************************
 * Test cases for the RS_HashFlatMap::insert()
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
insertSinglePair(RS_HashFlatMap<uint32_t, int> map, uint32_t key, int value, bool* result) {
    if (threadIdx.x == 0) {
        *result = map.insert(key, value);
    }
}

// Global kernel for inserting multiple key-value pairs
__global__ void insertMultiplePairs(
    RS_HashFlatMap<uint32_t, int> map,
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
template <int GroupSize>
__global__ void insertWithGroup(
    RS_HashFlatMap<uint32_t, int> map,
    uint32_t* keys,
    int* values,
    int count,
    bool* results
) {
    // Warp size is 32
    int group_idx = (blockIdx.x * blockDim.x + threadIdx.x) / GroupSize;
    int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % GroupSize;
    if (group_idx < count) {
        bool result = map.insert<GroupSize>(keys[group_idx], values[group_idx]);
        if (lane_id == 0) results[group_idx] = result;
    }
}

__global__ void getCount(RS_HashFlatMap<uint32_t, int> map, RSize* count) {
    if (threadIdx.x == 0) {
        *count = map.count();
    }
}

TEST_F(RS_HashFlatMapTest, InsertIntoEmptyHashMap) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

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

TEST_F(RS_HashFlatMapTest, MultipleInsertions) {
    RSize max_entry_num = 2048;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);
    const int count = 10000;
    const int unique_keys = 1000;
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

TEST_F(RS_HashFlatMapTest, InsertDuplicateKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

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

TEST_F(RS_HashFlatMapTest, InsertUntilFull) {
    RSize max_entry_num = 2048;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

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

TEST_F(RS_HashFlatMapTest, InsertWithGroup32) {
    RSize max_entry_num = 16384;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

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

TEST_F(RS_HashFlatMapTest, InsertWithGroup4) {
    RSize max_entry_num = 2048;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);
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

/*****************************************************************
 * Test cases for the RS_HashFlatMap::erase()
 *
 * Test cases:
 * 1. Erase existing key.
 * 2. Erase non-existent key.
 * 3. Erase from empty hash map.
 * 4. Erase all keys.
 * 5. Erase with group size 32.
 * 6. Erase with group size 4.
 *****************************************************************/

__global__ void eraseSingleKey(RS_HashFlatMap<uint32_t, int> map, uint32_t key, bool* result) {
    if (threadIdx.x == 0) {
        *result = map.erase(key);
    }
}
__global__ void
eraseMultipleKeys(RS_HashFlatMap<uint32_t, int> map, uint32_t* keys, int count, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        results[idx] = map.erase(keys[idx]);
        if (not results[idx]) printf("Erase %d failed\n", keys[idx]);
    }
}
template <int GroupSize>
__global__ void
eraseWithGroup(RS_HashFlatMap<uint32_t, int> map, uint32_t* keys, int count, bool* results) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) / GroupSize;
    int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % GroupSize;
    if (idx < count) {
        bool result = map.erase<GroupSize>(keys[idx]);
        if (lane_id == 0) results[idx] = result;
    }
}

TEST_F(RS_HashFlatMapTest, EraseExistingKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Insert a key-value pair
    uint32_t key = 42;
    int value = 100;
    bool* d_insert_result;
    cudaMalloc(&d_insert_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_map, key, value, d_insert_result);
    cudaFree(d_insert_result);

    // Erase the key
    bool* d_erase_result;
    bool h_erase_result;
    cudaMalloc(&d_erase_result, sizeof(bool));
    eraseSingleKey<<<1, 1>>>(hash_map, key, d_erase_result);
    cudaMemcpy(&h_erase_result, d_erase_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_erase_result);

    EXPECT_TRUE(h_erase_result);

    // Verify the key is no longer in the hash map
    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);
    EXPECT_EQ(host_hash_map.find(key), nullptr);

    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, EraseNonExistentKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Attempt to erase a key that doesn't exist
    uint32_t key = 12345;
    bool* d_erase_result;
    bool h_erase_result;
    cudaMalloc(&d_erase_result, sizeof(bool));
    eraseSingleKey<<<1, 1>>>(hash_map, key, d_erase_result);
    cudaMemcpy(&h_erase_result, d_erase_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_erase_result);

    EXPECT_FALSE(h_erase_result);

    // Verify the hash map is still empty
    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);

    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, EraseFromEmptyHashMap) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Attempt to erase a key from an empty hash map
    uint32_t key = 0;
    bool* d_erase_result;
    bool h_erase_result;
    cudaMalloc(&d_erase_result, sizeof(bool));
    eraseSingleKey<<<1, 1>>>(hash_map, key, d_erase_result);
    cudaMemcpy(&h_erase_result, d_erase_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_erase_result);

    EXPECT_FALSE(h_erase_result);

    // Verify the hash map is still empty
    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);

    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, EraseAllKeys) {
    RSize max_entry_num = 2048;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Insert multiple key-value pairs
    const int count = 100;
    uint32_t* h_keys = new uint32_t[count];
    int* h_values = new int[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_insert_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_insert_results, count * sizeof(bool));
    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertMultiplePairs<<<RS_CEIL_DIV(count, 256), 256>>>(
        hash_map, d_keys, d_values, count, d_insert_results
    );

    // auto h_hash_map = hash_map.move(&host_provider);
    // for (int t = 0; t < 2; t++) {
    //     std::cout << "table " << t << std::endl;
    //     for (int b = 0; b < h_hash_map.bucket_num_per_table; b++) {
    //         std::cout << "bucket " << b << ":";
    //         for (int e = 0; e < 32; e++)
    //             std::cout << h_hash_map.tables[t].bucket_keys[b].keys[e] << " ";
    //         std::cout << std::endl;
    //     }
    // }

    // Erase all keys
    bool* d_erase_results;
    bool* h_erase_results = new bool[count];
    cudaMalloc(&d_erase_results, count * sizeof(bool));
    eraseMultipleKeys<<<RS_CEIL_DIV(count, 256), 256>>>(hash_map, d_keys, count, d_erase_results);

    cudaMemcpy(h_erase_results, d_erase_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_insert_results);
    cudaFree(d_erase_results);

    // Verify all erase operations succeeded
    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_erase_results[i]);
    }

    // Verify the hash map is empty
    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);

    // Clean up
    delete[] h_keys;
    delete[] h_values;
    delete[] h_erase_results;

    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, EraseWithGroup32) {
    RSize max_entry_num = 16384;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    const int group_size = 32;  // Cooperative group size

    // Insert multiple key-value pairs
    const int count = 100;
    uint32_t* h_keys = new uint32_t[count];
    int* h_values = new int[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_insert_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_insert_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertWithGroup<group_size><<<RS_CEIL_DIV(group_size * count, 256), 256>>>(
        hash_map, d_keys, d_values, count, d_insert_results
    );

    // Erase all keys using group erase
    bool* d_erase_results;
    bool* h_erase_results = new bool[count];
    cudaMalloc(&d_erase_results, count * sizeof(bool));

    eraseWithGroup<group_size>
        <<<RS_CEIL_DIV(group_size * count, 256), 256>>>(hash_map, d_keys, count, d_erase_results);

    cudaMemcpy(h_erase_results, d_erase_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_insert_results);
    cudaFree(d_erase_results);

    // Verify all erase operations succeeded
    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_erase_results[i]);
    }

    // Verify the hash map is empty
    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);

    // Clean up
    delete[] h_keys;
    delete[] h_values;
    delete[] h_erase_results;

    host_hash_map.free();
    hash_map.free();
}

/*****************************************************************
 * Test cases for the RS_HashFlatMap::find()
 *
 * Test cases:
 * 1. Find existing key.
 * 2. Find non-existent key.
 * 3. Find in empty hash map.
 * 4. Find after erase.
 * 5. Find with group size 32.
 * 6. Find with group size 4.
 *****************************************************************/
__global__ void
findSingleKey(RS_HashFlatMap<uint32_t, int> map, uint32_t key, int* value, bool* found) {
    if (threadIdx.x == 0) {
        int* ptr = map.find(key);
        if (ptr) {
            *value = *ptr;
            *found = true;
        } else {
            *found = false;
        }
    }
}
__global__ void findMultipleKeys(
    RS_HashFlatMap<uint32_t, int> map,
    uint32_t* keys,
    int* values,
    bool* found,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int* ptr = map.find(keys[idx]);
        if (ptr) {
            values[idx] = *ptr;
            found[idx] = true;
        } else {
            found[idx] = false;
        }
    }
}
template <int GroupSize>
__global__ void findWithGroup(
    RS_HashFlatMap<uint32_t, int> map,
    uint32_t* keys,
    int* values,
    bool* found,
    int count
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) / GroupSize;
    int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % GroupSize;
    if (idx < count) {
        int* ptr = map.find<GroupSize>(keys[idx]);
        if (ptr && lane_id == 0) {
            values[idx] = *ptr;
            found[idx] = true;
        } else if (lane_id == 0) {
            found[idx] = false;
        }
    }
}
TEST_F(RS_HashFlatMapTest, FindExistingKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Insert a key-value pair
    uint32_t key = 42;
    int value = 100;
    bool* d_insert_result;
    cudaMalloc(&d_insert_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_map, key, value, d_insert_result);
    cudaFree(d_insert_result);

    // Find the key
    int* d_value;
    bool* d_found;
    int h_value;
    bool h_found;
    cudaMalloc(&d_value, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    findSingleKey<<<1, 1>>>(hash_map, key, d_value, d_found);

    cudaMemcpy(&h_value, d_value, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_value);
    cudaFree(d_found);

    EXPECT_TRUE(h_found);
    EXPECT_EQ(h_value, value);

    // Clean up
    hash_map.free();
}
TEST_F(RS_HashFlatMapTest, FindNonExistentKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Attempt to find a key that doesn't exist
    uint32_t key = 12345;
    int* d_value;
    bool* d_found;
    bool h_found;
    cudaMalloc(&d_value, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    findSingleKey<<<1, 1>>>(hash_map, key, d_value, d_found);

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_value);
    cudaFree(d_found);

    EXPECT_FALSE(h_found);

    // Clean up
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, FindInEmptyHashMap) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Attempt to find any key in an empty hash map
    uint32_t key = 0;
    int* d_value;
    bool* d_found;
    bool h_found;
    cudaMalloc(&d_value, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    findSingleKey<<<1, 1>>>(hash_map, key, d_value, d_found);

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_value);
    cudaFree(d_found);

    EXPECT_FALSE(h_found);

    // Clean up
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, FindAfterErase) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    // Insert a key-value pair
    uint32_t key = 99;
    int value = 999;
    bool* d_insert_result;
    cudaMalloc(&d_insert_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_map, key, value, d_insert_result);
    cudaFree(d_insert_result);

    // Erase the key
    bool* d_erase_result;
    cudaMalloc(&d_erase_result, sizeof(bool));
    eraseSingleKey<<<1, 1>>>(hash_map, key, d_erase_result);
    cudaFree(d_erase_result);

    // Attempt to find the key after erasure
    int* d_value;
    bool* d_found;
    bool h_found;
    cudaMalloc(&d_value, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    findSingleKey<<<1, 1>>>(hash_map, key, d_value, d_found);

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_value);
    cudaFree(d_found);

    EXPECT_FALSE(h_found);

    // Clean up
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, FindWithGroup32) {
    RSize max_entry_num = 16384;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    const int group_size = 32;  // Cooperative group size

    // Insert multiple key-value pairs
    const int count = 64;  // Should be multiple of group size
    uint32_t* h_keys = new uint32_t[count];
    int* h_values = new int[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_insert_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_insert_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertWithGroup<group_size><<<RS_CEIL_DIV(group_size * count, 256), 256>>>(
        hash_map, d_keys, d_values, count, d_insert_results
    );

    // Attempt to find the keys using group find
    int* d_found_values;
    bool* d_found_flags;
    int* h_found_values = new int[count];
    bool* h_found_flags = new bool[count];
    cudaMalloc(&d_found_values, count * sizeof(int));
    cudaMalloc(&d_found_flags, count * sizeof(bool));

    findWithGroup<group_size><<<RS_CEIL_DIV(group_size * count, 256), 256>>>(
        hash_map, d_keys, d_found_values, d_found_flags, count
    );

    cudaMemcpy(h_found_values, d_found_values, count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_found_flags, d_found_flags, count * sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify all find operations succeeded and values are correct
    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_found_flags[i]);
        EXPECT_EQ(h_found_values[i], h_values[i]);
    }

    // Clean up
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_insert_results);
    cudaFree(d_found_values);
    cudaFree(d_found_flags);

    delete[] h_keys;
    delete[] h_values;
    delete[] h_found_values;
    delete[] h_found_flags;

    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, FindWithGroup4) {
    RSize max_entry_num = 2048;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    const int group_size = 4;  // Cooperative group size

    // Insert multiple key-value pairs
    const int count = 64;  // Should be multiple of group size
    uint32_t* h_keys = new uint32_t[count];
    int* h_values = new int[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_insert_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_insert_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertWithGroup<group_size><<<RS_CEIL_DIV(group_size * count, 256), 256>>>(
        hash_map, d_keys, d_values, count, d_insert_results
    );

    // Attempt to find the keys using group find
    int* d_found_values;
    bool* d_found_flags;
    int* h_found_values = new int[count];
    bool* h_found_flags = new bool[count];
    cudaMalloc(&d_found_values, count * sizeof(int));
    cudaMalloc(&d_found_flags, count * sizeof(bool));

    findWithGroup<group_size><<<RS_CEIL_DIV(group_size * count, 256), 256>>>(
        hash_map, d_keys, d_found_values, d_found_flags, count
    );

    cudaMemcpy(h_found_values, d_found_values, count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_found_flags, d_found_flags, count * sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify all find operations succeeded and values are correct
    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_found_flags[i]);
        EXPECT_EQ(h_found_values[i], h_values[i]);
    }

    // Clean up
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_insert_results);
    cudaFree(d_found_values);
    cudaFree(d_found_flags);

    delete[] h_keys;
    delete[] h_values;
    delete[] h_found_values;
    delete[] h_found_flags;

    hash_map.free();
}

/*****************************************************************
 * Test cases for the RS_HashFlatMap::resize()
 *
 * Test cases:
 * 1. Insert, resize and find existing key.
 *****************************************************************/
TEST_F(RS_HashFlatMapTest, ResizeAndFindKey) {
    RSize initial_max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(initial_max_entry_num, &provider);

    // Insert a key-value pair into the hash map
    uint32_t key = 42;
    int initial_value = 100;
    bool* d_insert_result;
    cudaMalloc(&d_insert_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_map, key, initial_value, d_insert_result);

    // Resize the hash map to a larger capacity
    RSize new_max_entry_num = 2048;  // New size for the hash map
    hash_map.resize(new_max_entry_num);

    // Attempt to find the key after resizing
    int* d_found_value;
    bool* d_found;
    cudaMalloc(&d_found_value, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    findSingleKey<<<1, 1>>>(hash_map, key, d_found_value, d_found);

    // Copy results back to host for verification
    int h_found_value;
    bool h_found;
    cudaMemcpy(&h_found_value, d_found_value, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify the key was found and the value is correct
    EXPECT_TRUE(h_found);
    EXPECT_EQ(h_found_value, initial_value);

    // Cleanup
    cudaFree(d_insert_result);
    cudaFree(d_found_value);
    cudaFree(d_found);
    hash_map.free();
}