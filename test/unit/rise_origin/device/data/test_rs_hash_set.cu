#include <gtest/gtest.h>

// #define DEBUG_HASHMAP
#include "rise_origin/device/data/rs_hashmap.h"

using namespace rise;
using namespace rise::device;

class RS_HashSetTest : public ::testing::Test {
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
 * Test cases for the RS_HashSet constructor
 *
 * Test cases:
 * 1. Construct with valid parameters.
 * 2. Construct with zero capacity.
 * 3. Construct with large capacity.
 * 4. Default constructor.
 *****************************************************************/

TEST_F(RS_HashSetTest, ConstructWithValidParameters) {
    // Initialize with a valid maximum entry number
    RSize max_entry_num = 1024;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Calculate expected bucket number
    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_set.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_set.empty());
    EXPECT_EQ(
        hash_set.size(), expected_bucket_num_per_table * 2 * bucket_size
    );  // 2 tables, BucketSize = bucket_size
}

TEST_F(RS_HashSetTest, ConstructWithZeroCapacity) {
    // Initialize with zero maximum entry number
    RSize max_entry_num = 0;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Even with zero capacity, bucket_num_per_table should be at least BucketSize
    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_set.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_set.empty());
    EXPECT_EQ(hash_set.size(), expected_bucket_num_per_table * 2 * bucket_size);
}

TEST_F(RS_HashSetTest, ConstructWithLargeCapacity) {
    // Initialize with a large maximum entry number
    RSize max_entry_num = 1 << 20;  // 1 million entries
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_set.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_set.empty());
    EXPECT_EQ(hash_set.size(), expected_bucket_num_per_table * 2 * bucket_size);
}

TEST_F(RS_HashSetTest, DefaultConstructor) {
    // Use the default constructor
    RS_HashSet<uint32_t> hash_set;

    // Since no initialization, bucket_num_per_table should be zero
    EXPECT_EQ(hash_set.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_set.empty());
    EXPECT_EQ(hash_set.size(), 0);
}

/*****************************************************************
 * Test cases for the RS_HashSet::free()
 *
 * Test cases:
 * 1. Free after initialization.
 * 2. Multiple frees.
 *****************************************************************/
TEST_F(RS_HashSetTest, FreeAfterInitialization) {
    // Initialize with a valid maximum entry number
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Free the hash set
    hash_set.free();

    // Check that resources are deallocated
    EXPECT_EQ(hash_set.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_set.empty());
}

TEST_F(RS_HashSetTest, MultipleFrees) {
    // Initialize with a valid maximum entry number
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Free the hash set multiple times
    hash_set.free();
    hash_set.free();
    hash_set.free();

    // Check that resources are deallocated and no errors occur
    EXPECT_EQ(hash_set.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_set.empty());
}

/*****************************************************************
 * Test cases for the RS_HashSet::insert()
 *
 * Test cases:
 * 1. Insert into empty hash set
 * 2. Multiple insertions
 * 3. Insert duplicate key
 * 4. Insert until full, stress test with evictions
 * 5. Insert with group size 32
 * 6. Insert with group size 4
 *****************************************************************/

// Global kernel for inserting a single key
__global__ void insertSingleKey(RS_HashSet<uint32_t> set, uint32_t key, bool* result) {
    if (threadIdx.x == 0) {
        *result = set.insert(key);
    }
}

// Global kernel for inserting multiple keys
__global__ void
insertMultipleKeys(RS_HashSet<uint32_t> set, uint32_t* keys, int count, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        results[idx] = set.insert(keys[idx]);
    }
}

// Global kernel for group insertion
template <int GroupSize, typename HashSet>
__global__ void insertWithGroup(HashSet set, uint32_t* keys, int count, bool* results) {
    // Warp size is 32
    int group_idx = (blockIdx.x * blockDim.x + threadIdx.x) / GroupSize;
    int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % GroupSize;
    if (group_idx < count) {
        bool result = set.insert<GroupSize>(keys[group_idx]);
        if (lane_id == 0) results[group_idx] = result;
    }
}

__global__ void getCount(RS_HashSet<uint32_t> set, RSize* count) {
    if (threadIdx.x == 0) {
        *count = set.count();
    }
}

TEST_F(RS_HashSetTest, InsertIntoEmptyHashSet) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    uint32_t key = 42;
    bool* d_result;
    bool h_result;

    cudaMalloc(&d_result, sizeof(bool));
    insertSingleKey<<<1, 1>>>(hash_set, key, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    EXPECT_TRUE(h_result);
    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 1);
    host_hash_set.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, MultipleInsertions) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);
    const int count = 10000;
    const int unique_keys = 1000;
    auto keys = new uint32_t[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i % unique_keys;
    }

    uint32_t* d_keys;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    insertMultipleKeys<<<RS_CEIL_DIV(count, 256), 256>>>(hash_set, d_keys, count, d_results);
    cudaMemcpy(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_results);

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(results[i]);
    }

    delete[] keys;
    delete[] results;

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), unique_keys);
    host_hash_set.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, InsertDuplicateKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    uint32_t key = 42;
    bool* d_result;
    bool h_result1, h_result2;

    cudaMalloc(&d_result, sizeof(bool));

    insertSingleKey<<<1, 1>>>(hash_set, key, d_result);
    cudaMemcpy(&h_result1, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    insertSingleKey<<<1, 1>>>(hash_set, key, d_result);
    cudaMemcpy(&h_result2, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    EXPECT_TRUE(h_result1);
    EXPECT_TRUE(h_result2);
    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 1);
    host_hash_set.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, InsertUntilFull) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    const int count = 10000;
    int successful_inserts = 0;
    auto keys = new uint32_t[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i;
    }

    uint32_t* d_keys;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    insertMultipleKeys<<<RS_CEIL_DIV(count, 256), 256>>>(hash_set, d_keys, count, d_results);
    cudaMemcpy(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_results);

    for (int i = 0; i < count; ++i) {
        successful_inserts += results[i] ? 1 : 0;
    }

    delete[] keys;
    delete[] results;

    EXPECT_EQ(successful_inserts, max_entry_num);
    hash_set.free();
}

TEST_F(RS_HashSetTest, InsertWithGroup32) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    const int count = 10000;
    const int unique_keys = 1000;
    auto keys = new uint32_t[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i % unique_keys;
    }

    uint32_t* d_keys;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    insertWithGroup<32><<<RS_CEIL_DIV(count * 32, 256), 256>>>(hash_set, d_keys, count, d_results);
    cudaMemcpy(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_results);

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(results[i]);
    }

    delete[] keys;
    delete[] results;

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), unique_keys);
    host_hash_set.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, InsertWithGroup4) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);
    const int count = 10000;
    const int unique_keys = 1000;
    auto keys = new uint32_t[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i % unique_keys;
    }

    uint32_t* d_keys;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    insertWithGroup<4><<<RS_CEIL_DIV(count * 4, 256), 256>>>(hash_set, d_keys, count, d_results);
    cudaMemcpy(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_results);

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(results[i]);
    }

    delete[] keys;
    delete[] results;

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), unique_keys);
    host_hash_set.free();
    hash_set.free();
}

/*****************************************************************
 * Test cases for the RS_HashSet::erase()
 *
 * Test cases:
 * 1. Erase existing key.
 * 2. Erase non-existent key.
 * 3. Erase from empty hash set.
 * 4. Erase all keys.
 * 5. Erase with group size 32.
 * 6. Erase with group size 4.
 *****************************************************************/

__global__ void eraseSingleKey(RS_HashSet<uint32_t> set, uint32_t key, bool* result) {
    if (threadIdx.x == 0) {
        *result = set.erase(key);
    }
}
__global__ void
eraseMultipleKeys(RS_HashSet<uint32_t> set, uint32_t* keys, int count, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        results[idx] = set.erase(keys[idx]);
        if (!results[idx]) printf("Erase %d failed\n", keys[idx]);
    }
}
template <int GroupSize, typename HashSet>
__global__ void eraseWithGroup(HashSet set, uint32_t* keys, int count, bool* results) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) / GroupSize;
    int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % GroupSize;
    if (idx < count) {
        bool result = set.erase<GroupSize>(keys[idx]);
        if (lane_id == 0) results[idx] = result;
    }
}

TEST_F(RS_HashSetTest, EraseExistingKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Insert a key
    uint32_t key = 42;
    bool* d_insert_result;
    cudaMalloc(&d_insert_result, sizeof(bool));
    insertSingleKey<<<1, 1>>>(hash_set, key, d_insert_result);
    cudaFree(d_insert_result);

    // Erase the key
    bool* d_erase_result;
    bool h_erase_result;
    cudaMalloc(&d_erase_result, sizeof(bool));
    eraseSingleKey<<<1, 1>>>(hash_set, key, d_erase_result);
    cudaMemcpy(&h_erase_result, d_erase_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_erase_result);

    EXPECT_TRUE(h_erase_result);

    // Verify the key is no longer in the hash set
    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);
    EXPECT_FALSE(host_hash_set.contains(key));

    host_hash_set.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, EraseNonExistentKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Attempt to erase a key that doesn't exist
    uint32_t key = 12345;
    bool* d_erase_result;
    bool h_erase_result;
    cudaMalloc(&d_erase_result, sizeof(bool));
    eraseSingleKey<<<1, 1>>>(hash_set, key, d_erase_result);
    cudaMemcpy(&h_erase_result, d_erase_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_erase_result);

    EXPECT_FALSE(h_erase_result);

    // Verify the hash set is still empty
    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);

    host_hash_set.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, EraseFromEmptyHashSet) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Attempt to erase a key from an empty hash set
    uint32_t key = 0;
    bool* d_erase_result;
    bool h_erase_result;
    cudaMalloc(&d_erase_result, sizeof(bool));
    eraseSingleKey<<<1, 1>>>(hash_set, key, d_erase_result);
    cudaMemcpy(&h_erase_result, d_erase_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_erase_result);

    EXPECT_FALSE(h_erase_result);

    // Verify the hash set is still empty
    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);

    host_hash_set.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, EraseAllKeys) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Insert multiple keys
    const int count = 100;
    uint32_t* h_keys = new uint32_t[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
    }

    uint32_t* d_keys;
    bool* d_insert_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_insert_results, count * sizeof(bool));
    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    insertMultipleKeys<<<RS_CEIL_DIV(count, 256), 256>>>(hash_set, d_keys, count, d_insert_results);

    // Erase all keys
    bool* d_erase_results;
    bool* h_erase_results = new bool[count];
    cudaMalloc(&d_erase_results, count * sizeof(bool));
    eraseMultipleKeys<<<RS_CEIL_DIV(count, 256), 256>>>(hash_set, d_keys, count, d_erase_results);

    cudaMemcpy(h_erase_results, d_erase_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_insert_results);
    cudaFree(d_erase_results);

    // Verify all erase operations succeeded
    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_erase_results[i]);
    }

    // Verify the hash set is empty
    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);

    // Clean up
    delete[] h_keys;
    delete[] h_erase_results;

    host_hash_set.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, EraseWithGroup32) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t, uint32_t, RS_GenericHasher, 32> hash_set(max_entry_num, &provider);

    const int group_size = 32;  // Cooperative group size

    // Insert multiple keys
    const int count = 100;
    uint32_t* h_keys = new uint32_t[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
    }

    uint32_t* d_keys;
    bool* d_insert_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_insert_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    insertWithGroup<group_size>
        <<<RS_CEIL_DIV(group_size * count, 256), 256>>>(hash_set, d_keys, count, d_insert_results);

    // Erase all keys using group erase
    bool* d_erase_results;
    bool* h_erase_results = new bool[count];
    cudaMalloc(&d_erase_results, count * sizeof(bool));

    eraseWithGroup<group_size>
        <<<RS_CEIL_DIV(group_size * count, 256), 256>>>(hash_set, d_keys, count, d_erase_results);

    cudaMemcpy(h_erase_results, d_erase_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_insert_results);
    cudaFree(d_erase_results);

    // Verify all erase operations succeeded
    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_erase_results[i]);
    }

    // Verify the hash set is empty
    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);

    // Clean up
    delete[] h_keys;
    delete[] h_erase_results;

    host_hash_set.free();
    hash_set.free();
}

/*****************************************************************
 * Test cases for the RS_HashSet::contains()
 *
 * Test cases:
 * 1. Contains existing key.
 * 2. Contains non-existent key.
 * 3. Contains in empty hash set.
 * 4. Contains after erase.
 * 5. Contains with group size 32.
 * 6. Contains with group size 4.
 *****************************************************************/

__global__ void containsSingleKey(RS_HashSet<uint32_t> set, uint32_t key, bool* found) {
    if (threadIdx.x == 0) {
        *found = set.contains(key);
    }
}
__global__ void
containsMultipleKeys(RS_HashSet<uint32_t> set, uint32_t* keys, bool* found, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        found[idx] = set.contains(keys[idx]);
    }
}
template <int GroupSize>
__global__ void
containsWithGroup(RS_HashSet<uint32_t> set, uint32_t* keys, bool* found, int count) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) / GroupSize;
    int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % GroupSize;
    if (idx < count) {
        bool result = set.contains<GroupSize>(keys[idx]);
        if (lane_id == 0) found[idx] = result;
    }
}
TEST_F(RS_HashSetTest, ContainsExistingKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Insert a key
    uint32_t key = 42;
    bool* d_insert_result;
    cudaMalloc(&d_insert_result, sizeof(bool));
    insertSingleKey<<<1, 1>>>(hash_set, key, d_insert_result);
    cudaFree(d_insert_result);

    // Check if the key is contained
    bool* d_found;
    bool h_found;
    cudaMalloc(&d_found, sizeof(bool));

    containsSingleKey<<<1, 1>>>(hash_set, key, d_found);

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_found);

    EXPECT_TRUE(h_found);

    // Clean up
    hash_set.free();
}
TEST_F(RS_HashSetTest, ContainsNonExistentKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Attempt to check a key that doesn't exist
    uint32_t key = 12345;
    bool* d_found;
    bool h_found;
    cudaMalloc(&d_found, sizeof(bool));

    containsSingleKey<<<1, 1>>>(hash_set, key, d_found);

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_found);

    EXPECT_FALSE(h_found);

    // Clean up
    hash_set.free();
}

TEST_F(RS_HashSetTest, ContainsInEmptyHashSet) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Attempt to check any key in an empty hash set
    uint32_t key = 0;
    bool* d_found;
    bool h_found;
    cudaMalloc(&d_found, sizeof(bool));

    containsSingleKey<<<1, 1>>>(hash_set, key, d_found);

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_found);

    EXPECT_FALSE(h_found);

    // Clean up
    hash_set.free();
}

TEST_F(RS_HashSetTest, ContainsAfterErase) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    // Insert a key
    uint32_t key = 99;
    bool* d_insert_result;
    cudaMalloc(&d_insert_result, sizeof(bool));
    insertSingleKey<<<1, 1>>>(hash_set, key, d_insert_result);
    cudaFree(d_insert_result);

    // Erase the key
    bool* d_erase_result;
    cudaMalloc(&d_erase_result, sizeof(bool));
    eraseSingleKey<<<1, 1>>>(hash_set, key, d_erase_result);
    cudaFree(d_erase_result);

    // Attempt to check if the key is contained after erasure
    bool* d_found;
    bool h_found;
    cudaMalloc(&d_found, sizeof(bool));

    containsSingleKey<<<1, 1>>>(hash_set, key, d_found);

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_found);

    EXPECT_FALSE(h_found);

    // Clean up
    hash_set.free();
}

TEST_F(RS_HashSetTest, ContainsWithGroup32) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    const int group_size = 32;  // Cooperative group size

    // Insert multiple keys
    const int count = 64;  // Should be multiple of group size
    uint32_t* h_keys = new uint32_t[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
    }

    uint32_t* d_keys;
    bool* d_insert_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_insert_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    insertWithGroup<group_size>
        <<<RS_CEIL_DIV(group_size * count, 256), 256>>>(hash_set, d_keys, count, d_insert_results);

    // Attempt to check if the keys are contained using group contains
    bool* d_found_flags;
    bool* h_found_flags = new bool[count];
    cudaMalloc(&d_found_flags, count * sizeof(bool));

    containsWithGroup<group_size>
        <<<RS_CEIL_DIV(group_size * count, 256), 256>>>(hash_set, d_keys, d_found_flags, count);

    cudaMemcpy(h_found_flags, d_found_flags, count * sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify all contains operations succeeded
    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_found_flags[i]);
    }

    // Clean up
    cudaFree(d_keys);
    cudaFree(d_insert_results);
    cudaFree(d_found_flags);

    delete[] h_keys;
    delete[] h_found_flags;

    hash_set.free();
}

TEST_F(RS_HashSetTest, ContainsWithGroup4) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    const int group_size = 4;  // Cooperative group size

    // Insert multiple keys
    const int count = 64;  // Should be multiple of group size
    uint32_t* h_keys = new uint32_t[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
    }

    uint32_t* d_keys;
    bool* d_insert_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_insert_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    insertWithGroup<group_size>
        <<<RS_CEIL_DIV(group_size * count, 256), 256>>>(hash_set, d_keys, count, d_insert_results);

    // Attempt to check if the keys are contained using group contains
    bool* d_found_flags;
    bool* h_found_flags = new bool[count];
    cudaMalloc(&d_found_flags, count * sizeof(bool));

    containsWithGroup<group_size>
        <<<RS_CEIL_DIV(group_size * count, 256), 256>>>(hash_set, d_keys, d_found_flags, count);

    cudaMemcpy(h_found_flags, d_found_flags, count * sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify all contains operations succeeded
    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_found_flags[i]);
    }

    // Clean up
    cudaFree(d_keys);
    cudaFree(d_insert_results);
    cudaFree(d_found_flags);

    delete[] h_keys;
    delete[] h_found_flags;

    hash_set.free();
}

/*****************************************************************
 * Test cases for the RS_HashSet::resize()
 *
 * Test cases:
 * 1. Insert, resize and find existing key.
 *****************************************************************/
TEST_F(RS_HashSetTest, ResizeAndFindKey) {
    RSize initial_max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(initial_max_entry_num, &provider);

    // Insert a key into the hash set
    uint32_t key = 42;
    bool* d_insert_result;
    cudaMalloc(&d_insert_result, sizeof(bool));
    insertSingleKey<<<1, 1>>>(hash_set, key, d_insert_result);

    // Resize the hash set to a larger capacity
    RSize new_max_entry_num = 2048;  // New size for the hash set
    hash_set.resize(new_max_entry_num);

    // Attempt to check if the key exists after resizing
    bool* d_found;
    cudaMalloc(&d_found, sizeof(bool));

    containsSingleKey<<<1, 1>>>(hash_set, key, d_found);

    // Copy results back to host for verification
    bool h_found;
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify the key was found
    EXPECT_TRUE(h_found);

    // Cleanup
    cudaFree(d_insert_result);
    cudaFree(d_found);
    hash_set.free();
}