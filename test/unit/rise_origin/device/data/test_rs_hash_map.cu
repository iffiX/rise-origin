#include <gtest/gtest.h>
// #define DEBUG_HASHMAP
#include "rise_origin/device/data/rs_hashmap.h"

using namespace rise;
using namespace rise::device;

struct LargeValue {
    int data[3];  // 12 bytes
};

class RS_HashMapTest : public ::testing::Test {
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
 * Test cases for the RS_HashMap constructor
 *
 * Test cases:
 * 1. Construct with valid parameters.
 * 2. Construct with zero capacity.
 * 3. Construct with large capacity.
 * 4. Default constructor.
 *****************************************************************/

TEST_F(RS_HashMapTest, ConstructWithValidParameters) {
    // Initialize with a valid maximum entry number
    RSize max_entry_num = 1024;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    // Calculate expected bucket number
    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(
        hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size
    );  // 2 tables, BucketSize = 32
}

TEST_F(RS_HashMapTest, ConstructWithZeroCapacity) {
    // Initialize with zero maximum entry number
    RSize max_entry_num = 0;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    // Even with zero capacity, bucket_num_per_table should be at least BucketSize
    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
}

TEST_F(RS_HashMapTest, ConstructWithLargeCapacity) {
    // Initialize with a large maximum entry number
    RSize max_entry_num = 1 << 20;  // 1 million entries
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
}

TEST_F(RS_HashMapTest, DefaultConstructor) {
    // Use the default constructor
    RS_HashMap<uint32_t, LargeValue> hash_map;

    // Since no initialization, bucket_num_per_table should be zero
    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), 0);
}

/*****************************************************************
 * Test cases for the RS_HashMap::free()
 *
 * Test cases:
 * 1. Free after initialization.
 * 2. Multiple frees.
 *****************************************************************/
TEST_F(RS_HashMapTest, FreeAfterInitialization) {
    // Initialize with a valid maximum entry number
    RSize max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    // Free the hash map
    hash_map.free();

    // Check that resources are deallocated
    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
}

TEST_F(RS_HashMapTest, MultipleFrees) {
    // Initialize with a valid maximum entry number
    RSize max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    // Free the hash map multiple times
    hash_map.free();
    hash_map.free();
    hash_map.free();

    // Check that resources are deallocated and no errors occur
    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
}

/*****************************************************************
 * Test cases for the RS_HashMap::insert()
 *
 * Test cases:
 * 1. Insert into empty hash map
 * 2. Multiple insertions
 * 3. Insert duplicate key
 * 4. Insert until full, stress test with evictions
 * -> 4.1 Insert until full, stress test with evictions, on host
 * 5. Insert with group size 32
 * 6. Insert with group size 4
 *****************************************************************/

// Global kernel for inserting a single key-value pair
__global__ void insertSinglePair(
    RS_HashMap<uint32_t, LargeValue> map,
    uint32_t key,
    LargeValue value,
    bool* result
) {
    if (threadIdx.x == 0) {
        *result = map.insert(key, value);
    }
}

// Global kernel for inserting multiple key-value pairs
__global__ void insertMultiplePairs(
    RS_HashMap<uint32_t, LargeValue> map,
    uint32_t* keys,
    LargeValue* values,
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
insertWithGroup(HashMap map, uint32_t* keys, LargeValue* values, int count, bool* results) {
    // Warp size is 32
    int group_idx = (blockIdx.x * blockDim.x + threadIdx.x) / GroupSize;
    int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % GroupSize;
    if (group_idx < count) {
        bool result = map.template insert<GroupSize>(keys[group_idx], values[group_idx]);
        if (lane_id == 0) results[group_idx] = result;
    }
}

TEST_F(RS_HashMapTest, InsertIntoEmptyHashMap) {
    RSize max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    uint32_t key = 42;
    LargeValue value = {{100, 200, 300}};
    bool* d_result;
    bool h_result;

    cudaMalloc(&d_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_map, key, value, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    EXPECT_TRUE(h_result);
    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 1);

    host_hash_map.insert(42, value);

    // Verify the value
    LargeValue* val_ptr = host_hash_map.find(key);
    ASSERT_NE(val_ptr, nullptr);
    EXPECT_EQ(val_ptr->data[0], value.data[0]);
    EXPECT_EQ(val_ptr->data[1], value.data[1]);
    EXPECT_EQ(val_ptr->data[2], value.data[2]);

    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashMapTest, MultipleInsertions) {
    RSize max_entry_num = 2048;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);
    const int count = 1000;
    auto keys = new uint32_t[count];
    auto values = new LargeValue[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i;
        values[i] = {{i, i * 2, i * 3}};
    }

    uint32_t* d_keys;
    LargeValue* d_values;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(LargeValue));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, count * sizeof(LargeValue), cudaMemcpyHostToDevice);

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
    EXPECT_EQ(host_hash_map.count(), count);

    // Verify the values
    for (int i = 0; i < count; ++i) {
        LargeValue* val_ptr = host_hash_map.find(i);
        ASSERT_NE(val_ptr, nullptr);
        EXPECT_EQ(val_ptr->data[0], i);
        EXPECT_EQ(val_ptr->data[1], i * 2);
        EXPECT_EQ(val_ptr->data[2], i * 3);
    }

    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashMapTest, InsertDuplicateKey) {
    RSize max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    uint32_t key = 42;
    LargeValue value1 = {{100, 200, 300}};
    LargeValue value2 = {{400, 500, 600}};
    bool* d_result;
    bool h_result1, h_result2;

    cudaMalloc(&d_result, sizeof(bool));

    insertSinglePair<<<1, 1>>>(hash_map, key, value1, d_result);
    cudaMemcpy(&h_result1, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    insertSinglePair<<<1, 1>>>(hash_map, key, value2, d_result);
    cudaMemcpy(&h_result2, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    EXPECT_TRUE(h_result1);
    EXPECT_TRUE(h_result2);  // Should overwrite the existing value

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 1);

    // Verify the value corresponds to the last inserted value
    LargeValue* val_ptr = host_hash_map.find(key);
    ASSERT_NE(val_ptr, nullptr);
    EXPECT_EQ(val_ptr->data[0], value2.data[0]);
    EXPECT_EQ(val_ptr->data[1], value2.data[1]);
    EXPECT_EQ(val_ptr->data[2], value2.data[2]);

    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashMapTest, InsertUntilFull) {
    RSize max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    const int count = 200000;
    int successful_inserts = 0;
    auto keys = new uint32_t[count];
    auto values = new LargeValue[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i;
        values[i] = {{i, i * 2, i * 3}};
    }

    uint32_t* d_keys;
    LargeValue* d_values;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(LargeValue));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, count * sizeof(LargeValue), cudaMemcpyHostToDevice);

    insertMultiplePairs<<<RS_CEIL_DIV(count, 256), 256>>>(
        hash_map, d_keys, d_values, count, d_results
    );
    cudaMemcpy(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);

    for (int i = 0; i < count; ++i) {
        if (results[i]) successful_inserts++;
    }

    delete[] keys;
    delete[] values;
    delete[] results;

    // Since the map can hold up to max_entry_num entries, we expect that many successful inserts
    EXPECT_EQ(successful_inserts, max_entry_num);

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), max_entry_num);

    // Optionally, verify the inserted values
    int valid_count = 0;
    for (uint32_t i = 0; i < count; ++i) {
        LargeValue* val_ptr = host_hash_map.find(i);
        if (val_ptr != nullptr) {
            EXPECT_EQ(val_ptr->data[0], static_cast<int>(i));
            EXPECT_EQ(val_ptr->data[1], static_cast<int>(i * 2));
            EXPECT_EQ(val_ptr->data[2], static_cast<int>(i * 3));
            valid_count++;
        }
    }
    EXPECT_EQ(valid_count, max_entry_num);

    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashMapTest, InsertUntilFullHost) {
    RSize max_entry_num = 1024;
    RSize count = 200000;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &host_provider);

    int successful_inserts = 0;
    int failed_inserts = 0;

    // Perform insertions in a single thread on the host
    for (uint32_t i = 0; i < count; ++i) {
        LargeValue value = {
            {static_cast<int>(i), static_cast<int>(i * 2), static_cast<int>(i * 3)}
        };
        bool result = hash_map.insert(i, value);

        if (result) {
            successful_inserts++;
        } else {
            failed_inserts++;
        }
    }

    // Since the map can hold up to max_entry_num entries, we expect that many successful inserts
    EXPECT_EQ(successful_inserts, max_entry_num);

    // Verify the hash map's count
    EXPECT_EQ(hash_map.count(), max_entry_num);

    // Optionally, verify the inserted values
    int valid_count = 0;
    for (uint32_t i = 0; i < count; ++i) {
        LargeValue* val_ptr = hash_map.find(i);
        if (val_ptr != nullptr) {
            EXPECT_EQ(val_ptr->data[0], static_cast<int>(i));
            EXPECT_EQ(val_ptr->data[1], static_cast<int>(i * 2));
            EXPECT_EQ(val_ptr->data[2], static_cast<int>(i * 3));
            valid_count++;
        }
    }
    EXPECT_EQ(valid_count, max_entry_num);
    // Clean up
    hash_map.free();
}

TEST_F(RS_HashMapTest, InsertWithGroup32) {
    RSize max_entry_num = 2048;
    RS_HashMap<uint32_t, LargeValue, uint32_t, RS_GenericHasher, 32> hash_map(
        max_entry_num, &provider
    );
    const int count = 1000;
    auto keys = new uint32_t[count];
    auto values = new LargeValue[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i;
        values[i] = {{i, i * 2, i * 3}};
    }

    uint32_t* d_keys;
    LargeValue* d_values;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(LargeValue));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, count * sizeof(LargeValue), cudaMemcpyHostToDevice);

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
    EXPECT_EQ(host_hash_map.count(), count);

    // Verify the values
    for (int i = 0; i < count; ++i) {
        LargeValue* val_ptr = host_hash_map.find(i);
        ASSERT_NE(val_ptr, nullptr);
        EXPECT_EQ(val_ptr->data[0], i);
        EXPECT_EQ(val_ptr->data[1], i * 2);
        EXPECT_EQ(val_ptr->data[2], i * 3);
    }

    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashMapTest, InsertWithGroup4) {
    RSize max_entry_num = 2048;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);
    const int count = 1000;
    auto keys = new uint32_t[count];
    auto values = new LargeValue[count];
    auto results = new bool[count];

    for (int i = 0; i < count; ++i) {
        keys[i] = i;
        values[i] = {{i, i * 2, i * 3}};
    }

    uint32_t* d_keys;
    LargeValue* d_values;
    bool* d_results;

    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(LargeValue));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, count * sizeof(LargeValue), cudaMemcpyHostToDevice);

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
    EXPECT_EQ(host_hash_map.count(), count);

    // Verify the values
    for (int i = 0; i < count; ++i) {
        LargeValue* val_ptr = host_hash_map.find(i);
        ASSERT_NE(val_ptr, nullptr);
        EXPECT_EQ(val_ptr->data[0], i);
        EXPECT_EQ(val_ptr->data[1], i * 2);
        EXPECT_EQ(val_ptr->data[2], i * 3);
    }

    host_hash_map.free();
    hash_map.free();
}

/*****************************************************************
 * Test cases for the RS_HashMap::resize()
 *
 * Test cases:
 * 1. Insert, resize and find existing key.
 *****************************************************************/
__global__ void findSingleKey(
    RS_HashMap<uint32_t, LargeValue> map,
    uint32_t key,
    LargeValue* foundValue,
    bool* foundFlag
) {
    if (threadIdx.x == 0) {
        // Attempt to find the key in the map
        LargeValue* value = map.find(key);

        if (value != nullptr) {
            // If the key was found, copy the value to the output parameter
            *foundValue = *value;
            *foundFlag = true;
        } else {
            // If the key was not found, set the found flag to false
            *foundFlag = false;
        }
    }
}

TEST_F(RS_HashMapTest, ResizeAndFindExistingKey) {
    RSize initial_max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(initial_max_entry_num, &provider);

    uint32_t key = 42;
    LargeValue initial_value = {{100, 200, 300}};
    bool* d_result;
    cudaMalloc(&d_result, sizeof(bool));

    // Insert a single key-value pair
    insertSinglePair<<<1, 1>>>(hash_map, key, initial_value, d_result);

    // Resize the hash map to a larger capacity
    RSize new_max_entry_num = 2048;
    hash_map.resize(new_max_entry_num);

    // Allocate memory on the device for the found value and found flag
    LargeValue* d_found_value;
    bool* d_found_flag;
    cudaMalloc(&d_found_value, sizeof(LargeValue));
    cudaMalloc(&d_found_flag, sizeof(bool));

    // Attempt to find the key after resizing
    findSingleKey<<<1, 1>>>(hash_map, key, d_found_value, d_found_flag);

    // Copy the found flag and value back to host for verification
    bool h_found_flag;
    LargeValue h_found_value;
    cudaMemcpy(&h_found_flag, d_found_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found_value, d_found_value, sizeof(LargeValue), cudaMemcpyDeviceToHost);

    // Verify the key was found and the value is correct
    EXPECT_TRUE(h_found_flag);
    EXPECT_EQ(h_found_value.data[0], initial_value.data[0]);
    EXPECT_EQ(h_found_value.data[1], initial_value.data[1]);
    EXPECT_EQ(h_found_value.data[2], initial_value.data[2]);

    // Cleanup
    cudaFree(d_result);
    cudaFree(d_found_value);
    cudaFree(d_found_flag);
    hash_map.free();
}