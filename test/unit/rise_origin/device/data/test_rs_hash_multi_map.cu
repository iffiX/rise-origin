#include <gtest/gtest.h>
// #define DEBUG_HASHMAP
#include "rise_origin/device/data/rs_hashmap.h"

using namespace rise;
using namespace rise::device;

class RS_HashMultiMapTest : public ::testing::Test {
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
 * Test cases for the RS_HashMultiMap::insert()
 *
 * Test cases:
 * 1. Insert into empty multimap.
 * 2. Insert multiple values for the same key.
 * 3. Insert multiple unique keys.
 * 4. Insert duplicate key-value pair.
 * 5. Insert until hashmap full.
 * 6. Insert with group size 32
 * 7. Insert with group size 4
 *****************************************************************/

// Global kernel for inserting a single key-value pair
__global__ void
insertSinglePair(RS_HashMultiMap<uint32_t, int> map, uint32_t key, int value, bool* result) {
    if (threadIdx.x == 0) {
        *result = map.insert(key, value);
    }
}

// Global kernel for inserting multiple key-value pairs
__global__ void insertMultiplePairs(
    RS_HashMultiMap<uint32_t, int> map,
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
    RS_HashMultiMap<uint32_t, int> map,
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

// Kernel to find and retrieve the number of elements in the list view for a key
__global__ void
findKeySize(RS_HashMultiMap<uint32_t, int> map, uint32_t key, int* list_size, bool* found) {
    if (threadIdx.x == 0) {
        auto list_view = map.find(key);
        if (list_view.size() > 0) {
            *list_size = list_view.size();
            *found = true;
        } else {
            *found = false;
        }
    }
}

// Kernel to retrieve values from an RS_ListView associated with a key
__global__ void
getListValues(RS_HashMultiMap<uint32_t, int> map, uint32_t key, int* values, int max_values) {
    if (threadIdx.x == 0) {
        auto list_view = map.find(key);
        for (int i = 0; i < list_view.size() && i < max_values; ++i) {
            values[i] = list_view.get(i);
        }
    }
}

TEST_F(RS_HashMultiMapTest, InsertIntoEmptyMultimap) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    uint32_t key = 42;
    int value = 100;
    bool* d_result;
    bool h_result;

    cudaMalloc(&d_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_multi_map, key, value, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    EXPECT_TRUE(h_result);

    // Retrieve the size of the list view associated with the key
    int* d_list_size;
    int h_list_size;
    bool* d_found;
    bool h_found;

    cudaMalloc(&d_list_size, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    findKeySize<<<1, 1>>>(hash_multi_map, key, d_list_size, d_found);
    cudaMemcpy(&h_list_size, d_list_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_list_size);
    cudaFree(d_found);

    EXPECT_TRUE(h_found);
    EXPECT_EQ(h_list_size, 1);

    // Retrieve the values from the list view
    int* d_values;
    int h_values[1];
    cudaMalloc(&d_values, sizeof(int));

    getListValues<<<1, 1>>>(hash_multi_map, key, d_values, 1);
    cudaMemcpy(h_values, d_values, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_values);

    EXPECT_EQ(h_values[0], value);

    // Clean up
    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertMultipleValuesForSameKey) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    uint32_t key = 42;
    int values_to_insert[] = {100, 200, 300};
    bool* d_result;
    bool h_result;

    cudaMalloc(&d_result, sizeof(bool));

    // Insert multiple values under the same key
    for (int i = 0; i < 3; ++i) {
        insertSinglePair<<<1, 1>>>(hash_multi_map, key, values_to_insert[i], d_result);
        cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
        EXPECT_TRUE(h_result);
    }
    cudaFree(d_result);

    // Retrieve the size of the list view associated with the key
    int* d_list_size;
    int h_list_size;
    bool* d_found;
    bool h_found;

    cudaMalloc(&d_list_size, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    findKeySize<<<1, 1>>>(hash_multi_map, key, d_list_size, d_found);
    cudaMemcpy(&h_list_size, d_list_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_list_size);
    cudaFree(d_found);

    EXPECT_TRUE(h_found);
    EXPECT_EQ(h_list_size, 3);

    // Retrieve the values from the list view
    int* d_values;
    int h_values[3];
    cudaMalloc(&d_values, 3 * sizeof(int));

    getListValues<<<1, 1>>>(hash_multi_map, key, d_values, 3);
    cudaMemcpy(h_values, d_values, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_values);

    // Verify that all values are present
    std::vector<int> expected_values = {100, 200, 300};
    std::vector<int> retrieved_values(h_values, h_values + 3);
    EXPECT_EQ(retrieved_values.size(), expected_values.size());
    for (int val : expected_values) {
        EXPECT_NE(
            std::find(retrieved_values.begin(), retrieved_values.end(), val), retrieved_values.end()
        );
    }

    // Clean up
    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertMultipleUniqueKeys) {
    RSize max_entry_num = 2048;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    const int count = 1000;
    uint32_t* h_keys = new uint32_t[count];
    int* h_values = new int[count];
    bool* h_results = new bool[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertMultiplePairs<<<RS_CEIL_DIV(count, 256), 256>>>(
        hash_multi_map, d_keys, d_values, count, d_results
    );
    cudaMemcpy(h_results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_results[i]);
    }

    // Check if values are correct on device
    int* d_retrieved_values;
    cudaMalloc(&d_retrieved_values, sizeof(int));

    for (int i = 0; i < count; ++i) {
        cudaMemset(d_retrieved_values, 0, sizeof(int));
        getListValues<<<1, 1>>>(hash_multi_map, h_keys[i], d_retrieved_values, 1);

        int retrieved_value = 0;
        cudaMemcpy(&retrieved_value, d_retrieved_values, sizeof(int), cudaMemcpyDeviceToHost);

        EXPECT_EQ(retrieved_value, h_values[i]);
    }

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);
    cudaFree(d_retrieved_values);
    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertUntilHashMapFull) {
    RSize max_entry_num = 1024;
    // Due to parallelism, if the memory pool only has the same number of entry as the hashtable,
    // allocation may fail, so we give many times more space to prevent that
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num * 1000, &provider);

    const int count = 10000;
    uint32_t* h_keys = new uint32_t[count];
    int* h_values = new int[count];
    bool* h_results = new bool[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;  // Ensure unique keys
        h_values[i] = i;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertMultiplePairs<<<RS_CEIL_DIV(count, 256), 256>>>(
        hash_multi_map, d_keys, d_values, count, d_results
    );
    cudaMemcpy(h_results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    int successful_inserts = 0;
    for (int i = 0; i < count; ++i) {
        if (h_results[i]) {
            successful_inserts++;
        }
    }

    EXPECT_EQ(successful_inserts, max_entry_num);

    // Check if values are correct on device
    int* d_list_size;
    bool* d_found;
    cudaMalloc(&d_list_size, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    for (int i = 0; i < count; ++i) {
        if (h_results[i]) {
            findKeySize<<<1, 1>>>(hash_multi_map, h_keys[i], d_list_size, d_found);

            int h_list_size;
            bool h_found;
            cudaMemcpy(&h_list_size, d_list_size, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

            if (h_found && h_list_size == 1) {
                int* d_retrieved_values;
                cudaMalloc(&d_retrieved_values, sizeof(int));
                getListValues<<<1, 1>>>(hash_multi_map, h_keys[i], d_retrieved_values, 1);

                int retrieved_value;
                cudaMemcpy(
                    &retrieved_value, d_retrieved_values, sizeof(int), cudaMemcpyDeviceToHost
                );
                cudaFree(d_retrieved_values);
            }
        }
    }

    cudaFree(d_list_size);
    cudaFree(d_found);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);
    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertUntilHashMapFullHost) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &host_provider);

    const int count = 10000;
    uint32_t* h_keys = new uint32_t[count];
    int* h_values = new int[count];
    bool* h_results = new bool[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;  // Ensure unique keys
        h_values[i] = i;
    }

    for (int i = 0; i < count; ++i) {
        h_results[i] = hash_multi_map.insert(h_keys[i], h_values[i]);
    }

    EXPECT_EQ(hash_multi_map.count(), max_entry_num);

    // Check if values are correct on host
    for (int i = 0; i < count; ++i) {
        auto list_view = hash_multi_map.find(h_keys[i]);
        if (list_view.size() == 1) {
            EXPECT_EQ(list_view.get(0), h_values[i]);
        }
    }

    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertWithGroup32) {
    RSize max_entry_num = 2048;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    const int count = 1000;
    uint32_t* h_keys = new uint32_t[count];
    int* h_values = new int[count];
    bool* h_results = new bool[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertWithGroup<32>
        <<<RS_CEIL_DIV(count * 32, 256), 256>>>(hash_multi_map, d_keys, d_values, count, d_results);
    cudaMemcpy(h_results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_results[i]);
    }

    // Check if values are correct on device
    int* d_retrieved_values;
    cudaMalloc(&d_retrieved_values, sizeof(int));

    for (int i = 0; i < count; ++i) {
        cudaMemset(d_retrieved_values, 0, sizeof(int));
        getListValues<<<1, 1>>>(hash_multi_map, h_keys[i], d_retrieved_values, 1);

        int retrieved_value = 0;
        cudaMemcpy(&retrieved_value, d_retrieved_values, sizeof(int), cudaMemcpyDeviceToHost);

        EXPECT_EQ(retrieved_value, h_values[i]);
    }

    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    cudaFree(d_retrieved_values);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);

    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertWithGroup4) {
    RSize max_entry_num = 2048;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    const int count = 1000;
    uint32_t* h_keys = new uint32_t[count];
    int* h_values = new int[count];
    bool* h_results = new bool[count];

    for (int i = 0; i < count; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 10;
    }

    uint32_t* d_keys;
    int* d_values;
    bool* d_results;
    cudaMalloc(&d_keys, count * sizeof(uint32_t));
    cudaMalloc(&d_values, count * sizeof(int));
    cudaMalloc(&d_results, count * sizeof(bool));

    cudaMemcpy(d_keys, h_keys, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, count * sizeof(int), cudaMemcpyHostToDevice);

    insertWithGroup<4>
        <<<RS_CEIL_DIV(count * 4, 256), 256>>>(hash_multi_map, d_keys, d_values, count, d_results);
    cudaMemcpy(h_results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(h_results[i]);
    }

    // Check if values are correct on device
    int* d_retrieved_values;
    cudaMalloc(&d_retrieved_values, sizeof(int));

    for (int i = 0; i < count; ++i) {
        getListValues<<<1, 1>>>(hash_multi_map, h_keys[i], d_retrieved_values, 1);

        int retrieved_value;
        cudaMemcpy(&retrieved_value, d_retrieved_values, sizeof(int), cudaMemcpyDeviceToHost);

        EXPECT_EQ(retrieved_value, h_values[i]);
    }

    delete[] h_keys;
    delete[] h_values;
    delete[] h_results;

    cudaFree(d_retrieved_values);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);

    hash_multi_map.free();
}

/*****************************************************************
 * Test cases for the RS_HashMultiMap::move() and copy()
 *
 * Test cases:
 * 1. Move to host after single insertion.
 *****************************************************************/
TEST_F(RS_HashMultiMapTest, MoveToHostAfterSingleInsertion) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    uint32_t key = 42;
    int value = 100;
    bool* d_result;
    bool h_result;

    cudaMalloc(&d_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_multi_map, key, value, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    EXPECT_TRUE(h_result);

    // Move to host and verify
    auto host_hash_multi_map = hash_multi_map.move(&host_provider);
    EXPECT_EQ(host_hash_multi_map.count(), 1);
    auto list_view_host = host_hash_multi_map.find(key);
    EXPECT_EQ(list_view_host.size(), 1);
    EXPECT_EQ(list_view_host.get(0), value);
    host_hash_multi_map.free();

    // Clean up
    hash_multi_map.free();
}

/*****************************************************************
 * Test cases for the RS_HashMultiMap::copy()
 *
 * Test cases:
 * 1. Copy to a new device hash multimap after single insertion.
 *****************************************************************/
TEST_F(RS_HashMultiMapTest, CopyToDeviceAfterSingleInsertion) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    uint32_t key = 42;
    int value = 100;
    bool* d_result;
    bool h_result;

    cudaMalloc(&d_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_multi_map, key, value, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    EXPECT_TRUE(h_result);

    // Copy to a new device hash multimap and verify
    RS_HashMultiMap<uint32_t, int> hash_multi_map_copy(max_entry_num, max_entry_num, &provider);
    hash_multi_map_copy.copy(hash_multi_map);

    int* d_list_size;
    int h_list_size;
    bool* d_found;
    bool h_found;

    cudaMalloc(&d_list_size, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    findKeySize<<<1, 1>>>(hash_multi_map_copy, key, d_list_size, d_found);
    cudaMemcpy(&h_list_size, d_list_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_list_size);
    cudaFree(d_found);

    EXPECT_TRUE(h_found);
    EXPECT_EQ(h_list_size, 1);

    int* d_values;
    int h_values[1];
    cudaMalloc(&d_values, sizeof(int));

    getListValues<<<1, 1>>>(hash_multi_map_copy, key, d_values, 1);
    cudaMemcpy(h_values, d_values, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_values);

    EXPECT_EQ(h_values[0], value);

    // Clean up
    hash_multi_map.free();
    hash_multi_map_copy.free();
}

/*****************************************************************
 * Test cases for the RS_HashMultiMap::resize()
 *
 * Test cases:
 * 1. Insert, resize and find existing key.
 *****************************************************************/
TEST_F(RS_HashMultiMapTest, ResizeAndFindExistingKey) {
    RSize initial_max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(
        initial_max_entry_num, initial_max_entry_num, &provider
    );

    uint32_t key = 42;
    int value = 100;
    bool* d_insert_result;
    cudaMalloc(&d_insert_result, sizeof(bool));
    insertSinglePair<<<1, 1>>>(hash_multi_map, key, value, d_insert_result);

    // Resize the hash multi-map to a larger capacity
    RSize new_max_entry_num = 2048;
    // Assuming resize() functionality is supported and implemented for RS_HashMultiMap
    hash_multi_map.resize(new_max_entry_num, 0, true);

    // Attempt to find the key after resizing
    int* d_list_size;
    bool* d_found;
    cudaMalloc(&d_list_size, sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    findKeySize<<<1, 1>>>(hash_multi_map, key, d_list_size, d_found);

    // Copy the results back to host for verification
    int h_list_size;
    bool h_found;
    cudaMemcpy(&h_list_size, d_list_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    EXPECT_TRUE(h_found);
    EXPECT_EQ(h_list_size, 1);

    // Retrieve the values from the list view
    int* d_values;
    int h_values[1];  // Adjust size if expecting more values per key
    cudaMalloc(&d_values, sizeof(int) * h_list_size);

    getListValues<<<1, 1>>>(hash_multi_map, key, d_values, h_list_size);
    cudaMemcpy(h_values, d_values, sizeof(int) * h_list_size, cudaMemcpyDeviceToHost);

    // Verify the value(s) are correct
    EXPECT_EQ(h_values[0], value);

    // Cleanup
    cudaFree(d_insert_result);
    cudaFree(d_list_size);
    cudaFree(d_found);
    cudaFree(d_values);
    hash_multi_map.free();
}
