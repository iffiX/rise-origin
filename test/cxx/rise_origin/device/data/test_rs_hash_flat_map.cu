#include <gtest/gtest.h>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_hashmap.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_HashFlatMap is a device hash map that stores values inside
 * the hash table across two cuckoo tables. Each call adds 1 to
 * the key so that 0 marks an empty cell.
 *
 * - Construction & Initialization
 *   Constructor with capacity, zero capacity, large capacity,
 *   default constructor, bucket_num_per_table, size(), empty()
 *
 * - Lifetime
 *   free(), multiple free()
 *
 * - Insert (Device)
 *   insert() one pair, many pairs, duplicate key, fill until full
 *
 * - Erase (Device)
 *   erase() existing key, missing key, empty map, all keys
 *
 * - Find (Device)
 *   find() existing key, missing key, after erase
 *
 * - Resize
 *   resize() keeps inserted pair reachable
 *
 * The device probes below use the bare RS_ComputeFunction path.
 * Each thread works on one key with group size 1 and writes its
 * outcome into an int result vector that is moved to the host and
 * checked. Found values are written into a separate result vector.
 *****************************************************************/

class RS_HashFlatMapTest : public ::testing::Test {
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
 * Device probes
 *****************************************************************/

class RSCF_HashFlatMapInsert : public RS_ComputeFunction<RSCF_HashFlatMapInsert> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_HashFlatMap<uint32_t, int> map,
        RS_Vector<uint32_t> keys,
        RS_Vector<int> values,
        RS_Vector<int> results
    ) {
        if (index1D() < keys.size()) {
            results[index1D()] = map.insert(keys[index1D()], values[index1D()]) ? 1 : 0;
        }
    }
};

class RSCF_HashFlatMapErase : public RS_ComputeFunction<RSCF_HashFlatMapErase> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_HashFlatMap<uint32_t, int> map, RS_Vector<uint32_t> keys, RS_Vector<int> results) {
        if (index1D() < keys.size()) {
            results[index1D()] = map.erase(keys[index1D()]) ? 1 : 0;
        }
    }
};

class RSCF_HashFlatMapFind : public RS_ComputeFunction<RSCF_HashFlatMapFind> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_HashFlatMap<uint32_t, int> map,
        RS_Vector<uint32_t> keys,
        RS_Vector<int> found,
        RS_Vector<int> values
    ) {
        if (index1D() < keys.size()) {
            int *ptr = map.find(keys[index1D()]);
            if (ptr != nullptr) {
                values[index1D()] = *ptr;
                found[index1D()] = 1;
            } else {
                found[index1D()] = 0;
            }
        }
    }
};

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
template <int GroupSize, typename HashMap>
class RSCF_HashFlatMapInsertGroup
    : public RS_ComputeFunction<RSCF_HashFlatMapInsertGroup<GroupSize, HashMap>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashFlatMapInsertGroup<GroupSize, HashMap>>;
    using BaseType::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void
    run(HashMap map, RS_Vector<uint32_t> keys, RS_Vector<int> values, RS_Vector<int> results) {
        RIndex group_idx = index1D() / GroupSize;
        RIndex lane_id = index1D() % GroupSize;
        if (group_idx < keys.size()) {
            bool is_inserted = map.template insert<GroupSize>(keys[group_idx], values[group_idx]);
            if (lane_id == 0) results[group_idx] = is_inserted ? 1 : 0;
        }
    }
};

template <int GroupSize, typename HashMap>
class RSCF_HashFlatMapEraseGroup
    : public RS_ComputeFunction<RSCF_HashFlatMapEraseGroup<GroupSize, HashMap>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashFlatMapEraseGroup<GroupSize, HashMap>>;
    using BaseType::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void
    run(HashMap map, RS_Vector<uint32_t> keys, RS_Vector<int> results) {
        RIndex group_idx = index1D() / GroupSize;
        RIndex lane_id = index1D() % GroupSize;
        if (group_idx < keys.size()) {
            bool is_erased = map.template erase<GroupSize>(keys[group_idx]);
            if (lane_id == 0) results[group_idx] = is_erased ? 1 : 0;
        }
    }
};

template <int GroupSize, typename HashMap>
class RSCF_HashFlatMapFindGroup
    : public RS_ComputeFunction<RSCF_HashFlatMapFindGroup<GroupSize, HashMap>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashFlatMapFindGroup<GroupSize, HashMap>>;
    using BaseType::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void
    run(HashMap map, RS_Vector<uint32_t> keys, RS_Vector<int> found, RS_Vector<int> values) {
        RIndex group_idx = index1D() / GroupSize;
        RIndex lane_id = index1D() % GroupSize;
        if (group_idx < keys.size()) {
            int *ptr = map.template find<GroupSize>(keys[group_idx]);
            if (lane_id == 0) {
                if (ptr != nullptr) {
                    values[group_idx] = *ptr;
                    found[group_idx] = 1;
                } else {
                    found[group_idx] = 0;
                }
            }
        }
    }
};
#endif

/*****************************************************************
 * Test cases for RS_HashFlatMap construction and initialization
 *
 * Test cases:
 * 1. Construct with valid parameters.
 * 2. Construct with zero capacity.
 * 3. Construct with large capacity.
 * 4. Default constructor.
 *****************************************************************/

TEST_F(RS_HashFlatMapTest, ConstructWithValidParameters) {
    RSize max_entry_num = 1024;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, ConstructWithZeroCapacity) {
    RSize max_entry_num = 0;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, ConstructWithLargeCapacity) {
    RSize max_entry_num = 1 << 20;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, DefaultConstructor) {
    RS_HashFlatMap<uint32_t, int> hash_map;
    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), 0);
}

/*****************************************************************
 * Test cases for RS_HashFlatMap::free()
 *
 * Test cases:
 * 1. Free after initialization.
 * 2. Multiple frees.
 *****************************************************************/

TEST_F(RS_HashFlatMapTest, FreeAfterInitialization) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);
    hash_map.free();
    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
}

TEST_F(RS_HashFlatMapTest, MultipleFrees) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);
    hash_map.free();
    hash_map.free();
    hash_map.free();
    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
}

/*****************************************************************
 * Test cases for RS_HashFlatMap::insert()
 *
 * Test cases:
 * 1. Insert into empty hash map.
 * 2. Multiple insertions.
 * 3. Insert duplicate key.
 * 4. Insert until full.
 *****************************************************************/

TEST_F(RS_HashFlatMapTest, InsertIntoEmptyHashMap) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    RS_Vector<int> values(1, &provider);
    uint32_t key_data[1] = {42};
    int value_data[1] = {100};
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> results(1, &provider);
    results.fill(0);

    provider.run<RSCF_HashFlatMapInsert>({1}, hash_map, keys, values, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    EXPECT_EQ(host_results[0], 1);

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 1);

    host_results.free();
    host_hash_map.free();
    keys.free();
    values.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, MultipleInsertions) {
    RSize max_entry_num = 2048;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    const int count = 10000;
    const int unique_keys = 1000;
    auto key_data = new uint32_t[count];
    auto value_data = new int[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i % unique_keys;
        value_data[i] = i * 10;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashFlatMapInsert>({(RSize)count}, hash_map, keys, values, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), unique_keys);

    delete[] key_data;
    delete[] value_data;
    host_results.free();
    host_hash_map.free();
    keys.free();
    values.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, InsertDuplicateKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {42};
    keys.write(key_data);

    RS_Vector<int> values1(1, &provider);
    int value_data1[1] = {100};
    values1.write(value_data1);
    RS_Vector<int> results1(1, &provider);
    results1.fill(0);
    provider.run<RSCF_HashFlatMapInsert>({1}, hash_map, keys, values1, results1);
    RS_Vector<int> host_results1 = results1.move(&host_provider);
    EXPECT_EQ(host_results1[0], 1);

    RS_Vector<int> values2(1, &provider);
    int value_data2[1] = {200};
    values2.write(value_data2);
    RS_Vector<int> results2(1, &provider);
    results2.fill(0);
    provider.run<RSCF_HashFlatMapInsert>({1}, hash_map, keys, values2, results2);
    RS_Vector<int> host_results2 = results2.move(&host_provider);
    EXPECT_EQ(host_results2[0], 1);

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 1);

    host_results1.free();
    host_results2.free();
    host_hash_map.free();
    keys.free();
    values1.free();
    values2.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, InsertUntilFull) {
    RSize max_entry_num = 2048;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    const int count = 10000;
    auto key_data = new uint32_t[count];
    auto value_data = new int[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
        value_data[i] = i * 10;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashFlatMapInsert>({(RSize)count}, hash_map, keys, values, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    int successful_inserts = 0;
    for (int i = 0; i < count; ++i) {
        successful_inserts += host_results[i];
    }
    EXPECT_EQ((RSize)successful_inserts, max_entry_num);

    delete[] key_data;
    delete[] value_data;
    host_results.free();
    keys.free();
    values.free();
    hash_map.free();
}

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
TEST_F(RS_HashFlatMapTest, InsertWithGroup32) {
    RSize max_entry_num = 16384;
    using HashMapType = RS_HashFlatMap<uint32_t, int>;
    HashMapType hash_map(max_entry_num, &provider);

    const int count = 1000;
    auto key_data = new uint32_t[count];
    auto value_data = new int[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
        value_data[i] = i * 10;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashFlatMapInsertGroup<32, HashMapType>>(
        {static_cast<RSize>(count) * 32}, hash_map, keys, values, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), count);

    delete[] key_data;
    delete[] value_data;
    host_results.free();
    host_hash_map.free();
    keys.free();
    values.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, InsertWithGroup4) {
    RSize max_entry_num = 2048;
    using HashMapType = RS_HashFlatMap<uint32_t, int>;
    HashMapType hash_map(max_entry_num, &provider);

    const int count = 1000;
    auto key_data = new uint32_t[count];
    auto value_data = new int[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
        value_data[i] = i * 10;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashFlatMapInsertGroup<4, HashMapType>>(
        {static_cast<RSize>(count) * 4}, hash_map, keys, values, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), count);

    delete[] key_data;
    delete[] value_data;
    host_results.free();
    host_hash_map.free();
    keys.free();
    values.free();
    hash_map.free();
}
#endif

/*****************************************************************
 * Test cases for RS_HashFlatMap::erase()
 *
 * Test cases:
 * 1. Erase existing key.
 * 2. Erase non-existent key.
 * 3. Erase from empty hash map.
 * 4. Erase all keys.
 * 5. Erase with group size 32.
 *****************************************************************/

TEST_F(RS_HashFlatMapTest, EraseExistingKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    RS_Vector<int> values(1, &provider);
    uint32_t key_data[1] = {42};
    int value_data[1] = {100};
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashFlatMapInsert>({1}, hash_map, keys, values, insert_results);

    RS_Vector<int> erase_results(1, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashFlatMapErase>({1}, hash_map, keys, erase_results);

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    EXPECT_EQ(host_erase[0], 1);

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);
    EXPECT_EQ(host_hash_map.find(key_data[0]), nullptr);

    insert_results.free();
    host_erase.free();
    host_hash_map.free();
    keys.free();
    values.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, EraseNonExistentKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {12345};
    keys.write(key_data);
    RS_Vector<int> erase_results(1, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashFlatMapErase>({1}, hash_map, keys, erase_results);

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    EXPECT_EQ(host_erase[0], 0);

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);

    host_erase.free();
    host_hash_map.free();
    keys.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, EraseFromEmptyHashMap) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {7};
    keys.write(key_data);
    RS_Vector<int> erase_results(1, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashFlatMapErase>({1}, hash_map, keys, erase_results);

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    EXPECT_EQ(host_erase[0], 0);

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);

    host_erase.free();
    host_hash_map.free();
    keys.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, EraseAllKeys) {
    RSize max_entry_num = 2048;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    const int count = 100;
    auto key_data = new uint32_t[count];
    auto value_data = new int[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
        value_data[i] = i * 10;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> insert_results(count, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashFlatMapInsert>({(RSize)count}, hash_map, keys, values, insert_results);

    RS_Vector<int> erase_results(count, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashFlatMapErase>({(RSize)count}, hash_map, keys, erase_results);

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_erase[i], 1);
    }

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);

    delete[] key_data;
    delete[] value_data;
    insert_results.free();
    host_erase.free();
    host_hash_map.free();
    keys.free();
    values.free();
    hash_map.free();
}

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
TEST_F(RS_HashFlatMapTest, EraseWithGroup32) {
    RSize max_entry_num = 16384;
    using HashMapType = RS_HashFlatMap<uint32_t, int>;
    HashMapType hash_map(max_entry_num, &provider);

    const int count = 100;
    auto key_data = new uint32_t[count];
    auto value_data = new int[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
        value_data[i] = i * 10;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> insert_results(count, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashFlatMapInsertGroup<32, HashMapType>>(
        {static_cast<RSize>(count) * 32}, hash_map, keys, values, insert_results
    );

    RS_Vector<int> erase_results(count, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashFlatMapEraseGroup<32, HashMapType>>(
        {static_cast<RSize>(count) * 32}, hash_map, keys, erase_results
    );

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_erase[i], 1);
    }

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 0);

    delete[] key_data;
    delete[] value_data;
    insert_results.free();
    host_erase.free();
    host_hash_map.free();
    keys.free();
    values.free();
    hash_map.free();
}
#endif

/*****************************************************************
 * Test cases for RS_HashFlatMap::find()
 *
 * Test cases:
 * 1. Find existing key.
 * 2. Find non-existent key.
 * 3. Find after erase.
 * 4. Find with group size 32.
 *****************************************************************/

TEST_F(RS_HashFlatMapTest, FindExistingKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    RS_Vector<int> values(1, &provider);
    uint32_t key_data[1] = {42};
    int value_data[1] = {100};
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashFlatMapInsert>({1}, hash_map, keys, values, insert_results);

    RS_Vector<int> found(1, &provider);
    RS_Vector<int> found_values(1, &provider);
    found.fill(0);
    found_values.fill(0);
    provider.run<RSCF_HashFlatMapFind>({1}, hash_map, keys, found, found_values);

    RS_Vector<int> host_found = found.move(&host_provider);
    RS_Vector<int> host_values = found_values.move(&host_provider);
    EXPECT_EQ(host_found[0], 1);
    EXPECT_EQ(host_values[0], value_data[0]);

    insert_results.free();
    host_found.free();
    host_values.free();
    keys.free();
    values.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, FindNonExistentKey) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {12345};
    keys.write(key_data);
    RS_Vector<int> found(1, &provider);
    RS_Vector<int> found_values(1, &provider);
    found.fill(0);
    found_values.fill(0);
    provider.run<RSCF_HashFlatMapFind>({1}, hash_map, keys, found, found_values);

    RS_Vector<int> host_found = found.move(&host_provider);
    EXPECT_EQ(host_found[0], 0);

    host_found.free();
    found_values.free();
    keys.free();
    hash_map.free();
}

TEST_F(RS_HashFlatMapTest, FindAfterErase) {
    RSize max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    RS_Vector<int> values(1, &provider);
    uint32_t key_data[1] = {99};
    int value_data[1] = {999};
    keys.write(key_data);
    values.write(value_data);

    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashFlatMapInsert>({1}, hash_map, keys, values, insert_results);

    RS_Vector<int> erase_results(1, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashFlatMapErase>({1}, hash_map, keys, erase_results);

    RS_Vector<int> found(1, &provider);
    RS_Vector<int> found_values(1, &provider);
    found.fill(0);
    found_values.fill(0);
    provider.run<RSCF_HashFlatMapFind>({1}, hash_map, keys, found, found_values);

    RS_Vector<int> host_found = found.move(&host_provider);
    EXPECT_EQ(host_found[0], 0);

    insert_results.free();
    erase_results.free();
    host_found.free();
    found_values.free();
    keys.free();
    values.free();
    hash_map.free();
}

/*****************************************************************
 * Test cases for RS_HashFlatMap::resize()
 *
 * Test cases:
 * 1. Insert, resize and find existing key.
 *****************************************************************/

TEST_F(RS_HashFlatMapTest, ResizeAndFindKey) {
    RSize initial_max_entry_num = 1024;
    RS_HashFlatMap<uint32_t, int> hash_map(initial_max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    RS_Vector<int> values(1, &provider);
    uint32_t key_data[1] = {42};
    int value_data[1] = {100};
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashFlatMapInsert>({1}, hash_map, keys, values, insert_results);

    RSize new_max_entry_num = 2048;
    hash_map.resize(new_max_entry_num);

    RS_Vector<int> found(1, &provider);
    RS_Vector<int> found_values(1, &provider);
    found.fill(0);
    found_values.fill(0);
    provider.run<RSCF_HashFlatMapFind>({1}, hash_map, keys, found, found_values);

    RS_Vector<int> host_found = found.move(&host_provider);
    RS_Vector<int> host_values = found_values.move(&host_provider);
    EXPECT_EQ(host_found[0], 1);
    EXPECT_EQ(host_values[0], value_data[0]);

    insert_results.free();
    host_found.free();
    host_values.free();
    keys.free();
    values.free();
    hash_map.free();
}

/*****************************************************************
 * Test cases for RS_HashFlatMap grouped find
 *
 * Test cases:
 * 1. Find with group size 32 returns inserted values.
 *****************************************************************/

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
TEST_F(RS_HashFlatMapTest, FindWithGroup32) {
    RSize max_entry_num = 16384;
    using HashMapType = RS_HashFlatMap<uint32_t, int>;
    HashMapType hash_map(max_entry_num, &provider);

    const int count = 64;
    auto key_data = new uint32_t[count];
    auto value_data = new int[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
        value_data[i] = i * 10;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> insert_results(count, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashFlatMapInsertGroup<32, HashMapType>>(
        {static_cast<RSize>(count) * 32}, hash_map, keys, values, insert_results
    );

    RS_Vector<int> found(count, &provider);
    RS_Vector<int> found_values(count, &provider);
    found.fill(0);
    found_values.fill(0);
    provider.run<RSCF_HashFlatMapFindGroup<32, HashMapType>>(
        {static_cast<RSize>(count) * 32}, hash_map, keys, found, found_values
    );

    RS_Vector<int> host_found = found.move(&host_provider);
    RS_Vector<int> host_values = found_values.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_found[i], 1);
        EXPECT_EQ(host_values[i], value_data[i]);
    }

    delete[] key_data;
    delete[] value_data;
    insert_results.free();
    host_found.free();
    host_values.free();
    keys.free();
    values.free();
    hash_map.free();
}
#endif
