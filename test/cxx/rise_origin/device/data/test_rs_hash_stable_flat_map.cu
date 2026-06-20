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
 * RS_HashStableFlatMap stores values inside the hash table like
 * RS_HashFlatMap but uses a power of two insertion strategy so
 * that once a key is inserted its value pointer stays stable.
 * Each call adds 1 to the key so that 0 marks an empty cell.
 *
 * - Insert (Device)
 *   insert() one pair, many pairs, duplicate key, fill until full
 *
 * - Find (Device)
 *   find() returns the inserted value
 *
 * erase(), find(), init(), free(), resize(), count() are inherited
 * from RS_HashFlatMap. The device probes below use the bare
 * RS_ComputeFunction path. Each thread works on one key with group
 * size 1 and writes its outcome into an int result vector that is
 * moved to the host and checked.
 *****************************************************************/

class RS_HashStableFlatMapTest : public ::testing::Test {
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

class RSCF_HashStableFlatMapInsert : public RS_ComputeFunction<RSCF_HashStableFlatMapInsert> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_HashStableFlatMap<uint32_t, int> map,
        RS_Vector<uint32_t> keys,
        RS_Vector<int> values,
        RS_Vector<int> results
    ) {
        if (index1D() < keys.size()) {
            results[index1D()] = map.insert(keys[index1D()], values[index1D()]) ? 1 : 0;
        }
    }
};

class RSCF_HashStableFlatMapFind : public RS_ComputeFunction<RSCF_HashStableFlatMapFind> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_HashStableFlatMap<uint32_t, int> map,
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
class RSCF_HashStableFlatMapInsertGroup
    : public RS_ComputeFunction<RSCF_HashStableFlatMapInsertGroup<GroupSize, HashMap>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashStableFlatMapInsertGroup<GroupSize, HashMap>>;
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
#endif

/*****************************************************************
 * Test cases for RS_HashStableFlatMap::insert()
 *
 * Test cases:
 * 1. Insert into empty hash map.
 * 2. Multiple insertions.
 * 3. Insert duplicate key.
 * 4. Insert until full.
 * 5. Insert with group size 32.
 * 6. Insert with group size 16.
 *****************************************************************/

TEST_F(RS_HashStableFlatMapTest, InsertIntoEmptyHashMap) {
    RSize max_entry_num = 1024;
    RS_HashStableFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    RS_Vector<int> values(1, &provider);
    uint32_t key_data[1] = {42};
    int value_data[1] = {100};
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> results(1, &provider);
    results.fill(0);

    provider.run<RSCF_HashStableFlatMapInsert>({1}, hash_map, keys, values, results);

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

TEST_F(RS_HashStableFlatMapTest, MultipleInsertions) {
    RSize max_entry_num = 2048;
    RS_HashStableFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    const int count = 10000;
    const int unique_keys = 100;
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

    provider.run<RSCF_HashStableFlatMapInsert>({(RSize)count}, hash_map, keys, values, results);

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

TEST_F(RS_HashStableFlatMapTest, InsertDuplicateKey) {
    RSize max_entry_num = 1024;
    RS_HashStableFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {42};
    keys.write(key_data);

    RS_Vector<int> values1(1, &provider);
    int value_data1[1] = {100};
    values1.write(value_data1);
    RS_Vector<int> results1(1, &provider);
    results1.fill(0);
    provider.run<RSCF_HashStableFlatMapInsert>({1}, hash_map, keys, values1, results1);
    RS_Vector<int> host_results1 = results1.move(&host_provider);
    EXPECT_EQ(host_results1[0], 1);

    RS_Vector<int> values2(1, &provider);
    int value_data2[1] = {200};
    values2.write(value_data2);
    RS_Vector<int> results2(1, &provider);
    results2.fill(0);
    provider.run<RSCF_HashStableFlatMapInsert>({1}, hash_map, keys, values2, results2);
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

TEST_F(RS_HashStableFlatMapTest, InsertUntilFull) {
    RSize max_entry_num = 2048;
    RS_HashStableFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

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

    provider.run<RSCF_HashStableFlatMapInsert>({(RSize)count}, hash_map, keys, values, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    int successful_inserts = 0;
    for (int i = 0; i < count; ++i) {
        successful_inserts += host_results[i];
    }
    EXPECT_LE((RSize)successful_inserts, max_entry_num);
    EXPECT_GT(successful_inserts, 0);

    delete[] key_data;
    delete[] value_data;
    host_results.free();
    keys.free();
    values.free();
    hash_map.free();
}

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
TEST_F(RS_HashStableFlatMapTest, InsertWithGroup32) {
    RSize max_entry_num = 2048;
    using HashMapType = RS_HashStableFlatMap<uint32_t, int, uint32_t, RS_GenericHasher, 32>;
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
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashStableFlatMapInsertGroup<32, HashMapType>>(
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

TEST_F(RS_HashStableFlatMapTest, InsertWithGroup16) {
    RSize max_entry_num = 2048;
    using HashMapType = RS_HashStableFlatMap<uint32_t, int, uint32_t, RS_GenericHasher, 16>;
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
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashStableFlatMapInsertGroup<16, HashMapType>>(
        {static_cast<RSize>(count) * 16}, hash_map, keys, values, results
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
 * Test cases for RS_HashStableFlatMap::find()
 *
 * Test cases:
 * 1. Find returns the inserted value.
 *****************************************************************/

TEST_F(RS_HashStableFlatMapTest, FindExistingKey) {
    RSize max_entry_num = 1024;
    RS_HashStableFlatMap<uint32_t, int> hash_map(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    RS_Vector<int> values(1, &provider);
    uint32_t key_data[1] = {42};
    int value_data[1] = {100};
    keys.write(key_data);
    values.write(value_data);
    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashStableFlatMapInsert>({1}, hash_map, keys, values, insert_results);

    RS_Vector<int> found(1, &provider);
    RS_Vector<int> found_values(1, &provider);
    found.fill(0);
    found_values.fill(0);
    provider.run<RSCF_HashStableFlatMapFind>({1}, hash_map, keys, found, found_values);

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
