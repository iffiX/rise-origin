#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_hashmap.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_HashMultiMap is a concurrent hash map on the device that
 * allows many values per key. The values for a key are kept in a
 * list backed by a memory pool, so a single find() returns a list
 * view over every value stored under the key.
 *
 * - Hash Multi Map Operations on the device
 *   insert() single, multiple, and grouped
 *   find() returning a list view with size() and get()
 *
 * - Capacity & Size
 *   count()
 *
 * - Memory Management
 *   move(), copy(), resize() with data kept
 *****************************************************************/

class RS_HashMultiMapTest : public ::testing::Test {
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
 * Compute functions that probe the hash multi map on the device
 *****************************************************************/

class RSCF_MultiMapInsertSingle : public RS_ComputeFunction<RSCF_MultiMapInsertSingle> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_HashMultiMap<uint32_t, int> map, uint32_t key, int value, RS_Vector<int> results) {
        if (index1D() == 0) {
            results[0] = map.insert(key, value) ? 1 : 0;
        }
    }
};

class RSCF_MultiMapInsertMany : public RS_ComputeFunction<RSCF_MultiMapInsertMany> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_HashMultiMap<uint32_t, int> map,
        RS_Vector<uint32_t> keys,
        RS_Vector<int> values,
        RS_Vector<int> results
    ) {
        RIndex idx = index1D();
        if (idx < keys.size()) {
            results[idx] = map.insert(keys[idx], values[idx]) ? 1 : 0;
        }
    }
};

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
template <int GroupSize>
class RSCF_MultiMapInsertGroup : public RS_ComputeFunction<RSCF_MultiMapInsertGroup<GroupSize>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_MultiMapInsertGroup<GroupSize>>;
    using BaseType::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_HashMultiMap<uint32_t, int> map,
        RS_Vector<uint32_t> keys,
        RS_Vector<int> values,
        RS_Vector<int> results
    ) {
        RIndex group_idx = index1D() / GroupSize;
        RIndex lane_id = index1D() % GroupSize;
        if (group_idx < keys.size()) {
            bool is_inserted = map.template insert<GroupSize>(keys[group_idx], values[group_idx]);
            if (lane_id == 0) results[group_idx] = is_inserted ? 1 : 0;
        }
    }
};
#endif

// Writes the list size of a key into results[0], and up to max_values list
// values starting at results[1].
class RSCF_MultiMapFind : public RS_ComputeFunction<RSCF_MultiMapFind> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_HashMultiMap<uint32_t, int> map, uint32_t key, int max_values, RS_Vector<int> results) {
        if (index1D() == 0) {
            auto list_view = map.find(key);
            results[0] = static_cast<int>(list_view.size());
            for (int i = 0; i < list_view.size() && i < max_values; ++i) {
                results[1 + i] = list_view.get(i);
            }
        }
    }
};

/*****************************************************************
 * Test cases for RS_HashMultiMap::insert()
 *
 * Test cases:
 * 1. Insert into an empty multimap.
 * 2. Insert multiple values for the same key.
 * 3. Insert multiple unique keys.
 * 4. Insert until the hash map is full.
 * 5. Insert with group size 32.
 * 6. Insert with group size 4.
 *****************************************************************/

TEST_F(RS_HashMultiMapTest, InsertIntoEmptyMultimap) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    uint32_t key = 42;
    int value = 100;

    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_MultiMapInsertSingle>({1}, hash_multi_map, key, value, insert_results);
    RS_Vector<int> host_insert = insert_results.move(&host_provider);
    EXPECT_EQ(host_insert[0], 1);

    RS_Vector<int> find_results(2, &provider);
    find_results.fill(0);
    provider.run<RSCF_MultiMapFind>({1}, hash_multi_map, key, 1, find_results);
    RS_Vector<int> host_find = find_results.move(&host_provider);
    EXPECT_EQ(host_find[0], 1);
    EXPECT_EQ(host_find[1], value);

    insert_results.free();
    host_insert.free();
    find_results.free();
    host_find.free();
    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertMultipleValuesForSameKey) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    uint32_t key = 42;
    int values_to_insert[] = {100, 200, 300};

    for (int i = 0; i < 3; ++i) {
        RS_Vector<int> insert_results(1, &provider);
        insert_results.fill(0);
        provider.run<RSCF_MultiMapInsertSingle>(
            {1}, hash_multi_map, key, values_to_insert[i], insert_results
        );
        RS_Vector<int> host_insert = insert_results.move(&host_provider);
        EXPECT_EQ(host_insert[0], 1);
        insert_results.free();
        host_insert.free();
    }

    RS_Vector<int> find_results(4, &provider);
    find_results.fill(0);
    provider.run<RSCF_MultiMapFind>({1}, hash_multi_map, key, 3, find_results);
    RS_Vector<int> host_find = find_results.move(&host_provider);
    EXPECT_EQ(host_find[0], 3);

    std::vector<int> expected_values = {100, 200, 300};
    std::vector<int> retrieved_values = {host_find[1], host_find[2], host_find[3]};
    for (int val : expected_values) {
        EXPECT_NE(
            std::find(retrieved_values.begin(), retrieved_values.end(), val),
            retrieved_values.end()
        );
    }

    find_results.free();
    host_find.free();
    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertMultipleUniqueKeys) {
    RSize max_entry_num = 2048;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    const int count = 1000;
    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    auto host_keys = new uint32_t[count];
    auto host_values = new int[count];
    for (int i = 0; i < count; ++i) {
        host_keys[i] = i;
        host_values[i] = i * 10;
    }
    keys.write(host_keys);
    values.write(host_values);

    provider.run<RSCF_MultiMapInsertMany>(
        {static_cast<RSize>(count)}, hash_multi_map, keys, values, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    for (int i = 0; i < count; ++i) {
        RS_Vector<int> find_results(2, &provider);
        find_results.fill(0);
        provider.run<RSCF_MultiMapFind>({1}, hash_multi_map, host_keys[i], 1, find_results);
        RS_Vector<int> host_find = find_results.move(&host_provider);
        EXPECT_EQ(host_find[0], 1);
        EXPECT_EQ(host_find[1], host_values[i]);
        find_results.free();
        host_find.free();
    }

    delete[] host_keys;
    delete[] host_values;
    keys.free();
    values.free();
    results.free();
    host_results.free();
    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertUntilHashMapFull) {
    RSize max_entry_num = 1024;
    // Due to parallelism, if the pool only has the same number of entries as the table,
    // allocation may fail, so we give many times more space to prevent that.
    RS_HashMultiMap<uint32_t, int> hash_multi_map(
        max_entry_num, max_entry_num * 1000, &provider
    );

    const int count = 10000;
    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    auto host_keys = new uint32_t[count];
    auto host_values = new int[count];
    for (int i = 0; i < count; ++i) {
        host_keys[i] = i;
        host_values[i] = i;
    }
    keys.write(host_keys);
    values.write(host_values);

    provider.run<RSCF_MultiMapInsertMany>(
        {static_cast<RSize>(count)}, hash_multi_map, keys, values, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    int successful_inserts = 0;
    for (int i = 0; i < count; ++i) {
        if (host_results[i] == 1) successful_inserts++;
    }
    EXPECT_EQ(successful_inserts, max_entry_num);

    delete[] host_keys;
    delete[] host_values;
    keys.free();
    values.free();
    results.free();
    host_results.free();
    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertUntilHashMapFullHost) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &host_provider);

    const int count = 10000;
    for (int i = 0; i < count; ++i) {
        hash_multi_map.insert(i, i);
    }

    EXPECT_EQ(hash_multi_map.count(), max_entry_num);

    for (int i = 0; i < count; ++i) {
        auto list_view = hash_multi_map.find(i);
        if (list_view.size() == 1) {
            EXPECT_EQ(list_view.get(0), i);
        }
    }

    hash_multi_map.free();
}

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
TEST_F(RS_HashMultiMapTest, InsertWithGroup32) {
    RSize max_entry_num = 2048;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    const int count = 1000;
    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    auto host_keys = new uint32_t[count];
    auto host_values = new int[count];
    for (int i = 0; i < count; ++i) {
        host_keys[i] = i;
        host_values[i] = i * 10;
    }
    keys.write(host_keys);
    values.write(host_values);

    provider.run<RSCF_MultiMapInsertGroup<32>>(
        {static_cast<RSize>(count) * 32}, hash_multi_map, keys, values, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    for (int i = 0; i < count; ++i) {
        RS_Vector<int> find_results(2, &provider);
        find_results.fill(0);
        provider.run<RSCF_MultiMapFind>({1}, hash_multi_map, host_keys[i], 1, find_results);
        RS_Vector<int> host_find = find_results.move(&host_provider);
        EXPECT_EQ(host_find[1], host_values[i]);
        find_results.free();
        host_find.free();
    }

    delete[] host_keys;
    delete[] host_values;
    keys.free();
    values.free();
    results.free();
    host_results.free();
    hash_multi_map.free();
}

TEST_F(RS_HashMultiMapTest, InsertWithGroup4) {
    RSize max_entry_num = 2048;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    const int count = 1000;
    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<int> values(count, &provider);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    auto host_keys = new uint32_t[count];
    auto host_values = new int[count];
    for (int i = 0; i < count; ++i) {
        host_keys[i] = i;
        host_values[i] = i * 10;
    }
    keys.write(host_keys);
    values.write(host_values);

    provider.run<RSCF_MultiMapInsertGroup<4>>(
        {static_cast<RSize>(count) * 4}, hash_multi_map, keys, values, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    for (int i = 0; i < count; ++i) {
        RS_Vector<int> find_results(2, &provider);
        find_results.fill(0);
        provider.run<RSCF_MultiMapFind>({1}, hash_multi_map, host_keys[i], 1, find_results);
        RS_Vector<int> host_find = find_results.move(&host_provider);
        EXPECT_EQ(host_find[1], host_values[i]);
        find_results.free();
        host_find.free();
    }

    delete[] host_keys;
    delete[] host_values;
    keys.free();
    values.free();
    results.free();
    host_results.free();
    hash_multi_map.free();
}
#endif

/*****************************************************************
 * Test cases for RS_HashMultiMap::move()
 *
 * Test cases:
 * 1. Move to host after a single insertion.
 *****************************************************************/

TEST_F(RS_HashMultiMapTest, MoveToHostAfterSingleInsertion) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    uint32_t key = 42;
    int value = 100;

    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_MultiMapInsertSingle>({1}, hash_multi_map, key, value, insert_results);
    RS_Vector<int> host_insert = insert_results.move(&host_provider);
    EXPECT_EQ(host_insert[0], 1);

    auto host_hash_multi_map = hash_multi_map.move(&host_provider);
    EXPECT_EQ(host_hash_multi_map.count(), 1);
    auto list_view_host = host_hash_multi_map.find(key);
    EXPECT_EQ(list_view_host.size(), 1);
    EXPECT_EQ(list_view_host.get(0), value);

    insert_results.free();
    host_insert.free();
    host_hash_multi_map.free();
    hash_multi_map.free();
}

/*****************************************************************
 * Test cases for RS_HashMultiMap::copy()
 *
 * Test cases:
 * 1. Copy to a new device multimap after a single insertion.
 *****************************************************************/

TEST_F(RS_HashMultiMapTest, CopyToDeviceAfterSingleInsertion) {
    RSize max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(max_entry_num, max_entry_num, &provider);

    uint32_t key = 42;
    int value = 100;

    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_MultiMapInsertSingle>({1}, hash_multi_map, key, value, insert_results);
    RS_Vector<int> host_insert = insert_results.move(&host_provider);
    EXPECT_EQ(host_insert[0], 1);

    RS_HashMultiMap<uint32_t, int> hash_multi_map_copy(max_entry_num, max_entry_num, &provider);
    hash_multi_map_copy.copy(hash_multi_map);

    RS_Vector<int> find_results(2, &provider);
    find_results.fill(0);
    provider.run<RSCF_MultiMapFind>({1}, hash_multi_map_copy, key, 1, find_results);
    RS_Vector<int> host_find = find_results.move(&host_provider);
    EXPECT_EQ(host_find[0], 1);
    EXPECT_EQ(host_find[1], value);

    insert_results.free();
    host_insert.free();
    find_results.free();
    host_find.free();
    hash_multi_map.free();
    hash_multi_map_copy.free();
}

/*****************************************************************
 * Test cases for RS_HashMultiMap::resize()
 *
 * Test cases:
 * 1. Insert, resize and find an existing key.
 *****************************************************************/

TEST_F(RS_HashMultiMapTest, ResizeAndFindExistingKey) {
    RSize initial_max_entry_num = 1024;
    RS_HashMultiMap<uint32_t, int> hash_multi_map(
        initial_max_entry_num, initial_max_entry_num, &provider
    );

    uint32_t key = 42;
    int value = 100;

    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_MultiMapInsertSingle>({1}, hash_multi_map, key, value, insert_results);

    RSize new_max_entry_num = 2048;
    hash_multi_map.resize(new_max_entry_num, 0, true);

    RS_Vector<int> find_results(2, &provider);
    find_results.fill(0);
    provider.run<RSCF_MultiMapFind>({1}, hash_multi_map, key, 1, find_results);
    RS_Vector<int> host_find = find_results.move(&host_provider);
    EXPECT_EQ(host_find[0], 1);
    EXPECT_EQ(host_find[1], value);

    insert_results.free();
    find_results.free();
    host_find.free();
    hash_multi_map.free();
}
