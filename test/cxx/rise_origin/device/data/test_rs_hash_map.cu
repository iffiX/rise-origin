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
 * RS_HashMap is a concurrent hash map on the device that uses a
 * cuckoo strategy across two tables. The values live in a separate
 * array, and the tables store the index of each value. This keeps
 * the cost of eviction low during insert.
 *
 * - Construction & Initialization
 *   Constructor with capacity, default constructor, free()
 *
 * - Capacity & Size
 *   empty(), size(), bucket_num_per_table, count()
 *
 * - Hash Map Operations on the device
 *   insert() single and grouped, find()
 *
 * - Memory Management
 *   resize() with data kept
 *****************************************************************/

struct LargeValue {
    int data[3];
};

class RS_HashMapTest : public ::testing::Test {
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
 * Compute functions that probe the hash map on the device
 *****************************************************************/

class RSCF_HashMapInsertSingle : public RS_ComputeFunction<RSCF_HashMapInsertSingle> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_HashMap<uint32_t, LargeValue> map, uint32_t key, LargeValue value, RS_Vector<int> results) {
        if (index1D() == 0) {
            results[0] = map.insert(key, value) ? 1 : 0;
        }
    }
};

class RSCF_HashMapInsertMany : public RS_ComputeFunction<RSCF_HashMapInsertMany> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_HashMap<uint32_t, LargeValue> map,
        RS_Vector<uint32_t> keys,
        RS_Vector<LargeValue> values,
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
class RSCF_HashMapInsertGroup : public RS_ComputeFunction<RSCF_HashMapInsertGroup<GroupSize>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashMapInsertGroup<GroupSize>>;
    using BaseType::index1D;

    template <typename HashMap>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        HashMap map,
        RS_Vector<uint32_t> keys,
        RS_Vector<LargeValue> values,
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

class RSCF_HashMapFindSingle : public RS_ComputeFunction<RSCF_HashMapFindSingle> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_HashMap<uint32_t, LargeValue> map, uint32_t key, RS_Vector<int> results) {
        if (index1D() == 0) {
            LargeValue* value = map.find(key);
            if (value != nullptr) {
                results[0] = 1;
                results[1] = value->data[0];
                results[2] = value->data[1];
                results[3] = value->data[2];
            } else {
                results[0] = 0;
            }
        }
    }
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
    RSize max_entry_num = 1024;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
    hash_map.free();
}

TEST_F(RS_HashMapTest, ConstructWithZeroCapacity) {
    RSize max_entry_num = 0;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
    hash_map.free();
}

TEST_F(RS_HashMapTest, ConstructWithLargeCapacity) {
    RSize max_entry_num = 1 << 20;
    RSize bucket_size = RS_HashFlatMap<uint32_t, int>::BucketSize;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_map.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), expected_bucket_num_per_table * 2 * bucket_size);
    hash_map.free();
}

TEST_F(RS_HashMapTest, DefaultConstructor) {
    RS_HashMap<uint32_t, LargeValue> hash_map;

    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
    EXPECT_EQ(hash_map.size(), 0);
}

/*****************************************************************
 * Test cases for RS_HashMap::free()
 *
 * Test cases:
 * 1. Free after initialization.
 * 2. Multiple frees.
 *****************************************************************/

TEST_F(RS_HashMapTest, FreeAfterInitialization) {
    RSize max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    hash_map.free();

    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
}

TEST_F(RS_HashMapTest, MultipleFrees) {
    RSize max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    hash_map.free();
    hash_map.free();
    hash_map.free();

    EXPECT_EQ(hash_map.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_map.empty());
}

/*****************************************************************
 * Test cases for RS_HashMap::insert()
 *
 * Test cases:
 * 1. Insert into an empty hash map.
 * 2. Multiple insertions.
 * 3. Insert a duplicate key.
 * 4. Insert with group size 32.
 * 5. Insert with group size 4.
 *****************************************************************/

TEST_F(RS_HashMapTest, InsertIntoEmptyHashMap) {
    RSize max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    uint32_t key = 42;
    LargeValue value = {{100, 200, 300}};

    RS_Vector<int> results(1, &provider);
    results.fill(0);

    provider.run<RSCF_HashMapInsertSingle>({1}, hash_map, key, value, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    EXPECT_EQ(host_results[0], 1);

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 1);

    LargeValue* val_ptr = host_hash_map.find(key);
    ASSERT_NE(val_ptr, nullptr);
    EXPECT_EQ(val_ptr->data[0], value.data[0]);
    EXPECT_EQ(val_ptr->data[1], value.data[1]);
    EXPECT_EQ(val_ptr->data[2], value.data[2]);

    results.free();
    host_results.free();
    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashMapTest, MultipleInsertions) {
    RSize max_entry_num = 2048;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);
    const int count = 1000;

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<LargeValue> values(count, &provider);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    auto host_keys = new uint32_t[count];
    auto host_values = new LargeValue[count];
    for (int i = 0; i < count; ++i) {
        host_keys[i] = i;
        host_values[i] = {{i, i * 2, i * 3}};
    }
    keys.write(host_keys);
    values.write(host_values);

    provider.run<RSCF_HashMapInsertMany>(
        {static_cast<RSize>(count)}, hash_map, keys, values, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), count);

    for (int i = 0; i < count; ++i) {
        LargeValue* val_ptr = host_hash_map.find(i);
        ASSERT_NE(val_ptr, nullptr);
        EXPECT_EQ(val_ptr->data[0], i);
        EXPECT_EQ(val_ptr->data[1], i * 2);
        EXPECT_EQ(val_ptr->data[2], i * 3);
    }

    delete[] host_keys;
    delete[] host_values;
    keys.free();
    values.free();
    results.free();
    host_results.free();
    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashMapTest, InsertDuplicateKey) {
    RSize max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);

    uint32_t key = 42;
    LargeValue value1 = {{100, 200, 300}};
    LargeValue value2 = {{400, 500, 600}};

    RS_Vector<int> results(1, &provider);
    results.fill(0);

    provider.run<RSCF_HashMapInsertSingle>({1}, hash_map, key, value1, results);
    RS_Vector<int> host_results1 = results.move(&host_provider);
    EXPECT_EQ(host_results1[0], 1);

    results.init(1, &provider);
    results.fill(0);
    provider.run<RSCF_HashMapInsertSingle>({1}, hash_map, key, value2, results);
    RS_Vector<int> host_results2 = results.move(&host_provider);
    EXPECT_EQ(host_results2[0], 1);

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), 1);

    LargeValue* val_ptr = host_hash_map.find(key);
    ASSERT_NE(val_ptr, nullptr);
    EXPECT_EQ(val_ptr->data[0], value2.data[0]);
    EXPECT_EQ(val_ptr->data[1], value2.data[1]);
    EXPECT_EQ(val_ptr->data[2], value2.data[2]);

    results.free();
    host_results1.free();
    host_results2.free();
    host_hash_map.free();
    hash_map.free();
}

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
TEST_F(RS_HashMapTest, InsertWithGroup32) {
    RSize max_entry_num = 2048;
    RS_HashMap<uint32_t, LargeValue, uint32_t, RS_GenericHasher, 32> hash_map(
        max_entry_num, &provider
    );
    const int count = 1000;

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<LargeValue> values(count, &provider);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    auto host_keys = new uint32_t[count];
    auto host_values = new LargeValue[count];
    for (int i = 0; i < count; ++i) {
        host_keys[i] = i;
        host_values[i] = {{i, i * 2, i * 3}};
    }
    keys.write(host_keys);
    values.write(host_values);

    provider.run<RSCF_HashMapInsertGroup<32>>(
        {static_cast<RSize>(count) * 32}, hash_map, keys, values, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), count);

    for (int i = 0; i < count; ++i) {
        LargeValue* val_ptr = host_hash_map.find(i);
        ASSERT_NE(val_ptr, nullptr);
        EXPECT_EQ(val_ptr->data[0], i);
        EXPECT_EQ(val_ptr->data[1], i * 2);
        EXPECT_EQ(val_ptr->data[2], i * 3);
    }

    delete[] host_keys;
    delete[] host_values;
    keys.free();
    values.free();
    results.free();
    host_results.free();
    host_hash_map.free();
    hash_map.free();
}

TEST_F(RS_HashMapTest, InsertWithGroup4) {
    RSize max_entry_num = 2048;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &provider);
    const int count = 1000;

    RS_Vector<uint32_t> keys(count, &provider);
    RS_Vector<LargeValue> values(count, &provider);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    auto host_keys = new uint32_t[count];
    auto host_values = new LargeValue[count];
    for (int i = 0; i < count; ++i) {
        host_keys[i] = i;
        host_values[i] = {{i, i * 2, i * 3}};
    }
    keys.write(host_keys);
    values.write(host_values);

    provider.run<RSCF_HashMapInsertGroup<4>>(
        {static_cast<RSize>(count) * 4}, hash_map, keys, values, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    auto host_hash_map = hash_map.move(&host_provider);
    EXPECT_EQ(host_hash_map.count(), count);

    for (int i = 0; i < count; ++i) {
        LargeValue* val_ptr = host_hash_map.find(i);
        ASSERT_NE(val_ptr, nullptr);
        EXPECT_EQ(val_ptr->data[0], i);
        EXPECT_EQ(val_ptr->data[1], i * 2);
        EXPECT_EQ(val_ptr->data[2], i * 3);
    }

    delete[] host_keys;
    delete[] host_values;
    keys.free();
    values.free();
    results.free();
    host_results.free();
    host_hash_map.free();
    hash_map.free();
}
#endif

/*****************************************************************
 * Test cases for RS_HashMap insert on host
 *
 * Test cases:
 * 1. Insert until full on host, check successful inserts and values.
 *****************************************************************/

TEST_F(RS_HashMapTest, InsertUntilFullHost) {
    RSize max_entry_num = 1024;
    RSize count = 200000;
    RS_HashMap<uint32_t, LargeValue> hash_map(max_entry_num, &host_provider);

    int successful_inserts = 0;
    for (uint32_t i = 0; i < count; ++i) {
        LargeValue value = {
            {static_cast<int>(i), static_cast<int>(i * 2), static_cast<int>(i * 3)}
        };
        if (hash_map.insert(i, value)) {
            successful_inserts++;
        }
    }

    EXPECT_EQ(successful_inserts, max_entry_num);
    EXPECT_EQ(hash_map.count(), max_entry_num);

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
    hash_map.free();
}

/*****************************************************************
 * Test cases for RS_HashMap::resize()
 *
 * Test cases:
 * 1. Insert, resize and find an existing key.
 *****************************************************************/

TEST_F(RS_HashMapTest, ResizeAndFindExistingKey) {
    RSize initial_max_entry_num = 1024;
    RS_HashMap<uint32_t, LargeValue> hash_map(initial_max_entry_num, &provider);

    uint32_t key = 42;
    LargeValue initial_value = {{100, 200, 300}};

    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashMapInsertSingle>({1}, hash_map, key, initial_value, insert_results);

    RSize new_max_entry_num = 2048;
    hash_map.resize(new_max_entry_num);

    RS_Vector<int> find_results(4, &provider);
    find_results.fill(0);
    provider.run<RSCF_HashMapFindSingle>({1}, hash_map, key, find_results);

    RS_Vector<int> host_find = find_results.move(&host_provider);
    EXPECT_EQ(host_find[0], 1);
    EXPECT_EQ(host_find[1], initial_value.data[0]);
    EXPECT_EQ(host_find[2], initial_value.data[1]);
    EXPECT_EQ(host_find[3], initial_value.data[2]);

    insert_results.free();
    find_results.free();
    host_find.free();
    hash_map.free();
}
