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
 * RS_HashSet is a device hash set that stores keys across two
 * cuckoo tables. Each call adds 1 to the key so that 0 marks an
 * empty cell.
 *
 * - Construction & Initialization
 *   Constructor with capacity, zero capacity, large capacity,
 *   default constructor, bucket_num_per_table, size(), empty()
 *
 * - Lifetime
 *   free(), multiple free()
 *
 * - Insert (Device)
 *   insert() one key, many keys, duplicate key, fill until full
 *
 * - Erase (Device)
 *   erase() existing key, missing key, empty set, all keys
 *
 * - Contains (Device)
 *   contains() existing key, missing key, after erase
 *
 * - Resize
 *   resize() keeps inserted keys reachable
 *
 * The device probes below use the bare RS_ComputeFunction path.
 * Each thread works on one key with group size 1 and writes its
 * boolean outcome into an int result vector that is moved to the
 * host and checked. The host then verifies set state with the
 * host side count() and contains() methods.
 *****************************************************************/

class RS_HashSetTest : public ::testing::Test {
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

class RSCF_HashSetInsert : public RS_ComputeFunction<RSCF_HashSetInsert> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_HashSet<uint32_t> set, RS_Vector<uint32_t> keys, RS_Vector<int> results) {
        if (index1D() < keys.size()) {
            results[index1D()] = set.insert(keys[index1D()]) ? 1 : 0;
        }
    }
};

class RSCF_HashSetErase : public RS_ComputeFunction<RSCF_HashSetErase> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_HashSet<uint32_t> set, RS_Vector<uint32_t> keys, RS_Vector<int> results) {
        if (index1D() < keys.size()) {
            results[index1D()] = set.erase(keys[index1D()]) ? 1 : 0;
        }
    }
};

class RSCF_HashSetContains : public RS_ComputeFunction<RSCF_HashSetContains> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_HashSet<uint32_t> set, RS_Vector<uint32_t> keys, RS_Vector<int> results) {
        if (index1D() < keys.size()) {
            results[index1D()] = set.contains(keys[index1D()]) ? 1 : 0;
        }
    }
};

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
template <int GroupSize, typename HashSet>
class RSCF_HashSetInsertGroup : public RS_ComputeFunction<RSCF_HashSetInsertGroup<GroupSize, HashSet>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashSetInsertGroup<GroupSize, HashSet>>;
    using BaseType::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void
    run(HashSet set, RS_Vector<uint32_t> keys, RS_Vector<int> results) {
        RIndex group_idx = index1D() / GroupSize;
        RIndex lane_id = index1D() % GroupSize;
        if (group_idx < keys.size()) {
            bool is_inserted = set.template insert<GroupSize>(keys[group_idx]);
            if (lane_id == 0) results[group_idx] = is_inserted ? 1 : 0;
        }
    }
};

template <int GroupSize, typename HashSet>
class RSCF_HashSetEraseGroup : public RS_ComputeFunction<RSCF_HashSetEraseGroup<GroupSize, HashSet>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashSetEraseGroup<GroupSize, HashSet>>;
    using BaseType::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void
    run(HashSet set, RS_Vector<uint32_t> keys, RS_Vector<int> results) {
        RIndex group_idx = index1D() / GroupSize;
        RIndex lane_id = index1D() % GroupSize;
        if (group_idx < keys.size()) {
            bool is_erased = set.template erase<GroupSize>(keys[group_idx]);
            if (lane_id == 0) results[group_idx] = is_erased ? 1 : 0;
        }
    }
};

template <int GroupSize, typename HashSet>
class RSCF_HashSetContainsGroup
    : public RS_ComputeFunction<RSCF_HashSetContainsGroup<GroupSize, HashSet>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashSetContainsGroup<GroupSize, HashSet>>;
    using BaseType::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void
    run(HashSet set, RS_Vector<uint32_t> keys, RS_Vector<int> results) {
        RIndex group_idx = index1D() / GroupSize;
        RIndex lane_id = index1D() % GroupSize;
        if (group_idx < keys.size()) {
            bool is_found = set.template contains<GroupSize>(keys[group_idx]);
            if (lane_id == 0) results[group_idx] = is_found ? 1 : 0;
        }
    }
};
#endif

/*****************************************************************
 * Test cases for RS_HashSet construction and initialization
 *
 * Test cases:
 * 1. Construct with valid parameters.
 * 2. Construct with zero capacity.
 * 3. Construct with large capacity.
 * 4. Default constructor.
 *****************************************************************/

TEST_F(RS_HashSetTest, ConstructWithValidParameters) {
    RSize max_entry_num = 1024;
    RSize bucket_size = RS_HashSet<uint32_t>::BucketSize;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_set.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_set.empty());
    EXPECT_EQ(hash_set.size(), expected_bucket_num_per_table * 2 * bucket_size);
    hash_set.free();
}

TEST_F(RS_HashSetTest, ConstructWithZeroCapacity) {
    RSize max_entry_num = 0;
    RSize bucket_size = RS_HashSet<uint32_t>::BucketSize;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_set.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_set.empty());
    EXPECT_EQ(hash_set.size(), expected_bucket_num_per_table * 2 * bucket_size);
    hash_set.free();
}

TEST_F(RS_HashSetTest, ConstructWithLargeCapacity) {
    RSize max_entry_num = 1 << 20;
    RSize bucket_size = RS_HashSet<uint32_t>::BucketSize;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RSize expected_bucket_num_per_table =
        nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, bucket_size), bucket_size));
    EXPECT_EQ(hash_set.bucket_num_per_table, expected_bucket_num_per_table);
    EXPECT_FALSE(hash_set.empty());
    EXPECT_EQ(hash_set.size(), expected_bucket_num_per_table * 2 * bucket_size);
    hash_set.free();
}

TEST_F(RS_HashSetTest, DefaultConstructor) {
    RS_HashSet<uint32_t> hash_set;
    EXPECT_EQ(hash_set.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_set.empty());
    EXPECT_EQ(hash_set.size(), 0);
}

/*****************************************************************
 * Test cases for RS_HashSet::free()
 *
 * Test cases:
 * 1. Free after initialization.
 * 2. Multiple frees.
 *****************************************************************/

TEST_F(RS_HashSetTest, FreeAfterInitialization) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);
    hash_set.free();
    EXPECT_EQ(hash_set.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_set.empty());
}

TEST_F(RS_HashSetTest, MultipleFrees) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);
    hash_set.free();
    hash_set.free();
    hash_set.free();
    EXPECT_EQ(hash_set.bucket_num_per_table, 0);
    EXPECT_TRUE(hash_set.empty());
}

/*****************************************************************
 * Test cases for RS_HashSet::insert()
 *
 * Test cases:
 * 1. Insert into empty hash set.
 * 2. Multiple insertions.
 * 3. Insert duplicate key.
 * 4. Insert until full.
 *****************************************************************/

TEST_F(RS_HashSetTest, InsertIntoEmptyHashSet) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {42};
    keys.write(key_data);
    RS_Vector<int> results(1, &provider);
    results.fill(0);

    provider.run<RSCF_HashSetInsert>({1}, hash_set, keys, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    EXPECT_EQ(host_results[0], 1);

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 1);

    host_results.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, MultipleInsertions) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    const int count = 10000;
    const int unique_keys = 1000;
    auto key_data = new uint32_t[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i % unique_keys;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    keys.write(key_data);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashSetInsert>({(RSize)count}, hash_set, keys, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), unique_keys);

    delete[] key_data;
    host_results.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, InsertDuplicateKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {42};
    keys.write(key_data);
    RS_Vector<int> results(1, &provider);

    results.fill(0);
    provider.run<RSCF_HashSetInsert>({1}, hash_set, keys, results);
    RS_Vector<int> host_results1 = results.move(&host_provider);
    EXPECT_EQ(host_results1[0], 1);

    RS_Vector<int> results2(1, &provider);
    results2.fill(0);
    provider.run<RSCF_HashSetInsert>({1}, hash_set, keys, results2);
    RS_Vector<int> host_results2 = results2.move(&host_provider);
    EXPECT_EQ(host_results2[0], 1);

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 1);

    host_results1.free();
    host_results2.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, InsertUntilFull) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    const int count = 10000;
    auto key_data = new uint32_t[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    keys.write(key_data);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashSetInsert>({(RSize)count}, hash_set, keys, results);

    RS_Vector<int> host_results = results.move(&host_provider);
    int successful_inserts = 0;
    for (int i = 0; i < count; ++i) {
        successful_inserts += host_results[i];
    }
    EXPECT_EQ((RSize)successful_inserts, max_entry_num);

    delete[] key_data;
    host_results.free();
    keys.free();
    hash_set.free();
}

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
TEST_F(RS_HashSetTest, InsertWithGroup32) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t, uint32_t, RS_GenericHasher, 32> hash_set(max_entry_num, &provider);

    const int count = 1000;
    auto key_data = new uint32_t[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    keys.write(key_data);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashSetInsertGroup<32, RS_HashSet<uint32_t, uint32_t, RS_GenericHasher, 32>>>(
        {static_cast<RSize>(count) * 32}, hash_set, keys, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), count);

    delete[] key_data;
    host_results.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, InsertWithGroup4) {
    RSize max_entry_num = 2048;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    const int count = 1000;
    auto key_data = new uint32_t[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    keys.write(key_data);
    RS_Vector<int> results(count, &provider);
    results.fill(0);

    provider.run<RSCF_HashSetInsertGroup<4, RS_HashSet<uint32_t>>>(
        {static_cast<RSize>(count) * 4}, hash_set, keys, results
    );

    RS_Vector<int> host_results = results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_results[i], 1);
    }

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), count);

    delete[] key_data;
    host_results.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}
#endif

/*****************************************************************
 * Test cases for RS_HashSet::erase()
 *
 * Test cases:
 * 1. Erase existing key.
 * 2. Erase non-existent key.
 * 3. Erase from empty hash set.
 * 4. Erase all keys.
 * 5. Erase with group size 32.
 *****************************************************************/

TEST_F(RS_HashSetTest, EraseExistingKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {42};
    keys.write(key_data);
    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashSetInsert>({1}, hash_set, keys, insert_results);

    RS_Vector<int> erase_results(1, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashSetErase>({1}, hash_set, keys, erase_results);

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    EXPECT_EQ(host_erase[0], 1);

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);
    EXPECT_FALSE(host_hash_set.contains(key_data[0]));

    insert_results.free();
    host_erase.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, EraseNonExistentKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {12345};
    keys.write(key_data);
    RS_Vector<int> erase_results(1, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashSetErase>({1}, hash_set, keys, erase_results);

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    EXPECT_EQ(host_erase[0], 0);

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);

    host_erase.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, EraseFromEmptyHashSet) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {7};
    keys.write(key_data);
    RS_Vector<int> erase_results(1, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashSetErase>({1}, hash_set, keys, erase_results);

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    EXPECT_EQ(host_erase[0], 0);

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);

    host_erase.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, EraseAllKeys) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    const int count = 100;
    auto key_data = new uint32_t[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    keys.write(key_data);
    RS_Vector<int> insert_results(count, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashSetInsert>({(RSize)count}, hash_set, keys, insert_results);

    RS_Vector<int> erase_results(count, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashSetErase>({(RSize)count}, hash_set, keys, erase_results);

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_erase[i], 1);
    }

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);

    delete[] key_data;
    insert_results.free();
    host_erase.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
TEST_F(RS_HashSetTest, EraseWithGroup32) {
    RSize max_entry_num = 1024;
    using HashSetType = RS_HashSet<uint32_t, uint32_t, RS_GenericHasher, 32>;
    HashSetType hash_set(max_entry_num, &provider);

    const int count = 100;
    auto key_data = new uint32_t[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    keys.write(key_data);
    RS_Vector<int> insert_results(count, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashSetInsertGroup<32, HashSetType>>(
        {static_cast<RSize>(count) * 32}, hash_set, keys, insert_results
    );

    RS_Vector<int> erase_results(count, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashSetEraseGroup<32, HashSetType>>(
        {static_cast<RSize>(count) * 32}, hash_set, keys, erase_results
    );

    RS_Vector<int> host_erase = erase_results.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_erase[i], 1);
    }

    auto host_hash_set = hash_set.move(&host_provider);
    EXPECT_EQ(host_hash_set.count(), 0);

    delete[] key_data;
    insert_results.free();
    host_erase.free();
    host_hash_set.free();
    keys.free();
    hash_set.free();
}
#endif

/*****************************************************************
 * Test cases for RS_HashSet::contains()
 *
 * Test cases:
 * 1. Contains existing key.
 * 2. Contains non-existent key.
 * 3. Contains after erase.
 * 4. Contains with group size 32.
 *****************************************************************/

TEST_F(RS_HashSetTest, ContainsExistingKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {42};
    keys.write(key_data);
    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashSetInsert>({1}, hash_set, keys, insert_results);

    RS_Vector<int> found(1, &provider);
    found.fill(0);
    provider.run<RSCF_HashSetContains>({1}, hash_set, keys, found);

    RS_Vector<int> host_found = found.move(&host_provider);
    EXPECT_EQ(host_found[0], 1);

    insert_results.free();
    host_found.free();
    keys.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, ContainsNonExistentKey) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {12345};
    keys.write(key_data);
    RS_Vector<int> found(1, &provider);
    found.fill(0);
    provider.run<RSCF_HashSetContains>({1}, hash_set, keys, found);

    RS_Vector<int> host_found = found.move(&host_provider);
    EXPECT_EQ(host_found[0], 0);

    host_found.free();
    keys.free();
    hash_set.free();
}

TEST_F(RS_HashSetTest, ContainsAfterErase) {
    RSize max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {99};
    keys.write(key_data);

    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashSetInsert>({1}, hash_set, keys, insert_results);

    RS_Vector<int> erase_results(1, &provider);
    erase_results.fill(0);
    provider.run<RSCF_HashSetErase>({1}, hash_set, keys, erase_results);

    RS_Vector<int> found(1, &provider);
    found.fill(0);
    provider.run<RSCF_HashSetContains>({1}, hash_set, keys, found);

    RS_Vector<int> host_found = found.move(&host_provider);
    EXPECT_EQ(host_found[0], 0);

    insert_results.free();
    erase_results.free();
    host_found.free();
    keys.free();
    hash_set.free();
}

// Group size greater than one is only supported on the cuda backend.
#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
TEST_F(RS_HashSetTest, ContainsWithGroup32) {
    RSize max_entry_num = 2048;
    using HashSetType = RS_HashSet<uint32_t, uint32_t, RS_GenericHasher, 32>;
    HashSetType hash_set(max_entry_num, &provider);

    const int count = 64;
    auto key_data = new uint32_t[count];
    for (int i = 0; i < count; ++i) {
        key_data[i] = i;
    }

    RS_Vector<uint32_t> keys(count, &provider);
    keys.write(key_data);
    RS_Vector<int> insert_results(count, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashSetInsertGroup<32, HashSetType>>(
        {static_cast<RSize>(count) * 32}, hash_set, keys, insert_results
    );

    RS_Vector<int> found(count, &provider);
    found.fill(0);
    provider.run<RSCF_HashSetContainsGroup<32, HashSetType>>(
        {static_cast<RSize>(count) * 32}, hash_set, keys, found
    );

    RS_Vector<int> host_found = found.move(&host_provider);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host_found[i], 1);
    }

    delete[] key_data;
    insert_results.free();
    host_found.free();
    keys.free();
    hash_set.free();
}
#endif

/*****************************************************************
 * Test cases for RS_HashSet::resize()
 *
 * Test cases:
 * 1. Insert, resize and find existing key.
 *****************************************************************/

TEST_F(RS_HashSetTest, ResizeAndFindKey) {
    RSize initial_max_entry_num = 1024;
    RS_HashSet<uint32_t> hash_set(initial_max_entry_num, &provider);

    RS_Vector<uint32_t> keys(1, &provider);
    uint32_t key_data[1] = {42};
    keys.write(key_data);
    RS_Vector<int> insert_results(1, &provider);
    insert_results.fill(0);
    provider.run<RSCF_HashSetInsert>({1}, hash_set, keys, insert_results);

    RSize new_max_entry_num = 2048;
    hash_set.resize(new_max_entry_num);

    RS_Vector<int> found(1, &provider);
    found.fill(0);
    provider.run<RSCF_HashSetContains>({1}, hash_set, keys, found);

    RS_Vector<int> host_found = found.move(&host_provider);
    EXPECT_EQ(host_found[0], 1);

    insert_results.free();
    host_found.free();
    keys.free();
    hash_set.free();
}
