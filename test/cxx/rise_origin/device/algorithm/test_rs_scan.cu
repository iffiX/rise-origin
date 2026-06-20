#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/algorithm/rs_scan.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_Scan computes exclusive and inclusive prefix sums and prefix
 * scans with a custom operator.
 *
 * - Correctness Verification
 *   Scan output is checked element by element against a host
 *   reference computed with a simple prefix sum loop. Large random
 *   data sets give wide coverage.
 *
 * - Boundary Conditions
 *   Empty input, single element, and two element inputs guard
 *   against off by one errors.
 *
 * - Edge Cases
 *   Deterministic input verifies the init value of an exclusive scan
 *   and the running total of an inclusive scan.
 *
 * - Algorithm Specific Features
 *   exclusiveSum and inclusiveSum use built in addition.
 *   exclusiveScan takes a custom operator and an init value.
 *   inclusiveScan takes a custom operator.
 *****************************************************************/

struct SumOp {
    HOST_DEVICE_QUALIFIER int operator()(int lhs, int rhs) { return lhs + rhs; }
};

class RS_ScanTest : public ::testing::Test {
   protected:
    void SetUp() override {
        RS_TEST_SETUP_DEVICE_PROVIDER();
        host_provider.init(0);
    }

    void TearDown() override {
        provider.free();
        host_provider.free();
    }

    std::vector<int> getRandomData(RSize size) {
        static std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(0, 1000);
        std::vector<int> data(size);
        for (RSize i = 0; i < size; ++i) {
            data[i] = dist(rng);
        }
        return data;
    }

    std::vector<int> referenceExclusiveSum(const std::vector<int>& in) {
        std::vector<int> out(in.size());
        int sum = 0;
        for (RSize i = 0; i < in.size(); ++i) {
            out[i] = sum;
            sum += in[i];
        }
        return out;
    }

    std::vector<int> referenceInclusiveSum(const std::vector<int>& in) {
        std::vector<int> out(in.size());
        int sum = 0;
        for (RSize i = 0; i < in.size(); ++i) {
            sum += in[i];
            out[i] = sum;
        }
        return out;
    }

    RS_TEST_DEVICE_PROVIDER_TYPE provider;
    RS_MemoryCPUProvider host_provider;
};

/*****************************************************************
 * Test cases for RS_Scan::exclusiveSum
 *
 * Test cases:
 * 1. Random data verified against the host reference.
 * 2. Empty input.
 * 3. Single element gives a zero output.
 * 4. Two elements.
 * 5. Large random data.
 *****************************************************************/

TEST_F(RS_ScanTest, ExclusiveSum) {
    const RSize count = 10000;
    auto h_in = getRandomData(count);

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.exclusiveSum<int, int>(d_in, d_out);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());

    auto h_ref = referenceExclusiveSum(h_in);
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_out[i], h_ref[i]);
    }

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, ExclusiveSumEmpty) {
    const RSize count = 0;
    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);

    RS_Scan scanner(&provider);
    scanner.exclusiveSum<int, int>(d_in, d_out);

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, ExclusiveSumSingle) {
    const RSize count = 1;
    std::vector<int> h_in = {42};

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.exclusiveSum<int, int>(d_in, d_out);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());
    EXPECT_EQ(h_out[0], 0);

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, ExclusiveSumTwo) {
    const RSize count = 2;
    std::vector<int> h_in = {7, 11};

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.exclusiveSum<int, int>(d_in, d_out);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());
    EXPECT_EQ(h_out[0], 0);
    EXPECT_EQ(h_out[1], 7);

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, ExclusiveSumLargeRandom) {
    const RSize count = 100000;
    auto h_in = getRandomData(count);

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.exclusiveSum<int, int>(d_in, d_out);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());

    auto h_ref = referenceExclusiveSum(h_in);
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_out[i], h_ref[i]);
    }

    d_in.free();
    d_out.free();
    scanner.free();
}

/*****************************************************************
 * Test cases for RS_Scan::inclusiveSum
 *
 * Test cases:
 * 1. Random data verified against the host reference.
 * 2. Empty input.
 * 3. Single element gives the same value back.
 * 4. Two elements.
 * 5. Large random data.
 *****************************************************************/

TEST_F(RS_ScanTest, InclusiveSum) {
    const RSize count = 10000;
    auto h_in = getRandomData(count);

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.inclusiveSum<int, int>(d_in, d_out);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());

    auto h_ref = referenceInclusiveSum(h_in);
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_out[i], h_ref[i]);
    }

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, InclusiveSumEmpty) {
    const RSize count = 0;
    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);

    RS_Scan scanner(&provider);
    scanner.inclusiveSum<int, int>(d_in, d_out);

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, InclusiveSumSingle) {
    const RSize count = 1;
    std::vector<int> h_in = {42};

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.inclusiveSum<int, int>(d_in, d_out);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());
    EXPECT_EQ(h_out[0], 42);

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, InclusiveSumTwo) {
    const RSize count = 2;
    std::vector<int> h_in = {7, 11};

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.inclusiveSum<int, int>(d_in, d_out);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());
    EXPECT_EQ(h_out[0], 7);
    EXPECT_EQ(h_out[1], 18);

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, InclusiveSumLargeRandom) {
    const RSize count = 100000;
    auto h_in = getRandomData(count);

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.inclusiveSum<int, int>(d_in, d_out);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());

    auto h_ref = referenceInclusiveSum(h_in);
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_out[i], h_ref[i]);
    }

    d_in.free();
    d_out.free();
    scanner.free();
}

/*****************************************************************
 * Test cases for RS_Scan::exclusiveScan
 *
 * Test cases:
 * 1. Random data with a custom operator and init value.
 * 2. Empty input.
 * 3. Deterministic input verifies the init value is honored.
 *****************************************************************/

TEST_F(RS_ScanTest, ExclusiveScan) {
    const RSize count = 10000;
    const int init_value = 100;
    auto h_in = getRandomData(count);

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.exclusiveScan<int, int>(d_in, d_out, SumOp(), init_value);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());

    int acc = init_value;
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_out[i], acc);
        acc += h_in[i];
    }

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, ExclusiveScanEmpty) {
    const RSize count = 0;
    const int init_value = 7;
    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);

    RS_Scan scanner(&provider);
    scanner.exclusiveScan<int, int>(d_in, d_out, SumOp(), init_value);

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, ExclusiveScanInitValue) {
    const RSize count = 16;
    const int init_value = 999;

    std::vector<int> h_in(count);
    for (RSize i = 0; i < count; ++i) {
        h_in[i] = static_cast<int>(i + 1);
    }

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.exclusiveScan<int, int>(d_in, d_out, SumOp(), init_value);

    std::vector<int> h_out(count);
    d_out.read(h_out.data());

    EXPECT_EQ(h_out[0], init_value);
    int acc = init_value;
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_out[i], acc);
        acc += h_in[i];
    }

    d_in.free();
    d_out.free();
    scanner.free();
}

/*****************************************************************
 * Test cases for RS_Scan::inclusiveScan
 *
 * Test cases:
 * 1. Random data with a custom operator.
 * 2. Empty input.
 * 3. Deterministic input verifies the running total.
 *****************************************************************/

TEST_F(RS_ScanTest, InclusiveScan) {
    const RSize count = 10000;
    auto h_in = getRandomData(count);

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.inclusiveScan<int, int>(d_in, d_out, SumOp());

    std::vector<int> h_out(count);
    d_out.read(h_out.data());

    auto h_ref = referenceInclusiveSum(h_in);
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_out[i], h_ref[i]);
    }

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, InclusiveScanEmpty) {
    const RSize count = 0;
    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);

    RS_Scan scanner(&provider);
    scanner.inclusiveScan<int, int>(d_in, d_out, SumOp());

    d_in.free();
    d_out.free();
    scanner.free();
}

TEST_F(RS_ScanTest, InclusiveScanCustomOp) {
    const RSize count = 32;

    std::vector<int> h_in(count);
    for (RSize i = 0; i < count; ++i) {
        h_in[i] = static_cast<int>(i + 1);
    }

    RS_Array<int, 1> d_in({count}, &provider);
    RS_Array<int, 1> d_out({count}, &provider);
    d_in.write(h_in.data());

    RS_Scan scanner(&provider);
    scanner.inclusiveScan<int, int>(d_in, d_out, SumOp());

    std::vector<int> h_out(count);
    d_out.read(h_out.data());

    EXPECT_EQ(h_out[0], h_in[0]);
    int total = 0;
    for (RSize i = 0; i < count; ++i) total += h_in[i];
    EXPECT_EQ(h_out[count - 1], total);

    auto h_ref = referenceInclusiveSum(h_in);
    for (RSize i = 0; i < count; ++i) {
        EXPECT_EQ(h_out[i], h_ref[i]);
    }

    d_in.free();
    d_out.free();
    scanner.free();
}
