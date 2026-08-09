#include <gtest/gtest.h>

#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_morton.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_MortonCode interleaves three grid coordinates into one Z order
 * code and splits a code back into the coordinates it was built
 * from. It holds no state, so every member is static, and the code
 * width decides how many bits each coordinate gets, 10 for a 32 bit
 * code and 21 for a 64 bit one.
 *
 * - Encoding
 *   toMortonCode() against a bit by bit reference that shares no
 *   code with the mask sequence under test
 *
 * - Decoding
 *   toCoordinate() returns what toMortonCode() was given, over each
 *   single axis and over a sampled three axis sweep
 *
 * - Boundary Conditions
 *   Zero, the largest coordinate, and coordinates above it, which
 *   saturate at the upper grid bound
 *
 * - Device Operations
 *   The same round trip driven from a compute function
 *****************************************************************/

/**
 * @brief Encodes a coordinate on the device and decodes the code straight back.
 */
template <typename CodeType>
class RSCF_MortonRoundTrip : public RS_ComputeFunction<RSCF_MortonRoundTrip<CodeType>> {
   public:
    using RS_ComputeFunction<RSCF_MortonRoundTrip>::index1D;

    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_Array<CodeType, 1> coordinates,
        RS_Array<CodeType, 1> codes,
        RS_Array<CodeType, 1> decoded_coordinates
    ) {
        RSize i = index1D();
        CodeType code = RS_MortonCode<CodeType>::toMortonCode(
            coordinates[i * 3], coordinates[i * 3 + 1], coordinates[i * 3 + 2]
        );
        codes[i] = code;

        CodeType x = 0, y = 0, z = 0;
        RS_MortonCode<CodeType>::toCoordinate(code, x, y, z);
        decoded_coordinates[i * 3] = x;
        decoded_coordinates[i * 3 + 1] = y;
        decoded_coordinates[i * 3 + 2] = z;
    }
};

class RS_MortonCodeTest : public ::testing::Test {
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

    /**
     * @brief Interleaves one bit at a time, sharing nothing with the shift and
     * mask sequence under test.
     */
    template <typename CodeType>
    static CodeType referenceMortonCode(CodeType x, CodeType y, CodeType z) {
        CodeType code = 0;
        for (RIndex bit = 0; bit < RS_MortonCode<CodeType>::CoordinateBits; bit++) {
            code |= ((x >> bit) & CodeType(1)) << (bit * 3);
            code |= ((y >> bit) & CodeType(1)) << (bit * 3 + 1);
            code |= ((z >> bit) & CodeType(1)) << (bit * 3 + 2);
        }
        return code;
    }

    /// @brief Coordinates spread over the whole range, sparse enough to sweep in three axes.
    template <typename CodeType>
    static std::vector<CodeType> sampleCoordinates() {
        constexpr CodeType coordinate_max = RS_MortonCode<CodeType>::CoordinateMax;
        std::vector<CodeType> samples{0, 1, 2, 3, 7, 8, coordinate_max - 1, coordinate_max};
        for (CodeType value = 5; value < coordinate_max; value = value * 3 + 1) {
            samples.push_back(value);
        }
        return samples;
    }

    /// @brief Encodes and decodes on the device, one coordinate per thread.
    template <typename CodeType, typename Provider>
    static void runRoundTripOnDevice(
        Provider& run_provider,
        const std::vector<CodeType>& coordinates,
        std::vector<CodeType>& codes_out,
        std::vector<CodeType>& decoded_coordinates_out
    ) {
        RSize coordinate_value_num = RSize(coordinates.size());
        RSize coordinate_num = coordinate_value_num / 3;
        RS_Array<CodeType, 1> d_coordinates({coordinate_value_num}, &run_provider);
        RS_Array<CodeType, 1> d_codes({coordinate_num}, &run_provider);
        RS_Array<CodeType, 1> d_decoded_coordinates({coordinate_value_num}, &run_provider);
        d_coordinates.write(coordinates.data());
        d_codes.fill(0);
        d_decoded_coordinates.fill(0);

        run_provider.template run<RSCF_MortonRoundTrip<CodeType>>(
            {coordinate_num}, d_coordinates, d_codes, d_decoded_coordinates
        );

        codes_out.resize(coordinate_num);
        decoded_coordinates_out.resize(coordinates.size());
        d_codes.read(codes_out.data());
        d_decoded_coordinates.read(decoded_coordinates_out.data());

        d_coordinates.free();
        d_codes.free();
        d_decoded_coordinates.free();
    }
};

/*****************************************************************
 * Test cases for the code width each type carries
 *
 * Test cases:
 * 1. A 32 bit code gives each coordinate 10 bits.
 * 2. A 64 bit code gives each coordinate 21 bits.
 *****************************************************************/

TEST_F(RS_MortonCodeTest, CodeWidth) {
    EXPECT_EQ(RS_MortonCode<uint32_t>::CoordinateBits, 10u);
    EXPECT_EQ(RS_MortonCode<uint32_t>::CoordinateMax, 1023u);
    EXPECT_EQ(RS_MortonCode<uint64_t>::CoordinateBits, 21u);
    EXPECT_EQ(RS_MortonCode<uint64_t>::CoordinateMax, 2097151ull);
}

/*****************************************************************
 * Test cases for encoding against an independent reference
 *
 * Test cases:
 * 1. Each axis swept alone over its whole range, others zero.
 * 2. A sampled sweep over all three axes together.
 * 3. Known codes at the two ends of the range.
 *****************************************************************/

TEST_F(RS_MortonCodeTest, EncodeSingleAxisFor32Bit) {
    for (uint32_t value = 0; value <= RS_MortonCode<uint32_t>::CoordinateMax; value++) {
        EXPECT_EQ(
            RS_MortonCode<uint32_t>::toMortonCode(value, 0, 0),
            referenceMortonCode<uint32_t>(value, 0, 0)
        ) << "x axis differs at " << value;
        EXPECT_EQ(
            RS_MortonCode<uint32_t>::toMortonCode(0, value, 0),
            referenceMortonCode<uint32_t>(0, value, 0)
        ) << "y axis differs at " << value;
        EXPECT_EQ(
            RS_MortonCode<uint32_t>::toMortonCode(0, 0, value),
            referenceMortonCode<uint32_t>(0, 0, value)
        ) << "z axis differs at " << value;
    }
}

TEST_F(RS_MortonCodeTest, EncodeSingleAxisFor64Bit) {
    for (uint64_t value = 0; value <= RS_MortonCode<uint64_t>::CoordinateMax; value += 7) {
        EXPECT_EQ(
            RS_MortonCode<uint64_t>::toMortonCode(value, 0, 0),
            referenceMortonCode<uint64_t>(value, 0, 0)
        ) << "x axis differs at " << value;
        EXPECT_EQ(
            RS_MortonCode<uint64_t>::toMortonCode(0, value, 0),
            referenceMortonCode<uint64_t>(0, value, 0)
        ) << "y axis differs at " << value;
        EXPECT_EQ(
            RS_MortonCode<uint64_t>::toMortonCode(0, 0, value),
            referenceMortonCode<uint64_t>(0, 0, value)
        ) << "z axis differs at " << value;
    }
}

TEST_F(RS_MortonCodeTest, EncodeThreeAxesFor32Bit) {
    std::vector<uint32_t> samples = sampleCoordinates<uint32_t>();
    for (uint32_t x : samples) {
        for (uint32_t y : samples) {
            for (uint32_t z : samples) {
                EXPECT_EQ(
                    RS_MortonCode<uint32_t>::toMortonCode(x, y, z),
                    referenceMortonCode<uint32_t>(x, y, z)
                ) << "differs at " << x << ", " << y << ", " << z;
            }
        }
    }
}

TEST_F(RS_MortonCodeTest, EncodeThreeAxesFor64Bit) {
    std::vector<uint64_t> samples = sampleCoordinates<uint64_t>();
    for (uint64_t x : samples) {
        for (uint64_t y : samples) {
            for (uint64_t z : samples) {
                EXPECT_EQ(
                    RS_MortonCode<uint64_t>::toMortonCode(x, y, z),
                    referenceMortonCode<uint64_t>(x, y, z)
                ) << "differs at " << x << ", " << y << ", " << z;
            }
        }
    }
}

TEST_F(RS_MortonCodeTest, EncodeRangeEnds) {
    EXPECT_EQ(RS_MortonCode<uint32_t>::toMortonCode(0, 0, 0), 0u);
    EXPECT_EQ(RS_MortonCode<uint64_t>::toMortonCode(0, 0, 0), 0ull);

    // Every coordinate bit set fills the low 30 bits of a 32 bit code.
    constexpr uint32_t coordinate_max_32 = RS_MortonCode<uint32_t>::CoordinateMax;
    EXPECT_EQ(
        RS_MortonCode<uint32_t>::toMortonCode(coordinate_max_32, coordinate_max_32, coordinate_max_32),
        0x3FFFFFFFu
    );

    // And the low 63 bits of a 64 bit code.
    constexpr uint64_t coordinate_max_64 = RS_MortonCode<uint64_t>::CoordinateMax;
    EXPECT_EQ(
        RS_MortonCode<uint64_t>::toMortonCode(coordinate_max_64, coordinate_max_64, coordinate_max_64),
        0x7FFFFFFFFFFFFFFFull
    );
}

/*****************************************************************
 * Test cases for decoding back to a coordinate
 *
 * Test cases:
 * 1. Each axis swept alone round trips.
 * 2. A sampled sweep over all three axes round trips.
 *****************************************************************/

TEST_F(RS_MortonCodeTest, DecodeSingleAxisFor32Bit) {
    for (uint32_t value = 0; value <= RS_MortonCode<uint32_t>::CoordinateMax; value++) {
        uint32_t x = 0, y = 0, z = 0;
        RS_MortonCode<uint32_t>::toCoordinate(
            RS_MortonCode<uint32_t>::toMortonCode(value, 0, 0), x, y, z
        );
        EXPECT_EQ(x, value);
        EXPECT_EQ(y, 0u);
        EXPECT_EQ(z, 0u);

        RS_MortonCode<uint32_t>::toCoordinate(
            RS_MortonCode<uint32_t>::toMortonCode(0, value, 0), x, y, z
        );
        EXPECT_EQ(x, 0u);
        EXPECT_EQ(y, value);
        EXPECT_EQ(z, 0u);

        RS_MortonCode<uint32_t>::toCoordinate(
            RS_MortonCode<uint32_t>::toMortonCode(0, 0, value), x, y, z
        );
        EXPECT_EQ(x, 0u);
        EXPECT_EQ(y, 0u);
        EXPECT_EQ(z, value);
    }
}

TEST_F(RS_MortonCodeTest, DecodeThreeAxesFor32Bit) {
    std::vector<uint32_t> samples = sampleCoordinates<uint32_t>();
    for (uint32_t expected_x : samples) {
        for (uint32_t expected_y : samples) {
            for (uint32_t expected_z : samples) {
                uint32_t x = 0, y = 0, z = 0;
                RS_MortonCode<uint32_t>::toCoordinate(
                    RS_MortonCode<uint32_t>::toMortonCode(expected_x, expected_y, expected_z), x, y, z
                );
                EXPECT_EQ(x, expected_x);
                EXPECT_EQ(y, expected_y);
                EXPECT_EQ(z, expected_z);
            }
        }
    }
}

TEST_F(RS_MortonCodeTest, DecodeThreeAxesFor64Bit) {
    std::vector<uint64_t> samples = sampleCoordinates<uint64_t>();
    for (uint64_t expected_x : samples) {
        for (uint64_t expected_y : samples) {
            for (uint64_t expected_z : samples) {
                uint64_t x = 0, y = 0, z = 0;
                RS_MortonCode<uint64_t>::toCoordinate(
                    RS_MortonCode<uint64_t>::toMortonCode(expected_x, expected_y, expected_z), x, y, z
                );
                EXPECT_EQ(x, expected_x);
                EXPECT_EQ(y, expected_y);
                EXPECT_EQ(z, expected_z);
            }
        }
    }
}

/*****************************************************************
 * Test cases for coordinates the code cannot hold
 *
 * Test cases:
 * 1. A coordinate one above the largest saturates at the largest.
 * 2. A far larger coordinate saturates at the same place.
 * 3. Only the axis that overflows saturates.
 *****************************************************************/

TEST_F(RS_MortonCodeTest, SaturateAboveCoordinateMax) {
    constexpr uint32_t coordinate_max = RS_MortonCode<uint32_t>::CoordinateMax;
    uint32_t saturated_code = RS_MortonCode<uint32_t>::toMortonCode(coordinate_max, 0, 0);

    EXPECT_EQ(RS_MortonCode<uint32_t>::toMortonCode(coordinate_max + 1, 0, 0), saturated_code);
    EXPECT_EQ(RS_MortonCode<uint32_t>::toMortonCode(0xFFFFFFFFu, 0, 0), saturated_code);
    EXPECT_EQ(
        RS_MortonCode<uint32_t>::toMortonCode(0, coordinate_max + 1, 0),
        RS_MortonCode<uint32_t>::toMortonCode(0, coordinate_max, 0)
    );
    EXPECT_EQ(
        RS_MortonCode<uint32_t>::toMortonCode(0, 0, coordinate_max + 1),
        RS_MortonCode<uint32_t>::toMortonCode(0, 0, coordinate_max)
    );
}

TEST_F(RS_MortonCodeTest, SaturateLeavesOtherAxesAlone) {
    constexpr uint32_t coordinate_max = RS_MortonCode<uint32_t>::CoordinateMax;
    uint32_t x = 0, y = 0, z = 0;
    RS_MortonCode<uint32_t>::toCoordinate(
        RS_MortonCode<uint32_t>::toMortonCode(coordinate_max + 5, 17, 300), x, y, z
    );
    EXPECT_EQ(x, coordinate_max);
    EXPECT_EQ(y, 17u);
    EXPECT_EQ(z, 300u);
}

TEST_F(RS_MortonCodeTest, SaturateAboveCoordinateMaxFor64Bit) {
    constexpr uint64_t coordinate_max = RS_MortonCode<uint64_t>::CoordinateMax;
    EXPECT_EQ(
        RS_MortonCode<uint64_t>::toMortonCode(coordinate_max + 1, 0, 0),
        RS_MortonCode<uint64_t>::toMortonCode(coordinate_max, 0, 0)
    );
    EXPECT_EQ(
        RS_MortonCode<uint64_t>::toMortonCode(0, 0, 0xFFFFFFFFFFFFFFFFull),
        RS_MortonCode<uint64_t>::toMortonCode(0, 0, coordinate_max)
    );
}

/*****************************************************************
 * Test cases for the same conversion driven from a kernel
 *
 * Test cases:
 * 1. Round trip on the device, against the host reference.
 * 2. Round trip on the host provider, against the host reference.
 *****************************************************************/

TEST_F(RS_MortonCodeTest, RoundTripOnDevice) {
    std::vector<uint32_t> samples = sampleCoordinates<uint32_t>();
    std::vector<uint32_t> coordinates;
    for (uint32_t x : samples) {
        for (uint32_t y : samples) {
            coordinates.push_back(x);
            coordinates.push_back(y);
            coordinates.push_back(samples[(x + y) % samples.size()]);
        }
    }

    std::vector<uint32_t> codes;
    std::vector<uint32_t> decoded_coordinates;
    runRoundTripOnDevice<uint32_t>(provider, coordinates, codes, decoded_coordinates);

    for (RSize i = 0; i < codes.size(); i++) {
        EXPECT_EQ(
            codes[i],
            referenceMortonCode<uint32_t>(
                coordinates[i * 3], coordinates[i * 3 + 1], coordinates[i * 3 + 2]
            )
        );
        EXPECT_EQ(decoded_coordinates[i * 3], coordinates[i * 3]);
        EXPECT_EQ(decoded_coordinates[i * 3 + 1], coordinates[i * 3 + 1]);
        EXPECT_EQ(decoded_coordinates[i * 3 + 2], coordinates[i * 3 + 2]);
    }
}

TEST_F(RS_MortonCodeTest, RoundTripOnHost) {
    std::vector<uint64_t> samples = sampleCoordinates<uint64_t>();
    std::vector<uint64_t> coordinates;
    for (uint64_t x : samples) {
        for (uint64_t y : samples) {
            coordinates.push_back(x);
            coordinates.push_back(y);
            coordinates.push_back(samples[(x + y) % samples.size()]);
        }
    }

    std::vector<uint64_t> codes;
    std::vector<uint64_t> decoded_coordinates;
    runRoundTripOnDevice<uint64_t>(host_provider, coordinates, codes, decoded_coordinates);

    for (RSize i = 0; i < codes.size(); i++) {
        EXPECT_EQ(
            codes[i],
            referenceMortonCode<uint64_t>(
                coordinates[i * 3], coordinates[i * 3 + 1], coordinates[i * 3 + 2]
            )
        );
        EXPECT_EQ(decoded_coordinates[i * 3], coordinates[i * 3]);
        EXPECT_EQ(decoded_coordinates[i * 3 + 1], coordinates[i * 3 + 1]);
        EXPECT_EQ(decoded_coordinates[i * 3 + 2], coordinates[i * 3 + 2]);
    }
}
