#include <gtest/gtest.h>

#include <map>
#include <set>
#include <string>
#include <vector>

#include "conf/rs_test_conf.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_spatial_bin.h"
#include "rise_origin/device/rs_compute.h"

using namespace rise;
using namespace rise::device;

/*****************************************************************
 * Test Summary:
 *
 * RS_SpatialBin3D groups objects by the spatial code the caller
 * writes for each of them, and records how the occupied bins
 * neighbour each other. It is built by sorting rather than hashing,
 * so nothing atomic is involved and the exact bin count is known
 * before any payload is sized.
 *
 * - Direction Offsets
 *   spatialBinDirectionToOffset() for all 26 directions, checked
 *   against the name each direction carries
 *
 * - Code Conversion
 *   RS_SpatialBinMortonCodeConverter round trips a signed grid
 *   coordinate, and reports which coordinates the code can hold
 *
 * - Build
 *   resize(), the caller writing codes in place, then rebuild(),
 *   giving sorted bins, contiguous object runs, and bin_offsets
 *
 * - Lookup
 *   findBin() and getBin() from the host and from a kernel
 *
 * - Neighbourhood
 *   Every slot of every bin, under FACE, FULL and NONE, including
 *   bins whose neighbours are absent or out of the code's reach
 *
 * - Boundary Conditions
 *   Empty build, one object, one bin holding many objects, and a
 *   rebuild that shrinks the structure or empties it
 *****************************************************************/

/**
 * @brief Looks each query code up, and reads the bin's positive x neighbour.
 *
 * Walks the neighbourhood the way a consumer does, through getBin(), so an
 * absent neighbour and a bad index both come back as a null pointer.
 */
class RSCF_QuerySpatialBin : public RS_ComputeFunction<RSCF_QuerySpatialBin> {
   public:
    using RS_ComputeFunction<RSCF_QuerySpatialBin>::index1D;

    template <typename SpatialBin>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        SpatialBin bin,
        RS_Array<RIndex, 1> query_codes,
        RS_Array<RIndex, 1> found_bin_indices,
        RS_Array<RIndex, 1> found_neighbor_codes
    ) {
        RSize i = index1D();
        RIndex bin_index = bin.findBin(query_codes[i]);
        found_bin_indices[i] = bin_index;
        found_neighbor_codes[i] = RS_NULL_INDEX;

        auto* found = bin.getBin(bin_index);
        if (found == nullptr) return;
        auto* neighbor = bin.getBin(found->neighbor_bins[RSE_BIN_X_POS]);
        if (neighbor == nullptr) return;
        found_neighbor_codes[i] = neighbor->spatial_code;
    }
};

class RS_SpatialBinTest : public ::testing::Test {
   protected:
    using TestConverter = RS_SpatialBinMortonCodeConverter<RIndex>;
    using TestBinFace = RS_SpatialBin3D<
        RIndex,
        RS_SpatialBinObjectIdentifier,
        TestConverter,
        RSE_SPATIAL_BIN_NEIGHBORHOOD_FACE>;
    using TestBinFull = RS_SpatialBin3D<
        RIndex,
        RS_SpatialBinObjectIdentifier,
        TestConverter,
        RSE_SPATIAL_BIN_NEIGHBORHOOD_FULL>;
    using TestBinNone = RS_SpatialBin3D<
        RIndex,
        RS_SpatialBinObjectIdentifier,
        TestConverter,
        RSE_SPATIAL_BIN_NEIGHBORHOOD_NONE>;

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
     * @brief Writes one object per coordinate into the bin, then rebuilds it.
     *
     * The in place path a real consumer takes, with a host write standing in for
     * the domain kernel that would otherwise fill the two public arrays.
     */
    template <typename SpatialBin>
    static void buildFromCoordinates(SpatialBin& bin, const std::vector<RCoord3>& coordinates) {
        RSize object_num = coordinates.size();
        std::vector<RIndex> codes(object_num);
        std::vector<RS_SpatialBinObjectIdentifier> identifiers(object_num);
        for (RSize i = 0; i < object_num; i++) {
            codes[i] = TestConverter::toSpatialCode(coordinates[i]);
            identifiers[i].object_index = i;
        }

        bin.resize(object_num, false);
        if (object_num > 0) {
            bin.spatial_codes.writeSlice(codes.data(), object_num, 0);
            bin.object_identifiers.writeSlice(identifiers.data(), object_num, 0);
        }
        bin.rebuild();
    }

    /// @brief Reads every bin record back to the host.
    template <typename SpatialBin>
    static std::vector<typename SpatialBin::BinType> readBins(SpatialBin& bin) {
        std::vector<typename SpatialBin::BinType> bins(bin.getBinNum());
        if (!bins.empty()) bin.bins.readSlice(bins.data(), bins.size(), 0);
        return bins;
    }

    /// @brief Reads the sorted codes, identifiers and bin offsets back to the host.
    template <typename SpatialBin>
    static void readObjects(
        SpatialBin& bin,
        RSize object_num,
        std::vector<RIndex>& codes_out,
        std::vector<RS_SpatialBinObjectIdentifier>& identifiers_out,
        std::vector<RIndex>& bin_offsets_out
    ) {
        codes_out.resize(object_num);
        identifiers_out.resize(object_num);
        bin_offsets_out.resize(object_num);
        if (object_num == 0) return;
        bin.spatial_codes.readSlice(codes_out.data(), object_num, 0);
        bin.object_identifiers.readSlice(identifiers_out.data(), object_num, 0);
        bin.bin_offsets.readSlice(bin_offsets_out.data(), object_num, 0);
    }

    /**
     * @brief The bin holding a coordinate, searched over records already read back.
     *
     * findBin() reads the records where they live, so a bin built on the device
     * can only answer it from a kernel. This walks the host copy instead, and
     * being a plain sweep it also stands as a reference for the binary search.
     */
    template <typename BinType>
    static RIndex findBinOnHost(const std::vector<BinType>& bins, const RCoord3& coord) {
        RIndex spatial_code = TestConverter::toSpatialCode(coord);
        for (RIndex i = 0; i < RIndex(bins.size()); i++) {
            if (bins[i].spatial_code == spatial_code) return i;
        }
        return RS_NULL_INDEX;
    }

    /**
     * @brief The step one axis takes, read out of the direction's own name.
     *
     * Nothing here shares code with the packed literals under test.
     */
    static RSignedIndex getAxisStepFromName(const std::string& name, char axis) {
        if (name.find(std::string("_") + axis + "_POS") != std::string::npos) return 1;
        if (name.find(std::string("_") + axis + "_NEG") != std::string::npos) return -1;
        return 0;
    }

    /**
     * @brief Checks every slot of every bin against the occupied coordinate set.
     *
     * A slot holds a bin index when that surrounding coordinate is occupied and
     * the code can still reach it, and RS_NULL_INDEX otherwise.
     */
    template <typename SpatialBin>
    static void expectNeighborsResolved(const std::vector<typename SpatialBin::BinType>& bins) {
        std::map<RIndex, RIndex> bin_index_by_code;
        for (RIndex i = 0; i < RIndex(bins.size()); i++) {
            bin_index_by_code[bins[i].spatial_code] = i;
        }

        for (RIndex i = 0; i < RIndex(bins.size()); i++) {
            RCoord3 coord = TestConverter::toCoordinate(bins[i].spatial_code);
            for (RIndex direction = 0; direction < SpatialBin::NeighborNum; direction++) {
                RCoord3 neighbor =
                    coord + spatialBinDirectionToOffset(RSE_SpatialBinDirection(direction));
                RIndex expected_bin_index = RS_NULL_INDEX;
                if (TestConverter::isCoordinateInBound(neighbor)) {
                    auto found = bin_index_by_code.find(TestConverter::toSpatialCode(neighbor));
                    if (found != bin_index_by_code.end()) expected_bin_index = found->second;
                }
                EXPECT_EQ(bins[i].neighbor_bins[direction], expected_bin_index)
                    << "bin " << i << " slot "
                    << enumToString(RSE_SpatialBinDirection(direction));
            }
        }
    }

    /// @brief A solid block of coordinates, one object each, with a corner at origin.
    static std::vector<RCoord3> makeCoordinateBlock(RSignedIndex side_num) {
        std::vector<RCoord3> coordinates;
        for (RSignedIndex x = 0; x < side_num; x++) {
            for (RSignedIndex y = 0; y < side_num; y++) {
                for (RSignedIndex z = 0; z < side_num; z++) {
                    coordinates.emplace_back(x, y, z);
                }
            }
        }
        return coordinates;
    }
};

/*****************************************************************
 * Test cases for the direction offsets
 *
 * The three packed literals are written by hand, and a wrong field
 * silently resolves a wrong neighbour, so these check them from
 * every angle the enum allows.
 *
 * Test cases:
 * 1. Every offset matches the name its direction carries.
 * 2. The six faces are in the voxel link order.
 * 3. Faces move on one axis, edges on two, corners on three.
 * 4. No two directions share an offset, and none stands still.
 *****************************************************************/

TEST_F(RS_SpatialBinTest, DirectionOffsetsMatchTheirNames) {
    for (RIndex direction = 0; direction < RSE_SPATIAL_BIN_NEIGHBORHOOD_FULL; direction++) {
        RSE_SpatialBinDirection typed_direction = RSE_SpatialBinDirection(direction);
        std::string name = enumToString(typed_direction);
        RCoord3 offset = spatialBinDirectionToOffset(typed_direction);

        EXPECT_EQ(offset.x, getAxisStepFromName(name, 'X')) << name;
        EXPECT_EQ(offset.y, getAxisStepFromName(name, 'Y')) << name;
        EXPECT_EQ(offset.z, getAxisStepFromName(name, 'Z')) << name;
    }
}

TEST_F(RS_SpatialBinTest, FaceDirectionsFollowTheLinkOrder) {
    EXPECT_EQ(spatialBinDirectionToOffset(RSE_BIN_X_POS), RCoord3(1, 0, 0));
    EXPECT_EQ(spatialBinDirectionToOffset(RSE_BIN_X_NEG), RCoord3(-1, 0, 0));
    EXPECT_EQ(spatialBinDirectionToOffset(RSE_BIN_Y_POS), RCoord3(0, 1, 0));
    EXPECT_EQ(spatialBinDirectionToOffset(RSE_BIN_Y_NEG), RCoord3(0, -1, 0));
    EXPECT_EQ(spatialBinDirectionToOffset(RSE_BIN_Z_POS), RCoord3(0, 0, 1));
    EXPECT_EQ(spatialBinDirectionToOffset(RSE_BIN_Z_NEG), RCoord3(0, 0, -1));
}

TEST_F(RS_SpatialBinTest, DirectionGroupsMoveOnTheRightAxisCount) {
    for (RIndex direction = 0; direction < RSE_SPATIAL_BIN_NEIGHBORHOOD_FULL; direction++) {
        RCoord3 offset = spatialBinDirectionToOffset(RSE_SpatialBinDirection(direction));
        RIndex moved_axis_num = RIndex(offset.x != 0) + RIndex(offset.y != 0) + RIndex(offset.z != 0);

        if (direction < RSE_SPATIAL_BIN_NEIGHBORHOOD_FACE) {
            EXPECT_EQ(moved_axis_num, 1u) << "face " << direction;
        } else if (direction < RSE_SPATIAL_BIN_NEIGHBORHOOD_FACE_EDGE) {
            EXPECT_EQ(moved_axis_num, 2u) << "edge " << direction;
        } else {
            EXPECT_EQ(moved_axis_num, 3u) << "corner " << direction;
        }
    }
}

TEST_F(RS_SpatialBinTest, DirectionOffsetsAreDistinctAndNonZero) {
    std::set<std::tuple<RSignedIndex, RSignedIndex, RSignedIndex>> seen_offsets;
    for (RIndex direction = 0; direction < RSE_SPATIAL_BIN_NEIGHBORHOOD_FULL; direction++) {
        RCoord3 offset = spatialBinDirectionToOffset(RSE_SpatialBinDirection(direction));
        EXPECT_GE(offset.x, -1);
        EXPECT_LE(offset.x, 1);
        EXPECT_GE(offset.y, -1);
        EXPECT_LE(offset.y, 1);
        EXPECT_GE(offset.z, -1);
        EXPECT_LE(offset.z, 1);
        EXPECT_NE(offset, RCoord3(0, 0, 0)) << "direction " << direction << " stands still";
        seen_offsets.emplace(offset.x, offset.y, offset.z);
    }
    EXPECT_EQ(seen_offsets.size(), RSE_SPATIAL_BIN_NEIGHBORHOOD_FULL);
}

/*****************************************************************
 * Test cases for the spatial code converter
 *
 * Test cases:
 * 1. Signed coordinates round trip through a code.
 * 2. Distinct coordinates take distinct codes.
 * 3. isCoordinateInBound() at each end of the reachable range.
 * 4. A coordinate out of bound converts to InvalidCode.
 * 5. No coordinate in bound ever produces InvalidCode.
 *****************************************************************/

TEST_F(RS_SpatialBinTest, CodeRoundTripsSignedCoordinates) {
    const std::vector<RSignedIndex> samples{
        -TestConverter::CoordinateBias, -257, -256, -1, 0, 1, 255, TestConverter::CoordinateBias - 1};

    for (RSignedIndex x : samples) {
        for (RSignedIndex y : samples) {
            for (RSignedIndex z : samples) {
                RCoord3 coord(x, y, z);
                EXPECT_EQ(TestConverter::toCoordinate(TestConverter::toSpatialCode(coord)), coord);
            }
        }
    }
}

TEST_F(RS_SpatialBinTest, DistinctCoordinatesTakeDistinctCodes) {
    std::set<RIndex> seen_codes;
    for (RSignedIndex x = -4; x <= 4; x++) {
        for (RSignedIndex y = -4; y <= 4; y++) {
            for (RSignedIndex z = -4; z <= 4; z++) {
                seen_codes.insert(TestConverter::toSpatialCode(RCoord3(x, y, z)));
            }
        }
    }
    EXPECT_EQ(seen_codes.size(), 9u * 9u * 9u);
}

TEST_F(RS_SpatialBinTest, CoordinateBoundEnds) {
    constexpr RSignedIndex bias = TestConverter::CoordinateBias;

    EXPECT_TRUE(TestConverter::isCoordinateInBound(RCoord3(0, 0, 0)));
    EXPECT_TRUE(TestConverter::isCoordinateInBound(RCoord3(-bias, -bias, -bias)));
    EXPECT_TRUE(TestConverter::isCoordinateInBound(RCoord3(bias - 1, bias - 1, bias - 1)));

    EXPECT_FALSE(TestConverter::isCoordinateInBound(RCoord3(bias, 0, 0)));
    EXPECT_FALSE(TestConverter::isCoordinateInBound(RCoord3(0, bias, 0)));
    EXPECT_FALSE(TestConverter::isCoordinateInBound(RCoord3(0, 0, bias)));
    EXPECT_FALSE(TestConverter::isCoordinateInBound(RCoord3(-bias - 1, 0, 0)));
    EXPECT_FALSE(TestConverter::isCoordinateInBound(RCoord3(0, -bias - 1, 0)));
    EXPECT_FALSE(TestConverter::isCoordinateInBound(RCoord3(0, 0, -bias - 1)));
}

TEST_F(RS_SpatialBinTest, CoordinateOutOfBoundTakesInvalidCode) {
    constexpr RSignedIndex bias = TestConverter::CoordinateBias;

    EXPECT_EQ(TestConverter::toSpatialCode(RCoord3(bias, 0, 0)), TestConverter::InvalidCode);
    EXPECT_EQ(TestConverter::toSpatialCode(RCoord3(0, bias, 0)), TestConverter::InvalidCode);
    EXPECT_EQ(TestConverter::toSpatialCode(RCoord3(0, 0, bias)), TestConverter::InvalidCode);
    EXPECT_EQ(TestConverter::toSpatialCode(RCoord3(-bias - 1, 0, 0)), TestConverter::InvalidCode);
    EXPECT_EQ(TestConverter::toSpatialCode(RCoord3(0, -bias - 1, 0)), TestConverter::InvalidCode);
    EXPECT_EQ(TestConverter::toSpatialCode(RCoord3(0, 0, -bias - 1)), TestConverter::InvalidCode);
    EXPECT_EQ(
        TestConverter::toSpatialCode(RCoord3(1 << 20, -(1 << 20), 1 << 20)),
        TestConverter::InvalidCode
    );
}

TEST_F(RS_SpatialBinTest, CoordinateInBoundNeverTakesInvalidCode) {
    constexpr RSignedIndex bias = TestConverter::CoordinateBias;
    const std::vector<RSignedIndex> samples{-bias, -bias + 1, -1, 0, 1, bias - 2, bias - 1};

    for (RSignedIndex x : samples) {
        for (RSignedIndex y : samples) {
            for (RSignedIndex z : samples) {
                EXPECT_NE(
                    TestConverter::toSpatialCode(RCoord3(x, y, z)), TestConverter::InvalidCode
                ) << x << ", " << y << ", " << z;
            }
        }
    }
}

/*****************************************************************
 * Test cases for the bin record layout
 *
 * Test cases:
 * 1. A neighbourhood names as many slots as the record holds.
 * 2. The larger neighbourhoods take a sector alignment, the
 *    smaller ones the natural alignment of their fields.
 *****************************************************************/

TEST_F(RS_SpatialBinTest, NeighborSlotNum) {
    EXPECT_EQ(TestBinNone::NeighborNum, 0u);
    EXPECT_EQ(TestBinFace::NeighborNum, 6u);
    EXPECT_EQ(TestBinFull::NeighborNum, 26u);
}

TEST_F(RS_SpatialBinTest, BinAlignment) {
    EXPECT_EQ(alignof(TestBinNone::BinType), alignof(RIndex));
    EXPECT_EQ(alignof(TestBinFace::BinType), alignof(RIndex));
    EXPECT_EQ(alignof(TestBinFull::BinType), 32u);
    EXPECT_EQ(sizeof(TestBinFull::BinType) % 32u, 0u);
}

/*****************************************************************
 * Test cases for building the structure
 *
 * Test cases:
 * 1. No objects at all.
 * 2. One object, so one bin holding it.
 * 3. Many objects sharing one coordinate.
 * 4. A block of distinct coordinates, checking the bin order, the
 *    object runs, and bin_offsets for every object.
 * 5. Every object identifier survives the sort, in its own bin.
 *****************************************************************/

TEST_F(RS_SpatialBinTest, BuildEmpty) {
    TestBinFace bin(&provider);
    buildFromCoordinates(bin, {});

    EXPECT_EQ(bin.getBinNum(), 0u);
    EXPECT_EQ(bin.bins.size(), 0u);

    bin.free();
}

TEST_F(RS_SpatialBinTest, BuildOneObject) {
    TestBinFace bin(&provider);
    buildFromCoordinates(bin, {RCoord3(3, -4, 5)});

    ASSERT_EQ(bin.getBinNum(), 1u);
    std::vector<TestBinFace::BinType> bins = readBins(bin);
    EXPECT_EQ(bins[0].spatial_code, TestConverter::toSpatialCode(RCoord3(3, -4, 5)));
    EXPECT_EQ(bins[0].object_offset, 0u);
    EXPECT_EQ(bins[0].object_num, 1u);
    for (RIndex direction = 0; direction < TestBinFace::NeighborNum; direction++) {
        EXPECT_EQ(bins[0].neighbor_bins[direction], RS_NULL_INDEX)
            << "an isolated bin has no neighbours";
    }

    bin.free();
}

TEST_F(RS_SpatialBinTest, BuildManyObjectsInOneBin) {
    constexpr RSize object_num = 64;
    std::vector<RCoord3> coordinates(object_num, RCoord3(-7, 8, -9));

    TestBinFace bin(&provider);
    buildFromCoordinates(bin, coordinates);

    ASSERT_EQ(bin.getBinNum(), 1u);
    std::vector<TestBinFace::BinType> bins = readBins(bin);
    EXPECT_EQ(bins[0].object_offset, 0u);
    EXPECT_EQ(bins[0].object_num, object_num);

    std::vector<RIndex> codes;
    std::vector<RS_SpatialBinObjectIdentifier> identifiers;
    std::vector<RIndex> bin_offsets;
    readObjects(bin, object_num, codes, identifiers, bin_offsets);
    for (RSize i = 0; i < object_num; i++) {
        EXPECT_EQ(bin_offsets[i], 0u);
    }

    bin.free();
}

TEST_F(RS_SpatialBinTest, BuildBlockOfCoordinates) {
    constexpr RSignedIndex side_num = 4;
    std::vector<RCoord3> coordinates = makeCoordinateBlock(side_num);
    RSize object_num = coordinates.size();

    TestBinFace bin(&provider);
    buildFromCoordinates(bin, coordinates);

    ASSERT_EQ(bin.getBinNum(), object_num);
    std::vector<TestBinFace::BinType> bins = readBins(bin);

    // Bins come out sorted by code, with the object runs back to back.
    RIndex total_object_num = 0;
    for (RIndex i = 0; i < RIndex(bins.size()); i++) {
        if (i > 0) EXPECT_LT(bins[i - 1].spatial_code, bins[i].spatial_code);
        EXPECT_EQ(bins[i].object_offset, total_object_num);
        EXPECT_EQ(bins[i].object_num, 1u);
        total_object_num += bins[i].object_num;
    }
    EXPECT_EQ(total_object_num, object_num);

    bin.free();
}

TEST_F(RS_SpatialBinTest, BuildKeepsEveryObjectInItsOwnBin) {
    // Coordinates repeat, so bins hold uneven numbers of objects.
    std::vector<RCoord3> coordinates;
    for (RSignedIndex x = 0; x < 5; x++) {
        for (RSignedIndex count = 0; count <= x; count++) {
            coordinates.emplace_back(x, -x, x * 2);
        }
    }
    RSize object_num = coordinates.size();

    TestBinFace bin(&provider);
    buildFromCoordinates(bin, coordinates);

    ASSERT_EQ(bin.getBinNum(), 5u);
    std::vector<TestBinFace::BinType> bins = readBins(bin);

    std::vector<RIndex> codes;
    std::vector<RS_SpatialBinObjectIdentifier> identifiers;
    std::vector<RIndex> bin_offsets;
    readObjects(bin, object_num, codes, identifiers, bin_offsets);

    // Every object still carries the code of the coordinate it was written with,
    // and sits inside the run of the bin bin_offsets points at.
    std::set<RIndex> seen_object_indices;
    for (RSize i = 0; i < object_num; i++) {
        ASSERT_LT(identifiers[i].object_index, object_num);
        RCoord3 coord = coordinates[identifiers[i].object_index];
        EXPECT_EQ(codes[i], TestConverter::toSpatialCode(coord));

        RIndex bin_index = bin_offsets[i];
        ASSERT_LT(bin_index, bins.size());
        EXPECT_EQ(bins[bin_index].spatial_code, codes[i]);
        EXPECT_GE(i, bins[bin_index].object_offset);
        EXPECT_LT(i, bins[bin_index].object_offset + bins[bin_index].object_num);

        seen_object_indices.insert(identifiers[i].object_index);
    }
    EXPECT_EQ(seen_object_indices.size(), object_num) << "the sort dropped or copied an object";

    // A bin holds as many objects as the coordinate was written times.
    for (RIndex i = 0; i < RIndex(bins.size()); i++) {
        RCoord3 coord = TestConverter::toCoordinate(bins[i].spatial_code);
        EXPECT_EQ(bins[i].object_num, RIndex(coord.x) + 1u);
    }

    bin.free();
}

/*****************************************************************
 * Test cases for looking a bin up
 *
 * Test cases:
 * 1. findBin() on the host over every occupied and absent code, and over a
 *    code no coordinate in bound produces.
 * 2. findBin() and getBin() driven from a kernel on the device.
 * 3. findBin() and getBin() driven from a kernel on the host.
 *****************************************************************/

TEST_F(RS_SpatialBinTest, FindBinOnHost) {
    std::vector<RCoord3> coordinates = makeCoordinateBlock(3);

    TestBinFace bin(&host_provider);
    buildFromCoordinates(bin, coordinates);
    ASSERT_EQ(bin.getBinNum(), coordinates.size());

    for (const RCoord3& coord : coordinates) {
        RIndex bin_index = bin.findBin(TestConverter::toSpatialCode(coord));
        ASSERT_NE(bin_index, RS_NULL_INDEX);
        EXPECT_EQ(bin.bins[bin_index].spatial_code, TestConverter::toSpatialCode(coord));
    }

    // Codes below, above and inside the occupied range, none of them written.
    EXPECT_EQ(bin.findBin(TestConverter::toSpatialCode(RCoord3(-1, -1, -1))), RS_NULL_INDEX);
    EXPECT_EQ(bin.findBin(TestConverter::toSpatialCode(RCoord3(9, 9, 9))), RS_NULL_INDEX);
    EXPECT_EQ(bin.findBin(TestConverter::toSpatialCode(RCoord3(0, 0, 3))), RS_NULL_INDEX);

    // A coordinate the code cannot hold arrives as InvalidCode and finds nothing.
    EXPECT_EQ(bin.findBin(TestConverter::InvalidCode), RS_NULL_INDEX);
    EXPECT_EQ(
        bin.findBin(TestConverter::toSpatialCode(RCoord3(TestConverter::CoordinateBias, 0, 0))),
        RS_NULL_INDEX
    );

    EXPECT_EQ(bin.getBin(RS_NULL_INDEX), nullptr);
    EXPECT_EQ(bin.getBin(RIndex(bin.getBinNum())), nullptr);
    EXPECT_NE(bin.getBin(0), nullptr);

    bin.free();
}

TEST_F(RS_SpatialBinTest, QueryOnDevice) {
    // A row along x, so the positive x neighbour is present for all but the last.
    std::vector<RCoord3> coordinates;
    for (RSignedIndex x = 0; x < 6; x++) coordinates.emplace_back(x, 0, 0);

    TestBinFace bin(&provider);
    buildFromCoordinates(bin, coordinates);
    ASSERT_EQ(bin.getBinNum(), coordinates.size());

    std::vector<RIndex> query_codes;
    for (const RCoord3& coord : coordinates) {
        query_codes.push_back(TestConverter::toSpatialCode(coord));
    }
    // A coordinate nothing was written at.
    query_codes.push_back(TestConverter::toSpatialCode(RCoord3(0, 1, 0)));

    RSize query_num = RSize(query_codes.size());
    RS_Array<RIndex, 1> d_query_codes({query_num}, &provider);
    RS_Array<RIndex, 1> d_found_bin_indices({query_num}, &provider);
    RS_Array<RIndex, 1> d_found_neighbor_codes({query_num}, &provider);
    d_query_codes.write(query_codes.data());
    d_found_bin_indices.fill(0);
    d_found_neighbor_codes.fill(0);

    provider.run<RSCF_QuerySpatialBin>(
        {query_num}, bin, d_query_codes, d_found_bin_indices, d_found_neighbor_codes
    );

    std::vector<RIndex> found_bin_indices(query_codes.size());
    std::vector<RIndex> found_neighbor_codes(query_codes.size());
    d_found_bin_indices.read(found_bin_indices.data());
    d_found_neighbor_codes.read(found_neighbor_codes.data());

    for (RSize i = 0; i < coordinates.size(); i++) {
        ASSERT_NE(found_bin_indices[i], RS_NULL_INDEX) << "coordinate " << i << " should be found";
        bool is_last = (i + 1 == coordinates.size());
        RIndex expected_neighbor_code =
            is_last ? RS_NULL_INDEX
                    : TestConverter::toSpatialCode(RCoord3(RSignedIndex(i) + 1, 0, 0));
        EXPECT_EQ(found_neighbor_codes[i], expected_neighbor_code) << "coordinate " << i;
    }
    EXPECT_EQ(found_bin_indices.back(), RS_NULL_INDEX) << "an absent code finds no bin";
    EXPECT_EQ(found_neighbor_codes.back(), RS_NULL_INDEX);

    d_query_codes.free();
    d_found_bin_indices.free();
    d_found_neighbor_codes.free();
    bin.free();
}

TEST_F(RS_SpatialBinTest, QueryOnHost) {
    std::vector<RCoord3> coordinates;
    for (RSignedIndex x = 0; x < 6; x++) coordinates.emplace_back(x, 0, 0);

    TestBinFace bin(&host_provider);
    buildFromCoordinates(bin, coordinates);
    ASSERT_EQ(bin.getBinNum(), coordinates.size());

    std::vector<RIndex> query_codes;
    for (const RCoord3& coord : coordinates) {
        query_codes.push_back(TestConverter::toSpatialCode(coord));
    }
    query_codes.push_back(TestConverter::toSpatialCode(RCoord3(0, 1, 0)));

    RSize query_num = RSize(query_codes.size());
    RS_Array<RIndex, 1> d_query_codes({query_num}, &host_provider);
    RS_Array<RIndex, 1> d_found_bin_indices({query_num}, &host_provider);
    RS_Array<RIndex, 1> d_found_neighbor_codes({query_num}, &host_provider);
    d_query_codes.write(query_codes.data());
    d_found_bin_indices.fill(0);
    d_found_neighbor_codes.fill(0);

    host_provider.run<RSCF_QuerySpatialBin>(
        {query_num}, bin, d_query_codes, d_found_bin_indices, d_found_neighbor_codes
    );

    std::vector<RIndex> found_bin_indices(query_codes.size());
    std::vector<RIndex> found_neighbor_codes(query_codes.size());
    d_found_bin_indices.read(found_bin_indices.data());
    d_found_neighbor_codes.read(found_neighbor_codes.data());

    for (RSize i = 0; i + 1 < coordinates.size(); i++) {
        EXPECT_EQ(
            found_neighbor_codes[i], TestConverter::toSpatialCode(RCoord3(RSignedIndex(i) + 1, 0, 0))
        );
    }
    EXPECT_EQ(found_bin_indices.back(), RS_NULL_INDEX);

    d_query_codes.free();
    d_found_bin_indices.free();
    d_found_neighbor_codes.free();
    bin.free();
}

/*****************************************************************
 * Test cases for resolving the neighbourhood
 *
 * Test cases:
 * 1. FACE over a block, where the middle bin has all six and the
 *    surface bins are missing the slots pointing outwards.
 * 2. FULL over a 3 by 3 by 3 block, where the middle bin has 26.
 * 3. NONE builds and bins without recording anything.
 * 4. Bins scattered so most slots stay empty.
 *****************************************************************/

TEST_F(RS_SpatialBinTest, ResolveFaceNeighbors) {
    std::vector<RCoord3> coordinates = makeCoordinateBlock(3);

    TestBinFace bin(&provider);
    buildFromCoordinates(bin, coordinates);
    ASSERT_EQ(bin.getBinNum(), coordinates.size());

    std::vector<TestBinFace::BinType> bins = readBins(bin);
    expectNeighborsResolved<TestBinFace>(bins);

    // The middle of the block is surrounded on every face.
    RIndex middle_index = findBinOnHost(bins, RCoord3(1, 1, 1));
    ASSERT_NE(middle_index, RS_NULL_INDEX);
    for (RIndex direction = 0; direction < TestBinFace::NeighborNum; direction++) {
        EXPECT_NE(bins[middle_index].neighbor_bins[direction], RS_NULL_INDEX)
            << enumToString(RSE_SpatialBinDirection(direction));
    }

    // A block corner has three faces inside the block and three outside it.
    RIndex corner_index = findBinOnHost(bins, RCoord3(0, 0, 0));
    ASSERT_NE(corner_index, RS_NULL_INDEX);
    RIndex present_num = 0;
    for (RIndex direction = 0; direction < TestBinFace::NeighborNum; direction++) {
        if (bins[corner_index].neighbor_bins[direction] != RS_NULL_INDEX) present_num++;
    }
    EXPECT_EQ(present_num, 3u);

    bin.free();
}

TEST_F(RS_SpatialBinTest, ResolveFullNeighbors) {
    std::vector<RCoord3> coordinates = makeCoordinateBlock(3);

    TestBinFull bin(&provider);
    buildFromCoordinates(bin, coordinates);
    ASSERT_EQ(bin.getBinNum(), coordinates.size());

    std::vector<TestBinFull::BinType> bins = readBins(bin);
    expectNeighborsResolved<TestBinFull>(bins);

    RIndex middle_index = findBinOnHost(bins, RCoord3(1, 1, 1));
    ASSERT_NE(middle_index, RS_NULL_INDEX);
    std::set<RIndex> neighbor_indices;
    for (RIndex direction = 0; direction < TestBinFull::NeighborNum; direction++) {
        RIndex neighbor_index = bins[middle_index].neighbor_bins[direction];
        EXPECT_NE(neighbor_index, RS_NULL_INDEX)
            << enumToString(RSE_SpatialBinDirection(direction));
        neighbor_indices.insert(neighbor_index);
    }
    EXPECT_EQ(neighbor_indices.size(), 26u) << "two slots resolved to the same bin";
    EXPECT_EQ(neighbor_indices.count(middle_index), 0u) << "a bin is its own neighbour";

    bin.free();
}

TEST_F(RS_SpatialBinTest, ResolveNoNeighbors) {
    std::vector<RCoord3> coordinates = makeCoordinateBlock(3);

    TestBinNone bin(&provider);
    buildFromCoordinates(bin, coordinates);

    ASSERT_EQ(bin.getBinNum(), coordinates.size());
    std::vector<TestBinNone::BinType> bins = readBins(bin);
    for (RIndex i = 0; i < RIndex(bins.size()); i++) {
        EXPECT_EQ(bins[i].object_num, 1u);
    }
    EXPECT_NE(findBinOnHost(bins, RCoord3(1, 1, 1)), RS_NULL_INDEX);

    bin.free();
}

TEST_F(RS_SpatialBinTest, ResolveScatteredNeighbors) {
    // Far enough apart that only the pair at x 10 and 11 touch.
    std::vector<RCoord3> coordinates{
        RCoord3(-100, -100, -100), RCoord3(0, 0, 0), RCoord3(10, 0, 0), RCoord3(11, 0, 0),
        RCoord3(300, -300, 300)};

    TestBinFace bin(&provider);
    buildFromCoordinates(bin, coordinates);
    ASSERT_EQ(bin.getBinNum(), coordinates.size());

    std::vector<TestBinFace::BinType> bins = readBins(bin);
    expectNeighborsResolved<TestBinFace>(bins);

    RIndex left_index = findBinOnHost(bins, RCoord3(10, 0, 0));
    RIndex right_index = findBinOnHost(bins, RCoord3(11, 0, 0));
    ASSERT_NE(left_index, RS_NULL_INDEX);
    ASSERT_NE(right_index, RS_NULL_INDEX);
    EXPECT_EQ(bins[left_index].neighbor_bins[RSE_BIN_X_POS], right_index);
    EXPECT_EQ(bins[right_index].neighbor_bins[RSE_BIN_X_NEG], left_index);
    EXPECT_EQ(bins[left_index].neighbor_bins[RSE_BIN_X_NEG], RS_NULL_INDEX);
    EXPECT_EQ(bins[right_index].neighbor_bins[RSE_BIN_X_POS], RS_NULL_INDEX);

    bin.free();
}

/*****************************************************************
 * Test cases for coordinates at the edge of what a code reaches
 *
 * A step past the edge must leave the slot empty rather than
 * saturate onto a coordinate that is already occupied.
 *
 * Test cases:
 * 1. A bin at the upper end, where the positive steps run out.
 * 2. A bin at the lower end, where the negative steps run out.
 *****************************************************************/

TEST_F(RS_SpatialBinTest, ResolveNeighborsAtUpperCoordinateBound) {
    constexpr RSignedIndex bias = TestConverter::CoordinateBias;
    std::vector<RCoord3> coordinates{
        RCoord3(bias - 1, bias - 1, bias - 1), RCoord3(bias - 2, bias - 1, bias - 1)};

    TestBinFace bin(&provider);
    buildFromCoordinates(bin, coordinates);
    ASSERT_EQ(bin.getBinNum(), 2u);

    std::vector<TestBinFace::BinType> bins = readBins(bin);
    RIndex edge_index = findBinOnHost(bins, coordinates[0]);
    RIndex inner_index = findBinOnHost(bins, coordinates[1]);
    ASSERT_NE(edge_index, RS_NULL_INDEX);
    ASSERT_NE(inner_index, RS_NULL_INDEX);

    EXPECT_EQ(bins[edge_index].neighbor_bins[RSE_BIN_X_POS], RS_NULL_INDEX);
    EXPECT_EQ(bins[edge_index].neighbor_bins[RSE_BIN_Y_POS], RS_NULL_INDEX);
    EXPECT_EQ(bins[edge_index].neighbor_bins[RSE_BIN_Z_POS], RS_NULL_INDEX);
    EXPECT_EQ(bins[edge_index].neighbor_bins[RSE_BIN_X_NEG], inner_index);

    bin.free();
}

TEST_F(RS_SpatialBinTest, ResolveNeighborsAtLowerCoordinateBound) {
    constexpr RSignedIndex bias = TestConverter::CoordinateBias;
    std::vector<RCoord3> coordinates{RCoord3(-bias, -bias, -bias), RCoord3(-bias + 1, -bias, -bias)};

    TestBinFace bin(&provider);
    buildFromCoordinates(bin, coordinates);
    ASSERT_EQ(bin.getBinNum(), 2u);

    std::vector<TestBinFace::BinType> bins = readBins(bin);
    RIndex edge_index = findBinOnHost(bins, coordinates[0]);
    RIndex inner_index = findBinOnHost(bins, coordinates[1]);
    ASSERT_NE(edge_index, RS_NULL_INDEX);
    ASSERT_NE(inner_index, RS_NULL_INDEX);

    EXPECT_EQ(bins[edge_index].neighbor_bins[RSE_BIN_X_NEG], RS_NULL_INDEX);
    EXPECT_EQ(bins[edge_index].neighbor_bins[RSE_BIN_Y_NEG], RS_NULL_INDEX);
    EXPECT_EQ(bins[edge_index].neighbor_bins[RSE_BIN_Z_NEG], RS_NULL_INDEX);
    EXPECT_EQ(bins[edge_index].neighbor_bins[RSE_BIN_X_POS], inner_index);

    bin.free();
}

/*****************************************************************
 * Test cases for rebuilding the same structure again
 *
 * bins is refreshed by rebuild() rather than resized by the caller,
 * so a second build has to leave nothing of the first behind.
 *
 * Test cases:
 * 1. Rebuilding with the same objects gives the same bins.
 * 2. Rebuilding with fewer objects shrinks the bins.
 * 3. Rebuilding with no objects leaves no bin to find.
 *****************************************************************/

TEST_F(RS_SpatialBinTest, RebuildWithSameObjects) {
    std::vector<RCoord3> coordinates = makeCoordinateBlock(3);

    TestBinFace bin(&provider);
    buildFromCoordinates(bin, coordinates);
    std::vector<TestBinFace::BinType> first_bins = readBins(bin);

    buildFromCoordinates(bin, coordinates);
    std::vector<TestBinFace::BinType> second_bins = readBins(bin);

    ASSERT_EQ(second_bins.size(), first_bins.size());
    for (RIndex i = 0; i < RIndex(first_bins.size()); i++) {
        EXPECT_EQ(second_bins[i].spatial_code, first_bins[i].spatial_code);
        EXPECT_EQ(second_bins[i].object_offset, first_bins[i].object_offset);
        EXPECT_EQ(second_bins[i].object_num, first_bins[i].object_num);
    }
    expectNeighborsResolved<TestBinFace>(second_bins);

    bin.free();
}

TEST_F(RS_SpatialBinTest, RebuildWithFewerObjects) {
    TestBinFace bin(&provider);
    buildFromCoordinates(bin, makeCoordinateBlock(4));
    ASSERT_EQ(bin.getBinNum(), 64u);

    std::vector<RCoord3> fewer_coordinates{RCoord3(0, 0, 0), RCoord3(1, 0, 0)};
    buildFromCoordinates(bin, fewer_coordinates);

    ASSERT_EQ(bin.getBinNum(), 2u);
    EXPECT_EQ(bin.bins.size(), 2u) << "bins still reaches past the bins it now holds";

    std::vector<TestBinFace::BinType> bins = readBins(bin);
    expectNeighborsResolved<TestBinFace>(bins);
    // A coordinate of the first build, gone from the second.
    EXPECT_EQ(findBinOnHost(bins, RCoord3(3, 3, 3)), RS_NULL_INDEX);

    bin.free();
}

TEST_F(RS_SpatialBinTest, RebuildWithNoObjects) {
    TestBinFace bin(&provider);
    buildFromCoordinates(bin, makeCoordinateBlock(3));
    ASSERT_EQ(bin.getBinNum(), 27u);

    buildFromCoordinates(bin, {});

    EXPECT_EQ(bin.getBinNum(), 0u);
    EXPECT_EQ(bin.bins.size(), 0u) << "bins still reaches past the bins it now holds";
    EXPECT_EQ(findBinOnHost(readBins(bin), RCoord3(1, 1, 1)), RS_NULL_INDEX);

    bin.free();
}
