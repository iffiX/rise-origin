// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SPATIAL_BIN_H
#define RS_SPATIAL_BIN_H

#include "rise_origin/device/algorithm/rs_generate.h"
#include "rise_origin/device/algorithm/rs_scan.h"
#include "rise_origin/device/algorithm/rs_sort.h"
#include "rise_origin/device/data/rs_morton.h"
#include "rise_origin/device/data/rs_value.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/utils/math/rs_coord.h"
#include "rise_origin/utils/meta/rs_constexpr.h"
#include "rs_data_util.h"

namespace rise {

namespace device {

/**
 * @brief One of the 26 bins surrounding a bin.
 *
 * The value doubles as the slot in RS_SpatialBin::neighbor_bins, so a
 * caller reaches a neighbour by name. Faces come first in the voxel link order
 * RSE_X_POS to RSE_Z_NEG, so a slot index and a link direction are the same
 * number. Edges follow, then corners, which is what makes each neighbourhood a
 * prefix of this list.
 */
RS_ENUM(
    RSE_SpatialBinDirection,
    unsigned int,
    (RSE_BIN_X_POS, 0),               //!< Face
    (RSE_BIN_X_NEG, 1),               //!< Face
    (RSE_BIN_Y_POS, 2),               //!< Face
    (RSE_BIN_Y_NEG, 3),               //!< Face
    (RSE_BIN_Z_POS, 4),               //!< Face
    (RSE_BIN_Z_NEG, 5),               //!< Face
    (RSE_BIN_X_POS_Y_POS, 6),         //!< Edge
    (RSE_BIN_X_POS_Y_NEG, 7),         //!< Edge
    (RSE_BIN_X_NEG_Y_POS, 8),         //!< Edge
    (RSE_BIN_X_NEG_Y_NEG, 9),         //!< Edge
    (RSE_BIN_X_POS_Z_POS, 10),        //!< Edge
    (RSE_BIN_X_POS_Z_NEG, 11),        //!< Edge
    (RSE_BIN_X_NEG_Z_POS, 12),        //!< Edge
    (RSE_BIN_X_NEG_Z_NEG, 13),        //!< Edge
    (RSE_BIN_Y_POS_Z_POS, 14),        //!< Edge
    (RSE_BIN_Y_POS_Z_NEG, 15),        //!< Edge
    (RSE_BIN_Y_NEG_Z_POS, 16),        //!< Edge
    (RSE_BIN_Y_NEG_Z_NEG, 17),        //!< Edge
    (RSE_BIN_X_POS_Y_POS_Z_POS, 18),  //!< Corner
    (RSE_BIN_X_POS_Y_POS_Z_NEG, 19),  //!< Corner
    (RSE_BIN_X_POS_Y_NEG_Z_POS, 20),  //!< Corner
    (RSE_BIN_X_POS_Y_NEG_Z_NEG, 21),  //!< Corner
    (RSE_BIN_X_NEG_Y_POS_Z_POS, 22),  //!< Corner
    (RSE_BIN_X_NEG_Y_POS_Z_NEG, 23),  //!< Corner
    (RSE_BIN_X_NEG_Y_NEG_Z_POS, 24),  //!< Corner
    (RSE_BIN_X_NEG_Y_NEG_Z_NEG, 25)   //!< Corner
)

/**
 * @brief How many surrounding bins a bin records, excluding itself.
 *
 * Each is a prefix of RSE_SpatialBinDirection, so a slot means the same thing
 * under all of them and a stencil written against FACE still reads under FULL.
 */
RS_ENUM(
    RSE_SpatialBinNeighborhood,
    unsigned int,
    (RSE_SPATIAL_BIN_NEIGHBORHOOD_NONE, 0),        //!< No neighbours recorded
    (RSE_SPATIAL_BIN_NEIGHBORHOOD_FACE, 6),        //!< The six faces
    (RSE_SPATIAL_BIN_NEIGHBORHOOD_FACE_EDGE, 18),  //!< Faces and edges
    (RSE_SPATIAL_BIN_NEIGHBORHOOD_FULL, 26)        //!< Every surrounding bin
)

/**
 * @brief The coordinate step for one direction.
 *
 * Each literal is a lookup table of two bit steps, one per direction, in the
 * order of RSE_SpatialBinDirection: 00 is zero, 01 is positive, 11 is negative.
 * Shifting a field to the top of an int and back down sign extends it into the
 * step. Regenerate the three literals together with the enum.
 */
HOST_DEVICE_INLINE_QUALIFIER RCoord3
spatialBinDirectionToOffset(RSE_SpatialBinDirection direction) {
    RIndex shift = RIndex(direction) * 2;
    RSignedIndex x_offset =
        RSignedIndex(int32_t(uint32_t((0x000FF5500F5F500DULL >> shift) & 3) << 30) >> 30);
    RSignedIndex y_offset =
        RSignedIndex(int32_t(uint32_t((0x000F5F5F500DD0D0ULL >> shift) & 3) << 30) >> 30);
    RSignedIndex z_offset =
        RSignedIndex(int32_t(uint32_t((0x000DDDDDDDD00D00ULL >> shift) & 3) << 30) >> 30);
    return RCoord3(x_offset, y_offset, z_offset);
}

/**
 * @brief Reorders object identifiers into the order the sort produced.
 */
class RSCF_ReorderSpatialBinObject : public RS_ComputeFunction<RSCF_ReorderSpatialBinObject> {
   public:
    using RS_ComputeFunction<RSCF_ReorderSpatialBinObject>::index1D;

    template <typename IndexVector, typename ObjectIdentifierVector>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        IndexVector sorted_object_indices,
        ObjectIdentifierVector object_identifiers,
        ObjectIdentifierVector sorted_object_identifiers
    ) {
        RIndex i = index1D();
        if (RSize(i) >= sorted_object_indices.size()) return;
        sorted_object_identifiers[i] = object_identifiers[sorted_object_indices[i]];
    }
};

/**
 * @brief Flags the first object of each bin run in the sorted object array.
 */
class RSCF_MarkSpatialBinStart : public RS_ComputeFunction<RSCF_MarkSpatialBinStart> {
   public:
    using RS_ComputeFunction<RSCF_MarkSpatialBinStart>::index1D;

    template <typename SpatialCodeVector, typename FlagVector>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(SpatialCodeVector sorted_spatial_codes, FlagVector start_flags) {
        RIndex i = index1D();
        if (RSize(i) >= sorted_spatial_codes.size()) return;
        start_flags[i] = (i == 0 || sorted_spatial_codes[i] != sorted_spatial_codes[i - 1])
                             ? RIndex(1)
                             : RIndex(0);
    }
};

/**
 * @brief Reads the bin count out of the last entry of the scanned run starts.
 */
class RSCF_WriteSpatialBinNum : public RS_ComputeFunction<RSCF_WriteSpatialBinNum> {
   public:
    using RS_ComputeFunction<RSCF_WriteSpatialBinNum>::index1D;

    template <typename IndexVector>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(IndexVector start_prefix_sums, RS_MemoryPtr bin_num) {
        if (index1D() != 0) return;
        bin_num.get<RIndex>(0) = start_prefix_sums[start_prefix_sums.size() - 1];
    }
};

/**
 * @brief Writes one record per bin, and tells every object which bin it landed in.
 */
class RSCF_WriteSpatialBinRecord : public RS_ComputeFunction<RSCF_WriteSpatialBinRecord> {
   public:
    using RS_ComputeFunction<RSCF_WriteSpatialBinRecord>::index1D;

    template <typename SpatialCodeVector, typename IndexVector, typename BinVector>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        SpatialCodeVector sorted_spatial_codes,
        IndexVector start_flags,
        IndexVector start_prefix_sums,
        IndexVector bin_offsets,
        BinVector bins
    ) {
        RIndex i = index1D();
        RSize object_num = sorted_spatial_codes.size();
        if (RSize(i) >= object_num) return;

        // An inclusive prefix sum, so a run's ordinal is its sum minus one.
        RIndex bin_index = start_prefix_sums[i] - 1;
        bin_offsets[i] = bin_index;

        if (start_flags[i] != RIndex(0)) {
            bins[bin_index].spatial_code = sorted_spatial_codes[i];
            bins[bin_index].object_offset = i;
        }
    }
};

/**
 * @brief Fills each bin's object count, once the offsets are known.
 */
class RSCF_WriteSpatialBinObjectNum : public RS_ComputeFunction<RSCF_WriteSpatialBinObjectNum> {
   public:
    using RS_ComputeFunction<RSCF_WriteSpatialBinObjectNum>::index1D;

    template <typename BinVector>
    HOST_DEVICE_INLINE_QUALIFIER static void run(BinVector bins, RIndex bin_num, RSize object_num) {
        RIndex i = index1D();
        if (RSize(i) >= bin_num) return;
        RIndex end = (i + 1 < bin_num) ? bins[i + 1].object_offset : RIndex(object_num);
        bins[i].object_num = end - bins[i].object_offset;
    }
};

/**
 * @brief Resolves every bin's neighbourhood, once per rebuild.
 *
 * One binary search per bin per direction, so no lookup is needed afterwards.
 */
template <typename SpatialCodeConverter, RIndex NeighborNum>
class RSCF_WriteSpatialBinNeighbor
    : public RS_ComputeFunction<RSCF_WriteSpatialBinNeighbor<SpatialCodeConverter, NeighborNum>> {
   public:
    using RS_ComputeFunction<RSCF_WriteSpatialBinNeighbor>::index1D;

    template <typename BinVector>
    HOST_DEVICE_INLINE_QUALIFIER static void run(BinVector bins, RIndex bin_num) {
        RIndex i = index1D();
        if (RSize(i) >= bin_num) return;

        RCoord3 coord = SpatialCodeConverter::toCoordinate(bins[i].spatial_code);
        for (RIndex direction = 0; direction < NeighborNum; ++direction) {
            RCoord3 offset = spatialBinDirectionToOffset(RSE_SpatialBinDirection(direction));
            RCoord3 neighbor(coord.x + offset.x, coord.y + offset.y, coord.z + offset.z);
            bins[i].neighbor_bins[direction] =
                _findBin(bins, bin_num, SpatialCodeConverter::toSpatialCode(neighbor));
        }
    }

   private:
    template <typename BinVector, typename SpatialCodeType>
    HOST_DEVICE_INLINE_QUALIFIER static RIndex
    _findBin(BinVector bins, RIndex bin_num, SpatialCodeType spatial_code) {
        if (spatial_code == SpatialCodeConverter::InvalidCode) return RS_NULL_INDEX;
        RIndex lo = 0;
        RIndex hi = bin_num;
        while (lo < hi) {
            RIndex mid = (lo + hi) >> 1;
            if (bins[mid].spatial_code < spatial_code) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo >= bin_num || bins[lo].spatial_code != spatial_code) return RS_NULL_INDEX;
        return lo;
    }
};

/**
 * @brief Converts a grid coordinate to a Morton spatial code and back.
 *
 * Every converter declares InvalidCode and answers with it for a coordinate it
 * cannot hold, so a lookup reports a miss without the caller checking first.
 */
template <typename SpatialCodeType>
class RS_SpatialBinMortonCodeConverter {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    /// Bits each coordinate gets: 10 for a 32 bit code, 21 for a 64 bit one.
    static constexpr RIndex CoordinateBits = RS_MortonCode<SpatialCodeType>::CoordinateBits;
    /// Signed coordinates are biased by this to reach the unsigned packing range.
    static constexpr RSignedIndex CoordinateBias = RSignedIndex(1) << (CoordinateBits - 1);
    /// No coordinate reaches this, since interleaving leaves the top bits clear.
    static constexpr SpatialCodeType InvalidCode = ~SpatialCodeType(0);

    /*****************************************************************
     * Methods
     *****************************************************************/
    /// @brief The code for a grid coordinate, or InvalidCode when it is out of bound.
    HOST_DEVICE_INLINE_QUALIFIER static SpatialCodeType toSpatialCode(const RCoord3& coord) {
        if (!isCoordinateInBound(coord)) return InvalidCode;
        return RS_MortonCode<SpatialCodeType>::toMortonCode(
            SpatialCodeType(coord.x + CoordinateBias),
            SpatialCodeType(coord.y + CoordinateBias),
            SpatialCodeType(coord.z + CoordinateBias)
        );
    }

    /// @brief The grid coordinate a code was built from.
    HOST_DEVICE_INLINE_QUALIFIER static RCoord3 toCoordinate(SpatialCodeType spatial_code) {
        SpatialCodeType x, y, z;
        RS_MortonCode<SpatialCodeType>::toCoordinate(spatial_code, x, y, z);
        return RCoord3(
            RSignedIndex(x) - CoordinateBias,
            RSignedIndex(y) - CoordinateBias,
            RSignedIndex(z) - CoordinateBias
        );
    }

    /// @brief True when a coordinate fits the code without saturating.
    HOST_DEVICE_INLINE_QUALIFIER static bool isCoordinateInBound(const RCoord3& coord) {
        return coord.x >= -CoordinateBias && coord.x < CoordinateBias &&
               coord.y >= -CoordinateBias && coord.y < CoordinateBias &&
               coord.z >= -CoordinateBias && coord.z < CoordinateBias;
    }
};

/**
 * @brief Default object identifier, holding only where the object lives.
 *
 * A bare index suits a contiguous store. A sparse SoA or SoV store puts an
 * RHandle here instead, and anything else the caller wants to carry alongside.
 */
struct RS_SpatialBinObjectIdentifier {
    RIndex object_index = RS_NULL_INDEX;
};

/**
 * @brief Alignment a bin takes, a sector for the larger neighbourhoods.
 */
template <typename SpatialCodeType, RIndex NeighborNum>
inline constexpr RSize RS_SPATIAL_BIN_ALIGNMENT =
    NeighborNum >= RSE_SPATIAL_BIN_NEIGHBORHOOD_FACE_EDGE
        ? RSize(32)
        : (alignof(SpatialCodeType) > alignof(RIndex) ? alignof(SpatialCodeType) : alignof(RIndex));

/**
 * @brief One occupied bin, ordered by code so a lookup can binary search.
 */
template <typename SpatialCodeType, RIndex NeighborNum>
struct alignas(RS_SPATIAL_BIN_ALIGNMENT<SpatialCodeType, NeighborNum>) RS_SpatialBin {
    SpatialCodeType spatial_code;
    /// Where this bin's objects begin in the sorted arrays.
    RIndex object_offset = 0;
    RIndex object_num = 0;
    /// RS_NULL_INDEX where that neighbour holds nothing.
    RIndex neighbor_bins[NeighborNum];
};

/**
 * @brief A bin with no neighbourhood, since a zero length array is not valid.
 */
template <typename SpatialCodeType>
struct alignas(RS_SPATIAL_BIN_ALIGNMENT<SpatialCodeType, 0>) RS_SpatialBin<SpatialCodeType, 0> {
    SpatialCodeType spatial_code;
    RIndex object_offset = 0;
    RIndex object_num = 0;
};

/**
 * @brief Bins objects by spatial code, and records how the bins neighbour each other.
 *
 * Built by sorting rather than hashing, so nothing atomic is involved and the
 * exact bin count is known before any payload is sized. A caller writes one
 * spatial code per object into `spatial_codes`, calls rebuild, and afterwards
 * reads bins through `bin_offsets` or `findBin`.
 *
 * Whether a bin owns a memory region is not this structure's concern. It hands
 * out ranges and neighbourhoods; the caller sizes and indexes its own arrays.
 */
template <
    typename SpatialCodeType = RIndex,
    typename ObjectIdentifierType = RS_SpatialBinObjectIdentifier,
    typename SpatialCodeConverter = RS_SpatialBinMortonCodeConverter<SpatialCodeType>,
    RSE_SpatialBinNeighborhood Neighborhood = RSE_SPATIAL_BIN_NEIGHBORHOOD_FACE>
class RS_SpatialBin3D {
   public:
    static constexpr RIndex NeighborNum = Neighborhood;
    using BinType = RS_SpatialBin<SpatialCodeType, NeighborNum>;

    class RS_SpatialBin3DHostData {
       public:
        RS_RadixSortPairs sorter;
        RS_Scan scanner;

        RS_Vector<RIndex> object_indices;
        RS_Vector<RIndex> sorted_object_indices;
        RS_Vector<SpatialCodeType> sorted_spatial_codes;
        RS_Vector<ObjectIdentifierType> sorted_object_identifiers;

        RS_Vector<RIndex> start_flags;
        RS_Vector<RIndex> start_prefix_sums;
    };

    /*****************************************************************
     * Data
     *****************************************************************/
    RS_SpatialBin3DHostData* host_data = nullptr;
    RS_MemoryProvider* mem_provider = nullptr;

    /// One per object, written by the caller before rebuild, sorted by it.
    RS_Vector<SpatialCodeType> spatial_codes;
    RS_Vector<ObjectIdentifierType> object_identifiers;
    /// One per object, which bin it landed in.
    RS_Vector<RIndex> bin_offsets;
    /// One per occupied bin, ordered by code.
    RS_Vector<BinType> bins;
    /// How many bins are occupied.
    RS_Value<RIndex> bin_num;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_SpatialBin3D() = default;

    /**
     * @brief Construct a spatial bin with the given memory provider.
     */
    explicit RS_SpatialBin3D(RS_MemoryProvider* provider) { init(provider); }

    HOST_DEVICE_INLINE_QUALIFIER RS_SpatialBin3D(const RS_SpatialBin3D& bin)
        : host_data(bin.host_data),
          mem_provider(bin.mem_provider),
          spatial_codes(bin.spatial_codes),
          object_identifiers(bin.object_identifiers),
          bin_offsets(bin.bin_offsets),
          bins(bin.bins),
          bin_num(bin.bin_num) {}

    /**
     * @brief Initialize the spatial bin with the given memory provider.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        host_data = new RS_SpatialBin3DHostData();
        mem_provider = provider;
        host_data->sorter = RS_RadixSortPairs(provider);
        host_data->scanner = RS_Scan(provider);
        host_data->object_indices.init(provider);
        host_data->sorted_object_indices.init(provider);
        host_data->sorted_spatial_codes.init(provider);
        host_data->sorted_object_identifiers.init(provider);
        host_data->start_flags.init(provider);
        host_data->start_prefix_sums.init(provider);

        spatial_codes.init(provider);
        object_identifiers.init(provider);
        bin_offsets.init(provider);
        bins.init(provider);
        bin_num.init(provider);
        bin_num.fill(0);
    }

    /// @brief Free the spatial bin.
    void free() {
        mem_provider = nullptr;
        if (host_data != nullptr) {
            host_data->sorter.free();
            host_data->scanner.free();
            host_data->object_indices.free();
            host_data->sorted_object_indices.free();
            host_data->sorted_spatial_codes.free();
            host_data->sorted_object_identifiers.free();
            host_data->start_flags.free();
            host_data->start_prefix_sums.free();

            delete host_data;
            host_data = nullptr;
        }

        spatial_codes.free();
        object_identifiers.free();
        bin_offsets.free();
        bins.free();
        bin_num.free();
    }

    /**
     * @brief Size every buffer for `size` objects, leaving the caller to write
     * spatial_codes and object_identifiers.
     */
    void resize(RSize size, bool copy = true) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        _ensure(spatial_codes, size, copy);
        _ensure(object_identifiers, size, copy);
        _ensure(bin_offsets, size, copy);
        _ensure(host_data->object_indices, size, copy);
        _ensure(host_data->sorted_object_indices, size, copy);
        _ensure(host_data->sorted_spatial_codes, size, copy);
        _ensure(host_data->sorted_object_identifiers, size, copy);
        _ensure(host_data->start_flags, size, copy);
        _ensure(host_data->start_prefix_sums, size, copy);
        // bins is not sized here, rebuild sizes it to the exact bin count.
    }

    /**
     * @brief Sort by code, compact into bin records, then resolve every
     * neighbourhood. Operates in place on the current spatial_codes.
     * @param sync Whether the rebuild operation is synchronous or asynchronous.
     */
    void rebuild(bool sync = true) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        RSize object_num = spatial_codes.size();
        if (object_num == 0) {
            // Drop what the last rebuild left, so findBin cannot reach it.
            bin_num.fill(0);
            _ensure(bins, 0, false);
            return;
        }

        castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
            p_ptr->template run<RSCF_GenerateIndexArray1DLike>(
                {object_num, RS_AUTO_BLOCK_SIZE, false}, host_data->object_indices
            );

            host_data->sorter.sort(
                spatial_codes,
                host_data->object_indices,
                host_data->sorted_spatial_codes,
                host_data->sorted_object_indices
            );

            p_ptr->template run<RSCF_ReorderSpatialBinObject>(
                {object_num, RS_AUTO_BLOCK_SIZE, false},
                host_data->sorted_object_indices,
                object_identifiers,
                host_data->sorted_object_identifiers
            );

            // The sorted results become the public arrays, at no copy cost.
            exchange(spatial_codes, host_data->sorted_spatial_codes);
            exchange(object_identifiers, host_data->sorted_object_identifiers);

            p_ptr->template run<RSCF_MarkSpatialBinStart>(
                {object_num, RS_AUTO_BLOCK_SIZE, false}, spatial_codes, host_data->start_flags
            );
            host_data->scanner.inclusiveSum(host_data->start_flags, host_data->start_prefix_sums);

            // Synchronous, because getBinNum reads back what this writes.
            p_ptr->template run<RSCF_WriteSpatialBinNum>(
                {1, RS_AUTO_BLOCK_SIZE, true}, host_data->start_prefix_sums, bin_num.data
            );

            RSize occupied_bin_num = getBinNum();
            if (occupied_bin_num > 0) {
                _ensure(bins, occupied_bin_num, false);

                p_ptr->template run<RSCF_WriteSpatialBinRecord>(
                    {object_num, RS_AUTO_BLOCK_SIZE, false},
                    spatial_codes,
                    host_data->start_flags,
                    host_data->start_prefix_sums,
                    bin_offsets,
                    bins
                );

                p_ptr->template run<RSCF_WriteSpatialBinObjectNum>(
                    {occupied_bin_num, RS_AUTO_BLOCK_SIZE, false},
                    bins,
                    RIndex(occupied_bin_num),
                    object_num
                );

                if constexpr (NeighborNum > 0) {
                    p_ptr->template run<
                        RSCF_WriteSpatialBinNeighbor<SpatialCodeConverter, NeighborNum>>(
                        {occupied_bin_num, RS_AUTO_BLOCK_SIZE, false},
                        bins,
                        RIndex(occupied_bin_num)
                    );
                }
            }

            // Synchronize memory provider stream to make sure modifications are completed.
            if (sync) p_ptr->sync();
        });
    }

    /**
     * @brief How many bins are occupied, read back to the host.
     *
     * Exact after rebuild, so a caller that wants a memory region per bin sizes
     * it once with no overflow path.
     */
    RSize getBinNum() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RIndex value = 0;
        bin_num.read(&value);
        return RSize(value);
    }

    /// @brief Returns the memory provider.
    [[nodiscard]] RS_MemoryProvider* provider() const { return mem_provider; }

    /// @brief Returns the device location.
    [[nodiscard]] RS_DeviceLocation device() const { return spatial_codes.device(); }

    /// @brief Returns the base size in bytes, including the host data it holds.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize baseBytes() const {
        return sizeof(*this) +
               (host_data != nullptr ? RBigSize(sizeof(RS_SpatialBin3DHostData)) : 0);
    }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize extraBytes() const {
        return spatial_codes.extraBytes() + object_identifiers.extraBytes() +
               bin_offsets.extraBytes() + bins.extraBytes() + bin_num.extraBytes();
    }

    /**
     * @brief The bin holding a spatial code, or RS_NULL_INDEX when none does.
     *
     * A coordinate the converter cannot hold arrives as InvalidCode, so a lookup
     * out of bound needs no check of its own.
     */
    HOST_DEVICE_INLINE_QUALIFIER RIndex findBin(SpatialCodeType spatial_code) const {
        if (spatial_code == SpatialCodeConverter::InvalidCode) return RS_NULL_INDEX;
        RIndex lo = 0;
        RIndex hi = RIndex(bins.size());
        while (lo < hi) {
            RIndex mid = (lo + hi) >> 1;
            if (bins[mid].spatial_code < spatial_code) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo >= RIndex(bins.size()) || bins[lo].spatial_code != spatial_code)
            return RS_NULL_INDEX;
        return lo;
    }

    /**
     * @brief The record at bin_index, or nullptr when it is out of bounds.
     *
     * RS_NULL_INDEX is out of bounds by construction, so an absent neighbour and
     * a bad index need only this one check.
     */
    HOST_DEVICE_INLINE_QUALIFIER BinType* getBin(RIndex bin_index) {
        if (RSize(bin_index) >= bins.size()) return nullptr;
        return &bins[bin_index];
    }

   protected:
    /*****************************************************************
     * Methods
     *****************************************************************/
    /// @brief Grow a vector to hold `size` entries and mark them occupied.
    template <typename Vector>
    static void _ensure(Vector& vector, RSize size, bool copy) {
        if (vector.capacity() < size) vector.resize(size, copy);
        vector.occupy(size);
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_SPATIAL_BIN_H
