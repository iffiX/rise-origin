// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_LIGHT_BIN_H
#define RS_LIGHT_BIN_H

#include "rise_origin/device/algorithm/rs_search.h"
#include "rise_origin/device/data/rs_spatial_bin.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/utils/math/rs_coord.h"
#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/meta/rs_constexpr.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

/**
 * @brief Spatial code conversion for the light bin, starting from a world position.
 */
class RS_LightBinSpatialCodeConverter : public device::RS_SpatialBinMortonCodeConverter<RIndex> {
   public:
    using BaseType = device::RS_SpatialBinMortonCodeConverter<RIndex>;
    using BaseType::isCoordinateInBound;
    using BaseType::toCoordinate;
    using BaseType::toSpatialCode;

    /*****************************************************************
     * Methods
     *****************************************************************/
    /**
     * @brief The grid coordinate a world position falls in.
     *
     * @param position   World position.
     * @param cell_size  Cell edge in meters.
     */
    template <typename FloatType>
    HOST_DEVICE_INLINE_QUALIFIER static RCoord3
    toCoordinate(const RS_Vec3D<FloatType>& position, float cell_size) {
        using RS_FI = RS_FloatIntrinsics<FloatType>;
        FloatType inverse_cell_size = FloatType(1) / FloatType(cell_size);
        return RCoord3(
            RSignedIndex(RS_FI::rs_floor(position.x * inverse_cell_size)),
            RSignedIndex(RS_FI::rs_floor(position.y * inverse_cell_size)),
            RSignedIndex(RS_FI::rs_floor(position.z * inverse_cell_size))
        );
    }

    /**
     * @brief The code of the cell a world position falls in.
     *
     * @param position   World position.
     * @param cell_size  Cell edge in meters.
     */
    template <typename FloatType>
    HOST_DEVICE_INLINE_QUALIFIER static RIndex
    toSpatialCode(const RS_Vec3D<FloatType>& position, float cell_size) {
        return BaseType::toSpatialCode(toCoordinate(position, cell_size));
    }
};

/**
 * @brief What the light bin carries for each light it holds.
 */
struct RS_LightBinObjectIdentifier {
    /// Emissive voxel this light is; null where the bin holds nothing.
    RHandle voxel = RS_NULL_HANDLE;
};

using RS_LightBinSpatialBin = device::RS_SpatialBin3D<
    RIndex,
    RS_LightBinObjectIdentifier,
    RS_LightBinSpatialCodeConverter,
    device::RSE_SPATIAL_BIN_NEIGHBORHOOD_FACE>;

/**
 * @brief Sums the weight each bin holds, so a probe reads one number.
 */
class RSCF_WriteLightBinTotal : public device::RS_ComputeFunction<RSCF_WriteLightBinTotal> {
   public:
    using device::RS_ComputeFunction<RSCF_WriteLightBinTotal>::index1D;

    template <typename BinVector, typename WeightVector>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(BinVector bins, WeightVector weight_cdf, WeightVector bin_total_weights) {
        RIndex i = index1D();
        if (RSize(i) >= bins.size()) return;
        if (bins[i].object_num == 0) {
            bin_total_weights[i] = 0.0f;
            return;
        }

        RIndex begin = bins[i].object_offset;
        RIndex end = begin + bins[i].object_num;
        float base = begin > 0 ? weight_cdf[begin - 1] : 0.0f;
        float total = weight_cdf[end - 1] - base;
        bin_total_weights[i] = total > 0.0f ? total : 0.0f;
    }
};

/**
 * @brief Which lights a shading point can reach, and how their power divides.
 *
 * A spatial bin of the lights with an inclusive weight scan over it. A shading
 * point draws from its own cell and the six touching it, in proportion to power.
 * A bin's total weight is a difference of two scan entries.
 */
class RS_LightBin : public RS_LightBinSpatialBin {
   public:
    using BaseType = RS_LightBinSpatialBin;

    /// Cells a shading point may draw from: its own plus the six faces.
    static constexpr RIndex MaxProbeNum = 1 + BaseType::NeighborNum;

    /*****************************************************************
     * Data
     *****************************************************************/
    /// One per light, written by the caller in object order and sorted by rebuild.
    device::RS_Vector<float> weights;
    /// Inclusive scan of weights.
    device::RS_Vector<float> weight_cdf;
    /// One per bin, the weight it holds, so a probe reads one number.
    device::RS_Vector<float> bin_total_weights;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_LightBin() = default;

    /**
     * @brief Construct a light bin with the given memory provider.
     *
     * @param provider  Memory provider to allocate against.
     */
    explicit RS_LightBin(RS_MemoryProvider* provider) { init(provider); }

    HOST_DEVICE_INLINE_QUALIFIER RS_LightBin(const RS_LightBin& light_bin)
        : BaseType(light_bin),
          weights(light_bin.weights),
          weight_cdf(light_bin.weight_cdf),
          bin_total_weights(light_bin.bin_total_weights),
          _sorted_weights(light_bin._sorted_weights) {}

    /**
     * @brief Initialize the light bin with the given memory provider.
     *
     * @param provider  Memory provider to allocate against.
     */
    void init(RS_MemoryProvider* provider) {
        BaseType::init(provider);
        weights.init(provider);
        weight_cdf.init(provider);
        bin_total_weights.init(provider);
        _sorted_weights.init(provider);
    }

    /// @brief Free the light bin.
    void free() {
        weights.free();
        weight_cdf.free();
        bin_total_weights.free();
        _sorted_weights.free();
        BaseType::free();
    }

    /**
     * @brief Size every buffer, leaving the caller to write the light arrays.
     *
     * @param light_num  How many lights the bin will hold.
     * @param copy       Keep the current contents when a buffer grows.
     */
    void resize(RSize light_num, bool copy = true) {
        BaseType::resize(light_num, copy);
        _ensure(weights, light_num, copy);
        _ensure(weight_cdf, light_num, copy);
        _ensure(_sorted_weights, light_num, copy);
    }

    /**
     * @brief Sort the lights into cells, then scan the weight they carry.
     *
     * @param sync Whether the rebuild operation is synchronous or asynchronous.
     */
    void rebuild(bool sync = true) {
        // The weight passes below queue after the base, so one sync at the end covers both.
        BaseType::rebuild(false);

        RSize light_num = object_identifiers.size();
        if (light_num == 0) {
            _ensure(bin_total_weights, 0, false);
            return;
        }

        device::castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
            p_ptr->template run<device::RSCF_ReorderSpatialBinObject>(
                {light_num, RS_AUTO_BLOCK_SIZE, false},
                host_data->sorted_object_indices,
                weights,
                _sorted_weights
            );
        });
        exchange(weights, _sorted_weights);
        host_data->scanner.template inclusiveSum<float, float>(weights, weight_cdf);

        RSize occupied_bin_num = bins.size();
        _ensure(bin_total_weights, occupied_bin_num, false);
        device::castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
            p_ptr->template run<RSCF_WriteLightBinTotal>(
                {occupied_bin_num, RS_AUTO_BLOCK_SIZE, false}, bins, weight_cdf, bin_total_weights
            );

            // Synchronize memory provider stream to make sure modifications are completed.
            if (sync) p_ptr->sync();
        });
    }

    /// @brief How many lights the bin holds.
    HOST_DEVICE_INLINE_QUALIFIER RSize getLightNum() const { return object_identifiers.size(); }

    /**
     * @brief Pick one light near a shading point, in proportion to its power.
     *
     * @param position          World space shading point.
     * @param cell_size         Cell edge in meters.
     * @param u                 Uniform sample in [0, 1).
     * @param pdf_out           Probability the returned light was picked.
     * @param total_probed_out  Summed weight of every cell this point draws from.
     * @return The light picked, default constructed when no cell holds one.
     */
    template <RIndex ProbeNum, typename FloatType>
    HOST_DEVICE_INLINE_QUALIFIER RS_LightBinObjectIdentifier sampleLight(
        const RS_Vec3D<FloatType>& position,
        float cell_size,
        FloatType u,
        FloatType& pdf_out,
        float& total_probed_out
    ) {
        static_assert(
            ProbeNum == 1 || ProbeNum == MaxProbeNum, "ProbeNum must be 1 or MaxProbeNum"
        );
        pdf_out = FloatType(0);
        total_probed_out = 0.0f;
        if (getLightNum() == 0) return RS_LightBinObjectIdentifier{};

        // Probe the own bin and its neighbours, skipping any that does not exist.
        RCoord3 coord = RS_LightBinSpatialCodeConverter::toCoordinate(position, cell_size);
        RIndex probe_bins[ProbeNum];
        float probe_totals[ProbeNum];
        probe_bins[0] = findBin(RS_LightBinSpatialCodeConverter::toSpatialCode(coord));

        if constexpr (ProbeNum > 1) {
            auto* own_bin = getBin(probe_bins[0]);
#pragma unroll
            for (RIndex probe = 1; probe < ProbeNum; probe++) {
                probe_bins[probe] = _getNeighborBin(own_bin, coord, probe - 1);
            }
        }

        float total_probed = 0.0f;
#pragma unroll
        for (RIndex probe = 0; probe < ProbeNum; probe++) {
            probe_totals[probe] = _evalBinTotal(probe_bins[probe]);
            total_probed += probe_totals[probe];
        }
        total_probed_out = total_probed;
        if (!(total_probed > 0.0f)) return RS_LightBinObjectIdentifier{};

        // Select a bin by u over the total weight of the probed bins.
        float bin_target = float(u) * total_probed;
        RIndex chosen_bin_index = RS_NULL_INDEX;
        float chosen_total = 0.0f;
        float chosen_base = 0.0f;
        float running = 0.0f;
#pragma unroll
        for (RIndex probe = 0; probe < ProbeNum; probe++) {
            if (probe_totals[probe] > 0.0f) {
                chosen_bin_index = probe_bins[probe];
                chosen_total = probe_totals[probe];
                chosen_base = running;
                if (bin_target < running + probe_totals[probe]) break;
            }
            running += probe_totals[probe];
        }
        if (!(chosen_total > 0.0f)) return RS_LightBinObjectIdentifier{};

        auto* chosen_bin = getBin(chosen_bin_index);
        if (chosen_bin == nullptr) return RS_LightBinObjectIdentifier{};

        // Reuse what is left of the uniform inside the chosen bin. The scan is
        // global, so the target is absolute too.
        RIndex begin = chosen_bin->object_offset;
        float base = begin > 0 ? weight_cdf[begin - 1] : 0.0f;
        float target = base + (bin_target - chosen_base);
        RIndex found = device::RSCF_BinaryPrefixSearchFloat<float>::run(
            target, weight_cdf.ptr() + begin, chosen_bin->object_num
        );
        if (found == RS_NULL_INDEX) return RS_LightBinObjectIdentifier{};

        RIndex light_slot = begin + found;
        pdf_out = FloatType(weights[light_slot]) / FloatType(total_probed);
        return object_identifiers[light_slot];
    }

    /**
     * @brief Summed weight of every cell a shading point draws from.
     *
     * @param position   World space shading point.
     * @param cell_size  Cell edge in meters.
     */
    template <RIndex ProbeNum, typename FloatType>
    HOST_DEVICE_INLINE_QUALIFIER float
    evalProbedTotalWeight(const RS_Vec3D<FloatType>& position, float cell_size) {
        static_assert(
            ProbeNum == 1 || ProbeNum == MaxProbeNum, "ProbeNum must be 1 or MaxProbeNum"
        );
        if (getLightNum() == 0) return 0.0f;

        RCoord3 coord = RS_LightBinSpatialCodeConverter::toCoordinate(position, cell_size);
        RIndex own_bin_index = findBin(RS_LightBinSpatialCodeConverter::toSpatialCode(coord));
        float total_probed = _evalBinTotal(own_bin_index);

        if constexpr (ProbeNum > 1) {
            auto* own_bin = getBin(own_bin_index);
#pragma unroll
            for (RIndex probe = 1; probe < ProbeNum; probe++) {
                total_probed += _evalBinTotal(_getNeighborBin(own_bin, coord, probe - 1));
            }
        }
        return total_probed;
    }

    /**
     * @brief Probability sampleLight at `position` would have picked a light of
     * `weight` sitting at `light_position`.
     *
     * @param light_position  World space position of the light being tested.
     * @param position        World space shading point the pick was made from.
     * @param cell_size       Cell edge in meters.
     * @param weight          The light's bin weight.
     * @param total_probed    Summed weight of the cells `position` draws from,
     *                        out of sampleLight or evalProbedTotalWeight.
     * @return The pick probability, or zero when the light is out of reach.
     */
    template <RIndex ProbeNum, typename FloatType>
    HOST_DEVICE_INLINE_QUALIFIER FloatType evalLightPdf(
        const RS_Vec3D<FloatType>& light_position,
        const RS_Vec3D<FloatType>& position,
        float cell_size,
        FloatType weight,
        FloatType total_probed
    ) {
        static_assert(
            ProbeNum == 1 || ProbeNum == MaxProbeNum, "ProbeNum must be 1 or MaxProbeNum"
        );
        if (getLightNum() == 0 || !(weight > FloatType(0))) return FloatType(0);
        if (!(total_probed > FloatType(0))) return FloatType(0);

        RCoord3 coord = RS_LightBinSpatialCodeConverter::toCoordinate(position, cell_size);
        RCoord3 light_coord =
            RS_LightBinSpatialCodeConverter::toCoordinate(light_position, cell_size);
        // One probe reaches only the point's own cell. Seven probes also reach a
        // face neighbour, which is any cell one step away in a single direction.
        RSignedIndex distance = RS_ABS(light_coord.x - coord.x) + RS_ABS(light_coord.y - coord.y) +
                                RS_ABS(light_coord.z - coord.z);
        if (distance > (ProbeNum > 1 ? 1 : 0)) return FloatType(0);
        return weight / total_probed;
    }

    /// @brief Returns the base size in bytes, including the host data it holds.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize baseBytes() const {
        return sizeof(*this) +
               (host_data != nullptr ? RBigSize(sizeof(BaseType::RS_SpatialBin3DHostData)) : 0);
    }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize extraBytes() const {
        return BaseType::extraBytes() + weights.extraBytes() + weight_cdf.extraBytes() +
               bin_total_weights.extraBytes() + _sorted_weights.extraBytes();
    }

   private:
    /*****************************************************************
     * Data
     *****************************************************************/
    /// Where the reorder writes, exchanged into weights after it runs.
    device::RS_Vector<float> _sorted_weights;

    /*****************************************************************
     * Methods
     *****************************************************************/
    /**
     * @brief The bin one step from a coordinate.
     *
     * Read off the own bin where it holds one, and searched for by code where
     * the point stands in a cell no light sits in.
     *
     * @param own_bin    The bin the coordinate itself falls in, or nullptr.
     * @param coord      Grid coordinate of the shading point.
     * @param direction  Which of the surrounding cells to reach.
     */
    HOST_DEVICE_INLINE_QUALIFIER RIndex
    _getNeighborBin(const BinType* own_bin, const RCoord3& coord, RIndex direction) {
        if (own_bin != nullptr) return own_bin->neighbor_bins[direction];
        return findBin(
            RS_LightBinSpatialCodeConverter::toSpatialCode(
                coord +
                device::spatialBinDirectionToOffset(device::RSE_SpatialBinDirection(direction))
            )
        );
    }

    /// @brief Summed weight of one bin, zero where the bin holds nothing.
    HOST_DEVICE_INLINE_QUALIFIER float _evalBinTotal(RIndex bin_index) {
        if (RSize(bin_index) >= bin_total_weights.size()) return 0.0f;
        return bin_total_weights[bin_index];
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_LIGHT_BIN_H
