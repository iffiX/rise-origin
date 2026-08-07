// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BVH_H
#define RS_BVH_H

#include "rise_origin/device/algorithm/rs_copy.h"
#include "rise_origin/device/algorithm/rs_sort.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/utils/math/rs_bound.h"
#include "rs_data_util.h"

namespace rise {

namespace device {

RS_ENUM(
    RSE_BVH_Unit,
    unsigned int,
    (RSE_BVH_UNIT_METERS, 0),
    (RSE_BVH_UNIT_DECIMETERS, 1),
    (RSE_BVH_UNIT_CENTIMETERS, 2),
    (RSE_BVH_UNIT_MILLIMETERS, 3)
)

/**
 * @brief Internal (non-leaf) BVH node holding both child indices and each child's full AABB.
 *
 * The child indices carry the leaf flag in their high bit (see IsLeafNodeMask).
 * Caching both child AABBs in the parent lets one node fetch during traversal
 * test both children — no scattered loads from separate child / bound arrays. A
 * node's own bound is union(left_bound, right_bound). Child bounds are filled
 * bottom-up by RSCF_BuildBVHTree.
 *
 * Padded to 32 bytes so a node starts on a sector boundary. Packed, the 56 byte
 * node starts at any multiple of 4 and a fetch touches three 32 byte sectors
 * where two would do; padding to 64 makes every fetch exactly two sectors, at
 * the cost of an eighth more memory for the node array.
 */
template <typename BoundType>
struct alignas(32) RS_BVHInternalNode {
    RIndex left = RS_NULL_INDEX;
    RIndex right = RS_NULL_INDEX;
    BoundType left_bound;
    BoundType right_bound;
};

template <RSize MortonResolutionModifier = 1, RSE_BVH_Unit Unit = RSE_BVH_UNIT_METERS>
class RSCF_GenerateMortonSpatialCode
    : public RS_ComputeFunction<RSCF_GenerateMortonSpatialCode<MortonResolutionModifier, Unit>> {
   public:
    using RS_ComputeFunction<RSCF_GenerateMortonSpatialCode>::index1D;

    template <typename BoundVector, typename SpatialCodeVector>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(BoundVector bounds, SpatialCodeVector spatial_codes) {
        RIndex idx = index1D();

        using BoundVec = typename BoundVector::ObjectType;
        using SpatialCode = typename SpatialCodeVector::ObjectType;
        using Float = typename BoundVec::ElementType;
        using RS_FI = RS_FloatIntrinsics<Float>;

        // Base range per unit (meters). 10 bits per axis for 32-bit, 21 bits for 64-bit.
        // range = base_range * unit_scale / MortonResolutionModifier
        //   base_range: 32-bit = 1024.0, 64-bit = 2097152.0 (covers ±range/2 in meters)
        //   unit_scale: meters=1, dm=0.1, cm=0.01, mm=0.001
        static constexpr Float unit_scale = Unit == RSE_BVH_UNIT_METERS        ? Float(1.0)
                                            : Unit == RSE_BVH_UNIT_DECIMETERS  ? Float(0.1)
                                            : Unit == RSE_BVH_UNIT_CENTIMETERS ? Float(0.01)
                                                                               : Float(0.001);
        static constexpr Float base_range = sizeof(SpatialCode) == 4 ? Float(1024.0)
                                                                     : Float(2097152.0);
        static constexpr Float range = base_range * unit_scale / Float(MortonResolutionModifier);
        static constexpr Float scale = sizeof(SpatialCode) == 4 ? Float(1024) : Float(2097152);

        auto centroid = bounds[idx].center();
        SpatialCode x = static_cast<SpatialCode>(
            RS_FI::rs_round_down((centroid.x / range + Float(0.5)) * scale)
        );
        SpatialCode y = static_cast<SpatialCode>(
            RS_FI::rs_round_down((centroid.y / range + Float(0.5)) * scale)
        );
        SpatialCode z = static_cast<SpatialCode>(
            RS_FI::rs_round_down((centroid.z / range + Float(0.5)) * scale)
        );

        SpatialCode c = _leftShift3(x) | (_leftShift3(y) << 1) | (_leftShift3(z) << 2);
        spatial_codes[idx] = c;
    }

   private:
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER static T _leftShift3(T x) {
        if constexpr (sizeof(T) == 4)
            return _leftShift3For32(x);
        else
            return _leftShift3For64(x);
    }

    HOST_DEVICE_INLINE_QUALIFIER static uint32_t _leftShift3For32(uint32_t x) {
        // Clamp to lower 10 bits; out-of-range centroids saturate at the upper grid bound.
        x = RS_MIN(x, (uint32_t)0x3ff);

        /* Below code is like this, each time, you shift a group of code
         * to the left, until you shift all bit to their corresponding place.
         * x = (x | (x << 16)) & 0b00000011000000000000000011111111;
         * x = (x | (x << 8)) & 0b00000011000000001111000000001111;
         * x = (x | (x << 4)) & 0b00000011000011000011000011000011;
         * x = (x | (x << 2)) & 0b00001001001001001001001001001001;
         */

        x = (x | (x << 16)) & 0x30000ff;
        x = (x | (x << 8)) & 0x300f00f;
        x = (x | (x << 4)) & 0x30c30c3;
        x = (x | (x << 2)) & 0x9249249;

        return x;
    }

    HOST_DEVICE_INLINE_QUALIFIER static uint64_t _leftShift3For64(uint64_t x) {
        // Clamp to lower 21 bits; out-of-range centroids saturate at the upper grid bound.
        x = RS_MIN(x, (uint64_t)0x1fffffULL);

        x = (x | (x << 32)) & 0x1F00000000FFFFULL;
        x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
        x = (x | (x << 8)) & 0x100F00F00F00F00FULL;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3ULL;
        x = (x | (x << 2)) & 0x1249249249249249ULL;

        return x;
    }
};

class RSCF_BuildBVHRadixTree : public RS_ComputeFunction<RSCF_BuildBVHRadixTree> {
   public:
    template <typename SpatialCodeVector, typename InternalNodeVector, typename NodeVector>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        SpatialCodeVector spatial_codes,
        InternalNodeVector internal_nodes,
        NodeVector internal_parent_nodes,
        NodeVector leaf_parent_nodes
    ) {
        // Since we are using SignedSize to do computation, the max node num could not exceed
        // half of the unsigned size. We can just use the highest bit of Node index to determine
        // if it is internal node
        static constexpr RIndex IsLeafNodeMask = sizeof(RIndex) == 4 ? 0x80000000
                                                                     : 0x8000000000000000;

        // Launch per internal node
        RSignedIndex i = index1D();

        using SpatialCode = typename SpatialCodeVector::ObjectType;
        SpatialCode i_spatial_code = spatial_codes[i];

        // Determine direction of the range (+1 or -1)
        RSignedSize d = _commonPrefix(i, i + 1, i_spatial_code, spatial_codes) >
                                _commonPrefix(i, i - 1, i_spatial_code, spatial_codes)
                            ? 1
                            : -1;

        // Compute upper bound for the length of the range
        RSignedSize delta_min = _commonPrefix(i, i - d, i_spatial_code, spatial_codes);
        RSignedSize l_max = 2;

        while (_commonPrefix(i, i + l_max * d, i_spatial_code, spatial_codes) > delta_min) {
            l_max <<= 1;
        }

        // Find the other end using binary search
        RSignedSize l = 0;
        for (RSignedSize t = l_max >> 1; t >= 1; t >>= 1) {
            if (_commonPrefix(i, i + (l + t) * d, i_spatial_code, spatial_codes) > delta_min) {
                l += t;
            }
        }

        RSignedSize j = i + l * d;

        //  Find the split position using binary search
        RSignedSize delta_node = _commonPrefix(i, j, i_spatial_code, spatial_codes);
        RSignedSize s = 0;

        for (RSignedSize t = (l + 1) >> 1; t > 0; t = (t + 1) >> 1) {
            if (_commonPrefix(i, i + (s + t) * d, i_spatial_code, spatial_codes) > delta_node) {
                s = s + t;
            }
            if (t == 1) break;
        }

        RSignedSize gamma = i + s * d + RS_MIN(d, 0);

        // Output child nodes
        if (RS_MIN(i, j) == gamma) {
            internal_nodes[i].left = static_cast<RIndex>(gamma) | IsLeafNodeMask;
            leaf_parent_nodes[gamma] = static_cast<RIndex>(i);
        } else {
            internal_nodes[i].left = static_cast<RIndex>(gamma);
            internal_parent_nodes[gamma] = static_cast<RIndex>(i);
        }

        if (RS_MAX(i, j) == gamma + 1) {
            internal_nodes[i].right = static_cast<RIndex>(gamma + 1) | IsLeafNodeMask;
            leaf_parent_nodes[gamma + 1] = static_cast<RIndex>(i);
        } else {
            internal_nodes[i].right = static_cast<RIndex>(gamma + 1);
            internal_parent_nodes[gamma + 1] = static_cast<RIndex>(i);
        }

#ifdef RS_DEBUG_BVH
        if (RS_MIN(i, j) == gamma && RS_MAX(i, j) == gamma + 1) {
            printf("(%d) -> L <%d> R <%d>\n", i, gamma, gamma + 1);
        } else if (RS_MIN(i, j) == gamma) {
            printf("(%d) -> L <%d> R (%d)\n", i, gamma, gamma + 1);
        } else if (RS_MAX(i, j) == gamma + 1) {
            printf("(%d) -> L (%d) R <%d>\n", i, gamma, gamma + 1);
        } else {
            printf("(%d) -> L (%d) R (%d)\n", i, gamma, gamma + 1);
        }
#endif
    }

   private:
    template <typename SpatialCode, typename SpatialCodeVector>
    HOST_DEVICE_INLINE_QUALIFIER static int _commonPrefix(
        RSignedIndex i,
        RSignedIndex j,
        const SpatialCode& i_spatial_code,
        SpatialCodeVector& spatial_codes
    ) {
        if (j >= 0 && j < spatial_codes.size()) {
            auto code_j = spatial_codes[j];
            // If morton code is the same, use index as fall back, as described in paper sec 4
            auto same_bits =
                i_spatial_code != code_j
                    ? countLeadingZeros(i_spatial_code ^ code_j)
                    : 64 + countLeadingZeros(static_cast<RIndex>(i) ^ static_cast<RIndex>(j));
            return static_cast<int>(same_bits);
        }
        return -1;
    }
};

class RSCF_BuildBVHTree : public RS_ComputeFunction<RSCF_BuildBVHTree> {
   public:
    template <
        typename InternalNodeVector,
        typename BoundVector,
        typename NodeVector,
        typename NodeCountVector>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        InternalNodeVector internal_nodes,
        BoundVector leaf_bounds,
        NodeVector internal_parent_nodes,
        NodeVector leaf_parent_nodes,
        NodeCountVector node_operation_counts
    ) {
        using Bound = typename BoundVector::ObjectType;

        // Launch per leaf node (object)
        RIndex i = index1D();

        RIndex current_internal_node = _getNodeIndex(leaf_parent_nodes[i]);
        int old_operation_count = deviceAtomicAdd(&node_operation_counts[current_internal_node], 1);

        while (old_operation_count != 0) {
            // while current thread is not the first one to process this node, otherwise kill it.
            // The second (final) arrival owns the node: both children are done, so cache each
            // child's full bound here. A child's full bound is its leaf bound, or the union of
            // its own cached child bounds.
            RIndex raw_left_node = internal_nodes[current_internal_node].left;
            RIndex raw_right_node = internal_nodes[current_internal_node].right;

            Bound left_bound;
            if (_isLeafNode(raw_left_node)) {
                left_bound = leaf_bounds[_getNodeIndex(raw_left_node)];
            } else {
                auto child = internal_nodes[_getNodeIndex(raw_left_node)];
                left_bound = child.left_bound.unionWith(child.right_bound);
            }

            Bound right_bound;
            if (_isLeafNode(raw_right_node)) {
                right_bound = leaf_bounds[_getNodeIndex(raw_right_node)];
            } else {
                auto child = internal_nodes[_getNodeIndex(raw_right_node)];
                right_bound = child.left_bound.unionWith(child.right_bound);
            }

            internal_nodes[current_internal_node].left_bound = left_bound;
            internal_nodes[current_internal_node].right_bound = right_bound;

            // Is root node, break
            if (current_internal_node == 0) break;

            // Goto parent node
            current_internal_node = _getNodeIndex(internal_parent_nodes[current_internal_node]);

            // Make sure node-bound writes are visible to other threads
            threadfence();

            old_operation_count = deviceAtomicAdd(&node_operation_counts[current_internal_node], 1);
        }
    }

   private:
    HOST_DEVICE_INLINE_QUALIFIER static bool _isLeafNode(RIndex node) {
        static constexpr RIndex IsLeafNodeMask = sizeof(RIndex) == 4 ? 0x80000000
                                                                     : 0x8000000000000000;

        return node & IsLeafNodeMask;
    }

    HOST_DEVICE_INLINE_QUALIFIER static RIndex _getNodeIndex(RIndex node) {
        static constexpr RIndex NodeIndexMask = sizeof(RIndex) == 4 ? 0x7FFFFFFF
                                                                    : 0x7FFFFFFFFFFFFFFF;

        return node & NodeIndexMask;
    }
};

class RSCF_GenerateIndexVector : public RS_ComputeFunction<RSCF_GenerateIndexVector> {
   public:
    template <typename IndexVector>
    HOST_DEVICE_INLINE_QUALIFIER static void run(IndexVector indices) {
        RIndex i = index1D();
        indices[i] = i;
    }
};

class RSCF_ReorderObjectIdentifiersAndBounds
    : public RS_ComputeFunction<RSCF_ReorderObjectIdentifiersAndBounds> {
   public:
    template <typename IndexVector, typename ObjectIdentifierVector, typename BoundVector>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        IndexVector indices,
        ObjectIdentifierVector input_object_identifiers,
        ObjectIdentifierVector output_object_identifiers,
        BoundVector input_bounds,
        BoundVector output_bounds
    ) {
        RIndex i = index1D();
        RIndex idx = indices[i];
        output_object_identifiers[i] = input_object_identifiers[idx];
        output_bounds[i] = input_bounds[idx];
    }
};

struct RS_BVHObjectIdentifier {
    /// Per-leaf cached state for efficient traversal-time filtering.
    RState object_metadata = 0;
    /// Handle pointing to the actual leaf data.
    RHandle object_handle = RS_NULL_HANDLE;

    bool operator==(const RS_BVHObjectIdentifier& other) const {
        return object_metadata == other.object_metadata && object_handle == other.object_handle;
    }
};

struct RS_BVHObjectIdentifierHostHasher {
    std::size_t operator()(const RS_BVHObjectIdentifier& identifier) const {
        return std::hash<RState>()(identifier.object_metadata) ^
               std::hash<RHandle>()(identifier.object_handle);
    }
};

/**
 * Implemented BVH construction and optimization algorithm from:
 *
 * Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
 * by Tero Karras
 * https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/
 * 2012/11/karras2012hpg_paper.pdf
 */
template <
    typename SpatialCodeType = RBigIndex,
    typename ObjectIdentifierType = RS_BVHObjectIdentifier,
    typename ObjectIdentifierHostHasher = RS_BVHObjectIdentifierHostHasher,
    typename SpatialCodeComputeFunction =
        RSCF_GenerateMortonSpatialCode<1, RSE_BVH_UNIT_CENTIMETERS>,
    typename BoundType = RBound3f>
class RS_BVHBase {
   public:
    static constexpr RSize MaxObjectNum = sizeof(RIndex) == 4 ? 0x7FFFFFFE : 0x7FFFFFFFFFFFFFFE;

    class RS_BVHHostData {
       public:
        std::unordered_map<ObjectIdentifierType, RIndex, ObjectIdentifierHostHasher> object_indices;

        RS_Vector<RIndex> indices;
        RS_Vector<RIndex> sorted_indices;

        RS_Vector<SpatialCodeType> sorted_spatial_codes;
        RS_Vector<ObjectIdentifierType> sorted_object_identifiers;
        RS_Vector<BoundType> sorted_bounds;

        RS_Vector<int> node_operation_count;
    };

    class RS_BVHAccessor {
       public:
        static constexpr RIndex IsLeafNodeMask = sizeof(RIndex) == 4 ? 0x80000000
                                                                     : 0x8000000000000000;
        static constexpr RIndex NodeIndexMask = sizeof(RIndex) == 4 ? 0x7FFFFFFF
                                                                    : 0x7FFFFFFFFFFFFFFF;

        ObjectIdentifierType* object_identifiers_ptr = nullptr;
        RS_BVHInternalNode<BoundType>* internal_nodes_ptr = nullptr;
        BoundType* leaf_bounds_ptr = nullptr;

        RIndex current_node = RS_NULL_INDEX;

        HOST_DEVICE_INLINE_QUALIFIER RIndex left() const {
            // If current node is invalid, or current node is leaf node
            if (current_node == RS_NULL_INDEX || (current_node & IsLeafNodeMask))
                return RS_NULL_INDEX;
            return internal_nodes_ptr[current_node & NodeIndexMask].left;
        }

        HOST_DEVICE_INLINE_QUALIFIER RIndex right() const {
            // If current node is invalid, or current node is leaf node
            if (current_node == RS_NULL_INDEX || (current_node & IsLeafNodeMask))
                return RS_NULL_INDEX;
            return internal_nodes_ptr[current_node & NodeIndexMask].right;
        }

        HOST_DEVICE_INLINE_QUALIFIER RIndex current() const { return current_node; }

        HOST_DEVICE_INLINE_QUALIFIER bool isLeaf() const {
            if (current_node == RS_NULL_INDEX) return false;
            return current_node & IsLeafNodeMask;
        }

        HOST_DEVICE_INLINE_QUALIFIER void moveTo(RIndex node) { current_node = node; }

        HOST_DEVICE_INLINE_QUALIFIER BoundType bound() const {
            if (current_node == RS_NULL_INDEX) return BoundType::zero();
            if (current_node & IsLeafNodeMask) return leaf_bounds_ptr[current_node & NodeIndexMask];
            // An internal node's own bound is the union of its two child bounds.
            const RS_BVHInternalNode<BoundType>& node =
                internal_nodes_ptr[current_node & NodeIndexMask];
            return node.left_bound.unionWith(node.right_bound);
        }

        HOST_DEVICE_INLINE_QUALIFIER RS_BVHInternalNode<BoundType> internalNode() const {
            return internal_nodes_ptr[current_node & NodeIndexMask];
        }

        HOST_DEVICE_INLINE_QUALIFIER ObjectIdentifierType* object() const {
            if (current_node == RS_NULL_INDEX || !(current_node & IsLeafNodeMask)) return nullptr;
            return object_identifiers_ptr + (current_node & NodeIndexMask);
        }
    };

    /*****************************************************************
     * Data
     *****************************************************************/
    RS_BVHHostData* host_data = nullptr;
    RS_MemoryProvider* mem_provider = nullptr;
    RS_RadixSortPairs sorter;
    RS_Vector<SpatialCodeType> spatial_codes;
    RS_Vector<ObjectIdentifierType> object_identifiers;

    RS_Vector<RS_BVHInternalNode<BoundType>> internal_nodes;
    RS_Vector<RIndex> internal_parent_nodes;

    RS_Vector<RIndex> leaf_parent_nodes;
    RS_Vector<BoundType> leaf_bounds;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_BVHBase() = default;

    /**
     * @brief Construct a bvh tree with the given memory provider.
     * @param provider The memory provider.
     */
    explicit RS_BVHBase(RS_MemoryProvider* provider) { init(provider); }

    /**
     * @brief Initialize the bvh tree with the given memory provider.
     * @param provider The memory provider.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        host_data = new RS_BVHHostData();
        mem_provider = provider;
        sorter = RS_RadixSortPairs(provider);
        spatial_codes.init(provider);
        object_identifiers.init(provider);

        internal_nodes.init(provider);
        internal_parent_nodes.init(provider);

        leaf_parent_nodes.init(provider);
        leaf_bounds.init(provider);

        host_data->indices.init(provider);
        host_data->sorted_indices.init(provider);
        host_data->sorted_spatial_codes.init(provider);
        host_data->sorted_object_identifiers.init(provider);
        host_data->sorted_bounds.init(provider);
        host_data->node_operation_count.init(provider);
    }

    HOST_DEVICE_INLINE_QUALIFIER RS_BVHBase(const RS_BVHBase& bvh)
        : host_data(bvh.host_data),
          mem_provider(bvh.mem_provider),
          sorter(bvh.sorter),
          spatial_codes(bvh.spatial_codes),
          object_identifiers(bvh.object_identifiers),
          internal_nodes(bvh.internal_nodes),
          internal_parent_nodes(bvh.internal_parent_nodes),
          leaf_parent_nodes(bvh.leaf_parent_nodes),
          leaf_bounds(bvh.leaf_bounds) {}

    /// @brief Free the bvh tree.
    void free() {
        mem_provider = nullptr;
        if (host_data != nullptr) {
            host_data->indices.free();
            host_data->sorted_indices.free();
            host_data->sorted_spatial_codes.free();
            host_data->sorted_object_identifiers.free();
            host_data->sorted_bounds.free();
            host_data->node_operation_count.free();

            delete host_data;
            host_data = nullptr;
        }

        sorter.free();
        spatial_codes.free();
        object_identifiers.free();

        internal_nodes.free();
        internal_parent_nodes.free();

        leaf_parent_nodes.free();
        leaf_bounds.free();
    }

    /// @brief Copy from another bvh tree.
    void copy(const RS_BVHBase& bvh) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(bvh.mem_provider != nullptr);

        if (bvh.spatial_codes.data == spatial_codes.data) return;

        spatial_codes.copy(bvh.spatial_codes);
        object_identifiers.copy(bvh.object_identifiers);
        internal_nodes.copy(bvh.internal_nodes);
        internal_parent_nodes.copy(bvh.internal_parent_nodes);
        leaf_parent_nodes.copy(bvh.leaf_parent_nodes);
        leaf_bounds.copy(bvh.leaf_bounds);

        host_data->indices.resize(bvh.host_data->indices.capacity(), false);
        host_data->sorted_indices.resize(bvh.host_data->sorted_indices.capacity(), false);
        host_data->sorted_spatial_codes.resize(
            bvh.host_data->sorted_spatial_codes.capacity(), false
        );
        host_data->sorted_object_identifiers.resize(
            bvh.host_data->sorted_object_identifiers.capacity(), false
        );
        host_data->sorted_bounds.resize(bvh.host_data->sorted_bounds.capacity(), false);
        host_data->node_operation_count.resize(
            bvh.host_data->node_operation_count.capacity(), false
        );

        rebuildHostIndices();
    }

    /// @brief Move the bvh tree to the specified memory provider.
    RS_BVHBase move(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_BVHBase new_bvh;

        new_bvh.host_data = new RS_BVHHostData(*host_data);
        new_bvh.mem_provider = provider;
        new_bvh.sorter = RS_RadixSortPairs(provider);
        new_bvh.spatial_codes = spatial_codes.move(provider);
        new_bvh.object_identifiers = object_identifiers.move(provider);
        new_bvh.internal_nodes = internal_nodes.move(provider);
        new_bvh.internal_parent_nodes = internal_parent_nodes.move(provider);
        new_bvh.leaf_parent_nodes = leaf_parent_nodes.move(provider);
        new_bvh.leaf_bounds = leaf_bounds.move(provider);

        new_bvh.host_data->indices.init(host_data->indices.capacity(), mem_provider);
        new_bvh.host_data->sorted_indices.init(host_data->sorted_indices.capacity(), mem_provider);
        new_bvh.host_data->sorted_spatial_codes.init(
            host_data->sorted_spatial_codes.capacity(), mem_provider
        );
        new_bvh.host_data->sorted_object_identifiers.init(
            host_data->sorted_object_identifiers.capacity(), mem_provider
        );
        new_bvh.host_data->sorted_bounds.init(host_data->sorted_bounds.capacity(), mem_provider);
        new_bvh.host_data->node_operation_count.init(
            host_data->node_operation_count.capacity(), mem_provider
        );

        return new_bvh;
    }

    /**
     * @brief Manually resize the nodes in the bvh tree.
     * @param size The new size.
     * @param copy Whether to copy the existing data.
     */
    void resize(RSize size, bool copy = true) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        if (size == 0) {
            // Clear everything.
            object_identifiers.clear();
            spatial_codes.clear();

            internal_nodes.clear();
            internal_parent_nodes.clear();

            leaf_parent_nodes.clear();
            leaf_bounds.clear();

            host_data->indices.clear();
            host_data->sorted_indices.clear();
            host_data->sorted_spatial_codes.clear();
            host_data->sorted_object_identifiers.clear();
            host_data->sorted_bounds.clear();
            host_data->node_operation_count.clear();
        } else {
            if (object_identifiers.capacity() < size) {
                RSize capacity = size;
                object_identifiers.resize(capacity, copy);
                spatial_codes.resize(capacity, copy);

                internal_nodes.resize(capacity, copy);
                internal_parent_nodes.resize(capacity, copy);

                leaf_parent_nodes.resize(capacity, copy);
                leaf_bounds.resize(capacity, copy);

                host_data->indices.resize(capacity, false);
                host_data->sorted_indices.resize(capacity, false);
                host_data->sorted_spatial_codes.resize(capacity, false);
                host_data->sorted_object_identifiers.resize(capacity, false);
                host_data->sorted_bounds.resize(capacity, false);
                host_data->node_operation_count.resize(capacity, false);
            }

            object_identifiers.occupy(size);
            spatial_codes.occupy(size);

            internal_nodes.occupy(size - 1);
            internal_parent_nodes.occupy(size - 1);

            leaf_parent_nodes.occupy(size);
            leaf_bounds.occupy(size);

            host_data->indices.occupy(size);
            host_data->sorted_indices.occupy(size);
            host_data->sorted_spatial_codes.occupy(size);
            host_data->sorted_object_identifiers.occupy(size);
            host_data->sorted_bounds.occupy(size);
            host_data->node_operation_count.occupy(size);
        }
    }

    /**
     * @brief Build the bvh tree from the provided object identifiers and bounds.
     * @param object_identifiers The object identifiers of the objects in the bvh tree.
     * @param bounds The bounds of the objects in the bvh tree.
     */
    template <typename ObjectIdentifierContainer, typename BoundContainer>
    void build(ObjectIdentifierContainer object_identifiers, BoundContainer bounds) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_EXPECT_ARGUMENT(
            object_identifiers.size() == bounds.size(),
            "object_identifiers.size()={}, bounds.size()={}",
            object_identifiers.size(),
            bounds.size()
        );
        RS_DATA_EXPECT_ARGUMENT(
            object_identifiers.size() < MaxObjectNum,
            "object_identifiers.size()={}, MaxObjectNum={}",
            object_identifiers.size(),
            MaxObjectNum
        );

        if (object_identifiers.size() == 0) return;
        RSize new_size = object_identifiers.size();

        resize(new_size);

        device::copy(this->object_identifiers, object_identifiers, new_size);
        device::copy(leaf_bounds, bounds, new_size);

        rebuild();
        rebuildHostIndices();
    }

    /**
     * @brief Build the bvh tree from the provided object identifiers and bounds.
     * @param object_identifiers The object identifiers of the objects in the bvh tree.
     * @param bounds The bounds of the objects in the bvh tree.
     */
    void
    build(std::vector<ObjectIdentifierType>& object_identifiers, std::vector<BoundType>& bounds) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_EXPECT_ARGUMENT(
            object_identifiers.size() == bounds.size(),
            "object_identifiers.size()={}, bounds.size()={}",
            object_identifiers.size(),
            bounds.size()
        );
        RS_DATA_EXPECT_ARGUMENT(
            object_identifiers.size() < MaxObjectNum,
            "object_identifiers.size()={}, MaxObjectNum={}",
            object_identifiers.size(),
            MaxObjectNum
        );

        if (object_identifiers.empty()) return;
        RSize new_size = object_identifiers.size();

        resize(new_size);

        this->object_identifiers.write(object_identifiers.data());
        leaf_bounds.write(bounds.data());

        rebuild();
        rebuildHostIndices();
    }

    /**
     * @brief Refit the bvh tree in place: re-runs only the bottom-up bound
     * propagation pass against the current `leaf_bounds`, recomputing the cached
     * child bounds in `internal_nodes`. Topology (the child links in
     * `internal_nodes`, `internal_parent_nodes`, `leaf_parent_nodes`, and the
     * Morton-sorted spatial codes / object identifiers) is left untouched.
     * Requires that the bvh has been built or rebuilt at least once before; a
     * no-op when the tree is empty.
     */
    void refit() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
            RSize size = object_identifiers.size();
            if (size == 0) return;

            // Clear node operation count so RSCF_BuildBVHTree's atomic
            // per-internal-node arrival counter starts from zero.
            host_data->node_operation_count.fill(0, false);

            if (size > 1) {
                // launch per leaf node (object); propagates leaf_bounds up
                // through the existing topology.
                p_ptr->template run<RSCF_BuildBVHTree>(
                    {size, RS_AUTO_BLOCK_SIZE, false},
                    internal_nodes,
                    leaf_bounds,
                    internal_parent_nodes,
                    leaf_parent_nodes,
                    host_data->node_operation_count
                );
            }
        });
    }

    /**
     * @brief Rebuild the bvh tree, call these after you directly write data to
     * `object_identifiers` and `leaf_bounds` of the bvh tree.
     */
    void rebuild() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
            RSize size = object_identifiers.size();

            if (size == 0) return;

            p_ptr->template run<SpatialCodeComputeFunction>(
                {size, RS_AUTO_BLOCK_SIZE, false}, leaf_bounds, spatial_codes
            );

            // Clear node operation count
            host_data->node_operation_count.fill(0, false);

            p_ptr->template run<RSCF_GenerateIndexVector>(
                {size, RS_AUTO_BLOCK_SIZE, false}, host_data->indices
            );

            // Sort
            sorter.sort(
                spatial_codes,
                host_data->indices,
                host_data->sorted_spatial_codes,
                host_data->sorted_indices
            );

            // Reorder data
            p_ptr->template run<RSCF_ReorderObjectIdentifiersAndBounds>(
                {size, RS_AUTO_BLOCK_SIZE, false},
                host_data->sorted_indices,
                object_identifiers,
                host_data->sorted_object_identifiers,
                leaf_bounds,
                host_data->sorted_bounds
            );

            exchange(spatial_codes, host_data->sorted_spatial_codes);
            exchange(object_identifiers, host_data->sorted_object_identifiers);
            exchange(leaf_bounds, host_data->sorted_bounds);

            if (size > 1) {
                // launch per internal node
                p_ptr->template run<RSCF_BuildBVHRadixTree>(
                    {size - 1, RS_AUTO_BLOCK_SIZE, false},
                    spatial_codes,
                    internal_nodes,
                    internal_parent_nodes,
                    leaf_parent_nodes
                );

                // launch per leaf node (object)
                p_ptr->template run<RSCF_BuildBVHTree>(
                    {size, RS_AUTO_BLOCK_SIZE, false},
                    internal_nodes,
                    leaf_bounds,
                    internal_parent_nodes,
                    leaf_parent_nodes,
                    host_data->node_operation_count
                );
            }
        });
    }

    /**
     * @brief Rebuild host indices of the tree, can be separate from  @ref rebuild . Only
     * needed if @ref insert, @ref erase, @ref update @ref find, @ref contains are needed.
     */
    void rebuildHostIndices() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        // Update host object indices
        RS_MemoryPtr host_object_identifiers_ptr = mem_provider->allocateHost(
            sizeof(ObjectIdentifierType) * object_identifiers.size(), true
        );
        host_data->sorted_object_identifiers.read(
            host_object_identifiers_ptr.as<ObjectIdentifierType>()
        );
        host_data->object_indices.clear();
        for (RIndex i = 0; i < object_identifiers.size(); i++) {
            host_data->object_indices[host_object_identifiers_ptr.get<ObjectIdentifierType>(i)] = i;
        }
        host_object_identifiers_ptr.free();
    }

    /**
     * @brief Insert an object into the bvh tree
     * @param object_identifier The object identifier of the object to insert
     * @param bound The bound of the object to insert
     */
    void insert(ObjectIdentifierType object_identifier, BoundType bound) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_EXPECT_ARGUMENT(
            object_identifiers.size() < MaxObjectNum,
            "object_identifiers.size()={}, MaxObjectNum={}",
            object_identifiers.size(),
            MaxObjectNum
        );

        auto it = host_data->object_indices.find(object_identifier);
        if (it != host_data->object_indices.end()) {
            update(object_identifier, bound);
        } else {
            RIndex offset = object_identifiers.size();
            resize(object_identifiers.size() + 1);

            object_identifiers.writeSlice(&object_identifier, 1, offset);
            leaf_bounds.writeSlice(&bound, 1, offset);

            rebuild();
        }
    }

    /**
     * @brief Remove an object from the bvh tree.
     * @param object_identifier The object identifier of the object to remove.
     */
    void erase(ObjectIdentifierType object_identifier) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        auto it = host_data->object_indices.find(object_identifier);
        if (it == host_data->object_indices.end()) {
            RS_THROW(RS_OutOfBoundsException("Object identifier does not exist in BVH"));
        }

        RIndex leaf_index = it->second;

        RSize new_size = object_identifiers.size() - 1;

        object_identifiers.erase(leaf_index);
        leaf_bounds.erase(leaf_index);

        resize(new_size);

        rebuild();
    }

    /**
     * @brief Update bounds of an object in the bvh tree.
     * @param object_identifier The object identifier of the object to update.
     * @param bound The bound of the object to update.
     */
    void update(ObjectIdentifierType object_identifier, BoundType bound) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        auto it = host_data->object_indices.find(object_identifier);
        if (it == host_data->object_indices.end()) {
            RS_THROW(RS_OutOfBoundsException("Object identifier does not exist in BVH"));
        }

        RIndex leaf_index = it->second;
        leaf_bounds.writeSlice(&bound, 1, leaf_index);

        rebuild();
    }

    /**
     * @brief Find the bounds of an object in the bvh tree.
     * @param object_identifier The object identifier of the object to find.
     * @return The bound of the object.
     */
    BoundType find(ObjectIdentifierType object_identifier) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        auto it = host_data->object_indices.find(object_identifier);
        if (it == host_data->object_indices.end()) {
            return BoundType::empty();
        }
        BoundType bound;
        leaf_bounds.readSlice(&bound, 1, it->second);
        return bound;
    }

    /**
     * @brief Check if an object is in the bvh tree.
     * @param object_identifier The object identifier of the object to check.
     * @return True if the object is in the bvh tree, false otherwise.
     */
    bool contains(ObjectIdentifierType object_identifier) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        auto it = host_data->object_indices.find(object_identifier);
        return it != host_data->object_indices.end();
    }

    /// @brief Get the root node of the bvh tree.
    HOST_DEVICE_INLINE_QUALIFIER RS_BVHAccessor root() {
        // If bvh is empty, initialize with null index
        static constexpr RIndex FirstLeafNode = sizeof(RIndex) == 4 ? 0x80000000
                                                                    : 0x8000000000000000;

        return {
            object_identifiers.ptr(),
            internal_nodes.ptr(),
            leaf_bounds.ptr(),
            object_identifiers.size() > 1
                ? 0
                : (object_identifiers.size() == 1 ? FirstLeafNode : RS_NULL_INDEX)
        };
    }
};

using RS_BVH = RS_BVHBase<>;

}  // namespace device

}  // namespace rise

#endif  // RS_BVH_H
