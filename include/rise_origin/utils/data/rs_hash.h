// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_HASH_H
#define RS_HASH_H

#include <functional>

#include "rise_origin/utils/data/rs_nary.h"

namespace rise {

struct RS_UnorderedPairHash {
    template <typename T>
    std::size_t operator()(const RS_UnorderedPair<T>& pair) const {
        return std::hash<T>()(pair.value1) ^ std::hash<T>()(pair.value2);
    }
};

struct RS_OrderedPairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const RS_OrderedPair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.value1) ^ std::hash<T2>()(pair.value2);
    }
};

struct RS_UnorderedTripleHash {
    template <typename T>
    std::size_t operator()(const RS_UnorderedTriple<T>& triple) const {
        return std::hash<T>()(triple.value1) ^ std::hash<T>()(triple.value2) ^
               std::hash<T>()(triple.value3);
    }
};

template <typename T1, typename T2, typename T3>
struct RS_OrderedTripleHash {
    std::size_t operator()(const RS_OrderedTriple<T1, T2, T3>& triple) const {
        return std::hash<T1>()(triple.value1) ^ std::hash<T2>()(triple.value2) ^
               std::hash<T3>()(triple.value3);
    }
};

struct RS_UnorderedQuadrupleHash {
    template <typename T>
    std::size_t operator()(const RS_UnorderedQuadruple<T>& quad) const {
        return std::hash<T>()(quad.value1) ^ std::hash<T>()(quad.value2) ^
               std::hash<T>()(quad.value3) ^ std::hash<T>()(quad.value4);
    }
};

struct RS_OrderedQuadrupleHash {
    template <typename T1, typename T2, typename T3, typename T4>
    std::size_t operator()(const RS_OrderedQuadruple<T1, T2, T3, T4>& quad) const {
        return std::hash<T1>()(quad.value1) ^ (std::hash<T2>()(quad.value2) << 1) ^
               (std::hash<T3>()(quad.value3) << 2) ^ (std::hash<T4>()(quad.value4) << 3);
    }
};

}  // namespace rise

#endif  // RS_HASH_H
