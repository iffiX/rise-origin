// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_TUPLE_HASH_H
#define RS_TUPLE_HASH_H

#include <functional>

namespace rise {

template <typename T>
struct RS_UnorderedPair {
    T value1{}, value2{};
    RS_UnorderedPair() = default;
    RS_UnorderedPair(T value1, T value2) : value1(value1), value2(value2) {}
    bool operator==(const RS_UnorderedPair& other) const {
        return (value1 == other.value1 && value2 == other.value2) ||
               (value2 == other.value1 && value1 == other.value2);
    }
};

struct RS_UnorderedPairHash {
    template <typename T>
    std::size_t operator()(const RS_UnorderedPair<T>& pair) const {
        return std::hash<T>()(pair.value1) ^ std::hash<T>()(pair.value2);
    }
};

template <typename T1, typename T2>
struct RS_OrderedPair {
    T1 value1{};
    T2 value2{};
    RS_OrderedPair() = default;
    RS_OrderedPair(T1 value1, T2 value2) : value1(value1), value2(value2) {}
    bool operator==(const RS_OrderedPair& other) const {
        return value1 == other.value1 && value2 == other.value2;
    }
    bool operator<(const RS_OrderedPair& other) const {
        return value1 < other.value1 || (value1 == other.value1 && value2 < other.value2);
    }
};

struct RS_OrderedPairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const RS_OrderedPair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.value1) ^ std::hash<T2>()(pair.value2);
    }
};

template <typename T>
struct RS_UnorderedTriple {
    T value1{}, value2{}, value3{};
    RS_UnorderedTriple() = default;
    RS_UnorderedTriple(T value1, T value2, T value3)
        : value1(value1), value2(value2), value3(value3) {}
    bool operator==(const RS_UnorderedTriple& other) const {
        bool used[3] = {false, false, false};
        const T* this_values = reinterpret_cast<const T*>(this);
        const T* other_values = reinterpret_cast<const T*>(&other);

        for (int i = 0; i < 3; ++i) {
            bool found = false;
            for (int j = 0; j < 3; ++j) {
                if (!used[j] && this_values[i] == other_values[j]) {
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
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
struct RS_OrderedTriple {
    T1 value1{};
    T2 value2{};
    T3 value3{};
    RS_OrderedTriple() = default;
    RS_OrderedTriple(T1 value1, T2 value2, T3 value3)
        : value1(value1), value2(value2), value3(value3) {}
    bool operator==(const RS_OrderedTriple& other) const {
        return value1 == other.value1 && value2 == other.value2 && value3 == other.value3;
    }
    bool operator<(const RS_OrderedTriple& other) const {
        return value1 < other.value1 ||
               (value1 == other.value1 &&
                (value2 < other.value2 || (value2 == other.value2 && value3 < other.value3)));
    }
};

template <typename T1, typename T2, typename T3>
struct RS_OrderedTripleHash {
    std::size_t operator()(const RS_OrderedTriple<T1, T2, T3>& triple) const {
        return std::hash<T1>()(triple.value1) ^ std::hash<T2>()(triple.value2) ^
               std::hash<T3>()(triple.value3);
    }
};

template <typename T>
struct RS_UnorderedQuadruple {
    T value1{}, value2{}, value3{}, value4{};
    RS_UnorderedQuadruple() = default;
    RS_UnorderedQuadruple(T v1, T v2, T v3, T v4)
        : value1(v1), value2(v2), value3(v3), value4(v4) {}
    bool operator==(const RS_UnorderedQuadruple& other) const {
        bool used[4] = {false, false, false, false};
        const T* this_values = reinterpret_cast<const T*>(this);
        const T* other_values = reinterpret_cast<const T*>(&other);

        for (int i = 0; i < 4; ++i) {
            bool found = false;
            for (int j = 0; j < 4; ++j) {
                if (!used[j] && this_values[i] == other_values[j]) {
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }
};

struct RS_UnorderedQuadrupleHash {
    template <typename T>
    std::size_t operator()(const RS_UnorderedQuadruple<T>& quad) const {
        return std::hash<T>()(quad.value1) ^ std::hash<T>()(quad.value2) ^
               std::hash<T>()(quad.value3) ^ std::hash<T>()(quad.value4);
    }
};

template <typename T1, typename T2, typename T3, typename T4>
struct RS_OrderedQuadruple {
    T1 value1{};
    T2 value2{};
    T3 value3{};
    T4 value4{};
    RS_OrderedQuadruple() = default;
    RS_OrderedQuadruple(T1 v1, T2 v2, T3 v3, T4 v4)
        : value1(v1), value2(v2), value3(v3), value4(v4) {}
    bool operator==(const RS_OrderedQuadruple& other) const {
        return value1 == other.value1 && value2 == other.value2 && value3 == other.value3 &&
               value4 == other.value4;
    }
    bool operator<(const RS_OrderedQuadruple& other) const {
        return value1 < other.value1 ||
               (value1 == other.value1 &&
                (value2 < other.value2 ||
                 (value2 == other.value2 &&
                  (value3 < other.value3 || (value3 == other.value3 && value4 < other.value4)))));
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

#endif  // RS_TUPLE_HASH_H