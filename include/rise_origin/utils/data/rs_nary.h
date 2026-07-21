// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_NARY_H
#define RS_NARY_H

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

    template <class Serializer>
    void serialize(Serializer& s) {
        s("value1", value1);
        s("value2", value2);
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

    template <class Serializer>
    void serialize(Serializer& s) {
        s("value1", value1);
        s("value2", value2);
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

    template <class Serializer>
    void serialize(Serializer& s) {
        s("value1", value1);
        s("value2", value2);
        s("value3", value3);
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

    template <class Serializer>
    void serialize(Serializer& s) {
        s("value1", value1);
        s("value2", value2);
        s("value3", value3);
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

    template <class Serializer>
    void serialize(Serializer& s) {
        s("value1", value1);
        s("value2", value2);
        s("value3", value3);
        s("value4", value4);
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

    template <class Serializer>
    void serialize(Serializer& s) {
        s("value1", value1);
        s("value2", value2);
        s("value3", value3);
        s("value4", value4);
    }
};

}  // namespace rise

#endif  // RS_NARY_H
