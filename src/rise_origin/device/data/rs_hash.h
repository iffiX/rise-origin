// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_HASH_H
#define RS_HASH_H

#include "rise_origin/utils/meta/rs_num_traits.h"

namespace rise {

namespace device {

// For source of below hash functions, refer to
// https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html

HOST_DEVICE_INLINE_QUALIFIER uint32_t hash0(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint64_t hash0(uint64_t key) {
    key ^= (key >> 33);
    key *= 0xff51afd7ed558ccd;
    key ^= (key >> 33);
    key *= 0xc4ceb9fe1a85ec53;
    key ^= (key >> 33);
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint64_t hash1(uint64_t key) {
    key ^= (key >> 31);
    key *= 0x7fb5d329728ea185;
    key ^= (key >> 27);
    key *= 0x81dadef4bc2dd44d;
    key ^= (key >> 33);
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint32_t hash1(uint32_t key) {
    key ^= key >> 15;
    key *= 0xcc9e2d51;
    key ^= key >> 13;
    key *= 0x1b873593;
    key ^= key >> 16;
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint32_t hash2(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca77;
    key ^= key >> 13;
    key *= 0xc2b2ae3d;
    key ^= key >> 16;
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint64_t hash2(uint64_t key) {
    key ^= (key >> 33);
    key *= 0x64dd81482cbd31d7;
    key ^= (key >> 31);
    key *= 0xe36aa5c613612997;
    key ^= (key >> 31);
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint32_t hash3(uint32_t key) {
    key ^= key >> 17;
    key *= 0xed5ad4bb;
    key ^= key >> 11;
    key *= 0xac4c1b51;
    key ^= key >> 15;
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint64_t hash3(uint64_t key) {
    key ^= (key >> 31);
    key *= 0x99bcf6822b23ca35;
    key ^= (key >> 30);
    key *= 0x14020a57acced8b7;
    key ^= (key >> 33);
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint32_t hash4(uint32_t key) {
    key ^= key >> 15;
    key *= 0x2c1b3c6d;
    key ^= key >> 12;
    key *= 0x297a2d39;
    key ^= key >> 15;
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint64_t hash4(uint64_t key) {
    key ^= (key >> 33);
    key *= 0x62a9d9ed799705f5;
    key ^= (key >> 28);
    key *= 0xcb24d0a5c88c35b3;
    key ^= (key >> 32);
    return key;
}

HOST_DEVICE_INLINE_QUALIFIER uint32_t compress64To32(uint64_t key) {
    return (uint32_t)(key >> 32) ^ (uint32_t)(key & 0xFFFFFFFF);
}

HOST_DEVICE_INLINE_QUALIFIER uint32_t reduce(uint32_t x, uint32_t N) {
    return ((uint64_t)x * (uint64_t)N) >> 32;
}

template <typename InputType, typename OutputType, RIndex HashType>
struct RS_GenericHasher {
    static_assert(std::is_integral_v<InputType>, "InputType must be an integral type.");
    static_assert(std::is_integral_v<OutputType>, "OutputType must be an integral type.");
    static_assert(
        HashType == 0 || HashType == 1 || HashType == 2 || HashType == 3 || HashType == 4,
        "HashType must be 0, 1, 2, 3, or 4."
    );

    HOST_DEVICE_INLINE_QUALIFIER static bool isValidKey(InputType key) {
        return key < RS_NumTraits<InputType>::highest() - 1;
    }

    HOST_DEVICE_INLINE_QUALIFIER static bool isNullKey(InputType key) { return key == 0; }

    HOST_DEVICE_INLINE_QUALIFIER static void setNullKey(InputType &key) { key = 0; }

    HOST_DEVICE_INLINE_QUALIFIER OutputType operator()(InputType key) const {
        OutputType result;
        if constexpr (sizeof(InputType) == sizeof(uint64_t) &&
                      sizeof(OutputType) == sizeof(uint32_t)) {
            if constexpr (HashType == 0) {
                result = compress64To32(hash0(key));
            } else if constexpr (HashType == 1) {
                result = compress64To32(hash1(key));
            } else if constexpr (HashType == 2) {
                result = compress64To32(hash2(key));
            } else if constexpr (HashType == 3) {
                result = compress64To32(hash3(key));
            } else if constexpr (HashType == 4) {
                result = compress64To32(hash4(key));
            }
        } else {
            if constexpr (HashType == 0) {
                result = hash0(key);
            } else if constexpr (HashType == 1) {
                result = hash1(key);
            } else if constexpr (HashType == 2) {
                result = hash2(key);
            } else if constexpr (HashType == 3) {
                result = hash3(key);
            } else if constexpr (HashType == 4) {
                result = hash4(key);
            }
        }
        return result;
    }
};

}  // namespace device

}  // namespace rise

#endif