// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_UTIL_H
#define RS_RSC_UTIL_H

#include <cstdint>
#include <string>
#include <vector>

#include "rise_origin/rsc/rs_rsc_file_tree_reader.h"
#include "rise_origin/rsc/rs_rsc_file_tree_writer.h"
#include "rise_origin/utils/math/rs_bound.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_mat.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"

namespace rise {

namespace rsc {

/**
 * @brief Read an enum from its name string under key.
 */
template <typename EnumT>
inline EnumT readEnum(const RS_RSCFileTreeReader &tree, const std::string &key) {
    EnumT value{};
    std::string str = tree.get<std::string>(key);
    try {
        stringToEnum(str, value);
    } catch (const std::invalid_argument &e) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Invalid enum value for '{}': '{}' ({})", key, str, e.what()
        ));
    }
    return value;
}

/**
 * @brief Read an enum from its name string under key, or fallback when absent.
 */
template <typename EnumT>
inline EnumT readEnum(const RS_RSCFileTreeReader &tree, const std::string &key, EnumT fallback) {
    return tree.hasChild(key) ? readEnum<EnumT>(tree, key) : fallback;
}

/**
 * @brief Write an enum as its name string under key.
 */
template <typename EnumT>
inline void writeEnum(RS_RSCFileTreeWriter &tree, const std::string &key, EnumT value) {
    tree.add(key, std::string(enumToString(value)));
}

/**
 * @brief Read a byte color from a [r, g, b] array under key.
 */
inline RColor3b readColor3b(const RS_RSCFileTreeReader &tree, const std::string &key) {
    auto a = tree.get<std::vector<int>>(key);
    if (a.size() != 3) {
        RS_THROW(RS_InvalidCharacterizationException("{} must hold 3 values for RColor3b", key));
    }
    return {
        static_cast<unsigned char>(a[0]),
        static_cast<unsigned char>(a[1]),
        static_cast<unsigned char>(a[2])
    };
}

/**
 * @brief Write a byte color as a [r, g, b] array under key.
 */
inline void writeColor3b(RS_RSCFileTreeWriter &tree, const std::string &key, const RColor3b &c) {
    tree.add(key, std::vector<int>{c.r, c.g, c.b});
}

/**
 * @brief Read a float color from a [r, g, b] array under key.
 */
inline RColor3f readColor3f(const RS_RSCFileTreeReader &tree, const std::string &key) {
    auto a = tree.get<std::vector<float>>(key);
    if (a.size() != 3) {
        RS_THROW(RS_InvalidCharacterizationException("{} must hold 3 values for RColor3f", key));
    }
    return {a[0], a[1], a[2]};
}

/**
 * @brief Read a float color from a [r, g, b] array under key, or fallback when absent.
 */
inline RColor3f
readColor3f(const RS_RSCFileTreeReader &tree, const std::string &key, const RColor3f &fallback) {
    return tree.hasChild(key) ? readColor3f(tree, key) : fallback;
}

/**
 * @brief Write a float color as a [r, g, b] array under key.
 */
inline void writeColor3f(RS_RSCFileTreeWriter &tree, const std::string &key, const RColor3f &c) {
    tree.add(key, std::vector<float>{c.r, c.g, c.b});
}

/**
 * @brief Read a vector from an [x, y, z] array under key.
 */
inline RVec3rf readVec3rf(const RS_RSCFileTreeReader &tree, const std::string &key) {
    auto a = tree.get<std::vector<RFloat>>(key);
    if (a.size() != 3) {
        RS_THROW(RS_InvalidCharacterizationException("{} must hold 3 values for RVec3rf", key));
    }
    return {a[0], a[1], a[2]};
}

/**
 * @brief Read a vector from an [x, y, z] array under key, or fallback when absent.
 */
inline RVec3rf
readVec3rf(const RS_RSCFileTreeReader &tree, const std::string &key, const RVec3rf &fallback) {
    return tree.hasChild(key) ? readVec3rf(tree, key) : fallback;
}

/**
 * @brief Write a vector as an [x, y, z] array under key.
 */
inline void writeVec3rf(RS_RSCFileTreeWriter &tree, const std::string &key, const RVec3rf &v) {
    tree.add(key, std::vector<RFloat>{v.x, v.y, v.z});
}

/**
 * @brief Read a quaternion from an [x, y, z, w] array under key.
 */
inline RQuat3rf readQuat3rf(const RS_RSCFileTreeReader &tree, const std::string &key) {
    auto a = tree.get<std::vector<RFloat>>(key);
    if (a.size() != 4) {
        RS_THROW(RS_InvalidCharacterizationException("{} must hold 4 values for RQuat3rf", key));
    }
    return {a[0], a[1], a[2], a[3]};
}

/**
 * @brief Read a quaternion from an [x, y, z, w] array under key, or fallback when absent.
 */
inline RQuat3rf
readQuat3rf(const RS_RSCFileTreeReader &tree, const std::string &key, const RQuat3rf &fallback) {
    return tree.hasChild(key) ? readQuat3rf(tree, key) : fallback;
}

/**
 * @brief Write a quaternion as an [x, y, z, w] array under key.
 */
inline void writeQuat3rf(RS_RSCFileTreeWriter &tree, const std::string &key, const RQuat3rf &q) {
    tree.add(key, std::vector<RFloat>{q.x, q.y, q.z, q.w});
}

/**
 * @brief Read a row major 3x3 matrix from a nine value array under key.
 */
inline RMat3rf readMat3rf(const RS_RSCFileTreeReader &tree, const std::string &key) {
    auto a = tree.get<std::vector<RFloat>>(key);
    if (a.size() != 9) {
        RS_THROW(RS_InvalidCharacterizationException("{} must hold 9 values for RMat3rf", key));
    }
    return {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]};
}

/**
 * @brief Read a row major 3x3 matrix array under key, or fallback when absent.
 */
inline RMat3rf
readMat3rf(const RS_RSCFileTreeReader &tree, const std::string &key, const RMat3rf &fallback) {
    return tree.hasChild(key) ? readMat3rf(tree, key) : fallback;
}

/**
 * @brief Write a row major 3x3 matrix as a nine value array under key.
 */
inline void writeMat3rf(RS_RSCFileTreeWriter &tree, const std::string &key, const RMat3rf &m) {
    tree.add(
        key,
        std::vector<RFloat>{
            m.data[0],
            m.data[1],
            m.data[2],
            m.data[3],
            m.data[4],
            m.data[5],
            m.data[6],
            m.data[7],
            m.data[8]
        }
    );
}

/**
 * @brief Read an inclusive index box from a [min_x, min_y, min_z, max_x, max_y, max_z] array.
 */
inline RBound3ri readBound3ri(const RS_RSCFileTreeReader &tree, const std::string &key) {
    auto a = tree.get<std::vector<RSignedIndex>>(key);
    if (a.size() != 6) {
        RS_THROW(RS_InvalidCharacterizationException("{} must hold 6 values for RBound3ri", key));
    }
    return {a[0], a[1], a[2], a[3], a[4], a[5]};
}

/**
 * @brief Read an inclusive index box array under key, or fallback when absent.
 */
inline RBound3ri
readBound3ri(const RS_RSCFileTreeReader &tree, const std::string &key, const RBound3ri &fallback) {
    return tree.hasChild(key) ? readBound3ri(tree, key) : fallback;
}

/**
 * @brief Write an inclusive index box as a [min_x, min_y, min_z, max_x, max_y, max_z] array.
 */
inline void writeBound3ri(RS_RSCFileTreeWriter &tree, const std::string &key, const RBound3ri &b) {
    tree.add(key, std::vector<RSignedIndex>{b.min_x, b.min_y, b.min_z, b.max_x, b.max_y, b.max_z});
}

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_UTIL_H
