// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_UTIL_H
#define RS_RSC_UTIL_H

#include <array>
#include <cstdint>
#include <highfive/highfive.hpp>
#include <type_traits>

#include "rise_origin/rsc/rs_config_property_tree.h"
#include "rise_origin/utils/rs_bound.h"
#include "rise_origin/utils/rs_color.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_quat.h"
#include "rise_origin/utils/rs_string.h"
#include "rise_origin/utils/rs_vec.h"
namespace rise {

namespace detail {

enum CharClass : uint8_t { DELIM = 1, DIGIT = 2, SIGN = 4 };

inline constexpr auto CHAR_LUT = [] {
    std::array<uint8_t, 256> t{};
    t[','] = DELIM;
    t[' '] = DELIM;
    t['\t'] = DELIM;
    t['+'] = SIGN;
    t['-'] = SIGN;
    for (int c = '0'; c <= '9'; ++c) t[c] = DIGIT;
    return t;
}();

}  // namespace detail

inline RSE_FloatType readFloatType(const std::string &s, const char *field) {
    if (s == "Float32") return RSE_FLOAT_32;
    if (s == "Float64") return RSE_FLOAT_64;
    RS_THROW(RS_InvalidConfigurationException(
        "Invalid RS_SimulationConfig: {} must be 'Float32' or 'Float64', got '{}'", field, s
    ));
}

inline RColor3b readColor3c(const std::string &str) {
    auto split_values = split(str, ",");

    if (split_values.size() == 3) {
        try {
            return {
                static_cast<unsigned char>(std::stoi(split_values[0])),
                static_cast<unsigned char>(std::stoi(split_values[1])),
                static_cast<unsigned char>(std::stoi(split_values[2]))
            };
        } catch ([[maybe_unused]] const std::invalid_argument &e) {
            RS_THROW(
                RS_InvalidConfigurationException("Invalid input {} for RColor3b data type", str)
            );
        }
    }
    RS_THROW(RS_InvalidConfigurationException("Invalid input {} for RColor3b data type", str));
}

inline RVec3rf readVec3rf(const std::string &str) {
    auto split_values = split(str, ",");

    if (split_values.size() == 3) {
        try {
            return {
                static_cast<RFloat>(std::stod(split_values[0])),
                static_cast<RFloat>(std::stod(split_values[1])),
                static_cast<RFloat>(std::stod(split_values[2]))
            };
        } catch ([[maybe_unused]] const std::invalid_argument &e) {
            RS_THROW(
                RS_InvalidConfigurationException("Invalid input {} for RVec3rf data type", str)
            );
        }
    }
    RS_THROW(RS_InvalidConfigurationException("Invalid input {} for RVec3rf data type", str));
}

inline RQuat3rf readQuat3rf(const std::string &str) {
    auto split_values = split(str, ",");

    if (split_values.size() == 4) {
        try {
            return {
                static_cast<RFloat>(std::stod(split_values[0])),
                static_cast<RFloat>(std::stod(split_values[1])),
                static_cast<RFloat>(std::stod(split_values[2])),
                static_cast<RFloat>(std::stod(split_values[3]))
            };
        } catch ([[maybe_unused]] const std::invalid_argument &e) {
            RS_THROW(
                RS_InvalidConfigurationException("Invalid input {} for RQuat3rf data type", str)
            );
        }
    }
    RS_THROW(RS_InvalidConfigurationException("Invalid input {} for RQuat3rf data type", str));
}

inline RBound3ri readBound3ri(const std::string &str) {
    auto split_values = split(str, ",");

    if (split_values.size() == 6) {
        try {
            return {
                static_cast<RSignedIndex>(std::stoi(split_values[0])),
                static_cast<RSignedIndex>(std::stoi(split_values[1])),
                static_cast<RSignedIndex>(std::stoi(split_values[2])),
                static_cast<RSignedIndex>(std::stoi(split_values[3])),
                static_cast<RSignedIndex>(std::stoi(split_values[4])),
                static_cast<RSignedIndex>(std::stoi(split_values[5]))
            };
        } catch ([[maybe_unused]] const std::invalid_argument &e) {
            RS_THROW(
                RS_InvalidConfigurationException("Invalid input {} for RBound3ri data type", str)
            );
        }
    }
    RS_THROW(RS_InvalidConfigurationException("Invalid input {} for RBound3ri data type", str));
}

inline void readIndexLayersFromXML(
    const RS_ConfigPropertyTree &body_tree,
    const std::string &section,
    size_t voxel_num,
    size_t voxel_per_layer_num,
    std::vector<RIndex> &layer_data,
    bool required = true
) {
    layer_data.clear();
    if (!body_tree.isChildExist(section)) {
        if (required)
            RS_THROW(RS_InvalidConfigurationException("Missing required section {:s}", section));
        else
            return;
    }

    // Access the boost ptree children directly to avoid RS_ConfigPropertyTree
    // copies (each copy duplicates the entire root tree).
    const auto &section_node = body_tree.current_node->get_child(section);
    std::vector<std::string_view> layer_strings;
    for (const auto &child : section_node) {
        if (child.first[0] == '<') continue;
        layer_strings.emplace_back(child.second.data());
    }

    layer_data.resize(voxel_num);
    size_t layer_num = layer_strings.size();

    for (size_t l = 0; l < layer_num; ++l) {
        const char *ptr = layer_strings[l].data();
        const char *end = ptr + layer_strings[l].size();
        size_t count = 0;
        size_t base = l * voxel_per_layer_num;
        while (ptr < end && count < voxel_per_layer_num) {
            // Skip delimiters
            while (ptr < end && (detail::CHAR_LUT[static_cast<uint8_t>(*ptr)] & detail::DELIM))
                ++ptr;
            if (ptr >= end) break;
            // Parse sign
            bool neg = false;
            if (detail::CHAR_LUT[static_cast<uint8_t>(*ptr)] & detail::SIGN) {
                neg = (*ptr == '-');
                ++ptr;
            }
            // Parse digits
            if (ptr >= end || !(detail::CHAR_LUT[static_cast<uint8_t>(*ptr)] & detail::DIGIT))
                break;
            unsigned long long uval = 0;
            do {
                uval = uval * 10 + (*ptr - '0');
                ++ptr;
            } while (ptr < end && (detail::CHAR_LUT[static_cast<uint8_t>(*ptr)] & detail::DIGIT));
            long long val = neg ? -static_cast<long long>(uval) : static_cast<long long>(uval);
            layer_data[base + count] = val < 0 ? static_cast<RIndex>(RS_NULL_INDEX)
                                               : static_cast<RIndex>(val);
            ++count;
        }
        if (count != voxel_per_layer_num)
            RS_THROW(RS_InvalidConfigurationException(
                "{} layer {} size is {}, required {}", section, l, count, voxel_per_layer_num
            ));
    }
}

inline void readIndexLayersFromHDF5(
    const HighFive::File &hdf5_file,
    const RS_ConfigPropertyTree &body_tree,
    const std::string &section,
    size_t voxel_num,
    std::vector<RIndex> &layer_data,
    bool required
) {
    layer_data.clear();
    if (!body_tree.isChildExist(section)) {
        if (required)
            RS_THROW(RS_InvalidConfigurationException("Missing required section {:s}", section));
        else
            return;
    }
    HighFive::DataSetAccessProps access_props;
    access_props.add(HighFive::Caching(8192, 131072));
    auto dataset = hdf5_file.getDataSet(body_tree.get<std::string>(section), access_props);
    std::vector<long long> tmp_data;
    dataset.read(tmp_data);
    layer_data.resize(tmp_data.size());
    for (size_t i = 0; i < tmp_data.size(); ++i) {
        layer_data[i] = tmp_data[i] < 0 ? static_cast<RIndex>(RS_NULL_INDEX)
                                        : static_cast<RIndex>(tmp_data[i]);
    }
    if (layer_data.size() != voxel_num)
        RS_THROW(RS_InvalidConfigurationException(
            "HDF5 {} size is {}, required {}", section, layer_data.size(), voxel_num
        ));
}

template <typename Enum>
inline void readEnumLayersFromXML(
    const RS_ConfigPropertyTree &body_tree,
    const std::string &section,
    size_t voxel_num,
    size_t voxel_per_layer_num,
    std::vector<Enum> &layer_data,
    bool required = true
) {
    static_assert(std::is_enum_v<Enum>, "readEnumLayersFromXML requires an enum element type");
    layer_data.clear();
    if (!body_tree.isChildExist(section)) {
        if (required)
            RS_THROW(RS_InvalidConfigurationException("Missing required section {:s}", section));
        else
            return;
    }

    const auto &section_node = body_tree.current_node->get_child(section);
    std::vector<std::string_view> layer_strings;
    for (const auto &child : section_node) {
        if (child.first[0] == '<') continue;
        layer_strings.emplace_back(child.second.data());
    }

    layer_data.resize(voxel_num);
    size_t layer_num = layer_strings.size();
    for (size_t l = 0; l < layer_num; ++l) {
        const char *ptr = layer_strings[l].data();
        const char *end = ptr + layer_strings[l].size();
        size_t count = 0;
        size_t base = l * voxel_per_layer_num;
        while (ptr < end && count < voxel_per_layer_num) {
            while (ptr < end && (detail::CHAR_LUT[static_cast<uint8_t>(*ptr)] & detail::DELIM))
                ++ptr;
            if (ptr >= end) break;
            bool neg = false;
            if (detail::CHAR_LUT[static_cast<uint8_t>(*ptr)] & detail::SIGN) {
                neg = (*ptr == '-');
                ++ptr;
            }
            if (ptr >= end || !(detail::CHAR_LUT[static_cast<uint8_t>(*ptr)] & detail::DIGIT))
                break;
            unsigned long long uval = 0;
            do {
                uval = uval * 10 + (*ptr - '0');
                ++ptr;
            } while (ptr < end && (detail::CHAR_LUT[static_cast<uint8_t>(*ptr)] & detail::DIGIT));
            int64_t val = neg ? -static_cast<int64_t>(uval) : static_cast<int64_t>(uval);
            Enum candidate = static_cast<Enum>(val);
            if (!isValidEnum(candidate)) {
                RS_THROW(RS_InvalidConfigurationException(
                    "{} layer {} contains invalid enum value {}", section, l, val
                ));
            }
            layer_data[base + count] = candidate;
            ++count;
        }
        if (count != voxel_per_layer_num)
            RS_THROW(RS_InvalidConfigurationException(
                "{} layer {} size is {}, required {}", section, l, count, voxel_per_layer_num
            ));
    }
}

template <typename Enum>
inline void readEnumLayersFromHDF5(
    const HighFive::File &hdf5_file,
    const RS_ConfigPropertyTree &body_tree,
    const std::string &section,
    size_t voxel_num,
    std::vector<Enum> &layer_data,
    bool required
) {
    static_assert(std::is_enum_v<Enum>, "readEnumLayersFromHDF5 requires an enum element type");
    layer_data.clear();
    if (!body_tree.isChildExist(section)) {
        if (required)
            RS_THROW(RS_InvalidConfigurationException("Missing required section {:s}", section));
        else
            return;
    }
    HighFive::DataSetAccessProps access_props;
    access_props.add(HighFive::Caching(8192, 131072));
    auto dataset = hdf5_file.getDataSet(body_tree.get<std::string>(section), access_props);
    std::vector<int64_t> tmp_data;
    dataset.read(tmp_data);
    if (tmp_data.size() != voxel_num)
        RS_THROW(RS_InvalidConfigurationException(
            "HDF5 {} size is {}, required {}", section, tmp_data.size(), voxel_num
        ));
    layer_data.resize(tmp_data.size());
    for (size_t i = 0; i < tmp_data.size(); ++i) {
        Enum candidate = static_cast<Enum>(tmp_data[i]);
        if (!isValidEnum(candidate)) {
            RS_THROW(RS_InvalidConfigurationException(
                "HDF5 {} index {} contains invalid enum value {}", section, i, tmp_data[i]
            ));
        }
        layer_data[i] = candidate;
    }
}

}  // namespace rise

#endif  // RS_RSC_UTIL_H
