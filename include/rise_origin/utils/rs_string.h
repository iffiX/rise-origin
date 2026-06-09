// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_STRING_H
#define RS_STRING_H

#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace rise {

/// @brief Slice string from start to end. Returns a std::string.
inline std::string slice(const std::string_view& str, RIndex start, RIndex end = RS_NULL_INDEX) {
    if (end == RS_NULL_INDEX) {
        end = str.length();
    }
    return std::string(str.data() + start, end - start);
}

/// @brief Slice string from start to end. Returns a std::string_view.
inline std::string_view
sliceToView(const std::string_view& str, RIndex start, RIndex end = RS_NULL_INDEX) {
    if (end == RS_NULL_INDEX) {
        end = str.length();
    }
    return str.substr(start, end - start);
}

/// @brief Capitalize first letter of string, returns a new string.
inline std::string capitalize(const std::string_view str) {
    if (str.empty()) {
        return {};
    }

    std::string new_str(str);
    new_str[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(new_str[0])));
    return new_str;
}

/// @brief Returns new string with all characters converted to lowercase.
inline std::string lower(const std::string_view str) {
    std::string new_str(str);
    std::transform(new_str.begin(), new_str.end(), new_str.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return new_str;
}

/// @brief Returns new string with all characters converted to uppercase.
inline std::string upper(const std::string_view str) {
    std::string new_str(str);
    std::transform(new_str.begin(), new_str.end(), new_str.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return new_str;
}

/// @brief Replace all occurrences of old_str with new_str.
inline std::string replace(
    const std::string_view& str,
    const std::string_view& old_str,
    const std::string_view& new_str
) {
    std::string result(str);
    size_t pos = 0;
    while ((pos = result.find(old_str, pos)) != std::string::npos) {
        result.replace(pos, old_str.length(), new_str);
        pos += new_str.length();
    }
    return result;
}

/// @brief Strip characters specified in strip_chars from beginning and end of string. If
/// strip_chars is empty, strip whitespace by default.
inline std::string strip(const std::string_view& str, const std::string_view& strip_chars = "") {
    std::array<bool, 256> char_mask = {false};

    if (strip_chars.empty()) {
        for (char c : {' ', '\t', '\n', '\v', '\f', '\r'}) {
            char_mask[static_cast<unsigned char>(c)] = true;
        }
    } else {
        for (char c : strip_chars) {
            char_mask[static_cast<unsigned char>(c)] = true;
        }
    }

    auto start = std::find_if_not(str.begin(), str.end(), [&char_mask](unsigned char c) {
        return char_mask[c];
    });

    auto end = std::find_if_not(str.rbegin(), str.rend(), [&char_mask](unsigned char c) {
                   return char_mask[c];
               }).base();

    return (start < end) ? std::string(start, end) : "";
}

/// @brief Join strings with separator.
template <typename String>
std::string join(const std::string_view sep, const std::vector<String>& strs) {
    std::string result;
    for (const auto& str : strs) {
        if (!result.empty()) {
            result += sep;
        }
        result += str;
    }
    return result;
}

/// @brief Split string by delimiter. Returns vector of strings.
inline std::vector<std::string> split(const std::string_view& str, const std::string_view& delim) {
    std::vector<std::string> result;
    std::string_view::size_type pos = 0, next_pos = 0;
    while ((next_pos = str.find(delim, next_pos)) != std::string_view::npos) {
        result.emplace_back(str.substr(pos, next_pos - pos));
        pos = next_pos = next_pos + delim.length();
    }
    result.emplace_back(str.substr(pos, next_pos));
    return result;
}

/// @brief Split string by delimiter. Returns vector of string views. Need to make sure input str
/// persists.
inline std::vector<std::string_view>
splitToView(const std::string_view& str, const std::string_view& delim) {
    std::vector<std::string_view> result;
    std::string_view::size_type pos = 0, next_pos = 0;
    while ((next_pos = str.find(delim, next_pos)) != std::string_view::npos) {
        result.push_back(str.substr(pos, next_pos - pos));
        pos = next_pos = next_pos + delim.length();
    }
    result.push_back(str.substr(pos, next_pos));
    return result;
}

/// @brief Count number of substrings in string.
inline RSize count(const std::string_view str, const std::string_view substr) {
    RSize count = 0;
    std::string_view::size_type pos = 0;

    while ((pos = str.find(substr, pos)) != std::string_view::npos) {
        ++count;
        pos += substr.length();
    }

    return count;
}

/// @brief Check if string ends with suffix.
inline bool endsWith(const std::string_view& str, const std::string_view& suffix) {
    // str is shorter than suffix; cannot end with suffix
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

/// @brief Check if string starts with prefix.
inline bool startsWith(const std::string_view& str, const std::string_view& prefix) {
    // str is shorter than prefix; cannot start with prefix
    if (str.length() < prefix.length()) {
        return false;
    }
    return str.compare(0, prefix.length(), prefix) == 0;
}

/// @brief Find the first occurrence of substring in string, if not found returns RS_NULL_INDEX.
inline RIndex find(const std::string_view& str, const std::string_view& substr) {
    auto pos = str.find(substr);
    return pos == std::string_view::npos ? RS_NULL_INDEX : pos;
}

/// @brief Check if string is not empty and contains only alphanumeric.
inline bool isAlnum(const std::string_view& str) {
    return !str.empty() &&
           std::all_of(str.begin(), str.end(), [](char c) { return std::isalnum(c); });
}

/// @brief Check if string is not empty and contains only alpha.
inline bool isAlpha(const std::string_view& str) {
    return !str.empty() &&
           std::all_of(str.begin(), str.end(), [](char c) { return std::isalpha(c); });
}

/// @brief Check if string is not empty and contains only digits.
inline bool isDigit(const std::string_view& str) {
    return !str.empty() &&
           std::all_of(str.begin(), str.end(), [](char c) { return std::isdigit(c); });
}

/// @brief Check if string is not empty and contains only lowercase.
inline bool isLower(const std::string_view& str) {
    return !str.empty() &&
           std::all_of(str.begin(), str.end(), [](char c) { return std::islower(c); });
}

/// @brief Check if string is not empty and contains only uppercase.
inline bool isUpper(const std::string_view& str) {
    return !str.empty() &&
           std::all_of(str.begin(), str.end(), [](char c) { return std::isupper(c); });
}

/// @brief Check whether string is a valid float.
inline bool isFloat(const std::string_view& str) {
    static const std::regex float_regex(R"(^[-+]?(\d*\.\d+|\d+\.\d*)([eE][-+]?\d+)?$)");
    return std::regex_match(str.begin(), str.end(), float_regex);
}

/// @brief Check whether string is a valid integer.
inline bool isInteger(const std::string_view& str) {
    static const std::regex int_regex(R"(^[-+]?\d+$)");
    return std::regex_match(str.begin(), str.end(), int_regex);
}

/// @brief Check whether string is a valid number.
inline bool isNumber(const std::string_view& str) { return isInteger(str) || isFloat(str); }

}  // namespace rise

#endif  // RS_STRING_H
