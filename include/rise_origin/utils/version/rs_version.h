#ifndef RS_VERSION_H
#define RS_VERSION_H

#include <string>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rise_origin/utils/string/rs_format.h"
#include "rise_origin/utils/string/rs_string.h"

namespace rise {

/*****************************************************************
 * Exceptions
 *****************************************************************/
RS_MAKE_EXCEPTION(RS_InvalidVersionException, RS_DataException)

/*****************************************************************
 * Class Definitions
 *****************************************************************/
/**
 * @brief Semantic versioning class with comparison operators
 *
 * RS_Version provides semantic versioning (major.minor.patch) functionality with support
 * for version comparison operations similar to pip and conda package managers. The class
 * uses RIndex (uint32_t) for version components to ensure efficient storage and comparison.
 *
 * Version format: major.minor.patch
 * - major: Incompatible API changes
 * - minor: Backwards-compatible functionality additions
 * - patch: Backwards-compatible bug fixes
 *
 * Supported comparison operations:
 * - Equality: ==, !=
 * - Ordering: <, <=, >, >=
 * - Matching: matches() with version specifiers (==, !=, <, <=, >, >=, ~=, ^)
 *
 * Example usage:
 * @code
 *   RS_Version v1(1, 2, 3);
 *   RS_Version v2 = RS_Version::fromString("2.0.0");
 *
 *   if (v1 < v2) { ... }
 *   if (v2.matches(">=2.0.0")) { ... }
 *   if (v1.matches("~=1.2")) { ... } // compatible release
 *   if (v2.matches("^2")) { ... } // caret (compatible within major)
 * @endcode
 */
class RS_Version {
   public:
    RIndex major;  ///< Major version number (breaking changes)
    RIndex minor;  ///< Minor version number (backwards-compatible features)
    RIndex patch;  ///< Patch version number (backwards-compatible fixes)

    /**
     * @brief Default constructor - creates version 0.0.0
     */
    RS_Version() : major(0), minor(0), patch(0) {}

    /**
     * @brief Constructs version with specified components
     * @param major Major version number
     * @param minor Minor version number (default: 0)
     * @param patch Patch version number (default: 0)
     */
    RS_Version(RIndex major, RIndex minor = 0, RIndex patch = 0)
        : major(major), minor(minor), patch(patch) {}

    /**
     * @brief Equality operator
     * @param other Version to compare against
     * @return True if versions are identical
     */
    bool operator==(const RS_Version& other) const {
        return major == other.major && minor == other.minor && patch == other.patch;
    }

    /**
     * @brief Inequality operator
     * @param other Version to compare against
     * @return True if versions differ
     */
    bool operator!=(const RS_Version& other) const { return !(*this == other); }

    /**
     * @brief Less-than operator
     * @param other Version to compare against
     * @return True if this version is less than other
     */
    bool operator<(const RS_Version& other) const {
        if (major != other.major) return major < other.major;
        if (minor != other.minor) return minor < other.minor;
        return patch < other.patch;
    }

    /**
     * @brief Less-than-or-equal operator
     * @param other Version to compare against
     * @return True if this version is less than or equal to other
     */
    bool operator<=(const RS_Version& other) const { return *this < other || *this == other; }

    /**
     * @brief Greater-than operator
     * @param other Version to compare against
     * @return True if this version is greater than other
     */
    bool operator>(const RS_Version& other) const { return other < *this; }

    /**
     * @brief Greater-than-or-equal operator
     * @param other Version to compare against
     * @return True if this version is greater than or equal to other
     */
    bool operator>=(const RS_Version& other) const { return *this > other || *this == other; }

    /**
     * @brief Returns a string representation of the version
     * @return Version string in format "major.minor.patch"
     */
    std::string toString() const { return rise::format("{}.{}.{}", major, minor, patch); }

    /**
     * @brief Parses a version string into a RS_Version object
     *
     * Accepts version strings in the following formats:
     * - "major.minor.patch" (e.g., "1.2.3")
     * - "major.minor" (patch defaults to 0)
     * - "major" (minor and patch default to 0)
     *
     * @param version_str Version string to parse
     * @throw RS_InvalidVersionException if version string is malformed
     */
    void fromString(const std::string& version_str) {
        if (version_str.empty()) {
            RS_THROW(RS_InvalidVersionException("Version string cannot be empty"));
        }

        auto components = splitToView(version_str, ".");

        if (components.size() > 3) {
            RS_THROW(RS_InvalidVersionException(
                "Invalid version format '{}': too many components", version_str
            ));
        }

        for (auto&& [idx, component] : enumerate(components)) {
            if (!isInteger(component)) {
                RS_THROW(RS_InvalidVersionException(
                    "Invalid version format '{}': non-integer component", version_str
                ));
            }
            try {
                if (idx == 0) {
                    major = static_cast<RIndex>(std::stoul(std::string(component)));
                } else if (idx == 1) {
                    minor = static_cast<RIndex>(std::stoul(std::string(component)));
                } else if (idx == 2) {
                    patch = static_cast<RIndex>(std::stoul(std::string(component)));
                }
            } catch (const std::exception& e) {
                RS_THROW(RS_InvalidVersionException(
                    "Invalid version format '{}': {}", version_str, e.what()
                ));
            }
        }

        minor = components.size() > 1 ? minor : 0;
        patch = components.size() > 2 ? patch : 0;
    }

    /**
     * @brief Checks if this version matches a version specifier
     *
     * Supports the following specifier formats:
     * - "==X.Y.Z" : Exactly equal to X.Y.Z
     * - "!=X.Y.Z" : Not equal to X.Y.Z
     * - "<X.Y.Z"  : Less than X.Y.Z
     * - "<=X.Y.Z" : Less than or equal to X.Y.Z
     * - ">X.Y.Z"  : Greater than X.Y.Z
     * - ">=X.Y.Z" : Greater than or equal to X.Y.Z
     * - "~=X.Y"   : Compatible release (>= X.Y.0, < X+1.0.0)
     * - "^X"      : Caret (>= X.0.0, < X+1.0.0) - npm-style
     * - "^X.Y"    : Caret (>= X.Y.0, < X+1.0.0) - npm-style
     * - "^X.Y.Z"  : Caret (>= X.Y.Z, < X+1.0.0) - npm-style
     *
     * @param specifier Version specifier string
     * @return True if this version matches the specifier
     * @throw RS_InvalidVersionException if specifier format is invalid
     */
    bool matches(const std::string& specifier) const {
        if (specifier.empty()) {
            RS_THROW(RS_InvalidVersionException("Version specifier cannot be empty"));
        }

        auto spec = strip(specifier);

        // Compatible release operator (~=)
        if (spec.size() >= 2 && sliceToView(spec, 0, 2) == "~=") {
            auto version_str = strip(sliceToView(spec, 2));
            RS_Version target_version;
            target_version.fromString(version_str);
            RS_Version upper_bound(target_version.major + 1, 0, 0);
            return *this >= target_version && *this < upper_bound;
        }

        // Caret operator (^) - npm-style
        if (spec.size() >= 2 && spec[0] == '^') {
            auto version_str = strip(sliceToView(spec, 1));
            RS_Version target_version;
            target_version.fromString(version_str);
            RS_Version upper_bound(target_version.major + 1, 0, 0);
            return *this >= target_version && *this < upper_bound;
        }

        // Two-character operators (==, !=, <=, >=)
        if (spec.size() >= 3 &&
            (sliceToView(spec, 0, 2) == "==" || sliceToView(spec, 0, 2) == "!=" ||
             sliceToView(spec, 0, 2) == "<=" || sliceToView(spec, 0, 2) == ">=")) {
            auto op = sliceToView(spec, 0, 2);
            auto version_str = strip(sliceToView(spec, 2));
            RS_Version target_version;
            target_version.fromString(version_str);

            if (op == "==") return *this == target_version;
            if (op == "!=") return *this != target_version;
            if (op == "<=") return *this <= target_version;
            if (op == ">=") return *this >= target_version;
        }

        // Single-character operators (<, >)
        if (spec.size() >= 2 && (spec[0] == '<' || spec[0] == '>')) {
            auto op = sliceToView(spec, 0, 1);
            auto version_str = strip(sliceToView(spec, 1));
            RS_Version target_version;
            target_version.fromString(version_str);

            if (op == "<") return *this < target_version;
            if (op == ">") return *this > target_version;
        }

        // If no operator, assume exact match
        RS_Version target_version;
        target_version.fromString(spec);
        return *this == target_version;
    }

    /**
     * @brief Checks if this version is compatible with another version
     *
     * Compatibility rules:
     * - Same major version (major > 0): minor and patch can differ
     * - Major version 0: minor must match, patch can differ
     *
     * @param other Version to check compatibility with
     * @return True if versions are compatible
     */
    bool isCompatibleWith(const RS_Version& other) const {
        if (major == 0 && other.major == 0) {
            // For 0.x.y versions, minor must match
            return minor == other.minor;
        }
        // For x.y.z versions (x > 0), major must match
        return major == other.major && major > 0;
    }

    /**
     * @brief Returns the next major version (increments major, resets minor and patch to 0)
     * @return New version with incremented major number
     */
    RS_Version nextMajor() const { return RS_Version(major + 1, 0, 0); }

    /**
     * @brief Returns the next minor version (increments minor, resets patch to 0)
     * @return New version with incremented minor number
     */
    RS_Version nextMinor() const { return RS_Version(major, minor + 1, 0); }

    /**
     * @brief Returns the next patch version (increments patch)
     * @return New version with incremented patch number
     */
    RS_Version nextPatch() const { return RS_Version(major, minor, patch + 1); }

    /**
     * @brief Checks if this is a pre-release version (0.x.y)
     * @return True if major version is 0
     */
    bool isPreRelease() const { return major == 0; }

    /**
     * @brief Checks if this is a stable release version (x.y.z where x > 0)
     * @return True if major version is greater than 0
     */
    bool isStable() const { return major > 0; }

    /// Generic serialization
    template <class Serializer>
    void serialize(Serializer& s) {
        s("major", major);
        s("minor", minor);
        s("patch", patch);
    }
};

}  // namespace rise

#endif  // RS_VERSION_H
