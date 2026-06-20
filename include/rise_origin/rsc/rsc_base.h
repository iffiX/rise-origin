// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_BASE_H
#define RSC_BASE_H

#include "rise_origin/utils/rs_def.h"
#include "rs_characterization_property_tree.h"

namespace rise {

namespace rsc {

/**
 * @brief Root base class for all Rise Characterization (RSC) types.
 *
 * Every RSC class inherits (directly or transitively) from this. Provides the
 * read/validate contract driven by RS_CharacterizationPropertyTree.
 */
class RSC_Base {
   public:
    virtual ~RSC_Base() = default;

    /**
     * @brief Read the characterization from a property tree.
     * @param rsc_tree The property tree to read the characterization from.
     * @param validate Whether to validate the characterization after reading it.
     * @throw RS_InvalidCharacterizationException When the characterization is invalid.
     */
    void read(const RS_CharacterizationPropertyTree& rsc_tree, bool validate = true) {
        _read(rsc_tree);
        if (validate) {
            _validate();
        }
    }

    /**
     * @brief Validate the characterization.
     * @throw RS_InvalidCharacterizationException When the characterization is invalid.
     */
    void validate() const { _validate(); }

    bool operator==(const RSC_Base&) const { return true; }

   protected:
    virtual void _validate() const;
    virtual void _read(const RS_CharacterizationPropertyTree& rsc_tree) = 0;
};

}  // namespace rsc

}  // namespace rise

#define RSC_EXPECT_NOT_NULL(class_name, attribute, value)                           \
    if (attribute == static_cast<decltype(attribute)>(value)) {                     \
        RS_THROW(RS_InvalidCharacterizationException(                                  \
            "Invalid " #class_name ", " #attribute " must not be null index" \
        ));                                                                         \
    }

#define RSC_EXPECT_NOT_EMPTY(class_name, attribute)                            \
    if (attribute.empty()) {                                                   \
        RS_THROW(RS_InvalidCharacterizationException(                             \
            "Invalid " #class_name ", " #attribute " must not be empty" \
        ));                                                                    \
    }

#define RSC_EXPECT_NE(class_name, attribute, value)                                       \
    if (attribute == static_cast<decltype(attribute)>(value)) {                           \
        RS_THROW(RS_InvalidCharacterizationException(                                        \
            "Invalid " #class_name ", " #attribute " must not be equal to " #value \
        ));                                                                               \
    }

#define RSC_EXPECT_EQ(class_name, attribute, value)                                   \
    if (attribute != static_cast<decltype(attribute)>(value)) {                       \
        RS_THROW(RS_InvalidCharacterizationException(                                    \
            "Invalid " #class_name ", " #attribute " must be equal to " #value \
        ));                                                                           \
    }

#define RSC_EXPECT_GT(class_name, attribute, value)                                       \
    if (attribute <= static_cast<decltype(attribute)>(value)) {                           \
        RS_THROW(RS_InvalidCharacterizationException(                                        \
            "Invalid " #class_name ", " #attribute " must be greater than " #value \
        ));                                                                               \
    }

#define RSC_EXPECT_GE(class_name, attribute, value)            \
    if (attribute < static_cast<decltype(attribute)>(value)) { \
        RS_THROW(RS_InvalidCharacterizationException(             \
            "Invalid " #class_name ", " #attribute      \
            " must be greater than or equal to " #value        \
        ));                                                    \
    }

#define RSC_EXPECT_LT(class_name, attribute, value)                                    \
    if (attribute >= static_cast<decltype(attribute)>(value)) {                        \
        RS_THROW(RS_InvalidCharacterizationException(                                     \
            "Invalid " #class_name ", " #attribute " must be less than " #value \
        ));                                                                            \
    }

#define RSC_EXPECT_LE(class_name, attribute, value)                                                \
    if (attribute > static_cast<decltype(attribute)>(value)) {                                     \
        RS_THROW(RS_InvalidCharacterizationException(                                                 \
            "Invalid " #class_name ", " #attribute " must be less than or equal to " #value \
        ));                                                                                        \
    }

#define RSC_EXPECT_CLOSE(class_name, attribute, value, tolerance)                     \
    if (RS_ABS(attribute - value) > tolerance) {                                      \
        RS_THROW(RS_InvalidCharacterizationException(                                    \
            "Invalid " #class_name ", " #attribute " must be close to " #value \
            " within " #tolerance                                                     \
        ));                                                                           \
    }

#endif  // RSC_BASE_H
