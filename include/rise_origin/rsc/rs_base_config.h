// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BASE_CONFIG_H
#define RS_BASE_CONFIG_H

#include "rise_origin/utils/rs_def.h"
#include "rs_config_property_tree.h"

namespace rise {

class RS_BaseConfig {
   public:
    virtual ~RS_BaseConfig() = default;

    /**
     * Read the configuration from a property tree.
     * @param config_tree The property tree to read the configuration from.
     * @param validate Whether to validate the configuration after reading it.
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void read(const RS_ConfigPropertyTree &config_tree, bool validate = true) {
        _read(config_tree);
        if (validate) {
            _validate();
        }
    }

    /**
     * Validate the configuration.
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void validate() const { _validate(); }

   protected:
    virtual void _validate() const {};
    virtual void _read(const RS_ConfigPropertyTree &config_tree) = 0;
};

}  // namespace rise

#define RS_CONFIG_EXPECT_ANY_SET(class_name, any_member, type_label)                 \
    if ((any_member).empty()) {                                                      \
        RS_THROW(RS_InvalidConfigurationException(                                   \
            "Invalid " #class_name ": '" #any_member "' is not set, but 'type' "     \
            "requires a " type_label ". Assign a " type_label " instance to "        \
            #any_member "."                                                          \
        ));                                                                          \
    }

#define RS_CONFIG_EXPECT_NOT_NULL(class_name, attribute, value)                     \
    if (attribute == static_cast<decltype(attribute)>(value)) {                     \
        RS_THROW(RS_InvalidConfigurationException(                                  \
            "Invalid " #class_name " config, " #attribute " must not be null index" \
        ));                                                                         \
    }

#define RS_CONFIG_EXPECT_NE(class_name, attribute, value)                                 \
    if (attribute == static_cast<decltype(attribute)>(value)) {                           \
        RS_THROW(RS_InvalidConfigurationException(                                        \
            "Invalid " #class_name " config, " #attribute " must not be equal to " #value \
        ));                                                                               \
    }

#define RS_CONFIG_EXPECT_EQ(class_name, attribute, value)                             \
    if (attribute != static_cast<decltype(attribute)>(value)) {                       \
        RS_THROW(RS_InvalidConfigurationException(                                    \
            "Invalid " #class_name " config, " #attribute " must be equal to " #value \
        ));                                                                           \
    }

#define RS_CONFIG_EXPECT_GT(class_name, attribute, value)                                 \
    if (attribute <= static_cast<decltype(attribute)>(value)) {                           \
        RS_THROW(RS_InvalidConfigurationException(                                        \
            "Invalid " #class_name " config, " #attribute " must be greater than " #value \
        ));                                                                               \
    }

#define RS_CONFIG_EXPECT_GE(class_name, attribute, value)      \
    if (attribute < static_cast<decltype(attribute)>(value)) { \
        RS_THROW(RS_InvalidConfigurationException(             \
            "Invalid " #class_name " config, " #attribute      \
            " must be greater than or equal to " #value        \
        ));                                                    \
    }

#define RS_CONFIG_EXPECT_LT(class_name, attribute, value)                              \
    if (attribute >= static_cast<decltype(attribute)>(value)) {                        \
        RS_THROW(RS_InvalidConfigurationException(                                     \
            "Invalid " #class_name " config, " #attribute " must be less than " #value \
        ));                                                                            \
    }

#define RS_CONFIG_EXPECT_LE(class_name, attribute, value)                                          \
    if (attribute > static_cast<decltype(attribute)>(value)) {                                     \
        RS_THROW(RS_InvalidConfigurationException(                                                 \
            "Invalid " #class_name " config, " #attribute " must be less than or equal to " #value \
        ));                                                                                        \
    }

#define RS_CONFIG_EXPECT_CLOSE(class_name, attribute, value, tolerance)               \
    if (RS_ABS(attribute - value) > tolerance) {                                      \
        RS_THROW(RS_InvalidConfigurationException(                                    \
            "Invalid " #class_name " config, " #attribute " must be close to " #value \
            " within " #tolerance                                                     \
        ));                                                                           \
    }

#endif  // RS_BASE_CONFIG_H
