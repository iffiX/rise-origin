// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_ANY_H
#define RS_ANY_H

#include <any>

#include "rs_exception.h"

namespace rise {

RS_MAKE_EXCEPTION(RS_InvalidCastException, RS_DataException)

/**
 * @brief A wrapper around std::any with enhanced type conversion capabilities
 *
 * RS_Any extends std::any by providing automatic type instantiation when accessing values,
 * simplifying serialization and deserialization workflows.
 */
class RS_Any {
   public:
    /**
     * @brief The underlying std::any storage
     */
    std::any value;

    /**
     * @brief Default constructor
     */
    RS_Any() = default;

    /**
     * @brief Constructs RS_Any with the given value
     * @tparam T Type of the value to store
     * @param val Value to store
     */
    template <typename T>
    RS_Any(T val) : value(std::move(val)) {}

    /**
     * @brief Assignment operator from any type
     * @tparam T Type of the value to store
     * @param val Value to store
     * @return Reference to this object
     */
    template <typename T>
    RS_Any& operator=(T val) {
        value = std::move(val);
        return *this;
    }

    /**
     * @brief Checks if the container holds a value
     * @return True if the container holds a value
     */
    [[nodiscard]] bool empty() const { return !value.has_value(); }

    /**
     * @brief Reset the container to empty state
     */
    void reset() { value.reset(); }

    /**
     * @brief Get the type info of the contained value
     * @return Type info of the contained value
     */
    [[nodiscard]] const std::type_info& type() const { return value.type(); }

    /**
     * @brief Access the contained value as the specified type
     *
     * If the container is empty or holds a different type, a default-constructed
     * instance of the requested type will be created before returning a reference.
     * This ensures safe access for both read and write operations, particularly
     * during deserialization processes.
     *
     * @tparam T Type to access the value as
     * @return Reference to the value of type T
     */
    template <typename T>
    T& as() {
        try {
            if (!value.has_value() || value.type() != typeid(T)) {
                value = T{};
            }
            return std::any_cast<T&>(value);
        } catch (const std::bad_any_cast&) {
            RS_THROW(RS_InvalidCastException(
                "Cannot cast from {} to {}", value.type().name(), typeid(T).name()
            ));
        }
    }

    /**
     * @brief Const access to the contained value as the specified type
     *
     * @tparam T Type to access the value as
     * @return Const reference to the value of type T
     * @throw RS_InvalidCastException if the value cannot be cast to type T
     */
    template <typename T>
    const T& as() const {
        try {
            return std::any_cast<const T&>(value);
        } catch (const std::bad_any_cast&) {
            if (value.has_value()) {
                RS_THROW(RS_InvalidCastException(
                    "Cannot cast from {} to {}", value.type().name(), typeid(T).name()
                ));
            } else {
                RS_THROW(
                    RS_InvalidCastException("Cannot cast from empty value to {}", typeid(T).name())
                );
            }
        }
    }
};

}  // namespace rise

#endif  // RS_ANY_H