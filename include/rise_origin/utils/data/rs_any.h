// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_ANY_H
#define RS_ANY_H

#include <any>

#include "rise_origin/utils/rs_exception.h"

namespace rise {

/**
 * @brief A container that can hold any type of value.
 *
 * @note RS_Any is not recommended for designing C++ internal classes requiring serialization;
 * use RS_Variant instead for a cleaner design. For communication between the RISE system and
 * outside (e.g. Python), RS_Any with a type enum is the recommended pattern.
 */
class RS_Any {
   public:
    /**
     * @brief The underlying std::any storage.
     */
    std::any data;

    RS_Any() = default;
    RS_Any(const RS_Any&) = default;
    RS_Any(RS_Any&&) = default;

    template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, RS_Any>>>
    RS_Any(T&& val) : data(std::forward<T>(val)) {}

    RS_Any& operator=(const RS_Any&) = default;
    RS_Any& operator=(RS_Any&&) = default;

    template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, RS_Any>>>
    RS_Any& operator=(T&& val) {
        data = std::forward<T>(val);
        return *this;
    }

    [[nodiscard]] bool empty() const { return !data.has_value(); }
    explicit operator bool() const { return data.has_value(); }
    void reset() { data.reset(); }
    [[nodiscard]] const std::type_info& type() const { return data.type(); }

    template <typename T>
    [[nodiscard]] bool is() const {
        return data.has_value() && data.type() == typeid(T);
    }

    /**
     * @brief Access the contained value as the specified type as an lvalue.
     *
     * If the container is empty or holds a different type, a default-constructed instance of the
     * requested type replaces the stored value before returning a reference. If the type is not
     * default-constructible and the container is empty, an exception is thrown. This keeps
     * deserialization code simple.
     */
    template <typename T>
    T& as() & {
        try {
            if (!data.has_value() || data.type() != typeid(T)) {
                if constexpr (std::is_default_constructible_v<T>) {
                    data = T{};
                } else {
                    RS_THROW(RS_InvalidCastException(
                        "Cannot cast from empty value to {}", typeid(T).name()
                    ));
                }
            }
            return std::any_cast<T&>(data);
        } catch (const std::bad_any_cast&) {
            RS_THROW(RS_InvalidCastException(
                "Cannot cast from {} to {}", data.type().name(), typeid(T).name()
            ));
        }
    }

    template <typename T>
    const T& as() const& {
        try {
            return std::any_cast<const T&>(data);
        } catch (const std::bad_any_cast&) {
            if (data.has_value()) {
                RS_THROW(RS_InvalidCastException(
                    "Cannot cast from {} to {}", data.type().name(), typeid(T).name()
                ));
            } else {
                RS_THROW(
                    RS_InvalidCastException("Cannot cast from empty value to {}", typeid(T).name())
                );
            }
        }
    }

    template <typename T>
    T&& as() && {
        try {
            return std::move(std::any_cast<T&>(data));
        } catch (const std::bad_any_cast&) {
            RS_THROW(RS_InvalidCastException(
                "Cannot cast from {} to {}", data.type().name(), typeid(T).name()
            ));
        }
    }
};

}  // namespace rise

#endif  // RS_ANY_H
