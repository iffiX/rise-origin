// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_OPTIONAL_H
#define RS_OPTIONAL_H

#include <optional>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"

namespace rise {

/**
 * @brief An optional container that can hold a value or be empty.
 *
 * @tparam T The type of the value that may or may not be present.
 */
template <typename T>
class RS_Optional {
   public:
    std::optional<T> data;

    RS_Optional() = default;
    RS_Optional(const RS_Optional&) = default;
    RS_Optional(RS_Optional&&) = default;

    template <
        typename U,
        typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, RS_Optional>>>
    RS_Optional(U&& val) : data(std::forward<U>(val)) {}

    RS_Optional(RS_NoneType) : data(std::nullopt) {}

    RS_Optional& operator=(const RS_Optional&) = default;
    RS_Optional& operator=(RS_Optional&&) = default;

    template <
        typename U,
        typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, RS_Optional>>>
    RS_Optional& operator=(U&& val) {
        data = std::forward<U>(val);
        return *this;
    }

    RS_Optional& operator=(RS_NoneType) {
        data = std::nullopt;
        return *this;
    }

    bool empty() const { return !data.has_value(); }
    explicit operator bool() const { return data.has_value(); }
    void reset() { data.reset(); }

    template <typename... Args>
    T& emplace(Args&&... args) {
        return data.emplace(std::forward<Args>(args)...);
    }

    T& get() & {
        if (!data.has_value()) {
            RS_THROW(RS_UninitializedException("Attempted to access value of an empty optional"));
        }
        return data.value();
    }

    const T& get() const& {
        if (!data.has_value()) {
            RS_THROW(RS_UninitializedException("Attempted to access value of an empty optional"));
        }
        return data.value();
    }

    T&& get() && {
        if (!data.has_value()) {
            RS_THROW(RS_UninitializedException("Attempted to access value of an empty optional"));
        }
        return std::move(data.value());
    }

    T& operator*() & { return get(); }
    const T& operator*() const& { return get(); }
    T&& operator*() && { return get(); }

    T* operator->() {
        if (!data.has_value()) {
            RS_THROW(RS_UninitializedException("Attempted to access value of an empty optional"));
        }
        return &data.value();
    }

    const T* operator->() const {
        if (!data.has_value()) {
            RS_THROW(RS_UninitializedException("Attempted to access value of an empty optional"));
        }
        return &data.value();
    }

    T valueOr(const T& default_value) const { return data.value_or(default_value); }
    T valueOr(T&& default_value) const { return data.value_or(std::move(default_value)); }

    bool operator==(const RS_Optional& other) const { return data == other.data; }
    bool operator!=(const RS_Optional& other) const { return data != other.data; }

    bool operator==(const T& other_value) const {
        return data.has_value() && data.value() == other_value;
    }
    bool operator!=(const T& other_value) const {
        return !data.has_value() || data.value() != other_value;
    }

    /// Generic serialization.
    template <typename Serializer>
    void serialize(Serializer& s) {
        if (s.isSerialize()) {
            bool has_value = data.has_value();
            s("has_value", has_value);
            if (has_value) {
                s("data", data.value());
            }
        } else {
            bool has_value = false;
            s("has_value", has_value);
            if (has_value) {
                if (data.has_value()) data.reset();
                data.emplace();
                s("data", data.value());
            } else {
                data.reset();
            }
        }
    }
};

}  // namespace rise

#endif  // RS_OPTIONAL_H
