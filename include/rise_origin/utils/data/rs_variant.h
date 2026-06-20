// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VARIANT_H
#define RS_VARIANT_H

#include <variant>

#include "rise_origin/utils/meta/rs_constexpr.h"
#include "rise_origin/utils/rs_exception.h"

namespace rise {

/**
 * @brief A container that can hold one of multiple types of values.
 *
 * Serialization is baked in and dispatches on the runtime variant index.
 * On-disk layout: { "variant_index": <RIndex>, "data": <held alternative> }.
 *
 * @tparam Types The types that the variant can hold.
 */
template <typename... Types>
class RS_Variant {
   public:
    std::variant<Types...> data;

    RS_Variant() = default;
    RS_Variant(const RS_Variant&) = default;
    RS_Variant(RS_Variant&&) = default;

    template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, RS_Variant>>>
    RS_Variant(T&& val) : data(std::forward<T>(val)) {}

    RS_Variant& operator=(const RS_Variant&) = default;
    RS_Variant& operator=(RS_Variant&&) = default;

    template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, RS_Variant>>>
    RS_Variant& operator=(T&& val) {
        data = std::forward<T>(val);
        return *this;
    }

    template <typename T>
    static constexpr bool contains() {
        return std::disjunction<std::is_same<T, Types>...>::value;
    }

    /**
     * @brief Zero-based index of the currently held alternative, or RS_NULL_INDEX when empty.
     */
    [[nodiscard]] RSize index() const {
        size_t i = data.index();
        if (i == std::variant_npos) {
            return RS_NULL_INDEX;
        }
        return static_cast<RSize>(i);
    }

    [[nodiscard]] const std::type_info& type() const {
        const std::type_info* type_info = nullptr;
        RIndex variant_index = index();
        if (variant_index == RS_NULL_INDEX) {
            return typeid(void);
        }
        switchOver<0, sizeof...(Types)>(variant_index, [&](auto i) {
            type_info = &typeid(std::variant_alternative_t<i.value, std::variant<Types...>>);
        });
        return *type_info;
    }

    template <typename T>
    [[nodiscard]] bool is() const {
        return std::holds_alternative<T>(data);
    }

    template <typename T>
    T& as() & {
        try {
            return std::get<T>(data);
        } catch (const std::bad_variant_access&) {
            RS_THROW(RS_InvalidCastException("Cannot cast to {}", typeid(T).name()));
        }
    }

    template <typename T>
    const T& as() const& {
        try {
            return std::get<T>(data);
        } catch (const std::bad_variant_access&) {
            RS_THROW(RS_InvalidCastException("Cannot cast to {}", typeid(T).name()));
        }
    }

    template <typename T>
    T&& as() && {
        try {
            return std::move(std::get<T>(data));
        } catch (const std::bad_variant_access&) {
            RS_THROW(RS_InvalidCastException("Cannot cast to {}", typeid(T).name()));
        }
    }

    template <size_t I>
    auto& get() & {
        return std::get<I>(data);
    }

    template <size_t I>
    const auto& get() const& {
        return std::get<I>(data);
    }

    template <size_t I>
    auto&& get() && {
        return std::move(std::get<I>(data));
    }

    template <typename T>
    T* getIf() {
        return std::get_if<T>(&data);
    }

    template <typename T>
    const T* getIf() const {
        return std::get_if<T>(&data);
    }

    template <typename T, typename... Args>
    T& emplace(Args&&... args) {
        return data.template emplace<T>(std::forward<Args>(args)...);
    }

    template <size_t I, typename... Args>
    auto& emplace(Args&&... args) {
        return data.template emplace<I>(std::forward<Args>(args)...);
    }

    template <typename Visitor>
    decltype(auto) visit(Visitor&& visitor) {
        return std::visit(std::forward<Visitor>(visitor), data);
    }

    template <typename Visitor>
    decltype(auto) visit(Visitor&& visitor) const {
        return std::visit(std::forward<Visitor>(visitor), data);
    }

    bool operator==(const RS_Variant& other) const { return data == other.data; }
    bool operator!=(const RS_Variant& other) const { return data != other.data; }

    /// Generic serialization (works with both RS_MessagePackSerializer and ...Deserializer).
    template <typename Serializer>
    void serialize(Serializer& s) {
        if (s.isSerialize()) {
            RIndex variant_index = index();
            s("variant_index", variant_index);

            if (variant_index != RS_NULL_INDEX) {
                switchOver<0, sizeof...(Types)>(variant_index, [&](auto i) {
                    s("data", std::get<i.value>(data));
                });
            }
        } else {
            RIndex variant_index = RS_NULL_INDEX;
            s("variant_index", variant_index);

            if (variant_index != RS_NULL_INDEX) {
                switchOver<0, sizeof...(Types)>(variant_index, [&](auto i) {
                    data.template emplace<i.value>();
                    s("data", std::get<i.value>(data));
                });
            }
        }
    }
};

}  // namespace rise

#endif  // RS_VARIANT_H
