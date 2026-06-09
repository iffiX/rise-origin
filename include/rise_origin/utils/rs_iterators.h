// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_ITERATORS_H
#define RS_ITERATORS_H

#include <iterator>
#include <tuple>

namespace rise {

/**
 * @brief Template for performing enumerate function in python
 * @snippet for (auto &&[index, element] : enumerate(iterable)) { ... }
 */
template <typename Iterable>
auto enumerate(Iterable&& iterable) {
    using Iterator = decltype(std::begin(std::declval<Iterable>()));
    using T = decltype(*std::declval<Iterator>());

    struct Enumerated {
        std::size_t index;
        decltype(*std::declval<Iterator>()) element;
    };

    struct Enumerator {
        Iterator iterator;
        std::size_t index;

        auto operator!=(const Enumerator& other) const { return iterator != other.iterator; }

        auto& operator++() {
            ++iterator;
            ++index;
            return *this;
        }

        auto operator*() const { return Enumerated{index, *iterator}; }
    };

    struct Wrapper {
        Iterable& iterable;

        [[nodiscard]] auto begin() const { return Enumerator{std::begin(iterable), 0U}; }

        [[nodiscard]] auto end() const { return Enumerator{std::end(iterable), 0U}; }
    };

    return Wrapper{std::forward<Iterable>(iterable)};
}

/**
 * @brief Template for performing zip function in python
 * @snippet for (auto &&[element1, element2] : zip(iterable1, iterable2)) { ... }
 */

namespace zip_helper {
template <typename... Containers>
std::ptrdiff_t minDistance(Containers&&... containers) {
    // For each container, compute distance(begin, end).
    // Then return the std::min of all those distances.
    std::ptrdiff_t distances[] = {
        static_cast<std::ptrdiff_t>(std::distance(std::begin(containers), std::end(containers)))...
    };
    // Compute the global min
    auto n = sizeof...(containers);
    std::ptrdiff_t result = distances[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (distances[i] < result) {
            result = distances[i];
        }
    }
    return result;
}
}  // namespace zip_helper

template <typename... Iterables>
auto zip(Iterables&&... iterables) {
    struct Zipper {
        std::tuple<decltype(std::begin(std::declval<Iterables>()))...> iterators;

        bool operator!=(const Zipper& other) const {
            return std::get<0>(iterators) != std::get<0>(other.iterators);
        }

        Zipper& operator++() {
            std::apply([](auto&... its) { (..., ++its); }, iterators);
            return *this;
        }

        auto operator*() const {
            return std::apply(
                [](auto&... its) { return std::tuple<decltype(*its)...>(*its...); }, iterators
            );
        }
    };

    struct Wrapper {
        std::tuple<Iterables&&...> refs;

        [[nodiscard]] auto begin() const {
            return Zipper{
                std::apply([](auto&&... cs) { return std::tuple{(std::begin(cs))...}; }, refs)
            };
        }

        [[nodiscard]] auto end() const {
            std::ptrdiff_t dist =
                std::apply([](auto&&... cs) { return zip_helper::minDistance(cs...); }, refs);
            return Zipper{std::apply(
                [dist](auto&&... cs) { return std::tuple{(std::begin(cs) + dist)...}; }, refs
            )};
        }
    };

    return Wrapper{std::forward_as_tuple(std::forward<Iterables>(iterables)...)};
}

}  // namespace rise

#endif  // RS_ITERATORS_H
