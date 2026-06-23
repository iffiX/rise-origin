// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_BASE_H
#define RSC_BASE_H

#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_rsc_property_tree_reader.h"
#include "rs_rsc_property_tree_writer.h"

namespace rise {

class RS_RSCValidator;

namespace rsc {

/**
 * @brief Root base class for all Rise Characterization (RSC) types.
 *
 * Every RSC class inherits (directly or transitively) from this. Provides the
 * read/write/validate contract driven by RS_RSCPropertyTreeReader, plus the
 * shared reference composition utilities (RSCRef, RSCVec, MemoRef) and the
 * shallowCopy/deepCopy contract used to build the RSC tree as a reference
 * counted directed acyclic graph.
 */
class RSC_Base {
   public:
    /// Map from an original node to its deep copied node, used to preserve sharing.
    using MemoMap = std::unordered_map<const RSC_Base*, std::shared_ptr<RSC_Base>>;

    /**
     * @brief Shared pointer to a deepCopy memo map, default constructing to null.
     *
     * Lets deepCopy take a defaulted argument (MemoRef memo = {}) yet still detect
     * the top level call by a null check, where it lazily creates the one shared
     * map. Passing the pointer down by value shares that map so a node referenced by
     * many parents is deep copied once and the sharing structure is preserved.
     */
    class MemoRef : public std::shared_ptr<MemoMap> {
       public:
        using std::shared_ptr<MemoMap>::shared_ptr;
        MemoRef(std::shared_ptr<MemoMap> ptr) : std::shared_ptr<MemoMap>(std::move(ptr)) {}
    };

    /**
     * @brief Shared pointer to a single embedded sub RSC.
     *
     * Holds a sub RSC by a shared pointer so that copying a parent RSC shares
     * the child instead of deep copying it. T is always an RSC type. Copy and
     * assignment share the referenced node; deepCopy produces an independent node.
     */
    template <typename T>
    class RSCRef {
       public:
        RSCRef() = default;
        RSCRef(std::nullptr_t) {}

        /// Wrap a value by allocating a shared node holding a copy of it.
        RSCRef(const T& value) : _ptr(std::make_shared<T>(value)) {}

        /// Adopt an existing shared node.
        RSCRef(std::shared_ptr<T> ptr) : _ptr(std::move(ptr)) {}

        /// Share from a ref to a derived RSC type.
        template <typename U, typename = std::enable_if_t<std::is_convertible<U*, T*>::value>>
        RSCRef(const RSCRef<U>& other) : _ptr(other.shared()) {}

        /// Adopt a shared node of a derived RSC type.
        template <typename U, typename = std::enable_if_t<std::is_convertible<U*, T*>::value>>
        RSCRef(std::shared_ptr<U> ptr) : _ptr(std::move(ptr)) {}

        /// Make a ref by allocating a shared copy of the object at ptr, as the pointed to type.
        template <typename U>
        static RSCRef<U> fromThis(const U* ptr) {
            return RSCRef<U>(std::make_shared<U>(*ptr));
        }

        T* operator->() const { return _ptr.get(); }
        T& operator*() const { return *_ptr; }
        T* get() const { return _ptr.get(); }
        explicit operator bool() const { return static_cast<bool>(_ptr); }
        bool isNull() const { return _ptr == nullptr; }

        /// Behave like a reference to the held value, so the ref binds wherever a T& is expected.
        operator T&() { return *_ptr; }
        operator const T&() const { return *_ptr; }

        const std::shared_ptr<T>& shared() const { return _ptr; }

        /// Compare by referenced value, treating two null refs as equal.
        bool operator==(const RSCRef& other) const {
            if (_ptr == nullptr || other._ptr == nullptr) {
                return _ptr == other._ptr;
            }
            return *_ptr == *other._ptr;
        }
        bool operator!=(const RSCRef& other) const { return !(*this == other); }

        /// Serialize as the referenced value, allocating one when reading into a null ref.
        template <typename Serializer>
        void serialize(Serializer& s) {
            if (_ptr == nullptr) {
                _ptr = std::make_shared<T>();
            }
            _ptr->serialize(s);
        }

        /// Produce an independent ref by deep copying the referenced node.
        RSCRef deepCopy(MemoRef memo) const {
            if (_ptr == nullptr) {
                return RSCRef();
            }
            if constexpr (std::is_base_of<RSC_Base, T>::value) {
                return RSCRef(std::static_pointer_cast<T>(_ptr->deepCopy(memo).shared()));
            } else {
                return RSCRef(std::make_shared<T>(rscDeepCopyValue(*_ptr, memo)));
            }
        }

       private:
        std::shared_ptr<T> _ptr;
    };

    /**
     * @brief Shared vector of embedded sub RSCs.
     *
     * Stores a list of sub RSCs as shared pointers so that copying the owning
     * RSC shares every element. Element access dereferences to T& so callers read
     * elements[i].field. T is always an RSC type.
     */
    template <typename T>
    class RSCVec {
       public:
        using value_type = T;

        RSCVec() = default;

        void push_back(const T& value) { _data.emplace_back(std::make_shared<T>(value)); }
        void push_back(const RSCRef<T>& ref) { _data.push_back(ref); }

        template <typename... Args>
        T& emplace_back(Args&&... args) {
            _data.emplace_back(std::make_shared<T>(std::forward<Args>(args)...));
            return *_data.back();
        }

        T& operator[](std::size_t index) { return *_data[index]; }
        const T& operator[](std::size_t index) const { return *_data[index]; }
        T& at(std::size_t index) { return *_data.at(index); }
        const T& at(std::size_t index) const { return *_data.at(index); }

        const RSCRef<T>& ref(std::size_t index) const { return _data[index]; }

        std::size_t size() const { return _data.size(); }
        bool empty() const { return _data.empty(); }
        void clear() { _data.clear(); }
        void reserve(std::size_t capacity) { _data.reserve(capacity); }
        void erase(std::size_t index) { _data.erase(_data.begin() + index); }
        void insert(std::size_t index, const T& value) {
            _data.insert(_data.begin() + index, std::make_shared<T>(value));
        }

        /// Compare element by element by referenced value.
        bool operator==(const RSCVec& other) const {
            if (_data.size() != other._data.size()) {
                return false;
            }
            for (std::size_t i = 0; i < _data.size(); i++) {
                if (!(*_data[i] == *other._data[i])) {
                    return false;
                }
            }
            return true;
        }
        bool operator!=(const RSCVec& other) const { return !(*this == other); }

        /// Forward iterator that dereferences each pointer to T&.
        template <typename VecIt, typename Ref>
        class IteratorBase {
           public:
            explicit IteratorBase(VecIt it) : _it(it) {}
            Ref operator*() const { return **_it; }
            IteratorBase& operator++() {
                ++_it;
                return *this;
            }
            bool operator==(const IteratorBase& other) const { return _it == other._it; }
            bool operator!=(const IteratorBase& other) const { return _it != other._it; }

           private:
            VecIt _it;
        };

        using iterator = IteratorBase<typename std::vector<RSCRef<T>>::iterator, T&>;
        using const_iterator =
            IteratorBase<typename std::vector<RSCRef<T>>::const_iterator, const T&>;

        iterator begin() { return iterator(_data.begin()); }
        iterator end() { return iterator(_data.end()); }
        const_iterator begin() const { return const_iterator(_data.begin()); }
        const_iterator end() const { return const_iterator(_data.end()); }

        /// Produce an independent vector by deep copying every element.
        RSCVec deepCopy(MemoRef memo) const {
            RSCVec out;
            out._data.reserve(_data.size());
            for (const auto& element : _data) {
                out._data.push_back(element.deepCopy(memo));
            }
            return out;
        }

       private:
        std::vector<RSCRef<T>> _data;
    };

    /// RSC file key naming the concrete type of a variant element.
    static constexpr const char* VAR_TYPE_KEY = "type";

    /// RSC file key holding the payload fields of a variant element.
    static constexpr const char* VAR_DATA_KEY = "data";

    virtual ~RSC_Base() = default;

    /**
     * @brief Return a new node that shares this node's sub RSC children.
     * @details Generic. Concrete types only supply _shallowCopy.
     */
    RSCRef<RSC_Base> shallowCopy() const { return _shallowCopy(); }

    /**
     * @brief Return an independent deep copy of this node and its sub RSC children.
     * @details Generic. memo preserves shared sub nodes across the copy so a node
     * referenced by many parents is copied once. Concrete types supply _shallowCopy,
     * and composites override _deepCopyShared to deep copy their shared members.
     */
    RSCRef<RSC_Base> deepCopy(MemoRef memo = {}) const {
        if (!memo) {
            memo = MemoRef(std::make_shared<MemoMap>());
        }
        auto found = memo->find(this);
        if (found != memo->end()) {
            return found->second;
        }
        RSCRef<RSC_Base> copy = _shallowCopy();
        (*memo)[this] = copy.shared();
        copy->_deepCopyShared(memo);
        return copy;
    }
    
    /**
     * @brief Read the characterization from a reader.
     */
    void read(const RS_RSCPropertyTreeReader& reader) { _read(reader); }

    /**
     * @brief Write the characterization to a writer.
     */
    void write(RS_RSCPropertyTreeWriter writer) const { _write(writer); }

    /**
     * @brief Submit this characterization's checks to a validator.
     * @details The submit half of validation. A parent calls this on its children to register their
     * validator functions. The validator runs them and throws on the first failure.
     */
    void validate(RS_RSCValidator& validator) const { _validate(validator); }

    bool operator==(const RSC_Base&) const { return true; }

   protected:
    virtual void _validate(RS_RSCValidator& validator) const;
    virtual void _read(const RS_RSCPropertyTreeReader& reader) = 0;
    virtual void _write(RS_RSCPropertyTreeWriter& writer) const = 0;

    /// Allocate a copy of the concrete type that shares its sub RSC children.
    virtual RSCRef<RSC_Base> _shallowCopy() const = 0;

    /// Replace shared sub RSC members with deep copies. Default does nothing; composites override.
    virtual void _deepCopyShared(MemoRef memo) { (void)memo; }
};

}  // namespace rsc

namespace mp_helper {

/// Forward declaration from the message pack serializer.
template <typename T>
struct is_container;

/// Serialize an RSCVec exactly like the std::vector it replaced, as an array of elements.
template <typename T>
struct is_container<rsc::RSC_Base::RSCVec<T>> {
    static const bool value = true;
};

}  // namespace mp_helper

/// Deep copy the RSC held by a variant, preserving sharing across the copy through the memo.
template <typename... Types>
RS_Variant<Types...>
rscDeepCopyValue(const RS_Variant<Types...>& variant, rsc::RSC_Base::MemoRef memo) {
    RS_Variant<Types...> out;
    if (variant.index() == RS_NULL_INDEX) {
        return out;
    }
    variant.visit([&](const auto& held) {
        using Held = std::decay_t<decltype(held)>;
        out = *std::static_pointer_cast<Held>(held.deepCopy(memo).shared());
    });
    return out;
}

}  // namespace rise

#define RSC_EXPECT_NOT_NULL(class_name, attribute, value)                    \
    if (attribute == static_cast<decltype(attribute)>(value)) {              \
        RS_THROW(RS_InvalidCharacterizationException(                        \
            "Invalid " #class_name ", " #attribute " must not be null index" \
        ));                                                                  \
    }

#define RSC_EXPECT_NOT_EMPTY(class_name, attribute)                     \
    if (attribute.empty()) {                                            \
        RS_THROW(RS_InvalidCharacterizationException(                   \
            "Invalid " #class_name ", " #attribute " must not be empty" \
        ));                                                             \
    }

#define RSC_EXPECT_NE(class_name, attribute, value)                                \
    if (attribute == static_cast<decltype(attribute)>(value)) {                    \
        RS_THROW(RS_InvalidCharacterizationException(                              \
            "Invalid " #class_name ", " #attribute " must not be equal to " #value \
        ));                                                                        \
    }

#define RSC_EXPECT_EQ(class_name, attribute, value)                            \
    if (attribute != static_cast<decltype(attribute)>(value)) {                \
        RS_THROW(RS_InvalidCharacterizationException(                          \
            "Invalid " #class_name ", " #attribute " must be equal to " #value \
        ));                                                                    \
    }

#define RSC_EXPECT_GT(class_name, attribute, value)                                \
    if (attribute <= static_cast<decltype(attribute)>(value)) {                    \
        RS_THROW(RS_InvalidCharacterizationException(                              \
            "Invalid " #class_name ", " #attribute " must be greater than " #value \
        ));                                                                        \
    }

#define RSC_EXPECT_GE(class_name, attribute, value)                                            \
    if (attribute < static_cast<decltype(attribute)>(value)) {                                 \
        RS_THROW(RS_InvalidCharacterizationException(                                          \
            "Invalid " #class_name ", " #attribute " must be greater than or equal to " #value \
        ));                                                                                    \
    }

#define RSC_EXPECT_LT(class_name, attribute, value)                             \
    if (attribute >= static_cast<decltype(attribute)>(value)) {                 \
        RS_THROW(RS_InvalidCharacterizationException(                           \
            "Invalid " #class_name ", " #attribute " must be less than " #value \
        ));                                                                     \
    }

#define RSC_EXPECT_LE(class_name, attribute, value)                                         \
    if (attribute > static_cast<decltype(attribute)>(value)) {                              \
        RS_THROW(RS_InvalidCharacterizationException(                                       \
            "Invalid " #class_name ", " #attribute " must be less than or equal to " #value \
        ));                                                                                 \
    }

#define RSC_EXPECT_CLOSE(class_name, attribute, value, tolerance)              \
    if (RS_ABS(attribute - value) > tolerance) {                               \
        RS_THROW(RS_InvalidCharacterizationException(                          \
            "Invalid " #class_name ", " #attribute " must be close to " #value \
            " within " #tolerance                                              \
        ));                                                                    \
    }

#endif  // RSC_BASE_H
