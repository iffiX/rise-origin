// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RSC_REGISTRY_H
#define RS_RSC_REGISTRY_H

#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/rs_exception.h"
#include "rs_rsc_file_tree_reader.h"
#include "rs_rsc_file_tree_writer.h"
#include "rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief Singleton mapping a type name to the read and write of its RSC type.
 * @details A variant element is stored as { type, data }. This turns the name into a
 * constructed and read node, and a held alternative back into its name and data.
 */
class RS_RSCRegistry {
   public:
    RS_RSCRegistry(const RS_RSCRegistry &) = delete;
    RS_RSCRegistry &operator=(const RS_RSCRegistry &) = delete;

    /**
     * @brief Get the process wide registry, built on first use.
     */
    static RS_RSCRegistry &getInstance() {
        static RS_RSCRegistry instance;
        return instance;
    }

    /**
     * @brief Set a node into the matching variant alternative, false when none matches.
     */
    template <typename... Ts>
    static bool tryAssign(RS_Variant<Ts...> &out, const RSC_Base::RSCRef<RSC_Base> &node) {
        return ([&] {
            if (auto *held = dynamic_cast<Ts *>(node.get())) {
                out = *held;
                return true;
            }
            return false;
        }() || ...);
    }

    /**
     * @brief Read a { type, data } element into the matching variant alternative.
     */
    template <typename... Ts>
    static void read(RS_Variant<Ts...> &out, const RS_RSCFileTreeReader &element) {
        std::string type = element.get<std::string>(RSC_Base::VAR_TYPE_KEY);
        RSC_Base::RSCRef<RSC_Base> node = make(type, element.getChild(RSC_Base::VAR_DATA_KEY));
        if (!tryAssign(out, node)) {
            RS_THROW(RS_InvalidCharacterizationException("Unexpected RSC type {}", type));
        }
    }

    /**
     * @brief Write the held variant alternative as a { type, data } element.
     */
    template <typename... Ts>
    static void write(const RS_Variant<Ts...> &in, RS_RSCFileTreeWriter element) {
        in.visit([&](const auto &held) {
            using T = std::decay_t<decltype(held)>;
            element.add(RSC_Base::VAR_TYPE_KEY, getName(typeid(T)));
            held.write(element.addChild(RSC_Base::VAR_DATA_KEY));
        });
    }

    /**
     * @brief Construct a node by type name and read it.
     */
    static RSC_Base::RSCRef<RSC_Base> make(std::string_view type, const RS_RSCFileTreeReader &data);

    /**
     * @brief Get the registered name of a concrete RSC type.
     */
    static const std::string &getName(const std::type_info &type);

    /**
     * @brief Get the registered type names grouped by category.
     */
    static std::map<std::string, std::vector<std::string>> getCategory();

    /**
     * @brief Construct a default node of a type name, as an example to fill in.
     */
    static RSC_Base::RSCRef<RSC_Base> getExample(std::string_view type);

   private:
    using ReadFn = std::function<RSC_Base::RSCRef<RSC_Base>(const RS_RSCFileTreeReader &)>;
    using MakeFn = std::function<RSC_Base::RSCRef<RSC_Base>()>;

    /// Maps type name -> construct and read into a base node.
    std::unordered_map<std::string, ReadFn> _read_map;

    /// Maps type name -> construct a default base node.
    std::unordered_map<std::string, MakeFn> _make_map;

    /// Maps type_index -> registered type name.
    std::unordered_map<std::type_index, std::string> _name_map;

    /// Maps category -> its type names, in registration order.
    std::map<std::string, std::vector<std::string>> _category_map;

    RS_RSCRegistry() { _registerAll(); }

    /**
     * @brief Register every savable RSC type, defined as one list in the source file.
     */
    void _registerAll();

    /**
     * @brief Register one concrete RSC type under name and category.
     */
    template <typename T>
    void _register(const std::string &name, const std::string &category) {
        _read_map[name] = [](const RS_RSCFileTreeReader &data) -> RSC_Base::RSCRef<RSC_Base> {
            auto node = std::make_shared<T>();
            node->read(data);
            return node;
        };
        _make_map[name] = []() -> RSC_Base::RSCRef<RSC_Base> { return std::make_shared<T>(); };
        _name_map[std::type_index(typeid(T))] = name;
        _category_map[category].push_back(name);
    }

    /**
     * @brief Construct a node by type name and read it.
     */
    RSC_Base::RSCRef<RSC_Base> _make(std::string_view type, const RS_RSCFileTreeReader &data) const;

    /**
     * @brief Get the registered name of a concrete RSC type.
     */
    const std::string &_getName(const std::type_info &type) const;

    /**
     * @brief Get the registered type names grouped by category.
     */
    std::map<std::string, std::vector<std::string>> _getCategory() const;

    /**
     * @brief Construct a default node of a type name, as an example to fill in.
     */
    RSC_Base::RSCRef<RSC_Base> _getExample(std::string_view type) const;
};

}  // namespace rsc

}  // namespace rise

#endif  // RS_RSC_REGISTRY_H
