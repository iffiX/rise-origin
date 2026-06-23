// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <algorithm>

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/device/algorithm/rs_copy.h"
#include "rise_origin/logic/utils/rs_material_pair_c.h"
#include "rise_origin/logic/vision/rs_vision_material_m.h"
#include "rs_kernel_material_manager.h"
#include "rs_kernel_texture_manager.h"

using namespace std;
using namespace rise::rsc;
using namespace rise::logic;
using namespace rise::device;

namespace rise {

/*****************************************************************
 * Construction
 *****************************************************************/
RS_KernelMaterialManager::RS_KernelMaterialManager() { _createLutTables(); }

void RS_KernelMaterialManager::_createLutTables() {
    _unary_lut = {
        {"density",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (auto it = m.find(typeid(RSC_MaterialPhysicalPropertyRigid)); it != m.end()) {
                  return it->second.as<RSC_MaterialPhysicalPropertyRigid>().density;
              }
              if (auto it = m.find(typeid(RSC_MaterialPhysicalPropertySoft)); it != m.end()) {
                  return it->second.as<RSC_MaterialPhysicalPropertySoft>().density;
              }
              return RS_NONE;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPhysicalPropertyRigid)) != m.end() ||
                     m.find(typeid(RSC_MaterialPhysicalPropertySoft)) != m.end();
          }}},
        {"fixed",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (m.find(typeid(RSC_MaterialPhysicalPropertyFixed)) != m.end()) return true;
              return RS_NONE;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPhysicalPropertyFixed)) != m.end();
          }}},
        {"rigid",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (m.find(typeid(RSC_MaterialPhysicalPropertyRigid)) != m.end()) return true;
              return RS_NONE;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPhysicalPropertyRigid)) != m.end();
          }}},
        {"atlas",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (m.find(typeid(RSC_MaterialPhysicalPropertyAtlas)) != m.end()) return true;
              return RS_NONE;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPhysicalPropertyAtlas)) != m.end();
          }}},
        {"soft",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (auto it = m.find(typeid(RSC_MaterialPhysicalPropertySoft)); it != m.end()) {
                  const auto& p = it->second.as<RSC_MaterialPhysicalPropertySoft>();
                  return std::make_tuple(
                      p.min_expansion,
                      p.max_expansion,
                      p.elastic_modulus,
                      p.fail_stress,
                      p.poissons_ratio
                  );
              }
              return RS_NONE;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPhysicalPropertySoft)) != m.end();
          }}},
        {"mass",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (auto it = m.find(typeid(RSC_MaterialPhysicalPropertyAtlas)); it != m.end()) {
                  return it->second.as<RSC_MaterialPhysicalPropertyAtlas>().mass;
              }
              return RS_NONE;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPhysicalPropertyAtlas)) != m.end();
          }}},
        {"com_position",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (auto it = m.find(typeid(RSC_MaterialPhysicalPropertyAtlas)); it != m.end()) {
                  return it->second.as<RSC_MaterialPhysicalPropertyAtlas>().com_position;
              }
              return RS_NONE;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPhysicalPropertyAtlas)) != m.end();
          }}},
        {"inertia",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (auto it = m.find(typeid(RSC_MaterialPhysicalPropertyAtlas)); it != m.end()) {
                  return it->second.as<RSC_MaterialPhysicalPropertyAtlas>().inertia;
              }
              return RS_NONE;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPhysicalPropertyAtlas)) != m.end();
          }}},
        {"is_transparent",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (auto it = m.find(typeid(RSC_MaterialOpticalPropertySimpleMicrofacet));
                  it != m.end()) {
                  return it->second.as<RSC_MaterialOpticalPropertySimpleMicrofacet>()
                      .is_transparent;
              }
              return false;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialOpticalPropertySimpleMicrofacet)) != m.end();
          }}},
        {"is_emissive",
         {[](const UnaryPropertiesMap& m) -> RS_Any {
              if (auto it = m.find(typeid(RSC_MaterialOpticalPropertySimpleMicrofacet));
                  it != m.end()) {
                  const auto& p = it->second.as<RSC_MaterialOpticalPropertySimpleMicrofacet>();
                  return !p.emission_texture_reference.empty() && p.emission_intensity > 0;
              }
              return false;
          },
          [](const UnaryPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialOpticalPropertySimpleMicrofacet)) != m.end();
          }}},
    };

    _pair_lut = {
        {"coulomb_friction",
         {[](const PairPropertiesMap& m) -> RS_Any {
              if (auto it = m.find(typeid(RSC_MaterialPairPropertyCoulombFriction));
                  it != m.end()) {
                  const auto& p = it->second.as<RSC_MaterialPairPropertyCoulombFriction>();
                  return std::make_tuple(p.u_static, p.u_dynamic);
              }
              return RS_NONE;
          },
          [](const PairPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPairPropertyCoulombFriction)) != m.end();
          }}},
        {"restitution",
         {[](const PairPropertiesMap& m) -> RS_Any {
              if (auto it = m.find(typeid(RSC_MaterialPairPropertyRestitution)); it != m.end()) {
                  return it->second.as<RSC_MaterialPairPropertyRestitution>().restitution;
              }
              return RS_NONE;
          },
          [](const PairPropertiesMap& m) {
              return m.find(typeid(RSC_MaterialPairPropertyRestitution)) != m.end();
          }}},
    };
}

/*****************************************************************
 * Lifecycle
 *****************************************************************/
void RS_KernelMaterialManager::init(
    RS_KernelManager* kernel_manager,
    float default_u_static,
    float default_u_dynamic,
    float default_restitution
) {
    _kernel_manager = kernel_manager;
    _default_u_static = default_u_static;
    _default_u_dynamic = default_u_dynamic;
    _default_restitution = default_restitution;
    _current_table_dim = 0;
    _kernel_manager->enterKernelDataCriticalZone();
    _initKernelTablesData();
    _kernel_manager->exitKernelDataCriticalZone();
}

void RS_KernelMaterialManager::free() {
    if (_kernel_manager == nullptr) return;

    _kernel_manager->enterKernelDataCriticalZone();
    _freeKernelTablesData();
    _kernel_manager->exitKernelDataCriticalZone();

    // Vision-slot texture references release with the slots.
    _materials.clear();
    _material_registrations.clear();
    _material_use_count.clear();
    _unary_property_name_to_loc.clear();
    _pair_property_name_to_loc.clear();
    _material_unary_properties.clear();
    _material_is_emissive_at_init.clear();
    _pair_properties.clear();
    _vision_slots.clear();
    _dirty_vision_materials.clear();
    _is_tables_dirty = false;
    _pending_table_dim = 0;
    _current_table_dim = 0;
    _kernel_manager = nullptr;
}

/*****************************************************************
 * Material
 *****************************************************************/
void RS_KernelMaterialManager::addMaterial(const RSC_Material& material) {
    if (_material_registrations.count(material.name) > 0) {
        RS_THROW(RS_DuplicateException("Material name '{}' already in use", material.name));
    }

    RS_KernelGidRegistration registration(_kernel_manager, RSE_KERNEL_GID_ESSENCE, material.name);
    const RIndex material_gid = registration.gid();
    _materials[material_gid] = material;
    _material_registrations.emplace(material.name, std::move(registration));
    _material_use_count[material_gid] = 0;

    // Tables grow to fit this gid at the next commit; the vision slot waits for the first
    // reference.
    _pending_table_dim = RS_MAX(_pending_table_dim, static_cast<RSize>(material_gid + 1));
    _is_tables_dirty = true;
}

RIndex RS_KernelMaterialManager::findMaterial(const string& name) const {
    auto it = _material_registrations.find(name);
    return it == _material_registrations.end() ? RS_NULL_INDEX : it->second.gid();
}

RHandle RS_KernelMaterialManager::findVisionMaterialHandle(const string& name) const {
    auto it = _material_registrations.find(name);
    if (it == _material_registrations.end()) return RS_NULL_HANDLE;
    auto sit = _vision_slots.find(it->second.gid());
    if (sit == _vision_slots.end()) return RS_NULL_HANDLE;
    return sit->second.vision_material;
}

const rsc::RSC_Material* RS_KernelMaterialManager::getMaterialByName(const string& name) const {
    auto it = _material_registrations.find(name);
    if (it == _material_registrations.end()) return nullptr;
    auto mit = _materials.find(it->second.gid());
    return mit == _materials.end() ? nullptr : &mit->second;
}

void RS_KernelMaterialManager::removeMaterial(const string& name) {
    auto it = _material_registrations.find(name);
    if (it == _material_registrations.end()) {
        RS_THROW(RS_NotFoundException("Material '{}' not found", name));
    }
    const RIndex material_gid = it->second.gid();

    if (auto uc_it = _material_use_count.find(material_gid);
        uc_it != _material_use_count.end() && uc_it->second > 0) {
        RS_THROW(RS_SimulationException(
            "Cannot remove material '{}': still in use ({} reference(s))", name, uc_it->second
        ));
    }

    // Purge all unary properties on this material (by name).
    for (auto p_it = _unary_property_name_to_loc.begin();
         p_it != _unary_property_name_to_loc.end();) {
        if (p_it->second.first == material_gid) {
            p_it = _unary_property_name_to_loc.erase(p_it);
        } else {
            ++p_it;
        }
    }
    _material_unary_properties.erase(material_gid);

    // Purge all pair properties (by name) involving this gid.
    for (auto p_it = _pair_property_name_to_loc.begin();
         p_it != _pair_property_name_to_loc.end();) {
        const auto& key = p_it->second.first;
        if (key.first == material_gid || key.second == material_gid) {
            p_it = _pair_property_name_to_loc.erase(p_it);
        } else {
            ++p_it;
        }
    }
    for (auto p_it = _pair_properties.begin(); p_it != _pair_properties.end();) {
        if (p_it->first.first == material_gid || p_it->first.second == material_gid) {
            p_it = _pair_properties.erase(p_it);
        } else {
            ++p_it;
        }
    }

    // A slot only exists if the material was referenced then dereferenced; its device data is
    // freed by the next commit (use_count is 0).
    if (_vision_slots.count(material_gid) > 0) {
        _dirty_vision_materials.insert(material_gid);
    }

    _materials.erase(material_gid);
    _material_registrations.erase(it);
    _material_use_count.erase(material_gid);
    _material_is_emissive_at_init.erase(material_gid);
    // Device 2D-table cells for removed gid are stale but harmless; gid is never reissued.
}

void RS_KernelMaterialManager::referenceMaterial(const string& name) {
    auto it = _material_registrations.find(name);
    if (it == _material_registrations.end()) {
        RS_THROW(RS_NotFoundException("Material '{}' not found", name));
    }
    const RIndex material_gid = it->second.gid();
    RSize& uc = _material_use_count[material_gid];
    uc += 1;
    if (uc == 1) {
        // First reference resolves the optical channels host side; the slot is committed later.
        _vision_slots.emplace(material_gid, MaterialVisionSlot{});
        _resolveVisionSlot(material_gid);
        _dirty_vision_materials.insert(material_gid);
    }
}

void RS_KernelMaterialManager::dereferenceMaterial(const string& name) {
    auto it = _material_registrations.find(name);
    if (it == _material_registrations.end()) {
        RS_THROW(RS_NotFoundException("Material '{}' not found", name));
    }
    const RIndex material_gid = it->second.gid();
    auto uc_it = _material_use_count.find(material_gid);
    if (uc_it == _material_use_count.end() || uc_it->second == 0) {
        RS_THROW(
            RS_HostExecutionException("Material '{}' use_count underflow on dereference", name)
        );
    }
    uc_it->second -= 1;
    if (uc_it->second == 0 && _vision_slots.count(material_gid) > 0) {
        // Drop the texture references now so the texture commit sees the lower use_count; the
        // device slot is freed at the next commit.
        auto& slot = _vision_slots.at(material_gid);
        slot.albedo = MaterialVisionTextureReference{};
        slot.roughness_metalness = MaterialVisionTextureReference{};
        slot.emission = MaterialVisionTextureReference{};
        _dirty_vision_materials.insert(material_gid);
    }
}

void RS_KernelMaterialManager::commit() {
    if (!_is_tables_dirty && _dirty_vision_materials.empty()) return;
    _kernel_manager->enterKernelDataCriticalZone();

    if (_is_tables_dirty) {
        _resizeKernelTablesData(_pending_table_dim);
        // Reapply every pair property over the resized tables.
        for (auto& [key, props_map] : _pair_properties) {
            // Note: Structured bindings cannot be captured by a lambda
            // before C++20, so bind to a plain variable first.
            const auto& pair_key = key;
            for (auto& [ti, prop_variant] : props_map) {
                prop_variant.visit([&](auto& p) {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, RSC_MaterialPairPropertyCoulombFriction>) {
                        _updateKernelCoulombFrictionData(pair_key.first, pair_key.second, p);
                    } else if constexpr (std::is_same_v<T, RSC_MaterialPairPropertyRestitution>) {
                        _updateKernelRestitutionData(pair_key.first, pair_key.second, p);
                    }
                });
            }
        }
        _is_tables_dirty = false;
    }

    for (RIndex gid : _dirty_vision_materials) {
        auto slot_it = _vision_slots.find(gid);
        if (slot_it == _vision_slots.end()) continue;
        const RSize uc = _material_use_count.count(gid) ? _material_use_count.at(gid) : 0;
        if (uc > 0) {
            _initKernelVisionSlotData(gid);
        } else {
            _freeKernelVisionSlotData(gid);
            _vision_slots.erase(gid);
        }
    }
    _dirty_vision_materials.clear();

    _kernel_manager->exitKernelDataCriticalZone();
}

RSize RS_KernelMaterialManager::getMaterialUseCount(const string& name) const {
    auto it = _material_registrations.find(name);
    if (it == _material_registrations.end()) {
        RS_THROW(RS_NotFoundException("Material '{}' not found", name));
    }
    auto uc_it = _material_use_count.find(it->second.gid());
    return uc_it == _material_use_count.end() ? 0 : uc_it->second;
}

/*****************************************************************
 * Material property (unary)
 *****************************************************************/
void RS_KernelMaterialManager::addMaterialProperty(const UnaryPropertyVariant& property) {
    property.visit([&](auto& p) {
        using T = std::decay_t<decltype(p)>;
        const auto& prop_name = p.name;
        const auto& mat_ref = p.material_reference;

        auto mat_it = _material_registrations.find(mat_ref);
        if (mat_it == _material_registrations.end()) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Material property '{}' references unknown material '{}'", prop_name, mat_ref
            ));
        }
        const RIndex material_gid = mat_it->second.gid();

        // Reject physical-property add when material is already in use.
        // Optical SimpleMicrofacet is updatable live (vision slot rebuilt on the fly).
        if constexpr (!std::is_same_v<T, RSC_MaterialOpticalPropertySimpleMicrofacet>) {
            const RSize uc =
                _material_use_count.count(material_gid) ? _material_use_count.at(material_gid) : 0;
            if (uc > 0) {
                RS_THROW(RS_SimulationException(
                    "Cannot add physical property '{}' on material '{}': material is already in "
                    "use ({} reference(s))",
                    prop_name,
                    mat_ref,
                    uc
                ));
            }
        }

        if (_unary_property_name_to_loc.count(prop_name) > 0 ||
            _pair_property_name_to_loc.count(prop_name) > 0) {
            RS_THROW(
                RS_DuplicateException("Material property name '{}' already in use", prop_name)
            );
        }

        auto& props_map = _material_unary_properties[material_gid];
        const std::type_index ti{typeid(T)};
        if (props_map.find(ti) != props_map.end()) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Material '{}' already has a property of type '{}'", mat_ref, ti.name()
            ));
        }
        props_map.emplace(ti, property);
        _unary_property_name_to_loc.emplace(prop_name, std::make_pair(material_gid, ti));

        if constexpr (std::is_same_v<T, RSC_MaterialOpticalPropertySimpleMicrofacet>) {
            // Validate texture references now (textures are registered before essences) so the
            // ctx mutation in referenceMaterial can only fail on OOM, never a missing texture.
            auto& tex_mgr = _kernel_manager->getKernelTextureManager();
            const std::string* tex_refs[] = {
                &p.albedo_texture_reference,
                &p.roughness_metalness_texture_reference,
                &p.emission_texture_reference
            };
            for (const std::string* tex_ref_ptr : tex_refs) {
                const std::string& tex_ref = *tex_ref_ptr;
                if (!tex_ref.empty() && !tex_mgr.hasTexture(tex_ref)) {
                    RS_THROW(RS_InvalidCharacterizationException(
                        "Material optical property '{}' references unknown texture '{}'",
                        prop_name,
                        tex_ref
                    ));
                }
            }
            // Re-resolve the vision slot if the material is currently referenced.
            if (_vision_slots.count(material_gid) > 0) {
                _resolveVisionSlot(material_gid);
                _dirty_vision_materials.insert(material_gid);
            }
        }
    });
}

/*****************************************************************
 * Material property (unordered pair)
 *****************************************************************/
void RS_KernelMaterialManager::addMaterialProperty(const PairPropertyVariant& property) {
    property.visit([&](auto& p) {
        using T = std::decay_t<decltype(p)>;
        const auto& prop_name = p.name;
        const auto& first_ref = p.first_material_reference;
        const auto& second_ref = p.second_material_reference;

        auto a_it = _material_registrations.find(first_ref);
        auto b_it = _material_registrations.find(second_ref);
        if (a_it == _material_registrations.end()) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Material pair property '{}' references unknown material '{}'", prop_name, first_ref
            ));
        }
        if (b_it == _material_registrations.end()) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Material pair property '{}' references unknown material '{}'",
                prop_name,
                second_ref
            ));
        }
        const RIndex a_gid = a_it->second.gid();
        const RIndex b_gid = b_it->second.gid();

        // Pair properties are physical; reject if either material is in use.
        const RSize uc_a = _material_use_count.count(a_gid) ? _material_use_count.at(a_gid) : 0;
        const RSize uc_b = _material_use_count.count(b_gid) ? _material_use_count.at(b_gid) : 0;
        if (uc_a > 0 || uc_b > 0) {
            RS_THROW(RS_SimulationException(
                "Cannot add pair property '{}' on materials ('{}', '{}'): in use ({} / {} "
                "reference(s))",
                prop_name,
                first_ref,
                second_ref,
                uc_a,
                uc_b
            ));
        }

        if (_unary_property_name_to_loc.count(prop_name) > 0 ||
            _pair_property_name_to_loc.count(prop_name) > 0) {
            RS_THROW(
                RS_DuplicateException("Material property name '{}' already in use", prop_name)
            );
        }

        const PairKey key = makePairKey(a_gid, b_gid);
        auto& props_map = _pair_properties[key];
        const std::type_index ti{typeid(T)};
        if (props_map.find(ti) != props_map.end()) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Material pair ('{}', '{}') already has a property of type '{}'",
                first_ref,
                second_ref,
                ti.name()
            ));
        }
        props_map.emplace(ti, property);
        _pair_property_name_to_loc.emplace(prop_name, std::make_pair(key, ti));

        // The table cell is written when the next commit() reassembles the tables.
        _is_tables_dirty = true;
    });
}

bool RS_KernelMaterialManager::hasMaterialProperty(const string& property_name) const {
    return _unary_property_name_to_loc.count(property_name) > 0 ||
           _pair_property_name_to_loc.count(property_name) > 0;
}

void RS_KernelMaterialManager::removeMaterialProperty(const string& property_name) {
    if (auto it = _unary_property_name_to_loc.find(property_name);
        it != _unary_property_name_to_loc.end()) {
        const auto [material_gid, ti] = it->second;
        const bool was_optical =
            (ti == std::type_index(typeid(RSC_MaterialOpticalPropertySimpleMicrofacet)));
        // Reject physical-property remove when material is in use.
        if (!was_optical) {
            const RSize uc =
                _material_use_count.count(material_gid) ? _material_use_count.at(material_gid) : 0;
            if (uc > 0) {
                RS_THROW(RS_SimulationException(
                    "Cannot remove physical property '{}': parent material is in use ({} "
                    "reference(s))",
                    property_name,
                    uc
                ));
            }
        }
        auto& props_map = _material_unary_properties.at(material_gid);
        props_map.erase(ti);
        _unary_property_name_to_loc.erase(it);
        if (was_optical && _vision_slots.count(material_gid) > 0) {
            // Re-resolve to the default (black) channels; commit() writes it.
            _resolveVisionSlot(material_gid);
            _dirty_vision_materials.insert(material_gid);
        }
        return;
    }
    if (auto it = _pair_property_name_to_loc.find(property_name);
        it != _pair_property_name_to_loc.end()) {
        const auto [key, ti] = it->second;
        // Pair properties are physical; reject if either material is in use.
        const RSize uc_a = _material_use_count.count(key.first) ? _material_use_count.at(key.first)
                                                                : 0;
        const RSize uc_b =
            _material_use_count.count(key.second) ? _material_use_count.at(key.second) : 0;
        if (uc_a > 0 || uc_b > 0) {
            RS_THROW(RS_SimulationException(
                "Cannot remove pair property '{}': involved materials are in use ({} / {} "
                "reference(s))",
                property_name,
                uc_a,
                uc_b
            ));
        }
        _pair_properties.at(key).erase(ti);
        _pair_property_name_to_loc.erase(it);
        // Device 2D-table cell stays until overwritten by a fresh pair property.
        return;
    }
    RS_THROW(RS_NotFoundException("Material property '{}' not found", property_name));
}

/*****************************************************************
 * Snapshot
 *****************************************************************/
vector<RSC_Material> RS_KernelMaterialManager::getMaterialRSCs() const {
    vector<RSC_Material> out;
    out.reserve(_materials.size());
    for (auto& kv : _materials) out.push_back(kv.second);
    return out;
}

vector<RS_KernelMaterialManager::UnaryPropertyVariant>
RS_KernelMaterialManager::getMaterialUnaryPropertyRSCs() const {
    vector<UnaryPropertyVariant> out;
    for (auto& [_, props_map] : _material_unary_properties) {
        for (auto& [__, prop_variant] : props_map) out.push_back(prop_variant);
    }
    return out;
}

vector<RS_KernelMaterialManager::PairPropertyVariant>
RS_KernelMaterialManager::getMaterialPairPropertyRSCs() const {
    vector<PairPropertyVariant> out;
    for (auto& [_, props_map] : _pair_properties) {
        for (auto& [__, prop_variant] : props_map) out.push_back(prop_variant);
    }
    return out;
}

/*****************************************************************
 * Query / has
 *****************************************************************/
RS_Any RS_KernelMaterialManager::query(const string& material, const string& property) const {
    auto lut_it = _unary_lut.find(property);
    if (lut_it == _unary_lut.end()) {
        RS_THROW(RS_InvalidArgumentException("Unknown unary property '{}'", property));
    }
    auto gid_it = _material_registrations.find(material);
    if (gid_it == _material_registrations.end()) return RS_NONE;
    auto props_it = _material_unary_properties.find(gid_it->second.gid());
    if (props_it == _material_unary_properties.end()) return RS_NONE;
    return lut_it->second.query_fn(props_it->second);
}

RS_Any RS_KernelMaterialManager::query(
    const string& first_material,
    const string& second_material,
    const string& property
) const {
    auto lut_it = _pair_lut.find(property);
    if (lut_it == _pair_lut.end()) {
        RS_THROW(RS_InvalidArgumentException("Unknown pair property '{}'", property));
    }
    auto a_it = _material_registrations.find(first_material);
    auto b_it = _material_registrations.find(second_material);
    if (a_it == _material_registrations.end() || b_it == _material_registrations.end()) {
        return RS_NONE;
    }
    auto props_it = _pair_properties.find(makePairKey(a_it->second.gid(), b_it->second.gid()));
    if (props_it == _pair_properties.end()) return RS_NONE;
    return lut_it->second.query_fn(props_it->second);
}

bool RS_KernelMaterialManager::has(const string& material, const string& property) const {
    auto lut_it = _unary_lut.find(property);
    if (lut_it == _unary_lut.end()) {
        RS_THROW(RS_InvalidArgumentException("Unknown unary property '{}'", property));
    }
    auto gid_it = _material_registrations.find(material);
    if (gid_it == _material_registrations.end()) return false;
    auto props_it = _material_unary_properties.find(gid_it->second.gid());
    if (props_it == _material_unary_properties.end()) return false;
    return lut_it->second.has_fn(props_it->second);
}

bool RS_KernelMaterialManager::has(
    const string& first_material,
    const string& second_material,
    const string& property
) const {
    auto lut_it = _pair_lut.find(property);
    if (lut_it == _pair_lut.end()) {
        RS_THROW(RS_InvalidArgumentException("Unknown pair property '{}'", property));
    }
    auto a_it = _material_registrations.find(first_material);
    auto b_it = _material_registrations.find(second_material);
    if (a_it == _material_registrations.end() || b_it == _material_registrations.end())
        return false;
    auto props_it = _pair_properties.find(makePairKey(a_it->second.gid(), b_it->second.gid()));
    if (props_it == _pair_properties.end()) return false;
    return lut_it->second.has_fn(props_it->second);
}

/*****************************************************************
 * Vision slot
 *****************************************************************/
void RS_KernelMaterialManager::_resolveVisionSlot(RIndex material_gid) {
    auto slot_it = _vision_slots.find(material_gid);
    if (slot_it == _vision_slots.end()) return;
    auto& slot = slot_it->second;

    // Resolve current optical property (may be absent).
    const RSC_MaterialOpticalPropertySimpleMicrofacet* optical = nullptr;
    if (auto pit = _material_unary_properties.find(material_gid);
        pit != _material_unary_properties.end()) {
        if (auto oit = pit->second.find(typeid(RSC_MaterialOpticalPropertySimpleMicrofacet));
            oit != pit->second.end()) {
            optical = &oit->second.as<RSC_MaterialOpticalPropertySimpleMicrofacet>();
        }
    }

    // Forbid post-init changes to emissive status. Structure managers cache their emissive
    // voxel set at structure init and would not observe a flip here.
    const bool current_is_emissive = optical != nullptr &&
                                     !optical->emission_texture_reference.empty() &&
                                     optical->emission_intensity > 0;
    auto init_it = _material_is_emissive_at_init.find(material_gid);
    if (init_it == _material_is_emissive_at_init.end()) {
        _material_is_emissive_at_init.emplace(material_gid, current_is_emissive);
    } else if (init_it->second != current_is_emissive) {
        RS_THROW(RS_SimulationException(
            "Cannot change emissive status of material gid {} after first use", material_gid
        ));
    }

    // Reassigning each channel releases the previously-held texture reference.
    if (optical != nullptr) {
        slot.albedo = _resolveVisionSampleInput(
            optical->albedo_texture_reference,
            RColor3f::zero(),
            RSE_TEXTURE_COLOR_SPACE_SRGB,
            RSE_TEXTURE_STORAGE_FORMAT_CONSTANT
        );
        slot.roughness_metalness = _resolveVisionSampleInput(
            optical->roughness_metalness_texture_reference,
            RColor3f(1.0f, 0.0f, 0.0f),
            RSE_TEXTURE_COLOR_SPACE_LINEAR,
            RSE_TEXTURE_STORAGE_FORMAT_CONSTANT
        );
        slot.emission = _resolveVisionSampleInput(
            optical->emission_texture_reference,
            RColor3f::zero(),
            RSE_TEXTURE_COLOR_SPACE_SRGB,
            RSE_TEXTURE_STORAGE_FORMAT_CONSTANT
        );
    } else {
        slot.albedo = MaterialVisionTextureReference{};
        slot.roughness_metalness = MaterialVisionTextureReference{
            RSE_VISION_SAMPLE_INPUT_CONSTANT, RColor3f(1.0f, 0.0f, 0.0f), {}
        };
        slot.emission = MaterialVisionTextureReference{};
    }
}

RS_KernelMaterialManager::MaterialVisionTextureReference
RS_KernelMaterialManager::_resolveVisionSampleInput(
    const std::string& tex_ref,
    RColor3f default_constant,
    RSE_TextureColorSpace color_space,
    RSE_TextureStorageFormat storage_format
) {
    MaterialVisionTextureReference channel;
    if (tex_ref.empty()) {
        channel.input_type = RSE_VISION_SAMPLE_INPUT_CONSTANT;
        channel.constant_value = default_constant;
        return channel;
    }
    auto& tex_mgr = _kernel_manager->getKernelTextureManager();
    // Always reference (constants too) so texture removal stays guarded.
    channel.texture_reference =
        RS_KernelTextureReference(_kernel_manager, tex_ref, color_space, storage_format);
    if (tex_mgr.isConstant(tex_ref)) {
        channel.input_type = RSE_VISION_SAMPLE_INPUT_CONSTANT;
        channel.constant_value = tex_mgr.getConstant(tex_ref);
        return channel;
    }
    switch (tex_mgr.getTextureStorageFormat(tex_ref)) {
        case RSE_TEXTURE_STORAGE_FORMAT_GRID:
            channel.input_type = RSE_VISION_SAMPLE_INPUT_GRID;
            break;
        default:
            channel.input_type = RSE_VISION_SAMPLE_INPUT_CONSTANT;
            channel.constant_value = default_constant;
            break;
    }
    return channel;
}

void RS_KernelMaterialManager::_initKernelVisionSlotData(RIndex material_gid) {
    auto slot_it = _vision_slots.find(material_gid);
    if (slot_it == _vision_slots.end()) return;
    auto& slot = slot_it->second;

    const RSC_MaterialOpticalPropertySimpleMicrofacet* optical = nullptr;
    if (auto pit = _material_unary_properties.find(material_gid);
        pit != _material_unary_properties.end()) {
        if (auto oit = pit->second.find(typeid(RSC_MaterialOpticalPropertySimpleMicrofacet));
            oit != pit->second.end()) {
            optical = &oit->second.as<RSC_MaterialOpticalPropertySimpleMicrofacet>();
        }
    }

    // Read each channel's device payload host side; grid handles resolve by name post texture
    // commit.
    auto& tex_mgr = _kernel_manager->getKernelTextureManager();
    RS_VisionSampleInputMem<RColor3f> albedo =
        slot.albedo.input_type == RSE_VISION_SAMPLE_INPUT_CONSTANT
            ? RS_VisionSampleInputMem<RColor3f>(slot.albedo.constant_value)
            : RS_VisionSampleInputMem<RColor3f>(
                  tex_mgr.findTextureHandle(slot.albedo.texture_reference.name())
              );
    RS_VisionSampleInputMem<RColor3f> roughness_metalness =
        slot.roughness_metalness.input_type == RSE_VISION_SAMPLE_INPUT_CONSTANT
            ? RS_VisionSampleInputMem<RColor3f>(slot.roughness_metalness.constant_value)
            : RS_VisionSampleInputMem<RColor3f>(
                  tex_mgr.findTextureHandle(slot.roughness_metalness.texture_reference.name())
              );
    RS_VisionSampleInputMem<RColor3f> emission_color =
        slot.emission.input_type == RSE_VISION_SAMPLE_INPUT_CONSTANT
            ? RS_VisionSampleInputMem<RColor3f>(slot.emission.constant_value)
            : RS_VisionSampleInputMem<RColor3f>(
                  tex_mgr.findTextureHandle(slot.emission.texture_reference.name())
              );

    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;
        using ContextType = decltype(k_ptr->ctx);
        using FloatM = typename RS_VisionMaterialMem<ContextType::FloatMemAccuracy>::FloatM;

        // Allocate the device slot on first commit; the pooled offset is stable afterwards.
        if (slot.vision_material == RS_NULL_HANDLE) {
            ctx.host_data->vision_material_layout.add(1, material_gid);
            ctx.host_data->vision_material_layout.syncDeviceLayout(ctx.vision_material_layout);
            ctx.vision_materials.resize(ctx.host_data->vision_material_layout.capacity());
            slot.vision_material =
                static_cast<RHandle>(ctx.host_data->vision_material_layout.segmentOffset(
                    ctx.host_data->vision_material_layout.find(material_gid)
                ));
        }

        RS_VisionMaterialMem<ContextType::FloatMemAccuracy> vm{};
        vm.material_gid = material_gid;
        vm.albedo = albedo;
        vm.roughness_metalness = roughness_metalness;
        vm.emission_color = emission_color;

        if (optical != nullptr) {
            vm.ior = static_cast<FloatM>(optical->ior);
            vm.emission_intensity = static_cast<FloatM>(optical->emission_intensity);
        }

        // Emission precedence over transparency over opaque, because
        // emission required special data flow in the vision system.
        RSE_VisionMaterialType inferred_type;
        if (optical != nullptr && optical->emission_intensity > 0) {
            inferred_type = RSE_VISION_MATERIAL_EMISSIVE;
        } else if (optical != nullptr && optical->is_transparent) {
            inferred_type = RSE_VISION_MATERIAL_TRANSPARENT;
        } else {
            inferred_type = RSE_VISION_MATERIAL_OPAQUE;
        }

        RS_VisionMaterialStateUnion state_union{0};
        state_union.vision_material_state.render_type = inferred_type;
        state_union.vision_material_state.albedo_input_type = slot.albedo.input_type;
        state_union.vision_material_state.roughness_metalness_input_type =
            slot.roughness_metalness.input_type;
        state_union.vision_material_state.emission_input_type = slot.emission.input_type;
        vm.state = state_union.vision_material_state_packed;

        const RSize offset = ctx.host_data->vision_material_layout.segmentOffset(
            ctx.host_data->vision_material_layout.find(material_gid)
        );
        ctx.vision_materials.writeSlice(&vm, 1, offset);

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelMaterialManager::_freeKernelVisionSlotData(RIndex material_gid) {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;
        ctx.host_data->vision_material_layout.remove(
            ctx.host_data->vision_material_layout.find(material_gid)
        );
        ctx.host_data->vision_material_layout.syncDeviceLayout(ctx.vision_material_layout);
        ctx.vision_materials.resize(ctx.host_data->vision_material_layout.capacity());

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

/*****************************************************************
 * Pair tables
 *****************************************************************/
void RS_KernelMaterialManager::_initKernelTablesData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto* device_provider = _kernel_manager->getDeviceProvider();
        auto& ctx = k_ptr->ctx;
        ctx.material_friction_static_table.init(device_provider);
        ctx.material_friction_dynamic_table.init(device_provider);
        ctx.material_restitution_table.init(device_provider);
        ctx.vision_materials.init(0, device_provider);
        ctx.vision_material_layout.init(device_provider);
    });
}

void RS_KernelMaterialManager::_freeKernelTablesData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;
        ctx.material_friction_static_table.free();
        ctx.material_friction_dynamic_table.free();
        ctx.material_restitution_table.free();
        ctx.vision_materials.free();
        ctx.vision_material_layout.free();

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelMaterialManager::_resizeKernelTablesData(RSize material_count) {
    if (material_count <= _current_table_dim) return;

    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto* device_provider = _kernel_manager->getDeviceProvider();
        auto& ctx = k_ptr->ctx;

        RSize old_dim = _current_table_dim;
        RSize new_dim = material_count;

        auto resize_table = [&](RS_Array<float, 2>& table, float default_value) {
            RS_Array<float, 2> new_table(device_provider);
            new_table.resize({new_dim, new_dim});
            new_table.fill(default_value, false);
            if (old_dim > 0) {
                copy2D(new_table, table, old_dim, RS_AUTO_BLOCK_SIZE, false);
            }
            device_provider->sync();
            table.free();
            table = new_table;
        };

        resize_table(ctx.material_friction_static_table, _default_u_static);
        resize_table(ctx.material_friction_dynamic_table, _default_u_dynamic);
        resize_table(ctx.material_restitution_table, _default_restitution);

        _current_table_dim = new_dim;

        // Context data is modified, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelMaterialManager::_updateKernelCoulombFrictionData(
    RIndex gid_a,
    RIndex gid_b,
    const rsc::RSC_MaterialPairPropertyCoulombFriction& cfg
) {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto* device_provider = _kernel_manager->getDeviceProvider();
        auto& ctx = k_ptr->ctx;
        const float us = static_cast<float>(cfg.u_static);
        const float ud = static_cast<float>(cfg.u_dynamic);

        _kernel_manager->syncKernelIfNeeded();
        castProviderPtrAndCall(device_provider, [&](auto p_ptr) {
            p_ptr->template run<RSCF_SetMaterialFriction>(
                {1, 1, true},
                ctx.material_friction_static_table,
                ctx.material_friction_dynamic_table,
                gid_a,
                gid_b,
                us,
                ud
            );
        });
    });
}

void RS_KernelMaterialManager::_updateKernelRestitutionData(
    RIndex gid_a,
    RIndex gid_b,
    const rsc::RSC_MaterialPairPropertyRestitution& cfg
) {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto* device_provider = _kernel_manager->getDeviceProvider();
        auto& ctx = k_ptr->ctx;
        const float r = static_cast<float>(cfg.restitution);

        _kernel_manager->syncKernelIfNeeded();
        castProviderPtrAndCall(device_provider, [&](auto p_ptr) {
            p_ptr->template run<RSCF_SetMaterialRestitution>(
                {1, 1, true}, ctx.material_restitution_table, gid_a, gid_b, r
            );
        });
    });
}

/*****************************************************************
 * RS_KernelMaterialReference
 *****************************************************************/
RS_KernelMaterialReference::RS_KernelMaterialReference(
    RS_KernelManager* kernel_manager,
    std::string name
)
    : _kernel_manager(kernel_manager), _name(std::move(name)) {
    _kernel_manager->getKernelMaterialManager().referenceMaterial(_name);
}

RS_KernelMaterialReference::~RS_KernelMaterialReference() { _release(); }

RS_KernelMaterialReference::RS_KernelMaterialReference(RS_KernelMaterialReference&& other) noexcept
    : _kernel_manager(other._kernel_manager), _name(std::move(other._name)) {
    other._kernel_manager = nullptr;
}

RS_KernelMaterialReference& RS_KernelMaterialReference::operator=(
    RS_KernelMaterialReference&& other
) noexcept {
    if (this != &other) {
        _release();
        _kernel_manager = other._kernel_manager;
        _name = std::move(other._name);
        other._kernel_manager = nullptr;
    }
    return *this;
}

void RS_KernelMaterialReference::_release() {
    if (_kernel_manager == nullptr) return;
    _kernel_manager->getKernelMaterialManager().dereferenceMaterial(_name);
    _kernel_manager = nullptr;
}

}  // namespace rise
