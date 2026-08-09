// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_MATERIAL_MANAGER_H
#define RS_KERNEL_MATERIAL_MANAGER_H

#include <functional>
#include <map>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rise_origin/core/kernel_manager/rs_kernel_gid_registration.h"
#include "rise_origin/core/kernel_manager/rs_kernel_texture_manager.h"
#include "rise_origin/logic/vision/rs_vision_material_m.h"
#include "rise_origin/rsc/essence/rsc_material.h"
#include "rise_origin/rsc/essence/rsc_material_pair_property.h"
#include "rise_origin/rsc/essence/rsc_material_unary_property.h"
#include "rise_origin/utils/data/rs_any.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

class RS_KernelManager;

/**
 * Query and return types:
 *
 * Unary Properties:
 * - "density" :
 *     Return type: Union[RFloat, RS_NoneType]
 *     Detail: Density of the material (unit: kg/m^3), sourced from the rigid or
 *             soft physical property when present, return RS_NONE if not defined
 * - "fixed" :
 *     Return type: Union[bool, RS_NoneType]
 *     Detail: Whether the material is fixed (i.e. never changes position or orientation),
 *             return RS_NONE if not defined
 * - "rigid" :
 *     Return type: Union[bool, RS_NoneType]
 *     Detail: Whether the material is rigid (i.e. mass and inertia are computed
 *             from the voxel composition), return RS_NONE if not defined
 * - "atlas" :
 *     Return type: Union[bool, RS_NoneType]
 *     Detail: Whether the material is an atlas rigid body (prescribed mass, COM
 *             and inertia), return RS_NONE if not defined
 * - "soft" :
 *     Return type: Union[RS_Tuple[RFloat, RFloat, RFloat, RFloat, RFloat], RS_NoneType]
 *     Detail: A tuple containing:
 *             0) min_expansion:
 *                Range: [-1, 0],
 *                Detail: Minimum expansion allowed by the material,
 *                        0 means no contraction is allowed and -1 means min contraction
 *                        ratio is -100%
 *             1) max_expansion:
 *                Range: [0, 1],
 *                Detail: Maximum expansion allowed by the material,
 *                        0 means no expansion is allowed and 1 means max expansion
 *                        ratio is 100%
 *             2) elastic_modulus:
 *                Range: [0, inf),
 *                Detail: Elastic modulus of the material (unit: Pa),
 *             3) fail_stress:
 *                Range: [0, inf),
 *                Detail: Maximum failure stress of the material (unit: N/m^2),
 *             4) poissons_ratio:
 *                Range: [0, 0.5],
 *                Detail: Poisson's ratio of the material,
 *             return RS_NONE if not defined
 * - "mass" :
 *     Return type: Union[RFloat, RS_NoneType]
 *     Detail: Mass of the atlas rigid body (unit: kg), return RS_NONE if not defined
 * - "com_position" :
 *     Return type: Union[RVec3rf, RS_NoneType]
 *     Detail: Center of mass of the atlas rigid body in the rigid-body segment's local frame
 *             (origin at the segment's minimum bounding-box corner, body-local axes, unit: m),
 *             return RS_NONE if not defined
 * - "inertia" :
 *     Return type: Union[RMat3rf, RS_NoneType]
 *     Detail: 3x3 moment of inertia tensor of the atlas rigid body about its center of mass,
 *             in body-local axes (unit: kg m^2), return RS_NONE if not defined
 * - "is_transparent" :
 *     Return type: bool
 *     Detail: True iff the material's optical SimpleMicrofacet property is transparent,
 *             return false if the optical property is absent
 * - "is_emissive" :
 *     Return type: bool
 *     Detail: True iff the material's optical SimpleMicrofacet property has both a
 *             populated emission texture reference and a positive emission_radiance,
 *             return false if the optical property is absent
 *
 * Unordered Pair Properties:
 * - "coulomb_friction" :
 *     Return type: Union[RS_Tuple[RFloat, RFloat], RS_NoneType]
 *     Detail: A tuple containing:
 *             0) u_static:
 *                Range: [0, 1],
 *                Detail: Static friction coefficient,
 *             1) u_dynamic:
 *                Range: [0, 1],
 *                Detail: Dynamic friction coefficient,
 *             return RS_NONE if not defined
 * - "restitution" :
 *     Return type: Union[RFloat, RS_NoneType]
 *     Detail: Coefficient of restitution (0 = perfectly inelastic, 1 = perfectly
 *             elastic), return RS_NONE if not defined
 */

/**
 * @brief Singular manager for materials and their properties.
 *
 * @details Owns materials, per-material unary properties (physical + optical),
 * and unordered-pair properties (friction, restitution). Provides name-keyed
 * add/remove/find and composer-style query/has over a fixed property-name
 * vocabulary. Internally manages vision-material device slots (one per
 * material) for surface rendering and the 2D GPU lookup tables for Coulomb
 * friction and restitution.
 */
class RS_KernelMaterialManager {
   public:
    using UnaryPropertyVariant = RS_Variant<
        rsc::RSC_MaterialPhysicalPropertyFixed,
        rsc::RSC_MaterialPhysicalPropertyRigid,
        rsc::RSC_MaterialPhysicalPropertyAtlas,
        rsc::RSC_MaterialPhysicalPropertySoft,
        rsc::RSC_MaterialOpticalPropertySimpleMicrofacet>;

    using PairPropertyVariant = rsc::RSC_MaterialPairPropertyVariant;

    /**
     * @brief Construct an empty material manager. Property-name LUTs are
     *        initialized here; init() must still be called before use.
     */
    RS_KernelMaterialManager();

    /**
     * @brief Bind to the kernel manager and seed the default pair-property
     *        defaults (Coulomb friction, restitution) used when no per-pair
     *        override is registered.
     */
    void init(
        RS_KernelManager *kernel_manager,
        float default_u_static = 0.5f,
        float default_u_dynamic = 0.5f,
        float default_restitution = 0.5f
    );

    /**
     * @brief Release all material vision slots and friction/restitution GPU
     *        tables. Characterizations are dropped.
     */
    void free();

    /**
     * @brief Add a material.
     */
    void addMaterial(const rsc::RSC_Material &material);

    /**
     * @brief Find a material's essence gid by name.
     * @return The essence gid of the material, RS_NULL_INDEX if not found.
     */
    RIndex findMaterial(const std::string &name) const;

    /**
     * @brief Resolve a material optical property's vision-material array handle
     *        (offset into ctx.vision_materials).
     * @param name Material name.
     * @param material_optical_name Optical property name, empty selects the
     *        material's first optical property.
     * @return The vision-material array handle, RS_NULL_HANDLE if the material or
     *         optical property is unknown or not referenced.
     */
    RHandle findVisionMaterialHandle(
        const std::string &name,
        const std::string &material_optical_name = ""
    ) const;

    /**
     * @brief Remove a material.
     */
    void removeMaterial(const std::string &name);

    /**
     * @brief Increment material use_count; first reference resolves the optical-property textures
     *        host side and marks the vision slot dirty for the next commit().
     */
    void referenceMaterial(const std::string &name);

    /**
     * @brief Decrement material use_count; last dereference drops the texture references and marks
     *        the vision slot dirty so the next commit() frees it.
     */
    void dereferenceMaterial(const std::string &name);

    /**
     * @brief Reassemble the dirty friction/restitution tables and reconcile the dirty vision slots
     *        against host state. Run after the texture manager commit() so grid handles resolve.
     */
    void commit();

    /**
     * @brief Get the use count of a material.
     */
    RSize getMaterialUseCount(const std::string &name) const;

    /**
     * @brief Add a material property.
     */
    void addMaterialProperty(const UnaryPropertyVariant &property);

    /**
     * @brief Add a material property.
     */
    void addMaterialProperty(const PairPropertyVariant &property);

    /**
     * @brief Check if a material property exists.
     */
    bool hasMaterialProperty(const std::string &property_name) const;

    /**
     * @brief Check whether a unary material property belongs to a material.
     * @param material_name Material name.
     * @param material_property_name Unary property name.
     * @return True if the property exists and its parent material is material_name.
     */
    bool isMaterialPropertyBelongTo(
        const std::string &material_name,
        const std::string &material_property_name
    ) const;

    /**
     * @brief Remove a material property.
     */
    void removeMaterialProperty(const std::string &property_name);

    /**
     * @brief Get a material characterization by name.
     * @return The material, nullptr if not found.
     */
    const rsc::RSC_Material *getMaterialRSCByName(const std::string &name) const;

    /**
     * @brief Get a unary property characterization by name (variant-wrapped).
     * @return The unary property, nullptr if not found.
     */
    const UnaryPropertyVariant *getMaterialUnaryPropertyRSCByName(const std::string &name) const;

    /**
     * @brief Get a pair property characterization by name (variant-wrapped).
     * @return The pair property, nullptr if not found.
     */
    const PairPropertyVariant *getMaterialPairPropertyRSCByName(const std::string &name) const;

    /**
     * @brief Get the snapshot of every registered material characterization, for getWorld().
     */
    std::vector<rsc::RSC_Material> getMaterialRSCs() const;

    /**
     * @brief Get the snapshot of every registered unary property characterization
     * (variant-wrapped), for getWorld().
     */
    std::vector<UnaryPropertyVariant> getMaterialUnaryPropertyRSCs() const;

    /**
     * @brief Get the snapshot of every registered pair property characterization (variant-wrapped),
     * for getWorld().
     */
    std::vector<PairPropertyVariant> getMaterialPairPropertyRSCs() const;

    /**
     * @brief Query a material property.
     */
    RS_Any query(const std::string &material, const std::string &property) const;

    /**
     * @brief Query a pair property.
     */
    RS_Any query(
        const std::string &first_material,
        const std::string &second_material,
        const std::string &property
    ) const;

    /**
     * @brief Check if a material property exists.
     */
    bool has(const std::string &material, const std::string &property) const;

    /**
     * @brief Check if a pair property exists.
     */
    bool has(
        const std::string &first_material,
        const std::string &second_material,
        const std::string &property
    ) const;

   private:
    /// Per-type unary property map, a type with several instances keeps them in registration order.
    using UnaryPropertiesMap =
        std::unordered_map<std::type_index, std::vector<UnaryPropertyVariant>>;
    /// Per-type pair property map.
    using PairPropertiesMap = std::unordered_map<std::type_index, PairPropertyVariant>;

    using PairKey = std::pair<RIndex, RIndex>;
    static PairKey makePairKey(RIndex a, RIndex b) { return {std::min(a, b), std::max(a, b)}; }
    struct PairKeyHash {
        std::size_t operator()(const PairKey &p) const noexcept {
            return std::hash<RIndex>{}(p.first) ^ (std::hash<RIndex>{}(p.second) << 1);
        }
    };

    struct UnaryLutEntry {
        std::function<RS_Any(const UnaryPropertiesMap &)> query_fn;
        std::function<bool(const UnaryPropertiesMap &)> has_fn;
    };
    struct PairLutEntry {
        std::function<RS_Any(const PairPropertiesMap &)> query_fn;
        std::function<bool(const PairPropertiesMap &)> has_fn;
    };

    /// One optical channel resolved host side; holds its texture reference.
    struct MaterialVisionTextureReference {
        logic::RSE_VisionSampleInputType input_type = logic::RSE_VISION_SAMPLE_INPUT_CONSTANT;
        RColor3f constant_value = RColor3f::zero();
        RS_KernelTextureReference texture_reference;
    };

    /// One optical property's channels resolved host side.
    struct MaterialVisionOptical {
        MaterialVisionTextureReference albedo;
        MaterialVisionTextureReference roughness_metalness;
        MaterialVisionTextureReference emission;
    };

    struct MaterialVisionSlot {
        RHandle vision_material_base = RS_NULL_HANDLE;
        RSize optical_count = 0;
        std::vector<MaterialVisionOptical> opticals;
    };

    RS_KernelManager *_kernel_manager = nullptr;

    std::unordered_map<std::string, UnaryLutEntry> _unary_lut;
    std::unordered_map<std::string, PairLutEntry> _pair_lut;

    RIndex _next_gid = 0;
    std::unordered_map<RIndex, rsc::RSC_Material> _materials;
    /// Held essence gid registration per material, keyed by name.
    std::unordered_map<std::string, RS_KernelGidRegistration> _material_registrations;
    /// Per-material live use_count keyed by gid (0 when no live consumer holds it).
    std::unordered_map<RIndex, RSize> _material_use_count;

    std::unordered_map<std::string, std::pair<RIndex, std::type_index>> _unary_property_name_to_loc;
    std::unordered_map<std::string, std::pair<PairKey, std::type_index>> _pair_property_name_to_loc;

    std::unordered_map<RIndex, UnaryPropertiesMap> _material_unary_properties;
    std::unordered_map<PairKey, PairPropertiesMap, PairKeyHash> _material_pair_properties;

    std::unordered_map<RIndex, MaterialVisionSlot> _vision_slots;

    /// Emissive status observed at first vision-slot population per material.
    /// Subsequent rebuilds throw if the new emissive status differs, since
    /// structure managers cache their emissive voxel set at init and would
    /// otherwise miss the change.
    std::unordered_map<RIndex, bool> _material_is_emissive_at_init;

    /// Material gids whose vision slot the next commit() must reconcile.
    std::unordered_set<RIndex> _dirty_vision_materials;
    /// Whether the next commit() must reassemble the friction/restitution tables.
    bool _is_tables_dirty = false;
    /// Target table dimension for the next reassemble.
    RSize _pending_table_dim = 0;

    RSize _current_table_dim = 0;
    float _default_u_static = 0.5f;
    float _default_u_dynamic = 0.5f;
    float _default_restitution = 0.5f;

    void _createLutTables();

    /**
     * @brief Resolve the material's optical channels host side into its vision slot.
     */
    void _resolveVisionSlot(RIndex material_gid);

    /**
     * @brief Resolve one optical-channel texture reference host side, taking a texture reference
     *        when tex_ref is non-empty.
     * @param tex_ref Texture resource name (may be empty).
     * @param default_constant Fallback constant when ref is empty.
     * @param color_space sRGB or linear decode at load time.
     * @param storage_format Per-field storage format policy.
     */
    MaterialVisionTextureReference _resolveVisionSampleInput(
        const std::string &tex_ref,
        RColor3f default_constant,
        RSE_TextureColorSpace color_space,
        RSE_TextureStorageFormat storage_format
    );

    void _initKernelVisionSlotData(RIndex material_gid);
    void _freeKernelVisionSlotData(RIndex material_gid);

    void _initKernelTablesData();
    void _freeKernelTablesData();
    void _resizeKernelTablesData(RSize material_count);
    void _updateKernelCoulombFrictionData(
        RIndex gid_a,
        RIndex gid_b,
        const rsc::RSC_MaterialPairPropertyCoulombFriction &cfg
    );
    void _updateKernelRestitutionData(
        RIndex gid_a,
        RIndex gid_b,
        const rsc::RSC_MaterialPairPropertyRestitution &cfg
    );
};

/**
 * @brief References a material on construction and dereferences it on destruction.
 */
class RS_KernelMaterialReference {
   public:
    RS_KernelMaterialReference() = default;
    RS_KernelMaterialReference(RS_KernelManager *kernel_manager, std::string name);
    ~RS_KernelMaterialReference();

    RS_KernelMaterialReference(RS_KernelMaterialReference &&other) noexcept;
    RS_KernelMaterialReference &operator=(RS_KernelMaterialReference &&other) noexcept;
    RS_KernelMaterialReference(const RS_KernelMaterialReference &) = delete;
    RS_KernelMaterialReference &operator=(const RS_KernelMaterialReference &) = delete;

    [[nodiscard]] const std::string &name() const { return _name; }

   private:
    void _release();

    RS_KernelManager *_kernel_manager = nullptr;
    std::string _name;
};

}  // namespace rise

#endif  // RS_KERNEL_MATERIAL_MANAGER_H
