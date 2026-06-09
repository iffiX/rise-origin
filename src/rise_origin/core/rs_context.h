// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONTEXT_H
#define RS_CONTEXT_H

#include <unordered_map>
#include <vector>

#include "rise_origin/core/rs_simulation_record.h"
#include "rise_origin/device/data/rs_array_tree.h"
#include "rise_origin/device/data/rs_buffer.h"
#include "rise_origin/device/data/rs_bvh.h"
#include "rise_origin/device/data/rs_soa.h"
#include "rise_origin/device/data/rs_sov.h"
#include "rise_origin/device/data/rs_sparse.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/logic/mesh/rs_mesh_triangle_m.h"
#include "rise_origin/logic/rigid/rs_constraint_m.h"
#include "rise_origin/logic/rigid/rs_contact_m.h"
#include "rise_origin/logic/rigid/rs_rigid_body_m.h"
#include "rise_origin/logic/rigid/rs_rigid_body_surface_voxel_m.h"
#include "rise_origin/logic/signal/rs_force_m.h"
#include "rise_origin/logic/signal/rs_torque_m.h"
#include "rise_origin/logic/soft/rs_link_m.h"
#include "rise_origin/logic/soft/rs_link_material_m.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"
#include "rise_origin/logic/soft/rs_voxel_material_m.h"
#include "rise_origin/logic/vision/rs_camera_m.h"
#include "rise_origin/logic/vision/rs_vision_material_m.h"
#include "rise_origin/utils/rs_color.h"
#include "rs_layout.h"

namespace rise {

using RS_VoxelBVH = device::RS_BVHBase<
    RBigIndex,
    logic::RS_VoxelBVHObjectIdentifier,
    logic::RS_VoxelBVHObjectIdentifierHostHasher>;

struct RS_ArrayFloorContext {
    // In RSC.Simulation.Floor
    RFloat x_size = 0;
    RFloat y_size = 0;
    RSize x_values = 0;  // cell count after upsampling
    RSize y_values = 0;
    RFloat inv_cell_x = 0;  // x_values / x_size, precomputed
    RFloat inv_cell_y = 0;
    bool is_x_repeat = false;
    bool is_y_repeat = false;
    RFloat height_min = 0;  // min z over heights[], for finite-z slab of array-walk repeat
    RFloat height_max = 0;  // max z over heights[], same purpose
    device::RS_DivMod<RSize> x_divmod;
    device::RS_DivMod<RSize> y_divmod;
    RSize x_wrap_offset = 0;
    RSize y_wrap_offset = 0;
    device::RS_Array<RFloat, 2> heights;  // absolute Z in meters, shape {y_values, x_values}
    device::RS_Array<RVec3f, 2> normals;  // unit normal per cell, shape {y_values, x_values}
};

struct RS_ContextHostData {
    // Structure context data layouts
    RS_PooledLayout structure_link_material_layout;
    RS_PooledLayout structure_voxel_material_layout;
    RS_PooledLayout structure_link_layout;
    RS_PooledLayout structure_voxel_layout;
    RS_PooledLayout structure_rigid_body_layout;
    RS_PooledLayout structure_rigid_body_surface_voxel_layout;
    RS_PooledLayout structure_constraint_joint_layout;

    RS_CompactLayout structure_constraint_joint_block_range_layout;

    // Structure signal layouts
    RS_PooledLayout structure_expansion_signal_layout;
    RS_PooledLayout structure_hinge_signal_layout;
    RS_PooledLayout structure_prismatic_signal_layout;

    // Structure record layouts
    RS_CompactLayout structure_record_layout;
    RS_CompactLayout structure_voxel_record_layout;
    RS_CompactLayout structure_link_record_layout;
    RS_CompactLayout structure_rigid_body_record_layout;
    // For joints, because number doesn't match between simulated elements
    // and recorded elements, need two layouts
    RS_CompactLayout structure_joint_record_layout;
    RS_CompactLayout structure_constraint_joint_record_layout;

    // Structure rigid body surface voxel to rigid body reduction layout
    RS_ReductionLayout structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout;

    RS_CompactLayout camera_layout;
    RS_CompactLayout camera_info_ray_layout;
    RS_CompactLayout camera_rgb_ray_layout;
    RS_CompactLayout camera_record_layout;
    RS_CompactLayout camera_pixel_record_layout;
    RS_CompactLayout camera_state_record_layout;

    // Camera signal layout, note that all signals are packed together
    RS_PooledLayout camera_control_signal_layout;

    RS_PooledLayout vision_material_layout;

    // Force signal layout
    RS_PooledLayout force_signal_layout;
    RS_CompactLayout force_rigid_layout;
    RS_CompactLayout force_soft_layout;

    // Torque signal layout (rigid bodies only)
    RS_PooledLayout torque_signal_layout;
    RS_CompactLayout torque_rigid_layout;

    /// Per-structure host map of emissive voxel handles, 
    /// maps structure_gid -> emissive voxel handles
    std::unordered_map<RIndex, std::vector<RHandle>> structure_emissive_voxel_handles_map;
};

template <RSE_FloatType FloatMemAccuracy_>
struct RS_Context {
    constexpr static RSE_FloatType FloatMemAccuracy = FloatMemAccuracy_;

    using LinkMaterialMemType = logic::RS_LinkMaterialMem<FloatMemAccuracy_>;
    using VoxelMaterialMemType = logic::RS_VoxelMaterialMem<FloatMemAccuracy_>;
    using LinkMemType = logic::RS_LinkMem<FloatMemAccuracy_>;
    using VoxelMemType = logic::RS_VoxelMem<FloatMemAccuracy_>;
    using RigidBodyMemType = logic::RS_RigidBodyMem<FloatMemAccuracy_>;
    using RigidBodySurfaceVoxelMemType = logic::RS_RigidBodySurfaceVoxelMem<FloatMemAccuracy_>;
    using ConstraintJointMemType = logic::RS_ConstraintJointMem<FloatMemAccuracy_>;
    using RigidContactMemType = logic::RS_RigidContactMem<FloatMemAccuracy_>;

    using CameraMemType = logic::RS_CameraMem<FloatMemAccuracy_>;
    using VisionMaterialMemType = logic::RS_VisionMaterialMem<FloatMemAccuracy_>;

    RS_ArrayFloorContext floor;

    RS_VoxelBVH static_voxel_bvh;
    RS_VoxelBVH dynamic_voxel_bvh;
    device::RS_BVH floor_triangle_bvh;

    device::RS_Array<logic::RS_MeshTriangleMem, 1> mesh_triangles;
    device::RS_DeviceLayout mesh_triangle_layout;

    device::RS_StructOfArrays<LinkMaterialMemType> structure_link_materials;
    device::RS_StructOfArrays<VoxelMaterialMemType> structure_voxel_materials;
    device::RS_StructOfArrays<LinkMemType> structure_links;
    device::RS_StructOfArrays<VoxelMemType> structure_voxels;
    device::RS_StructOfArrays<RigidBodyMemType> structure_rigid_bodies;
    device::RS_StructOfArrays<RigidBodySurfaceVoxelMemType> structure_rigid_body_surface_voxels;
    device::RS_StructOfArrays<ConstraintJointMemType> structure_constraint_joints;

    device::RS_Vector<logic::RS_ConstraintJointBlockRange> structure_constraint_joint_block_ranges;

    device::RS_StructOfVectors<RigidContactMemType> structure_rigid_contacts;
    device::RS_StructOfVectors<RigidContactMemType> structure_rigid_sorted_contacts;
    device::RS_Vector<logic::RS_RigidContactKey> structure_rigid_contact_keys;
    device::RS_Vector<logic::RS_RigidContactKey> structure_rigid_sorted_contact_keys;
    device::RS_Vector<RIndex> structure_rigid_contact_indices;
    device::RS_Vector<RIndex> structure_rigid_sorted_contact_indices;
    device::RS_Vector<RIndex> structure_rigid_contact_flags;
    device::RS_Vector<RIndex> structure_rigid_contact_prefix_sums;
    device::RS_Vector<logic::RS_RigidManifoldDeepest> structure_rigid_manifold_deepest_in;
    device::RS_Vector<logic::RS_RigidManifoldDeepest> structure_rigid_manifold_deepest_out;
    device::RS_Vector<logic::RS_RigidManifoldFarthest> structure_rigid_manifold_farthest_in;
    device::RS_Vector<logic::RS_RigidManifoldFarthest> structure_rigid_manifold_farthest_out;

    device::RS_Vector<RHandle> structure_emissive_voxel_handles;

    device::RS_Vector<RIndex> structure_voxel_bvh_flags;
    device::RS_Vector<RIndex> structure_voxel_bvh_prefix_sums;
    device::RS_Vector<RHandle> structure_voxel_bvh_active_handles;

    device::RS_Array<RFloat, 1> structure_expansion_signals;
    device::RS_Array<RFloat, 1> structure_hinge_signals;
    device::RS_Array<RFloat, 1> structure_prismatic_signals;

    device::RS_DeviceLayout structure_link_material_layout;
    device::RS_DeviceLayout structure_voxel_material_layout;
    device::RS_DeviceLayout structure_link_layout;
    device::RS_DeviceLayout structure_voxel_layout;
    device::RS_DeviceLayout structure_rigid_body_layout;
    device::RS_DeviceLayout structure_rigid_body_surface_voxel_layout;
    device::RS_DeviceLayout structure_constraint_joint_layout;

    device::RS_DeviceLayout structure_constraint_joint_block_range_layout;

    device::RS_DeviceLayout structure_expansion_signal_layout;
    device::RS_DeviceLayout structure_hinge_signal_layout;
    device::RS_DeviceLayout structure_prismatic_signal_layout;

    device::RS_Vector<RS_SimulationStructureRecord> structure_records;
    device::RS_Vector<device::RS_DoubleBuffer> structure_recorder_buffers;
    device::RS_Vector<device::RS_DirectBuffer> structure_controller_buffers;
    device::RS_Vector<device::RS_Array<RHandle, 1>> structure_voxel_record_handles;
    device::RS_Vector<device::RS_Array<RHandle, 1>> structure_link_record_handles;
    device::RS_Vector<device::RS_Array<RHandle, 1>> structure_rigid_body_record_handles;
    device::RS_Vector<device::RS_Array<RHandle, 1>> structure_constraint_joint_record_handles;

    device::RS_DeviceLayout structure_voxel_record_layout;
    device::RS_DeviceLayout structure_link_record_layout;
    device::RS_DeviceLayout structure_rigid_body_record_layout;
    device::RS_DeviceLayout structure_joint_record_layout;
    device::RS_DeviceLayout structure_constraint_joint_record_layout;

    device::RS_DeviceLayout structure_rigid_body_surface_voxel_to_rigid_body_reduction_input_layout;
    device::RS_DeviceLayout
        structure_rigid_body_surface_voxel_to_rigid_body_reduction_output_layout;

    device::RS_Vector<CameraMemType> cameras;
    device::RS_Vector<RS_SimulationCameraRecord> camera_records;
    device::RS_Vector<device::RS_DoubleBuffer> camera_recorder_buffers;
    device::RS_Vector<device::RS_DirectBuffer> camera_controller_buffers;
    device::RS_Vector<logic::RS_CameraRGBCanvas> camera_current_rgb_canvas;
    device::RS_Vector<logic::RS_CameraRGBCanvas> camera_next_rgb_canvas;
    // Basic and restir shared primary aux buffers.
    device::RS_Vector<logic::RS_CameraAuxNormal> camera_aux_normal;
    device::RS_Vector<logic::RS_CameraAuxDepthZ> camera_aux_depth_z;
    device::RS_Vector<logic::RS_CameraAuxHitPosition> camera_aux_hit_position;
    device::RS_Vector<logic::RS_CameraAuxObjectKey> camera_aux_object_key;
    device::RS_Vector<logic::RS_CameraAuxRoughness> camera_aux_roughness;
    device::RS_Vector<logic::RS_CameraAuxScreenMotion> camera_aux_screen_motion;
    // Restir only aux buffers.
    device::RS_Vector<logic::RS_CameraAuxHitPosition> camera_aux_endpoint_hit_position;
    device::RS_Vector<logic::RS_CameraAuxNormal> camera_aux_endpoint_normal;
    device::RS_Vector<logic::RS_CameraAuxObjectKey> camera_aux_endpoint_object_key;
    device::RS_Vector<logic::RS_CameraAuxAlbedo> camera_aux_endpoint_albedo;
    device::RS_Vector<logic::RS_CameraAuxRoughness> camera_aux_endpoint_roughness;
    device::RS_Vector<logic::RS_CameraAuxMetalness> camera_aux_endpoint_metalness;
    device::RS_Vector<logic::RS_CameraAuxIntensity> camera_aux_endpoint_intensity;
    device::RS_Vector<logic::RS_CameraAuxMaterialType> camera_aux_endpoint_material_type;
    device::RS_Vector<logic::RS_CameraAuxScreenMotion> camera_aux_endpoint_screen_motion;
    device::RS_Vector<logic::RS_CameraAuxChainThroughput> camera_aux_chain_throughput;
    device::RS_Vector<logic::RS_CameraAuxChainTerminalKind> camera_aux_chain_terminal_kind;
    device::RS_Vector<logic::RS_CameraReservoirDI> camera_prev_reservoir_di;
    device::RS_Vector<logic::RS_CameraReservoirDI> camera_current_reservoir_di;
    device::RS_Vector<logic::RS_CameraReservoirGI> camera_prev_reservoir_gi;
    device::RS_Vector<logic::RS_CameraReservoirGI> camera_current_reservoir_gi;
    device::RS_Vector<logic::RS_CameraReservoirDI> camera_spatial_reservoir_di;
    device::RS_Vector<logic::RS_CameraReservoirGI> camera_spatial_reservoir_gi;
    // SVGF prev-frame aux snapshots.
    device::RS_Vector<logic::RS_CameraPrevAuxNormal> camera_prev_aux_normal;
    device::RS_Vector<logic::RS_CameraPrevAuxDepthZ> camera_prev_aux_depth_z;
    device::RS_Vector<logic::RS_CameraPrevAuxObjectKey> camera_prev_aux_object_key;
    // SVGF temporal history (current and ping-pong prev).
    device::RS_Vector<logic::RS_CameraDenoiserColorHistory> camera_denoiser_color_history;
    device::RS_Vector<logic::RS_CameraDenoiserMomentHistory> camera_denoiser_moment_history;
    device::RS_Vector<logic::RS_CameraDenoiserHistoryLength> camera_denoiser_history_length;
    device::RS_Vector<logic::RS_CameraDenoiserVariance> camera_denoiser_variance;
    device::RS_Vector<logic::RS_CameraDenoiserColorHistory> camera_prev_denoiser_color_history;
    device::RS_Vector<logic::RS_CameraDenoiserMomentHistory> camera_prev_denoiser_moment_history;
    device::RS_Vector<logic::RS_CameraDenoiserHistoryLength> camera_prev_denoiser_history_length;

    device::RS_Vector<RFloat> camera_control_signals;

    device::RS_DeviceLayout camera_layout;
    device::RS_DeviceLayout camera_info_ray_layout;
    device::RS_DeviceLayout camera_rgb_ray_layout;
    device::RS_DeviceLayout camera_record_layout;
    device::RS_DeviceLayout camera_pixel_record_layout;
    device::RS_DeviceLayout camera_state_record_layout;
    device::RS_DeviceLayout camera_control_signal_layout;

    device::RS_StructOfArrays<VisionMaterialMemType> vision_materials;
    device::RS_DeviceLayout vision_material_layout;

    /// Note: ReSTIR DI light list: index 0 is sun, indices 1..N correspond to
    /// structure_emissive_voxel_handles. light_weight stores the luminance of
    /// each light's emitted radiance; light_cdf is the inclusive prefix sum.
    device::RS_Vector<float> light_weight;
    device::RS_Vector<float> light_cdf;

    device::RS_Array<RFloat, 1> force_signals;
    device::RS_DeviceLayout force_signal_layout;
    device::RS_Vector<logic::RS_RigidForceMem> force_rigid;
    device::RS_Vector<logic::RS_SoftForceMem> force_soft;

    device::RS_Array<RFloat, 1> torque_signals;
    device::RS_DeviceLayout torque_signal_layout;
    device::RS_Vector<logic::RS_RigidTorqueMem> torque_rigid;

    RS_ContextHostData *host_data = nullptr;
};

struct RS_AnyContext {
    char data[16384];
};

}  // namespace rise

#ifndef DEBUG_SOA
#define DEBUG_ARGS(method)
#else
#define DEBUG_ARGS(method) , #method ": @file " __FILE__ " @line " TOSTRING(__LINE__)
#endif

#define CTX_G(context, object, property, index)                                                \
    (context.object.template get<Pr(typename decltype(context.object)::ObjectType, property)>( \
        index DEBUG_ARGS(object_get)                                                           \
    ))

#define CTX_S(context, object, property, index, value)                                        \
    context.object.template set<Pr(typename decltype(context.object)::ObjectType, property)>( \
        index, value DEBUG_ARGS(object_get)                                                   \
    )

// See https://stackoverflow.com/questions/3046889/optional-parameters-with-c-macros
// Empty macro are placed there for safely clear replacements when argument number is
// invalid. voxel get
#define V_G_0()
#define V_G_1(property)                                                                    \
    (ctx.structure_voxels                                                                  \
         .template get<Pr(typename decltype(ctx.structure_voxels)::ObjectType, property)>( \
             voxel DEBUG_ARGS(voxel_get)                                                   \
         ))
#define V_G_2(voxel_idx, property)                                                         \
    (ctx.structure_voxels                                                                  \
         .template get<Pr(typename decltype(ctx.structure_voxels)::ObjectType, property)>( \
             voxel_idx DEBUG_ARGS(voxel_get)                                               \
         ))
#define V_G_X(x, A, B, FUNC, ...) FUNC
#define V_G(...) V_G_X(, ##__VA_ARGS__, V_G_2(__VA_ARGS__), V_G_1(__VA_ARGS__), V_G_0(__VA_ARGS__))

#define V_S_0()
#define V_S_1()
#define V_S_2(property, value)                                                            \
    ctx.structure_voxels                                                                  \
        .template set<Pr(typename decltype(ctx.structure_voxels)::ObjectType, property)>( \
            voxel, value DEBUG_ARGS(voxel_set)                                            \
        )
#define V_S_3(voxel_idx, property, value)                                                 \
    ctx.structure_voxels                                                                  \
        .template set<Pr(typename decltype(ctx.structure_voxels)::ObjectType, property)>( \
            voxel_idx, value DEBUG_ARGS(voxel_set)                                        \
        )
#define V_S_X(x, A, B, C, FUNC, ...) FUNC
#define V_S(...)            \
    V_S_X(                  \
        ,                   \
        ##__VA_ARGS__,      \
        V_S_3(__VA_ARGS__), \
        V_S_2(__VA_ARGS__), \
        V_S_1(__VA_ARGS__), \
        V_S_0(__VA_ARGS__)  \
    )

#define RB_G_0()
#define RB_G_1(property)                                                                         \
    (ctx.structure_rigid_bodies                                                                  \
         .template get<Pr(typename decltype(ctx.structure_rigid_bodies)::ObjectType, property)>( \
             rigid_body DEBUG_ARGS(rigid_body_get)                                               \
         ))
#define RB_G_2(rigid_body_idx, property)                                                         \
    (ctx.structure_rigid_bodies                                                                  \
         .template get<Pr(typename decltype(ctx.structure_rigid_bodies)::ObjectType, property)>( \
             rigid_body_idx DEBUG_ARGS(rigid_body_get)                                           \
         ))
#define RB_G_X(x, A, B, FUNC, ...) FUNC
#define RB_G(...) \
    RB_G_X(, ##__VA_ARGS__, RB_G_2(__VA_ARGS__), RB_G_1(__VA_ARGS__), RB_G_0(__VA_ARGS__))

#define RB_S_0()
#define RB_S_1()
#define RB_S_2(property, value)                                                                 \
    ctx.structure_rigid_bodies                                                                  \
        .template set<Pr(typename decltype(ctx.structure_rigid_bodies)::ObjectType, property)>( \
            rigid_body, value DEBUG_ARGS(rigid_body_set)                                        \
        )
#define RB_S_3(rigid_body_idx, property, value)                                                 \
    ctx.structure_rigid_bodies                                                                  \
        .template set<Pr(typename decltype(ctx.structure_rigid_bodies)::ObjectType, property)>( \
            rigid_body_idx, value DEBUG_ARGS(rigid_body_set)                                    \
        )
#define RB_S_X(x, A, B, C, FUNC, ...) FUNC
#define RB_S(...)            \
    RB_S_X(                  \
        ,                    \
        ##__VA_ARGS__,       \
        RB_S_3(__VA_ARGS__), \
        RB_S_2(__VA_ARGS__), \
        RB_S_1(__VA_ARGS__), \
        RB_S_0(__VA_ARGS__)  \
    )

#define RBSV_G_0()
#define RBSV_G_1(property)                                                                     \
    (ctx.structure_rigid_body_surface_voxels.template get<                                     \
        Pr(typename decltype(ctx.structure_rigid_body_surface_voxels)::ObjectType, property)>( \
        rigid_body_surface_voxel DEBUG_ARGS(rigid_body_surface_voxel_get)                      \
    ))
#define RBSV_G_2(rigid_body_surface_voxel_idx, property)                                       \
    (ctx.structure_rigid_body_surface_voxels.template get<                                     \
        Pr(typename decltype(ctx.structure_rigid_body_surface_voxels)::ObjectType, property)>( \
        rigid_body_surface_voxel_idx DEBUG_ARGS(rigid_body_surface_voxel_get)                  \
    ))
#define RBSV_G_X(x, A, B, FUNC, ...) FUNC
#define RBSV_G(...) \
    RBSV_G_X(, ##__VA_ARGS__, RBSV_G_2(__VA_ARGS__), RBSV_G_1(__VA_ARGS__), RBSV_G_0(__VA_ARGS__))

#define RBSV_S_0()
#define RBSV_S_1()
#define RBSV_S_2(property, value)                                                              \
    ctx.structure_rigid_body_surface_voxels.template set<                                      \
        Pr(typename decltype(ctx.structure_rigid_body_surface_voxels)::ObjectType, property)>( \
        rigid_body_surface_voxel, value DEBUG_ARGS(rigid_body_surface_voxel_set)               \
    )
#define RBSV_S_3(rigid_body_surface_voxel_idx, property, value)                                \
    ctx.structure_rigid_body_surface_voxels.template set<                                      \
        Pr(typename decltype(ctx.structure_rigid_body_surface_voxels)::ObjectType, property)>( \
        rigid_body_surface_voxel_idx, value DEBUG_ARGS(rigid_body_surface_voxel_set)           \
    )
#define RBSV_S_X(x, A, B, C, FUNC, ...) FUNC
#define RBSV_S(...)            \
    RBSV_S_X(                  \
        ,                      \
        ##__VA_ARGS__,         \
        RBSV_S_3(__VA_ARGS__), \
        RBSV_S_2(__VA_ARGS__), \
        RBSV_S_1(__VA_ARGS__), \
        RBSV_S_0(__VA_ARGS__)  \
    )

#define CJ_G_0()
#define CJ_G_1(property)                                                               \
    (ctx.structure_constraint_joints.template get<                                     \
        Pr(typename decltype(ctx.structure_constraint_joints)::ObjectType, property)>( \
        constraint_joint DEBUG_ARGS(constraint_joint_get)                              \
    ))
#define CJ_G_2(constraint_joint_idx, property)                                         \
    (ctx.structure_constraint_joints.template get<                                     \
        Pr(typename decltype(ctx.structure_constraint_joints)::ObjectType, property)>( \
        constraint_joint_idx DEBUG_ARGS(constraint_joint_get)                          \
    ))
#define CJ_G_X(x, A, B, FUNC, ...) FUNC
#define CJ_G(...) \
    CJ_G_X(, ##__VA_ARGS__, CJ_G_2(__VA_ARGS__), CJ_G_1(__VA_ARGS__), CJ_G_0(__VA_ARGS__))

#define CJ_S_0()
#define CJ_S_1()
#define CJ_S_2(property, value)                                                        \
    ctx.structure_constraint_joints.template set<                                      \
        Pr(typename decltype(ctx.structure_constraint_joints)::ObjectType, property)>( \
        constraint_joint, value DEBUG_ARGS(constraint_joint_set)                       \
    )
#define CJ_S_3(constraint_joint_idx, property, value)                                  \
    ctx.structure_constraint_joints.template set<                                      \
        Pr(typename decltype(ctx.structure_constraint_joints)::ObjectType, property)>( \
        constraint_joint_idx, value DEBUG_ARGS(constraint_joint_set)                   \
    )
#define CJ_S_X(x, A, B, C, FUNC, ...) FUNC
#define CJ_S(...)            \
    CJ_S_X(                  \
        ,                    \
        ##__VA_ARGS__,       \
        CJ_S_3(__VA_ARGS__), \
        CJ_S_2(__VA_ARGS__), \
        CJ_S_1(__VA_ARGS__), \
        CJ_S_0(__VA_ARGS__)  \
    )

#define L_G_0()
#define L_G_1(property)                                                                   \
    (ctx.structure_links                                                                  \
         .template get<Pr(typename decltype(ctx.structure_links)::ObjectType, property)>( \
             link DEBUG_ARGS(link_get)                                                    \
         ))
#define L_G_2(link_idx, property)                                                         \
    (ctx.structure_links                                                                  \
         .template get<Pr(typename decltype(ctx.structure_links)::ObjectType, property)>( \
             link_idx DEBUG_ARGS(link_get)                                                \
         ))
#define L_G_X(x, A, B, FUNC, ...) FUNC
#define L_G(...) L_G_X(, ##__VA_ARGS__, L_G_2(__VA_ARGS__), L_G_1(__VA_ARGS__), L_G_0(__VA_ARGS__))

#define L_S_0()
#define L_S_1()
#define L_S_2(property, value)                                                           \
    ctx.structure_links                                                                  \
        .template set<Pr(typename decltype(ctx.structure_links)::ObjectType, property)>( \
            link, value DEBUG_ARGS(link_set)                                             \
        )
#define L_S_3(link_idx, property, value)                                                 \
    ctx.structure_links                                                                  \
        .template set<Pr(typename decltype(ctx.structure_links)::ObjectType, property)>( \
            link_idx, value DEBUG_ARGS(link_set)                                         \
        )
#define L_S_X(x, A, B, C, FUNC, ...) FUNC
#define L_S(...)            \
    L_S_X(                  \
        ,                   \
        ##__VA_ARGS__,      \
        L_S_3(__VA_ARGS__), \
        L_S_2(__VA_ARGS__), \
        L_S_1(__VA_ARGS__), \
        L_S_0(__VA_ARGS__)  \
    )

#define LM_G_0()
#define LM_G_1(property)                                                                           \
    (ctx.structure_link_materials                                                                  \
         .template get<Pr(typename decltype(ctx.structure_link_materials)::ObjectType, property)>( \
             link_material DEBUG_ARGS(link_material_get)                                           \
         ))

#define LM_G_2(link_material_idx, property)                                                        \
    (ctx.structure_link_materials                                                                  \
         .template get<Pr(typename decltype(ctx.structure_link_materials)::ObjectType, property)>( \
             link_material_idx DEBUG_ARGS(link_material_get)                                       \
         ))
#define LM_G_X(x, A, B, FUNC, ...) FUNC
#define LM_G(...) \
    LM_G_X(, ##__VA_ARGS__, LM_G_2(__VA_ARGS__), LM_G_1(__VA_ARGS__), LM_G_0(__VA_ARGS__))

#define LM_S_0()
#define LM_S_1()
#define LM_S_2(property, value)                                                                   \
    ctx.structure_link_materials                                                                  \
        .template set<Pr(typename decltype(ctx.structure_link_materials)::ObjectType, property)>( \
            link_material, value DEBUG_ARGS(link_material_set)                                    \
        )

#define LM_S_3(link_material_idx, property, value)                                                \
    ctx.structure_link_materials                                                                  \
        .template set<Pr(typename decltype(ctx.structure_link_materials)::ObjectType, property)>( \
            link_material_idx, value DEBUG_ARGS(link_material_set)                                \
        )

#define LM_S_X(x, A, B, C, FUNC, ...) FUNC
#define LM_S(...)            \
    LM_S_X(                  \
        ,                    \
        ##__VA_ARGS__,       \
        LM_S_3(__VA_ARGS__), \
        LM_S_2(__VA_ARGS__), \
        LM_S_1(__VA_ARGS__), \
        LM_S_0(__VA_ARGS__)  \
    )

#define VM_G_0()
#define VM_G_1(property)                                                             \
    (ctx.structure_voxel_materials.template get<                                     \
        Pr(typename decltype(ctx.structure_voxel_materials)::ObjectType, property)>( \
        voxel_material DEBUG_ARGS(voxel_material_get)                                \
    ))

#define VM_G_2(voxel_material_idx, property)                                         \
    (ctx.structure_voxel_materials.template get<                                     \
        Pr(typename decltype(ctx.structure_voxel_materials)::ObjectType, property)>( \
        voxel_material_idx DEBUG_ARGS(voxel_material_get)                            \
    ))

#define VM_G_X(x, A, B, FUNC, ...) FUNC
#define VM_G(...) \
    VM_G_X(, ##__VA_ARGS__, VM_G_2(__VA_ARGS__), VM_G_1(__VA_ARGS__), VM_G_0(__VA_ARGS__))

#define VM_S_0()
#define VM_S_1()
#define VM_S_2(property, value)                                                                    \
    ctx.structure_voxel_materials                                                                  \
        .template set<Pr(typename decltype(ctx.structure_voxel_materials)::ObjectType, property)>( \
            voxel_material, value DEBUG_ARGS(voxel_material_set)                                   \
        )

#define VM_S_3(voxel_material_idx, property, value)                                                \
    ctx.structure_voxel_materials                                                                  \
        .template set<Pr(typename decltype(ctx.structure_voxel_materials)::ObjectType, property)>( \
            voxel_material_idx, value DEBUG_ARGS(voxel_material_set)                               \
        )
#define VM_S_X(x, A, B, C, FUNC, ...) FUNC
#define VM_S(...)            \
    VM_S_X(                  \
        ,                    \
        ##__VA_ARGS__,       \
        VM_S_3(__VA_ARGS__), \
        VM_S_2(__VA_ARGS__), \
        VM_S_1(__VA_ARGS__), \
        VM_S_0(__VA_ARGS__)  \
    )

#define CA_G_0()
#define CA_G_1(property) ctx.cameras[camera].property
#define CA_G_2(camera_idx, property) ctx.cameras[camera_idx].property
#define CA_G_X(x, A, B, FUNC, ...) FUNC
#define CA_G(...) \
    CA_G_X(, ##__VA_ARGS__, CA_G_2(__VA_ARGS__), CA_G_1(__VA_ARGS__), CA_G_0(__VA_ARGS__))

#define CA_S_0()
#define CA_S_1()
#define CA_S_2(property, value) ctx.cameras[camera].property = value
#define CA_S_3(camera_idx, property, value) ctx.cameras[camera_idx].property = value
#define CA_S_X(x, A, B, C, FUNC, ...) FUNC
#define CA_S(...)            \
    CA_S_X(                  \
        ,                    \
        ##__VA_ARGS__,       \
        CA_S_3(__VA_ARGS__), \
        CA_S_2(__VA_ARGS__), \
        CA_S_1(__VA_ARGS__), \
        CA_S_0(__VA_ARGS__)  \
    )

#define VSM_G_0()
#define VSM_G_1(property)                                                                  \
    (ctx.vision_materials                                                                  \
         .template get<Pr(typename decltype(ctx.vision_materials)::ObjectType, property)>( \
             vision_material DEBUG_ARGS(vision_material_get)                               \
         ))

#define VSM_G_2(vision_material_idx, property)                                             \
    (ctx.vision_materials                                                                  \
         .template get<Pr(typename decltype(ctx.vision_materials)::ObjectType, property)>( \
             vision_material_idx DEBUG_ARGS(vision_material_get)                           \
         ))
#define VSM_G_X(x, A, B, FUNC, ...) FUNC
#define VSM_G(...) \
    VSM_G_X(, ##__VA_ARGS__, VSM_G_2(__VA_ARGS__), VSM_G_1(__VA_ARGS__), VSM_G_0(__VA_ARGS__))

#define VSM_S_0()
#define VSM_S_1()
#define VSM_S_2(property, value)                                                          \
    ctx.vision_materials                                                                  \
        .template set<Pr(typename decltype(ctx.vision_materials)::ObjectType, property)>( \
            vision_material, value DEBUG_ARGS(vision_material_set)                        \
        )
#define VSM_S_3(vision_material_idx, property, value)                                     \
    ctx.vision_materials                                                                  \
        .template set<Pr(typename decltype(ctx.vision_materials)::ObjectType, property)>( \
            vision_material_idx, value DEBUG_ARGS(vision_material_set)                    \
        )

#define VSM_S_X(x, A, B, C, FUNC, ...) FUNC
#define VSM_S(...)            \
    VSM_S_X(                  \
        ,                     \
        ##__VA_ARGS__,        \
        VSM_S_3(__VA_ARGS__), \
        VSM_S_2(__VA_ARGS__), \
        VSM_S_1(__VA_ARGS__), \
        VSM_S_0(__VA_ARGS__)  \
    )

#endif  // RS_CONTEXT_H
