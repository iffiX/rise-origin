// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RAY_SENSOR_RAY_C_H
#define RS_RAY_SENSOR_RAY_C_H

#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/ray/rs_ray_sensor_m.h"
#include "rise_origin/logic/record/rs_ray_sensor_record_m.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"
#include "rise_origin/logic/vision/rs_vision_floor_m.h"
#include "rise_origin/logic/vision/rs_vision_material_m.h"
#include "rise_origin/utils/math/rs_mat.h"
#include "rise_origin/utils/math/rs_triangle.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_conf.h"

namespace rise {

namespace logic {

/**
 * @brief Closest-hit ray query for ray sensing. A slim voxel BVH plus floor walk that
 * keeps only the geometry and material id readouts a ray sensor reports.
 */
template <RSE_FloatType FloatComputeAccuracy = RSE_FLOAT_32>
class RS_RaySensorRayCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;
    using Mat3C = RS_Mat3D<FloatC>;
    using RS_FI = RS_FloatIntrinsics<FloatC>;

    static constexpr FloatC RAY_T_MIN = FloatC(1e-4);

    /// Stack-allocated sensor ray.
    struct Ray {
        Vec3C ray_origin;
        Vec3C ray_direction;

        Ray() = default;
        HOST_DEVICE_INLINE_QUALIFIER Ray(const Vec3C& origin, const Vec3C& direction)
            : ray_origin(origin), ray_direction(direction) {}

        HOST_DEVICE_INLINE_QUALIFIER Vec3C at(FloatC t) const {
            return ray_origin + t * ray_direction;
        }
    };

    /// Closest hit geometry and material id readout at a ray.
    struct HitRecord {
        Vec3C p = Vec3C::zero();
        Vec3C normal = Vec3C(0, 0, 1);
        FloatC t = FloatC(0);
        FloatC emission = FloatC(0);
        RHandle voxel = RS_NULL_HANDLE;
        RIndex structure_gid = RS_NULL_INDEX;
        RIndex material_gid = RS_NULL_INDEX;
        bool is_floor_hit = false;
    };

    class RSCF_TraceRaySensorRays : public device::RS_ComputeFunction<RSCF_TraceRaySensorRays> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void
        run(Kernel* k, RS_RaySensorRecordMem& record, RHandle ray_sensor, RHandle ray) {
            if (k == nullptr || ray_sensor == RS_NULL_HANDLE || !record.is_valid) return;
            auto& ctx = k->ctx;

            Vec3C sensor_position = Vec3C(RAY_SENSOR_G(position));
            Quat3C sensor_orientation = Quat3C(RAY_SENSOR_G(orientation));
            FloatC max_range = FloatC(RAY_SENSOR_G(max_range));

            Vec3C local_dir = Vec3C(ctx.ray_sensor_directions[ray_sensor][ray]);
            Vec3C world_dir = sensor_orientation.rotateVec3D(local_dir).normalized();
            Ray world_ray(sensor_position, world_dir);

            HitRecord rec;
            bool is_hit = _findHit(k, world_ray, max_range, rec);

            _writeHit(record, ray, is_hit, world_ray, max_range, rec);
            _writeEntityInfo(record, ray, is_hit, rec);
            _writeEntityDetail(k, record, ray, is_hit, rec);
            _writeMaterial(record, ray, is_hit, rec);
            _writeNormal(record, ray, is_hit, rec);
            _writeVelocity(k, record, ray, ray_sensor, is_hit, world_dir, rec);
        }

       private:
        HOST_DEVICE_INLINE_QUALIFIER static void _writeHit(
            RS_RaySensorRecordMem& record,
            RHandle ray,
            bool is_hit,
            const Ray& world_ray,
            FloatC max_range,
            const HitRecord& rec
        ) {
            if (record.hit.object_num == 0) return;
            record.hit.is_hit[ray] = is_hit ? RByte(1) : RByte(0);
            record.hit.distance[ray] = is_hit ? RFloat(rec.t) : RFloat(max_range);
            Vec3C p = is_hit ? rec.p : world_ray.at(max_range);
            record.hit.hit_position[ray] = RVec3rf(RFloat(p.x), RFloat(p.y), RFloat(p.z));
        }

        HOST_DEVICE_INLINE_QUALIFIER static void _writeEntityInfo(
            RS_RaySensorRecordMem& record,
            RHandle ray,
            bool is_hit,
            const HitRecord& rec
        ) {
            if (record.entity_info.object_num == 0) return;
            if (!is_hit) {
                record.entity_info.entity_gid[ray] = RS_WORLD_RSC_COSMIC_BACKGROUND_GID;
                record.entity_info.entity_type[ray] = RSE_WORLD_ENTITY_TYPE_COSMIC_BACKGROUND;
            } else if (rec.is_floor_hit) {
                record.entity_info.entity_gid[ray] = RS_WORLD_RSC_FLOOR_GID;
                record.entity_info.entity_type[ray] = RSE_WORLD_ENTITY_TYPE_FLOOR;
            } else {
                record.entity_info.entity_gid[ray] = rec.structure_gid;
                record.entity_info.entity_type[ray] = RSE_WORLD_ENTITY_TYPE_STRUCTURE_VOXEL;
            }
        }

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _writeEntityDetail(
            const Kernel* k,
            RS_RaySensorRecordMem& record,
            RHandle ray,
            bool is_hit,
            const HitRecord& rec
        ) {
            if (record.entity_detail.object_num == 0) return;
            auto& ctx = k->ctx;
            bool is_voxel = is_hit && !rec.is_floor_hit && rec.voxel != RS_NULL_HANDLE;
            RHandle voxel = rec.voxel;
            record.entity_detail.body_sid[ray] = is_voxel ? V_G(voxel, body_sid) : RS_NULL_INDEX;
            record.entity_detail.segment_bid[ray] = is_voxel ? V_G(voxel, segment_bid)
                                                             : RS_NULL_INDEX;
            record.entity_detail.voxel_sid[ray] = is_voxel ? V_G(voxel, sid) : RS_NULL_INDEX;
        }

        HOST_DEVICE_INLINE_QUALIFIER static void _writeMaterial(
            RS_RaySensorRecordMem& record,
            RHandle ray,
            bool is_hit,
            const HitRecord& rec
        ) {
            if (record.material.object_num == 0) return;
            bool is_voxel = is_hit && !rec.is_floor_hit;
            record.material.material_id[ray] = is_voxel ? rec.material_gid : RS_NULL_INDEX;
            record.material.emission[ray] = is_voxel ? RFloat(rec.emission) : RFloat(0);
        }

        HOST_DEVICE_INLINE_QUALIFIER static void _writeNormal(
            RS_RaySensorRecordMem& record,
            RHandle ray,
            bool is_hit,
            const HitRecord& rec
        ) {
            if (record.normal.object_num == 0) return;
            Vec3C n = is_hit ? rec.normal : Vec3C::zero();
            record.normal.normal[ray] = RVec3rf(RFloat(n.x), RFloat(n.y), RFloat(n.z));
        }

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void _writeVelocity(
            const Kernel* k,
            RS_RaySensorRecordMem& record,
            RHandle ray,
            RHandle ray_sensor,
            bool is_hit,
            const Vec3C& world_dir,
            const HitRecord& rec
        ) {
            if (record.velocity.object_num == 0) return;
            auto& ctx = k->ctx;

            Vec3C v_hit = Vec3C::zero();
            if (is_hit && !rec.is_floor_hit) {
                _computeVoxelVelocity(k, rec.voxel, rec.p, v_hit);
            }

            // The sensor's own velocity is a per-sensor value shared by all rays, tracked by
            // the pose update. It is not a per-ray output column.
            Vec3C v_sensor = Vec3C(RAY_SENSOR_G(velocity));

            Vec3C v_rel = v_hit - v_sensor;
            FloatC v_radial = v_rel.dot(world_dir);
            record.velocity.v_hit[ray] = RVec3rf(RFloat(v_hit.x), RFloat(v_hit.y), RFloat(v_hit.z));
            record.velocity.v_rel[ray] = RVec3rf(RFloat(v_rel.x), RFloat(v_rel.y), RFloat(v_rel.z));
            record.velocity.v_radial[ray] = RFloat(v_radial);
        }
    };

   private:
    /*****************************************************************
     * BVH / intersection
     *****************************************************************/

    /**
     * @brief Slab test of a ray against an axis-aligned box.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _isRayIntersectingAABB(
        const Ray& ray,
        const Vec3C& aabb_min,
        const Vec3C& aabb_max,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_enter_out,
        FloatC& t_exit_out
    ) {
        Vec3C dir_reciprocal = ray.ray_direction.reciprocalFast();
        Vec3C t0 = (aabb_min - ray.ray_origin).elemMul(dir_reciprocal);
        Vec3C t1 = (aabb_max - ray.ray_origin).elemMul(dir_reciprocal);
        Vec3C t_min_vec = t0.min(t1);
        Vec3C t_max_vec = t0.max(t1);
        t_enter_out = t_min_vec.max();
        t_exit_out = t_max_vec.min();
        return t_exit_out >= t_enter_out && t_exit_out >= t_min && t_enter_out <= t_max;
    }

    /**
     * @brief First real boundary face along the ray inside a voxel box.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _voxelFirstBoundary(
        Kernel* k,
        RHandle voxel,
        const Ray& ray,
        FloatC t_min,
        FloatC& t_out,
        int& face_out,
        bool& is_front_out,
        Vec3C& normal_out
    ) {
        auto& ctx = k->ctx;
        Vec3C nnn = V_G(voxel, nnn_offset);
        Vec3C ppp = V_G(voxel, ppp_offset);
        Vec3C position = V_G(voxel, position);
        Quat3C orientation = V_G(voxel, orientation);
        Vec3C local_axes[3] = {
            orientation.toOrthoAxisX(), orientation.toOrthoAxisY(), orientation.toOrthoAxisZ()
        };
        Vec3C local_ray_origin = orientation.rotateVec3DInv(ray.ray_origin - position);
        Vec3C local_ray_dir = orientation.rotateVec3DInv(ray.ray_direction);
        Vec3C local_ray_dir_reciprocal = local_ray_dir.reciprocalFast();
        Vec3C t0 = (nnn - local_ray_origin).elemMul(local_ray_dir_reciprocal);
        Vec3C t1 = (ppp - local_ray_origin).elemMul(local_ray_dir_reciprocal);
        Vec3C t_min_vec = t0.min(t1);
        Vec3C t_max_vec = t0.max(t1);
        FloatC t_enter = t_min_vec.max();
        FloatC t_exit = t_max_vec.min();
        if (t_exit < t_enter || t_exit < t_min) return false;
        RState voxel_state = V_G(voxel, state);

        bool is_entry_reached = t_enter >= t_min;
        for (int pass = 0; pass < 2; ++pass) {
            bool is_entry = pass == 0;
            if (is_entry && !is_entry_reached) continue;
            FloatC t_cross = is_entry ? t_enter : t_exit;
            int axis = is_entry ? t_min_vec.argmax() : t_max_vec.argmin();
            Vec3C local_hit_point = local_ray_origin + local_ray_dir * t_cross;
            Vec3C center_offset = local_hit_point - (nnn + ppp) * FloatC(0.5);
            bool is_pos = center_offset[axis] > 0;
            int face = 2 * axis + (is_pos ? 0 : 1);
            RSE_VoxelFaceRole role = getFaceRole(voxel_state, face);
            if (role == RSE_VOXEL_FACE_INTERIOR) continue;
            if (!is_entry && role != RSE_VOXEL_FACE_AIR) continue;
            t_out = t_cross;
            face_out = face;
            normal_out = is_pos ? local_axes[axis] : -local_axes[axis];
            is_front_out = ray.ray_direction.dot(normal_out) < FloatC(0);
            return true;
        }
        return false;
    }

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _isRayIntersectingVoxel(
        Kernel* k,
        const Ray& ray,
        RHandle voxel,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_cross_out
    ) {
        int face;
        bool is_front;
        Vec3C normal;
        if (!_voxelFirstBoundary(k, voxel, ray, t_min, t_cross_out, face, is_front, normal))
            return false;
        return t_cross_out <= t_max;
    }

    /**
     * @brief Closest-hit descent over one voxel BVH.
     */
    template <typename Kernel, typename BVHAccessor>
    HOST_DEVICE_INLINE_QUALIFIER static void _findHitVoxelInBVH(
        Kernel* k,
        BVHAccessor root,
        const Ray& ray,
        FloatC t_min,
        FloatC& t_closest_in_out,
        RHandle& voxel_out
    ) {
        if (root.current() == RS_NULL_INDEX) return;
        static constexpr int STACK_SIZE = 32;
        RIndex stack[STACK_SIZE];
        int stack_top = 0;
        stack[stack_top++] = root.current();
        while (stack_top > 0) {
            RIndex idx = stack[--stack_top];
            auto node = root;
            node.moveTo(idx);
            if (node.current() == RS_NULL_INDEX) continue;
            if (node.isLeaf()) {
                FloatC t_enter, t_exit;
                auto bnd = node.bound();
                if (!_isRayIntersectingAABB(
                        ray,
                        Vec3C(bnd.minBound()),
                        Vec3C(bnd.maxBound()),
                        t_min,
                        t_closest_in_out,
                        t_enter,
                        t_exit
                    ))
                    continue;
                auto obj_ptr = node.object();
                if (obj_ptr == nullptr) continue;
                RHandle voxel = obj_ptr->object_handle;
                FloatC t_hit;
                if (_isRayIntersectingVoxel(k, ray, voxel, t_min, t_closest_in_out, t_hit) &&
                    t_hit < t_closest_in_out) {
                    t_closest_in_out = t_hit;
                    voxel_out = voxel;
                }
                continue;
            }

            auto internal_node = node.internalNode();
            FloatC t_left_enter, t_right_enter, t_exit_unused;
            bool is_left_hit = internal_node.left != RS_NULL_INDEX &&
                               _isRayIntersectingAABB(
                                   ray,
                                   Vec3C(internal_node.left_bound.minBound()),
                                   Vec3C(internal_node.left_bound.maxBound()),
                                   t_min,
                                   t_closest_in_out,
                                   t_left_enter,
                                   t_exit_unused
                               );
            bool is_right_hit = internal_node.right != RS_NULL_INDEX &&
                                _isRayIntersectingAABB(
                                    ray,
                                    Vec3C(internal_node.right_bound.minBound()),
                                    Vec3C(internal_node.right_bound.maxBound()),
                                    t_min,
                                    t_closest_in_out,
                                    t_right_enter,
                                    t_exit_unused
                                );
            if (is_left_hit && is_right_hit && stack_top + 2 <= STACK_SIZE) {
                if (t_left_enter < t_right_enter) {
                    stack[stack_top++] = internal_node.right;
                    stack[stack_top++] = internal_node.left;
                } else {
                    stack[stack_top++] = internal_node.left;
                    stack[stack_top++] = internal_node.right;
                }
            } else if (is_left_hit && stack_top + 1 <= STACK_SIZE) {
                stack[stack_top++] = internal_node.left;
            } else if (is_right_hit && stack_top + 1 <= STACK_SIZE) {
                stack[stack_top++] = internal_node.right;
            }
        }
    }

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _findHitVoxel(
        Kernel* k,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        RHandle& voxel_out,
        FloatC& t_out
    ) {
        auto& ctx = k->ctx;
        FloatC t_closest = t_max;
        voxel_out = RS_NULL_HANDLE;
        _findHitVoxelInBVH(k, ctx.static_voxel_bvh.root(), ray, t_min, t_closest, voxel_out);
        _findHitVoxelInBVH(k, ctx.dynamic_voxel_bvh.root(), ray, t_min, t_closest, voxel_out);
        t_out = t_closest;
        return voxel_out != RS_NULL_HANDLE;
    }

    /**
     * @brief Floor triangle narrow-phase: Moller-Trumbore via RS_Triangle.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _isRayIntersectingFloorTriangle(
        Kernel* k,
        const Ray& ray,
        RHandle triangle,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        Vec3C& normal_out
    ) {
        auto& ctx = k->ctx;
        auto& tri = ctx.mesh_triangles.get({triangle});
        Vec3C v0(tri.vertex_0);
        Vec3C v1(tri.vertex_1);
        Vec3C v2(tri.vertex_2);
        FloatC u, v;
        if (!RS_Triangle<FloatC>::rayIntersect(
                ray.ray_origin, ray.ray_direction, v0, v1, v2, t_min, t_max, t_out, u, v
            ))
            return false;
        normal_out = Vec3C(tri.face_normal);
        return true;
    }

    /**
     * @brief Closest-hit floor triangle BVH walk.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool _findHitFloorTriangle(
        Kernel* k,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        Vec3C& normal_out
    ) {
        auto& ctx = k->ctx;
        auto root = ctx.floor_triangle_bvh.root();
        if (root.current() == RS_NULL_INDEX) return false;

        FloatC t_closest = t_max;
        bool found = false;
        static constexpr int STACK_SIZE = 32;
        RIndex stack[STACK_SIZE];
        int stack_top = 0;
        stack[stack_top++] = root.current();
        while (stack_top > 0) {
            RIndex idx = stack[--stack_top];
            auto node = root;
            node.moveTo(idx);
            if (node.current() == RS_NULL_INDEX) continue;
            FloatC t_enter, t_exit;
            auto bnd = node.bound();
            if (!_isRayIntersectingAABB(
                    ray,
                    Vec3C(bnd.minBound()),
                    Vec3C(bnd.maxBound()),
                    t_min,
                    t_closest,
                    t_enter,
                    t_exit
                ))
                continue;
            if (node.isLeaf()) {
                auto obj_ptr = node.object();
                if (obj_ptr == nullptr) continue;
                RHandle tri = obj_ptr->object_handle;
                FloatC t_hit;
                Vec3C tri_normal;
                if (_isRayIntersectingFloorTriangle(
                        k, ray, tri, t_min, t_closest, t_hit, tri_normal
                    ) &&
                    t_hit < t_closest) {
                    t_closest = t_hit;
                    normal_out = tri_normal;
                    found = true;
                }
                continue;
            }
            RIndex l = node.left();
            RIndex r = node.right();
            auto ln = root;
            ln.moveTo(l);
            auto rn = root;
            rn.moveTo(r);
            FloatC t_left_enter, t_right_enter, t_exit_unused;
            bool is_left_hit = ln.current() != RS_NULL_INDEX && _isRayIntersectingAABB(
                                                                    ray,
                                                                    Vec3C(ln.bound().minBound()),
                                                                    Vec3C(ln.bound().maxBound()),
                                                                    t_min,
                                                                    t_closest,
                                                                    t_left_enter,
                                                                    t_exit_unused
                                                                );
            bool is_right_hit = rn.current() != RS_NULL_INDEX && _isRayIntersectingAABB(
                                                                     ray,
                                                                     Vec3C(rn.bound().minBound()),
                                                                     Vec3C(rn.bound().maxBound()),
                                                                     t_min,
                                                                     t_closest,
                                                                     t_right_enter,
                                                                     t_exit_unused
                                                                 );
            if (is_left_hit && is_right_hit && stack_top + 2 <= STACK_SIZE) {
                if (t_left_enter < t_right_enter) {
                    stack[stack_top++] = r;
                    stack[stack_top++] = l;
                } else {
                    stack[stack_top++] = l;
                    stack[stack_top++] = r;
                }
            } else if (is_left_hit && stack_top + 1 <= STACK_SIZE) {
                stack[stack_top++] = l;
            } else if (is_right_hit && stack_top + 1 <= STACK_SIZE) {
                stack[stack_top++] = r;
            }
        }
        t_out = t_closest;
        return found;
    }

    /**
     * @brief Closest hit across the voxel BVHs and the floor within max_range.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool
    _findHit(Kernel* k, const Ray& ray, FloatC max_range, HitRecord& rec_out) {
        auto& ctx = k->ctx;
        FloatC t_voxel;
        RHandle voxel;
        bool voxel_hit = _findHitVoxel(k, ray, RAY_T_MIN, max_range, voxel, t_voxel);

        FloatC t_floor = voxel_hit ? t_voxel : max_range;
        Vec3C floor_normal(0, 0, 1);
        bool floor_hit = false;
        if (ctx.floor_vision_mem.data_type == RSE_VISION_FLOOR_DATA_ARRAY) {
            floor_hit = _findHitFloorTriangle(k, ray, RAY_T_MIN, t_floor, t_floor, floor_normal);
        } else if (ctx.floor_vision_mem.data_type == RSE_VISION_FLOOR_DATA_FLAT) {
            FloatC dz = ray.ray_direction.z;
            if (RS_ABS(dz) > FloatC(1e-8)) {
                FloatC t_ground = (FloatC(ctx.floor.height) - ray.ray_origin.z) / dz;
                if (t_ground > RAY_T_MIN && t_ground < t_floor) {
                    t_floor = t_ground;
                    floor_normal = Vec3C(0, 0, 1);
                    floor_hit = true;
                }
            }
        }

        if (!voxel_hit && !floor_hit) return false;

        bool floor_wins = floor_hit && (!voxel_hit || t_floor < t_voxel);
        if (floor_wins) {
            rec_out.is_floor_hit = true;
            rec_out.voxel = RS_NULL_HANDLE;
            rec_out.t = t_floor;
            rec_out.p = ray.at(t_floor);
            bool is_front = ray.ray_direction.dot(floor_normal) < FloatC(0);
            rec_out.normal = is_front ? floor_normal : -floor_normal;
        } else {
            _readVoxelHit(k, voxel, ray, rec_out);
        }
        return true;
    }

    /*****************************************************************
     * Material-property resolution
     *****************************************************************/

    /**
     * @brief Read the crossed voxel boundary into the hit record: normal, structure id,
     * material id and emission.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    _readVoxelHit(Kernel* k, RHandle voxel, const Ray& ray, HitRecord& rec_out) {
        auto& ctx = k->ctx;
        FloatC t;
        int face;
        bool is_front;
        Vec3C face_normal;
        _voxelFirstBoundary(k, voxel, ray, RAY_T_MIN, t, face, is_front, face_normal);

        rec_out.t = t;
        rec_out.p = ray.at(t);
        rec_out.normal = is_front ? face_normal : -face_normal;
        rec_out.voxel = voxel;
        rec_out.structure_gid = V_G(voxel, structure_gid);
        RHandle vision_material = V_G(voxel, vision_material);
        rec_out.material_gid =
            RS_VisionMaterialKeyUnion{VSM_G(vision_material, vision_material_key)}
                .vision_material_key.material_gid;
        rec_out.emission = FloatC(VSM_G(vision_material, emission_intensity));
    }

    /*****************************************************************
     * Motion / temporal helpers
     *****************************************************************/

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    _computeVoxelVelocity(const Kernel* k, RHandle voxel, const Vec3C& point, Vec3C& velocity_out) {
        auto& ctx = k->ctx;
        if (voxel == RS_NULL_HANDLE) {
            velocity_out = Vec3C::zero();
            return;
        }
        RS_VoxelStateUnion state_union{V_G(voxel, state)};
        if (state_union.voxel_state.is_rigid) {
            RHandle rigid_body = V_G(voxel, rigid_body);
            _computeRigidBodyVelocity(k, rigid_body, point, velocity_out);
        } else {
            RHandle voxel_material = V_G(voxel, voxel_material);
            FloatC mass_inverse = FloatC(VM_G(voxel_material, mass_inverse));
            FloatC inertia_inverse = FloatC(VM_G(voxel_material, moment_inertia_inverse));
            Vec3C lin_vel = Vec3C(V_G(voxel, linear_momentum)) * mass_inverse;
            Vec3C ang_vel = Vec3C(V_G(voxel, angular_momentum)) * inertia_inverse;
            Vec3C voxel_position = Vec3C(V_G(voxel, position));
            velocity_out = lin_vel + ang_vel.cross(point - voxel_position);
        }
    }

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _computeRigidBodyVelocity(
        const Kernel* k,
        RHandle rigid_body,
        const Vec3C& point,
        Vec3C& velocity_out
    ) {
        auto& ctx = k->ctx;
        if (rigid_body == RS_NULL_HANDLE) {
            velocity_out = Vec3C::zero();
            return;
        }
        FloatC mass_inverse = FloatC(RB_G(rigid_body, mass_inverse));
        Vec3C lin_vel = Vec3C(RB_G(rigid_body, linear_momentum)) * mass_inverse;
        Vec3C ang_vel = Mat3C(RB_G(rigid_body, moment_inertia_inverse)) *
                        Vec3C(RB_G(rigid_body, angular_momentum));
        Vec3C com = Vec3C(RB_G(rigid_body, com));
        velocity_out = lin_vel + ang_vel.cross(point - com);
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_RAY_SENSOR_RAY_C_H
