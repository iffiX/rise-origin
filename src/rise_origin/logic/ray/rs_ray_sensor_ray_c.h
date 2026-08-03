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
        RHandle vision_material = RS_NULL_HANDLE;
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
     * @brief Nearest crossing of a ray with one deformed quad, tested as two triangles.
     *
     * @details The corners arrive in the winding getVoxelFaceCornerIndices produces, so the cross
     * product of the first and last edge is the outward geometric normal. A deformed quad need
     * not be planar, so both triangles are tested and the nearer crossing wins.
     *
     * @param ray        the ray being traced.
     * @param corner_0   first quad corner in winding order.
     * @param corner_1   second quad corner.
     * @param corner_2   third quad corner, opposite the first.
     * @param corner_3   fourth quad corner.
     * @param t_min    nearest accepted distance along the ray.
     * @param t_max    furthest accepted distance along the ray.
     * @param t_out    distance to the crossing.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _isRayIntersectingQuad(
        const Ray& ray,
        const Vec3C& corner_0,
        const Vec3C& corner_1,
        const Vec3C& corner_2,
        const Vec3C& corner_3,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out
    ) {
        FloatC u, v, t_cross;
        FloatC t_near = t_max;
        bool has_hit = false;
        if (RS_Triangle<FloatC>::rayIntersect(
                ray.ray_origin,
                ray.ray_direction,
                corner_0,
                corner_1,
                corner_2,
                t_min,
                t_near,
                t_cross,
                u,
                v
            )) {
            t_near = t_cross;
            has_hit = true;
        }
        if (RS_Triangle<FloatC>::rayIntersect(
                ray.ray_origin,
                ray.ray_direction,
                corner_0,
                corner_2,
                corner_3,
                t_min,
                t_near,
                t_cross,
                u,
                v
            )) {
            t_near = t_cross;
            has_hit = true;
        }
        if (!has_hit) return false;
        t_out = t_near;
        return true;
    }

    /**
     * @brief Slab test of a ray already in a box's own frame, the box centered on the origin.
     *
     * @details Reports both crossings and the axis each one lies on, so the caller can turn
     * them into faces. The ray starts within the box when the entry distance falls short of
     * t_min.
     *
     * @param local_origin    ray origin in the box frame.
     * @param local_direction ray direction in the box frame.
     * @param half_size       half the box edge length.
     * @param t_min           nearest accepted distance along the ray.
     * @param t_enter_out     distance to the crossing where the ray goes in.
     * @param enter_axis_out  axis that crossing lies on, 0 for x, 1 for y and 2 for z.
     * @param t_exit_out      distance to the crossing where the ray comes out.
     * @param exit_axis_out   axis that crossing lies on.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _isRayIntersectingBox(
        const Vec3C& local_origin,
        const Vec3C& local_direction,
        FloatC half_size,
        FloatC t_min,
        FloatC& t_enter_out,
        int& enter_axis_out,
        FloatC& t_exit_out,
        int& exit_axis_out
    ) {
        Vec3C direction_reciprocal = local_direction.reciprocalFast();
        Vec3C t0 = (Vec3C::full(-half_size) - local_origin).elemMul(direction_reciprocal);
        Vec3C t1 = (Vec3C::full(half_size) - local_origin).elemMul(direction_reciprocal);
        Vec3C t_min_vec = t0.min(t1);
        Vec3C t_max_vec = t0.max(t1);
        t_enter_out = t_min_vec.max();
        t_exit_out = t_max_vec.min();
        if (t_exit_out < t_enter_out || t_exit_out < t_min) return false;
        enter_axis_out = t_min_vec.argmax();
        exit_axis_out = t_max_vec.argmin();
        return true;
    }

    /**
     * @brief First real boundary face along the ray inside a voxel hexahedron.
     *
     * @details The leaf carries eight corners that deformation can pull out of any box shape, so
     * each face is crossed as a quad rather than a slab. Only faces the roles expose are tested:
     * an interior face can never carry a boundary, an entry is reported for any exposed face, and
     * an exit only where the face meets air.
     *
     * @param leaf         ray-BVH leaf holding the corners and roles.
     * @param ray          the ray being traced.
     * @param t_min        nearest accepted distance along the ray.
     * @param t_max        furthest accepted distance along the ray.
     * @param t_out        distance to the crossing.
     * @param face_out     crossed face index in RSE_LinkDirection order.
     * @param is_front_out whether the ray meets the face from outside the voxel.
     * @param normal_out   outward normal of the crossed face, unit length.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _voxelHexahedronFirstBoundary(
        RState metadata,
        const RS_VoxelRayBVHObjectIdentifier& leaf,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        int& face_out,
        bool& is_front_out,
        Vec3C& normal_out
    ) {
        Vec3C center = Vec3C(leaf.position);

        FloatC t_near = t_max;
        bool has_hit = false;
        Vec3C best_normal;
        for (int face = 0; face < 6; face++) {
            RSE_VoxelFaceRole role = getRayLeafFaceRole(metadata, face);
            if (role == RSE_VOXEL_FACE_INTERIOR) continue;

            int index_0, index_1, index_2, index_3;
            getVoxelFaceCornerIndices(
                face >> 1, (face & 1) == 0, index_0, index_1, index_2, index_3
            );
            const auto& corner = leaf.shape.hexahedron.corner_offset;
            Vec3C corner_0 = center + corner[index_0].template toVec3D<FloatC>();
            Vec3C corner_1 = center + corner[index_1].template toVec3D<FloatC>();
            Vec3C corner_2 = center + corner[index_2].template toVec3D<FloatC>();
            Vec3C corner_3 = center + corner[index_3].template toVec3D<FloatC>();

            // The diagonals give the normal of the whole face, so either triangle of a face
            // bent out of plane reads the same way.
            Vec3C normal = (corner_2 - corner_0).cross(corner_3 - corner_1);
            bool is_front = ray.ray_direction.dot(normal) < FloatC(0);

            // One interface belongs to two voxels, and the voxel being entered reports it, so
            // a face met from within is skipped unless it opens onto air, where there is no
            // other voxel to report it.
            if (!is_front && role != RSE_VOXEL_FACE_AIR) continue;

            FloatC t_cross;
            if (!_isRayIntersectingQuad(
                    ray, corner_0, corner_1, corner_2, corner_3, t_min, t_near, t_cross
                ))
                continue;

            t_near = t_cross;
            face_out = face;
            is_front_out = is_front;
            best_normal = normal;
            has_hit = true;
        }
        if (!has_hit) return false;

        // Only the winning face pays for a root.
        t_out = t_near;
        normal_out = best_normal.normalized();
        return true;
    }

    /**
     * @brief First real boundary face along the ray inside a voxel box.
     *
     * @details A non-deformable voxel keeps its box, so one slab test covers all six faces and
     * only the two it crosses need a role. The entry face comes first when the ray starts
     * outside, then the exit face, and the first that is a real boundary wins.
     *
     * @param leaf         ray-BVH leaf holding the box and the roles.
     * @param ray          the ray being traced.
     * @param t_min        nearest accepted distance along the ray.
     * @param t_max        furthest accepted distance along the ray.
     * @param t_out        distance to the crossing.
     * @param face_out     crossed face index in RSE_LinkDirection order.
     * @param is_front_out whether the ray meets the face from outside the voxel.
     * @param normal_out   outward normal of the crossed face, unit length.
     */
    HOST_DEVICE_INLINE_QUALIFIER static bool _voxelBoxFirstBoundary(
        RState metadata,
        const RS_VoxelRayBVHObjectIdentifier& leaf,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        int& face_out,
        bool& is_front_out,
        Vec3C& normal_out
    ) {
        Quat3C orientation = Quat3C(leaf.shape.box.orientation);
        FloatC half_size = FloatC(0.5) * FloatC(leaf.shape.box.size);

        Vec3C local_origin = orientation.rotateVec3DInv(ray.ray_origin - Vec3C(leaf.position));
        Vec3C local_direction = orientation.rotateVec3DInv(ray.ray_direction);

        FloatC t_enter, t_exit;
        int enter_axis, exit_axis;
        if (!_isRayIntersectingBox(
                local_origin,
                local_direction,
                half_size,
                t_min,
                t_enter,
                enter_axis,
                t_exit,
                exit_axis
            ))
            return false;

        for (int pass = 0; pass < 2; ++pass) {
            bool is_entry = pass == 0;
            if (is_entry && t_enter < t_min) continue;
            FloatC t_cross = is_entry ? t_enter : t_exit;
            if (t_cross > t_max) return false;

            int axis = is_entry ? enter_axis : exit_axis;
            Vec3C local_hit = local_origin + local_direction * t_cross;
            bool is_positive = local_hit[axis] > FloatC(0);
            int face = 2 * axis + (is_positive ? 0 : 1);
            RSE_VoxelFaceRole role = getRayLeafFaceRole(metadata, face);
            if (role == RSE_VOXEL_FACE_INTERIOR) continue;

            // One interface belongs to two voxels, and the voxel being entered reports it, so
            // a face met from within is skipped unless it opens onto air, where there is no
            // other voxel to report it.
            if (!is_entry && role != RSE_VOXEL_FACE_AIR) continue;

            FloatC side = is_positive ? FloatC(1) : FloatC(-1);
            Vec3C local_normal(
                axis == 0 ? side : FloatC(0),
                axis == 1 ? side : FloatC(0),
                axis == 2 ? side : FloatC(0)
            );

            t_out = t_cross;
            face_out = face;
            normal_out = orientation.rotateVec3D(local_normal);
            is_front_out = is_entry;
            return true;
        }
        return false;
    }

    /** @brief First real boundary face along the ray, taking whichever shape the leaf holds. */
    HOST_DEVICE_INLINE_QUALIFIER static bool _voxelFirstBoundary(
        const RS_VoxelRayBVHObjectIdentifier& leaf,
        const Ray& ray,
        FloatC t_min,
        FloatC t_max,
        FloatC& t_out,
        int& face_out,
        bool& is_front_out,
        Vec3C& normal_out
    ) {
        RState metadata = leaf.object_metadata;
        RS_VoxelRayBVHLeafMetadataUnion m_union{metadata};
        if (m_union.meta.is_box)
            return _voxelBoxFirstBoundary(
                metadata, leaf, ray, t_min, t_max, t_out, face_out, is_front_out, normal_out
            );
        return _voxelHexahedronFirstBoundary(
            metadata, leaf, ray, t_min, t_max, t_out, face_out, is_front_out, normal_out
        );
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
        HitRecord& rec_in_out
    ) {
        if (root.current() == RS_NULL_INDEX) return;
        RIndex stack[RS_RUNTIME_BVH_STACK_SIZE];
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
                FloatC t_hit;
                int face;
                bool is_front;
                Vec3C normal;
                if (!_voxelFirstBoundary(
                        *obj_ptr, ray, t_min, t_closest_in_out, t_hit, face, is_front, normal
                    ))
                    continue;
                t_closest_in_out = t_hit;
                rec_in_out.t = t_hit;
                rec_in_out.normal = is_front ? normal : -normal;
                rec_in_out.voxel = obj_ptr->object_handle;
                rec_in_out.vision_material = obj_ptr->vision_material;
                rec_in_out.structure_gid = obj_ptr->structure_gid;
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
            if (is_left_hit && is_right_hit && stack_top + 2 <= RS_RUNTIME_BVH_STACK_SIZE) {
                if (t_left_enter < t_right_enter) {
                    stack[stack_top++] = internal_node.right;
                    stack[stack_top++] = internal_node.left;
                } else {
                    stack[stack_top++] = internal_node.left;
                    stack[stack_top++] = internal_node.right;
                }
            } else if (is_left_hit && stack_top + 1 <= RS_RUNTIME_BVH_STACK_SIZE) {
                stack[stack_top++] = internal_node.left;
            } else if (is_right_hit && stack_top + 1 <= RS_RUNTIME_BVH_STACK_SIZE) {
                stack[stack_top++] = internal_node.right;
            }
        }
    }

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static bool
    _findHitVoxel(Kernel* k, const Ray& ray, FloatC t_min, FloatC t_max, HitRecord& rec_in_out) {
        auto& ctx = k->ctx;
        FloatC t_closest = t_max;
        rec_in_out.voxel = RS_NULL_HANDLE;
        _findHitVoxelInBVH(k, ctx.static_voxel_ray_bvh.root(), ray, t_min, t_closest, rec_in_out);
        _findHitVoxelInBVH(k, ctx.dynamic_voxel_ray_bvh.root(), ray, t_min, t_closest, rec_in_out);
        return rec_in_out.voxel != RS_NULL_HANDLE;
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
        RIndex stack[RS_RUNTIME_BVH_STACK_SIZE];
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
            if (is_left_hit && is_right_hit && stack_top + 2 <= RS_RUNTIME_BVH_STACK_SIZE) {
                if (t_left_enter < t_right_enter) {
                    stack[stack_top++] = r;
                    stack[stack_top++] = l;
                } else {
                    stack[stack_top++] = l;
                    stack[stack_top++] = r;
                }
            } else if (is_left_hit && stack_top + 1 <= RS_RUNTIME_BVH_STACK_SIZE) {
                stack[stack_top++] = l;
            } else if (is_right_hit && stack_top + 1 <= RS_RUNTIME_BVH_STACK_SIZE) {
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
        bool voxel_hit = _findHitVoxel(k, ray, RAY_T_MIN, max_range, rec_out);
        FloatC t_voxel = voxel_hit ? rec_out.t : max_range;

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
            rec_out.is_floor_hit = false;
            rec_out.p = ray.at(rec_out.t);
            _readVoxelHit(k, rec_out);
        }
        return true;
    }

    /*****************************************************************
     * Material-property resolution
     *****************************************************************/

    /**
     * @brief Read the hit voxel's material id and emission into the hit record. The BVH walk
     * already resolved the crossed boundary, so only the material remains.
     */
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void _readVoxelHit(Kernel* k, HitRecord& rec_in_out) {
        auto& ctx = k->ctx;
        RHandle vision_material = rec_in_out.vision_material;
        rec_in_out.material_gid =
            RS_VisionMaterialKeyUnion{VSM_G(vision_material, vision_material_key)}
                .vision_material_key.material_gid;
        rec_in_out.emission = FloatC(VSM_G(vision_material, emission_intensity));
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
