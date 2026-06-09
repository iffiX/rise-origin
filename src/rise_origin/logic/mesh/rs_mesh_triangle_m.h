// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MESH_TRIANGLE_M_H
#define RS_MESH_TRIANGLE_M_H

#include <refl.hpp>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_vec.h"

namespace rise {

namespace logic {

/// Flattened mesh triangle: vertex positions + face normal + tangent stored
/// inline (no shared vertex buffer). Used only by the floor in rise-origin;
/// no general mesh geometry is planned.
class RS_MeshTriangleMem {
   public:
    /// Triangle vertex 0 (world-space, CCW with vertex_1 / vertex_2).
    RVec3f vertex_0 = RVec3f::zero();

    /// Triangle vertex 1.
    RVec3f vertex_1 = RVec3f::zero();

    /// Triangle vertex 2.
    RVec3f vertex_2 = RVec3f::zero();

    /// Precomputed face normal (outward).
    RVec3f face_normal = RVec3f::zero();

    /// Precomputed face tangent.
    RVec3f face_tangent = RVec3f::zero();
};

}  // namespace logic

}  // namespace rise

REFL_AUTO(
    type(rise::logic::RS_MeshTriangleMem),
    field(vertex_0),
    field(vertex_1),
    field(vertex_2),
    field(face_normal),
    field(face_tangent)
)

#endif  // RS_MESH_TRIANGLE_M_H
