// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <algorithm>
#include <array>
#include <random>
#include <tuple>
#include <unordered_set>

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/device/algorithm/rs_generate.h"
#include "rise_origin/logic/record/rs_basic_record_c.h"
#include "rise_origin/logic/rigid/rs_constraint_c.h"
#include "rise_origin/logic/rigid/rs_rigid_body_c.h"
#include "rise_origin/logic/rigid/rs_rigid_body_surface_voxel_c.h"
#include "rise_origin/logic/soft/rs_link_c.h"
#include "rise_origin/logic/soft/rs_link_material_c.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/logic/soft/rs_voxel_material_c.h"
#include "rise_origin/logic/utils/rs_launch_c.h"
#include "rise_origin/utils/concurrency/rs_barrier.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rise_origin/utils/time/rs_timer.h"
#include "rs_kernel_material_manager.h"
#include "rs_kernel_structure_voxel_manager.h"

using namespace std;
using namespace rise::device;
using namespace rise::logic;

namespace rise {

/*****************************************************************
 * RS_KernelStructureVoxelManager
 *****************************************************************/
RS_KernelStructureVoxelManager::RS_KernelStructureVoxelManager(
    RS_KernelManager *kernel_manager,
    const SupportedRSC &structure_rsc
)
    : _kernel_manager(kernel_manager), _rsc(structure_rsc) {
    // Reject a structure whose bounding box leaves the world before doing any expensive work.
    RBound3f world_bound = _computeWorldBound(_structure());
    RFloat range = RFLOAT(RS_WORLD_MAX_RANGE);
    RBound3f world_range(-range, -range, -range, range, range, range);
    if (world_bound.isValid() && !world_range.isContaining(world_bound)) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure \"{}\" bounding box [({}, {}, {}), ({}, {}, {})] exceeds the world range of "
            "+/-{} m in at least one axis",
            _structure().name,
            world_bound.min_x,
            world_bound.min_y,
            world_bound.min_z,
            world_bound.max_x,
            world_bound.max_y,
            world_bound.max_z,
            range
        ));
    }

    _entity_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_ENTITY, _structure().name);
    _record_registration = RS_KernelGidRegistration(
        _kernel_manager, RSE_KERNEL_GID_RECORD, _structure().name + ":record"
    );
    _signal_registration = RS_KernelGidRegistration(
        _kernel_manager, RSE_KERNEL_GID_SIGNAL, _structure().name + ":signal"
    );
    _structure_gid = _entity_registration.gid();
    _record_gid = _record_registration.gid();
    _signal_gid = _signal_registration.gid();

    _scoped_host_provider =
        std::make_unique<RS_MemoryScopedProvider>(_kernel_manager->getHostProvider());
    _scoped_device_provider =
        std::make_unique<RS_MemoryScopedProvider>(_kernel_manager->getDeviceProvider());

    RS_Timer timer;

    _is_constrained = !_structure().is_fixed && !_structure().constraints.empty();

    _resolveMaterials();

    for (auto &body_rsc : _structure().bodies) {
        _addBody(body_rsc);
    }
    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Structure {} build body time: {}",
        _structure().name,
        timer.elapsedTimeString()
    );
    timer.reset();

    if (!_structure().is_fixed) {
        for (auto &constraint_rsc : _structure().constraints) {
            _addConstraint(constraint_rsc);
        }

        _addSignals();
    }
    _addMaterials();

    if (!_structure().is_fixed) {
        _colorConstraintJoints();
    }
    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Structure {} build constraint, material and optimize time: {}",
        _structure().name,
        timer.elapsedTimeString()
    );

    // Validate the characterization before the kernel context is touched at commit.
    _checkStructureLimits();
    _checkRigidBodyInertia();

    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Structure {} has voxel: {} link: {} rigid body: {} rigid body surface voxel: {} joint: {}",
        _structure().name,
        _voxel_num,
        _link_num,
        _rigid_body_num,
        _rigid_body_surface_voxel_num,
        _joint_num
    );
}

RS_KernelStructureVoxelManager::~RS_KernelStructureVoxelManager() {
    for (auto &voxel_tmp_index_map : _voxel_tmp_index_map) voxel_tmp_index_map.second.free();
}

void RS_KernelStructureVoxelManager::commit() {
    RS_Timer timer;
    _kernel_manager->enterKernelDataCriticalZone();
    _initKernelSignalData();
    _initKernelData();
    _initKernelRecordData();
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;
    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Structure {} init kernel data time: {}",
        _structure().name,
        timer.elapsedTimeString()
    );
}

void RS_KernelStructureVoxelManager::decommit() {
    if (_is_kernel_data_inited) {
        _kernel_manager->enterKernelDataCriticalZone();
        _freeKernelRecordData();
        _freeKernelData();
        _freeKernelSignalData();
        _kernel_manager->exitKernelDataCriticalZone();
        _is_kernel_data_inited = false;
    }
}

void RS_KernelStructureVoxelManager::freeze() {
    // A permanently fixed structure has no dynamic voxels and no active compute layouts to pause.
    if (_structure().is_fixed || _is_frozen) return;
    _kernel_manager->enterKernelDataCriticalZone();
    _freezeKernelData(true);
    _kernel_manager->exitKernelDataCriticalZone();
    _is_frozen = true;
}

void RS_KernelStructureVoxelManager::unfreeze() {
    if (_structure().is_fixed || !_is_frozen) return;
    _kernel_manager->enterKernelDataCriticalZone();
    _freezeKernelData(false);
    _kernel_manager->exitKernelDataCriticalZone();
    _is_frozen = false;
}

bool RS_KernelStructureVoxelManager::isFrozen() const { return _is_frozen; }

void RS_KernelStructureVoxelManager::reference() { _use_count += 1; }

void RS_KernelStructureVoxelManager::dereference() {
    if (_use_count == 0) {
        RS_THROW(RS_HostExecutionException(
            "Structure '{}' use_count underflow on dereference", _structure().name
        ));
    }
    _use_count -= 1;
}

RSize RS_KernelStructureVoxelManager::getUseCount() const { return _use_count; }

void RS_KernelStructureVoxelManager::syncRecorderRecords() {
    _recorder_buffer.swapAndCopy(true, false, false);
}

void RS_KernelStructureVoxelManager::waitForRecorderRecordsSync() {
    _recorder_buffer.waitForCopyFinish();
}

std::vector<RS_WorldStructureVoxelRecord> RS_KernelStructureVoxelManager::getRecorderRecords() {
    vector<RS_WorldStructureVoxelRecord> records;
    for (RIndex i = 0, total = _recorder_buffer.getHostNum(); i < total; ++i) {
        RS_WorldStructureVoxelRecord record;
        record.initFromMemory(
            _recorder_buffer.getHostChunk(i).ptr(),
            _voxel_record_num,
            _link_record_num,
            _rigid_body_record_num,
            _joint_record_num
        );
        record.structure_gid = _structure_gid;
        record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
        records.emplace_back(record);
    }
    return records;
}

bool RS_KernelStructureVoxelManager::hasControllerSignal() {
    return !_structure().is_fixed && (_structure().signal->expansion_signal_num > 0 ||
                                      _structure().signal->hinge_signal_num > 0 ||
                                      _structure().signal->prismatic_signal_num > 0);
}

void RS_KernelStructureVoxelManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());
    }
}

RS_WorldStructureVoxelSignal RS_KernelStructureVoxelManager::getControllerSignal() {
    if (hasControllerSignal()) {
        RS_WorldStructureVoxelSignal signal;
        signal.initFromMemory(
            _host_signal_buffer.ptr(),
            _expansion_signal_num,
            _hinge_signal_num,
            _prismatic_signal_num
        );
        signal.structure_gid = _structure_gid;
        signal.device = _host_signal_buffer.device();
        return signal;
    }
    return {};
}

bool RS_KernelStructureVoxelManager::hasControllerRecord() {
    return _voxel_record_num > 0 || _link_record_num > 0 || _rigid_body_record_num > 0 ||
           _joint_record_num > 0;
}

RS_WorldStructureVoxelRecord RS_KernelStructureVoxelManager::getControllerRecord() {
    RS_WorldStructureVoxelRecord record;
    record.initFromMemory(
        _controller_buffer.getHostChunk(0).ptr(),
        _voxel_record_num,
        _link_record_num,
        _rigid_body_record_num,
        _joint_record_num
    );
    record.structure_gid = _structure_gid;
    record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
    return record;
}

void RS_KernelStructureVoxelManager::syncControllerRecord() {
    if (hasControllerRecord()) {
        _controller_buffer.sync(true);
    }
}

const RS_KernelStructureVoxelManager::SupportedRSC &RS_KernelStructureVoxelManager::getRSC() const {
    return _rsc;
}

const unordered_set<RIndex> &RS_KernelStructureVoxelManager::getDependentMaterialGID() const {
    return _dependent_material_gid_set;
}

std::string RS_KernelStructureVoxelManager::getEntityName() const { return _structure().name; }

std::string RS_KernelStructureVoxelManager::getRecordName() const {
    return _structure().record->name;
}

std::string RS_KernelStructureVoxelManager::getSignalName() const {
    return _structure().signal->name;
}

std::string RS_KernelStructureVoxelManager::getRecordControllerName() const {
    return _structure().record->target_controller_reference;
}

std::string RS_KernelStructureVoxelManager::getRecordRecorderName() const {
    return _structure().record->target_recorder_reference;
}

std::string RS_KernelStructureVoxelManager::getSignalControllerName() const {
    return _structure().signal->source_controller_name;
}

RIndex RS_KernelStructureVoxelManager::getEntityGid() const { return _structure_gid; }

RIndex RS_KernelStructureVoxelManager::getRecordGid() const { return _record_gid; }

RIndex RS_KernelStructureVoxelManager::getSignalGid() const { return _signal_gid; }

RSize RS_KernelStructureVoxelManager::getVoxelNum() const { return _voxel_num; }

RSize RS_KernelStructureVoxelManager::getEmissiveVoxelNum() const { return _emissive_voxel_num; }

RSize RS_KernelStructureVoxelManager::getLinkNum() const { return _link_num; }

RSize RS_KernelStructureVoxelManager::getRigidBodyNum() const { return _rigid_body_num; }

RSize RS_KernelStructureVoxelManager::getJointNum() const { return _joint_num; }

RSize RS_KernelStructureVoxelManager::getMotorNum() const { return _motor_num; }

RSize RS_KernelStructureVoxelManager::getConstraintJointNum() const {
    return _constraint_joint_num;
}

RSize RS_KernelStructureVoxelManager::getConstraintJointColorNum() const {
    return _constraint_joint_color_num;
}

RSize RS_KernelStructureVoxelManager::getExpansionSignalNum() const {
    return _expansion_signal_num;
}

RSize RS_KernelStructureVoxelManager::getHingeSignalNum() const { return _hinge_signal_num; }

RSize RS_KernelStructureVoxelManager::getPrismaticSignalNum() const {
    return _prismatic_signal_num;
}

RIndex RS_KernelStructureVoxelManager::findRigidBodySID(RIndex body_sid, RIndex segment_bid) const {
    auto it = _rigid_body_sid_map.find({body_sid, segment_bid});
    if (it == _rigid_body_sid_map.end()) return RS_NULL_INDEX;
    return it->second;
}

RVec3rf RS_KernelStructureVoxelManager::getRigidBodyPositionFromRelativeAnchor(
    RIndex body_sid,
    RIndex rigid_body_sid,
    const RVec3rf &relative_anchor
) const {
    if (rigid_body_sid >= _rigid_body_num)
        RS_THROW(RS_InvalidArgumentException("Invalid rigid body sid {}", rigid_body_sid));
    auto &body_rsc = _structure().bodies[body_sid];
    auto relative_diff = relative_anchor - body_rsc.relative_origin_position;
    auto body_local_anchor = body_rsc.relative_orientation.inverse().rotateVec3D(relative_diff);
    auto result = body_local_anchor - _rigid_bodies[rigid_body_sid].origin_b_position;
    return result;
}

RSE_StructureVoxelMaterialType
RS_KernelStructureVoxelManager::getSegmentPhysicalType(RIndex body_sid, RIndex segment_bid) const {
    // Check if the segment exists by looking for it in voxels
    auto it = _segment_phys_type_map.find({body_sid, segment_bid});
    if (it != _segment_phys_type_map.end()) {
        return it->second;
    } else {
        RS_THROW(RS_InvalidArgumentException(
            "Segment with body_sid {} and segment_bid {} not found in structure {}",
            body_sid,
            segment_bid,
            _structure().name
        ));
    }
}

RSize RS_KernelStructureVoxelManager::getSegmentVoxelNumInRange(
    RIndex body_sid,
    RIndex segment_bid,
    const RBound3ri &voxel_range
) const {
    RSize count = 0;

    for (const auto &voxel : _voxels) {
        // Check if voxel belongs to the specified body and segment
        if (voxel.body_sid != body_sid || voxel.segment_bid != segment_bid) {
            continue;
        }

        // Check if voxel is soft (not part of a rigid body)
        if (voxel.rigid_body_sid != RS_NULL_INDEX) {
            continue;
        }

        // Check if voxel is within the specified range
        if (voxel.index_x >= voxel_range.min_x && voxel.index_x <= voxel_range.max_x &&
            voxel.index_y >= voxel_range.min_y && voxel.index_y <= voxel_range.max_y &&
            voxel.index_z >= voxel_range.min_z && voxel.index_z <= voxel_range.max_z) {
            count++;
        }
    }

    return count;
}

void RS_KernelStructureVoxelManager::_resolveMaterials() {
    auto &mat_mgr = _kernel_manager->getKernelMaterialManager();

    // Resolve each material reference once into _mat_resolved_data, indexed by material_sid (the
    // structure's material reference index). A reference is acquired on the first occurrence of
    // each gid so the material cannot be removed while this structure is alive.
    _mat_resolved_data.resize(_structure().material_references.size());

    for (auto &&[material_sid, material_reference] : enumerate(_structure().material_references)) {
        RIndex material_gid = mat_mgr.findMaterial(material_reference);
        if (material_gid == RS_NULL_INDEX) {
            RS_THROW(RS_InvalidCharacterizationException(
                "Material reference \"{}\" (index {}) in structure characterization doesn't exist",
                material_reference,
                material_sid
            ));
        }

        MaterialResolvedData &resolved = _mat_resolved_data[material_sid];
        resolved.material_gid = material_gid;
        resolved.name = material_reference;

        const auto &name = material_reference;
        const bool has_fixed = mat_mgr.has(name, "fixed");
        const bool has_rigid = mat_mgr.has(name, "rigid");
        const bool has_atlas = mat_mgr.has(name, "atlas");
        const bool has_soft = mat_mgr.has(name, "soft");

        if ((has_rigid || has_atlas) && has_fixed) {
            // A rigid or atlas material plus fixed is an immovable (anchored) rigid body.
            resolved.phys_type = RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED;
        } else if (has_fixed) {
            resolved.phys_type = RSE_STRUCTURE_VOXEL_MATERIAL_FIXED;
        } else if (has_rigid || has_atlas) {
            // An atlas material is a movable rigid body with prescribed mass, com and inertia.
            resolved.phys_type = RSE_STRUCTURE_VOXEL_MATERIAL_RIGID;
        } else if (has_soft) {
            resolved.phys_type = RSE_STRUCTURE_VOXEL_MATERIAL_SOFT;
        } else {
            RS_THROW(RS_InvalidCharacterizationException(
                "Material \"{}\" has no physical property attached", name
            ));
        }

        if (mat_mgr.has(name, "density")) {
            resolved.density = mat_mgr.query(name, "density").as<RFloat>();
        }
        if (resolved.phys_type == RSE_STRUCTURE_VOXEL_MATERIAL_SOFT) {
            auto soft_tuple = mat_mgr.query(name, "soft")
                                  .as<std::tuple<RFloat, RFloat, RFloat, RFloat, RFloat>>();
            resolved.min_expansion = std::get<0>(soft_tuple);
            resolved.max_expansion = std::get<1>(soft_tuple);
            resolved.elastic_modulus = std::get<2>(soft_tuple);
            resolved.fail_stress = std::get<3>(soft_tuple);
            resolved.poissons_ratio = std::get<4>(soft_tuple);
        }
        if (has_atlas) {
            resolved.is_atlas = true;
            resolved.mass = mat_mgr.query(name, "mass").as<RFloat>();
            resolved.com_position = mat_mgr.query(name, "com_position").as<RVec3rf>();
            resolved.inertia = mat_mgr.query(name, "inertia").as<RMat3rf>();
        }
        if (mat_mgr.has(name, "is_transparent")) {
            resolved.is_transparent = mat_mgr.query(name, "is_transparent").as<bool>();
        }
        if (mat_mgr.has(name, "is_emissive")) {
            resolved.is_emissive = mat_mgr.query(name, "is_emissive").as<bool>();
        }

        if (_dependent_material_gid_set.insert(material_gid).second) {
            _material_references.emplace_back(_kernel_manager, material_reference);
        }
    }
}

void RS_KernelStructureVoxelManager::_addBody(const rsc::RSC_StructureVoxelBody &body_rsc) {
    RS_Timer timer;

    RIndex body_sid = _body_num++;

    RSize thread_count = _kernel_manager->getThreadPool()->size();
    RS_Barrier barrier(thread_count);

    _voxel_tmp_index_map[body_sid].init(_scoped_host_provider.get());

    auto initial_body_orientation = _structure().orientation * body_rsc.relative_orientation;
    auto initial_body_origin_position =
        _structure().origin_position +
        _structure().orientation.rotateVec3D(body_rsc.relative_origin_position);
    auto cuboids = _partitionSpace(body_rsc.x_voxels, body_rsc.y_voxels, body_rsc.z_voxels);

    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Add body {} preparation time {}",
        body_sid,
        timer.elapsedTimeString()
    );
    timer.reset();

    //----------------------------------------------------------------
    // Step 1: Create temporary voxels
    //----------------------------------------------------------------

    // Note: some of the temporary voxels, such as rigid voxels that's
    // not the surface of the rigid body, will not be added to the
    // global _voxels array

    // Thread local info
    vector<RCoord3> local_min_bound(thread_count, RCoord3::upperBound());
    vector<RCoord3> local_max_bound(thread_count, RCoord3::lowerBound());
    // Note: this is the actual physical bound, not coordinate
    vector<RBound3f> local_space_bound(thread_count, RBound3f::lowest());
    vector<RSize> local_voxel_count(thread_count, 0);
    vector<RSize> local_voxel_offset(thread_count, 0);
    vector<RSize> local_emissive_count(thread_count, 0);

    // Global info
    RCoord3 actual_min_bound = RCoord3::upperBound();
    RCoord3 actual_max_bound = RCoord3::lowerBound();
    RSize cuboid_count = cuboids.size();
    RSize voxel_block_size = RS_CEIL_DIV(cuboid_count, thread_count);

    auto &tmp_voxels = _all_voxels[body_sid];
    RSize tmp_voxel_num = 0;

    _kernel_manager->getThreadPool()
        ->submitLoop(
            0,
            thread_count,
            [&](RIndex t) {
                RCoord3 local_min_bound_v = RCoord3::upperBound();
                RCoord3 local_max_bound_v = RCoord3::lowerBound();

                std::vector<RIndex> local_array_indices;
                std::vector<RCoord3> local_voxel_coords;

                // -------------------------------------
                // Phase 1: Collect valid voxel indices.
                // -------------------------------------
                RIndex start_c = t * voxel_block_size;
                RIndex end_c = RS_MIN(start_c + voxel_block_size, cuboid_count);

                // Gather array indices of all non-empty voxels in per-thread vector.
                for (RIndex c = start_c; c < end_c; c++) {
                    const auto &cub = cuboids[c];
                    for (RIndex z = cub.z_start; z < cub.z_end; z++) {
                        for (RIndex y = cub.y_start; y < cub.y_end; y++) {
                            for (RIndex x = cub.x_start; x < cub.x_end; x++) {
                                RSize array_index = z * body_rsc.x_voxels * body_rsc.y_voxels +
                                                    y * body_rsc.x_voxels + x;
                                RCoord3 voxel_coord(
                                    static_cast<RSignedIndex>(x),
                                    static_cast<RSignedIndex>(y),
                                    static_cast<RSignedIndex>(z)
                                );

                                if (body_rsc.material_reference_sid[array_index] != RS_NULL_INDEX) {
                                    local_array_indices.push_back(array_index);
                                    local_voxel_coords.push_back(voxel_coord);
                                    local_max_bound_v =
                                        local_max_bound_v.max(voxel_coord + RCoord3::one());
                                    local_min_bound_v = local_min_bound_v.min(voxel_coord);
                                }
                            }
                        }
                    }
                }

                local_min_bound[t] = local_min_bound_v;
                local_max_bound[t] = local_max_bound_v;
                local_voxel_count[t] = local_array_indices.size();
                barrier.wait();

                // ----------------------------------------------------------
                // Phase 2: Compute global offsets in the main (t==0) thread.
                // ----------------------------------------------------------
                if (t == 0) {
                    actual_min_bound = local_min_bound[0];
                    actual_max_bound = local_max_bound[0];
                    local_voxel_offset[0] = 0;
                    for (RIndex i = 1; i < thread_count; i++) {
                        actual_min_bound = actual_min_bound.min(local_min_bound[i]);
                        actual_max_bound = actual_max_bound.max(local_max_bound[i]);
                        local_voxel_offset[i] =
                            local_voxel_offset[i - 1] + local_voxel_count[i - 1];
                    }

                    // The total number of newly added voxels is sum of all local_count.
                    tmp_voxel_num = local_voxel_offset[thread_count - 1] + local_voxel_count.back();

                    // Resize the temporary voxel array
                    tmp_voxels.resize(tmp_voxel_num);
                }
                barrier.wait();

                // -----------------------------------------
                // Phase 3: Fill the voxel data in parallel.
                // -----------------------------------------
                RIndex local_offset = local_voxel_offset[t];
                RBound3f local_space_bound_v = RBound3f::lowest();

                for (RSize i = 0; i < local_array_indices.size(); i++) {
                    RSize array_index = local_array_indices[i];
                    const RCoord3 &voxel_coord = local_voxel_coords[i];

                    auto voxel = VoxelData::preset();

                    voxel.tmp_index = local_offset + i;

                    // Note: voxel.sid is not set here, it will be set after we find
                    // the actual needed voxels

                    voxel.body_sid = body_sid;
                    voxel.array_index = array_index;

                    voxel.index_x = voxel_coord.x;
                    voxel.index_y = voxel_coord.y;
                    voxel.index_z = voxel_coord.z;

                    voxel.material_sid = body_rsc.material_reference_sid[array_index];

                    voxel.segment_bid = body_rsc.segment_bid[array_index];

                    voxel.b_position = RVec3rf(
                        (RFLOAT(voxel_coord.x) + RFLOAT(0.5)) * body_rsc.voxel_size,
                        (RFLOAT(voxel_coord.y) + RFLOAT(0.5)) * body_rsc.voxel_size,
                        (RFLOAT(voxel_coord.z) + RFLOAT(0.5)) * body_rsc.voxel_size
                    );
                    voxel.g_position = initial_body_origin_position +
                                       initial_body_orientation.rotateVec3D(voxel.b_position);
                    voxel.g_orientation = initial_body_orientation;

                    // 0.87 > sqrt(3)/2, diagonal length of voxel
                    RFloat radius = body_rsc.voxel_size * RFLOAT(0.87);
                    RBound3f voxel_bound = RBound3f(
                        voxel.g_position - RVec3rf(radius, radius, radius),
                        voxel.g_position + RVec3rf(radius, radius, radius)
                    );
                    local_space_bound_v = local_space_bound_v.unionWith(voxel_bound);

                    if (_mat_resolved_data[voxel.material_sid].is_emissive) {
                        ++local_emissive_count[t];
                    }

                    // material_type is the physical kind; rigid_body_sid is assigned later in the
                    // gather-rigid-body-voxel step, after the segment-type pass emits rigid bodies.
                    voxel.material_type = _mat_resolved_data[voxel.material_sid].phys_type;
                    if (!body_rsc.expansion_signal_sid.empty()) {
                        voxel.expansion_signal_sid = body_rsc.expansion_signal_sid[array_index];
                    }
                    if (!body_rsc.expansion_control_mode.empty()) {
                        voxel.expansion_control_mode = body_rsc.expansion_control_mode[array_index];
                    }

                    // Write it back into our temporary voxel array
                    tmp_voxels[local_offset + i] = voxel;
                }

                local_space_bound[t] = local_space_bound_v;
            }
        )
        .wait();

    // Compute body space bound
    RBound3f body_space_bound = local_space_bound[0];
    for (RIndex i = 1; i < local_space_bound.size(); i++) {
        body_space_bound = body_space_bound.unionWith(local_space_bound[i]);
    }

    // Sum per-thread emissive counts into the structure-wide counter.
    for (RSize n : local_emissive_count) _emissive_voxel_num += n;

    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Add body {} create temporary voxel time {}",
        body_sid,
        timer.elapsedTimeString()
    );
    timer.reset();

    //----------------------------------------------------------------
    // Step 2: Process segment types
    //----------------------------------------------------------------
    // Pass 1: record, per segment, which materials it contains as a bitmask over material
    // references. This is light (one bit set per temporary voxel) and dedups materials per segment.
    // A std::map keeps the segment order deterministic so rigid body ids are stable.
    std::map<RIndex, std::vector<bool>> segment_material_flags;
    for (auto &voxel : tmp_voxels) {
        // Pass the vector constructor arguments so the bitmask is built only on first insert.
        auto [it, inserted] =
            segment_material_flags.try_emplace(voxel.segment_bid, _mat_resolved_data.size(), false);
        it->second[voxel.material_sid] = true;
    }

    // Pass 2: validate each segment is a single uniform allowed kind (all soft, all rigid, all
    // fixed, all rigid-fixed, all atlas with one shared atlas property, or all atlas-fixed with one
    // shared atlas property), then emit a rigid body for rigid/rigid-fixed/atlas segments.
    vector<RIndex> added_rigid_body_sid;
    for (auto &[segment_bid, material_flags] : segment_material_flags) {
        RIndex ref_material_sid = RS_NULL_INDEX;
        for (auto &&[material_sid, is_present] : enumerate(material_flags)) {
            if (!is_present) continue;
            if (ref_material_sid == RS_NULL_INDEX) {
                ref_material_sid = material_sid;
                continue;
            }
            const auto &ref = _mat_resolved_data[ref_material_sid];
            const auto &cur = _mat_resolved_data[material_sid];
            if (cur.phys_type != ref.phys_type) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Segment BID {} in body SID {} mixes incompatible material physical types: "
                    "material \"{}\" is {} but material \"{}\" is {}",
                    segment_bid,
                    body_sid,
                    ref.name,
                    enumToString(ref.phys_type),
                    cur.name,
                    enumToString(cur.phys_type)
                ));
            }
            if (cur.is_atlas != ref.is_atlas) {
                const auto &atlas_mat = cur.is_atlas ? cur : ref;
                const auto &plain_mat = cur.is_atlas ? ref : cur;
                RS_THROW(RS_InvalidCharacterizationException(
                    "Segment BID {} in body SID {} mixes atlas and non-atlas materials of the same "
                    "physical type {}: material \"{}\" is atlas but material \"{}\" is not",
                    segment_bid,
                    body_sid,
                    enumToString(ref.phys_type),
                    atlas_mat.name,
                    plain_mat.name
                ));
            }
            if (cur.is_atlas && !(cur.mass == ref.mass && cur.com_position == ref.com_position &&
                                  cur.inertia == ref.inertia)) {
                RS_THROW(RS_InvalidCharacterizationException(
                    "Segment BID {} in body SID {} mixes different atlas material properties: "
                    "material \"{}\" has mass={}, com=({}, {}, {}), inertia diagonal=({}, {}, {}) "
                    "but material \"{}\" has mass={}, com=({}, {}, {}), inertia diagonal=({}, {}, "
                    "{})",
                    segment_bid,
                    body_sid,
                    ref.name,
                    ref.mass,
                    ref.com_position.x,
                    ref.com_position.y,
                    ref.com_position.z,
                    ref.inertia[0][0],
                    ref.inertia[1][1],
                    ref.inertia[2][2],
                    cur.name,
                    cur.mass,
                    cur.com_position.x,
                    cur.com_position.y,
                    cur.com_position.z,
                    cur.inertia[0][0],
                    cur.inertia[1][1],
                    cur.inertia[2][2]
                ));
            }
        }

        // Rigid body discovery only applies to non-fixed structures.
        if (_structure().is_fixed) continue;

        const auto &ref = _mat_resolved_data[ref_material_sid];
        _segment_phys_type_map[{body_sid, segment_bid}] = ref.phys_type;

        // A rigid or rigid-fixed (including atlas) segment forms a rigid body; rigid-fixed is an
        // immovable anchor (zero inverse mass, set when its inertia is computed).
        const bool is_mat_rigid = ref.phys_type == RSE_STRUCTURE_VOXEL_MATERIAL_RIGID ||
                                  ref.phys_type == RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED;
        if (is_mat_rigid) {
            RIndex rigid_body_sid = _rigid_body_num++;

            auto rigid_body = RigidBodyData::preset();
            rigid_body.sid = rigid_body_sid;
            rigid_body.body_sid = body_sid;
            rigid_body.segment_bid = segment_bid;
            rigid_body.g_orientation = initial_body_orientation;
            // The mask is read only for constrained structures, which are capped at the mask width;
            // an unconstrained structure leaves it zero and may hold far more rigid bodies than the
            // mask can index.
            if (_is_constrained) {
                // A body never collides with itself; connected peers are set later.
                rigid_body.connected_body_mask.setOn(rigid_body_sid);
            }

            _rigid_bodies.emplace_back(rigid_body);
            _rigid_body_sid_map[{body_sid, segment_bid}] = rigid_body_sid;
            added_rigid_body_sid.push_back(rigid_body_sid);
        }
    }

    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Add body {} process segment types time {}",
        body_sid,
        timer.elapsedTimeString()
    );
    timer.reset();

    //----------------------------------------------------------------
    // Step 3: Create voxel hashmap (single hashmap level, 50% load factor)
    //----------------------------------------------------------------
    // Each voxel maps directly to one hashmap entry. The 2-flat hashmap's total capacity is about
    // the requested max_entry_num, so request twice the voxel count for a ~50% load factor. On
    // overflow, double the capacity, clear, and re-insert every voxel from the start.
    auto &voxel_tmp_index_map = _voxel_tmp_index_map[body_sid];

    RSize voxel_tmp_index_capacity = RS_MAX(tmp_voxel_num * 2, RSize(1));
    RSize voxel_tmp_insert_block_size = RS_CEIL_DIV(tmp_voxel_num, thread_count);

    RSize voxel_tmp_insert_retry = 10;
    bool is_voxel_tmp_insert_successful = true;

    voxel_tmp_index_map.resize(voxel_tmp_index_capacity, false);
    voxel_tmp_index_map.clear();

    _kernel_manager->getThreadPool()
        ->submitLoop(
            0,
            thread_count,
            [&](RIndex t) {
                while (voxel_tmp_insert_retry > 0) {
                    RIndex start = t * voxel_tmp_insert_block_size;
                    RIndex end = RS_MIN(start + voxel_tmp_insert_block_size, tmp_voxel_num);

                    for (; start < end; start++) {
                        auto &voxel = tmp_voxels[start];
                        if (!voxel_tmp_index_map.insert(
                                RS_VoxelKey(RCoord3(voxel.index_x, voxel.index_y, voxel.index_z)),
                                voxel.tmp_index
                            )) {
                            is_voxel_tmp_insert_successful = false;
                            break;
                        }
                    }

                    barrier.wait();
                    if (is_voxel_tmp_insert_successful) {
                        break;
                    }
                    barrier.wait();
                    if (t == 0) {
                        voxel_tmp_insert_retry--;

                        if (voxel_tmp_insert_retry > 0) {
                            is_voxel_tmp_insert_successful = true;
                            voxel_tmp_index_capacity *= 2;
                            voxel_tmp_index_map.resize(voxel_tmp_index_capacity, false);
                            voxel_tmp_index_map.clear();
                            RS_LOG(
                                _kernel_manager->getLogger(),
                                debug,
                                "Create voxel hashmap pass {} failed, resize voxel hashmap to "
                                "capacity {} and retry from start",
                                10 - voxel_tmp_insert_retry,
                                voxel_tmp_index_capacity
                            );
                        }
                    }
                    barrier.wait();
                }
            }
        )
        .wait();

    if (!is_voxel_tmp_insert_successful) {
        RS_THROW(RS_RuntimeException("Failed to insert all voxels in hashmap"));
    }

    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Add body {} create voxel hashmap time {}",
        body_sid,
        timer.elapsedTimeString()
    );
    timer.reset();

    //----------------------------------------------------------------
    // Step 4: Process properties of tmp_voxels
    //----------------------------------------------------------------
    _kernel_manager->getThreadPool()
        ->submitBlocks(
            0,
            tmp_voxel_num,
            [&](RIndex start, RIndex end) {
                for (RIndex v = start; v < end; v++) {
                    auto &voxel = tmp_voxels[v];

                    // Per face, assign the role:
                    // RSE_VOXEL_FACE_AIR: the neighbor cell is empty space.
                    // RSE_VOXEL_FACE_INTERIOR: the neighbor shares this voxel's material and
                    // segment.
                    // RSE_VOXEL_FACE_TRANSPARENT_BOUNDARY: the neighbor is a different transparent
                    // material, or a different segment.
                    // RSE_VOXEL_FACE_OPAQUE_BOUNDARY: the neighbor is a different non-transparent
                    // material, or a different segment.
                    // is_surface is set when any face is empty space, or, for a non-fixed body,
                    // faces a different segment.
                    bool is_surface = false;
                    bool has_soft_neighbor = false;
                    for (int l = 0; l < 6; l++) {
                        RCoord3 neighbor_coord(
                            voxel.index_x +
                                linkDirectionToXOffset(static_cast<RSE_LinkDirection>(l)),
                            voxel.index_y +
                                linkDirectionToYOffset(static_cast<RSE_LinkDirection>(l)),
                            voxel.index_z +
                                linkDirectionToZOffset(static_cast<RSE_LinkDirection>(l))
                        );

                        auto neighbor_voxel = voxel_tmp_index_map.find(RS_VoxelKey(neighbor_coord));

                        RSE_VoxelFaceRole role;
                        if (!neighbor_voxel) {
                            is_surface = true;
                            role = RSE_VOXEL_FACE_AIR;
                        } else {
                            auto &neighbor = tmp_voxels[*neighbor_voxel];
                            bool is_cross_segment = !_structure().is_fixed &&
                                                    neighbor.segment_bid != voxel.segment_bid;
                            if (is_cross_segment) {
                                is_surface = true;
                            }
                            if (neighbor.material_type == RSE_STRUCTURE_VOXEL_MATERIAL_SOFT) {
                                has_soft_neighbor = true;
                            }
                            // Two non-soft segments are separate parts that do not stick
                            // together, so the face between them is a real air gap. A soft
                            // neighbor instead bonds across the seam through a link.
                            bool is_non_soft_air_gap =
                                is_cross_segment &&
                                voxel.material_type != RSE_STRUCTURE_VOXEL_MATERIAL_SOFT &&
                                neighbor.material_type != RSE_STRUCTURE_VOXEL_MATERIAL_SOFT;
                            if (is_non_soft_air_gap) {
                                role = RSE_VOXEL_FACE_AIR;
                            } else if (neighbor.material_sid == voxel.material_sid &&
                                       neighbor.segment_bid == voxel.segment_bid) {
                                role = RSE_VOXEL_FACE_INTERIOR;
                            } else {
                                role = _mat_resolved_data[neighbor.material_sid].is_transparent
                                           ? RSE_VOXEL_FACE_TRANSPARENT_BOUNDARY
                                           : RSE_VOXEL_FACE_OPAQUE_BOUNDARY;
                            }
                        }
                        setFaceRole(voxel.role_state, l, role);
                    }
                    voxel.is_surface = is_surface;
                    voxel.is_render_active = isRenderActive(voxel.role_state);
                    voxel.is_contact_active = isContactActive(voxel.role_state);
                    voxel.has_soft_neighbor = has_soft_neighbor;

                    // Assign the rigid body now that the segment-type pass has emitted rigid
                    // bodies.
                    if (!_structure().is_fixed &&
                        (voxel.material_type == RSE_STRUCTURE_VOXEL_MATERIAL_RIGID ||
                         voxel.material_type == RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED)) {
                        voxel.rigid_body_sid =
                            _rigid_body_sid_map.at({body_sid, voxel.segment_bid});
                    }
                }
            }
        )
        .wait();

    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Add body {} process voxel properties time {}",
        body_sid,
        timer.elapsedTimeString()
    );
    timer.reset();

    //----------------------------------------------------------------
    // Step 5: Find valid voxels, assign sid, and save them
    //----------------------------------------------------------------
    RIndex added_voxel_start = _voxel_num;
    if (_structure().is_fixed) {
        // For a fixed body, only add voxels that are ray-traceable or collidable.
        for (RIndex v = 0; v < tmp_voxel_num; v++) {
            if (tmp_voxels[v].is_render_active || tmp_voxels[v].is_contact_active) {
                tmp_voxels[v].sid = _voxel_num++;
                _voxels.emplace_back(tmp_voxels[v]);
            }
        }
    } else {
        // For a non-fixed body, add every soft voxel, plus rigid/fixed voxels that are
        // ray-traceable, collidable, or a soft link endpoint.
        for (RIndex v = 0; v < tmp_voxel_num; v++) {
            if (tmp_voxels[v].material_type == RSE_STRUCTURE_VOXEL_MATERIAL_SOFT ||
                tmp_voxels[v].is_render_active || tmp_voxels[v].is_contact_active ||
                tmp_voxels[v].has_soft_neighbor) {
                tmp_voxels[v].sid = _voxel_num++;
                _voxels.emplace_back(tmp_voxels[v]);
            }
        }
    }

    _body_size[body_sid] = _voxel_num - added_voxel_start;

    RS_LOG(
        _kernel_manager->getLogger(),
        debug,
        "Add body {} create voxel time {}",
        body_sid,
        timer.elapsedTimeString()
    );
    timer.reset();

    if (!_structure().is_fixed) {
        //----------------------------------------------------------------
        // Step 6: Gather rigid body voxel info
        //----------------------------------------------------------------
        RIndex v_index = 0;
        for (auto &voxel : tmp_voxels) {
            if (voxel.material_type == RSE_STRUCTURE_VOXEL_MATERIAL_RIGID ||
                voxel.material_type == RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED) {
                _rigid_bodies[voxel.rigid_body_sid].contained_voxel_tmp_indices.emplace_back(
                    v_index
                );
                _rigid_bodies[voxel.rigid_body_sid].contained_material_sid.emplace_back(
                    voxel.material_sid
                );
                _rigid_bodies[voxel.rigid_body_sid].contained_material_rb_positions.emplace_back(
                    voxel.b_position
                );
            }
            ++v_index;
        }

        RS_LOG(
            _kernel_manager->getLogger(),
            debug,
            "Add body {} find rigid body voxel time {}",
            body_sid,
            timer.elapsedTimeString()
        );
        timer.reset();

        //----------------------------------------------------------------
        // Step 7: Create links
        //----------------------------------------------------------------

        // Add links, scan for +X, +Y, +Z direction from each voxel
        // Note: this layout increases cache locality, much more efficient than
        // scanning for X, then Y, then Z, which increases memory coalescing rate
        // but has much worse cache performance
        RIndex added_link_start = _links.size();
        vector<RSize> local_link_count = vector<RSize>(thread_count, 0);
        vector<RSize> local_link_offset = vector<RSize>(thread_count, 0);
        RSize block_size = RS_CEIL_DIV(tmp_voxel_num, thread_count);

        _kernel_manager->getThreadPool()
            ->submitLoop(
                static_cast<RIndex>(0),
                thread_count,
                [&](RIndex t) {
                    RIndex start = t * block_size;
                    RIndex end = RS_MIN(start + block_size, tmp_voxel_num);
                    vector<tuple<RIndex, RIndex, int>> local_links;

                    // --------------------
                    // Phase 1: Find links.
                    // --------------------
                    for (RIndex v = start; v < end; v++) {
                        auto &voxel = tmp_voxels[v];
                        auto material_type = voxel.material_type;

                        for (int l = 0; l < 6; l += 2) {
                            if (body_rsc.link.empty() ||
                                (body_rsc.link[voxel.array_index] & (1 << l))) {
                                // If link is not specified, default to create link in all
                                // directions
                                // If specified, only create link in the specified direction
                                RCoord3 neighbor_coord(
                                    voxel.index_x +
                                        linkDirectionToXOffset(static_cast<RSE_LinkDirection>(l)),
                                    voxel.index_y +
                                        linkDirectionToYOffset(static_cast<RSE_LinkDirection>(l)),
                                    voxel.index_z +
                                        linkDirectionToZOffset(static_cast<RSE_LinkDirection>(l))
                                );
                                if (neighbor_coord.x >= 0 && neighbor_coord.x < body_rsc.x_voxels &&
                                    neighbor_coord.y >= 0 && neighbor_coord.y < body_rsc.y_voxels &&
                                    neighbor_coord.z >= 0 && neighbor_coord.z < body_rsc.z_voxels) {
                                    auto neighbor_index_ptr =
                                        voxel_tmp_index_map.find(RS_VoxelKey(neighbor_coord));
                                    if (neighbor_index_ptr != nullptr) {
                                        RIndex neighbor_index = *neighbor_index_ptr;
                                        auto neighbor_material_type =
                                            tmp_voxels[neighbor_index].material_type;
                                        if (material_type == RSE_STRUCTURE_VOXEL_MATERIAL_SOFT ||
                                            neighbor_material_type ==
                                                RSE_STRUCTURE_VOXEL_MATERIAL_SOFT) {
                                            // Create link only if we find a valid neighbor voxel
                                            // and one of current/neighbor is soft
                                            local_links.emplace_back(
                                                voxel.sid, tmp_voxels[neighbor_index].sid, l
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    local_link_count[t] = local_links.size();
                    barrier.wait();

                    // ----------------------------------------------------------
                    // Phase 2: Compute global offsets in the main (t==0) thread.
                    // ----------------------------------------------------------
                    if (t == 0) {
                        local_link_offset[0] = 0;
                        for (RIndex i = 1; i < thread_count; i++) {
                            local_link_offset[i] =
                                local_link_offset[i - 1] + local_link_count[i - 1];
                        }

                        _link_num += local_link_offset[thread_count - 1] + local_link_count.back();
                        _links.resize(_link_num);
                    }

                    barrier.wait();

                    // ----------------------------------------
                    // Phase 3: Fill the link data in parallel.
                    // ----------------------------------------
                    for (RIndex i = 0; i < local_links.size(); i++) {
                        RIndex link_sid = local_link_offset[t] + i + added_link_start;

                        auto link = LinkData::preset();

                        RIndex voxel = get<0>(local_links[i]);
                        RIndex neighbor_voxel = get<1>(local_links[i]);
                        int l = get<2>(local_links[i]);
                        // Make sure that link is always pointed to +X, +Y and +Z direction
                        // from negative voxel to positive voxel
                        link.sid = link_sid;
                        link.body_sid = body_sid;
                        link.axis = linkDirectionToAxis(static_cast<RSE_LinkDirection>(l));
                        link.voxel_neg_sid = voxel;
                        link.voxel_pos_sid = neighbor_voxel;

                        link.voxel_neg_material_sid = _voxels[voxel].material_sid;
                        link.voxel_pos_material_sid = _voxels[neighbor_voxel].material_sid;

                        auto pos_dir = static_cast<RSE_LinkDirection>(l);
                        _voxels[voxel].link_sid[pos_dir] = link.sid;
                        _voxels[voxel].neighbor_voxel_sid[pos_dir] = neighbor_voxel;

                        auto neg_dir = static_cast<RSE_LinkDirection>(l + 1);
                        _voxels[neighbor_voxel].link_sid[neg_dir] = link.sid;
                        _voxels[neighbor_voxel].neighbor_voxel_sid[neg_dir] = voxel;

                        _links[link_sid] = link;
                    }
                }
            )
            .wait();

        RS_LOG(
            _kernel_manager->getLogger(),
            debug,
            "Add body {} create link time {}",
            body_sid,
            timer.elapsedTimeString()
        );
        timer.reset();

        // Process at last to make sure that rigid body surface voxels are contiguous
        // in rigid body index and ordered by rigid body index
        for (auto rigid_body_sid : added_rigid_body_sid) {
            auto &rigid_body = _rigid_bodies[rigid_body_sid];

            // Compute origin of the rigid body by finding minimum bounding box corner.
            // contained_material_rb_positions hold voxel centers, so shift by half a voxel to land
            // on the true minimum bounding box corner rather than the corner voxel's center.
            rigid_body.origin_b_position = rigid_body.contained_material_rb_positions[0];
            for (auto &material_local_position : rigid_body.contained_material_rb_positions) {
                rigid_body.origin_b_position =
                    rigid_body.origin_b_position.min(material_local_position);
            }
            RFloat half_voxel_size =
                _structure().bodies[rigid_body.body_sid].voxel_size / RFLOAT(2);
            rigid_body.origin_b_position -= RVec3rf::full(half_voxel_size);

            // Compute local coordinates of material points
            for (auto &material_local_position : rigid_body.contained_material_rb_positions) {
                material_local_position -= rigid_body.origin_b_position;
            }

            // Transform rigid body origin position to global scope
            rigid_body.origin_g_position =
                initial_body_origin_position +
                initial_body_orientation.rotateVec3D(rigid_body.origin_b_position);

            // Create rigid body surface voxels
            rigid_body.rigid_body_surface_voxel_start = _rigid_body_surface_voxel_num;
            for (auto &&[voxel_tmp_index, material_rb_position] :
                 zip(rigid_body.contained_voxel_tmp_indices,
                     rigid_body.contained_material_rb_positions)) {
                // Add rigid body voxel if this rigid voxel is a surface voxel
                auto &voxel = tmp_voxels[voxel_tmp_index];

                if (voxel.is_surface) {
                    auto rigid_body_surface_voxel = RigidBodySurfaceVoxelData::preset();
                    rigid_body_surface_voxel.voxel_sid = tmp_voxels[voxel_tmp_index].sid;
                    rigid_body_surface_voxel.rigid_body_sid = rigid_body_sid;
                    rigid_body_surface_voxel.voxel_rb_position = material_rb_position;
                    _rigid_body_surface_voxels.emplace_back(rigid_body_surface_voxel);
                    _rigid_body_surface_voxel_num++;
                }
            }
            rigid_body.rigid_body_surface_voxel_end = _rigid_body_surface_voxel_num;
        }

        // Compute each rigid body's mass, center of mass, and local moment of inertia inverse
        // from its contained voxels.
        _kernel_manager->getThreadPool()
            ->submitLoop(
                0,
                added_rigid_body_sid.size(),
                [&](RIndex idx) {
                    auto &rigid_body = _rigid_bodies[added_rigid_body_sid[idx]];

                    // An atlas body has user-prescribed mass, com and inertia. Override the
                    // voxel-summed values directly. com_position is already in this rigid body's
                    // local frame (origin at the minimum bounding box corner) and inertia is
                    // already about the com in body-local axes, so no conversion is needed.
                    const auto &atlas_resolved =
                        _mat_resolved_data[rigid_body.contained_material_sid[0]];
                    if (atlas_resolved.is_atlas) {
                        rigid_body.mass = atlas_resolved.mass;
                        rigid_body.mass_inverse = RFLOAT(1) / atlas_resolved.mass;
                        rigid_body.com_rb_position = atlas_resolved.com_position;
                        rigid_body.local_moment_inertia_inverse = atlas_resolved.inertia.inverse();
                        return;
                    }

                    // Voxel size is uniform within a rigid body (it belongs to a single body)
                    RFloat voxel_size = _structure().bodies[rigid_body.body_sid].voxel_size;
                    RFloat voxel_volume = voxel_size * voxel_size * voxel_size;
                    // Self inertia of one voxel about its own center, per diagonal axis: a solid
                    // cube of side voxel_size contributes m * (s^2 + s^2) / 12 = m * s^2 / 6 to
                    // each of i_xx, i_yy, i_zz.
                    RFloat voxel_self_inertia_term = voxel_size * voxel_size / RFLOAT(6);

                    // Compute mass and center of mass first
                    RVec3rf mass_dot_pos = RVec3rf::zero();
                    RFloat mass = 0;
                    for (auto &&[material_sid, material_pos] :
                         zip(rigid_body.contained_material_sid,
                             rigid_body.contained_material_rb_positions)) {
                        RFloat voxel_mass = _mat_resolved_data[material_sid].density * voxel_volume;
                        mass_dot_pos += voxel_mass * material_pos;
                        mass += voxel_mass;
                    }
                    rigid_body.mass = mass;
                    rigid_body.com_rb_position = mass_dot_pos / mass;

                    // Rigid-fixed body is an immovable anchor: zero inverse mass/inertia
                    if (_mat_resolved_data[rigid_body.contained_material_sid[0]].phys_type ==
                        RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED) {
                        rigid_body.mass_inverse = 0;
                        rigid_body.local_moment_inertia_inverse = RMat3rf::zero();
                        return;
                    }
                    rigid_body.mass_inverse = RFLOAT(1) / mass;

                    // For computing local moment inertia in com coords, see
                    // https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/
                    // dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf
                    // Omit i_yx, i_zy, i_zx since they are equal to counter parts
                    RFloat i_xx = 0, i_yy = 0, i_zz = 0, i_xy = 0, i_xz = 0, i_yz = 0;

                    // Per-voxel cube self inertia shifted to the COM via the parallel axis theorem.
                    for (auto &&[material_sid, material_pos] :
                         zip(rigid_body.contained_material_sid,
                             rigid_body.contained_material_rb_positions)) {
                        RFloat voxel_mass = _mat_resolved_data[material_sid].density * voxel_volume;
                        auto d = material_pos - rigid_body.com_rb_position;
                        i_xx += voxel_mass * (d.y * d.y + d.z * d.z + voxel_self_inertia_term);
                        i_yy += voxel_mass * (d.x * d.x + d.z * d.z + voxel_self_inertia_term);
                        i_zz += voxel_mass * (d.x * d.x + d.y * d.y + voxel_self_inertia_term);
                        i_xy += voxel_mass * d.x * d.y;
                        i_xz += voxel_mass * d.x * d.z;
                        i_yz += voxel_mass * d.y * d.z;
                    }
                    RMat3rf local_moment_inertia = RMat3rf::zero();
                    local_moment_inertia[0][0] = i_xx;
                    local_moment_inertia[0][1] = -i_xy;
                    local_moment_inertia[0][2] = -i_xz;
                    local_moment_inertia[1][0] = -i_xy;
                    local_moment_inertia[1][1] = i_yy;
                    local_moment_inertia[1][2] = -i_yz;
                    local_moment_inertia[2][0] = -i_xz;
                    local_moment_inertia[2][1] = -i_yz;
                    local_moment_inertia[2][2] = i_zz;
                    rigid_body.local_moment_inertia_inverse = local_moment_inertia.inverse();
                }
            )
            .wait();

        RS_LOG(
            _kernel_manager->getLogger(),
            debug,
            "Add body {} create rigid body surface voxel time {}",
            body_sid,
            timer.elapsedTimeString()
        );
        timer.reset();
    }
}

void RS_KernelStructureVoxelManager::_addConstraint(
    const rsc::RSC_StructureVoxelConstraint &constraint_rsc
) {
    // Validate that constrained segments are rigid or rigid-fixed.
    const auto a_type =
        _segment_phys_type_map.at({constraint_rsc.a_body_sid, constraint_rsc.a_segment_bid});
    if (a_type != RSE_STRUCTURE_VOXEL_MATERIAL_RIGID &&
        a_type != RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Segment with BID {} from Body A with SID {} referenced in constraint is not a "
            "rigid segment",
            constraint_rsc.a_segment_bid,
            constraint_rsc.a_body_sid
        ));
    }
    const auto b_type =
        _segment_phys_type_map.at({constraint_rsc.b_body_sid, constraint_rsc.b_segment_bid});
    if (b_type != RSE_STRUCTURE_VOXEL_MATERIAL_RIGID &&
        b_type != RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Segment with BID {} from Body B with SID {} referenced in constraint is not a "
            "rigid segment",
            constraint_rsc.b_segment_bid,
            constraint_rsc.b_body_sid
        ));
    }

    // Record rigid-rigid connectivity so jointly-connected bodies skip mutual collision.
    // A constraint that enables collision leaves both bits clear so the pair reaches narrow phase.
    RIndex rb_a_sid =
        _rigid_body_sid_map.at({constraint_rsc.a_body_sid, constraint_rsc.a_segment_bid});
    RIndex rb_b_sid =
        _rigid_body_sid_map.at({constraint_rsc.b_body_sid, constraint_rsc.b_segment_bid});
    if (!constraint_rsc.is_collision_enabled) {
        _rigid_bodies[rb_a_sid].connected_body_mask.setOn(rb_b_sid);
        _rigid_bodies[rb_b_sid].connected_body_mask.setOn(rb_a_sid);
    }

    RIndex joint_sid = _joint_num++;
    if (constraint_rsc.type == rsc::RSE_STRUCTURE_VOXEL_BALL_AND_SOCKET_JOINT) {
        _constraint_joints.emplace_back(_createBasicConstraintJoint(
            constraint_rsc, RSE_BALL_AND_SOCKET_JOINT_TRANSLATION, joint_sid
        ));
    } else if (constraint_rsc.type == rsc::RSE_STRUCTURE_VOXEL_FIXED_JOINT) {
        RSE_ConstraintJointType types[2] = {RSE_FIXED_JOINT_TRANSLATION, RSE_FIXED_JOINT_ROTATION};
        for (auto &type : types) {
            auto constraint = _createBasicConstraintJoint(constraint_rsc, type, joint_sid);
            _constraint_joints.emplace_back(constraint);
        }
    } else if (constraint_rsc.type == rsc::RSE_STRUCTURE_VOXEL_HINGE_JOINT) {
        RSE_ConstraintJointType types[4] = {
            RSE_HINGE_JOINT_TRANSLATION,
            RSE_HINGE_JOINT_ROTATION,
            RSE_HINGE_JOINT_LIMIT,
            RSE_HINGE_JOINT_MOTOR
        };
        RIndex motor_sid = _motor_num++;

        for (auto &type : types) {
            if (type == RSE_HINGE_JOINT_LIMIT && constraint_rsc.hinge_min <= RFLOAT(-RS_PI) &&
                constraint_rsc.hinge_max >= RFLOAT(RS_PI))
                continue;
            if (type == RSE_HINGE_JOINT_MOTOR && constraint_rsc.hinge_signal_sid == RS_NULL_INDEX)
                continue;
            auto cj = _createBasicConstraintJoint(constraint_rsc, type, joint_sid);

            // Convert structure-relative hinge axis to body-local for each body
            auto &body_a_rsc = _structure().bodies[constraint_rsc.a_body_sid];
            auto &body_b_rsc = _structure().bodies[constraint_rsc.b_body_sid];
            cj.motor_sid = motor_sid;
            cj.hinge_control_mode =
                _convertToConstraintJointHingeControlMode(constraint_rsc.hinge_control_mode);
            cj.hinge_a_rb_axis = body_a_rsc.relative_orientation.inverse().rotateVec3D(
                constraint_rsc.hinge_relative_axis
            );
            RVec3rf b_local_axis = body_b_rsc.relative_orientation.inverse().rotateVec3D(
                constraint_rsc.hinge_relative_axis
            );
            cj.hinge_b_rb_orthogonal_axis_1 = b_local_axis.arbitraryNormal();
            cj.hinge_b_rb_orthogonal_axis_2 = b_local_axis.cross(cj.hinge_b_rb_orthogonal_axis_1);

            cj.hinge_min = constraint_rsc.hinge_min;
            cj.hinge_max = constraint_rsc.hinge_max;
            if (constraint_rsc.hinge_control_mode ==
                rsc::RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION) {
                cj.hinge_max_speed = constraint_rsc.hinge_max_speed;
                cj.hinge_max_torque = constraint_rsc.hinge_max_torque;
            } else if (constraint_rsc.hinge_control_mode ==
                       rsc::RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION) {
                cj.hinge_position_control_kp = constraint_rsc.hinge_position_control_kp;
                cj.hinge_position_control_kd = constraint_rsc.hinge_position_control_kd;
                cj.hinge_max_speed = constraint_rsc.hinge_max_speed;
                cj.hinge_max_torque = constraint_rsc.hinge_max_torque;
            }

            cj.hinge_signal_sid = constraint_rsc.hinge_signal_sid;

            if (type == RSE_HINGE_JOINT_MOTOR) {
                cj.motor_sid = _motor_num++;
            }

            _constraint_joints.emplace_back(cj);
        }
    } else if (constraint_rsc.type == rsc::RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT) {
        RSE_ConstraintJointType types[4] = {
            RSE_PRISMATIC_JOINT_PERPENDICULAR_TRANSLATION,
            RSE_PRISMATIC_JOINT_ROTATION,
            RSE_PRISMATIC_JOINT_LIMIT,
            RSE_PRISMATIC_JOINT_MOTOR
        };
        RIndex motor_sid = _motor_num++;

        for (auto &type : types) {
            if (type == RSE_PRISMATIC_JOINT_LIMIT && constraint_rsc.prismatic_min <= RFLOAT(-1e6) &&
                constraint_rsc.prismatic_max >= RFLOAT(1e6))
                continue;
            if (type == RSE_PRISMATIC_JOINT_MOTOR &&
                constraint_rsc.prismatic_signal_sid == RS_NULL_INDEX)
                continue;
            auto cj = _createBasicConstraintJoint(constraint_rsc, type, joint_sid);

            // Convert structure-relative slide axis to body-A-local
            auto &body_a_rsc = _structure().bodies[constraint_rsc.a_body_sid];
            auto &body_b_rsc = _structure().bodies[constraint_rsc.b_body_sid];
            cj.motor_sid = motor_sid;
            cj.prismatic_control_mode = _convertToConstraintJointPrismaticControlMode(
                constraint_rsc.prismatic_control_mode
            );
            cj.prismatic_a_rb_axis = body_a_rsc.relative_orientation.inverse().rotateVec3D(
                constraint_rsc.prismatic_relative_axis
            );
            // Precompute perpendicular axes in body B's local frame (stable, like hinge)
            RVec3rf b_local_axis = body_b_rsc.relative_orientation.inverse().rotateVec3D(
                constraint_rsc.prismatic_relative_axis
            );
            cj.prismatic_b_rb_orthogonal_axis_1 = b_local_axis.arbitraryNormal();
            cj.prismatic_b_rb_orthogonal_axis_2 =
                b_local_axis.cross(cj.prismatic_b_rb_orthogonal_axis_1);
            cj.prismatic_min = constraint_rsc.prismatic_min;
            cj.prismatic_max = constraint_rsc.prismatic_max;
            if (constraint_rsc.prismatic_control_mode ==
                rsc::RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION) {
                cj.prismatic_max_speed = constraint_rsc.prismatic_max_speed;
                cj.prismatic_max_force = constraint_rsc.prismatic_max_force;
            } else if (constraint_rsc.prismatic_control_mode ==
                       rsc::RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION) {
                cj.prismatic_position_control_kp = constraint_rsc.prismatic_position_control_kp;
                cj.prismatic_position_control_kd = constraint_rsc.prismatic_position_control_kd;
                cj.prismatic_max_speed = constraint_rsc.prismatic_max_speed;
                cj.prismatic_max_force = constraint_rsc.prismatic_max_force;
            }

            cj.prismatic_signal_sid = constraint_rsc.prismatic_signal_sid;

            if (type == RSE_PRISMATIC_JOINT_MOTOR) {
                cj.motor_sid = _motor_num++;
            }

            _constraint_joints.emplace_back(cj);
        }
    }
}

void RS_KernelStructureVoxelManager::_addMaterials() {
    // Per-body voxel and link materials (different bodies may have different voxel_size)
    vector<RFloat> body_voxel_sizes;
    for (auto &body_rsc : _structure().bodies) {
        body_voxel_sizes.push_back(body_rsc.voxel_size);
    }

    map<RS_OrderedPair<RIndex, RIndex>, RIndex> voxel_material_map;
    for (auto &voxel : _voxels) {
        if (voxel.material_sid == RS_NULL_INDEX) continue;
        RS_OrderedPair<RIndex, RIndex> key = {voxel.body_sid, voxel.material_sid};
        auto [it, inserted] = voxel_material_map.try_emplace(key, (int)voxel_material_map.size());
        voxel.voxel_material_sid = it->second;
    }
    _voxel_materials.resize(voxel_material_map.size());
    for (auto &[key, sid] : voxel_material_map) {
        VoxelMaterialData data;
        data.material_sid = key.value2;
        data.body_sid = key.value1;
        data.voxel_size = body_voxel_sizes[key.value1];
        _voxel_materials[sid] = data;
    }

    map<RS_OrderedTriple<RIndex, RIndex, RIndex>, RIndex> link_material_map;
    for (auto &link : _links) {
        auto [it, inserted] = link_material_map.try_emplace(
            {link.body_sid, link.voxel_neg_material_sid, link.voxel_pos_material_sid},
            (int)link_material_map.size()
        );
        link.link_material_sid = it->second;
    }
    _link_materials.resize(link_material_map.size());
    for (auto &[key, sid] : link_material_map) {
        LinkMaterialData data;
        data.neg_material_sid = key.value2;
        data.pos_material_sid = key.value3;
        data.body_sid = key.value1;
        data.voxel_size = body_voxel_sizes[key.value1];
        _link_materials[sid] = data;
    }
}

void RS_KernelStructureVoxelManager::_addSignals() {
    _expansion_signal_num = _structure().signal->expansion_signal_num;
    _hinge_signal_num = _structure().signal->hinge_signal_num;
    _prismatic_signal_num = _structure().signal->prismatic_signal_num;

    RBigSize total_size = RS_WorldStructureVoxelSignal::getRequiredMemorySize(
        _expansion_signal_num, _hinge_signal_num, _prismatic_signal_num
    );

    RS_MemoryProvider *memory_provider = _scoped_device_provider.get();
    _host_signal_buffer = memory_provider->allocateHost(total_size, true);
    _device_signal_buffer = memory_provider->allocate(total_size, true);
    _host_signal_buffer.set(0);
    _device_signal_buffer.set(0);

    _host_signal.initFromMemory(
        _host_signal_buffer.ptr(), _expansion_signal_num, _hinge_signal_num, _prismatic_signal_num
    );
    _host_signal.structure_gid = _structure_gid;
    _device_signal.initFromMemory(
        _device_signal_buffer.ptr(), _expansion_signal_num, _hinge_signal_num, _prismatic_signal_num
    );
    _device_signal.structure_gid = _structure_gid;
    _device_signal.device = _device_signal_buffer.device();
}

void RS_KernelStructureVoxelManager::_colorConstraintJoints() {
    if (_constraint_joint_num == 0) return;

    RS_Array<bool, 2> constraint_share_body(
        {_constraint_joint_num, _constraint_joint_num}, _scoped_host_provider.get()
    );
    RS_Array<RIndex, 1> constraint_colors(_constraint_joint_num, _scoped_host_provider.get());
    RS_Array<bool, 1> constraint_color_is_used(_constraint_joint_num, _scoped_host_provider.get());

    constraint_share_body.fill(false);
    constraint_colors.fill(RS_NULL_INDEX);

    for (auto &cj : _constraint_joints) {
        for (auto &cj1 : _constraint_joints) {
            if ((cj.sid != cj1.sid) && (cj.rigid_body_a_sid == cj1.rigid_body_a_sid ||
                                        cj.rigid_body_a_sid == cj1.rigid_body_b_sid ||
                                        cj.rigid_body_b_sid == cj1.rigid_body_a_sid ||
                                        cj.rigid_body_b_sid == cj1.rigid_body_b_sid)) {
                constraint_share_body[{cj1.sid, cj.sid}] = true;
                constraint_share_body[{cj.sid, cj1.sid}] = true;
            }
        }
    }

    // First constraint has 0 color
    constraint_colors[0] = 0;
    _constraint_joint_color_num = 1;

    for (RIndex cj_sid = 1; cj_sid < _constraint_joint_num; cj_sid++) {
        // Check color used by neighbors
        constraint_color_is_used.fill(false);
        for (RIndex n_sid = 0; n_sid < _constraint_joint_num; n_sid++) {
            if (constraint_share_body[{n_sid, cj_sid}] &&
                constraint_colors[n_sid] != RS_NULL_INDEX) {
                constraint_color_is_used[constraint_colors[n_sid]] = true;
            }
        }
        // Assign lowest available color
        for (RIndex color = 0; color < _constraint_joint_num; color++) {
            if (!constraint_color_is_used[color]) {
                constraint_colors[cj_sid] = color;
                _constraint_joint_color_num = RS_MAX(_constraint_joint_color_num, color + 1);
                break;
            }
        }
    }

    for (auto &cj : _constraint_joints) {
        cj.color = constraint_colors[cj.sid];
    }
    constraint_share_body.free();
    constraint_colors.free();
    constraint_color_is_used.free();
}

RS_KernelStructureVoxelManager::ConstraintJointData
RS_KernelStructureVoxelManager::_createBasicConstraintJoint(
    const rsc::RSC_StructureVoxelConstraint &constraint_rsc,
    RSE_ConstraintJointType constraint_type,
    RIndex joint_sid
) {
    auto constraint = ConstraintJointData();
    constraint.sid = _constraint_joint_num++;
    constraint.joint_sid = joint_sid;
    constraint.rigid_body_a_sid =
        _rigid_body_sid_map.at({constraint_rsc.a_body_sid, constraint_rsc.a_segment_bid});
    constraint.rigid_body_b_sid =
        _rigid_body_sid_map.at({constraint_rsc.b_body_sid, constraint_rsc.b_segment_bid});

    constraint.type = constraint_type;
    constraint.rigid_body_a_anchor_rb_position = getRigidBodyPositionFromRelativeAnchor(
        constraint_rsc.a_body_sid, constraint.rigid_body_a_sid, constraint_rsc.relative_anchor
    );
    constraint.rigid_body_b_anchor_rb_position = getRigidBodyPositionFromRelativeAnchor(
        constraint_rsc.b_body_sid, constraint.rigid_body_b_sid, constraint_rsc.relative_anchor
    );
    return constraint;
}

void RS_KernelStructureVoxelManager::_checkStructureLimits() const {
    if (_body_num > RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_BODY_NUM)
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure \"{}\" has {} bodies, exceeding limit {}",
            _structure().name,
            _body_num,
            RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_BODY_NUM
        ));
    if (_voxel_num > RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_VOXEL_NUM)
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure \"{}\" has {} voxels, exceeding limit {}",
            _structure().name,
            _voxel_num,
            RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_VOXEL_NUM
        ));
    if (_link_num > RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_LINK_NUM)
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure \"{}\" has {} links, exceeding limit {}",
            _structure().name,
            _link_num,
            RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_LINK_NUM
        ));
    if (_is_constrained) {
        if (_rigid_body_num > RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_RIGID_BODY_CONSTRAINED_NUM)
            RS_THROW(RS_InvalidCharacterizationException(
                "Constrained structure \"{}\" has {} rigid bodies, exceeding the constrained limit "
                "{}",
                _structure().name,
                _rigid_body_num,
                RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_RIGID_BODY_CONSTRAINED_NUM
            ));
    } else {
        if (_rigid_body_num > RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_RIGID_BODY_UNCONSTRAINED_NUM)
            RS_THROW(RS_InvalidCharacterizationException(
                "Unconstrained structure \"{}\" has {} rigid bodies, exceeding the unconstrained "
                "limit {}",
                _structure().name,
                _rigid_body_num,
                RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_RIGID_BODY_UNCONSTRAINED_NUM
            ));
    }
    if (_joint_num > RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_CONSTRAINT_NUM)
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure \"{}\" has {} constraints, exceeding limit {}",
            _structure().name,
            _joint_num,
            RS_WORLD_RSC_STRUCTURE_VOXEL_MAX_CONSTRAINT_NUM
        ));
}

void RS_KernelStructureVoxelManager::_checkRigidBodyInertia() const {
    // local_moment_inertia_inverse is already computed per rigid body in _addBody (voxel-summed for
    // rigid, prescribed for atlas), so just validate the stored value here.
    for (auto &rigid_body : _rigid_bodies) {
        // Rigid-fixed body is an immovable anchor: its inverse inertia is intentionally zero.
        if (_mat_resolved_data[rigid_body.contained_material_sid[0]].phys_type ==
            RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED) {
            continue;
        }

        if (!rigid_body.local_moment_inertia_inverse.isValid())
            RS_THROW(RS_InvalidCharacterizationException(
                "Rigid body in structure {} body {} segment {} has invalid moment inertia inverse",
                _structure().name,
                rigid_body.body_sid,
                rigid_body.segment_bid
            ));
    }
}

template <RSE_FloatType FloatMemAccuracy>
RS_KernelStructureVoxelManager::KernelStructureData<FloatMemAccuracy>
RS_KernelStructureVoxelManager::_getAll() const {
    using FloatM = typename RS_FloatDataType<FloatMemAccuracy>::DataType;
    using VoxelMem = RS_VoxelMem<FloatMemAccuracy>;
    using LinkMem = RS_LinkMem<FloatMemAccuracy>;
    using VoxelMaterialMem = RS_VoxelMaterialMem<FloatMemAccuracy>;
    using LinkMaterialMem = RS_LinkMaterialMem<FloatMemAccuracy>;
    using RigidBodyMem = RS_RigidBodyMem<FloatMemAccuracy>;
    using RigidBodySurfaceVoxelMem = RS_RigidBodySurfaceVoxelMem<FloatMemAccuracy>;
    using ConstraintJointMem = RS_ConstraintJointMem<FloatMemAccuracy>;

    using VoxelArray = RS_Array<VoxelMem, 1>;
    using LinkArray = RS_Array<LinkMem, 1>;
    using VoxelMaterialArray = RS_Array<RS_VoxelMaterialMem<FloatMemAccuracy>, 1>;
    using LinkMaterialArray = RS_Array<RS_LinkMaterialMem<FloatMemAccuracy>, 1>;
    using RigidBodyArray = RS_Array<RigidBodyMem, 1>;
    using RigidBodySurfaceVoxelArray = RS_Array<RigidBodySurfaceVoxelMem, 1>;
    using ConstraintJointArray = RS_Array<ConstraintJointMem, 1>;
    using RigidBodyFloatArray = RS_Array<RS_FloatMem<FloatMemAccuracy>, 1>;
    using RigidBodyVec3DArray = RS_Array<RS_Vec3DMem<FloatMemAccuracy>, 1>;
    using RigidBodyMat3DArray = RS_Array<RS_Mat3DMem<FloatMemAccuracy>, 1>;

    KernelStructureData<FloatMemAccuracy> data;

    RS_MemoryProvider *provider = _kernel_manager->getDeviceProvider();
    RS_MemoryProvider *host_provider = _scoped_host_provider.get();

    // RS_Timer timer;

    RS_MemoryPtr voxel_materials =
        host_provider->allocate(sizeof(VoxelMaterialMem) * _voxel_materials.size(), true);
    RS_MemoryPtr link_materials =
        host_provider->allocate(sizeof(LinkMaterialMem) * _link_materials.size(), true);
    RS_MemoryPtr voxels = host_provider->allocate(sizeof(VoxelMem) * _voxels.size(), true);
    RS_MemoryPtr links = host_provider->allocate(sizeof(LinkMem) * _links.size(), true);
    RS_MemoryPtr rigid_bodies =
        host_provider->allocate(sizeof(RigidBodyMem) * _rigid_bodies.size(), true);
    RS_MemoryPtr rigid_body_surface_voxels = host_provider->allocate(
        sizeof(RigidBodySurfaceVoxelMem) * _rigid_body_surface_voxels.size(), true
    );
    RS_MemoryPtr constraint_joints =
        host_provider->allocate(sizeof(ConstraintJointMem) * _constraint_joints.size(), true);

    map<RS_OrderedPair<RIndex, RIndex>, RIndex> voxel_material_map;

    // Resolve each dependent material's vision-material handle now that the material vision slots
    // are committed, indexed by material_sid.
    auto &mat_mgr = _kernel_manager->getKernelMaterialManager();
    std::vector<RHandle> vision_material_by_sid(_mat_resolved_data.size());
    for (auto &&[material_sid, resolved] : enumerate(_mat_resolved_data)) {
        vision_material_by_sid[material_sid] = mat_mgr.findVisionMaterialHandle(resolved.name);
    }

    // print("alloc {}\n", timer.elapsedTimeString());
    // timer.reset();

    for (auto &&[i, voxel_material] : enumerate(_voxel_materials)) {
        auto &resolved = _mat_resolved_data[voxel_material.material_sid];
        auto voxel_material_out = VoxelMaterialMem();
        voxel_material_out.structure_gid = _structure_gid;
        voxel_material_out.sid = i;
        voxel_material_out.material_gid = resolved.material_gid;
        voxel_material_out.E = resolved.elastic_modulus;
        voxel_material_out.fail_stress = resolved.fail_stress;
        voxel_material_out.nu = resolved.poissons_ratio;
        voxel_material_out.rho = resolved.density;
        voxel_material_out.zeta_internal = _kernel_manager->kernel.softbody_internal_damping_z;
        voxel_material_out.zeta_contact = _kernel_manager->kernel.softbody_contact_damping_z;
        voxel_material_out.zeta_global = _kernel_manager->kernel.softbody_global_damping_z;

        voxel_material_out.voxel_size = voxel_material.voxel_size;
        voxel_material_out.min_expansion = resolved.min_expansion;
        voxel_material_out.max_expansion = resolved.max_expansion;
        const bool is_rigid_or_fixed = resolved.phys_type != RSE_STRUCTURE_VOXEL_MATERIAL_SOFT;
        RS_VoxelMaterialCompute<FloatMemAccuracy>::initMaterial(
            voxel_material_out, is_rigid_or_fixed, _kernel_manager->kernel.world_grav_acc
        );
        voxel_materials.get<VoxelMaterialMem>(i) = voxel_material_out;
        voxel_material_map[{voxel_material.material_sid, voxel_material.body_sid}] = i;
    }

    for (auto &&[i, link_material] : enumerate(_link_materials)) {
        auto link_material_out = LinkMaterialMem();
        link_material_out.structure_gid = _structure_gid;
        link_material_out.sid = i;
        const bool is_neg_rigid_or_fixed =
            _mat_resolved_data[link_material.neg_material_sid].phys_type !=
            RSE_STRUCTURE_VOXEL_MATERIAL_SOFT;
        const bool is_pos_rigid_or_fixed =
            _mat_resolved_data[link_material.pos_material_sid].phys_type !=
            RSE_STRUCTURE_VOXEL_MATERIAL_SOFT;
        RS_LinkMaterialCompute<FloatMemAccuracy>::initMaterial(
            link_material_out,
            voxel_materials.get<VoxelMaterialMem>(
                voxel_material_map.at({link_material.neg_material_sid, link_material.body_sid})
            ),
            voxel_materials.get<VoxelMaterialMem>(
                voxel_material_map.at({link_material.pos_material_sid, link_material.body_sid})
            ),
            is_neg_rigid_or_fixed,
            is_pos_rigid_or_fixed,
            _kernel_manager->kernel.world_grav_acc
        );
        link_materials.get<LinkMaterialMem>(i) = link_material_out;
    }

    // print("m {}\n", timer.elapsedTimeString());
    // timer.reset();

    // Only a structure with at least one constraint runs the connected-body collision filter.
    if (!_voxels.empty())
        _kernel_manager->getThreadPool()
            ->submitLoop(
                0,
                _voxels.size(),
                [&](RIndex i) {
                    auto &voxel = _voxels[i];
                    auto voxel_out = VoxelMem();
                    voxel_out.structure_gid = _structure_gid;
                    voxel_out.sid = i;
                    voxel_out.body_sid = voxel.body_sid;
                    voxel_out.segment_bid = voxel.segment_bid;
                    voxel_out.rigid_body_sid = voxel.rigid_body_sid;
                    voxel_out.index_x = voxel.index_x;
                    voxel_out.index_y = voxel.index_y;
                    voxel_out.index_z = voxel.index_z;

                    logic::RS_VoxelContactFilterKeyUnion fk{0};
                    fk.contact_filter_key.body_sid = voxel.body_sid;
                    fk.contact_filter_key.constrained_rigid_body_sid =
                        (_is_constrained && voxel.rigid_body_sid != RS_NULL_INDEX)
                            ? voxel.rigid_body_sid
                            : RS_VOXEL_CONTACT_FILTER_KEY_NULL_CONSTRAINED_RIGID_BODY_SID;
                    fk.contact_filter_key.index_x_res = voxel.index_x & 0xF;
                    fk.contact_filter_key.index_y_res = voxel.index_y & 0xF;
                    fk.contact_filter_key.index_z_res = voxel.index_z & 0xF;
                    voxel_out.contact_filter_key = fk.contact_filter_key_packed;

                    voxel_out.voxel_material_sid = voxel.voxel_material_sid;
                    voxel_out.expansion_signal_sid = voxel.expansion_signal_sid;
                    voxel_out.structure_signal = _structure_signal;
                    for (int j = 0; j < 6; j++) {
                        voxel_out.link_sid[j] = voxel.link_sid[j];
                        voxel_out.neighbor_voxel_sid[j] = voxel.neighbor_voxel_sid[j];
                    }
                    voxel_out.vision_material = voxel.material_sid == RS_NULL_INDEX
                                                    ? RS_NULL_HANDLE
                                                    : vision_material_by_sid[voxel.material_sid];
                    voxel_out.position = voxel.g_position;
                    voxel_out.orientation = voxel.g_orientation;

                    FloatM body_voxel_size = _structure().bodies[voxel.body_sid].voxel_size;
                    FloatM half_voxel_size = 0.5 * body_voxel_size;
                    voxel_out.size = body_voxel_size;
                    voxel_out.nnn_offset =
                        RVec3rf(-half_voxel_size, -half_voxel_size, -half_voxel_size);
                    voxel_out.ppp_offset =
                        RS_Vec3D<FloatM>(half_voxel_size, half_voxel_size, half_voxel_size);
                    voxel_out.radius = half_voxel_size * std::sqrt(FloatM(3));

                    bool is_fixed =
                        voxel.material_type == RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED ||
                        voxel.material_type == RSE_STRUCTURE_VOXEL_MATERIAL_FIXED;
                    RS_VoxelStateUnion state{0};
                    state.voxel_state.previous_friction_state = RSE_VOXEL_NO_CONTACT_FRICTION;
                    state.voxel_state.is_enabled = true;
                    state.voxel_state.is_surface = voxel.is_surface;
                    state.voxel_state.is_static = is_fixed;
                    state.voxel_state.is_rigid = voxel.rigid_body_sid != RS_NULL_INDEX;
                    state.voxel_state.is_fixed = is_fixed;
                    state.voxel_state.is_structure_constrained = _is_constrained;

                    state.voxel_state.has_expansion = voxel.expansion_signal_sid != RS_NULL_INDEX;
                    state.voxel_state.expansion_control_mode = voxel.expansion_control_mode;

                    for (int l = 0; l < 6; l++) {
                        if (voxel.link_sid[l] != RS_NULL_INDEX) {
                            state.voxel_state.has_link = true;
                        }
                    }
                    state.voxel_state.face_role =
                        RS_VoxelStateUnion{voxel.role_state}.voxel_state.face_role;
                    voxel_out.state = state.voxel_state_packed;

                    voxels.get<VoxelMem>(i) = voxel_out;
                }
            )
            .wait();

    // print("v {}\n", timer.elapsedTimeString());
    // timer.reset();

    if (!_links.empty())
        _kernel_manager->getThreadPool()
            ->submitLoop(
                0,
                _links.size(),
                [&](RIndex i) {
                    auto &link = _links[i];
                    auto link_out = LinkMem();
                    auto link_material =
                        link_materials.get<LinkMaterialMem>(link.link_material_sid);
                    link_out.structure_gid = _structure_gid;
                    link_out.sid = i;
                    link_out.voxel_neg_sid = link.voxel_neg_sid;
                    link_out.voxel_pos_sid = link.voxel_pos_sid;
                    link_out.link_material_sid = link_material.sid;

                    link_out.strain_ratio = link_material.strain_ratio;
                    link_out.length = (voxels.get<VoxelMem>(link.voxel_pos_sid)
                                           .position.template toVec3D<RFloat>() -
                                       voxels.get<VoxelMem>(link.voxel_neg_sid)
                                           .position.template toVec3D<RFloat>())
                                          .length();

                    // Initial rest length is the distance between the two voxels
                    link_out.rest_length = link_out.length;

                    RS_LinkStateUnion state_union{0};
                    state_union.link_state.is_enabled = true;
                    state_union.link_state.axis = link.axis;
                    state_union.link_state.is_voxel_neg_rigid_or_fixed =
                        _voxels[link.voxel_neg_sid].material_type !=
                        RSE_STRUCTURE_VOXEL_MATERIAL_SOFT;
                    state_union.link_state.is_voxel_pos_rigid_or_fixed =
                        _voxels[link.voxel_pos_sid].material_type !=
                        RSE_STRUCTURE_VOXEL_MATERIAL_SOFT;
                    link_out.state = state_union.link_state_packed;
                    // Initial transverse area is the mean voxel transverse area
                    // between negative voxel and positive voxel
                    RFloat body_voxel_size = _structure().bodies[link.body_sid].voxel_size;
                    link_out.transverse_area = body_voxel_size * body_voxel_size;

                    links.get<LinkMem>(i) = link_out;
                }
            )
            .wait();

    // print("l {}\n", timer.elapsedTimeString());
    // timer.reset();

    if (!_rigid_bodies.empty())
        _kernel_manager->getThreadPool()
            ->submitLoop(
                0,
                _rigid_bodies.size(),
                [&](RIndex i) {
                    auto &rigid_body = _rigid_bodies[i];
                    auto rigid_body_out = RigidBodyMem();
                    rigid_body_out.structure_gid = _structure_gid;
                    rigid_body_out.sid = rigid_body.sid;
                    rigid_body_out.body_sid = rigid_body.body_sid;
                    rigid_body_out.segment_bid = rigid_body.segment_bid;
                    rigid_body_out.connected_body_mask_0 = rigid_body.connected_body_mask.data[0];
                    rigid_body_out.connected_body_mask_1 = rigid_body.connected_body_mask.data[1];
                    bool is_fixed =
                        _mat_resolved_data[rigid_body.contained_material_sid[0]].phys_type ==
                        RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED;
                    RS_RigidBodyStateUnion rb_state{0};
                    rb_state.rigid_body_state.is_static = is_fixed;
                    rb_state.rigid_body_state.is_fixed = is_fixed;
                    rigid_body_out.state = rb_state.rigid_body_state_packed;
                    rigid_body_out.initial_orientation = rigid_body.g_orientation;
                    rigid_body_out.orientation = rigid_body.g_orientation;
                    rigid_body_out.origin = rigid_body.origin_g_position;
                    rigid_bodies.get<RigidBodyMem>(i) = rigid_body_out;
                }
            )
            .wait();

    // Rigid body mass / center of mass / local moment of inertia inverse.
    vector<FloatM> rb_mass(_rigid_bodies.size());
    vector<FloatM> rb_mass_inverses(_rigid_bodies.size());
    vector<RS_Vec3DMem<FloatMemAccuracy>> rb_com_rb_positions(_rigid_bodies.size());
    vector<RS_Mat3DMem<FloatMemAccuracy>> rb_local_moment_inertia_inverses(_rigid_bodies.size());
    for (auto &&[i, rigid_body] : enumerate(_rigid_bodies)) {
        rb_mass[i] = rigid_body.mass;
        rb_mass_inverses[i] = rigid_body.mass_inverse;
        rb_com_rb_positions[i] = rigid_body.com_rb_position;
        rb_local_moment_inertia_inverses[i] = rigid_body.local_moment_inertia_inverse;
    }
    data.rigid_body_mass = RigidBodyFloatArray(_rigid_bodies.size(), provider);
    data.rigid_body_mass.write(rb_mass.data());
    data.rigid_body_mass_inverses = RigidBodyFloatArray(_rigid_bodies.size(), provider);
    data.rigid_body_mass_inverses.write(rb_mass_inverses.data());
    data.rigid_body_com_rb_positions = RigidBodyVec3DArray(_rigid_bodies.size(), provider);
    data.rigid_body_com_rb_positions.write(rb_com_rb_positions.data());
    data.rigid_body_local_moment_inertia_inverses =
        RigidBodyMat3DArray(_rigid_bodies.size(), provider);
    data.rigid_body_local_moment_inertia_inverses.write(rb_local_moment_inertia_inverses.data());

    // print("rb {}\n", timer.elapsedTimeString());
    // timer.reset();

    if (!_rigid_body_surface_voxels.empty())
        _kernel_manager->getThreadPool()
            ->submitLoop(
                0,
                _rigid_body_surface_voxels.size(),
                [&](RIndex i) {
                    auto &rigid_body_surface_voxel = _rigid_body_surface_voxels[i];
                    auto rigid_body_surface_voxel_out = RigidBodySurfaceVoxelMem();
                    rigid_body_surface_voxel_out.structure_gid = _structure_gid;
                    rigid_body_surface_voxel_out.voxel_sid = rigid_body_surface_voxel.voxel_sid;
                    rigid_body_surface_voxel_out.rigid_body_sid =
                        rigid_body_surface_voxel.rigid_body_sid;
                    rigid_body_surface_voxel_out.voxel_rb_position =
                        rigid_body_surface_voxel.voxel_rb_position;
                    rigid_body_surface_voxels.get<RigidBodySurfaceVoxelMem>(i) =
                        rigid_body_surface_voxel_out;
                }
            )
            .wait();

    // print("rbsv {}\n", timer.elapsedTimeString());
    // timer.reset();

    RFloat stiffness = _kernel_manager->kernel.rigidbody_constraint_stiffness_initial;
    if (!_constraint_joints.empty())
        _kernel_manager->getThreadPool()
            ->submitLoop(
                0,
                _constraint_joints.size(),
                [&](RIndex i) {
                    auto &cj = _constraint_joints[i];
                    auto cj_out = ConstraintJointMem();
                    RS_ConstraintJointStateUnion cj_state{0};
                    cj_state.constraint_joint_state.type = cj.type;
                    cj_state.constraint_joint_state.hinge_control_mode = cj.hinge_control_mode;
                    cj_state.constraint_joint_state.prismatic_control_mode =
                        cj.prismatic_control_mode;
                    cj_out.state = cj_state.constraint_joint_state_packed;
                    cj_out.structure_signal = _structure_signal;
                    cj_out.structure_gid = _structure_gid;
                    cj_out.sid = i;
                    cj_out.joint_sid = cj.joint_sid;
                    cj_out.motor_sid = cj.motor_sid;
                    cj_out.rigid_body_a_sid = cj.rigid_body_a_sid;
                    cj_out.rigid_body_b_sid = cj.rigid_body_b_sid;
                    cj_out.rigid_body_a_anchor_rb_position = cj.rigid_body_a_anchor_rb_position;
                    cj_out.rigid_body_b_anchor_rb_position = cj.rigid_body_b_anchor_rb_position;
                    cj_out.hinge_a_rb_axis = cj.hinge_a_rb_axis;
                    cj_out.hinge_b_rb_orthogonal_axis_1 = cj.hinge_b_rb_orthogonal_axis_1;
                    cj_out.hinge_b_rb_orthogonal_axis_2 = cj.hinge_b_rb_orthogonal_axis_2;
                    cj_out.hinge_min = cj.hinge_min;
                    cj_out.hinge_max = cj.hinge_max;
                    cj_out.hinge_position_control_kp = cj.hinge_position_control_kp;
                    cj_out.hinge_position_control_kd = cj.hinge_position_control_kd;
                    cj_out.hinge_max_speed = cj.hinge_max_speed;
                    cj_out.hinge_max_torque = cj.hinge_max_torque;
                    cj_out.hinge_signal_sid = cj.hinge_signal_sid;
                    cj_out.prismatic_a_rb_axis = cj.prismatic_a_rb_axis;
                    cj_out.prismatic_b_rb_orthogonal_axis_1 = cj.prismatic_b_rb_orthogonal_axis_1;
                    cj_out.prismatic_b_rb_orthogonal_axis_2 = cj.prismatic_b_rb_orthogonal_axis_2;
                    cj_out.prismatic_min = cj.prismatic_min;
                    cj_out.prismatic_max = cj.prismatic_max;
                    cj_out.prismatic_position_control_kp = cj.prismatic_position_control_kp;
                    cj_out.prismatic_position_control_kd = cj.prismatic_position_control_kd;
                    cj_out.prismatic_max_speed = cj.prismatic_max_speed;
                    cj_out.prismatic_max_force = cj.prismatic_max_force;
                    cj_out.prismatic_signal_sid = cj.prismatic_signal_sid;
                    cj_out.color = cj.color;
                    cj_out.stiffness = stiffness;

                    constraint_joints.get<ConstraintJointMem>(i) = cj_out;
                }
            )
            .wait();

    // print("cj {}\n", timer.elapsedTimeString());
    // timer.reset();

    data.voxel_materials = VoxelMaterialArray(_voxel_materials.size(), provider);
    data.voxel_materials.write(voxel_materials.as<VoxelMaterialMem>(), false);

    data.link_materials = LinkMaterialArray(_link_materials.size(), provider);
    data.link_materials.write(link_materials.as<LinkMaterialMem>(), false);

    data.voxels = VoxelArray(_voxels.size(), provider);
    data.voxels.write(voxels.as<VoxelMem>(), false);

    data.links = LinkArray(_links.size(), provider);
    data.links.write(links.as<LinkMem>(), false);

    data.rigid_bodies = RigidBodyArray(_rigid_bodies.size(), provider);
    data.rigid_bodies.write(rigid_bodies.as<RigidBodyMem>(), false);

    data.rigid_body_surface_voxels =
        RigidBodySurfaceVoxelArray(_rigid_body_surface_voxels.size(), provider);
    data.rigid_body_surface_voxels.write(
        rigid_body_surface_voxels.as<RigidBodySurfaceVoxelMem>(), false
    );

    data.constraint_joints = ConstraintJointArray(_constraint_joints.size(), provider);
    // Last write is synchronized
    data.constraint_joints.write(constraint_joints.as<ConstraintJointMem>());

    voxel_materials.free();
    link_materials.free();
    voxels.free();
    links.free();
    rigid_bodies.free();
    rigid_body_surface_voxels.free();
    constraint_joints.free();

    // print("write {}\n", timer.elapsedTimeString());
    // timer.reset();

    return move(data);
}

void RS_KernelStructureVoxelManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        auto &ctx = k_ptr->ctx;
        using ContextType = decltype(k_ptr->ctx);

        k_ptr->host_data->constraint_joint_max_colors =
            RS_MAX(k_ptr->host_data->constraint_joint_max_colors, getConstraintJointColorNum());
        k_ptr->host_data->max_constraint_joint_per_structure =
            RS_MAX(k_ptr->host_data->max_constraint_joint_per_structure, getConstraintJointNum());
        k_ptr->host_data->max_rigid_body_per_structure =
            RS_MAX(k_ptr->host_data->max_rigid_body_per_structure, getRigidBodyNum());

        auto structure = _getAll<ContextType::FloatMemAccuracy>();

        RSize link_material_num = structure.link_materials.size();
        RSize voxel_material_num = structure.voxel_materials.size();
        RSize voxel_num = structure.voxels.size();
        RSize link_num = structure.links.size();
        RSize rigid_body_num = structure.rigid_bodies.size();
        RSize rigid_body_surface_voxel_num = structure.rigid_body_surface_voxels.size();
        RSize constraint_joint_num = structure.constraint_joints.size();

        // Update host layout.
        // Note: Voxels must stay resolvable, collidable and renderable even when
        // the structure is fully fixed or frozen, so the voxel and voxel material layouts are
        // always active. The dynamic and static subsets are partitioned by stream compaction.
        bool is_active = !_structure().is_fixed;
        ctx.host_data->structure_link_material_layout.add(
            link_material_num, _structure_gid, is_active
        );
        ctx.host_data->structure_voxel_material_layout.add(
            voxel_material_num, _structure_gid, true
        );
        ctx.host_data->structure_link_layout.add(link_num, _structure_gid, is_active);
        ctx.host_data->structure_voxel_layout.add(voxel_num, _structure_gid, true);
        ctx.host_data->structure_rigid_body_layout.add(rigid_body_num, _structure_gid, is_active);
        ctx.host_data->structure_rigid_body_surface_voxel_layout.add(
            rigid_body_surface_voxel_num, _structure_gid, is_active
        );
        ctx.host_data->structure_constraint_joint_layout.add(
            constraint_joint_num, _structure_gid, is_active
        );
        ctx.host_data->structure_constraint_joint_block_range_layout.add(1, _structure_gid);

        if (!_structure().is_fixed) {
            for (auto &&[rigid_body_sid, rigid_body] : enumerate(_rigid_bodies)) {
                ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
                    .associate(
                        _structure_gid,
                        rigid_body.rigid_body_surface_voxel_start,
                        rigid_body.rigid_body_surface_voxel_end -
                            rigid_body.rigid_body_surface_voxel_start,
                        _structure_gid,
                        rigid_body_sid
                    );
            }
        }

        // Update device layout
        ctx.host_data->structure_link_material_layout.syncDeviceLayout(
            ctx.structure_link_material_layout
        );
        ctx.host_data->structure_voxel_material_layout.syncDeviceLayout(
            ctx.structure_voxel_material_layout
        );
        ctx.host_data->structure_link_layout.syncDeviceLayout(ctx.structure_link_layout);
        ctx.host_data->structure_voxel_layout.syncDeviceLayout(ctx.structure_voxel_layout);
        ctx.host_data->structure_rigid_body_layout.syncDeviceLayout(
            ctx.structure_rigid_body_layout
        );
        ctx.host_data->structure_rigid_body_surface_voxel_layout.syncDeviceLayout(
            ctx.structure_rigid_body_surface_voxel_layout
        );
        ctx.host_data->structure_constraint_joint_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_layout
        );
        ctx.host_data->structure_constraint_joint_block_range_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_block_range_layout
        );
        ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
            .syncDeviceInputLayout(
                ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_input_layout
            );
        ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
            .syncDeviceOutputLayout(
                ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_output_layout
            );

        // Adapt device storage size
        ctx.structure_link_materials.resize(
            ctx.host_data->structure_link_material_layout.capacity()
        );
        ctx.structure_voxel_materials.resize(
            ctx.host_data->structure_voxel_material_layout.capacity()
        );
        ctx.structure_links.resize(ctx.host_data->structure_link_layout.capacity());
        ctx.structure_voxels.resize(ctx.host_data->structure_voxel_layout.capacity());
        ctx.structure_rigid_bodies.resize(ctx.host_data->structure_rigid_body_layout.capacity());
        ctx.structure_rigid_body_surface_voxels.resize(
            ctx.host_data->structure_rigid_body_surface_voxel_layout.capacity()
        );
        ctx.structure_constraint_joints.resize(
            ctx.host_data->structure_constraint_joint_layout.capacity()
        );

        // Layout and SoA pointers changed, needs to update kernel.
        _kernel_manager->markKernelModified();

        // Transfer structure data to device context
        RSize link_material_offset = ctx.host_data->structure_link_material_layout.segmentOffset(
            ctx.host_data->structure_link_material_layout.find(_structure_gid)
        );
        copy(
            ctx.structure_link_materials,
            structure.link_materials,
            link_material_num,
            link_material_offset
        );

        RSize voxel_material_offset = ctx.host_data->structure_voxel_material_layout.segmentOffset(
            ctx.host_data->structure_voxel_material_layout.find(_structure_gid)
        );
        copy(
            ctx.structure_voxel_materials,
            structure.voxel_materials,
            voxel_material_num,
            voxel_material_offset
        );

        RSize link_offset = ctx.host_data->structure_link_layout.segmentOffset(
            ctx.host_data->structure_link_layout.find(_structure_gid)
        );
        copy(ctx.structure_links, structure.links, link_num, link_offset);

        RSize voxel_offset = ctx.host_data->structure_voxel_layout.segmentOffset(
            ctx.host_data->structure_voxel_layout.find(_structure_gid)
        );
        copy(ctx.structure_voxels, structure.voxels, voxel_num, voxel_offset);

        RSize rigid_body_offset = ctx.host_data->structure_rigid_body_layout.segmentOffset(
            ctx.host_data->structure_rigid_body_layout.find(_structure_gid)
        );
        copy(ctx.structure_rigid_bodies, structure.rigid_bodies, rigid_body_num, rigid_body_offset);

        RSize rigid_body_surface_voxel_offset =
            ctx.host_data->structure_rigid_body_surface_voxel_layout.segmentOffset(
                ctx.host_data->structure_rigid_body_surface_voxel_layout.find(_structure_gid)
            );
        copy(
            ctx.structure_rigid_body_surface_voxels,
            structure.rigid_body_surface_voxels,
            rigid_body_surface_voxel_num,
            rigid_body_surface_voxel_offset
        );

        RSize constraint_joint_offset =
            ctx.host_data->structure_constraint_joint_layout.segmentOffset(
                ctx.host_data->structure_constraint_joint_layout.find(_structure_gid)
            );
        copy(
            ctx.structure_constraint_joints,
            structure.constraint_joints,
            constraint_joint_num,
            constraint_joint_offset
        );

        logic::RS_ConstraintJointBlockRange constraint_joint_block_range{
            constraint_joint_offset,
            constraint_joint_num,
            rigid_body_offset,
            rigid_body_num,
        };
        ctx.structure_constraint_joint_block_ranges.append(constraint_joint_block_range);

        _initKernelEmissiveVoxelHandlesData(voxel_offset);

        // Resolve data dependency
        _kernel_manager->syncKernelIfNeeded();
        castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
            using KernelType = std::remove_pointer_t<decltype(k_ptr)>;

            p_ptr->template run<RSCF_LaunchDirect<RSCF_ResolveVoxelDependency, KernelType>>(
                {voxel_num}, voxel_offset, link_offset, voxel_material_offset, rigid_body_offset
            );
            p_ptr->template run<RSCF_LaunchDirect<RSCF_ResolveLinkDependency, KernelType>>(
                {link_num}, link_offset, voxel_offset, link_material_offset
            );
            p_ptr->template run<
                RSCF_LaunchDirect<RSCF_ResolveRigidBodySurfaceVoxelDependency, KernelType>>(
                {rigid_body_surface_voxel_num},
                rigid_body_surface_voxel_offset,
                voxel_offset,
                rigid_body_offset
            );
            p_ptr->template run<RSCF_LaunchDirect<RSCF_UpdateRigidBodyProperties, KernelType>>(
                {rigid_body_num},
                rigid_body_offset,
                structure.rigid_body_mass,
                structure.rigid_body_mass_inverses,
                structure.rigid_body_com_rb_positions,
                structure.rigid_body_local_moment_inertia_inverses
            );
            p_ptr->template run<
                RSCF_LaunchDirect<RSCF_ResolveConstraintJointDependency, KernelType>>(
                {constraint_joint_num}, constraint_joint_offset, rigid_body_offset
            );
        });

        structure.link_materials.free();
        structure.voxel_materials.free();
        structure.links.free();
        structure.voxels.free();
        structure.rigid_bodies.free();
        structure.rigid_body_surface_voxels.free();
        structure.constraint_joints.free();
        structure.rigid_body_mass.free();
        structure.rigid_body_mass_inverses.free();
        structure.rigid_body_com_rb_positions.free();
        structure.rigid_body_local_moment_inertia_inverses.free();
    });

    // Voxel data changed, needs to update partition.
    _kernel_manager->getKernelSpatialManager().markVoxelPartitionModified();
}

void RS_KernelStructureVoxelManager::_freeKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        if (!_is_kernel_data_inited) return;

        auto &ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        k_ptr->host_data->constraint_joint_max_colors = 0;
        k_ptr->host_data->max_constraint_joint_per_structure = 0;
        k_ptr->host_data->max_rigid_body_per_structure = 0;

        for (auto &structure_manager_p : _kernel_manager->getKernelStructureVoxelManagers()) {
            if (structure_manager_p.second->getEntityGid() != _structure_gid) {
                k_ptr->host_data->constraint_joint_max_colors = RS_MAX(
                    k_ptr->host_data->constraint_joint_max_colors,
                    structure_manager_p.second->getConstraintJointColorNum()
                );
                k_ptr->host_data->max_constraint_joint_per_structure = RS_MAX(
                    k_ptr->host_data->max_constraint_joint_per_structure,
                    structure_manager_p.second->getConstraintJointNum()
                );
                k_ptr->host_data->max_rigid_body_per_structure = RS_MAX(
                    k_ptr->host_data->max_rigid_body_per_structure,
                    structure_manager_p.second->getRigidBodyNum()
                );
            }
        }

        // Disable corresponding structure voxels and links
        _kernel_manager->syncKernelIfNeeded();
        castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
            using KernelType = std::remove_pointer_t<decltype(k_ptr)>;

            RIndex voxel_offset = ctx.host_data->structure_voxel_layout.segmentOffset(
                ctx.host_data->structure_voxel_layout.find(_structure_gid)
            );
            RIndex voxel_num = ctx.host_data->structure_voxel_layout.segmentSize(
                ctx.host_data->structure_voxel_layout.find(_structure_gid)
            );
            RIndex link_offset = ctx.host_data->structure_link_layout.segmentOffset(
                ctx.host_data->structure_link_layout.find(_structure_gid)
            );
            RIndex link_num = ctx.host_data->structure_link_layout.segmentSize(
                ctx.host_data->structure_link_layout.find(_structure_gid)
            );

            p_ptr->template run<RSCF_LaunchDirect<RSCF_DisableVoxel, KernelType>>(
                {voxel_num}, voxel_offset
            );
            p_ptr->template run<RSCF_LaunchDirect<RSCF_DisableLink, KernelType>>(
                {link_num}, link_offset
            );
        });

        RIndex constraint_joint_block_range_index =
            ctx.host_data->structure_constraint_joint_block_range_layout.find(_structure_gid);

        // Update host layout
        ctx.host_data->structure_link_material_layout.remove(
            ctx.host_data->structure_link_material_layout.find(_structure_gid)
        );
        ctx.host_data->structure_voxel_material_layout.remove(
            ctx.host_data->structure_voxel_material_layout.find(_structure_gid)
        );
        ctx.host_data->structure_link_layout.remove(
            ctx.host_data->structure_link_layout.find(_structure_gid)
        );
        ctx.host_data->structure_voxel_layout.remove(
            ctx.host_data->structure_voxel_layout.find(_structure_gid)
        );
        _freeKernelEmissiveVoxelHandlesData();
        ctx.host_data->structure_rigid_body_layout.remove(
            ctx.host_data->structure_rigid_body_layout.find(_structure_gid)
        );
        ctx.host_data->structure_rigid_body_surface_voxel_layout.remove(
            ctx.host_data->structure_rigid_body_surface_voxel_layout.find(_structure_gid)
        );
        ctx.host_data->structure_constraint_joint_layout.remove(
            ctx.host_data->structure_constraint_joint_layout.find(_structure_gid)
        );
        ctx.host_data->structure_constraint_joint_block_range_layout.remove(
            constraint_joint_block_range_index
        );

        // Update device layout
        ctx.host_data->structure_link_material_layout.syncDeviceLayout(
            ctx.structure_link_material_layout
        );
        ctx.host_data->structure_voxel_material_layout.syncDeviceLayout(
            ctx.structure_voxel_material_layout
        );
        ctx.host_data->structure_link_layout.syncDeviceLayout(ctx.structure_link_layout);
        ctx.host_data->structure_voxel_layout.syncDeviceLayout(ctx.structure_voxel_layout);
        ctx.host_data->structure_rigid_body_layout.syncDeviceLayout(
            ctx.structure_rigid_body_layout
        );
        ctx.host_data->structure_rigid_body_surface_voxel_layout.syncDeviceLayout(
            ctx.structure_rigid_body_surface_voxel_layout
        );
        ctx.host_data->structure_constraint_joint_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_layout
        );
        ctx.host_data->structure_constraint_joint_block_range_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_block_range_layout
        );
        ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
            .syncDeviceInputLayout(
                ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_input_layout
            );
        ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
            .syncDeviceOutputLayout(
                ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_output_layout
            );

        // Adapt device storage size
        ctx.structure_link_materials.resize(
            ctx.host_data->structure_link_material_layout.capacity()
        );
        ctx.structure_voxel_materials.resize(
            ctx.host_data->structure_voxel_material_layout.capacity()
        );
        ctx.structure_links.resize(ctx.host_data->structure_link_layout.capacity());
        ctx.structure_voxels.resize(ctx.host_data->structure_voxel_layout.capacity());
        ctx.structure_rigid_bodies.resize(ctx.host_data->structure_rigid_body_layout.capacity());
        ctx.structure_rigid_body_surface_voxels.resize(
            ctx.host_data->structure_rigid_body_surface_voxel_layout.capacity()
        );
        ctx.structure_constraint_joints.resize(
            ctx.host_data->structure_constraint_joint_layout.capacity()
        );
        ctx.structure_constraint_joint_block_ranges.erase(constraint_joint_block_range_index);

        // Layout and SoA pointers changed, needs to update kernel.
        _kernel_manager->markKernelModified();
    });

    // Voxel data changed, needs to update partition.
    _kernel_manager->getKernelSpatialManager().markVoxelPartitionModified();
}

void RS_KernelStructureVoxelManager::_initKernelRecordData() {
    auto memory_provider = _scoped_device_provider.get();
    auto buffer_frame_num = _kernel_manager->getKernelRecordBufferFrameNum();
    const rsc::RSC_StructureVoxelRecord &record_cfg = _structure().record;

    _voxel_record_num = 0;
    _link_record_num = 0;
    _rigid_body_record_num = 0;
    _joint_record_num = 0;

    // Voxel record sid/handles
    if (record_cfg.voxel_record_range == rsc::RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_ALL) {
        _voxel_record_sid = RS_Array<RIndex, 1>(_voxel_num, memory_provider);
        _voxel_record_handles = RS_Array<RHandle, 1>(_voxel_num, memory_provider);
        generateIndex(_voxel_record_sid);
    } else if (record_cfg.voxel_record_range ==
               rsc::RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_RANDOM) {
        RSize voxel_sample_num = RS_MIN(_voxel_num, record_cfg.voxel_sample_num);
        auto random_voxel_sid = _selectSID(_structure().name, voxel_sample_num, _voxel_num);
        _voxel_record_sid = RS_Array<RIndex, 1>(random_voxel_sid.size(), memory_provider);
        _voxel_record_handles = RS_Array<RHandle, 1>(random_voxel_sid.size(), memory_provider);
        _voxel_record_sid.write(random_voxel_sid.data());
    }
    _voxel_record_num = _voxel_record_sid.size();

    // Link record sid/handles
    if (record_cfg.link_record_range == rsc::RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_ALL) {
        _link_record_sid = RS_Array<RIndex, 1>(_link_num, memory_provider);
        _link_record_handles = RS_Array<RHandle, 1>(_link_num, memory_provider);
        generateIndex(_link_record_sid);
    } else if (record_cfg.link_record_range == rsc::RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_RANDOM) {
        RSize link_sample_num = RS_MIN(_link_num, record_cfg.link_sample_num);
        auto random_link_sid = _selectSID(_structure().name, link_sample_num, _link_num);
        _link_record_sid = RS_Array<RIndex, 1>(random_link_sid.size(), memory_provider);
        _link_record_handles = RS_Array<RHandle, 1>(random_link_sid.size(), memory_provider);
        _link_record_sid.write(random_link_sid.data());
    }
    _link_record_num = _link_record_sid.size();

    // Rigid body record sid/handles
    if (record_cfg.rigid_body_record_range ==
        rsc::RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_ALL) {
        _rigid_body_record_num = _rigid_body_num;
        _rigid_body_record_sid = RS_Array<RIndex, 1>(_rigid_body_num, memory_provider);
        _rigid_body_record_handles = RS_Array<RHandle, 1>(_rigid_body_num, memory_provider);
        generateIndex(_rigid_body_record_sid);
    }

    // Joint record sid/handles (handles are per constraint joint)
    if (record_cfg.joint_record_range == rsc::RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_ALL) {
        _joint_record_num = _joint_num;
        _constraint_joint_record_sid = RS_Array<RIndex, 1>(_constraint_joint_num, memory_provider);
        _constraint_joint_record_handles =
            RS_Array<RHandle, 1>(_constraint_joint_num, memory_provider);
        generateIndex(_constraint_joint_record_sid);
    }

    // Record buffers
    RBigSize frame_size = RS_WorldStructureVoxelRecord::getRequiredMemorySize(
        _voxel_record_num, _link_record_num, _rigid_body_record_num, _joint_record_num
    );
    _recorder_buffer.init(buffer_frame_num, buffer_frame_num * frame_size, memory_provider);
    _controller_buffer.init(frame_size, memory_provider);

    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Update host layout
        ctx.host_data->structure_record_layout.add(1, _structure_gid);
        ctx.host_data->structure_voxel_record_layout.add(_voxel_record_num, _structure_gid);
        ctx.host_data->structure_link_record_layout.add(_link_record_num, _structure_gid);
        ctx.host_data->structure_rigid_body_record_layout.add(
            _rigid_body_record_num, _structure_gid
        );
        ctx.host_data->structure_joint_record_layout.add(_joint_record_num, _structure_gid);
        ctx.host_data->structure_constraint_joint_record_layout.add(
            _constraint_joint_record_sid.size(), _structure_gid
        );

        // Update device layout
        ctx.host_data->structure_voxel_record_layout.syncDeviceLayout(
            ctx.structure_voxel_record_layout
        );
        ctx.host_data->structure_link_record_layout.syncDeviceLayout(
            ctx.structure_link_record_layout
        );
        ctx.host_data->structure_rigid_body_record_layout.syncDeviceLayout(
            ctx.structure_rigid_body_record_layout
        );
        ctx.host_data->structure_joint_record_layout.syncDeviceLayout(
            ctx.structure_joint_record_layout
        );
        ctx.host_data->structure_constraint_joint_record_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_record_layout
        );

        // Layout and SoA pointers changed, needs to update kernel.
        _kernel_manager->markKernelModified();

        // Resolve handles needed by the record
        _kernel_manager->syncKernelIfNeeded();
        castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
            p_ptr->template run<RSCF_ResolveRecordDependency>(
                {_voxel_record_sid.size()},
                ctx.host_data->structure_voxel_layout.segmentOffset(
                    ctx.host_data->structure_voxel_layout.find(_structure_gid)
                ),
                _voxel_record_sid,
                _voxel_record_handles
            );

            p_ptr->template run<RSCF_ResolveRecordDependency>(
                {_link_record_sid.size()},
                ctx.host_data->structure_link_layout.segmentOffset(
                    ctx.host_data->structure_link_layout.find(_structure_gid)
                ),
                _link_record_sid,
                _link_record_handles
            );

            p_ptr->template run<RSCF_ResolveRecordDependency>(
                {_rigid_body_record_sid.size()},
                ctx.host_data->structure_rigid_body_layout.segmentOffset(
                    ctx.host_data->structure_rigid_body_layout.find(_structure_gid)
                ),
                _rigid_body_record_sid,
                _rigid_body_record_handles
            );

            p_ptr->template run<RSCF_ResolveRecordDependency>(
                {_constraint_joint_record_sid.size()},
                ctx.host_data->structure_constraint_joint_layout.segmentOffset(
                    ctx.host_data->structure_constraint_joint_layout.find(_structure_gid)
                ),
                _constraint_joint_record_sid,
                _constraint_joint_record_handles
            );
        });

        // Append per-structure record slots to device vectors
        ctx.structure_records.append({});
        ctx.structure_recorder_buffers.append(_recorder_buffer);
        ctx.structure_controller_buffers.append(_controller_buffer);

        ctx.structure_voxel_record_handles.append(_voxel_record_handles);
        ctx.structure_link_record_handles.append(_link_record_handles);
        ctx.structure_rigid_body_record_handles.append(_rigid_body_record_handles);
        ctx.structure_constraint_joint_record_handles.append(_constraint_joint_record_handles);

        // Layout and SoA pointers changed, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelStructureVoxelManager::_freeKernelRecordData() {
    if (!_is_kernel_data_inited) return;

    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        RIndex record_index = ctx.host_data->structure_record_layout.find(_structure_gid);

        // Update host layout
        ctx.host_data->structure_record_layout.remove(record_index);
        ctx.host_data->structure_voxel_record_layout.remove(record_index);
        ctx.host_data->structure_link_record_layout.remove(record_index);
        ctx.host_data->structure_rigid_body_record_layout.remove(record_index);
        ctx.host_data->structure_joint_record_layout.remove(record_index);
        ctx.host_data->structure_constraint_joint_record_layout.remove(record_index);

        // Update device layout
        ctx.host_data->structure_voxel_record_layout.syncDeviceLayout(
            ctx.structure_voxel_record_layout
        );
        ctx.host_data->structure_link_record_layout.syncDeviceLayout(
            ctx.structure_link_record_layout
        );
        ctx.host_data->structure_rigid_body_record_layout.syncDeviceLayout(
            ctx.structure_rigid_body_record_layout
        );
        ctx.host_data->structure_joint_record_layout.syncDeviceLayout(
            ctx.structure_joint_record_layout
        );
        ctx.host_data->structure_constraint_joint_record_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_record_layout
        );

        // Erase per-structure record slots from device vectors
        ctx.structure_records.erase(record_index);
        ctx.structure_recorder_buffers.erase(record_index);
        ctx.structure_controller_buffers.erase(record_index);
        ctx.structure_voxel_record_handles.erase(record_index);
        ctx.structure_link_record_handles.erase(record_index);
        ctx.structure_rigid_body_record_handles.erase(record_index);
        ctx.structure_constraint_joint_record_handles.erase(record_index);

        // Layout and SoA pointers changed, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelStructureVoxelManager::_initKernelSignalData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        ctx.host_data->structure_signal_layout.add(1, _structure_gid);
        ctx.host_data->structure_signal_layout.syncDeviceLayout(ctx.structure_signal_layout);
        ctx.structure_signals.resize({ctx.host_data->structure_signal_layout.capacity()});
        _structure_signal = ctx.host_data->structure_signal_layout.segmentOffset(
            ctx.host_data->structure_signal_layout.find(_structure_gid)
        );
        ctx.structure_signals.writeSlice(&_device_signal, 1, _structure_signal);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelStructureVoxelManager::_freeKernelSignalData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;

        RIndex signal_index = ctx.host_data->structure_signal_layout.find(_structure_gid);
        ctx.host_data->structure_signal_layout.remove(signal_index);
        ctx.host_data->structure_signal_layout.syncDeviceLayout(ctx.structure_signal_layout);
        ctx.structure_signals.resize({ctx.host_data->structure_signal_layout.capacity()});

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelStructureVoxelManager::_initKernelEmissiveVoxelHandlesData(RSize voxel_offset) {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;
        auto &map = ctx.host_data->structure_emissive_voxel_handles_map;

        std::vector<RHandle> handles_this;
        handles_this.reserve(_emissive_voxel_num);
        for (auto &voxel : _voxels) {
            if (voxel.material_sid == RS_NULL_INDEX) continue;
            if (!_mat_resolved_data[voxel.material_sid].is_emissive) continue;
            handles_this.push_back(static_cast<RHandle>(voxel_offset + voxel.sid));
        }
        if (handles_this.empty()) {
            map.erase(_structure_gid);
        } else {
            map[_structure_gid] = std::move(handles_this);
        }

        std::vector<RHandle> all;
        RSize total = 0;
        for (auto &[gid, vec] : map) total += vec.size();
        all.reserve(total);
        for (auto &[gid, vec] : map) all.insert(all.end(), vec.begin(), vec.end());
        std::sort(all.begin(), all.end());

        ctx.structure_emissive_voxel_handles.resize(all.size(), false);
        ctx.structure_emissive_voxel_handles.occupy(all.size());
        if (!all.empty()) {
            ctx.structure_emissive_voxel_handles.data.copyFrom(
                RS_ConstMemoryPtr(all.data(), sizeof(RHandle) * all.size()),
                sizeof(RHandle) * all.size()
            );
        }

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
        // The light list shifted, so previous reservoirs may index a removed light.
        _kernel_manager->markVisionRestirReservoirStale();
    });
}

void RS_KernelStructureVoxelManager::_freeKernelEmissiveVoxelHandlesData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;
        auto &map = ctx.host_data->structure_emissive_voxel_handles_map;
        map.erase(_structure_gid);

        std::vector<RHandle> all;
        RSize total = 0;
        for (auto &[gid, vec] : map) total += vec.size();
        all.reserve(total);
        for (auto &[gid, vec] : map) all.insert(all.end(), vec.begin(), vec.end());
        std::sort(all.begin(), all.end());

        ctx.structure_emissive_voxel_handles.resize(all.size(), false);
        ctx.structure_emissive_voxel_handles.occupy(all.size());
        if (!all.empty()) {
            ctx.structure_emissive_voxel_handles.data.copyFrom(
                RS_ConstMemoryPtr(all.data(), sizeof(RHandle) * all.size()),
                sizeof(RHandle) * all.size()
            );
        }

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
        // The light list shifted, so previous reservoirs may index a removed light.
        _kernel_manager->markVisionRestirReservoirStale();
    });
}

void RS_KernelStructureVoxelManager::_freezeKernelData(bool is_freeze) {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto &ctx = k_ptr->ctx;
        using KernelType = std::remove_pointer_t<decltype(k_ptr)>;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        RIndex voxel_segment = ctx.host_data->structure_voxel_layout.find(_structure_gid);
        RSize voxel_offset = ctx.host_data->structure_voxel_layout.segmentOffset(voxel_segment);
        RSize voxel_num = ctx.host_data->structure_voxel_layout.segmentSize(voxel_segment);

        RIndex rigid_body_segment = ctx.host_data->structure_rigid_body_layout.find(_structure_gid);
        RSize rigid_body_offset =
            ctx.host_data->structure_rigid_body_layout.segmentOffset(rigid_body_segment);
        RSize rigid_body_num =
            ctx.host_data->structure_rigid_body_layout.segmentSize(rigid_body_segment);

        // Flip the effective immobility of every voxel and rigid body in the structure.
        _kernel_manager->syncKernelIfNeeded();
        castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
            p_ptr->template run<RSCF_LaunchDirect<RSCF_SetVoxelStaticOverSegment, KernelType>>(
                {voxel_num}, voxel_offset, is_freeze
            );
            p_ptr->template run<RSCF_LaunchDirect<RSCF_SetRigidBodyStaticOverSegment, KernelType>>(
                {rigid_body_num}, rigid_body_offset, is_freeze
            );
        });

        // Pause the compute, signal and record layouts while frozen; the voxel and voxel
        // material layouts stay active so the structure remains collidable and rendered.
        bool is_active = !is_freeze;
        ctx.host_data->structure_link_material_layout.setSegmentIsActive(
            ctx.host_data->structure_link_material_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_link_layout.setSegmentIsActive(
            ctx.host_data->structure_link_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_rigid_body_layout.setSegmentIsActive(
            ctx.host_data->structure_rigid_body_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_rigid_body_surface_voxel_layout.setSegmentIsActive(
            ctx.host_data->structure_rigid_body_surface_voxel_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_constraint_joint_layout.setSegmentIsActive(
            ctx.host_data->structure_constraint_joint_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_signal_layout.setSegmentIsActive(
            ctx.host_data->structure_signal_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_voxel_record_layout.setSegmentIsActive(
            ctx.host_data->structure_voxel_record_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_link_record_layout.setSegmentIsActive(
            ctx.host_data->structure_link_record_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_rigid_body_record_layout.setSegmentIsActive(
            ctx.host_data->structure_rigid_body_record_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_joint_record_layout.setSegmentIsActive(
            ctx.host_data->structure_joint_record_layout.find(_structure_gid), is_active
        );
        ctx.host_data->structure_constraint_joint_record_layout.setSegmentIsActive(
            ctx.host_data->structure_constraint_joint_record_layout.find(_structure_gid), is_active
        );

        ctx.host_data->structure_link_material_layout.syncDeviceLayout(
            ctx.structure_link_material_layout
        );
        ctx.host_data->structure_link_layout.syncDeviceLayout(ctx.structure_link_layout);
        ctx.host_data->structure_rigid_body_layout.syncDeviceLayout(
            ctx.structure_rigid_body_layout
        );
        ctx.host_data->structure_rigid_body_surface_voxel_layout.syncDeviceLayout(
            ctx.structure_rigid_body_surface_voxel_layout
        );
        ctx.host_data->structure_constraint_joint_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_layout
        );
        ctx.host_data->structure_signal_layout.syncDeviceLayout(ctx.structure_signal_layout);
        ctx.host_data->structure_voxel_record_layout.syncDeviceLayout(
            ctx.structure_voxel_record_layout
        );
        ctx.host_data->structure_link_record_layout.syncDeviceLayout(
            ctx.structure_link_record_layout
        );
        ctx.host_data->structure_rigid_body_record_layout.syncDeviceLayout(
            ctx.structure_rigid_body_record_layout
        );
        ctx.host_data->structure_joint_record_layout.syncDeviceLayout(
            ctx.structure_joint_record_layout
        );
        ctx.host_data->structure_constraint_joint_record_layout.syncDeviceLayout(
            ctx.structure_constraint_joint_record_layout
        );

        // Layout active state changed, needs to update kernel.
        _kernel_manager->markKernelModified();
    });

    // Voxels and rigid bodies moved between the dynamic and static subsets.
    _kernel_manager->getKernelSpatialManager().markVoxelPartitionModified();
}

logic::RSE_ConstraintJointHingeControlMode
RS_KernelStructureVoxelManager::_convertToConstraintJointHingeControlMode(
    rsc::RSE_StructureVoxelHingeControlMode mode
) {
    switch (mode) {
        case rsc::RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION:
            return RSE_CONSTRAINT_JOINT_HINGE_CONTROL_BASIC_POSITION;
        case rsc::RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION:
            return RSE_CONSTRAINT_JOINT_HINGE_CONTROL_PD_POSITION;
        case rsc::RSE_STRUCTURE_VOXEL_HINGE_CONTROL_TORQUE:
            return RSE_CONSTRAINT_JOINT_HINGE_CONTROL_TORQUE;
        case rsc::RSE_STRUCTURE_VOXEL_HINGE_CONTROL_VELOCITY:
            return RSE_CONSTRAINT_JOINT_HINGE_CONTROL_VELOCITY;
    }
    RS_THROW(
        RS_InvalidCharacterizationException("Unknown hinge control mode {}", enumToString(mode))
    );
}

logic::RSE_ConstraintJointPrismaticControlMode
RS_KernelStructureVoxelManager::_convertToConstraintJointPrismaticControlMode(
    rsc::RSE_StructureVoxelPrismaticControlMode mode
) {
    switch (mode) {
        case rsc::RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION:
            return RSE_CONSTRAINT_JOINT_PRISMATIC_CONTROL_BASIC_POSITION;
        case rsc::RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION:
            return RSE_CONSTRAINT_JOINT_PRISMATIC_CONTROL_PD_POSITION;
        case rsc::RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_FORCE:
            return RSE_CONSTRAINT_JOINT_PRISMATIC_CONTROL_FORCE;
        case rsc::RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_VELOCITY:
            return RSE_CONSTRAINT_JOINT_PRISMATIC_CONTROL_VELOCITY;
    }
    RS_THROW(
        RS_InvalidCharacterizationException("Unknown prismatic control mode {}", enumToString(mode))
    );
}

vector<RIndex> RS_KernelStructureVoxelManager::_selectSID(
    const std::string &structure_name,
    RSize select_num,
    RSize range
) {
    vector<RIndex> all_sids(range);
    for (RIndex i = 0; i < range; ++i) {
        all_sids[i] = i;
    }

    // Create a seed from the structure name for reproducibility
    seed_seq seed(structure_name.begin(), structure_name.end());

    // Initialize the random number generator with the seed
    mt19937 generator(seed);
    shuffle(all_sids.begin(), all_sids.end(), generator);
    vector<RIndex> record_sid(all_sids.begin(), all_sids.begin() + select_num);

    return record_sid;
}

// New partition method, an array of 24 links will access neighboring 8 voxels, ensuring cache
// efficiency (not affecting voxel sim efficiency too much, but speed up link sim efficiency)
vector<RS_KernelStructureVoxelManager::Cuboid>
RS_KernelStructureVoxelManager::_partitionSpace(RSize x_size, RSize y_size, RSize z_size) {
    // Fixed number of splits along each axis
    constexpr RSize splits_x = 2;
    constexpr RSize splits_y = 2;
    constexpr RSize splits_z = 2;

    // Compute the partition sizes using ceiling division to ensure full coverage.
    // (RS_CEIL_DIV(a, b) computes ceil(a/b)).
    RSize x_partition_size = RS_CEIL_DIV(x_size, splits_x);
    RSize y_partition_size = RS_CEIL_DIV(y_size, splits_y);
    RSize z_partition_size = RS_CEIL_DIV(z_size, splits_z);

    std::vector<Cuboid> cuboids;
    cuboids.reserve(splits_x * splits_y * splits_z);

    // Loop over each split in the z, y, and x directions.
    for (RSize z = 0; z < splits_z; ++z) {
        RIndex z_start = z * z_partition_size;
        RIndex z_end = RS_MIN(z_start + z_partition_size, z_size);

        for (RSize y = 0; y < splits_y; ++y) {
            RIndex y_start = y * y_partition_size;
            RIndex y_end = RS_MIN(y_start + y_partition_size, y_size);

            for (RSize x = 0; x < splits_x; ++x) {
                RIndex x_start = x * x_partition_size;
                RIndex x_end = RS_MIN(x_start + x_partition_size, x_size);

                cuboids.emplace_back(Cuboid{x_start, x_end, y_start, y_end, z_start, z_end});
            }
        }
    }

    return std::move(cuboids);
}

RBound3f RS_KernelStructureVoxelManager::_computeWorldBound(
    const rsc::RSC_StructureVoxel &structure_rsc
) {
    RBound3f world_bound = RBound3f::lowest();

    for (const auto &body : structure_rsc.bodies) {
        // Exact body-local voxel box: cells span [0, N * voxel_size] on each axis.
        RVec3rf local_min = RVec3rf::zero();
        RVec3rf local_max(
            RFLOAT(body.x_voxels) * body.voxel_size,
            RFLOAT(body.y_voxels) * body.voxel_size,
            RFLOAT(body.z_voxels) * body.voxel_size
        );

        // Same transform chain as _addBody, so rotation and offsets are accounted for.
        auto body_orientation = structure_rsc.orientation * body.relative_orientation;
        auto body_origin_position =
            structure_rsc.origin_position +
            structure_rsc.orientation.rotateVec3D(body.relative_origin_position);

        // Transform all 8 corners and union them; the rotated box becomes an AABB in world space.
        for (int corner = 0; corner < 8; corner++) {
            RVec3rf local_corner(
                (corner & 1) ? local_max.x : local_min.x,
                (corner & 2) ? local_max.y : local_min.y,
                (corner & 4) ? local_max.z : local_min.z
            );
            world_bound = world_bound.extend(
                body_origin_position + body_orientation.rotateVec3D(local_corner)
            );
        }
    }

    return world_bound;
}

/*****************************************************************
 * RS_KernelStructureVoxelReference
 *****************************************************************/
RS_KernelStructureVoxelReference::RS_KernelStructureVoxelReference(
    RS_KernelManager *kernel_manager,
    std::string name
)
    : _kernel_manager(kernel_manager), _name(std::move(name)) {
    auto &managers = _kernel_manager->getKernelStructureVoxelManagers();
    auto it = managers.find(_name);
    if (it == managers.end()) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure {} is not registered in the kernel manager", _name
        ));
    }
    it->second->reference();
    _gid = it->second->getEntityGid();
}

RS_KernelStructureVoxelReference::~RS_KernelStructureVoxelReference() { _release(); }

RS_KernelStructureVoxelReference::RS_KernelStructureVoxelReference(
    RS_KernelStructureVoxelReference &&other
) noexcept
    : _kernel_manager(other._kernel_manager), _name(std::move(other._name)), _gid(other._gid) {
    other._kernel_manager = nullptr;
    other._gid = RS_NULL_INDEX;
}

RS_KernelStructureVoxelReference &RS_KernelStructureVoxelReference::operator=(
    RS_KernelStructureVoxelReference &&other
) noexcept {
    if (this != &other) {
        _release();
        _kernel_manager = other._kernel_manager;
        _name = std::move(other._name);
        _gid = other._gid;
        other._kernel_manager = nullptr;
        other._gid = RS_NULL_INDEX;
    }
    return *this;
}

void RS_KernelStructureVoxelReference::_release() {
    if (_kernel_manager == nullptr) return;
    auto &managers = _kernel_manager->getKernelStructureVoxelManagers();
    if (auto it = managers.find(_name); it != managers.end()) {
        it->second->dereference();
    }
    _kernel_manager = nullptr;
    _gid = RS_NULL_INDEX;
}

}  // namespace rise
