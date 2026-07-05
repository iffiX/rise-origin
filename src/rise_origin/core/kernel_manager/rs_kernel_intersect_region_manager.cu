// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/logic/record/rs_intersect_region_c.h"
#include "rise_origin/logic/utils/rs_launch_c.h"
#include "rs_kernel_intersect_region_manager.h"

using namespace std;
using namespace rise::rsc;
using namespace rise::logic;
using namespace rise::device;

namespace rise {

RS_KernelIntersectRegionManager::RS_KernelIntersectRegionManager(
    RS_KernelManager* kernel_manager,
    const SupportedRSC& region_rsc
)
    : _kernel_manager(kernel_manager), _rsc(region_rsc) {
    _scoped_device_provider =
        std::make_unique<RS_MemoryScopedProvider>(_kernel_manager->getDeviceProvider());

    _record_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_RECORD, getRecordName());

    // Resolve the linked structure and rigid body anchor so a bad reference throws during
    // construction without touching the kernel context. A sphere is rotation invariant, so it has
    // no relative orientation and passes the unit quaternion.
    if (_rsc.is<RSC_IntersectRegionCuboidRigidBodyLinked>()) {
        auto& region_linked = _rsc.as<RSC_IntersectRegionCuboidRigidBodyLinked>();
        _resolveLinkedRigidBody(
            region_linked.structure_name,
            region_linked.body_sid,
            region_linked.segment_bid,
            region_linked.relative_anchor,
            region_linked.relative_orientation
        );
    } else if (_rsc.is<RSC_IntersectRegionSphereRigidBodyLinked>()) {
        auto& region_linked = _rsc.as<RSC_IntersectRegionSphereRigidBodyLinked>();
        _resolveLinkedRigidBody(
            region_linked.structure_name,
            region_linked.body_sid,
            region_linked.segment_bid,
            region_linked.relative_anchor,
            RQuat3rf::unit()
        );
    }

    // Stage an empty membership record so delivery is valid before the first record step.
    RBigSize empty_size = RS_WorldIntersectRegionRecord::getRequiredMemorySize(0);
    _host_record_buffer = _scoped_device_provider->allocateHost(empty_size, true);
    _host_record_buffer.set(0);
    _host_record.initFromMemory(_host_record_buffer.ptr(), 0);
    _host_record.record_gid = _record_registration.gid();
    _host_record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
    _membership_capacity = 0;
}

void RS_KernelIntersectRegionManager::commit() {
    _kernel_manager->enterKernelDataCriticalZone();
    _initKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;
}

void RS_KernelIntersectRegionManager::decommit() {
    if (_is_kernel_data_inited) {
        _kernel_manager->enterKernelDataCriticalZone();
        _freeKernelData();
        _kernel_manager->exitKernelDataCriticalZone();
        _is_kernel_data_inited = false;
    }
}

void RS_KernelIntersectRegionManager::syncRecorderRecords() {}

void RS_KernelIntersectRegionManager::waitForRecorderRecordsSync() {}

std::vector<RS_WorldIntersectRegionRecord> RS_KernelIntersectRegionManager::getRecorderRecords() {
    return {_host_record};
}

bool RS_KernelIntersectRegionManager::hasControllerRecord() {
    return !getRecordControllerName().empty();
}

void RS_KernelIntersectRegionManager::syncControllerRecord() {}

RS_WorldIntersectRegionRecord RS_KernelIntersectRegionManager::getControllerRecord() {
    return _host_record;
}

void RS_KernelIntersectRegionManager::setIntersectRegionRecord(
    const std::vector<RS_IntersectRegionPair>& pairs,
    RSize offset,
    RSize size,
    RBigIndex step,
    RBigFloat time
) {
    // Grow the host arena monotonically to the largest membership set ever seen.
    if (size > _membership_capacity) {
        RBigSize required = RS_WorldIntersectRegionRecord::getRequiredMemorySize(size);
        _host_record_buffer = _scoped_device_provider->allocateHost(required, true);
        _membership_capacity = size;
    }

    _host_record.initFromMemory(_host_record_buffer.ptr(), size);
    _host_record.record_gid = _record_registration.gid();
    _host_record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
    if (_host_record.step != nullptr) *_host_record.step = step;
    if (_host_record.time != nullptr) *_host_record.time = time;
    for (RSize i = 0; i < size; ++i) {
        _host_record.memberships.structure_gid[i] = pairs[offset + i].structure_gid;
        _host_record.memberships.closest_relative_position[i] =
            RVec3rf(pairs[offset + i].closest_relative_offset);
    }
}

std::string RS_KernelIntersectRegionManager::getRecordName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.name; });
}

std::string RS_KernelIntersectRegionManager::getRecordControllerName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.target_controller_reference; });
}

std::string RS_KernelIntersectRegionManager::getRecordRecorderName() const {
    return _rsc.visit([](auto& c) -> std::string { return c.target_recorder_reference; });
}

RIndex RS_KernelIntersectRegionManager::getRecordGid() const { return _record_registration.gid(); }

std::string RS_KernelIntersectRegionManager::getStructureName() const {
    if (_rsc.is<RSC_IntersectRegionCuboidRigidBodyLinked>())
        return _rsc.as<RSC_IntersectRegionCuboidRigidBodyLinked>().structure_name;
    if (_rsc.is<RSC_IntersectRegionSphereRigidBodyLinked>())
        return _rsc.as<RSC_IntersectRegionSphereRigidBodyLinked>().structure_name;
    return "";
}

RIndex RS_KernelIntersectRegionManager::getStructureGid() const {
    return _structure_reference.gid();
}

const RS_KernelIntersectRegionManager::SupportedRSC&
RS_KernelIntersectRegionManager::getRSC() const {
    return _rsc;
}

void RS_KernelIntersectRegionManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        RS_IntersectRegionMem mem;
        mem.gid = _record_registration.gid();
        _rsc.visit([&](auto& c) {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, RSC_IntersectRegionCuboid>) {
                mem.type = RSE_INTERSECT_REGION_CUBOID;
                mem.position = RVec3f(c.position);
                mem.orientation = RQuat3f(c.orientation);
                mem.half_size = RVec3f(c.size) * 0.5f;
            } else if constexpr (std::is_same_v<T, RSC_IntersectRegionSphere>) {
                mem.type = RSE_INTERSECT_REGION_SPHERE;
                mem.position = RVec3f(c.position);
                mem.radius = float(c.radius);
            } else if constexpr (std::is_same_v<T, RSC_IntersectRegionCuboidRigidBodyLinked>) {
                mem.type = RSE_INTERSECT_REGION_CUBOID_RIGID_BODY_LINKED;
                mem.rigid_body_sid = _rigid_body_sid;
                mem.half_size = RVec3f(c.size) * 0.5f;
                mem.rigid_body_anchor_rb_position = RVec3f(_rb_anchor);
                mem.rigid_body_anchor_rb_orientation = RQuat3f(_rb_orientation);
            } else if constexpr (std::is_same_v<T, RSC_IntersectRegionSphereRigidBodyLinked>) {
                mem.type = RSE_INTERSECT_REGION_SPHERE_RIGID_BODY_LINKED;
                mem.rigid_body_sid = _rigid_body_sid;
                mem.radius = float(c.radius);
                mem.rigid_body_anchor_rb_position = RVec3f(_rb_anchor);
                mem.rigid_body_anchor_rb_orientation = RQuat3f(_rb_orientation);
            }
        });

        ctx.host_data->intersect_region_record_layout.add(1, _record_registration.gid());
        ctx.intersect_regions.append(mem);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();

        // Resolve the linked rigid body handle and seed the initial world pose.
        if (_isRigidBodyLinked()) {
            _kernel_manager->syncKernelIfNeeded();
            castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
                using KernelType = std::remove_pointer_t<decltype(k_ptr)>;

                p_ptr->template run<
                    RSCF_LaunchDirect<RSCF_ResolveIntersectRegionDependency, KernelType>>(
                    {1},
                    ctx.intersect_regions.size() - 1,
                    ctx.host_data->structure_rigid_body_layout.segmentOffset(
                        ctx.host_data->structure_rigid_body_layout.find(_structure_reference.gid())
                    )
                );
            });
        }
    });
}

void RS_KernelIntersectRegionManager::_freeKernelData() {
    if (!_is_kernel_data_inited) return;
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        RIndex index =
            ctx.host_data->intersect_region_record_layout.find(_record_registration.gid());
        ctx.intersect_regions.erase(index);
        ctx.host_data->intersect_region_record_layout.remove(index);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

bool RS_KernelIntersectRegionManager::_isRigidBodyLinked() const {
    return _rsc.is<RSC_IntersectRegionCuboidRigidBodyLinked>() ||
           _rsc.is<RSC_IntersectRegionSphereRigidBodyLinked>();
}

void RS_KernelIntersectRegionManager::_resolveLinkedRigidBody(
    const std::string& structure_name,
    RIndex body_sid,
    RIndex segment_bid,
    const RVec3rf& relative_anchor,
    const RQuat3rf& relative_orientation
) {
    auto& structure_managers = _kernel_manager->getKernelStructureVoxelManagers();
    auto sm_it = structure_managers.find(structure_name);
    if (sm_it == structure_managers.end())
        RS_THROW(RS_InvalidCharacterizationException(
            "Structure {} is not registered in the kernel manager", structure_name
        ));
    auto& structure_manager = *sm_it->second;

    _rigid_body_sid = structure_manager.findRigidBodySID(body_sid, segment_bid);
    if (_rigid_body_sid == RS_NULL_INDEX)
        RS_THROW(RS_InvalidCharacterizationException(
            "Rigid body in body {} segment {} of structure {} does not exist",
            body_sid,
            segment_bid,
            structure_name
        ));

    _rb_anchor = structure_manager.getRigidBodyPositionFromRelativeAnchor(
        body_sid, _rigid_body_sid, relative_anchor
    );
    auto& body_rsc = structure_manager.getRSC().as<rsc::RSC_StructureVoxel>().bodies[body_sid];
    _rb_orientation = body_rsc.relative_orientation.inverse() * relative_orientation;

    // Reference the structure so it cannot be removed while this region is alive.
    _structure_reference = RS_KernelStructureVoxelReference(_kernel_manager, structure_name);
}

}  // namespace rise
