// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rise_origin/core/rs_kernel_manager.h"
#include "rise_origin/logic/ray/rs_ray_sensor_c.h"
#include "rise_origin/logic/signal/rs_ray_sensor_signal_m.h"
#include "rise_origin/logic/utils/rs_launch_c.h"
#include "rs_kernel_ray_sensor_manager.h"

using namespace std;
using namespace rise::rsc;
using namespace rise::logic;
using namespace rise::device;

namespace rise {

/*****************************************************************
 * RS_KernelRaySensorManager
 *****************************************************************/
RS_KernelRaySensorManager::RS_KernelRaySensorManager(
    RS_KernelManager* kernel_manager,
    const SupportedRSC& ray_sensor_rsc
)
    : _kernel_manager(kernel_manager), _rsc(ray_sensor_rsc) {
    _scoped_device_provider =
        std::make_unique<RS_MemoryScopedProvider>(_kernel_manager->getDeviceProvider());
    auto entity_name = _rsc.visit([](auto& s) -> std::string { return s.name; });
    _entity_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_ENTITY, entity_name);
    _record_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_RECORD, entity_name + ":record");
    _signal_registration =
        RS_KernelGidRegistration(_kernel_manager, RSE_KERNEL_GID_SIGNAL, entity_name + ":signal");

    auto memory_provider = _scoped_device_provider.get();
    auto buffer_frame_num = _kernel_manager->getKernelRecordBufferFrameNum();

    const std::vector<RVec3rf>& directions =
        _rsc.visit([](auto& s) -> const std::vector<RVec3rf>& { return s.ray_directions; });
    _ray_num = static_cast<RSize>(directions.size());
    _max_range = _rsc.visit([](auto& s) { return s.max_range; });

    if (_rsc.is<RSC_RaySensorRigidBodyLinked>()) {
        auto& linked = _rsc.as<RSC_RaySensorRigidBodyLinked>();
        auto& structure_managers = _kernel_manager->getKernelStructureVoxelManagers();
        auto sm_it = structure_managers.find(linked.structure_name);
        if (sm_it == structure_managers.end())
            RS_THROW(RS_InvalidCharacterizationException(
                "Structure {} is not registered in the kernel manager", linked.structure_name
            ));
        auto& structure_manager = *sm_it->second;

        RIndex rigid_body_sid =
            structure_manager.findRigidBodySID(linked.body_sid, linked.segment_bid);
        if (rigid_body_sid == RS_NULL_INDEX)
            RS_THROW(RS_InvalidCharacterizationException(
                "Rigid body in body {} segment {} of structure {} does not exist",
                linked.body_sid,
                linked.segment_bid,
                linked.structure_name
            ));

        _rigid_body_sid = rigid_body_sid;
        _ray_sensor_rb_position = structure_manager.getRigidBodyPositionFromRelativeAnchor(
            linked.body_sid, rigid_body_sid, linked.relative_anchor
        );
        auto& body_rsc =
            structure_manager.getRSC().as<rsc::RSC_StructureVoxel>().bodies[linked.body_sid];
        _ray_sensor_rb_orientation =
            body_rsc.relative_orientation.inverse() * linked.relative_orientation;

        // Reference the structure so it cannot be removed while this sensor is alive.
        _structure_reference =
            RS_KernelStructureVoxelReference(_kernel_manager, linked.structure_name);
    } else {
        auto& free = _rsc.as<RSC_RaySensorFree>();
        _origin_position = free.origin_position;
        _orientation = free.orientation;
    }

    // Upload the static ray directions.
    _ray_directions.init({_ray_num}, memory_provider);
    _ray_directions.writeSlice(directions.data(), _ray_num, 0);

    // Create record buffer.
    const auto& record_rsc = _record();
    RBigSize rn = RBigSize(_ray_num);
    RBigSize frame_size = RS_WorldRaySensorRecord::getRequiredMemorySize(
        record_rsc.has_hit ? rn : 0,
        record_rsc.has_entity_info ? rn : 0,
        record_rsc.has_entity_detail ? rn : 0,
        record_rsc.has_material ? rn : 0,
        record_rsc.has_normal ? rn : 0,
        record_rsc.has_velocity ? rn : 0,
        1
    );
    _recorder_buffer.init(buffer_frame_num, buffer_frame_num * frame_size, memory_provider);
    _controller_buffer.init(frame_size, memory_provider);

    // Create signal buffer.
    if (hasControllerSignal()) {
        _addSignals();
    }
}

void RS_KernelRaySensorManager::commit() {
    _kernel_manager->enterKernelDataCriticalZone();
    if (hasControllerSignal()) {
        _initKernelSignalData();
    }
    _initKernelData();
    _kernel_manager->exitKernelDataCriticalZone();
    _is_kernel_data_inited = true;
}

void RS_KernelRaySensorManager::decommit() {
    if (_is_kernel_data_inited) {
        _kernel_manager->enterKernelDataCriticalZone();
        _freeKernelData();
        if (hasControllerSignal()) {
            _freeKernelSignalData();
        }
        _kernel_manager->exitKernelDataCriticalZone();
        _is_kernel_data_inited = false;
    }
}

void RS_KernelRaySensorManager::freeze() {
    if (_is_frozen || !_is_kernel_data_inited) return;
    _kernel_manager->enterKernelDataCriticalZone();
    _freezeKernelData(true);
    _kernel_manager->exitKernelDataCriticalZone();
    _is_frozen = true;
}

void RS_KernelRaySensorManager::unfreeze() {
    if (!_is_frozen || !_is_kernel_data_inited) return;
    _kernel_manager->enterKernelDataCriticalZone();
    _freezeKernelData(false);
    _kernel_manager->exitKernelDataCriticalZone();
    _is_frozen = false;
}

bool RS_KernelRaySensorManager::isFrozen() const { return _is_frozen; }

void RS_KernelRaySensorManager::reference() { _use_count += 1; }

void RS_KernelRaySensorManager::dereference() {
    if (_use_count == 0) {
        RS_THROW(RS_HostExecutionException(
            "Ray sensor '{}' use_count underflow on dereference", getEntityName()
        ));
    }
    _use_count -= 1;
}

RSize RS_KernelRaySensorManager::getUseCount() const { return _use_count; }

void RS_KernelRaySensorManager::syncRecorderRecords() {
    _recorder_buffer.swapAndCopy(true, false, false);
}

void RS_KernelRaySensorManager::waitForRecorderRecordsSync() {
    _recorder_buffer.waitForCopyFinish();
}

std::vector<RS_WorldRaySensorRecord> RS_KernelRaySensorManager::getRecorderRecords() {
    vector<RS_WorldRaySensorRecord> records;
    const auto& record_rsc = _record();
    RBigSize rn = RBigSize(_ray_num);
    for (RIndex i = 0, total = _recorder_buffer.getHostNum(); i < total; ++i) {
        RS_WorldRaySensorRecord record;
        record.initFromMemory(
            _recorder_buffer.getHostChunk(i).ptr(),
            record_rsc.has_hit ? rn : 0,
            record_rsc.has_entity_info ? rn : 0,
            record_rsc.has_entity_detail ? rn : 0,
            record_rsc.has_material ? rn : 0,
            record_rsc.has_normal ? rn : 0,
            record_rsc.has_velocity ? rn : 0,
            1
        );
        record.sensor_gid = _entity_registration.gid();
        record.structure_gid = _structure_reference.gid();
        record.rigid_body_sid = _rigid_body_sid;
        record.ray_num = _ray_num;
        record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
        records.emplace_back(record);
    }
    return records;
}

bool RS_KernelRaySensorManager::hasControllerSignal() { return _rsc.is<RSC_RaySensorFree>(); }

void RS_KernelRaySensorManager::syncControllerSignal() {
    if (hasControllerSignal()) {
        _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());
    }
}

RS_WorldRaySensorSignal RS_KernelRaySensorManager::getControllerSignal() {
    if (hasControllerSignal()) {
        RS_WorldRaySensorSignal signal;
        signal.initFromMemory(_host_signal_buffer.ptr(), 1, 1);
        signal.sensor_gid = _entity_registration.gid();
        signal.device = _host_signal_buffer.device();
        return signal;
    }
    return {};
}

bool RS_KernelRaySensorManager::hasControllerRecord() {
    return _rsc.is<RSC_RaySensorFree>() || _rsc.is<RSC_RaySensorRigidBodyLinked>();
}

void RS_KernelRaySensorManager::syncControllerRecord() {
    if (hasControllerRecord()) {
        _controller_buffer.sync(true);
    }
}

RS_WorldRaySensorRecord RS_KernelRaySensorManager::getControllerRecord() {
    RS_WorldRaySensorRecord record;
    const auto& record_rsc = _record();
    RBigSize rn = RBigSize(_ray_num);
    record.initFromMemory(
        _controller_buffer.getHostChunk(0).ptr(),
        record_rsc.has_hit ? rn : 0,
        record_rsc.has_entity_info ? rn : 0,
        record_rsc.has_entity_detail ? rn : 0,
        record_rsc.has_material ? rn : 0,
        record_rsc.has_normal ? rn : 0,
        record_rsc.has_velocity ? rn : 0,
        1
    );
    record.sensor_gid = _entity_registration.gid();
    record.structure_gid = _structure_reference.gid();
    record.rigid_body_sid = _rigid_body_sid;
    record.ray_num = _ray_num;
    record.device = RS_DeviceLocation(RSE_DEVICE_TYPE_CPU, 0);
    return record;
}

std::string RS_KernelRaySensorManager::getEntityName() const {
    return _rsc.visit([](auto& s) -> std::string { return s.name; });
}

std::string RS_KernelRaySensorManager::getRecordName() const {
    return _rsc.visit([](auto& s) -> std::string { return s.record->name; });
}

std::string RS_KernelRaySensorManager::getSignalName() const {
    return _rsc.visit([](auto& s) -> std::string { return s.signal->name; });
}

std::string RS_KernelRaySensorManager::getRecordControllerName() const {
    return _rsc.visit([](auto& s) -> std::string { return s.record->target_controller_reference; });
}

std::string RS_KernelRaySensorManager::getRecordRecorderName() const {
    return _rsc.visit([](auto& s) -> std::string { return s.record->target_recorder_reference; });
}

std::string RS_KernelRaySensorManager::getSignalControllerName() const {
    return _rsc.visit([](auto& s) -> std::string { return s.signal->source_controller_reference; });
}

RIndex RS_KernelRaySensorManager::getEntityGid() const { return _entity_registration.gid(); }

RIndex RS_KernelRaySensorManager::getRecordGid() const { return _record_registration.gid(); }

RIndex RS_KernelRaySensorManager::getSignalGid() const { return _signal_registration.gid(); }

std::string RS_KernelRaySensorManager::getStructureName() const {
    if (!_rsc.is<RSC_RaySensorRigidBodyLinked>()) return "";
    return _rsc.as<RSC_RaySensorRigidBodyLinked>().structure_name;
}

RIndex RS_KernelRaySensorManager::getStructureGid() const {
    if (!_rsc.is<RSC_RaySensorRigidBodyLinked>()) return RS_NULL_INDEX;
    return _structure_reference.gid();
}

const RS_KernelRaySensorManager::SupportedRSC& RS_KernelRaySensorManager::getRSC() const {
    return _rsc;
}

void RS_KernelRaySensorManager::_initKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        _kernel_manager->getDeviceProvider()->sync();

        auto& ctx = k_ptr->ctx;
        using ContextType = decltype(k_ptr->ctx);
        using RaySensorMemType = RS_RaySensorMem<ContextType::FloatMemAccuracy>;

        RaySensorMemType sensor;
        sensor.gid = _entity_registration.gid();
        sensor.ray_num = _ray_num;
        sensor.max_range = _max_range;

        const auto& record_rsc = _record();
        RS_RaySensorStateUnion state{0};
        if (_rsc.is<RSC_RaySensorRigidBodyLinked>()) {
            state.ray_sensor_state.type = RSE_RAY_SENSOR_RIGID_BODY_LINKED;
            sensor.rigid_body_sid = _rigid_body_sid;
            sensor.rigid_body_anchor_rb_position = _ray_sensor_rb_position;
            sensor.rigid_body_anchor_rb_orientation = _ray_sensor_rb_orientation;
        } else {
            state.ray_sensor_state.type = RSE_RAY_SENSOR_FREE;
            sensor.position = _origin_position;
            sensor.orientation = _orientation;
            sensor.prev_position = _origin_position;
        }
        state.ray_sensor_state.has_hit = record_rsc.has_hit;
        state.ray_sensor_state.has_entity_info = record_rsc.has_entity_info;
        state.ray_sensor_state.has_entity_detail = record_rsc.has_entity_detail;
        state.ray_sensor_state.has_material = record_rsc.has_material;
        state.ray_sensor_state.has_normal = record_rsc.has_normal;
        state.ray_sensor_state.has_velocity = record_rsc.has_velocity;
        sensor.state = state.ray_sensor_state_packed;

        RIndex segment_gid = _entity_registration.gid();
        RSize rn = _ray_num;

        // Update host layout.
        ctx.host_data->ray_sensor_layout.add(1, segment_gid);
        ctx.host_data->ray_sensor_ray_layout.add(rn, segment_gid);
        ctx.host_data->ray_sensor_record_layout.add(1, segment_gid);
        ctx.host_data->ray_sensor_hit_record_layout.add(record_rsc.has_hit ? rn : 0, segment_gid);
        ctx.host_data->ray_sensor_entity_info_record_layout.add(
            record_rsc.has_entity_info ? rn : 0, segment_gid
        );
        ctx.host_data->ray_sensor_entity_detail_record_layout.add(
            record_rsc.has_entity_detail ? rn : 0, segment_gid
        );
        ctx.host_data->ray_sensor_material_record_layout.add(
            record_rsc.has_material ? rn : 0, segment_gid
        );
        ctx.host_data->ray_sensor_normal_record_layout.add(
            record_rsc.has_normal ? rn : 0, segment_gid
        );
        ctx.host_data->ray_sensor_velocity_record_layout.add(
            record_rsc.has_velocity ? rn : 0, segment_gid
        );
        ctx.host_data->ray_sensor_state_record_layout.add(1, segment_gid);

        // Update device layout.
        ctx.host_data->ray_sensor_layout.syncDeviceLayout(ctx.ray_sensor_layout);
        ctx.host_data->ray_sensor_ray_layout.syncDeviceLayout(ctx.ray_sensor_ray_layout);
        ctx.host_data->ray_sensor_record_layout.syncDeviceLayout(ctx.ray_sensor_record_layout);
        ctx.host_data->ray_sensor_hit_record_layout.syncDeviceLayout(
            ctx.ray_sensor_hit_record_layout
        );
        ctx.host_data->ray_sensor_entity_info_record_layout.syncDeviceLayout(
            ctx.ray_sensor_entity_info_record_layout
        );
        ctx.host_data->ray_sensor_entity_detail_record_layout.syncDeviceLayout(
            ctx.ray_sensor_entity_detail_record_layout
        );
        ctx.host_data->ray_sensor_material_record_layout.syncDeviceLayout(
            ctx.ray_sensor_material_record_layout
        );
        ctx.host_data->ray_sensor_normal_record_layout.syncDeviceLayout(
            ctx.ray_sensor_normal_record_layout
        );
        ctx.host_data->ray_sensor_velocity_record_layout.syncDeviceLayout(
            ctx.ray_sensor_velocity_record_layout
        );
        ctx.host_data->ray_sensor_state_record_layout.syncDeviceLayout(
            ctx.ray_sensor_state_record_layout
        );

        // Write data to device.
        if (hasControllerSignal()) {
            sensor.sensor_signal_handle = _ray_sensor_signal;
        }
        ctx.ray_sensors.append(sensor);
        ctx.ray_sensor_records.append({});
        ctx.ray_sensor_recorder_buffers.append(_recorder_buffer);
        ctx.ray_sensor_controller_buffers.append(_controller_buffer);
        ctx.ray_sensor_directions.append(_ray_directions);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();

        if (hasControllerSignal()) {
            // Seed initial values into the host signal buffer.
            _host_signal.position_signals.position[0] = _origin_position;
            _host_signal.orientation_signals.orientation[0] = _orientation;
            _device_signal_buffer.copyFrom(_host_signal_buffer, _host_signal_buffer.size());
        }

        // Resolve data dependency
        if (_rsc.is<RSC_RaySensorRigidBodyLinked>()) {
            _kernel_manager->syncKernelIfNeeded();
            castProviderPtrAndCall(_kernel_manager->getDeviceProvider(), [&](auto p_ptr) {
                using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
                p_ptr->template run<
                    RSCF_LaunchDirect<RSCF_ResolveRigidBodyLinkedRaySensorDependency, KernelType>>(
                    {1},
                    ctx.ray_sensors.size() - 1,
                    ctx.host_data->structure_rigid_body_layout.segmentOffset(
                        ctx.host_data->structure_rigid_body_layout.find(_structure_reference.gid())
                    )
                );
            });
        }
    });
}

void RS_KernelRaySensorManager::_freeKernelData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        if (!_is_kernel_data_inited) return;

        auto& ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Remove data from device
        RIndex segment_gid = _entity_registration.gid();
        RIndex sensor_index = ctx.host_data->ray_sensor_layout.find(segment_gid);
        ctx.ray_sensors.erase(sensor_index);
        ctx.ray_sensor_records.erase(sensor_index);
        ctx.ray_sensor_recorder_buffers.erase(sensor_index);
        ctx.ray_sensor_controller_buffers.erase(sensor_index);
        ctx.ray_sensor_directions.erase(sensor_index);

        // Update host layout
        ctx.host_data->ray_sensor_layout.remove(ctx.host_data->ray_sensor_layout.find(segment_gid));
        ctx.host_data->ray_sensor_ray_layout.remove(
            ctx.host_data->ray_sensor_ray_layout.find(segment_gid)
        );
        ctx.host_data->ray_sensor_record_layout.remove(
            ctx.host_data->ray_sensor_record_layout.find(segment_gid)
        );
        ctx.host_data->ray_sensor_hit_record_layout.remove(
            ctx.host_data->ray_sensor_hit_record_layout.find(segment_gid)
        );
        ctx.host_data->ray_sensor_entity_info_record_layout.remove(
            ctx.host_data->ray_sensor_entity_info_record_layout.find(segment_gid)
        );
        ctx.host_data->ray_sensor_entity_detail_record_layout.remove(
            ctx.host_data->ray_sensor_entity_detail_record_layout.find(segment_gid)
        );
        ctx.host_data->ray_sensor_material_record_layout.remove(
            ctx.host_data->ray_sensor_material_record_layout.find(segment_gid)
        );
        ctx.host_data->ray_sensor_normal_record_layout.remove(
            ctx.host_data->ray_sensor_normal_record_layout.find(segment_gid)
        );
        ctx.host_data->ray_sensor_velocity_record_layout.remove(
            ctx.host_data->ray_sensor_velocity_record_layout.find(segment_gid)
        );
        ctx.host_data->ray_sensor_state_record_layout.remove(
            ctx.host_data->ray_sensor_state_record_layout.find(segment_gid)
        );

        // Update device layout
        ctx.host_data->ray_sensor_layout.syncDeviceLayout(ctx.ray_sensor_layout);
        ctx.host_data->ray_sensor_ray_layout.syncDeviceLayout(ctx.ray_sensor_ray_layout);
        ctx.host_data->ray_sensor_record_layout.syncDeviceLayout(ctx.ray_sensor_record_layout);
        ctx.host_data->ray_sensor_hit_record_layout.syncDeviceLayout(
            ctx.ray_sensor_hit_record_layout
        );
        ctx.host_data->ray_sensor_entity_info_record_layout.syncDeviceLayout(
            ctx.ray_sensor_entity_info_record_layout
        );
        ctx.host_data->ray_sensor_entity_detail_record_layout.syncDeviceLayout(
            ctx.ray_sensor_entity_detail_record_layout
        );
        ctx.host_data->ray_sensor_material_record_layout.syncDeviceLayout(
            ctx.ray_sensor_material_record_layout
        );
        ctx.host_data->ray_sensor_normal_record_layout.syncDeviceLayout(
            ctx.ray_sensor_normal_record_layout
        );
        ctx.host_data->ray_sensor_velocity_record_layout.syncDeviceLayout(
            ctx.ray_sensor_velocity_record_layout
        );
        ctx.host_data->ray_sensor_state_record_layout.syncDeviceLayout(
            ctx.ray_sensor_state_record_layout
        );

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelRaySensorManager::_addSignals() {
    RBigSize total_size = RS_WorldRaySensorSignal::getRequiredMemorySize(1, 1);

    RS_MemoryProvider* memory_provider = _scoped_device_provider.get();
    _host_signal_buffer = memory_provider->allocateHost(total_size, true);
    _device_signal_buffer = memory_provider->allocate(total_size, true);
    _host_signal_buffer.set(0);
    _device_signal_buffer.set(0);

    _host_signal.initFromMemory(_host_signal_buffer.ptr(), 1, 1);
    _host_signal.sensor_gid = _entity_registration.gid();
    _device_signal.initFromMemory(_device_signal_buffer.ptr(), 1, 1);
    _device_signal.sensor_gid = _entity_registration.gid();
    _device_signal.device = _device_signal_buffer.device();
}

void RS_KernelRaySensorManager::_initKernelSignalData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        ctx.host_data->ray_sensor_signal_layout.add(1, _entity_registration.gid());
        ctx.host_data->ray_sensor_signal_layout.syncDeviceLayout(ctx.ray_sensor_signal_layout);
        ctx.ray_sensor_signals.resize({ctx.host_data->ray_sensor_signal_layout.capacity()});
        _ray_sensor_signal = ctx.host_data->ray_sensor_signal_layout.segmentOffset(
            ctx.host_data->ray_sensor_signal_layout.find(_entity_registration.gid())
        );
        RS_RaySensorSignalMem signal_mem;
        signal_mem.signal = _device_signal;
        ctx.ray_sensor_signals.writeSlice(&signal_mem, 1, _ray_sensor_signal);

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelRaySensorManager::_freeKernelSignalData() {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        RIndex signal_index =
            ctx.host_data->ray_sensor_signal_layout.find(_entity_registration.gid());
        ctx.host_data->ray_sensor_signal_layout.remove(signal_index);
        ctx.host_data->ray_sensor_signal_layout.syncDeviceLayout(ctx.ray_sensor_signal_layout);
        ctx.ray_sensor_signals.resize({ctx.host_data->ray_sensor_signal_layout.capacity()});

        // Context data is modified, needs to update kernel
        _kernel_manager->markKernelModified();
    });
}

void RS_KernelRaySensorManager::_freezeKernelData(bool is_freeze) {
    castKernelPtrAndCall(&_kernel_manager->kernel, [&](auto k_ptr) {
        auto& ctx = k_ptr->ctx;

        // Sync to make sure all operations in memory provider stream are finished
        _kernel_manager->getDeviceProvider()->sync();

        // Toggle every ray sensor layout so trace launches skip the sensor while frozen; the
        // last record is never overwritten, so it holds the last frame.
        bool is_active = !is_freeze;
        RIndex segment_gid = _entity_registration.gid();
        RIndex sensor_index = ctx.host_data->ray_sensor_layout.find(segment_gid);
        ctx.host_data->ray_sensor_layout.setSegmentIsActive(sensor_index, is_active);
        ctx.host_data->ray_sensor_ray_layout.setSegmentIsActive(
            ctx.host_data->ray_sensor_ray_layout.find(segment_gid), is_active
        );
        ctx.host_data->ray_sensor_record_layout.setSegmentIsActive(
            ctx.host_data->ray_sensor_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->ray_sensor_hit_record_layout.setSegmentIsActive(
            ctx.host_data->ray_sensor_hit_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->ray_sensor_entity_info_record_layout.setSegmentIsActive(
            ctx.host_data->ray_sensor_entity_info_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->ray_sensor_entity_detail_record_layout.setSegmentIsActive(
            ctx.host_data->ray_sensor_entity_detail_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->ray_sensor_material_record_layout.setSegmentIsActive(
            ctx.host_data->ray_sensor_material_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->ray_sensor_normal_record_layout.setSegmentIsActive(
            ctx.host_data->ray_sensor_normal_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->ray_sensor_velocity_record_layout.setSegmentIsActive(
            ctx.host_data->ray_sensor_velocity_record_layout.find(segment_gid), is_active
        );
        ctx.host_data->ray_sensor_state_record_layout.setSegmentIsActive(
            ctx.host_data->ray_sensor_state_record_layout.find(segment_gid), is_active
        );

        ctx.host_data->ray_sensor_layout.syncDeviceLayout(ctx.ray_sensor_layout);
        ctx.host_data->ray_sensor_ray_layout.syncDeviceLayout(ctx.ray_sensor_ray_layout);
        ctx.host_data->ray_sensor_record_layout.syncDeviceLayout(ctx.ray_sensor_record_layout);
        ctx.host_data->ray_sensor_hit_record_layout.syncDeviceLayout(
            ctx.ray_sensor_hit_record_layout
        );
        ctx.host_data->ray_sensor_entity_info_record_layout.syncDeviceLayout(
            ctx.ray_sensor_entity_info_record_layout
        );
        ctx.host_data->ray_sensor_entity_detail_record_layout.syncDeviceLayout(
            ctx.ray_sensor_entity_detail_record_layout
        );
        ctx.host_data->ray_sensor_material_record_layout.syncDeviceLayout(
            ctx.ray_sensor_material_record_layout
        );
        ctx.host_data->ray_sensor_normal_record_layout.syncDeviceLayout(
            ctx.ray_sensor_normal_record_layout
        );
        ctx.host_data->ray_sensor_velocity_record_layout.syncDeviceLayout(
            ctx.ray_sensor_velocity_record_layout
        );
        ctx.host_data->ray_sensor_state_record_layout.syncDeviceLayout(
            ctx.ray_sensor_state_record_layout
        );

        if (hasControllerSignal()) {
            ctx.host_data->ray_sensor_signal_layout.setSegmentIsActive(
                ctx.host_data->ray_sensor_signal_layout.find(_entity_registration.gid()), is_active
            );
            ctx.host_data->ray_sensor_signal_layout.syncDeviceLayout(ctx.ray_sensor_signal_layout);
        }

        // Layout active state changed, needs to update kernel.
        _kernel_manager->markKernelModified();
    });
}

/*****************************************************************
 * RS_KernelRaySensorReference
 *****************************************************************/
RS_KernelRaySensorReference::RS_KernelRaySensorReference(
    RS_KernelManager* kernel_manager,
    std::string name
)
    : _kernel_manager(kernel_manager), _name(std::move(name)) {
    auto& managers = _kernel_manager->getKernelRaySensorManagers();
    auto it = managers.find(_name);
    if (it == managers.end()) {
        RS_THROW(RS_InvalidCharacterizationException(
            "Ray sensor {} is not registered in the kernel manager", _name
        ));
    }
    it->second->reference();
    _gid = it->second->getEntityGid();
}

RS_KernelRaySensorReference::~RS_KernelRaySensorReference() { _release(); }

RS_KernelRaySensorReference::RS_KernelRaySensorReference(
    RS_KernelRaySensorReference&& other
) noexcept
    : _kernel_manager(other._kernel_manager), _name(std::move(other._name)), _gid(other._gid) {
    other._kernel_manager = nullptr;
    other._gid = RS_NULL_INDEX;
}

RS_KernelRaySensorReference& RS_KernelRaySensorReference::operator=(
    RS_KernelRaySensorReference&& other
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

void RS_KernelRaySensorReference::_release() {
    if (_kernel_manager == nullptr) return;
    auto& managers = _kernel_manager->getKernelRaySensorManagers();
    auto it = managers.find(_name);
    if (it != managers.end()) {
        it->second->dereference();
    }
    _kernel_manager = nullptr;
    _gid = RS_NULL_INDEX;
}

}  // namespace rise
