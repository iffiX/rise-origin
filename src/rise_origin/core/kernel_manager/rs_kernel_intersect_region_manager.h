// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_INTERSECT_REGION_MANAGER_H
#define RS_KERNEL_INTERSECT_REGION_MANAGER_H

#include <string>
#include <vector>

#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/logic/record/rs_intersect_region_m.h"
#include "rise_origin/rsc/record/rsc_intersect_region.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rs_kernel_gid_registration.h"
#include "rs_kernel_structure_voxel_manager.h"

namespace rise {

class RS_KernelManager;

/**
 * @brief Owns one intersect-region record (world-fixed cuboid, sphere, or rigid-body-linked
 *        cuboid), publishes its source mem into the context, and stages the per-step membership
 *        record for controller / recorder delivery.
 */
class RS_KernelIntersectRegionManager {
   public:
    using SupportedRSC = RS_Variant<
        rsc::RSC_IntersectRegionCuboid,
        rsc::RSC_IntersectRegionCuboidRigidBodyLinked,
        rsc::RSC_IntersectRegionSphere>;

    /**
     * @brief Register the record, build the region host data, and, for the linked variant,
     *        reference the host structure and resolve its rigid body.
     */
    RS_KernelIntersectRegionManager(RS_KernelManager *kernel_manager, const SupportedRSC &region_rsc);

    /**
     * @brief Commit the region source mem to the kernel context.
     */
    void commit();

    /**
     * @brief Decommit the region source mem from the kernel context.
     */
    void decommit();

    /**
     * @brief Stage the latest membership set into the recorder records; non-blocking. The set is
     *        already host-resident, so this is a no-op kept for the uniform recorder sync loop.
     */
    void syncRecorderRecords();

    /**
     * @brief Block until the most recent syncRecorderRecords() copy has completed; no-op since the
     *        membership set is host-resident.
     */
    void waitForRecorderRecordsSync();

    /**
     * @brief Pull the staged recorder records (one membership record).
     */
    std::vector<RS_WorldIntersectRegionRecord> getRecorderRecords();

    /**
     * @brief True iff this region emits a controller record (always true).
     */
    bool hasControllerRecord();

    /**
     * @brief Stage the latest controller record; no-op since the membership set is host-resident.
     */
    void syncControllerRecord();

    /**
     * @brief Read the staged controller record (host snapshot).
     */
    RS_WorldIntersectRegionRecord getControllerRecord();

    /**
     * @brief Populate this region's membership record from the contiguous slice
     *        pairs[offset, offset + size). A zero size yields an empty record.
     */
    void setIntersectRegionRecord(
        const std::vector<logic::RS_IntersectRegionPair> &pairs, RSize offset, RSize size
    );

    /**
     * @brief Record name from the active characterization.
     */
    std::string getRecordName() const;

    /**
     * @brief Name of the controller this region's record is sent to
     *        (record.target_controller_reference).
     */
    std::string getRecordControllerName() const;

    /**
     * @brief Name of the recorder this region's record is sent to
     *        (record.target_recorder_reference).
     */
    std::string getRecordRecorderName() const;

    /**
     * @brief Global id of the record entry.
     */
    RIndex getRecordGid() const;

    /**
     * @brief Linked structure name for the rigid-body-linked variant; empty otherwise.
     */
    std::string getStructureName() const;

    /**
     * @brief Linked structure gid for the rigid-body-linked variant; RS_NULL_INDEX otherwise.
     */
    RIndex getStructureGid() const;

    /**
     * @brief Read-only access to the active region characterization.
     */
    [[nodiscard]] const SupportedRSC &getRSC() const;

   private:
    RS_KernelManager *_kernel_manager = nullptr;
    std::unique_ptr<RS_MemoryScopedProvider> _scoped_provider;

    RS_KernelGidRegistration _record_registration;
    RS_KernelStructureVoxelReference _structure_reference;

    bool _is_kernel_data_inited = false;

    SupportedRSC _rsc;

    /// Linked structure rigid body pose, only valid for the rigid-body-linked variant.
    RIndex _rigid_body_sid = RS_NULL_INDEX;
    RVec3rf _rb_anchor = RVec3rf::zero();
    RQuat3rf _rb_orientation = RQuat3rf::unit();

    /// Host arena backing the staged membership record.
    RS_MemoryPtr _host_record_buffer;
    RSize _membership_capacity = 0;
    RS_WorldIntersectRegionRecord _host_record;

    void _initKernelData();

    void _freeKernelData();
};

}  // namespace rise

#endif  // RS_KERNEL_INTERSECT_REGION_MANAGER_H
