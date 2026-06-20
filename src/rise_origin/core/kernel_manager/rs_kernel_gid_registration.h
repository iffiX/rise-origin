// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_GID_REGISTRATION_H
#define RS_KERNEL_GID_REGISTRATION_H

#include <string>

#include "rise_origin/utils/rs_def.h"

namespace rise {

class RS_KernelManager;

/**
 * @brief The type of gid referenced by a registration.
 */
RS_ENUM(
    RSE_KernelGidType,
    RIndex,
    (RSE_KERNEL_GID_ENTITY, 0),
    (RSE_KERNEL_GID_ESSENCE, 1),
    (RSE_KERNEL_GID_RESOURCE, 2),
    (RSE_KERNEL_GID_RECORD, 3),
    (RSE_KERNEL_GID_SIGNAL, 4)
)

/**
 * @brief Registers a gid on construction and deregisters it on destruction.
 */
class RS_KernelGidRegistration {
   public:
    RS_KernelGidRegistration() = default;
    RS_KernelGidRegistration(
        RS_KernelManager *kernel_manager,
        RSE_KernelGidType type,
        std::string name
    );
    ~RS_KernelGidRegistration();

    RS_KernelGidRegistration(RS_KernelGidRegistration &&other) noexcept;
    RS_KernelGidRegistration &operator=(RS_KernelGidRegistration &&other) noexcept;
    RS_KernelGidRegistration(const RS_KernelGidRegistration &) = delete;
    RS_KernelGidRegistration &operator=(const RS_KernelGidRegistration &) = delete;

    [[nodiscard]] RIndex gid() const { return _gid; }

   private:
    void _release();

    RS_KernelManager *_kernel_manager = nullptr;
    RSE_KernelGidType _type = RSE_KERNEL_GID_ENTITY;
    std::string _name;
    RIndex _gid = RS_NULL_INDEX;
};

}  // namespace rise

#endif  // RS_KERNEL_GID_REGISTRATION_H
