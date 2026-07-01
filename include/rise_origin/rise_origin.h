// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RISE_ORIGIN_H
#define RISE_ORIGIN_H

#if defined(__GNUC__) && (__GNUC__ < 11)
#error "Minimum GCC version required is 11"
#endif

#include <memory>
#include <utility>

#include "rise_origin/core/rs_world_callback.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/rsc/rsc_world.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/memory/rs_device.h"

using namespace rise;

class RiseOriginImpl;

/**
 * @brief The RiseOrigin.
 *
 * Usage:
 * @code
 *   rise::rsc::RSC_World world({"config.xml"});
 *   RiseOrigin r;
 *   auto handle = r.hello(world);
 *   // World is yours
 * @endcode
 */
class RiseOrigin {
   public:
    /**
     * @brief Construct a local RiseOrigin instance on the specified device.
     *
     * @param device Device identifiers used by this process.
     */
    explicit RiseOrigin(const RS_DeviceLocation &device = {RSE_DEVICE_TYPE_CPU, 0});

    /**
     * @brief Say hello to the world.
     *
     * @param world World characterization describing all entities, essences, resources,
     *              etc that should exist before time starts.
     * @return The world handle and the rebindable world callback the engine drives each step.
     */
    std::pair<std::shared_ptr<RS_WorldHandle>, std::shared_ptr<RS_WorldCallback>>
    hello(const rsc::RSC_World &world);

    /**
     * @brief Connect to a world running on an already-launched remote instance.
     *
     * @param url Address of the remote instance hosting the world.
     * @return The world handle and the rebindable world callback the engine drives each step.
     */
    std::pair<std::shared_ptr<RS_WorldHandle>, std::shared_ptr<RS_WorldCallback>>
    connect(const std::string &url);

   private:
    std::shared_ptr<RiseOriginImpl> _rise_impl = nullptr;
};

#endif  // RISE_ORIGIN_H
